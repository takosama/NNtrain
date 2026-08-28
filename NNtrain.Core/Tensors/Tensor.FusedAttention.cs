namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Applies scaled dot-product multi-head attention to a fused QKV
    /// projection shaped [sequence, 3 * model] or
    /// [batch, sequence, 3 * model].
    /// </summary>
    public Tensor FusedMultiHeadAttention(
        int numHeads,
        bool causal = false)
    {
        if (Rank is not 2 and not 3)
        {
            throw new InvalidOperationException(
                "FusedMultiHeadAttention requires rank 2 or rank 3.");
        }

        if (numHeads <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(numHeads),
                numHeads,
                "Head count must be positive.");
        }

        int batch = Rank == 3 ? _shape[0] : 1;
        int sequence = Rank == 3 ? _shape[1] : _shape[0];
        int projectedWidth = _shape[^1];
        if (projectedWidth % 3 != 0)
        {
            throw new ArgumentException(
                "The final QKV dimension must be divisible by three.");
        }

        int modelWidth = projectedWidth / 3;
        if (modelWidth % numHeads != 0)
        {
            throw new ArgumentException(
                "The model width must be divisible by the head count.",
                nameof(numHeads));
        }

        int headWidth = modelWidth / numHeads;
        float scale = 1f / MathF.Sqrt(headWidth);
        bool directBFloat16Gradients = DType == TensorDType.BFloat16
            && TensorExecutionContext.UsesBFloat16GradientStorage;
        if (ExecutionDevice == TensorDevice.Cuda)
        {
            int deviceIndex = CudaDeviceIndex;
            if (DType == TensorDType.Bfp8)
            {
                Bfp8QuantizationDescriptor outputDescriptor =
                    SelectBfp8ResultDescriptor(this);
                var bfp8Context = CudaOperationProfiler.IsEnabled
                    ? CudaOperationProfiler.Measure(
                        "forward.attention",
                        () => TensorCudaKernels
                            .AttentionForwardBfp8Resident(
                                this,
                                outputDescriptor,
                                batch,
                                sequence,
                                modelWidth,
                                numHeads,
                                causal))
                    : TensorCudaKernels.AttentionForwardBfp8Resident(
                        this,
                        outputDescriptor,
                        batch,
                        sequence,
                        modelWidth,
                        numHeads,
                        causal);
                int[] bfp8OutputShape = Rank == 3
                    ? [batch, sequence, modelWidth]
                    : [sequence, modelWidth];
                Tensor bfp8Result;
                try
                {
                    using CudaBfp8OwnedBuffers bfp8Output =
                        bfp8Context.DetachEncodedOutput();
                    bfp8Result = FromCudaBfp8Result(
                        bfp8Output,
                        deviceIndex,
                        bfp8OutputShape,
                        [this]);
                }
                catch (Exception conversionFailure)
                {
                    try
                    {
                        bfp8Context.Dispose();
                    }
                    catch (Exception cleanupFailure)
                    {
                        throw new AggregateException(
                            "BFP8 attention result construction and " +
                            "saved-context cleanup failed.",
                            conversionFailure,
                            cleanupFailure);
                    }
                    System.Runtime.ExceptionServices.ExceptionDispatchInfo
                        .Capture(conversionFailure)
                        .Throw();
                    throw;
                }

                if (AutogradContext.IsRecordingEnabled)
                {
                    AutogradLease<TensorCudaKernels
                        .Bfp8AttentionResidentContext> lease =
                        AutogradLease<TensorCudaKernels
                            .Bfp8AttentionResidentContext>.Own(
                            bfp8Context,
                            AutogradLeaseMetadata.CudaOwned(
                                deviceIndex,
                                TensorDType.Bfp8,
                                DataVersion),
                            static saved => saved.Dispose());
                    bfp8Result.Node.SetBackward(lease, savedContext =>
                    {
                        if (CudaOperationProfiler.IsEnabled)
                        {
                            CudaOperationProfiler.Measure(
                                "backward.attention",
                                () => TensorCudaKernels
                                    .AttentionBackwardBfp8Resident(
                                        this,
                                        bfp8Result,
                                        savedContext,
                                        batch,
                                        sequence,
                                        modelWidth,
                                        numHeads,
                                        causal));
                        }
                        else
                        {
                            TensorCudaKernels
                                .AttentionBackwardBfp8Resident(
                                    this,
                                    bfp8Result,
                                    savedContext,
                                    batch,
                                    sequence,
                                    modelWidth,
                                    numHeads,
                                    causal);
                        }
                    });
                }
                else if (!CudaInferenceScope.TrackResource(bfp8Context))
                {
                    bfp8Context.Dispose();
                }
                return bfp8Result;
            }
            if (DType == TensorDType.BFloat16)
            {
                var bfloat16Context = CudaOperationProfiler.IsEnabled
                    ? CudaOperationProfiler.Measure(
                        "forward.attention",
                        () => TensorCudaKernels.AttentionForwardBFloat16Resident(
                            this,
                            batch,
                            sequence,
                            modelWidth,
                            numHeads,
                            causal))
                    : TensorCudaKernels.AttentionForwardBFloat16Resident(
                        this,
                        batch,
                        sequence,
                        modelWidth,
                        numHeads,
                        causal);
                int[] bfloat16OutputShape = Rank == 3
                    ? [batch, sequence, modelWidth]
                    : [sequence, modelWidth];
                Tensor bfloat16Result = FromCudaResult(
                    bfloat16Context.Output,
                    deviceIndex,
                    bfloat16OutputShape,
                    [this],
                    TensorDType.BFloat16);
                if (AutogradContext.IsRecordingEnabled)
                {
                    AutogradLease<TensorCudaKernels
                        .BFloat16AttentionResidentContext> lease =
                        AutogradLease<TensorCudaKernels
                            .BFloat16AttentionResidentContext>.Own(
                            bfloat16Context,
                            AutogradLeaseMetadata.CudaOwned(
                                deviceIndex,
                                TensorDType.BFloat16,
                                DataVersion),
                            static saved => saved.Dispose());
                    bfloat16Result.Node.SetBackward(lease, savedContext =>
                    {
                        if (CudaOperationProfiler.IsEnabled)
                        {
                            CudaOperationProfiler.Measure(
                                "backward.attention",
                                () => TensorCudaKernels
                                    .AttentionBackwardBFloat16Resident(
                                        this,
                                        bfloat16Result,
                                        savedContext,
                                        batch,
                                        sequence,
                                        modelWidth,
                                        numHeads,
                                        causal));
                        }
                        else
                        {
                            TensorCudaKernels.AttentionBackwardBFloat16Resident(
                                this,
                                bfloat16Result,
                                savedContext,
                                batch,
                                sequence,
                                modelWidth,
                                numHeads,
                                causal);
                        }
                    });
                }
                else if (!CudaInferenceScope.TrackResource(bfloat16Context))
                {
                    bfloat16Context.Dispose();
                }
                return bfloat16Result;
            }
            var cudaContext = CudaOperationProfiler.IsEnabled
                ? CudaOperationProfiler.Measure(
                    "forward.attention",
                    () => TensorCudaKernels.AttentionForwardResident(
                        this, batch, sequence, modelWidth, numHeads, causal))
                : TensorCudaKernels.AttentionForwardResident(
                    this, batch, sequence, modelWidth, numHeads, causal);
            int[] cudaOutputShape = Rank == 3
                ? [batch, sequence, modelWidth]
                : [sequence, modelWidth];
            Tensor cudaResult = FromCudaResult(
                cudaContext.Output, deviceIndex, cudaOutputShape, [this]);
            if (AutogradContext.IsRecordingEnabled)
            {
                AutogradLease<TensorCudaKernels.AttentionResidentContext>
                    lease = AutogradLease<TensorCudaKernels
                        .AttentionResidentContext>.Own(
                        cudaContext,
                        AutogradLeaseMetadata.CudaOwned(
                            deviceIndex,
                            TensorDType.Float32,
                            DataVersion),
                        static saved => saved.Dispose());
                cudaResult.Node.SetBackward(lease, savedContext =>
                {
                    if (CudaOperationProfiler.IsEnabled)
                    {
                        CudaOperationProfiler.Measure(
                            "backward.attention",
                            () => TensorCudaKernels.AttentionBackwardResident(
                                this, cudaResult, savedContext, batch,
                                sequence, modelWidth, numHeads, causal));
                    }
                    else
                    {
                        TensorCudaKernels.AttentionBackwardResident(
                            this, cudaResult, savedContext, batch, sequence,
                            modelWidth, numHeads, causal);
                    }
                });
            }
            else
            {
                if (!CudaInferenceScope.TrackResource(cudaContext))
                    cudaContext.Dispose();
            }
            return cudaResult;
        }

        // Only the CPU implementation reads the host-side QKV array.  Keeping
        // this below the CUDA dispatch avoids a full device-to-host copy and a
        // stream synchronization for every Transformer layer.
        EnsureHostDataCurrent();
        int workItemCount = checked(batch * numHeads);
        int probabilityMatrixLength = checked(sequence * sequence);
        float[] probabilities = new float[
            checked(workItemCount * probabilityMatrixLength)];
        float[] output = new float[checked(batch * sequence * modelWidth)];

        void ForwardHead(int workItem)
        {
            int batchIndex = workItem / numHeads;
            int head = workItem % numHeads;
            int headOffset = head * headWidth;
            int projectedBatchOffset =
                batchIndex * sequence * projectedWidth;
            int outputBatchOffset = batchIndex * sequence * modelWidth;
            int probabilityOffset = workItem * probabilityMatrixLength;

            for (int query = 0; query < sequence; query++)
            {
                int queryOffset = projectedBatchOffset
                    + query * projectedWidth
                    + headOffset;
                int probabilityRow =
                    probabilityOffset + query * sequence;
                int lastKey = causal ? query : sequence - 1;
                float maximum = float.NegativeInfinity;

                for (int key = 0; key <= lastKey; key++)
                {
                    int keyOffset = projectedBatchOffset
                        + key * projectedWidth
                        + modelWidth
                        + headOffset;
                    float score = scale * DotProduct(
                        _data,
                        queryOffset,
                        _data,
                        keyOffset,
                        headWidth);
                    probabilities[probabilityRow + key] = score;
                    if (score > maximum)
                        maximum = score;
                }

                int activeKeyCount = lastKey + 1;
                float sum = ExpShiftedValues(
                    probabilities,
                    probabilityRow,
                    maximum,
                    probabilities,
                    probabilityRow,
                    activeKeyCount);
                MultiplyValues(
                    probabilities,
                    probabilityRow,
                    1f / sum,
                    probabilities,
                    probabilityRow,
                    activeKeyCount);
                int outputOffset = outputBatchOffset
                    + query * modelWidth
                    + headOffset;
                for (int key = 0; key <= lastKey; key++)
                {
                    float probability = probabilities[probabilityRow + key];
                    float valueProbability = directBFloat16Gradients
                        ? TensorStorageCodec.RoundToBFloat16(probability)
                        : probability;
                    int valueOffset = projectedBatchOffset
                        + key * projectedWidth
                        + 2 * modelWidth
                        + headOffset;
                    AddScaledValues(
                        output,
                        outputOffset,
                        _data,
                        valueOffset,
                        valueProbability,
                        headWidth);
                }
            }
        }

        RunBatches(
            workItemCount,
            (long)sequence * sequence * headWidth,
            ForwardHead);

        int[] outputShape = Rank == 3
            ? [batch, sequence, modelWidth]
            : [sequence, modelWidth];
        var result = new Tensor(output, outputShape, [this]);
        result.EnsureHostGradientStorage();
        result.Node.BackwardAction = () =>
        {
            result.EnsureHostGradientStorage();
            EnsureHostGradientStorage();
            void BackwardHead(int workItem)
            {
                int batchIndex = workItem / numHeads;
                int head = workItem % numHeads;
                int headOffset = head * headWidth;
                int projectedBatchOffset =
                    batchIndex * sequence * projectedWidth;
                int outputBatchOffset =
                    batchIndex * sequence * modelWidth;
                int probabilityOffset =
                    workItem * probabilityMatrixLength;
                Span<float> probabilityGradients = sequence <= 256
                    ? stackalloc float[sequence]
                    : new float[sequence];

                for (int query = 0; query < sequence; query++)
                {
                    int queryOffset = projectedBatchOffset
                        + query * projectedWidth
                        + headOffset;
                    int outputOffset = outputBatchOffset
                        + query * modelWidth
                        + headOffset;
                    int probabilityRow =
                        probabilityOffset + query * sequence;
                    int lastKey = causal ? query : sequence - 1;
                    float softmaxDot = 0f;
                    float rowDelta = 0f;
                    if (directBFloat16Gradients)
                    {
                        for (int column = 0; column < headWidth; column++)
                        {
                            rowDelta += result._grad[outputOffset + column]
                                * result._data[outputOffset + column];
                        }
                    }

                    for (int key = 0; key <= lastKey; key++)
                    {
                        int valueOffset = projectedBatchOffset
                            + key * projectedWidth
                            + 2 * modelWidth
                            + headOffset;
                        float probability =
                            probabilities[probabilityRow + key];
                        float probabilityGradient;
                        if (directBFloat16Gradients)
                        {
                            probabilityGradient = 0f;
                            for (int column = 0; column < headWidth; column++)
                            {
                                probabilityGradient +=
                                    TensorStorageCodec.RoundToBFloat16(
                                        result._grad[outputOffset + column])
                                    * _data[valueOffset + column];
                            }
                        }
                        else
                        {
                            probabilityGradient = DotProduct(
                                result._grad,
                                outputOffset,
                                _data,
                                valueOffset,
                                headWidth);
                        }
                        probabilityGradients[key] = probabilityGradient;
                        softmaxDot += probabilityGradient * probability;
                        if (directBFloat16Gradients)
                        {
                            float valueProbability =
                                TensorStorageCodec.RoundToBFloat16(probability);
                            for (int column = 0; column < headWidth; column++)
                            {
                                _grad[valueOffset + column] +=
                                    TensorStorageCodec.RoundToBFloat16(
                                        result._grad[outputOffset + column])
                                    * valueProbability;
                            }
                        }
                        else
                        {
                            AddScaledValues(
                                _grad,
                                valueOffset,
                                result._grad,
                                outputOffset,
                                probability,
                                headWidth);
                        }
                    }

                    for (int key = 0; key <= lastKey; key++)
                    {
                        int keyOffset = projectedBatchOffset
                            + key * projectedWidth
                            + modelWidth
                            + headOffset;
                        float unscaledScoreGradient =
                            probabilities[probabilityRow + key]
                            * (probabilityGradients[key]
                                - (directBFloat16Gradients
                                    ? rowDelta
                                    : softmaxDot));
                        if (directBFloat16Gradients)
                        {
                            unscaledScoreGradient =
                                TensorStorageCodec.RoundToBFloat16(
                                    unscaledScoreGradient);
                        }
                        float scoreGradient = scale * unscaledScoreGradient;
                        AddScaledValues(
                            _grad,
                            queryOffset,
                            _data,
                            keyOffset,
                            scoreGradient,
                            headWidth);
                        AddScaledValues(
                            _grad,
                            keyOffset,
                            _data,
                            queryOffset,
                            scoreGradient,
                            headWidth);
                    }
                }
            }

            RunBatches(
                workItemCount,
                (long)sequence * sequence * headWidth,
                BackwardHead);
            if (directBFloat16Gradients)
            {
                for (int index = 0; index < _grad.Length; index++)
                {
                    _grad[index] = TensorStorageCodec.RoundToBFloat16(
                        _grad[index]);
                }
            }
        };

        return result;
    }
}
