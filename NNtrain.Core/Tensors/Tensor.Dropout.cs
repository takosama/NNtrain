namespace NNtrain;

partial class Tensor
{
    public Tensor Dropout(float probability, Random? random = null)
    {
        if (!float.IsFinite(probability)
            || probability < 0f
            || probability >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(probability),
                probability,
                "Dropout probability must be finite and in [0, 1).");
        }

        if (probability == 0f)
            return this;

        random ??= Random.Shared;
        uint seed = NextDropoutSeed(random);
        float scale = 1f / (1f - probability);
        uint dropThreshold = (uint)(probability * (uint.MaxValue + 1d));
        int columns = _shape[^1];
        int rows = Numel / columns;

        if (ExecutionDevice == TensorDevice.Cuda)
        {
            if (DType == TensorDType.BFloat16)
            {
                var bfloat16Buffer = CudaOperationProfiler.IsEnabled
                    ? CudaOperationProfiler.Measure(
                        "forward.dropout",
                        () => TensorCudaKernels.DropoutForwardBFloat16Resident(
                            this, seed, dropThreshold, scale))
                    : TensorCudaKernels.DropoutForwardBFloat16Resident(
                        this, seed, dropThreshold, scale);
                Tensor bfloat16Result = FromCudaResult(
                    bfloat16Buffer,
                    CudaDeviceIndex,
                    _shape,
                    [this],
                    TensorDType.BFloat16);
                if (AutogradContext.IsRecordingEnabled)
                {
                    bfloat16Result.Node.BackwardAction = () =>
                        TensorCudaKernels.DropoutBackwardResident(
                            bfloat16Result,
                            this,
                            seed,
                            dropThreshold,
                            scale);
                }
                return bfloat16Result;
            }
            var cudaBuffer = CudaOperationProfiler.IsEnabled
                ? CudaOperationProfiler.Measure(
                    "forward.dropout",
                    () => TensorCudaKernels.DropoutForwardResident(
                        this,
                        seed,
                        dropThreshold,
                        scale))
                : TensorCudaKernels.DropoutForwardResident(
                    this,
                    seed,
                    dropThreshold,
                    scale);
            Tensor cudaResult = FromCudaResult(
                cudaBuffer,
                CudaDeviceIndex,
                _shape,
                [this]);
            if (AutogradContext.IsRecordingEnabled)
            {
                cudaResult.Node.BackwardAction = () =>
                {
                    if (CudaOperationProfiler.IsEnabled)
                    {
                        CudaOperationProfiler.Measure(
                            "backward.dropout",
                            () => TensorCudaKernels.DropoutBackwardResident(
                                cudaResult,
                                this,
                                seed,
                                dropThreshold,
                                scale));
                    }
                    else
                    {
                        TensorCudaKernels.DropoutBackwardResident(
                            cudaResult,
                            this,
                            seed,
                            dropThreshold,
                            scale);
                    }
                };
            }
            return cudaResult;
        }

        var output = new float[Numel];
        void ForwardRow(int row)
        {
            int offset = row * columns;
            ApplyDropoutValues(
                _data,
                output,
                offset,
                columns,
                seed,
                dropThreshold,
                scale);
        }

        RunBatches(rows, columns * 8L, ForwardRow);

        var result = new Tensor(output, _shape, new[] { this });
        result.Node.BackwardAction = () =>
        {
            void BackwardRow(int row)
            {
                int offset = row * columns;
                AccumulateDropoutGradient(
                    _grad,
                    result._grad,
                    offset,
                    columns,
                    seed,
                    dropThreshold,
                    scale);
            }

            RunBatches(rows, columns * 3L, BackwardRow);
        };

        return result;
    }

    /// <summary>
    /// Computes <c>residual + dropout(branch)</c> as one allocation and one
    /// autograd node. A counter-based mask is generated per row so large
    /// activations use all configured workers instead of serial random-number
    /// generation. Backward regenerates the same mask from its seed rather
    /// than retaining another activation-sized array.
    /// </summary>
    public Tensor AddDropout(
        Tensor branch,
        float probability,
        Random? random = null)
    {
        ArgumentNullException.ThrowIfNull(branch);
        if (!float.IsFinite(probability)
            || probability < 0f
            || probability >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(probability),
                probability,
                "Dropout probability must be finite and in [0, 1).");
        }
        if (!_shape.AsSpan().SequenceEqual(branch._shape))
        {
            throw new ArgumentException(
                "Residual and dropout branch must have identical shapes.",
                nameof(branch));
        }
        if (probability == 0f)
            return this + branch;

        random ??= Random.Shared;
        uint seed = NextDropoutSeed(random);
        float scale = 1f / (1f - probability);
        uint dropThreshold = (uint)(probability * (uint.MaxValue + 1d));
        int columns = _shape[^1];
        int rows = Numel / columns;

        if (ExecutionDevice == TensorDevice.Cuda)
        {
            if (DType == TensorDType.BFloat16
                && branch.DType == TensorDType.BFloat16)
            {
                var bfloat16Buffer = CudaOperationProfiler.IsEnabled
                    ? CudaOperationProfiler.Measure(
                        "forward.residual_dropout",
                        () => TensorCudaKernels.AddDropoutForwardBFloat16Resident(
                            this, branch, seed, dropThreshold, scale))
                    : TensorCudaKernels.AddDropoutForwardBFloat16Resident(
                        this, branch, seed, dropThreshold, scale);
                Tensor bfloat16Result = FromCudaResult(
                    bfloat16Buffer,
                    CudaDeviceIndex,
                    _shape,
                    [this, branch],
                    TensorDType.BFloat16);
                if (AutogradContext.IsRecordingEnabled)
                {
                    bfloat16Result.Node.BackwardAction = () =>
                        TensorCudaKernels.AddDropoutBackwardResident(
                            bfloat16Result,
                            this,
                            branch,
                            ReferenceEquals(this, branch),
                            seed,
                            dropThreshold,
                            scale);
                }
                return bfloat16Result;
            }
            var cudaBuffer = CudaOperationProfiler.IsEnabled
                ? CudaOperationProfiler.Measure(
                    "forward.residual_dropout",
                    () => TensorCudaKernels.AddDropoutForwardResident(
                        this,
                        branch,
                        seed,
                        dropThreshold,
                        scale))
                : TensorCudaKernels.AddDropoutForwardResident(
                    this,
                    branch,
                    seed,
                    dropThreshold,
                    scale);
            Tensor cudaResult = FromCudaResult(
                cudaBuffer,
                CudaDeviceIndex,
                _shape,
                [this, branch]);
            if (AutogradContext.IsRecordingEnabled)
            {
                cudaResult.Node.BackwardAction = () =>
                {
                    if (CudaOperationProfiler.IsEnabled)
                    {
                        CudaOperationProfiler.Measure(
                            "backward.residual_dropout",
                            () => TensorCudaKernels.AddDropoutBackwardResident(
                                cudaResult,
                                this,
                                branch,
                                ReferenceEquals(this, branch),
                                seed,
                                dropThreshold,
                                scale));
                    }
                    else
                    {
                        TensorCudaKernels.AddDropoutBackwardResident(
                            cudaResult,
                            this,
                            branch,
                            ReferenceEquals(this, branch),
                            seed,
                            dropThreshold,
                            scale);
                    }
                };
            }
            return cudaResult;
        }

        var output = new float[Numel];
        void ForwardRow(int row)
        {
            int offset = row * columns;
            AddDropoutValues(
                _data,
                branch._data,
                output,
                offset,
                columns,
                seed,
                dropThreshold,
                scale);
        }

        RunBatches(rows, columns * 10L, ForwardRow);

        var result = new Tensor(output, _shape, new[] { this, branch });
        result.Node.BackwardAction = () =>
        {
            bool sameParent = ReferenceEquals(this, branch);

            void BackwardRow(int row)
            {
                int offset = row * columns;
                AccumulateResidualDropoutGradient(
                    _grad,
                    sameParent ? _grad : branch._grad,
                    result._grad,
                    offset,
                    columns,
                    sameParent,
                    seed,
                    dropThreshold,
                    scale);
            }

            RunBatches(rows, columns * 5L, BackwardRow);
        };

        return result;
    }

    private static void ApplyDropoutValues(
        TensorStorage input,
        float[] output,
        int offset,
        int length,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            for (; index < vectorizedLength; index += width)
            {
                StoreVector256(
                    LoadVector256(input, offset + index)
                        * CreateDropoutMask256(
                            seed,
                            offset + index,
                            dropThreshold,
                            scale),
                    output,
                    offset + index);
            }
        }
        if (CanUseVector128(length - index))
        {
            StoreVector128(
                LoadVector128(input, offset + index)
                    * CreateDropoutMask128(
                        seed,
                        offset + index,
                        dropThreshold,
                        scale),
                output,
                offset + index);
            index += Vector128<float>.Count;
        }
        for (; index < length; index++)
        {
            int valueIndex = offset + index;
            output[valueIndex] = input[valueIndex]
                * DropoutMultiplier(
                    seed,
                    valueIndex,
                    dropThreshold,
                    scale);
        }
    }

    private static void AddDropoutValues(
        TensorStorage residual,
        TensorStorage branch,
        float[] output,
        int offset,
        int length,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            for (; index < vectorizedLength; index += width)
            {
                StoreVector256(
                    Vector256.FusedMultiplyAdd(
                        LoadVector256(branch, offset + index),
                        CreateDropoutMask256(
                            seed,
                            offset + index,
                            dropThreshold,
                            scale),
                        LoadVector256(residual, offset + index)),
                    output,
                    offset + index);
            }
        }
        if (CanUseVector128(length - index))
        {
            StoreVector128(
                Vector128.FusedMultiplyAdd(
                    LoadVector128(branch, offset + index),
                    CreateDropoutMask128(
                        seed,
                        offset + index,
                        dropThreshold,
                        scale),
                    LoadVector128(residual, offset + index)),
                output,
                offset + index);
            index += Vector128<float>.Count;
        }
        for (; index < length; index++)
        {
            int valueIndex = offset + index;
            output[valueIndex] = residual[valueIndex]
                + branch[valueIndex]
                    * DropoutMultiplier(
                        seed,
                        valueIndex,
                        dropThreshold,
                        scale);
        }
    }

    private static void AccumulateDropoutGradient(
        float[] destination,
        float[] gradient,
        int offset,
        int length,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            for (; index < vectorizedLength; index += width)
            {
                StoreVector256(
                    Vector256.FusedMultiplyAdd(
                        LoadVector256(gradient, offset + index),
                        CreateDropoutMask256(
                            seed,
                            offset + index,
                            dropThreshold,
                            scale),
                        LoadVector256(destination, offset + index)),
                    destination,
                    offset + index);
            }
        }
        if (CanUseVector128(length - index))
        {
            StoreVector128(
                Vector128.FusedMultiplyAdd(
                    LoadVector128(gradient, offset + index),
                    CreateDropoutMask128(
                        seed,
                        offset + index,
                        dropThreshold,
                        scale),
                    LoadVector128(destination, offset + index)),
                destination,
                offset + index);
            index += Vector128<float>.Count;
        }
        for (; index < length; index++)
        {
            int valueIndex = offset + index;
            destination[valueIndex] +=
                gradient[valueIndex]
                    * DropoutMultiplier(
                        seed,
                        valueIndex,
                        dropThreshold,
                        scale);
        }
    }

    private static void AccumulateResidualDropoutGradient(
        float[] residualDestination,
        float[] branchDestination,
        float[] gradient,
        int offset,
        int length,
        bool sameParent,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> one = Vector256.Create(1f);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> gradientVector =
                    LoadVector256(gradient, offset + index);
                Vector256<float> maskVector = CreateDropoutMask256(
                    seed,
                    offset + index,
                    dropThreshold,
                    scale);
                StoreVector256(
                    LoadVector256(residualDestination, offset + index)
                        + gradientVector
                            * (sameParent ? one + maskVector : one),
                    residualDestination,
                    offset + index);
                if (!sameParent)
                {
                    StoreVector256(
                        Vector256.FusedMultiplyAdd(
                            gradientVector,
                            maskVector,
                            LoadVector256(
                                branchDestination,
                                offset + index)),
                        branchDestination,
                        offset + index);
                }
            }
        }
        if (CanUseVector128(length - index))
        {
            Vector128<float> gradientVector =
                LoadVector128(gradient, offset + index);
            Vector128<float> maskVector = CreateDropoutMask128(
                seed,
                offset + index,
                dropThreshold,
                scale);
            StoreVector128(
                LoadVector128(residualDestination, offset + index)
                    + gradientVector
                        * (sameParent
                            ? Vector128.Create(1f) + maskVector
                            : Vector128.Create(1f)),
                residualDestination,
                offset + index);
            if (!sameParent)
            {
                StoreVector128(
                    Vector128.FusedMultiplyAdd(
                        gradientVector,
                        maskVector,
                        LoadVector128(branchDestination, offset + index)),
                    branchDestination,
                    offset + index);
            }
            index += Vector128<float>.Count;
        }
        for (; index < length; index++)
        {
            int valueIndex = offset + index;
            float currentGradient = gradient[valueIndex];
            float multiplier = DropoutMultiplier(
                seed,
                valueIndex,
                dropThreshold,
                scale);
            residualDestination[valueIndex] += currentGradient
                * (sameParent ? 1f + multiplier : 1f);
            if (!sameParent)
            {
                branchDestination[valueIndex] +=
                    currentGradient * multiplier;
            }
        }
    }

}
