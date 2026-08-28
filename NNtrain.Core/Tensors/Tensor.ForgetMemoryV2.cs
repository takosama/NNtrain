namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Applies a causal, matrix-valued associative memory with a stable
    /// retention gate and a delta-rule write.
    /// </summary>
    /// <remarks>
    /// The packed input layout is [q, k, v, gate, beta]. Query and key are
    /// transformed elementwise to tanh(x) / sqrt(keyWidth). For each value row,
    /// g = floor + (1 - floor) sigmoid(gate),
    /// write = (1 - g) sigmoid(beta), and with the normalized key k,
    /// M[t] = g M[t-1] + write (v - M[t-1] k) k^T.
    /// The returned value is M[t] q. The recurrence stays sequential in time,
    /// while independent batches and the dense key dimension use parallel and
    /// SIMD kernels.
    /// </remarks>
    public Tensor ForgetMemoryV2(
        int keyWidth,
        int valueWidth,
        float retentionFloor)
        => ForgetMemory(
            keyWidth,
            valueWidth,
            retentionFloor,
            useV3: false);

    /// <summary>
    /// Applies the V3 matrix memory recurrence. V3 independently controls
    /// retention and writing, predicts from the retained memory, and uses an
    /// L2-normalized key.
    /// </summary>
    public Tensor ForgetMemoryV3(
        int keyWidth,
        int valueWidth,
        float retentionFloor)
        => ForgetMemory(
            keyWidth,
            valueWidth,
            retentionFloor,
            useV3: true,
            useDrn: false);

    /// <summary>
    /// Applies the delta, read-before-write, normalized-query/key memory
    /// recurrence while leaving V2 and V3 behavior unchanged.
    /// </summary>
    public Tensor ForgetMemoryDRN(
        int keyWidth,
        int valueWidth,
        float retentionFloor)
        => ForgetMemory(
            keyWidth,
            valueWidth,
            retentionFloor,
            useV3: false,
            useDrn: true);

    private Tensor ForgetMemory(
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        bool useV3,
        bool useDrn = false)
    {
        CheckRank(3);
        if (keyWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(keyWidth));
        if (valueWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(valueWidth));
        if (!float.IsFinite(retentionFloor)
            || retentionFloor < 0f
            || retentionFloor >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(retentionFloor),
                retentionFloor,
                "Retention floor must be finite and in [0, 1).");
        }

        int expectedProjectionWidth = checked(2 * keyWidth + 3 * valueWidth);
        if (_shape[2] != expectedProjectionWidth)
        {
            throw new InvalidOperationException(
                $"ForgetMemoryV2 projection width must be " +
                $"2 * keyWidth + 3 * valueWidth = " +
                $"{expectedProjectionWidth}.");
        }

        int batch = _shape[0];
        int sequence = _shape[1];
        int projectionWidth = _shape[2];
        int matrixSize = checked(valueWidth * keyWidth);
        if (ExecutionDevice == TensorDevice.Cuda)
        {
            return ForgetMemoryCuda(
                batch,
                sequence,
                projectionWidth,
                keyWidth,
                valueWidth,
                retentionFloor,
                useV3,
                useDrn);
        }
        var output = new float[checked(batch * sequence * valueWidth)];

        void ForwardBatch(int batchIndex)
        {
            var state = new float[matrixSize];
            ForwardForgetMemoryV2Batch(
                _data,
                output,
                state,
                batchIndex,
                sequence,
                projectionWidth,
                keyWidth,
                valueWidth,
                retentionFloor,
                states: null,
                useV3,
                useDrn);
        }
        RunBatches(
            batch,
            (long)sequence * matrixSize * 4,
            ForwardBatch);

        var result = new Tensor(
            output,
            [batch, sequence, valueWidth],
            [this]);
        if (!AutogradContext.IsRecordingEnabled)
            return result;

        result.Node.BackwardAction = () =>
        {
            void BackwardBatch(int batchIndex)
            {
                var states = new float[checked(sequence * matrixSize)];
                var finalState = new float[matrixSize];
                ForwardForgetMemoryV2Batch(
                    _data,
                    output: null,
                    finalState,
                    batchIndex,
                    sequence,
                    projectionWidth,
                    keyWidth,
                    valueWidth,
                    retentionFloor,
                    states,
                    useV3,
                    useDrn);
                BackwardForgetMemoryV2Batch(
                    _data,
                    _grad,
                    result._grad,
                    states,
                    batchIndex,
                    sequence,
                    projectionWidth,
                    keyWidth,
                    valueWidth,
                    retentionFloor,
                    useV3,
                    useDrn);
            }

            RunBatches(
                batch,
                (long)sequence * matrixSize * 12,
                BackwardBatch);
        };

        return result;
    }

    private Tensor ForgetMemoryCuda(
        int batch,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        bool useV3,
        bool useDrn)
    {
        if (DType == TensorDType.Bfp8)
        {
            return ForgetMemoryBfp8Cuda(
                batch,
                sequence,
                projectionWidth,
                keyWidth,
                valueWidth,
                retentionFloor,
                useV3,
                useDrn);
        }

        bool bfloat16Compute = DType == TensorDType.BFloat16;
        NNtrain.ForgetMemoryV2Cuda.ResidentForwardResult forward =
            NNtrain.ForgetMemoryV2Cuda.ForwardResident(
            this,
            batch,
            sequence,
            projectionWidth,
            keyWidth,
            valueWidth,
            retentionFloor,
            bfloat16Compute,
            useV3,
            useDrn);
        Tensor result = forward.OutputBFloat16 is not null
            ? FromCudaResult(
                forward.OutputBFloat16,
                forward.DeviceIndex,
                [batch, sequence, valueWidth],
                [this],
                TensorDType.BFloat16)
            : FromCudaResult(
                forward.OutputFloat32!,
                forward.DeviceIndex,
                [batch, sequence, valueWidth],
                [this]);
        if (!AutogradContext.IsRecordingEnabled)
        {
            forward.Dispose();
            return result;
        }

        AutogradLease<NNtrain.ForgetMemoryV2Cuda.ResidentForwardResult> lease =
            AutogradLease<NNtrain.ForgetMemoryV2Cuda
                .ResidentForwardResult>.Own(
                forward,
                AutogradLeaseMetadata.CudaOwned(
                    forward.DeviceIndex,
                    result.DType,
                    DataVersion),
                static saved => saved.Dispose());
        result.Node.SetBackward(lease, savedContext =>
        {
            NNtrain.ForgetMemoryV2Cuda.BackwardResident(
                this,
                result,
                savedContext,
                batch,
                sequence,
                projectionWidth,
                keyWidth,
                valueWidth,
                retentionFloor,
                bfloat16Compute,
                useV3,
                useDrn);
        });
        return result;
    }

    /// <summary>
    /// Runs the same recurrence as <see cref="ForgetMemoryV2"/> but starts
    /// from <paramref name="state"/> and leaves the final memory in it, so a
    /// caller can advance one token at a time without replaying the prefix.
    /// </summary>
    /// <remarks>
    /// Inference only. The tensor returned carries no backward action, so
    /// recording must be disabled; the batched entry point remains the one
    /// used for training.
    /// </remarks>
    internal Tensor ForgetMemoryV2Continue(
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        float[] state)
        => ForgetMemoryContinue(
            keyWidth,
            valueWidth,
            retentionFloor,
            state,
            useV3: false);

    internal Tensor ForgetMemoryV3Continue(
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        float[] state)
        => ForgetMemoryContinue(
            keyWidth,
            valueWidth,
            retentionFloor,
            state,
            useV3: true,
            useDrn: false);

    internal Tensor ForgetMemoryDRNContinue(
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        float[] state)
        => ForgetMemoryContinue(
            keyWidth,
            valueWidth,
            retentionFloor,
            state,
            useV3: false,
            useDrn: true);

    internal Tensor ForgetMemoryV2Continue(
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        ForgetMemoryRecurrentMemory state)
        => ForgetMemoryContinue(
            keyWidth,
            valueWidth,
            retentionFloor,
            state,
            useV3: false);

    internal Tensor ForgetMemoryV3Continue(
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        ForgetMemoryRecurrentMemory state)
        => ForgetMemoryContinue(
            keyWidth,
            valueWidth,
            retentionFloor,
            state,
            useV3: true,
            useDrn: false);

    internal Tensor ForgetMemoryDRNContinue(
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        ForgetMemoryRecurrentMemory state)
        => ForgetMemoryContinue(
            keyWidth,
            valueWidth,
            retentionFloor,
            state,
            useV3: false,
            useDrn: true);

    private Tensor ForgetMemoryContinue(
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        ForgetMemoryRecurrentMemory state,
        bool useV3,
        bool useDrn = false)
    {
        ArgumentNullException.ThrowIfNull(state);
        ValidateForgetMemoryContinuation(
            keyWidth,
            valueWidth,
            retentionFloor,
            state.Length);

        if (ExecutionDevice != TensorDevice.Cuda)
        {
            float[] hostState = state.HostForCpuMutation();
            try
            {
                return ForgetMemoryContinue(
                    keyWidth,
                    valueWidth,
                    retentionFloor,
                    hostState,
                    useV3,
                    useDrn);
            }
            finally
            {
                state.MarkHostMutated();
            }
        }

        int deviceIndex = CudaDeviceIndex;
        NativeCudaBuffer<float> cudaState =
            state.EnsureCudaBuffer(deviceIndex);
        try
        {
            return DType == TensorDType.Bfp8
                ? ForgetMemoryBfp8ContinueCuda(
                    _shape[1],
                    _shape[2],
                    keyWidth,
                    valueWidth,
                    retentionFloor,
                    cudaState,
                    useV3,
                    useDrn)
                : ForgetMemoryContinueCuda(
                    _shape[1],
                    _shape[2],
                    keyWidth,
                    valueWidth,
                    retentionFloor,
                    cudaState,
                    useV3,
                    useDrn);
        }
        finally
        {
            // A launch may have updated the borrowed state before a later
            // output conversion fails. Keep authority truthful on both the
            // success and rollback paths.
            state.MarkCudaMutated(deviceIndex);
        }
    }

    private Tensor ForgetMemoryContinue(
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        float[] state,
        bool useV3,
        bool useDrn = false)
    {
        ArgumentNullException.ThrowIfNull(state);
        ValidateForgetMemoryContinuation(
            keyWidth,
            valueWidth,
            retentionFloor,
            state.Length);

        int sequence = _shape[1];
        if (ExecutionDevice == TensorDevice.Cuda
            && DType == TensorDType.Bfp8)
        {
            return ForgetMemoryBfp8ContinueCuda(
                sequence,
                _shape[2],
                keyWidth,
                valueWidth,
                retentionFloor,
                state,
                useV3,
                useDrn);
        }

        var output = new float[checked(sequence * valueWidth)];
        ForwardForgetMemoryV2Batch(
            _data,
            output,
            state,
            batchIndex: 0,
            sequence,
            _shape[2],
            keyWidth,
            valueWidth,
            retentionFloor,
            states: null,
            useV3,
            useDrn);
        return new Tensor(output, [1, sequence, valueWidth], [this]);
    }

    private void ValidateForgetMemoryContinuation(
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        int stateLength)
    {
        CheckRank(3);
        if (keyWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(keyWidth));
        if (valueWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(valueWidth));
        if (!float.IsFinite(retentionFloor)
            || retentionFloor < 0f
            || retentionFloor >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(retentionFloor),
                retentionFloor,
                "Retention floor must be finite and in [0, 1).");
        }
        if (AutogradContext.IsRecordingEnabled)
        {
            throw new InvalidOperationException(
                "ForgetMemoryV2Continue is an inference path and cannot "
                + "record gradients. Wrap the call in torch.no_grad().");
        }

        int expectedProjectionWidth = checked(2 * keyWidth + 3 * valueWidth);
        if (_shape[2] != expectedProjectionWidth)
        {
            throw new InvalidOperationException(
                $"ForgetMemoryV2 projection width must be "
                + $"2 * keyWidth + 3 * valueWidth = "
                + $"{expectedProjectionWidth}.");
        }
        if (_shape[0] != 1)
        {
            throw new InvalidOperationException(
                "Recurrent stepping carries one memory, so the batch "
                + "dimension must be 1.");
        }
        if (stateLength != checked(valueWidth * keyWidth))
        {
            throw new ArgumentException(
                $"The recurrent state must hold valueWidth * keyWidth = "
                + $"{valueWidth * keyWidth} values.",
                "state");
        }
    }

    private Tensor ForgetMemoryContinueCuda(
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        NativeCudaBuffer<float> state,
        bool useV3,
        bool useDrn)
    {
        bool bfloat16Compute = DType == TensorDType.BFloat16;
        ForgetMemoryV2Cuda.ResidentForwardResult forward =
            ForgetMemoryV2Cuda.ForwardResident(
                this,
                batch: 1,
                sequence,
                projectionWidth,
                keyWidth,
                valueWidth,
                retentionFloor,
                bfloat16Compute,
                useV3,
                useDrn,
                recurrentState: state);
        Tensor result = forward.OutputBFloat16 is not null
            ? FromCudaResult(
                forward.OutputBFloat16,
                forward.DeviceIndex,
                [1, sequence, valueWidth],
                [this],
                TensorDType.BFloat16)
            : FromCudaResult(
                forward.OutputFloat32!,
                forward.DeviceIndex,
                [1, sequence, valueWidth],
                [this]);
        forward.Dispose();
        return result;
    }

    private static void ForwardForgetMemoryV2Batch(
        TensorStorage projected,
        float[]? output,
        float[] state,
        int batchIndex,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        float[]? states,
        bool useV3,
        bool useDrn)
    {
        int projectedBatchOffset = batchIndex * sequence * projectionWidth;
        int outputBatchOffset = batchIndex * sequence * valueWidth;
        int matrixSize = valueWidth * keyWidth;
        float inverseSqrtKeyWidth = 1f / MathF.Sqrt(keyWidth);
        var normalizedQuery = new float[keyWidth];
        var normalizedKey = new float[keyWidth];

        for (int time = 0; time < sequence; time++)
        {
            int projectedOffset =
                projectedBatchOffset + time * projectionWidth;
            int queryOffset = projectedOffset;
            int keyOffset = queryOffset + keyWidth;
            int valueOffset = keyOffset + keyWidth;
            int gateOffset = valueOffset + valueWidth;
            int betaOffset = gateOffset + valueWidth;

            for (int keyIndex = 0; keyIndex < keyWidth; keyIndex++)
            {
                float queryTanh = MathF.Tanh(
                    projected[queryOffset + keyIndex]);
                normalizedQuery[keyIndex] = useDrn
                    ? queryTanh
                    : queryTanh * inverseSqrtKeyWidth;
                float keyTanh = MathF.Tanh(
                    projected[keyOffset + keyIndex]);
                normalizedKey[keyIndex] = useV3 || useDrn
                    ? keyTanh
                    : keyTanh * inverseSqrtKeyWidth;
            }
            if (useDrn)
            {
                NormalizeForgetMemoryVector(normalizedQuery, 1e-8f);
                NormalizeForgetMemoryVector(normalizedKey, 1e-8f);
            }
            else if (useV3)
            {
                NormalizeForgetMemoryVector(normalizedKey, 1e-6f);
            }

            for (int valueIndex = 0; valueIndex < valueWidth; valueIndex++)
            {
                int stateRowOffset = valueIndex * keyWidth;
                float gateSigmoid = ForgetMemorySigmoid(
                    projected[gateOffset + valueIndex]);
                float retention = useDrn
                    ? gateSigmoid
                    : retentionFloor
                        + (1f - retentionFloor) * gateSigmoid;
                float beta = ForgetMemorySigmoid(
                    projected[betaOffset + valueIndex]);
                float value = MathF.Tanh(
                    projected[valueOffset + valueIndex]);

                if (useDrn && output is not null)
                {
                    int outputOffset = outputBatchOffset
                        + time * valueWidth;
                    output[outputOffset + valueIndex] = DotProduct(
                        state,
                        stateRowOffset,
                        normalizedQuery,
                        0,
                        keyWidth);
                }

                float predictedValue = DotProduct(
                    state, stateRowOffset, normalizedKey, 0, keyWidth);
                if (useV3)
                    predictedValue *= retention;

                float error = value - predictedValue;
                float write = useV3 || useDrn
                    ? beta
                    : (1f - retention) * beta;
                float delta = write * error;
                for (int keyIndex = 0; keyIndex < keyWidth; keyIndex++)
                {
                    int stateIndex = stateRowOffset + keyIndex;
                    state[stateIndex] = retention * state[stateIndex]
                        + delta * normalizedKey[keyIndex];
                }
            }

            if (!useDrn && output is not null)
            {
                int outputOffset = outputBatchOffset + time * valueWidth;
                for (int valueIndex = 0;
                    valueIndex < valueWidth;
                    valueIndex++)
                {
                    int stateRowOffset = valueIndex * keyWidth;
                    output[outputOffset + valueIndex] = DotProduct(
                        state, stateRowOffset, normalizedQuery, 0, keyWidth);
                }
            }

            if (states is not null)
            {
                state.AsSpan().CopyTo(
                    states.AsSpan(time * matrixSize, matrixSize));
            }
        }
    }

    private static void BackwardForgetMemoryV2Batch(
        TensorStorage projected,
        float[] projectedGradient,
        float[] outputGradient,
        float[] states,
        int batchIndex,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        bool useV3,
        bool useDrn)
    {
        int projectedBatchOffset = batchIndex * sequence * projectionWidth;
        int outputBatchOffset = batchIndex * sequence * valueWidth;
        int matrixSize = valueWidth * keyWidth;
        var stateGradient = new float[matrixSize];
        var previousStateGradient = new float[matrixSize];
        float inverseSqrtKeyWidth = 1f / MathF.Sqrt(keyWidth);
        var normalizedQuery = new float[keyWidth];
        var normalizedKey = new float[keyWidth];
        var queryDerivative = new float[keyWidth];
        var keyDerivative = new float[keyWidth];
        var queryTanhValues = new float[keyWidth];
        var keyTanhValues = new float[keyWidth];
        var normalizedQueryGradient = new float[keyWidth];
        var normalizedKeyGradient = new float[keyWidth];

        for (int time = sequence - 1; time >= 0; time--)
        {
            int projectedOffset =
                projectedBatchOffset + time * projectionWidth;
            int queryOffset = projectedOffset;
            int keyOffset = queryOffset + keyWidth;
            int valueOffset = keyOffset + keyWidth;
            int gateOffset = valueOffset + valueWidth;
            int betaOffset = gateOffset + valueWidth;
            int outputOffset = outputBatchOffset + time * valueWidth;
            int currentStateOffset = time * matrixSize;
            int previousStateOffset = (time - 1) * matrixSize;

            for (int keyIndex = 0; keyIndex < keyWidth; keyIndex++)
            {
                float queryTanh = MathF.Tanh(
                    projected[queryOffset + keyIndex]);
                float keyTanh = MathF.Tanh(
                    projected[keyOffset + keyIndex]);
                queryTanhValues[keyIndex] = queryTanh;
                keyTanhValues[keyIndex] = keyTanh;
                normalizedQuery[keyIndex] = useDrn
                    ? queryTanh
                    : queryTanh * inverseSqrtKeyWidth;
                queryDerivative[keyIndex] = (1f - queryTanh * queryTanh)
                    * (useDrn ? 1f : inverseSqrtKeyWidth);
                normalizedKey[keyIndex] = useV3 || useDrn
                    ? keyTanh
                    : keyTanh * inverseSqrtKeyWidth;
                keyDerivative[keyIndex] = (1f - keyTanh * keyTanh)
                    * (useV3 || useDrn ? 1f : inverseSqrtKeyWidth);
            }
            float queryNorm = 1f;
            float keyNorm = 1f;
            if (useDrn)
            {
                queryNorm = NormalizeForgetMemoryVector(
                    normalizedQuery,
                    1e-8f);
                keyNorm = NormalizeForgetMemoryVector(
                    normalizedKey,
                    1e-8f);
            }
            else if (useV3)
            {
                keyNorm = NormalizeForgetMemoryVector(
                    normalizedKey,
                    1e-6f);
            }

            Array.Clear(previousStateGradient);
            Array.Clear(normalizedQueryGradient);
            Array.Clear(normalizedKeyGradient);

            // DRN reads M[t-1]; V2/V3 read M[t].
            for (int valueIndex = 0; valueIndex < valueWidth; valueIndex++)
            {
                int stateRowOffset = valueIndex * keyWidth;
                float recalledGradient =
                    outputGradient[outputOffset + valueIndex];
                for (int keyIndex = 0; keyIndex < keyWidth; keyIndex++)
                {
                    float memory = useDrn
                        ? time == 0
                            ? 0f
                            : states[previousStateOffset
                                + stateRowOffset + keyIndex]
                        : states[currentStateOffset
                            + stateRowOffset + keyIndex];
                    if (useDrn)
                    {
                        normalizedQueryGradient[keyIndex] +=
                            memory * recalledGradient;
                        previousStateGradient[stateRowOffset + keyIndex] +=
                            normalizedQuery[keyIndex] * recalledGradient;
                    }
                    else
                    {
                        projectedGradient[queryOffset + keyIndex] +=
                            memory * recalledGradient
                            * queryDerivative[keyIndex];
                        stateGradient[stateRowOffset + keyIndex] +=
                            normalizedQuery[keyIndex] * recalledGradient;
                    }
                }
            }

            // Differentiate the stable forget + delta update.
            for (int valueIndex = 0; valueIndex < valueWidth; valueIndex++)
            {
                int stateRowOffset = valueIndex * keyWidth;
                float gateSigmoid = ForgetMemorySigmoid(
                    projected[gateOffset + valueIndex]);
                float retention = useDrn
                    ? gateSigmoid
                    : retentionFloor
                        + (1f - retentionFloor) * gateSigmoid;
                float beta = ForgetMemorySigmoid(
                    projected[betaOffset + valueIndex]);
                float write = useV3 || useDrn
                    ? beta
                    : (1f - retention) * beta;
                float value = MathF.Tanh(
                    projected[valueOffset + valueIndex]);
                float predictedValue = 0f;
                float stateGradientDotKey = 0f;
                float retentionGradient = 0f;
                for (int keyIndex = 0; keyIndex < keyWidth; keyIndex++)
                {
                    float previous = time == 0
                        ? 0f
                        : states[previousStateOffset + stateRowOffset + keyIndex];
                    float gradient = stateGradient[stateRowOffset + keyIndex];
                    predictedValue += previous * normalizedKey[keyIndex];
                    stateGradientDotKey += gradient * normalizedKey[keyIndex];
                    retentionGradient += gradient * previous;
                }

                float retainedPrediction = useV3
                    ? retention * predictedValue
                    : predictedValue;
                float error = value - retainedPrediction;
                float writeGradient = error * stateGradientDotKey;
                float errorGradient = write * stateGradientDotKey;
                if (useV3)
                    retentionGradient -= errorGradient * predictedValue;
                else if (!useDrn)
                    retentionGradient -= writeGradient * beta;
                projectedGradient[valueOffset + valueIndex] +=
                    errorGradient * (1f - value * value);
                projectedGradient[gateOffset + valueIndex] +=
                    retentionGradient
                    * (useDrn ? 1f : 1f - retentionFloor)
                    * gateSigmoid
                    * (1f - gateSigmoid);
                projectedGradient[betaOffset + valueIndex] +=
                    writeGradient
                    * (useV3 || useDrn ? 1f : 1f - retention)
                    * beta
                    * (1f - beta);

                for (int keyIndex = 0; keyIndex < keyWidth; keyIndex++)
                {
                    float previous = time == 0
                        ? 0f
                        : states[previousStateOffset + stateRowOffset + keyIndex];
                    float gradient = stateGradient[stateRowOffset + keyIndex];
                    float keyGradient = gradient * write * error
                        - previous * errorGradient
                            * (useV3 ? retention : 1f);
                    normalizedKeyGradient[keyIndex] += keyGradient;
                    float recurrentPreviousGradient = useV3
                        ? retention * (gradient
                            - normalizedKey[keyIndex] * errorGradient)
                        : gradient * retention
                            - normalizedKey[keyIndex] * errorGradient;
                    if (useDrn)
                    {
                        previousStateGradient[stateRowOffset + keyIndex] +=
                            recurrentPreviousGradient;
                    }
                    else
                    {
                        previousStateGradient[stateRowOffset + keyIndex] =
                            recurrentPreviousGradient;
                    }
                }
            }

            if (useDrn)
            {
                AccumulateNormalizedTanhGradient(
                    projectedGradient,
                    queryOffset,
                    queryTanhValues,
                    queryDerivative,
                    normalizedQueryGradient,
                    queryNorm);
            }

            if (useV3 || useDrn)
            {
                AccumulateNormalizedTanhGradient(
                    projectedGradient,
                    keyOffset,
                    keyTanhValues,
                    keyDerivative,
                    normalizedKeyGradient,
                    keyNorm);
            }
            else
            {
                for (int keyIndex = 0; keyIndex < keyWidth; keyIndex++)
                {
                    projectedGradient[keyOffset + keyIndex] +=
                        normalizedKeyGradient[keyIndex]
                        * keyDerivative[keyIndex];
                }
            }

            (stateGradient, previousStateGradient) =
                (previousStateGradient, stateGradient);
        }
    }

    private static float ForgetMemorySigmoid(float value)
    {
        if (value >= 0f)
        {
            float exponential = MathF.Exp(-value);
            return 1f / (1f + exponential);
        }

        float positiveExponential = MathF.Exp(value);
        return positiveExponential / (1f + positiveExponential);
    }

    private static float NormalizeForgetMemoryVector(
        float[] vector,
        float epsilon)
    {
        float squaredNorm = epsilon;
        for (int index = 0; index < vector.Length; index++)
            squaredNorm += vector[index] * vector[index];
        float norm = MathF.Sqrt(squaredNorm);
        float inverseNorm = 1f / norm;
        for (int index = 0; index < vector.Length; index++)
            vector[index] *= inverseNorm;
        return norm;
    }

    private static void AccumulateNormalizedTanhGradient(
        float[] projectedGradient,
        int projectedOffset,
        float[] tanhValues,
        float[] tanhDerivatives,
        float[] normalizedGradient,
        float norm)
    {
        float tanhDotGradient = 0f;
        for (int index = 0; index < tanhValues.Length; index++)
            tanhDotGradient += tanhValues[index] * normalizedGradient[index];

        float inverseNorm = 1f / norm;
        float inverseNormCubed = inverseNorm * inverseNorm * inverseNorm;
        for (int index = 0; index < tanhValues.Length; index++)
        {
            float tanhGradient = normalizedGradient[index] * inverseNorm
                - tanhValues[index] * tanhDotGradient * inverseNormCubed;
            projectedGradient[projectedOffset + index] +=
                tanhGradient * tanhDerivatives[index];
        }
    }

    private static void UpdateForgetMemoryState(
        float[] state,
        int stateOffset,
        TensorStorage key,
        int keyOffset,
        float retention,
        float delta,
        int length,
        bool bfloat16Compute = false)
    {
        if (bfloat16Compute)
        {
            for (int bf16Index = 0; bf16Index < length; bf16Index++)
            {
                int stateIndex = stateOffset + bf16Index;
                float retained = BFloat16Compute(
                    retention * state[stateIndex]);
                float written = BFloat16Compute(
                    delta * key[keyOffset + bf16Index]);
                state[stateIndex] = BFloat16Compute(retained + written);
            }
            return;
        }

        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> retentionVector = Vector256.Create(retention);
            Vector256<float> deltaVector = Vector256.Create(delta);
            for (; index < vectorizedLength; index += width)
            {
                StoreVector256(
                    Vector256.FusedMultiplyAdd(
                        LoadVector256(state, stateOffset + index),
                        retentionVector,
                        LoadVector256(key, keyOffset + index) * deltaVector),
                    state,
                    stateOffset + index);
            }
        }
        if (CanUseVector128(length - index))
        {
            StoreVector128(
                Vector128.FusedMultiplyAdd(
                    LoadVector128(state, stateOffset + index),
                    Vector128.Create(retention),
                    LoadVector128(key, keyOffset + index)
                        * Vector128.Create(delta)),
                state,
                stateOffset + index);
            index += Vector128<float>.Count;
        }
        for (; index < length; index++)
        {
            int stateIndex = stateOffset + index;
            state[stateIndex] = retention * state[stateIndex]
                + delta * key[keyOffset + index];
        }
    }

    private static float DotProductBFloat16(
        float[] left,
        int leftOffset,
        TensorStorage right,
        int rightOffset,
        int length)
    {
        float sum = 0f;
        for (int index = 0; index < length; index++)
        {
            float product = BFloat16Compute(
                left[leftOffset + index] * right[rightOffset + index]);
            sum = BFloat16Compute(sum + product);
        }
        return sum;
    }

    private static float BFloat16Compute(float value)
        => TensorStorageCodec.RoundToBFloat16Compute(value);

    private static void ComputeForgetMemoryBackwardDots(
        TensorStorage key,
        int keyOffset,
        float[] previousState,
        int previousStateOffset,
        float[] stateGradient,
        int stateGradientOffset,
        int length,
        bool hasPreviousState,
        out float predictedValue,
        out float stateGradientDotKey,
        out float retentionGradient)
    {
        if (!hasPreviousState)
        {
            predictedValue = 0f;
            retentionGradient = 0f;
            stateGradientDotKey = DotProduct(
                stateGradient,
                stateGradientOffset,
                key,
                keyOffset,
                length);
            return;
        }

        int index = 0;
        predictedValue = 0f;
        stateGradientDotKey = 0f;
        retentionGradient = 0f;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> predictedVector = Vector256<float>.Zero;
            Vector256<float> stateKeyVector = Vector256<float>.Zero;
            Vector256<float> retentionVector = Vector256<float>.Zero;
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> keyVector =
                    LoadVector256(key, keyOffset + index);
                Vector256<float> previousVector = LoadVector256(
                    previousState,
                    previousStateOffset + index);
                Vector256<float> gradientVector = LoadVector256(
                    stateGradient,
                    stateGradientOffset + index);
                predictedVector = Vector256.FusedMultiplyAdd(
                    previousVector,
                    keyVector,
                    predictedVector);
                stateKeyVector = Vector256.FusedMultiplyAdd(
                    gradientVector,
                    keyVector,
                    stateKeyVector);
                retentionVector = Vector256.FusedMultiplyAdd(
                    gradientVector,
                    previousVector,
                    retentionVector);
            }
            predictedValue += Vector256.Sum(predictedVector);
            stateGradientDotKey += Vector256.Sum(stateKeyVector);
            retentionGradient += Vector256.Sum(retentionVector);
        }
        if (CanUseVector128(length - index))
        {
            Vector128<float> keyVector =
                LoadVector128(key, keyOffset + index);
            Vector128<float> previousVector = LoadVector128(
                previousState,
                previousStateOffset + index);
            Vector128<float> gradientVector = LoadVector128(
                stateGradient,
                stateGradientOffset + index);
            predictedValue += Vector128.Sum(previousVector * keyVector);
            stateGradientDotKey += Vector128.Sum(
                gradientVector * keyVector);
            retentionGradient += Vector128.Sum(
                gradientVector * previousVector);
            index += Vector128<float>.Count;
        }
        for (; index < length; index++)
        {
            float keyValue = key[keyOffset + index];
            float previousValue =
                previousState[previousStateOffset + index];
            float gradientValue =
                stateGradient[stateGradientOffset + index];
            predictedValue += previousValue * keyValue;
            stateGradientDotKey += gradientValue * keyValue;
            retentionGradient += gradientValue * previousValue;
        }
    }

    private static void AccumulateForgetMemoryBackwardVectors(
        float[] keyGradient,
        int keyGradientOffset,
        float[] previousStateGradient,
        int previousStateGradientOffset,
        TensorStorage key,
        int keyOffset,
        float[] previousState,
        int previousStateOffset,
        float[] stateGradient,
        int stateGradientOffset,
        int length,
        bool hasPreviousState,
        float outerGradientScale,
        float errorGradient,
        float retention)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> outerScale =
                Vector256.Create(outerGradientScale);
            Vector256<float> negativeError =
                Vector256.Create(-errorGradient);
            Vector256<float> retentionVector = Vector256.Create(retention);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> gradientVector = LoadVector256(
                    stateGradient,
                    stateGradientOffset + index);
                Vector256<float> keyVector =
                    LoadVector256(key, keyOffset + index);
                Vector256<float> previousVector = hasPreviousState
                    ? LoadVector256(
                        previousState,
                        previousStateOffset + index)
                    : Vector256<float>.Zero;
                Vector256<float> keyGradientVector =
                    Vector256.FusedMultiplyAdd(
                        gradientVector,
                        outerScale,
                        LoadVector256(
                            keyGradient,
                            keyGradientOffset + index));
                keyGradientVector = Vector256.FusedMultiplyAdd(
                    previousVector,
                    negativeError,
                    keyGradientVector);
                StoreVector256(
                    keyGradientVector,
                    keyGradient,
                    keyGradientOffset + index);
                StoreVector256(
                    Vector256.FusedMultiplyAdd(
                        keyVector,
                        negativeError,
                        gradientVector * retentionVector),
                    previousStateGradient,
                    previousStateGradientOffset + index);
            }
        }
        if (CanUseVector128(length - index))
        {
            Vector128<float> gradientVector = LoadVector128(
                stateGradient,
                stateGradientOffset + index);
            Vector128<float> keyVector =
                LoadVector128(key, keyOffset + index);
            Vector128<float> previousVector = hasPreviousState
                ? LoadVector128(previousState, previousStateOffset + index)
                : Vector128<float>.Zero;
            Vector128<float> keyGradientVector =
                Vector128.FusedMultiplyAdd(
                    gradientVector,
                    Vector128.Create(outerGradientScale),
                    LoadVector128(keyGradient, keyGradientOffset + index));
            keyGradientVector = Vector128.FusedMultiplyAdd(
                previousVector,
                Vector128.Create(-errorGradient),
                keyGradientVector);
            StoreVector128(
                keyGradientVector,
                keyGradient,
                keyGradientOffset + index);
            StoreVector128(
                Vector128.FusedMultiplyAdd(
                    keyVector,
                    Vector128.Create(-errorGradient),
                    gradientVector * Vector128.Create(retention)),
                previousStateGradient,
                previousStateGradientOffset + index);
            index += Vector128<float>.Count;
        }
        for (; index < length; index++)
        {
            float previousValue = hasPreviousState
                ? previousState[previousStateOffset + index]
                : 0f;
            float gradientValue =
                stateGradient[stateGradientOffset + index];
            keyGradient[keyGradientOffset + index] +=
                gradientValue * outerGradientScale
                - errorGradient * previousValue;
            previousStateGradient[previousStateGradientOffset + index] =
                gradientValue * retention
                - errorGradient * key[keyOffset + index];
        }
    }
}
