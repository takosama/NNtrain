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
            return ForgetMemoryV2Cuda(
                batch,
                sequence,
                projectionWidth,
                keyWidth,
                valueWidth,
                retentionFloor);
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
                states: null);
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
                    states);
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
                    retentionFloor);
            }

            RunBatches(
                batch,
                (long)sequence * matrixSize * 12,
                BackwardBatch);
        };

        return result;
    }

    private Tensor ForgetMemoryV2Cuda(
        int batch,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor)
    {
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
            bfloat16Compute);
        Tensor result = FromCudaResult(
            forward.Output,
            forward.DeviceIndex,
            [batch, sequence, valueWidth],
            [this]);
        if (!AutogradContext.IsRecordingEnabled)
        {
            forward.Dispose();
            return result;
        }

        result.Node.RegisterResource(forward);
        result.Node.BackwardAction = () =>
        {
            NNtrain.ForgetMemoryV2Cuda.BackwardResident(
                this,
                result,
                forward,
                batch,
                sequence,
                projectionWidth,
                keyWidth,
                valueWidth,
                retentionFloor,
                bfloat16Compute);
        };
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
    {
        CheckRank(3);
        ArgumentNullException.ThrowIfNull(state);
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
        if (state.Length != checked(valueWidth * keyWidth))
        {
            throw new ArgumentException(
                $"The recurrent state must hold valueWidth * keyWidth = "
                + $"{valueWidth * keyWidth} values.",
                nameof(state));
        }

        int sequence = _shape[1];
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
            states: null);
        return new Tensor(output, [1, sequence, valueWidth], [this]);
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
        float[]? states)
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
                normalizedQuery[keyIndex] = MathF.Tanh(
                    projected[queryOffset + keyIndex]) * inverseSqrtKeyWidth;
                normalizedKey[keyIndex] = MathF.Tanh(
                    projected[keyOffset + keyIndex]) * inverseSqrtKeyWidth;
            }

            for (int valueIndex = 0; valueIndex < valueWidth; valueIndex++)
            {
                int stateRowOffset = valueIndex * keyWidth;
                float gateSigmoid = ForgetMemorySigmoid(
                    projected[gateOffset + valueIndex]);
                float retention = retentionFloor
                    + (1f - retentionFloor) * gateSigmoid;
                float beta = ForgetMemorySigmoid(
                    projected[betaOffset + valueIndex]);
                float write = (1f - retention) * beta;
                float value = MathF.Tanh(
                    projected[valueOffset + valueIndex]);
                float predictedValue = DotProduct(
                    state, stateRowOffset, normalizedKey, 0, keyWidth);

                float error = value - predictedValue;
                float delta = write * error;
                for (int keyIndex = 0; keyIndex < keyWidth; keyIndex++)
                {
                    int stateIndex = stateRowOffset + keyIndex;
                    state[stateIndex] = retention * state[stateIndex]
                        + delta * normalizedKey[keyIndex];
                }
            }

            if (output is not null)
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
        float retentionFloor)
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
                normalizedQuery[keyIndex] = queryTanh * inverseSqrtKeyWidth;
                normalizedKey[keyIndex] = keyTanh * inverseSqrtKeyWidth;
                queryDerivative[keyIndex] =
                    (1f - queryTanh * queryTanh) * inverseSqrtKeyWidth;
                keyDerivative[keyIndex] =
                    (1f - keyTanh * keyTanh) * inverseSqrtKeyWidth;
            }

            Array.Clear(previousStateGradient);

            // r[t] = M[t] q[t].
            for (int valueIndex = 0; valueIndex < valueWidth; valueIndex++)
            {
                int stateRowOffset = valueIndex * keyWidth;
                float recalledGradient =
                    outputGradient[outputOffset + valueIndex];
                for (int keyIndex = 0; keyIndex < keyWidth; keyIndex++)
                {
                    projectedGradient[queryOffset + keyIndex] +=
                        states[currentStateOffset + stateRowOffset + keyIndex]
                        * recalledGradient * queryDerivative[keyIndex];
                    stateGradient[stateRowOffset + keyIndex] +=
                        normalizedQuery[keyIndex] * recalledGradient;
                }
            }

            // Differentiate the stable forget + delta update.
            for (int valueIndex = 0; valueIndex < valueWidth; valueIndex++)
            {
                int stateRowOffset = valueIndex * keyWidth;
                float gateSigmoid = ForgetMemorySigmoid(
                    projected[gateOffset + valueIndex]);
                float retention = retentionFloor
                    + (1f - retentionFloor) * gateSigmoid;
                float beta = ForgetMemorySigmoid(
                    projected[betaOffset + valueIndex]);
                float write = (1f - retention) * beta;
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

                float error = value - predictedValue;
                float writeGradient = error * stateGradientDotKey;
                float errorGradient = write * stateGradientDotKey;
                retentionGradient -= writeGradient * beta;
                projectedGradient[valueOffset + valueIndex] +=
                    errorGradient * (1f - value * value);
                projectedGradient[gateOffset + valueIndex] +=
                    retentionGradient
                    * (1f - retentionFloor)
                    * gateSigmoid
                    * (1f - gateSigmoid);
                projectedGradient[betaOffset + valueIndex] +=
                    writeGradient
                    * (1f - retention)
                    * beta
                    * (1f - beta);

                for (int keyIndex = 0; keyIndex < keyWidth; keyIndex++)
                {
                    float previous = time == 0
                        ? 0f
                        : states[previousStateOffset + stateRowOffset + keyIndex];
                    float gradient = stateGradient[stateRowOffset + keyIndex];
                    float normalizedKeyGradient =
                        gradient * write * error - previous * errorGradient;
                    projectedGradient[keyOffset + keyIndex] +=
                        normalizedKeyGradient * keyDerivative[keyIndex];
                    previousStateGradient[stateRowOffset + keyIndex] =
                        gradient * retention
                        - normalizedKey[keyIndex] * errorGradient;
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
