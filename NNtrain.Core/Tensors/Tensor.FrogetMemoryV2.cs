namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Applies a causal, matrix-valued associative memory with a stable
    /// retention gate and a delta-rule write.
    /// </summary>
    /// <remarks>
    /// The packed input layout is [q, k, v, gate, beta]. For each value row,
    /// g = floor + (1 - floor) sigmoid(gate),
    /// write = (1 - g) sigmoid(beta), and
    /// M[t] = g M[t-1] + write (v - M[t-1] k) k^T.
    /// The returned value is M[t] q. The recurrence stays sequential in time,
    /// while independent batches and the dense key dimension use parallel and
    /// SIMD kernels.
    /// </remarks>
    public Tensor FrogetMemoryV2(
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
                $"FrogetMemoryV2 projection width must be " +
                $"2 * keyWidth + 3 * valueWidth = " +
                $"{expectedProjectionWidth}.");
        }

        int batch = _shape[0];
        int sequence = _shape[1];
        int projectionWidth = _shape[2];
        int matrixSize = checked(valueWidth * keyWidth);
        var output = new float[checked(batch * sequence * valueWidth)];

        void ForwardBatch(int batchIndex)
        {
            var state = new float[matrixSize];
            ForwardFrogetMemoryV2Batch(
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
                ForwardFrogetMemoryV2Batch(
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
                BackwardFrogetMemoryV2Batch(
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

    private static void ForwardFrogetMemoryV2Batch(
        float[] projected,
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

        for (int time = 0; time < sequence; time++)
        {
            int projectedOffset =
                projectedBatchOffset + time * projectionWidth;
            int queryOffset = projectedOffset;
            int keyOffset = queryOffset + keyWidth;
            int valueOffset = keyOffset + keyWidth;
            int gateOffset = valueOffset + valueWidth;
            int betaOffset = gateOffset + valueWidth;

            for (int valueIndex = 0; valueIndex < valueWidth; valueIndex++)
            {
                int stateRowOffset = valueIndex * keyWidth;
                float gateSigmoid = FrogetMemorySigmoid(
                    projected[gateOffset + valueIndex]);
                float retention = retentionFloor
                    + (1f - retentionFloor) * gateSigmoid;
                float beta = FrogetMemorySigmoid(
                    projected[betaOffset + valueIndex]);
                float write = (1f - retention) * beta;
                float value = MathF.Tanh(
                    projected[valueOffset + valueIndex]);
                float predictedValue = DotProduct(
                    state,
                    stateRowOffset,
                    projected,
                    keyOffset,
                    keyWidth);

                float error = value - predictedValue;
                UpdateFrogetMemoryState(
                    state,
                    stateRowOffset,
                    projected,
                    keyOffset,
                    retention,
                    write * error,
                    keyWidth);
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
                        state,
                        stateRowOffset,
                        projected,
                        queryOffset,
                        keyWidth);
                }
            }

            if (states is not null)
            {
                state.AsSpan().CopyTo(
                    states.AsSpan(time * matrixSize, matrixSize));
            }
        }
    }

    private static void BackwardFrogetMemoryV2Batch(
        float[] projected,
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

            Array.Clear(previousStateGradient);

            // r[t] = M[t] q[t].
            for (int valueIndex = 0; valueIndex < valueWidth; valueIndex++)
            {
                int stateRowOffset = valueIndex * keyWidth;
                float recalledGradient =
                    outputGradient[outputOffset + valueIndex];
                AddScaledValues(
                    projectedGradient,
                    queryOffset,
                    states,
                    currentStateOffset + stateRowOffset,
                    recalledGradient,
                    keyWidth);
                AddScaledValues(
                    stateGradient,
                    stateRowOffset,
                    projected,
                    queryOffset,
                    recalledGradient,
                    keyWidth);
            }

            // Differentiate the stable forget + delta update.
            for (int valueIndex = 0; valueIndex < valueWidth; valueIndex++)
            {
                int stateRowOffset = valueIndex * keyWidth;
                float gateSigmoid = FrogetMemorySigmoid(
                    projected[gateOffset + valueIndex]);
                float retention = retentionFloor
                    + (1f - retentionFloor) * gateSigmoid;
                float beta = FrogetMemorySigmoid(
                    projected[betaOffset + valueIndex]);
                float write = (1f - retention) * beta;
                float value = MathF.Tanh(
                    projected[valueOffset + valueIndex]);
                ComputeFrogetMemoryBackwardDots(
                    projected,
                    keyOffset,
                    states,
                    previousStateOffset + stateRowOffset,
                    stateGradient,
                    stateRowOffset,
                    keyWidth,
                    time != 0,
                    out float predictedValue,
                    out float stateGradientDotKey,
                    out float retentionGradient);

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

                AccumulateFrogetMemoryBackwardVectors(
                    projectedGradient,
                    keyOffset,
                    previousStateGradient,
                    stateRowOffset,
                    projected,
                    keyOffset,
                    states,
                    previousStateOffset + stateRowOffset,
                    stateGradient,
                    stateRowOffset,
                    keyWidth,
                    time != 0,
                    write * error,
                    errorGradient,
                    retention);
            }

            (stateGradient, previousStateGradient) =
                (previousStateGradient, stateGradient);
        }
    }

    private static float FrogetMemorySigmoid(float value)
    {
        if (value >= 0f)
        {
            float exponential = MathF.Exp(-value);
            return 1f / (1f + exponential);
        }

        float positiveExponential = MathF.Exp(value);
        return positiveExponential / (1f + positiveExponential);
    }

    private static void UpdateFrogetMemoryState(
        float[] state,
        int stateOffset,
        float[] key,
        int keyOffset,
        float retention,
        float delta,
        int length)
    {
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

    private static void ComputeFrogetMemoryBackwardDots(
        float[] key,
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

    private static void AccumulateFrogetMemoryBackwardVectors(
        float[] keyGradient,
        int keyGradientOffset,
        float[] previousStateGradient,
        int previousStateGradientOffset,
        float[] key,
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
