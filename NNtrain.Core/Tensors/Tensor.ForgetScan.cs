namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Applies sigmoid forget/input gates, a tanh value gate, and an inclusive
    /// associative affine prefix scan to a [batch, sequence, 3 * width]
    /// projection.
    /// </summary>
    public Tensor FusedForgetScan()
    {
        CheckRank(3);
        int batch = _shape[0];
        int sequence = _shape[1];
        int channels = _shape[2];
        if (channels % 3 != 0)
        {
            throw new InvalidOperationException(
                "ForgetScan projection width must be three times the model width.");
        }

        int width = channels / 3;
        if (ExecutionDevice == TensorDevice.Cuda
            && DType is TensorDType.Float32
                or TensorDType.BFloat16
                or TensorDType.Bfp8)
        {
            return ForgetScanCuda(batch, sequence, width);
        }
        ThrowIfCudaHostFallback(nameof(FusedForgetScan));
        int rows = checked(batch * sequence);
        int stateLength = checked(rows * width);
        var memory = new float[stateLength];
        int tileWidth = GetForgetScanTileWidth(batch, width);
        int tileCount = (width + tileWidth - 1) / tileWidth;
        int workItems = checked(batch * tileCount);

        if (!AutogradContext.IsRecordingEnabled)
        {
            void ScanTile(int workItem)
            {
                int batchIndex = workItem / tileCount;
                int tile = workItem % tileCount;
                int channelStart = tile * tileWidth;
                int channelCount = Math.Min(tileWidth, width - channelStart);
                int projectedBatchOffset = batchIndex * sequence * channels;
                int memoryBatchOffset = batchIndex * sequence * width;
                ForwardForgetScanInferenceTile(
                    _data,
                    memory,
                    projectedBatchOffset,
                    memoryBatchOffset,
                    channels,
                    width,
                    sequence,
                    channelStart,
                    channelCount);
            }

            RunBatches(
                workItems,
                (long)sequence * tileWidth * 4,
                ScanTile);
            return new Tensor(memory, [batch, sequence, width], [this]);
        }

        var forget = new float[stateLength];
        var input = new float[stateLength];
        var value = new float[stateLength];
        void ForwardTile(int workItem)
        {
            int batchIndex = workItem / tileCount;
            int tile = workItem % tileCount;
            int channelStart = tile * tileWidth;
            int channelCount = Math.Min(tileWidth, width - channelStart);
            ForwardForgetScanTrainingTile(
                _data,
                memory,
                forget,
                input,
                value,
                batchIndex * sequence * channels,
                batchIndex * sequence * width,
                channels,
                width,
                sequence,
                channelStart,
                channelCount);
        }

        RunBatches(
            workItems,
            (long)sequence * tileWidth * 7,
            ForwardTile);

        var result = new Tensor(memory, [batch, sequence, width], [this]);
        result.Node.BackwardAction = () =>
        {
            void BackwardTile(int workItem)
            {
                int batchIndex = workItem / tileCount;
                int tile = workItem % tileCount;
                int channelStart = tile * tileWidth;
                int channelCount = Math.Min(tileWidth, width - channelStart);
                int projectedBatchOffset = batchIndex * sequence * channels;
                int stateBatchOffset = batchIndex * sequence * width;
                BackwardForgetScanTile(
                    _grad,
                    result._grad,
                    memory,
                    forget,
                    input,
                    value,
                    projectedBatchOffset,
                    stateBatchOffset,
                    channels,
                    width,
                    sequence,
                    channelStart,
                    channelCount);
            }

            RunBatches(
                workItems,
                (long)sequence * tileWidth * 8,
                BackwardTile);
        };
        return result;
    }

    private static int GetForgetScanTileWidth(int batch, int width)
    {
        int tilesPerBatch = Math.Max(
            1,
            Math.Min(
                width,
                (EffectiveMaxDegreeOfParallelism + batch - 1) / batch));
        int unaligned = (width + tilesPerBatch - 1) / tilesPerBatch;
        return Math.Min(width, Math.Max(32, (unaligned + 31) / 32 * 32));
    }

    private static void ForwardForgetScanTrainingTile(
        TensorStorage projected,
        float[] memory,
        float[] forget,
        float[] input,
        float[] value,
        int projectedBatchOffset,
        int stateBatchOffset,
        int channels,
        int width,
        int sequence,
        int channelStart,
        int channelCount)
    {
        for (int time = 0; time < sequence; time++)
        {
            int projectedOffset = projectedBatchOffset
                + time * channels
                + channelStart;
            int stateOffset = stateBatchOffset
                + time * width
                + channelStart;
            int previousOffset = stateOffset - width;
            int index = 0;
            if (CanUseSimd(channelCount))
            {
                int vectorWidth = Vector256<float>.Count;
                int vectorEnd = channelCount - channelCount % vectorWidth;
                for (; index < vectorEnd; index += vectorWidth)
                {
                    Vector256<float> f = SigmoidValues(
                        LoadVector256(projected, projectedOffset + index));
                    Vector256<float> i = SigmoidValues(LoadVector256(
                        projected,
                        projectedOffset + width + index));
                    Vector256<float> v = TanhValues(LoadVector256(
                        projected,
                        projectedOffset + 2 * width + index));
                    Vector256<float> previous = time == 0
                        ? Vector256<float>.Zero
                        : LoadVector256(memory, previousOffset + index);
                    StoreVector256(f, forget, stateOffset + index);
                    StoreVector256(i, input, stateOffset + index);
                    StoreVector256(v, value, stateOffset + index);
                    StoreVector256(
                        Vector256.FusedMultiplyAdd(f, previous, i * v),
                        memory,
                        stateOffset + index);
                }
            }

            if (CanUseVector128(channelCount - index))
            {
                Vector128<float> f = SigmoidValues(
                    LoadVector128(projected, projectedOffset + index));
                Vector128<float> i = SigmoidValues(LoadVector128(
                    projected,
                    projectedOffset + width + index));
                Vector128<float> v = TanhValues(LoadVector128(
                    projected,
                    projectedOffset + 2 * width + index));
                Vector128<float> previous = time == 0
                    ? Vector128<float>.Zero
                    : LoadVector128(memory, previousOffset + index);
                StoreVector128(f, forget, stateOffset + index);
                StoreVector128(i, input, stateOffset + index);
                StoreVector128(v, value, stateOffset + index);
                StoreVector128(
                    Vector128.FusedMultiplyAdd(f, previous, i * v),
                    memory,
                    stateOffset + index);
                index += Vector128<float>.Count;
            }

            for (; index < channelCount; index++)
            {
                float f = Sigmoid(projected[projectedOffset + index]);
                float i = Sigmoid(projected[projectedOffset + width + index]);
                float v = MathF.Tanh(
                    projected[projectedOffset + 2 * width + index]);
                float previous = time == 0
                    ? 0f
                    : memory[previousOffset + index];
                forget[stateOffset + index] = f;
                input[stateOffset + index] = i;
                value[stateOffset + index] = v;
                memory[stateOffset + index] = f * previous + i * v;
            }
        }
    }

    private static void ForwardForgetScanInferenceTile(
        TensorStorage projected,
        float[] memory,
        int projectedBatchOffset,
        int stateBatchOffset,
        int channels,
        int width,
        int sequence,
        int channelStart,
        int channelCount)
    {
        for (int time = 0; time < sequence; time++)
        {
            int projectedOffset = projectedBatchOffset
                + time * channels
                + channelStart;
            int stateOffset = stateBatchOffset
                + time * width
                + channelStart;
            int previousOffset = stateOffset - width;
            int index = 0;
            if (CanUseSimd(channelCount))
            {
                int vectorWidth = Vector256<float>.Count;
                int vectorEnd = channelCount - channelCount % vectorWidth;
                for (; index < vectorEnd; index += vectorWidth)
                {
                    Vector256<float> f = SigmoidValues(
                        LoadVector256(projected, projectedOffset + index));
                    Vector256<float> i = SigmoidValues(LoadVector256(
                        projected,
                        projectedOffset + width + index));
                    Vector256<float> v = TanhValues(LoadVector256(
                        projected,
                        projectedOffset + 2 * width + index));
                    Vector256<float> previous = time == 0
                        ? Vector256<float>.Zero
                        : LoadVector256(memory, previousOffset + index);
                    StoreVector256(
                        Vector256.FusedMultiplyAdd(f, previous, i * v),
                        memory,
                        stateOffset + index);
                }
            }

            if (CanUseVector128(channelCount - index))
            {
                Vector128<float> f = SigmoidValues(
                    LoadVector128(projected, projectedOffset + index));
                Vector128<float> i = SigmoidValues(LoadVector128(
                    projected,
                    projectedOffset + width + index));
                Vector128<float> v = TanhValues(LoadVector128(
                    projected,
                    projectedOffset + 2 * width + index));
                Vector128<float> previous = time == 0
                    ? Vector128<float>.Zero
                    : LoadVector128(memory, previousOffset + index);
                StoreVector128(
                    Vector128.FusedMultiplyAdd(f, previous, i * v),
                    memory,
                    stateOffset + index);
                index += Vector128<float>.Count;
            }

            for (; index < channelCount; index++)
            {
                float f = Sigmoid(projected[projectedOffset + index]);
                float i = Sigmoid(projected[projectedOffset + width + index]);
                float v = MathF.Tanh(
                    projected[projectedOffset + 2 * width + index]);
                float previous = time == 0
                    ? 0f
                    : memory[previousOffset + index];
                memory[stateOffset + index] = f * previous + i * v;
            }
        }
    }

    private static void BackwardForgetScanTile(
        float[] projectedGradient,
        float[] memoryGradient,
        float[] memory,
        float[] forget,
        float[] input,
        float[] value,
        int projectedBatchOffset,
        int stateBatchOffset,
        int channels,
        int width,
        int sequence,
        int channelStart,
        int channelCount)
    {
        int index = 0;
        if (CanUseSimd(channelCount))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorEnd = channelCount - channelCount % vectorWidth;
            Vector256<float> one = Vector256.Create(1f);
            for (; index < vectorEnd; index += vectorWidth)
            {
                int channel = channelStart + index;
                Vector256<float> running = Vector256<float>.Zero;
                for (int time = sequence - 1; time >= 0; time--)
                {
                    int stateOffset = stateBatchOffset + time * width + channel;
                    int projectedOffset = projectedBatchOffset
                        + time * channels
                        + channel;
                    Vector256<float> total = running
                        + LoadVector256(memoryGradient, stateOffset);
                    Vector256<float> f = LoadVector256(forget, stateOffset);
                    Vector256<float> i = LoadVector256(input, stateOffset);
                    Vector256<float> v = LoadVector256(value, stateOffset);
                    Vector256<float> previous = time == 0
                        ? Vector256<float>.Zero
                        : LoadVector256(memory, stateOffset - width);
                    Vector256<float> forgetGradient =
                        total * previous * f * (one - f);
                    Vector256<float> inputGradient =
                        total * v * i * (one - i);
                    Vector256<float> valueGradient =
                        total * i * (one - v * v);
                    StoreVector256(
                        LoadVector256(projectedGradient, projectedOffset)
                            + forgetGradient,
                        projectedGradient,
                        projectedOffset);
                    StoreVector256(
                        LoadVector256(
                            projectedGradient,
                            projectedOffset + width)
                            + inputGradient,
                        projectedGradient,
                        projectedOffset + width);
                    StoreVector256(
                        LoadVector256(
                            projectedGradient,
                            projectedOffset + 2 * width)
                            + valueGradient,
                        projectedGradient,
                        projectedOffset + 2 * width);
                    running = total * f;
                }
            }
        }

        if (CanUseVector128(channelCount - index))
        {
            int channel = channelStart + index;
            Vector128<float> one = Vector128.Create(1f);
            Vector128<float> running = Vector128<float>.Zero;
            for (int time = sequence - 1; time >= 0; time--)
            {
                int stateOffset = stateBatchOffset + time * width + channel;
                int projectedOffset = projectedBatchOffset
                    + time * channels
                    + channel;
                Vector128<float> total = running
                    + LoadVector128(memoryGradient, stateOffset);
                Vector128<float> f = LoadVector128(forget, stateOffset);
                Vector128<float> i = LoadVector128(input, stateOffset);
                Vector128<float> v = LoadVector128(value, stateOffset);
                Vector128<float> previous = time == 0
                    ? Vector128<float>.Zero
                    : LoadVector128(memory, stateOffset - width);
                StoreVector128(
                    LoadVector128(projectedGradient, projectedOffset)
                        + total * previous * f * (one - f),
                    projectedGradient,
                    projectedOffset);
                StoreVector128(
                    LoadVector128(projectedGradient, projectedOffset + width)
                        + total * v * i * (one - i),
                    projectedGradient,
                    projectedOffset + width);
                StoreVector128(
                    LoadVector128(
                        projectedGradient,
                        projectedOffset + 2 * width)
                        + total * i * (one - v * v),
                    projectedGradient,
                    projectedOffset + 2 * width);
                running = total * f;
            }
            index += Vector128<float>.Count;
        }

        for (; index < channelCount; index++)
        {
            int channel = channelStart + index;
            float running = 0f;
            for (int time = sequence - 1; time >= 0; time--)
            {
                int stateOffset = stateBatchOffset + time * width + channel;
                int projectedOffset = projectedBatchOffset
                    + time * channels
                    + channel;
                float total = running + memoryGradient[stateOffset];
                float f = forget[stateOffset];
                float i = input[stateOffset];
                float v = value[stateOffset];
                float previous = time == 0
                    ? 0f
                    : memory[stateOffset - width];
                projectedGradient[projectedOffset] +=
                    total * previous * f * (1f - f);
                projectedGradient[projectedOffset + width] +=
                    total * v * i * (1f - i);
                projectedGradient[projectedOffset + 2 * width] +=
                    total * i * (1f - v * v);
                running = total * f;
            }
        }
    }

    private static float Sigmoid(float value)
        => 1f / (1f + MathF.Exp(-value));

    private static Vector256<float> SigmoidValues(Vector256<float> value)
    {
        Vector256<float> one = Vector256.Create(1f);
        return one / (one + FastExp(Vector256<float>.Zero - value));
    }

    private static Vector128<float> SigmoidValues(Vector128<float> value)
    {
        Vector128<float> one = Vector128.Create(1f);
        return one / (one + FastExp(Vector128<float>.Zero - value));
    }

    private static Vector256<float> TanhValues(Vector256<float> value)
    {
        Vector256<float> one = Vector256.Create(1f);
        Vector256<float> exponential = FastExp(Vector256.Create(-2f) * value);
        return Vector256.Create(2f) / (one + exponential) - one;
    }

    private static Vector128<float> TanhValues(Vector128<float> value)
    {
        Vector128<float> one = Vector128.Create(1f);
        Vector128<float> exponential = FastExp(Vector128.Create(-2f) * value);
        return Vector128.Create(2f) / (one + exponential) - one;
    }

    private static Vector256<float> FastExp(Vector256<float> value)
    {
        Vector256<float> x = Vector256.Min(
            Vector256.Max(value, Vector256.Create(-80f)),
            Vector256.Create(80f));
        Vector256<float> exponent = Vector256.Round(
            x * Vector256.Create(1.4426950408889634f));
        Vector256<float> remainder = Vector256.FusedMultiplyAdd(
            exponent,
            Vector256.Create(-0.693145751953125f),
            x);
        remainder = Vector256.FusedMultiplyAdd(
            exponent,
            Vector256.Create(-1.428606765330187e-6f),
            remainder);

        Vector256<float> polynomial = Vector256.Create(1f / 720f);
        polynomial = Vector256.FusedMultiplyAdd(
            polynomial,
            remainder,
            Vector256.Create(1f / 120f));
        polynomial = Vector256.FusedMultiplyAdd(
            polynomial,
            remainder,
            Vector256.Create(1f / 24f));
        polynomial = Vector256.FusedMultiplyAdd(
            polynomial,
            remainder,
            Vector256.Create(1f / 6f));
        polynomial = Vector256.FusedMultiplyAdd(
            polynomial,
            remainder,
            Vector256.Create(0.5f));
        polynomial = Vector256.FusedMultiplyAdd(
            polynomial,
            remainder,
            Vector256.Create(1f));
        polynomial = Vector256.FusedMultiplyAdd(
            polynomial,
            remainder,
            Vector256.Create(1f));

        Vector256<int> exponentBits = Vector256.ShiftLeft(
            Vector256.ConvertToInt32(exponent) + Vector256.Create(127),
            23);
        return polynomial * exponentBits.AsSingle();
    }

    private static Vector128<float> FastExp(Vector128<float> value)
    {
        Vector128<float> x = Vector128.Min(
            Vector128.Max(value, Vector128.Create(-80f)),
            Vector128.Create(80f));
        Vector128<float> exponent = Vector128.Round(
            x * Vector128.Create(1.4426950408889634f));
        Vector128<float> remainder = Vector128.FusedMultiplyAdd(
            exponent,
            Vector128.Create(-0.693145751953125f),
            x);
        remainder = Vector128.FusedMultiplyAdd(
            exponent,
            Vector128.Create(-1.428606765330187e-6f),
            remainder);

        Vector128<float> polynomial = Vector128.Create(1f / 720f);
        polynomial = Vector128.FusedMultiplyAdd(
            polynomial,
            remainder,
            Vector128.Create(1f / 120f));
        polynomial = Vector128.FusedMultiplyAdd(
            polynomial,
            remainder,
            Vector128.Create(1f / 24f));
        polynomial = Vector128.FusedMultiplyAdd(
            polynomial,
            remainder,
            Vector128.Create(1f / 6f));
        polynomial = Vector128.FusedMultiplyAdd(
            polynomial,
            remainder,
            Vector128.Create(0.5f));
        polynomial = Vector128.FusedMultiplyAdd(
            polynomial,
            remainder,
            Vector128.Create(1f));
        polynomial = Vector128.FusedMultiplyAdd(
            polynomial,
            remainder,
            Vector128.Create(1f));

        Vector128<int> exponentBits = Vector128.ShiftLeft(
            Vector128.ConvertToInt32(exponent) + Vector128.Create(127),
            23);
        return polynomial * exponentBits.AsSingle();
    }
}
