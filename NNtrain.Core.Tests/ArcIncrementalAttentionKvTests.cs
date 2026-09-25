using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcIncrementalAttentionKvTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Float32, TensorDType.Float32)]
    [InlineData(TensorPrecisionMode.Mix16_32, TensorDType.BFloat16)]
    [InlineData(TensorPrecisionMode.Mix8_32, TensorDType.Bfp8)]
    [InlineData(TensorPrecisionMode.Mix8_16, TensorDType.Bfp8)]
    public void CachedKeysAndValuesUseFullAttentionMatrixOperandRounding(
        TensorPrecisionMode precision, TensorDType storage)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int width = 64, heads = 2, sequence = 3, capacity = 4;
        float[] source = new float[sequence * 3 * width];
        for (int i = 0; i < source.Length; ++i)
            source[i] = MathF.Sin(i * .121f) * .07431f + .00987f;
        var qkv = new Tensor(source, [1, sequence, 3 * width]);
        if (storage == TensorDType.Bfp8)
            qkv.ConvertStorageInPlace(storage, Bfp8QuantizationDescriptor.Block(32));
        else if (storage == TensorDType.BFloat16)
            qkv.ConvertStorageInPlace(storage);
        float[] stored = qkv.Data.ToArray();

        using var execution = Tensor.BeginArcExecution(precision: precision);
        using var noGrad = AutogradContext.NoGrad();
        using var cache = new ArcAttentionKvCache(width, heads, capacity);
        Tensor first = new Tensor(source.AsSpan(0, 2 * 3 * width).ToArray(),
            [1, 2, 3 * width]);
        if (storage == TensorDType.Bfp8)
            first.ConvertStorageInPlace(storage, Bfp8QuantizationDescriptor.Block(32));
        else if (storage == TensorDType.BFloat16)
            first.ConvertStorageInPlace(storage);
        first.ArcPrefillAttentionKvCache(cache, 2);
        Tensor last = new Tensor(source.AsSpan(2 * 3 * width, 3 * width).ToArray(),
            [1, 1, 3 * width]);
        if (storage == TensorDType.Bfp8)
            last.ConvertStorageInPlace(storage, Bfp8QuantizationDescriptor.Block(32));
        else if (storage == TensorDType.BFloat16)
            last.ConvertStorageInPlace(storage);
        _ = last.ArcIncrementalCausalAttention(cache, 2);

        float[] keys = new float[capacity * width], values = new float[capacity * width];
        Tensor.ArcLane.Read(cache.Key, keys);
        Tensor.ArcLane.Read(cache.Value, values);
        for (int token = 0; token < sequence; ++token)
        for (int channel = 0; channel < width; ++channel)
        {
            int head = channel / (width / heads);
            int withinHead = channel % (width / heads);
            int cacheIndex = (head * capacity + token) * (width / heads) + withinHead;
            float expectedKey = stored[token * 3 * width + width + channel];
            float expectedValue = stored[token * 3 * width + 2 * width + channel];
            if (storage == TensorDType.Bfp8)
            {
                expectedKey = TensorStorageCodec.RoundToBFloat16(expectedKey);
                expectedValue = TensorStorageCodec.RoundToBFloat16(expectedValue);
            }
            Assert.Equal(expectedKey, keys[cacheIndex]);
            Assert.Equal(expectedValue, values[cacheIndex]);
        }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32, TensorDType.Float32, 1e-4f)]
    [InlineData(TensorPrecisionMode.Mix16_32, TensorDType.BFloat16, 8e-3f)]
    [InlineData(TensorPrecisionMode.Mix8_32, TensorDType.Bfp8, 3e-2f)]
    [InlineData(TensorPrecisionMode.Mix8_16, TensorDType.Bfp8, 3e-2f)]
    public void PrefillAndIncrementalTokensMatchFullCausalAttention(
        TensorPrecisionMode precision, TensorDType storage, float maximumRelativeRms)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int width = 64, heads = 2, prompt = 5, newTokens = 3, capacity = 12;
        float[] values = new float[(prompt + newTokens) * 3 * width];
        for (int token = 0; token < prompt + newTokens; ++token)
        for (int component = 0; component < 3; ++component)
        for (int channel = 0; channel < width; ++channel)
        {
            int index = token * 3 * width + component * width + channel;
            float phase = (token * 17 + channel * 11) * .019f;
            values[index] = component switch
            {
                0 => MathF.Sin(phase) * .07f,
                1 => MathF.Cos(phase) * .08f,
                _ => .09f + MathF.Sin(phase) * .02f,
            };
        }

        Tensor MakeQkv(int first, int tokens)
        {
            float[] selected = values.AsSpan(first * 3 * width, tokens * 3 * width).ToArray();
            var result = new Tensor(selected, [1, tokens, 3 * width]);
            if (storage == TensorDType.Bfp8)
                result.ConvertStorageInPlace(storage, Bfp8QuantizationDescriptor.Block(32));
            else if (storage == TensorDType.BFloat16)
                result.ConvertStorageInPlace(storage);
            return result;
        }

        using var execution = Tensor.BeginArcExecution(precision: precision);
        using var noGrad = AutogradContext.NoGrad();
        var lane = Tensor.ArcLane;
        using var cache = new ArcAttentionKvCache(width, heads, capacity);
        Tensor initialQkv = MakeQkv(0, prompt);
        long d2h = lane.D2HBytes;
        initialQkv.ArcPrefillAttentionKvCache(cache, prompt);
        lane.Synchronize();
        Assert.Equal(d2h, lane.D2HBytes);
        Assert.Equal(prompt, cache.Length);

        for (int position = prompt; position < prompt + newTokens; ++position)
        {
            Tensor currentQkv = MakeQkv(position, 1);
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                currentQkv.ArcIncrementalCausalAttention(cache, position + 1));
            d2h = lane.D2HBytes;
            Tensor actual = currentQkv.ArcIncrementalCausalAttention(cache, position);
            lane.Synchronize();
            Assert.Equal(d2h, lane.D2HBytes);
            Assert.Equal(position + 1, cache.Length);

            Tensor fullQkv = MakeQkv(0, position + 1);
            Tensor expected = fullQkv.FusedMultiHeadAttention(heads, causal: true);
            Assert.Equal(expected.DType, actual.DType);
            float[] expectedValues = expected.Data.TakeLast(width).ToArray();
            float[] actualValues = actual.Data.ToArray();
            Assert.Equal(width, actualValues.Length);
            double numerator = 0, denominator = 0;
            for (int i = 0; i < width; ++i)
            {
                Assert.True(float.IsFinite(actualValues[i]),
                    $"Nonfinite incremental output at position {position}, channel {i}.");
                double error = (double)actualValues[i] - expectedValues[i];
                numerator += error * error;
                denominator += (double)expectedValues[i] * expectedValues[i];
            }
            double relativeRms = Math.Sqrt(numerator / denominator);
            TestContext.Current.TestOutputHelper?.WriteLine(
                $"{precision}, position {position}: relative RMS {relativeRms:G6}");
            Assert.True(relativeRms <= maximumRelativeRms,
                $"{precision} position {position} relative RMS {relativeRms:G6} exceeds {maximumRelativeRms:G6}.");
        }
    }

    [Fact]
    public void CacheReleasesItsTwoResidentBuffers()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var execution = Tensor.BeginArcExecution();
        var lane = Tensor.ArcLane;
        long before = lane.AllocatedBytes;
        using (var cache = new ArcAttentionKvCache(width: 64, heads: 2, capacity: 16))
            Assert.Equal(before + 2L * 64 * 16 * sizeof(float), lane.AllocatedBytes);
        Assert.Equal(before, lane.AllocatedBytes);
    }
}
