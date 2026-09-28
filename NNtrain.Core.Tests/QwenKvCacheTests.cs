using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class QwenKvCacheTests
{
    [Theory]
    [InlineData(1, TensorPrecisionMode.Float32, TensorDType.Float32, 0.00001)]
    [InlineData(4, TensorPrecisionMode.Float32, TensorDType.Float32, 0.00001)]
    [InlineData(3, TensorPrecisionMode.Mix16_32, TensorDType.BFloat16, 0.01)]
    public void CachedAttentionMatchesFullPrefixWithAbsolutePositions(int prompt,
        TensorPrecisionMode precision, TensorDType dtype, double tolerance)
    {
        RequireArc();
        using var execution = Tensor.BeginArcExecution(precision: precision);
        using var noGrad = AutogradContext.NoGrad();
        const int heads = 4, kvHeads = 2, headWidth = 128, tokens = 8;
        float[] q = Values(tokens * heads * headWidth, 13);
        float[] k = Values(tokens * kvHeads * headWidth, 17);
        float[] v = Values(tokens * kvHeads * headWidth, 29);
        Tensor Part(float[] source, int first, int count, int width)
            => new(source.AsSpan(first * width, count * width).ToArray(),
                [1, count, width], dtype: dtype);

        using var cache = new ArcQwenKvCache(heads, kvHeads, headWidth, tokens);
        for (int position = 0; position < tokens; position = position == 0 ? prompt : position + 1)
        {
            int count = position == 0 ? prompt : 1;
            using var frame = Tensor.BeginArcInferenceFrame();
            Tensor query = Part(q, position, count, heads * headWidth);
            Tensor key = Part(k, position, count, kvHeads * headWidth);
            Tensor value = Part(v, position, count, kvHeads * headWidth);
            long readBytes = Tensor.ArcLane.D2HBytes;
            Tensor cached = query.ArcQwenCachedAttention(key, value, cache, position, 10_000f);
            Tensor.ArcLane.Synchronize();
            Assert.Equal(readBytes, Tensor.ArcLane.D2HBytes);
            Assert.Equal(position + count, cache.Length);
            Tensor full = Part(q, 0, position + count, heads * headWidth)
                .QwenGroupedQueryAttention(
                    Part(k, 0, position + count, kvHeads * headWidth),
                    Part(v, 0, position + count, kvHeads * headWidth),
                    heads, kvHeads, 10_000f);
            AssertClose(full.Data.TakeLast(count * heads * headWidth).ToArray(),
                cached.Data.ToArray(), tolerance);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void EveryCachedLastLogitMatchesFullQuantizedModel(int prompt)
    {
        RequireArc();
        using var execution = Tensor.BeginArcExecution();
        using var noGrad = AutogradContext.NoGrad();
        using Qwen2QuantizedForCausalLM model = CreateModel();
        int[] tokens = [1, 7, 3, 12, 5, 9, 2];
        ArcQwenKvCache[] caches = model.CreateArcKvCaches(tokens.Length);
        try
        {
            for (int position = 0; position < tokens.Length;
                 position = position == 0 ? prompt : position + 1)
            {
                using var frame = Tensor.BeginArcInferenceFrame();
                int count = position == 0 ? prompt : 1;
                float[] cached = model.ForwardLastLogits(
                    tokens.AsSpan(position, count).ToArray(), caches, position).Data.ToArray();
                float[] full = model.Forward(tokens[..(position + count)], 1, position + count)
                    .Data.TakeLast(model.VocabularySize).ToArray();
                AssertClose(full, cached, 0.0002);
                Assert.Equal(Array.IndexOf(full, full.Max()), Array.IndexOf(cached, cached.Max()));
            }
        }
        finally { foreach (ArcQwenKvCache cache in caches) cache.Dispose(); }
    }

    [Fact]
    public void CachedGenerationMatchesReferenceAndStreamsEveryToken()
    {
        RequireArc();
        int[] Generate(bool cached)
        {
            using var execution = Tensor.BeginArcExecution(options:
                new ArcExecutionOptions { QwenInferenceKvCache = cached });
            using Qwen2QuantizedForCausalLM model = CreateModel();
            int[] prompt = [1, 7, 3];
            var streamed = new List<int>();
            int[] result = model.GenerateTokenIds(prompt, 4, 0f, 1, null,
                new Random(29), streamed.Add);
            Assert.Equal(result.Skip(prompt.Length).ToArray(), streamed.ToArray());
            Tensor.ArcLane.Synchronize();
            long retained = Tensor.ArcLane.AllocatedBytes;
            Assert.Equal(result, model.GenerateTokenIds(prompt, 4, 0f, 1, null,
                new Random(29)));
            Tensor.ArcLane.Synchronize();
            Assert.Equal(retained, Tensor.ArcLane.AllocatedBytes);
            return result;
        }
        Assert.Equal(Generate(false), Generate(true));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CallbackFailureStopAndZeroTokensReleaseCaches(bool cached)
    {
        RequireArc();
        using var execution = Tensor.BeginArcExecution(options:
            new ArcExecutionOptions { QwenInferenceKvCache = cached });
        using Qwen2QuantizedForCausalLM model = CreateModel();
        int[] prompt = [1, 7, 3];
        int[] warmup = model.GenerateTokenIds(prompt, 3, 0f, 1, null, new Random(29));
        Tensor.ArcLane.Synchronize();
        long retained = Tensor.ArcLane.AllocatedBytes;
        var failure = new InvalidOperationException("Consumer failed.");
        int callbacks = 0;
        Assert.Same(failure, Assert.Throws<InvalidOperationException>(() =>
            model.GenerateTokenIds(prompt, 3, 0f, 1, null, new Random(29), _ =>
            {
                if (++callbacks == 2) throw failure;
            })));
        Assert.Equal(2, callbacks);
        Tensor.ArcLane.Synchronize();
        Assert.True(model.IsTraining);
        Assert.Equal(retained, Tensor.ArcLane.AllocatedBytes);

        var stopped = new List<int>();
        Assert.Equal(warmup.Take(4).ToArray(), model.GenerateTokenIds(prompt, 3,
            0f, 1, warmup[3], new Random(29), stopped.Add));
        Assert.Single(stopped);
        Assert.Equal(prompt, model.GenerateTokenIds(prompt, 0, 0f, 1, null,
            new Random(29), _ => throw new Exception("No token expected.")));
        Tensor.ArcLane.Synchronize();
        Assert.Equal(retained, Tensor.ArcLane.AllocatedBytes);
    }

    [Fact]
    public void LongDecodeUsesDeviceScratchAndDisposesCacheBuffers()
    {
        RequireArc();
        using var execution = Tensor.BeginArcExecution();
        using var noGrad = AutogradContext.NoGrad();
        var lane = Tensor.ArcLane;
        const int position = 4096, width = 4;
        const float theta = 10_000f;
        float[] queryValues = [0.31f, -0.67f, 0.21f, 0.09f];
        float[] rawKeys = Values((position + 1) * width, 31);
        float[] values = Enumerable.Range(0, rawKeys.Length)
            .Select(index => 0.25f + 0.1f * MathF.Sin(index * 0.031f)).ToArray();
        var rotatedKeys = new float[rawKeys.Length];
        float[] Rotate(float[] source, int offset, int absolutePosition)
        {
            var rotated = new float[width];
            for (int pair = 0; pair < width / 2; ++pair)
            {
                float angle = absolutePosition * MathF.Pow(theta, -2f * pair / width);
                float c = MathF.Cos(angle), s = MathF.Sin(angle);
                float a = source[offset + pair], b = source[offset + pair + width / 2];
                rotated[pair] = a * c - b * s;
                rotated[pair + width / 2] = b * c + a * s;
            }
            return rotated;
        }
        for (int token = 0; token <= position; ++token)
            Rotate(rawKeys, token * width, token).CopyTo(rotatedKeys, token * width);
        float[] rotatedQuery = Rotate(queryValues, 0, position);
        var scores = new float[position + 1];
        for (int token = 0; token <= position; ++token)
        {
            float dot = 0f;
            for (int pair = 0; pair < width / 2; ++pair)
            {
                dot = MathF.FusedMultiplyAdd(rotatedQuery[pair], rotatedKeys[token * width + pair], dot);
                dot = MathF.FusedMultiplyAdd(rotatedQuery[pair + width / 2],
                    rotatedKeys[token * width + pair + width / 2], dot);
            }
            scores[token] = dot / MathF.Sqrt(width);
        }
        float maximum = scores.Max(), sum = 0f;
        for (int token = 0; token <= position; ++token)
            sum += scores[token] = MathF.Exp(scores[token] - maximum);
        var expected = new float[width];
        for (int channel = 0; channel < width; ++channel)
        for (int token = 0; token <= position; ++token)
            expected[channel] = MathF.FusedMultiplyAdd(scores[token] / sum,
                values[token * width + channel], expected[channel]);

        long before = lane.AllocatedBytes;
        var cache = new ArcQwenKvCache(1, 1, width, position + 1);
        Assert.Equal(before + 2L * (position + 1) * width * sizeof(float), lane.AllocatedBytes);
        lane.Write(cache.Key, rotatedKeys);
        lane.Write(cache.Value, values);
        cache.MarkAppended(position);
        using (Tensor.BeginArcInferenceFrame())
        {
            var q = new Tensor(queryValues, [1, 1, width]);
            var k = new Tensor(rawKeys.AsSpan(position * width, width).ToArray(), [1, 1, width]);
            var v = new Tensor(values.AsSpan(position * width, width).ToArray(), [1, 1, width]);
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                q.ArcQwenCachedAttention(k, v, cache, position - 1, 10_000f));
            Tensor result = q.ArcQwenCachedAttention(k, v, cache, position, 10_000f);
            AssertClose(expected, result.Data.ToArray(), 0.0001);
            cache.Dispose();
            Assert.Throws<ObjectDisposedException>(() =>
                q.ArcQwenCachedAttention(k, v, cache, position, 10_000f));
        }
        // The three input leaf tensors remain session-owned; cache disposal is
        // checked independently of those buffers.
        cache.Dispose();
        lane.Synchronize();
        Assert.Equal(before + 3L * width * sizeof(float), lane.AllocatedBytes);
    }

    private static Qwen2QuantizedForCausalLM CreateModel()
    {
        const int width = 256, vocabulary = 16;
        const TensorDType dtype = TensorDType.Float32;
        var random = new Random(79);
        QwenQuantizedLinear Linear(int output)
        {
            byte[] payload = new byte[output * GgufQ4K.BlockBytes];
            random.NextBytes(payload);
            ushort scale = BitConverter.HalfToUInt16Bits((Half)0.0001f);
            ushort minimum = BitConverter.HalfToUInt16Bits((Half)0.0005f);
            for (int block = 0; block < output; ++block)
            {
                int offset = block * GgufQ4K.BlockBytes;
                payload[offset] = (byte)scale;
                payload[offset + 1] = (byte)(scale >> 8);
                payload[offset + 2] = (byte)minimum;
                payload[offset + 3] = (byte)(minimum >> 8);
            }
            return new QwenQuantizedLinear(payload, Qwen2Gguf.Q4KType,
                width, output, null, dtype);
        }
        QwenRmsNorm Norm() => new(width, 1e-6f, dtype);
        var blocks = Enumerable.Range(0, 2).Select(_ => new QwenQuantizedBlock(
            Norm(), new QwenQuantizedAttention(Linear(width), Linear(width / 2),
                Linear(width / 2), Linear(width), 4, 2, 10_000f, dtype),
            Norm(), new QwenQuantizedMlp(Linear(width), Linear(width), Linear(width), dtype),
            dtype)).ToArray();
        return new Qwen2QuantizedForCausalLM(vocabulary, 12, width, 4, 2,
            new Parameter(Values(vocabulary * width, 37), [vocabulary, width],
                "embedding", WeightDecayPolicy.Apply, dtype),
            blocks, Norm(), Linear(vocabulary), dtype);
    }

    private static float[] Values(int count, int seed)
        => Enumerable.Range(0, count)
            .Select(index => MathF.Sin(index * 0.173f + seed) * 0.3f).ToArray();

    private static void AssertClose(float[] expected, float[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        double squaredError = 0, squaredReference = 0;
        for (int i = 0; i < actual.Length; ++i)
        {
            Assert.True(float.IsFinite(actual[i]));
            squaredError += Math.Pow(actual[i] - expected[i], 2);
            squaredReference += Math.Pow(expected[i], 2);
        }
        double relativeRms = Math.Sqrt(squaredError / Math.Max(squaredReference, 1e-20));
        Assert.True(relativeRms <= tolerance,
            $"Relative RMS {relativeRms:G6} exceeds {tolerance:G6}.");
    }

    private static void RequireArc()
        => Assert.SkipWhen(!Tensor.IsArcAvailable(0), "Intel Arc GPU is required.");
}
