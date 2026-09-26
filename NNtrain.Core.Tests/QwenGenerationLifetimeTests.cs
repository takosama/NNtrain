using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class QwenGenerationLifetimeTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RepeatedGenerationsReleaseActivationsAndPreserveWeights(bool quantized)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(0), "Intel Arc GPU is required.");
        using IDisposable execution = Tensor.BeginArcExecution(0, TensorPrecisionMode.Float32);
        LanguageModel model = CreateModel(quantized);
        using IDisposable? weights = model as IDisposable;
        ArcExecutionLane lane = Tensor.ArcLane;
        int[] prompt = [1, 5];

        int[] first = Generate(model, prompt, 4);
        if (quantized)
            Assert.Equal([1, 5, 15, 15, 15, 15], first);
        lane.Synchronize();
        long retainedBytes = lane.AllocatedBytes;
        Assert.True(retainedBytes > 0, "The model weights should remain resident between generations.");
        Assert.True(model.IsTraining);

        for (int repetition = 0; repetition < 3; ++repetition)
        {
            int[] repeated = Generate(model, prompt, 4);
            lane.Synchronize();
            Assert.Equal(first, repeated);
            Assert.Equal(6, repeated.Length);
            Assert.Equal(retainedBytes, lane.AllocatedBytes);
        }

        // Sampling has already materialized the token before its frame ends.
        // A stop-token exit must release the same temporary buffers.
        int[] stopped = model.GenerateTokenIds(prompt, 4, 0f, 1,
            stopTokenId: first[prompt.Length], random: new Random(17));
        lane.Synchronize();
        Assert.Equal(first.Take(prompt.Length + 1).ToArray(), stopped);
        Assert.Equal(retainedBytes, lane.AllocatedBytes);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SamplingFailureReleasesActivationsAndRestoresTrainingMode(bool quantized)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(0), "Intel Arc GPU is required.");
        using IDisposable execution = Tensor.BeginArcExecution(0, TensorPrecisionMode.Float32);
        LanguageModel model = CreateModel(quantized);
        using IDisposable? weights = model as IDisposable;
        ArcExecutionLane lane = Tensor.ArcLane;
        int[] prompt = [1, 5];

        int[] expected = Generate(model, prompt, 1);
        lane.Synchronize();
        long retainedBytes = lane.AllocatedBytes;
        var failure = new InvalidOperationException("Sampler failed.");

        InvalidOperationException actual = Assert.Throws<InvalidOperationException>(() =>
            model.GenerateTokenIds(prompt, 2, temperature: 1f, topK: 4,
                stopTokenId: null, random: new ThrowingRandom(failure)));
        lane.Synchronize();
        Assert.Same(failure, actual);
        Assert.True(model.IsTraining);
        Assert.Equal(retainedBytes, lane.AllocatedBytes);
        Assert.Equal(expected, Generate(model, prompt, 1));
    }

    private static int[] Generate(LanguageModel model, int[] prompt, int count)
        => model.GenerateTokenIds(prompt, count, 0f, 1,
            stopTokenId: null, random: new Random(17));

    private static LanguageModel CreateModel(bool quantized)
    {
        if (!quantized)
            return new Qwen2ForCausalLM(16, 8, 32, 4, 2, 64, 1,
                random: new Random(13));

        const int width = 256;
        const int vocabulary = 16;
        const TensorDType dtype = TensorDType.Float32;
        QwenQuantizedLinear Linear(int input, int output, float[]? bias = null)
            => new(new byte[checked(output * (input / 256) * GgufQ4K.BlockBytes)],
                Qwen2Gguf.Q4KType, input, output, bias, dtype);
        QwenRmsNorm Norm() => new(width, 1e-6f, dtype);

        // Complete 256-value blocks are required by K-quantized matrices.
        // Zero quantized weights plus a distinct head bias give a known token,
        // while still exercising every layer and persistent encoded buffer.
        var attention = new QwenQuantizedAttention(
            Linear(width, width), Linear(width, width / 2),
            Linear(width, width / 2), Linear(width, width),
            queryHeads: 4, kvHeads: 2, ropeTheta: 1_000_000f, dtype);
        var mlp = new QwenQuantizedMlp(
            Linear(width, width), Linear(width, width), Linear(width, width), dtype);
        var block = new QwenQuantizedBlock(Norm(), attention, Norm(), mlp, dtype);
        float[] embedding = Enumerable.Range(0, vocabulary * width)
            .Select(index => (index % 19 - 9) * 0.01f).ToArray();
        return new Qwen2QuantizedForCausalLM(vocabulary, 8, width, 4, 2,
            new Parameter(embedding, [vocabulary, width], "embedding",
                WeightDecayPolicy.Apply, dtype),
            null, [block], Norm(),
            Linear(width, vocabulary, Enumerable.Range(0, vocabulary)
                .Select(index => index * 0.1f).ToArray()), dtype);
    }

    private sealed class ThrowingRandom(Exception failure) : Random
    {
        public override double NextDouble() => throw failure;
    }
}
