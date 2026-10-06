using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35MultimodalModelTests
{
    [Fact]
    public void ScalarPositionsWithoutOverridesMatchTextGeneration()
    {
        RequireArc();
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(
            tiedOutput: false, contextLength: 16);
        using Qwen35QuantizedModel text = Qwen35QuantizedModel.Load(file.Path, [0]);
        using Qwen35QuantizedModel mixed = Qwen35QuantizedModel.Load(file.Path, [0]);
        int[] ids = [1, 2, 0];
        Qwen35PromptToken[] prompt = ids.Select((id, position) =>
            new Qwen35PromptToken(id, null, Qwen35Position.Scalar(position))).ToArray();

        int[] expected = text.GenerateTokenIds(ids, 3);
        int[] actual = mixed.GenerateTokenIdsWithEmbeddings(prompt, 3, nextPosition: ids.Length,
            TestContext.Current.CancellationToken);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ImageEmbeddingAndSpatialPositionAdvanceTheMixedPrompt()
    {
        RequireArc();
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(
            tiedOutput: false, contextLength: 16);
        using Qwen35QuantizedModel model = Qwen35QuantizedModel.Load(file.Path, [0]);
        float[] image = Enumerable.Range(0, model.Descriptor.EmbeddingLength)
            .Select(i => (i % 9 - 4) * 0.01f).ToArray();
        Qwen35PromptToken[] prompt =
        [
            new(1, null, Qwen35Position.Scalar(0)),
            new(-1, image, new Qwen35Position(1, 0, 1)),
            new(2, null, Qwen35Position.Scalar(2))
        ];

        int[] generated = model.GenerateTokenIdsWithEmbeddings(prompt, 2, nextPosition: 3,
            TestContext.Current.CancellationToken);

        Assert.Equal([1, -1, 2], generated[..3]);
        Assert.Equal(5, generated.Length);
        Assert.All(generated[3..], id => Assert.InRange(id, 0, model.Descriptor.VocabularySize - 1));
    }

    [Fact]
    public void MixedPromptRejectsInvalidEmbeddingBeforeGpuMutation()
    {
        RequireArc();
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(tiedOutput: false);
        using Qwen35QuantizedModel model = Qwen35QuantizedModel.Load(file.Path, [0]);
        Qwen35PromptToken[] wrongWidth = [new(-1, new float[3], new Qwen35Position(0, 0, 1))];
        Qwen35PromptToken[] nonFinite = [new(-1,
            Enumerable.Repeat(float.NaN, model.Descriptor.EmbeddingLength).ToArray(),
            new Qwen35Position(0, 0, 1))];

        Assert.Throws<ArgumentException>(() => model.GenerateTokenIdsWithEmbeddings(
            wrongWidth, 1, nextPosition: 1, TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => model.GenerateTokenIdsWithEmbeddings(
            nonFinite, 1, nextPosition: 1, TestContext.Current.CancellationToken));
        Assert.Equal(2, model.GenerateTokenIds([1], 1).Length);
    }

    private static void RequireArc()
        => Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
}
