using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35MultimodalPrefillTests
{
    [Theory]
    [InlineData(3, 4, false, 1)]
    [InlineData(4, 4, true, 1)]
    [InlineData(5, 4, true, 2)]
    [InlineData(15, 16, false, 1)]
    [InlineData(16, 16, true, 1)]
    [InlineData(17, 16, true, 2)]
    [InlineData(31, 32, false, 1)]
    [InlineData(33, 32, true, 1)]
    public void MixedImageChunkMatchesScalarTokensAndCachedLogits(
        int promptLength, int chunkSize, bool lora, int deviceCount)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count < deviceCount,
            $"{deviceCount} Intel Arc GPU(s) are required.");
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(
            tiedOutput: false, contextLength: 40);
        int[] devices = Enumerable.Range(0, deviceCount).ToArray();
        using Qwen35QuantizedModel scalar = Qwen35QuantizedModel.Load(file.Path, devices, options: new()
        {
            InferencePrefillChunkTokens = 0
        });
        using Qwen35QuantizedModel chunked = Qwen35QuantizedModel.Load(file.Path, devices, options: new()
        {
            InferencePrefillChunkTokens = chunkSize
        });
        if (lora)
        {
            var options = new Qwen35LoraOptions { Rank = 2, Alpha = 4, IncludeOutput = true, Seed = 83 };
            scalar.AttachLora(options);
            chunked.AttachLora(options);
            var random = new Random(983);
            foreach (string name in scalar.LoraMatrices.Keys)
            {
                float[][] state = scalar.LoraMatrices[name].ReadState();
                foreach (int index in new[] { 0, 1 })
                    for (int i = 0; i < state[index].Length; i++)
                        state[index][i] = (float)(random.NextDouble() * .2 - .1);
                scalar.LoraMatrices[name].RestoreState(state, training: false);
                chunked.LoraMatrices[name].RestoreState(state, training: false);
            }
        }
        Qwen35PromptToken[] prompt = Enumerable.Range(0, promptLength).Select(i =>
            i > 0 && i < promptLength - 1 && i % 5 != 0
                ? new Qwen35PromptToken(-1, Enumerable.Range(0, scalar.Descriptor.EmbeddingLength)
                    .Select(component => ((component * 7 + i * 11) % 31 - 15) * .01f).ToArray(),
                    new Qwen35Position(1, 1 + (i - 1) / 4, 1 + (i - 1) % 4))
                : new Qwen35PromptToken(i % 4, null, Qwen35Position.Scalar(i + 3))).ToArray();
        int[] expected = scalar.GenerateTokenIdsWithEmbeddings(prompt, 3,
            promptLength + 3, TestContext.Current.CancellationToken);
        int[] actual = chunked.GenerateTokenIdsWithEmbeddings(prompt, 3,
            promptLength + 3, TestContext.Current.CancellationToken);
        Assert.Equal(expected, actual);
        // Compare the full logit vector, including recurrent and attention
        // caches, after generation. Argmax equality alone can hide drift.
        float[] expectedLogits = scalar.ForwardToken(3);
        float[] actualLogits = chunked.ForwardToken(3);
        for (int i = 0; i < expectedLogits.Length; i++)
            Assert.InRange(Math.Abs(expectedLogits[i] - actualLogits[i]), 0f,
                1e-6f + Math.Abs(expectedLogits[i]) * 1e-5f);
    }

    [Theory]
    [InlineData(4)]
    [InlineData(16)]
    public void ScalarPositionChunkWithoutImageMatchesExistingTextPath(int chunkSize)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(
            tiedOutput: false, contextLength: 40);
        var options = new Qwen35ExecutionOptions { InferencePrefillChunkTokens = chunkSize };
        using Qwen35QuantizedModel text = Qwen35QuantizedModel.Load(file.Path, [0], options: options);
        using Qwen35QuantizedModel mixed = Qwen35QuantizedModel.Load(file.Path, [0], options: options);
        int[] ids = Enumerable.Range(0, 19).Select(i => (i * 3 + 1) % 4).ToArray();
        Qwen35PromptToken[] prompt = ids.Select((id, position) =>
            new Qwen35PromptToken(id, null, Qwen35Position.Scalar(position))).ToArray();
        Assert.Equal(text.GenerateTokenIds(ids, 3), mixed.GenerateTokenIdsWithEmbeddings(prompt, 3,
            ids.Length, TestContext.Current.CancellationToken));
        Assert.Equal(text.ForwardToken(3), mixed.ForwardToken(3));
    }

    [Theory]
    [InlineData(2, false)]
    [InlineData(4, false)]
    [InlineData(8, false)]
    [InlineData(16, false)]
    [InlineData(4, true)]
    [InlineData(8, true)]
    public void DecodedWeightSharingRetainsMixedPromptLogits(int projectionRows, bool lora)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU required.");
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(false,
            contextLength: 40, iq2Qkv: true, iq2Gate: true, layerCount: 4);
        using Qwen35QuantizedModel scalar = Qwen35QuantizedModel.Load(file.Path, [0]);
        using Qwen35QuantizedModel shared = Qwen35QuantizedModel.Load(file.Path, [0], options: new()
        {
            InferencePrefillChunkTokens = 16,
            InferenceProjectionRows = projectionRows
        });
        if (lora)
        {
            var options = new Qwen35LoraOptions { Rank = 2, Alpha = 4, IncludeOutput = true, Seed = 83 };
            scalar.AttachLora(options);
            shared.AttachLora(options);
            var random = new Random(983);
            foreach (string name in scalar.LoraMatrices.Keys)
            {
                float[][] state = scalar.LoraMatrices[name].ReadState();
                foreach (int index in new[] { 0, 1 })
                    for (int i = 0; i < state[index].Length; i++)
                        state[index][i] = (float)(random.NextDouble() * .2 - .1);
                scalar.LoraMatrices[name].RestoreState(state, training: false);
                shared.LoraMatrices[name].RestoreState(state, training: false);
            }
        }
        Qwen35PromptToken[] prompt = Enumerable.Range(0, 19).Select(i => i > 0 && i < 18
            ? new Qwen35PromptToken(-1, Enumerable.Range(0, scalar.Descriptor.EmbeddingLength)
                .Select(component => ((component * 7 + i * 11) % 31 - 15) * .01f).ToArray(),
                new Qwen35Position(1, 1 + (i - 1) / 4, 1 + (i - 1) % 4))
            : new Qwen35PromptToken(i % 4, null, Qwen35Position.Scalar(i + 3))).ToArray();
        Assert.Equal(scalar.GenerateTokenIdsWithEmbeddings(prompt, 3, 22,
            TestContext.Current.CancellationToken), shared.GenerateTokenIdsWithEmbeddings(prompt, 3, 22,
            TestContext.Current.CancellationToken));
        float[] expectedLogits = scalar.ForwardToken(3), actualLogits = shared.ForwardToken(3);
        if (!lora) Assert.Equal(expectedLogits, actualLogits);
        else
            for (int i = 0; i < expectedLogits.Length; i++)
                Assert.InRange(Math.Abs(expectedLogits[i] - actualLogits[i]), 0,
                    1e-6f + Math.Abs(expectedLogits[i]) * 1e-5f);
    }

    [Fact]
    public void TextOnlyFusedLoraAndEosKeepTheirOriginalLogits()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU required.");
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(false,
            contextLength: 40, iq2Qkv: true, iq2Gate: true, layerCount: 4);
        using Qwen35QuantizedModel previous = Qwen35QuantizedModel.Load(file.Path, [0], options: new()
        {
            InferencePrefillChunkTokens = 16
        });
        using Qwen35QuantizedModel shared = Qwen35QuantizedModel.Load(file.Path, [0], options: new()
        {
            InferencePrefillChunkTokens = 16,
            InferenceProjectionRows = 4
        });
        var lora = new Qwen35LoraOptions { Rank = 2, Alpha = 4, IncludeOutput = true, Seed = 83 };
        previous.AttachLora(lora);
        shared.AttachLora(lora);
        var random = new Random(983);
        foreach (string name in previous.LoraMatrices.Keys)
        {
            float[][] state = previous.LoraMatrices[name].ReadState();
            foreach (int index in new[] { 0, 1 })
                for (int i = 0; i < state[index].Length; i++)
                    state[index][i] = (float)(random.NextDouble() * .2 - .1);
            previous.LoraMatrices[name].RestoreState(state, training: false);
            shared.LoraMatrices[name].RestoreState(state, training: false);
        }
        int[] ids = Enumerable.Range(0, 19).Select(i => i % 4).ToArray();
        var expectedStream = new List<int>();
        var actualStream = new List<int>();
        Assert.Equal(previous.GenerateTokenIds(ids, 8, eosTokenId: 2, onToken: expectedStream.Add),
            shared.GenerateTokenIds(ids, 8, eosTokenId: 2, onToken: actualStream.Add));
        Assert.Equal(expectedStream, actualStream);
        Assert.Equal(previous.ForwardToken(3), shared.ForwardToken(3));
    }
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void XmxTextPrefillRetainsTokensLoraAndEos(bool lora)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc required.");
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(false,
            contextLength: 40, iq2Qkv: true, iq2Gate: true, layerCount: 4);
        using Qwen35QuantizedModel previous = Qwen35QuantizedModel.Load(file.Path, [0], options: new()
        {
            InferencePrefillChunkTokens = 16, InferenceProjectionRows = 4,
            InferenceBatchRecurrent = false
        });
        using Qwen35QuantizedModel candidate = Qwen35QuantizedModel.Load(file.Path, [0], options: new()
        {
            InferencePrefillChunkTokens = 1024, InferenceProjectionRows = 4,
            InferenceXmxPrefill = true, InferenceXmxFactoredPrefill = true,
            InferenceBatchRecurrent = true
        });
        if (lora)
        {
            var options = new Qwen35LoraOptions { Rank = 2, Alpha = 4, IncludeOutput = true, Seed = 83 };
            previous.AttachLora(options); candidate.AttachLora(options);
            var random = new Random(983);
            foreach (string name in previous.LoraMatrices.Keys)
            {
                float[][] state = previous.LoraMatrices[name].ReadState();
                foreach (int index in new[] { 0, 1 })
                    for (int i = 0; i < state[index].Length; i++)
                        state[index][i] = (float)(random.NextDouble() * .2 - .1);
                previous.LoraMatrices[name].RestoreState(state, training: false);
                candidate.LoraMatrices[name].RestoreState(state, training: false);
            }
        }
        int[] ids = Enumerable.Range(0, 31).Select(i => i % 4).ToArray();
        var expectedStream = new List<int>(); var actualStream = new List<int>();
        Assert.Equal(previous.GenerateTokenIds(ids, 8, eosTokenId: 2, onToken: expectedStream.Add),
            candidate.GenerateTokenIds(ids, 8, eosTokenId: 2, onToken: actualStream.Add));
        Assert.Equal(expectedStream, actualStream);
        float[] expected = previous.ForwardToken(3), actual = candidate.ForwardToken(3);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(Math.Abs(expected[i] - actual[i]), 0f, 1e-6f + Math.Abs(expected[i]) * 1e-5f);
    }
}
