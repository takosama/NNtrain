using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35MixedPrefixReuseTests
{
    [Fact]
    public void CacheIdentityChecksImageBitsPositionsTokenIdsAndLength()
    {
        Qwen35PromptToken[] prefix = [new(1, null, Qwen35Position.Scalar(0)),
            new(-1, [0f, .2f], new(1, 2, 3))];
        Qwen35PromptToken[] same = prefix.Select(entry => new Qwen35PromptToken(entry.TokenId,
            entry.Embedding?.ToArray(), entry.Position)).Append(new(2, null, Qwen35Position.Scalar(4))).ToArray();
        Assert.True(Qwen35QuantizedModel.MixedPromptPrefixMatches(prefix, same));
        Assert.False(Qwen35QuantizedModel.MixedPromptPrefixMatches(prefix, prefix));
        Assert.True(Qwen35QuantizedModel.MixedPromptPrefixMatches(prefix, prefix, allowEqual: true));
        Assert.False(Qwen35QuantizedModel.MixedPromptPrefixMatches(prefix, same.Take(1).ToArray()));
        Qwen35PromptToken[] changed = same.ToArray();
        changed[1] = changed[1] with { Position = new(1, 2, 4) };
        Assert.False(Qwen35QuantizedModel.MixedPromptPrefixMatches(prefix, changed));
        changed[1] = same[1] with { TokenId = 2 };
        Assert.False(Qwen35QuantizedModel.MixedPromptPrefixMatches(prefix, changed));
        changed[1] = same[1] with { Embedding = [-0f, .2f] };
        Assert.False(Qwen35QuantizedModel.MixedPromptPrefixMatches(prefix, changed));
        changed[1] = same[1] with { Embedding = [0f, MathF.BitIncrement(.2f)] };
        Assert.False(Qwen35QuantizedModel.MixedPromptPrefixMatches(prefix, changed));
        changed[1] = same[1] with { Embedding = null };
        Assert.False(Qwen35QuantizedModel.MixedPromptPrefixMatches(prefix, changed));
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(4, false)]
    [InlineData(16, true)]
    public void ExplicitMixedPrimeRestoresAndRetainsRepeatedAnswersAndLogits(int chunk, bool lora)
    {
        RequireGpu();
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(false, contextLength: 40);
        var options = new Qwen35ExecutionOptions { InferencePrefillChunkTokens = chunk };
        using Qwen35QuantizedModel reference = Qwen35QuantizedModel.Load(file.Path, [0], options: options);
        using Qwen35QuantizedModel prepared = Qwen35QuantizedModel.Load(file.Path, [0], options: options);
        if (lora) AttachMatchingLora(reference, prepared);
        Qwen35PromptToken[] prompt = Prompt(prepared.Descriptor.EmbeddingLength);
        Qwen35PromptToken[] prefix = prompt.Take(13).ToArray();
        int[] expected = reference.GenerateTokenIdsWithEmbeddings(prompt, 3, 24, TestContext.Current.CancellationToken);
        Assert.True(prepared.PrimePromptWithEmbeddings(prefix, TestContext.Current.CancellationToken).Cached);
        for (int repeat = 0; repeat < 2; repeat++)
        {
            Assert.Equal(expected, prepared.GenerateTokenIdsWithEmbeddings(prompt, 3, 24, TestContext.Current.CancellationToken));
            Assert.Equal(prefix.Length, prepared.LastReusedPromptTokens);
            Assert.Equal((prefix.Length, true), prepared.PrimePromptWithEmbeddings(prefix, TestContext.Current.CancellationToken));
        }
        reference.PrimePromptWithEmbeddings(prompt, TestContext.Current.CancellationToken);
        prepared.PrimePromptWithEmbeddings(prompt, TestContext.Current.CancellationToken);
        float[] expectedLogits = reference.ForwardToken(3), actualLogits = prepared.ForwardToken(3);
        for (int i = 0; i < expectedLogits.Length; i++)
            Assert.InRange(Math.Abs(expectedLogits[i] - actualLogits[i]), 0,
                1e-6f + Math.Abs(expectedLogits[i]) * 1e-5f);
    }

    [Fact]
    public void ChangedImageCallerMutationResetAndCancellationCannotReuseOldState()
    {
        RequireGpu();
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(false, contextLength: 40);
        using Qwen35QuantizedModel reference = Qwen35QuantizedModel.Load(file.Path, [0]);
        using Qwen35QuantizedModel prepared = Qwen35QuantizedModel.Load(file.Path, [0]);
        Qwen35PromptToken[] prompt = Prompt(prepared.Descriptor.EmbeddingLength), prefix = prompt.Take(13).ToArray();
        Assert.True(prepared.PrimePromptWithEmbeddings(prefix, TestContext.Current.CancellationToken).Cached);
        prompt[1].Embedding![0] = MathF.BitIncrement(prompt[1].Embedding![0]);
        Assert.Equal(reference.GenerateTokenIdsWithEmbeddings(prompt, 3, 24, TestContext.Current.CancellationToken),
            prepared.GenerateTokenIdsWithEmbeddings(prompt, 3, 24, TestContext.Current.CancellationToken));
        Assert.Equal(0, prepared.LastReusedPromptTokens);
        Assert.True(prepared.PrimePromptWithEmbeddings(prefix, TestContext.Current.CancellationToken).Cached);
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        Assert.Throws<OperationCanceledException>(() => prepared.GenerateTokenIdsWithEmbeddings(prompt,
            3, 24, cancellation.Token, onToken: _ => cancellation.Cancel()));
        prepared.GenerateTokenIdsWithEmbeddings(prompt, 3, 24, TestContext.Current.CancellationToken);
        Assert.Equal(0, prepared.LastReusedPromptTokens);
        prepared.PrimePromptWithEmbeddings(prefix, TestContext.Current.CancellationToken);
        prepared.Reset();
        prepared.GenerateTokenIdsWithEmbeddings(prompt, 3, 24, TestContext.Current.CancellationToken);
        Assert.Equal(0, prepared.LastReusedPromptTokens);
        prepared.PrimePromptWithEmbeddings(prefix, TestContext.Current.CancellationToken);
        prepared.PrimePromptPrefix([1, 2], TestContext.Current.CancellationToken);
        prepared.GenerateTokenIdsWithEmbeddings(prompt, 3, 24, TestContext.Current.CancellationToken);
        Assert.Equal(0, prepared.LastReusedPromptTokens);
    }

    private static Qwen35PromptToken[] Prompt(int width) => Enumerable.Range(0, 20).Select(i =>
        i is > 0 and < 13 ? new Qwen35PromptToken(-1, Enumerable.Range(0, width)
            .Select(component => ((component * 7 + i * 11) % 31 - 15) * .01f).ToArray(),
            new Qwen35Position(1, 1 + (i - 1) / 4, 1 + (i - 1) % 4))
        : new Qwen35PromptToken(i % 4, null, Qwen35Position.Scalar(i + 3))).ToArray();

    private static void RequireGpu()
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_MIXED_PRIME_GPU") != "1", "Opt-in mixed checkpoint fixture.");
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc required.");
    }

    private static void AttachMatchingLora(Qwen35QuantizedModel left, Qwen35QuantizedModel right)
    {
        var options = new Qwen35LoraOptions { Rank = 2, Alpha = 4, IncludeOutput = true, Seed = 83 };
        left.AttachLora(options); right.AttachLora(options);
        var random = new Random(983);
        foreach (string name in left.LoraMatrices.Keys)
        {
            float[][] state = left.LoraMatrices[name].ReadState();
            foreach (int index in new[] { 0, 1 })
                for (int i = 0; i < state[index].Length; i++) state[index][i] = random.NextSingle() * .2f - .1f;
            left.LoraMatrices[name].RestoreState(state, training: false);
            right.LoraMatrices[name].RestoreState(state, training: false);
        }
    }
}
