using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35TextAttentionPrefillTests
{
    [Fact]
    public void TextBatchingRemainsAnIndependentOptIn()
    {
        var options = new Qwen35ExecutionOptions();
        Assert.False(options.InferenceBatchTextAttention);
        Assert.True(options.InferenceBatchMixedAttention);
        Assert.Equal(0, options.InferenceTextAttentionTileRows);
    }

    [Theory]
    [InlineData(8, false, false, 0)]
    [InlineData(16, true, false, 0)]
    [InlineData(32, true, true, 0)]
    [InlineData(33, false, false, 8)]
    [InlineData(33, true, true, 16)]
    public void TextBatchesPreserveLogitsAndKvContinuationAcrossReusedPrefixes(
        int chunkSize, bool fastNorm, bool lora, int tileRows)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU required.");
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(
            tiedOutput: false, contextLength: 80);
        var options = new Qwen35ExecutionOptions
        {
            InferencePrefillChunkTokens = chunkSize,
            InferenceFastRmsNorm = fastNorm,
            InferenceBatchTextAttention = false,
            // Ensure the text switch does not depend on the mixed-prompt switch.
            InferenceBatchMixedAttention = false,
            CollectKernelTimings = true
        };
        using Qwen35QuantizedModel previous = Qwen35QuantizedModel.Load(file.Path, [0], options: options);
        using Qwen35QuantizedModel batched = Qwen35QuantizedModel.Load(file.Path, [0],
            options: options with
            {
                InferenceBatchTextAttention = true,
                InferenceTextAttentionTileRows = tileRows
            });
        Assert.Equal(2, batched.Descriptor.HeadCount);
        Assert.Equal(1, batched.Descriptor.KvHeadCount);
        Assert.Equal(64, batched.Descriptor.RopeDimensionCount);
        Assert.Equal(128, batched.Descriptor.HeadWidth);
        if (lora) AttachMatchingLora(previous, batched);

        int[] prefix = [1, 3, 0, 2, 1];
        int[] history = [.. prefix, .. Enumerable.Range(prefix.Length, 47 - prefix.Length)
            .Select(i => (i * 3 + 1) % 4)];
        Assert.Equal(previous.PrimePromptPrefix(prefix, TestContext.Current.CancellationToken),
            batched.PrimePromptPrefix(prefix, TestContext.Current.CancellationToken));
        long previousUploaded = Assert.Single(previous.UploadedBytes);
        long batchedUploaded = Assert.Single(batched.UploadedBytes);
        long previousDownloaded = Assert.Single(previous.DownloadedBytes);
        long batchedDownloaded = Assert.Single(batched.DownloadedBytes);
        // The next attention chunk starts at absolute position five. Later
        // chunks cross both KV growth and power-of-two score capacity bounds.
        // The forced 8/16-row cases split a 33-row projection chunk and must
        // keep its one-row final attention tile at the correct absolute index.
        Assert.Equal(previous.PrimePromptPrefix(history, TestContext.Current.CancellationToken),
            batched.PrimePromptPrefix(history, TestContext.Current.CancellationToken));
        Assert.Equal(prefix.Length, batched.LastReusedPromptTokens);
        Assert.Equal(previous.ResidentStateBytes, batched.ResidentStateBytes);
        Assert.Equal(previousDownloaded, Assert.Single(previous.DownloadedBytes));
        Assert.Equal(batchedDownloaded, Assert.Single(batched.DownloadedBytes));
        // The only additional host traffic is one three-coordinate position
        // record per row. Q/K/V tiles and combined outputs stay on the GPU.
        Assert.Equal((history.Length - prefix.Length) * 3L * sizeof(int),
            (Assert.Single(batched.UploadedBytes) - batchedUploaded)
            - (Assert.Single(previous.UploadedBytes) - previousUploaded));
        foreach (int token in new[] { 3, 1, 2 })
            Assert.Equal(previous.ForwardToken(token), batched.ForwardToken(token));

        Assert.Contains("q35a_scores_rows", batched.KernelMilliseconds.Keys);
        Assert.Contains("q35a_attend_rows", batched.KernelMilliseconds.Keys);
        Assert.DoesNotContain("q35a_scores_rows", previous.KernelMilliseconds.Keys);
        Assert.DoesNotContain("q35a_attend_rows", previous.KernelMilliseconds.Keys);

        previous.Reset();
        batched.Reset();
        // Validate the generated last-prompt token and continuation after a
        // fresh prefill as well as the explicit prefix-cache path above.
        Assert.Equal(previous.GenerateTokenIds(history, 3), batched.GenerateTokenIds(history, 3));
        Assert.Equal(previous.ForwardToken(0), batched.ForwardToken(0));
    }

    private static void AttachMatchingLora(Qwen35QuantizedModel previous, Qwen35QuantizedModel batched)
    {
        var options = new Qwen35LoraOptions { Rank = 2, Alpha = 4, IncludeOutput = true, Seed = 97 };
        previous.AttachLora(options);
        batched.AttachLora(options);
        var random = new Random(991);
        foreach (string name in previous.LoraMatrices.Keys)
        {
            float[][] state = previous.LoraMatrices[name].ReadState();
            foreach (int index in new[] { 0, 1 })
                for (int i = 0; i < state[index].Length; i++)
                    state[index][i] = random.NextSingle() * .2f - .1f;
            previous.LoraMatrices[name].RestoreState(state, training: false);
            batched.LoraMatrices[name].RestoreState(state, training: false);
        }
    }
}
