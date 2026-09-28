using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35ResidentModelTests
{
    [Fact]
    public void SeededSamplingMatchesFullPrefillAfterPromptPrefixReuse()
    {
        RequireArcDevices(1);
        using TemporaryQwenGguf file = CreateFixture(tiedOutput: false, contextLength: 40);
        using Qwen35QuantizedModel model = Qwen35QuantizedModel.Load(file.Path, [0]);
        int[] firstPrompt = [1, 2, 0];
        int[] secondPrompt = [1, 2, 0, 3, 1];
        _ = model.GenerateTokenIdsWithPrefixReuse(firstPrompt, 3,
            temperature: 0.6f, topP: 0.95f, topK: 20, random: new Random(17));
        int[] reused = model.GenerateTokenIdsWithPrefixReuse(secondPrompt, 3,
            temperature: 0.6f, topP: 0.95f, topK: 20, random: new Random(28));
        Assert.Equal(firstPrompt.Length, model.LastReusedPromptTokens);
        int[] full = model.GenerateTokenIds(secondPrompt, 3,
            temperature: 0.6f, topP: 0.95f, topK: 20, random: new Random(28));
        Assert.Equal(full, reused);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void PrefixReuseRestoresHybridPromptStateAndFallsBackForChangedPrompts(int deviceCount)
    {
        RequireArcDevices(deviceCount);
        using TemporaryQwenGguf file = CreateFixture(tiedOutput: false, contextLength: 40);
        int[] devices = Enumerable.Range(0, deviceCount).ToArray();
        using Qwen35QuantizedModel baseline = Qwen35QuantizedModel.Load(file.Path, devices);
        using Qwen35QuantizedModel cached = Qwen35QuantizedModel.Load(file.Path, devices);
        int[] firstPrompt = [1, 2, 0];
        int[] secondPrompt = [1, 2, 0, 3, 1];

        Assert.Equal(baseline.GenerateTokenIds(firstPrompt, 4),
            cached.GenerateTokenIdsWithPrefixReuse(firstPrompt, 4));
        Assert.Equal(0, cached.LastReusedPromptTokens);
        long[] uploadedBeforeSecond = cached.UploadedBytes.ToArray();
        Assert.Equal(baseline.GenerateTokenIds(secondPrompt, 4),
            cached.GenerateTokenIdsWithPrefixReuse(secondPrompt, 4));
        Assert.Equal(firstPrompt.Length, cached.LastReusedPromptTokens);
        // Only the two-token suffix and three generated tokens are forwarded.
        // No checkpoint transfer crosses the host boundary.
        if (deviceCount == 1)
            Assert.Equal((secondPrompt.Length - firstPrompt.Length + 3) * sizeof(int),
                cached.UploadedBytes[0] - uploadedBeforeSecond[0]);

        int[] changedPrompt = [2, 1, 3];
        Assert.Equal(baseline.GenerateTokenIds(changedPrompt, 3),
            cached.GenerateTokenIdsWithPrefixReuse(changedPrompt, 3));
        Assert.Equal(0, cached.LastReusedPromptTokens);
        cached.Reset();
        Assert.Equal(baseline.GenerateTokenIds([2, 1, 3, 0], 2),
            cached.GenerateTokenIdsWithPrefixReuse([2, 1, 3, 0], 2));
        Assert.Equal(0, cached.LastReusedPromptTokens);
        Assert.Equal(baseline.GenerateTokenIds(firstPrompt, 2), cached.GenerateTokenIds(firstPrompt, 2));
        Assert.Equal(0, cached.LastReusedPromptTokens);

        cached.GenerateTokenIdsWithPrefixReuse(firstPrompt, 2);
        Assert.Throws<OperationCanceledException>(() => cached.GenerateTokenIdsWithPrefixReuse(
            [1, 2, 0, 3], 2, onToken: _ => throw new OperationCanceledException()));
        Assert.Equal(baseline.GenerateTokenIds(secondPrompt, 2),
            cached.GenerateTokenIdsWithPrefixReuse(secondPrompt, 2));
        Assert.Equal(0, cached.LastReusedPromptTokens);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HybridGenerationResetsStateAndKeepsEncodedWeightsResident(bool tiedOutput)
    {
        RequireArcDevices(1);
        using TemporaryQwenGguf file = CreateFixture(tiedOutput);
        using Qwen35QuantizedModel model = Qwen35QuantizedModel.Load(file.Path, [0]);
        long resident = Assert.Single(model.ResidentWeightBytes);
        Assert.True(resident > 0);
        long auxiliary = Assert.Single(model.ResidentAuxiliaryWeightBytes);
        Assert.True(auxiliary > 0);
        // Only encoded matrices and the small dense auxiliary weights are
        // uploaded at load time. Persistent states are zeroed on the GPU.
        Assert.Equal(resident + auxiliary, Assert.Single(model.UploadedBytes));
        Assert.Equal(0, Assert.Single(model.DownloadedBytes));
        Assert.True(Assert.Single(model.ResidentStateBytes) > 0);

        long beforeFirst = Assert.Single(model.UploadedBytes);
        float[] first = model.ForwardToken(1);
        AssertMeaningfulLogits(first);
        long afterFirst = Assert.Single(model.UploadedBytes);
        Assert.Equal(sizeof(int), afterFirst - beforeFirst);
        Assert.Equal(4 * sizeof(float), Assert.Single(model.DownloadedBytes));
        float[] second = model.ForwardToken(2);
        AssertMeaningfulLogits(second);
        long afterSecond = Assert.Single(model.UploadedBytes);
        Assert.Equal(sizeof(int), afterSecond - afterFirst);
        Assert.Equal(2 * 4 * sizeof(float), Assert.Single(model.DownloadedBytes));
        Assert.Equal(resident, Assert.Single(model.ResidentWeightBytes));

        long[] uploadedBeforeReset = model.UploadedBytes.ToArray();
        long[] downloadedBeforeReset = model.DownloadedBytes.ToArray();
        model.Reset();
        Assert.Equal(uploadedBeforeReset, model.UploadedBytes);
        Assert.Equal(downloadedBeforeReset, model.DownloadedBytes);
        Assert.Equal(first, model.ForwardToken(1));
        Assert.Equal(second, model.ForwardToken(2));

        // A fresh token at position zero has different logits from the same
        // token after a prefix, making the reset assertion state-sensitive.
        model.Reset();
        float[] withoutPrefix = model.ForwardToken(2);
        Assert.Contains(second.Zip(withoutPrefix, (left, right) => MathF.Abs(left - right)),
            difference => difference > 1e-7f);

        int[] prompt = [1, 2, 0];
        long uploadedBeforeGeneration = Assert.Single(model.UploadedBytes);
        long downloadedBeforeGeneration = Assert.Single(model.DownloadedBytes);
        int[] generated = model.GenerateTokenIds(prompt, 3);
        // Five tokens advance state: three prompt tokens, then the first two
        // generated tokens. Each of the three GPU argmax results is 8 bytes.
        Assert.Equal(5 * sizeof(int), Assert.Single(model.UploadedBytes) - uploadedBeforeGeneration);
        Assert.Equal(3 * 8, Assert.Single(model.DownloadedBytes) - downloadedBeforeGeneration);
        Assert.Equal(6, generated.Length);
        Assert.Equal(prompt, generated.Take(prompt.Length));
        Assert.All(generated, token => Assert.InRange(token, 0, 3));
        long[] stateAfterGeneration = model.ResidentStateBytes.ToArray();
        long[] liveAfterGeneration = model.LiveDeviceBytes.ToArray();
        Assert.Equal(generated, model.GenerateTokenIds(prompt, 3));
        Assert.Equal(stateAfterGeneration, model.ResidentStateBytes);
        Assert.Equal(liveAfterGeneration, model.LiveDeviceBytes);
        Assert.Equal(generated.Take(prompt.Length + 1),
            model.GenerateTokenIds(prompt, 3, eosTokenId: generated[prompt.Length]));
        Assert.Equal(resident, Assert.Single(model.ResidentWeightBytes));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TwoDevicesMatchSingleDeviceAcrossBothHybridLayers(bool tiedOutput)
    {
        RequireArcDevices(2);
        using TemporaryQwenGguf file = CreateFixture(tiedOutput);
        using Qwen35QuantizedModel single = Qwen35QuantizedModel.Load(file.Path, [0]);
        using Qwen35QuantizedModel split = Qwen35QuantizedModel.Load(file.Path, [0, 1]);

        Assert.Equal(single.ResidentWeightBytes.Sum(), split.ResidentWeightBytes.Sum());
        Assert.Equal(2, split.ResidentWeightBytes.Count);
        Assert.All(split.ResidentWeightBytes, bytes => Assert.True(bytes > 0));
        Assert.Equal(split.ResidentWeightBytes.Zip(split.ResidentAuxiliaryWeightBytes,
            (encoded, auxiliary) => encoded + auxiliary), split.UploadedBytes);
        Assert.All(split.DownloadedBytes, bytes => Assert.Equal(0, bytes));
        long hiddenBytes = split.Descriptor.EmbeddingLength * sizeof(float);
        long logitsBytes = split.Descriptor.VocabularySize * sizeof(float);
        foreach (int token in new[] { 1, 2, 0, 3 })
        {
            long[] uploadedBefore = split.UploadedBytes.ToArray();
            long[] downloadedBefore = split.DownloadedBytes.ToArray();
            float[] expected = single.ForwardToken(token);
            float[] actual = split.ForwardToken(token);
            AssertMeaningfulLogits(actual);
            for (int i = 0; i < expected.Length; i++)
                Assert.InRange(MathF.Abs(expected[i] - actual[i]), 0f, 1e-5f);
            // A tied head lives with the embedding on device 0, so the final
            // normalized hidden vector crosses back from device 1.
            Assert.Equal(sizeof(int) + (tiedOutput ? hiddenBytes : 0),
                split.UploadedBytes[0] - uploadedBefore[0]);
            Assert.Equal(hiddenBytes + (tiedOutput ? logitsBytes : 0),
                split.DownloadedBytes[0] - downloadedBefore[0]);
            Assert.Equal(hiddenBytes, split.UploadedBytes[1] - uploadedBefore[1]);
            Assert.Equal(tiedOutput ? hiddenBytes : logitsBytes,
                split.DownloadedBytes[1] - downloadedBefore[1]);
        }
        int[] expectedGenerated = single.GenerateTokenIds([1, 2], 4);
        long[] uploadedBeforeGeneration = split.UploadedBytes.ToArray();
        long[] downloadedBeforeGeneration = split.DownloadedBytes.ToArray();
        Assert.Equal(expectedGenerated, split.GenerateTokenIds([1, 2], 4));
        // All five forwarded tokens cross the layer boundary. Only four
        // require a head, and each argmax downloads exactly 8 bytes.
        Assert.Equal(5 * sizeof(int) + (tiedOutput ? 4 * hiddenBytes : 0),
            split.UploadedBytes[0] - uploadedBeforeGeneration[0]);
        Assert.Equal(5 * hiddenBytes + (tiedOutput ? 4 * 8 : 0),
            split.DownloadedBytes[0] - downloadedBeforeGeneration[0]);
        Assert.Equal(5 * hiddenBytes, split.UploadedBytes[1] - uploadedBeforeGeneration[1]);
        Assert.Equal(tiedOutput ? 4 * hiddenBytes : 4 * 8,
            split.DownloadedBytes[1] - downloadedBeforeGeneration[1]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OptimizedHybridModelMatchesReferenceThroughKvGrowthAndStreamingGeneration(bool tiedOutput)
    {
        RequireArcDevices(1);
        using TemporaryQwenGguf file = CreateFixture(tiedOutput, contextLength: 40);
        using Qwen35QuantizedModel reference = Qwen35QuantizedModel.Load(file.Path, [0], options: new()
        {
            QuantizedKernel = Qwen35QuantizedKernel.Reference,
            FusedDelta = false,
            CollectKernelTimings = true
        });
        using Qwen35QuantizedModel optimized = Qwen35QuantizedModel.Load(file.Path, [0], options: new()
        {
            QuantizedKernel = Qwen35QuantizedKernel.Auto,
            FusedDelta = true,
            CollectKernelTimings = true
        });
        long[] encodedBytes = optimized.ResidentWeightBytes.ToArray();
        Assert.Equal(reference.ResidentWeightBytes, encodedBytes);
        Assert.Equal(reference.ResidentAuxiliaryWeightBytes, optimized.ResidentAuxiliaryWeightBytes);
        long initialStateBytes = Assert.Single(optimized.ResidentStateBytes);
        long uploadedBefore = Assert.Single(optimized.UploadedBytes);
        long downloadedBefore = Assert.Single(optimized.DownloadedBytes);
        const int steps = 20;
        const float absoluteTolerance = 2e-4f, relativeTolerance = 2e-4f;
        for (int position = 0; position < steps; position++)
        {
            int token = (position * 3 + 1) % 4;
            float[] expected = reference.ForwardToken(token);
            float[] actual = optimized.ForwardToken(token);
            AssertMeaningfulLogits(expected);
            AssertMeaningfulLogits(actual);
            for (int index = 0; index < expected.Length; index++)
            {
                float error = MathF.Abs(expected[index] - actual[index]);
                float limit = absoluteTolerance + relativeTolerance * MathF.Abs(expected[index]);
                Assert.True(error <= limit,
                    $"Position {position}, logit {index}: reference={expected[index]:R}, optimized={actual[index]:R}, error={error:R}, limit={limit:R}.");
            }
        }
        // The hybrid model has one full-attention layer. Both routes grow its
        // K/V allocation from 16 to 32 slots while keeping weights encoded.
        Assert.Equal(initialStateBytes + 2L * 16 * 128 * sizeof(float),
            Assert.Single(optimized.ResidentStateBytes));
        Assert.Equal(reference.ResidentStateBytes, optimized.ResidentStateBytes);
        Assert.Equal(encodedBytes, optimized.ResidentWeightBytes);
        Assert.Equal(steps * sizeof(int), Assert.Single(optimized.UploadedBytes) - uploadedBefore);
        Assert.Equal(steps * 4 * sizeof(float), Assert.Single(optimized.DownloadedBytes) - downloadedBefore);
        Assert.Contains(optimized.KernelMilliseconds.Keys,
            kernel => kernel.StartsWith("q35l_", StringComparison.Ordinal));
        Assert.Contains("q35d_recurrent_gated_rmsnorm_fused128", optimized.KernelMilliseconds.Keys);
        Assert.DoesNotContain("q35d_recurrent_gated_rmsnorm_fused128", reference.KernelMilliseconds.Keys);

        int[] prompt = [1, 2, 0];
        var referenceCallbacks = new List<int>();
        var optimizedCallbacks = new List<int>();
        int[] referenceIds = reference.GenerateTokenIds(prompt, steps, onToken: referenceCallbacks.Add);
        uploadedBefore = Assert.Single(optimized.UploadedBytes);
        downloadedBefore = Assert.Single(optimized.DownloadedBytes);
        int[] optimizedIds = optimized.GenerateTokenIds(prompt, steps, onToken: optimizedCallbacks.Add);
        Assert.Equal(prompt.Length + steps, optimizedIds.Length);
        Assert.Equal(referenceIds, optimizedIds);
        Assert.Equal(referenceIds.Skip(prompt.Length), referenceCallbacks);
        Assert.Equal(optimizedIds.Skip(prompt.Length), optimizedCallbacks);
        Assert.Equal(referenceCallbacks, optimizedCallbacks);
        Assert.Equal((prompt.Length + steps - 1) * sizeof(int),
            Assert.Single(optimized.UploadedBytes) - uploadedBefore);
        Assert.Equal(steps * 8, Assert.Single(optimized.DownloadedBytes) - downloadedBefore);
        Assert.Equal(encodedBytes, optimized.ResidentWeightBytes);
    }

    [Fact]
    public void KvGrowthStaysOnDeviceAndResetReusesStateWithoutHostTransfers()
    {
        RequireArcDevices(1);
        using TemporaryQwenGguf file = CreateFixture(tiedOutput: false, contextLength: 40);
        using Qwen35QuantizedModel model = Qwen35QuantizedModel.Load(file.Path, [0]);
        long[] weights = model.ResidentWeightBytes.ToArray();
        long uploadedBefore = Assert.Single(model.UploadedBytes);
        long downloadedBefore = Assert.Single(model.DownloadedBytes);
        var logits = new Dictionary<int, float[]>();
        long atSixteen = 0, atThirtyTwo = 0;
        for (int position = 0; position < model.Descriptor.ContextLength; position++)
        {
            bool readLogits = position is 0 or 16 or 32 or 39;
            float[] result = model.ForwardToken(position % 4, readLogits);
            if (readLogits)
            {
                AssertMeaningfulLogits(result);
                logits.Add(position, result);
            }
            else Assert.Empty(result);
            if (position == 15) atSixteen = Assert.Single(model.ResidentStateBytes);
            if (position == 16)
            {
                // The fixture has one full-attention layer with 128-wide K/V.
                Assert.Equal(2L * 16 * 128 * sizeof(float),
                    Assert.Single(model.ResidentStateBytes) - atSixteen);
            }
            if (position == 31) atThirtyTwo = Assert.Single(model.ResidentStateBytes);
            if (position == 32)
                Assert.Equal(2L * 8 * 128 * sizeof(float),
                    Assert.Single(model.ResidentStateBytes) - atThirtyTwo);
        }
        Assert.Equal(40 * sizeof(int), Assert.Single(model.UploadedBytes) - uploadedBefore);
        Assert.Equal(4 * 4 * sizeof(float), Assert.Single(model.DownloadedBytes) - downloadedBefore);
        Assert.Equal(weights, model.ResidentWeightBytes);
        long[] stateAtCapacity = model.ResidentStateBytes.ToArray();
        long[] liveAtCapacity = model.LiveDeviceBytes.ToArray();
        long[] uploadsAtCapacity = model.UploadedBytes.ToArray();
        long[] downloadsAtCapacity = model.DownloadedBytes.ToArray();
        model.Reset();
        Assert.Equal(uploadsAtCapacity, model.UploadedBytes);
        Assert.Equal(downloadsAtCapacity, model.DownloadedBytes);
        Assert.Equal(stateAtCapacity, model.ResidentStateBytes);
        for (int position = 0; position < model.Descriptor.ContextLength; position++)
        {
            bool readLogits = logits.TryGetValue(position, out float[]? expected);
            float[] actual = model.ForwardToken(position % 4, readLogits);
            if (readLogits) Assert.Equal(expected, actual);
            else Assert.Empty(actual);
        }
        Assert.Equal(stateAtCapacity, model.ResidentStateBytes);
        Assert.Equal(liveAtCapacity, model.LiveDeviceBytes);
        Assert.Equal(weights, model.ResidentWeightBytes);
    }

    [Fact]
    public void ContextAndDisposalGuardsLeaveResetUsableBeforeDisposal()
    {
        RequireArcDevices(1);
        using TemporaryQwenGguf file = CreateFixture(tiedOutput: false);
        using Qwen35QuantizedModel model = Qwen35QuantizedModel.Load(file.Path, [0]);

        Assert.Throws<ArgumentOutOfRangeException>(() => model.ForwardToken(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.ForwardToken(4));
        float[] first = model.ForwardToken(1);
        for (int i = 1; i < model.Descriptor.ContextLength; i++)
            Assert.Empty(model.ForwardToken(i % 4, returnLogits: false));
        Assert.Contains("context length", Assert.Throws<InvalidOperationException>(
            () => model.ForwardToken(0)).Message);
        model.Reset();
        Assert.Equal(first, model.ForwardToken(1));

        model.Dispose();
        model.Dispose();
        Assert.Throws<ObjectDisposedException>(() => model.Reset());
        Assert.Throws<ObjectDisposedException>(() => model.ForwardToken(0));
        Assert.Throws<ObjectDisposedException>(() => model.GenerateTokenIds([0], 1));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NonFiniteOutputRequiresResetAfterStateWasAdvanced(bool greedyGeneration)
    {
        RequireArcDevices(1);
        using TemporaryQwenGguf file = CreateFixture(tiedOutput: false, nonFiniteOutputNorm: true);
        using Qwen35QuantizedModel model = Qwen35QuantizedModel.Load(file.Path, [0]);
        if (greedyGeneration)
            Assert.Throws<ArithmeticException>(() => model.GenerateTokenIds([1], 1));
        else
            Assert.Throws<ArithmeticException>(() => model.ForwardToken(1));
        Assert.Contains("Reset", Assert.Throws<InvalidOperationException>(
            () => model.ForwardToken(2)).Message);

        long[] uploadsBeforeReset = model.UploadedBytes.ToArray();
        long[] downloadsBeforeReset = model.DownloadedBytes.ToArray();
        model.Reset();
        Assert.Equal(uploadsBeforeReset, model.UploadedBytes);
        Assert.Equal(downloadsBeforeReset, model.DownloadedBytes);
        // The intentionally invalid weight still fails, but the previous
        // fault no longer blocks the execution of a freshly reset sequence.
        Assert.Throws<ArithmeticException>(() => model.ForwardToken(1));
    }

    private static void RequireArcDevices(int count)
        => Assert.SkipWhen(ArcDevices.Enumerate().Count < count,
            $"{count} Intel Arc GPU(s) required.");

    private static void AssertMeaningfulLogits(float[] logits)
    {
        Assert.Equal(4, logits.Length);
        Assert.All(logits, value => Assert.True(float.IsFinite(value)));
        Assert.True(logits.Max() - logits.Min() > 1e-5f,
            "The fixture must produce nonconstant logits to exercise real projections.");
    }

    internal static TemporaryQwenGguf CreateFixture(bool tiedOutput, uint contextLength = 8,
        bool nonFiniteOutputNorm = false)
    {
        var metadata = new Dictionary<string, object>
        {
            ["general.architecture"] = "qwen35",
            ["tokenizer.ggml.tokens"] = new[] { "a", "b", "c", "d" },
            ["qwen35.block_count"] = 2u,
            ["qwen35.embedding_length"] = 256u,
            ["qwen35.attention.head_count"] = 2u,
            ["qwen35.attention.head_count_kv"] = 1u,
            ["qwen35.attention.key_length"] = 128u,
            ["qwen35.attention.value_length"] = 128u,
            ["qwen35.context_length"] = contextLength,
            ["qwen35.feed_forward_length"] = 512u,
            ["qwen35.rope.dimension_count"] = 64u,
            ["qwen35.ssm.group_count"] = 1u,
            ["qwen35.ssm.time_step_rank"] = 2u,
            ["qwen35.ssm.state_size"] = 128u,
            ["qwen35.ssm.inner_size"] = 256u,
            ["qwen35.ssm.conv_kernel"] = 4u,
            ["qwen35.full_attention_interval"] = 2u
        };
        GgufTensorInfo[] directory =
        [
            new("token_embd.weight", [256, 4], Qwen2Gguf.Q4KType, 0),
            new("output_norm.weight", [256], Qwen2Gguf.F32Type, 0),
            new("output.weight", [256, 4], Qwen2Gguf.Q6KType, 0),
            new("blk.0.attn_norm.weight", [256], Qwen2Gguf.F32Type, 0),
            new("blk.0.post_attention_norm.weight", [256], Qwen2Gguf.F32Type, 0),
            new("blk.0.ffn_gate.weight", [256, 512], Qwen2Gguf.Q4KType, 0),
            new("blk.0.ffn_up.weight", [256, 512], Qwen2Gguf.Q4KType, 0),
            new("blk.0.ffn_down.weight", [512, 256], Qwen2Gguf.Q6KType, 0),
            new("blk.0.attn_qkv.weight", [256, 512], Qwen2Gguf.Q6KType, 0),
            new("blk.0.attn_gate.weight", [256, 256], Qwen2Gguf.Q4KType, 0),
            new("blk.0.ssm_alpha.weight", [256, 2], Qwen2Gguf.Q4KType, 0),
            new("blk.0.ssm_beta.weight", [256, 2], Qwen2Gguf.Q4KType, 0),
            new("blk.0.ssm_conv1d.weight", [4, 512], Qwen2Gguf.F32Type, 0),
            new("blk.0.ssm_a", [2], Qwen2Gguf.F32Type, 0),
            new("blk.0.ssm_dt.bias", [2], Qwen2Gguf.F32Type, 0),
            new("blk.0.ssm_norm.weight", [128], Qwen2Gguf.F32Type, 0),
            new("blk.0.ssm_out.weight", [256, 256], Qwen2Gguf.Q4KType, 0),
            new("blk.1.attn_norm.weight", [256], Qwen2Gguf.F32Type, 0),
            new("blk.1.post_attention_norm.weight", [256], Qwen2Gguf.F32Type, 0),
            new("blk.1.ffn_gate.weight", [256, 512], Qwen2Gguf.Q4KType, 0),
            new("blk.1.ffn_up.weight", [256, 512], Qwen2Gguf.Q4KType, 0),
            new("blk.1.ffn_down.weight", [512, 256], Qwen2Gguf.Q6KType, 0),
            new("blk.1.attn_q.weight", [256, 512], Qwen2Gguf.Q4KType, 0),
            new("blk.1.attn_k.weight", [256, 128], Qwen2Gguf.Q4KType, 0),
            new("blk.1.attn_v.weight", [256, 128], Qwen2Gguf.Q6KType, 0),
            new("blk.1.attn_q_norm.weight", [128], Qwen2Gguf.F32Type, 0),
            new("blk.1.attn_k_norm.weight", [128], Qwen2Gguf.F32Type, 0),
            new("blk.1.attn_output.weight", [256, 256], Qwen2Gguf.Q4KType, 0)
        ];
        using var payload = new MemoryStream();
        var tensors = new List<GgufTensorInfo>();
        foreach (GgufTensorInfo tensor in directory)
        {
            if (tiedOutput && tensor.Name == "output.weight") continue;
            while (payload.Position % 32 != 0) payload.WriteByte(0);
            tensors.Add(tensor with { Offset = (ulong)payload.Position });
            int elements = checked((int)tensor.Shape.Aggregate(1UL, (total, dimension) => total * dimension));
            byte[] bytes = tensor.Type == Qwen2Gguf.F32Type
                ? DensePayload(tensor.Name, elements)
                : QuantizedPayload(tensor.Name, tensor.Type, elements);
            if (nonFiniteOutputNorm && tensor.Name == "output_norm.weight")
                BitConverter.TryWriteBytes(bytes.AsSpan(0, sizeof(float)), float.NaN);
            payload.Write(bytes);
        }
        return new TemporaryQwenGguf(metadata, tensors, payload.ToArray());
    }

    private static byte[] DensePayload(string name, int elements)
    {
        var bytes = new byte[elements * sizeof(float)];
        for (int i = 0; i < elements; i++)
        {
            float value = name.EndsWith("ssm_a", StringComparison.Ordinal) ? -0.4f * (i + 1)
                : name.EndsWith("ssm_dt.bias", StringComparison.Ordinal) ? 0f
                : name.EndsWith("ssm_conv1d.weight", StringComparison.Ordinal)
                    ? (i % 4) switch { 0 => 0.03f, 1 => 0.07f, 2 => 0.15f, _ => 0.75f }
                : 1f;
            BitConverter.TryWriteBytes(bytes.AsSpan(i * sizeof(float)), value);
        }
        return bytes;
    }

    private static byte[] QuantizedPayload(string name, uint type, int elements)
    {
        // Fixed seed independent of randomized string hashes. Centered Q4
        // blocks and signed Q6 blocks keep projections small but nonzero.
        var random = new Random(name.Aggregate(17, (seed, ch) => unchecked(seed * 31 + ch)));
        int blockBytes = type == Qwen2Gguf.Q4KType ? GgufQ4K.BlockBytes : GgufQ6K.BlockBytes;
        var payload = new byte[elements / 256 * blockBytes];
        for (int offset = 0; offset < payload.Length; offset += blockBytes)
        {
            if (type == Qwen2Gguf.Q4KType)
            {
                WriteHalf(payload, offset, 1f / 1024f);
                WriteHalf(payload, offset + 2, 8f / 1024f);
                payload.AsSpan(offset + 4, 8).Fill(1);
                payload.AsSpan(offset + 12, 4).Fill(0x11);
                random.NextBytes(payload.AsSpan(offset + 16, 128));
            }
            else
            {
                random.NextBytes(payload.AsSpan(offset, 192));
                for (int i = 0; i < 16; i++)
                    payload[offset + 192 + i] = unchecked((byte)(sbyte)(i % 2 == 0 ? 1 : -1));
                WriteHalf(payload, offset + 208, 1f / 4096f);
            }
        }
        return payload;
    }

    private static void WriteHalf(byte[] bytes, int offset, float value)
        => BitConverter.TryWriteBytes(bytes.AsSpan(offset), BitConverter.HalfToUInt16Bits((Half)value));
}
