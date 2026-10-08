using System.Reflection;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35SplitOutputHeadTests
{
    private const string RangeKernel = "q35l_q5_k_sg16_range";
    private const long WorkspaceReserve = 64L * 1024 * 1024;

    [Fact]
    public void SplitOutputHeadRemainsOptIn()
        => Assert.False(new Qwen35ExecutionOptions().InferenceSplitOutputHead);

    [Theory]
    [InlineData(Qwen35Gguf.Q5KType, 256, 5, 2, false, true, false, false, true)]
    [InlineData(Qwen35Gguf.Q5KType, 5120, 248320, 3, false, true, false, false, true)]
    [InlineData(Qwen2Gguf.Q6KType, 256, 5, 2, false, true, false, false, false)]
    [InlineData(Qwen35Gguf.Q5KType, 0, 5, 2, false, true, false, false, false)]
    [InlineData(Qwen35Gguf.Q5KType, 257, 5, 2, false, true, false, false, false)]
    [InlineData(Qwen35Gguf.Q5KType, 256, 1, 2, false, true, false, false, false)]
    [InlineData(Qwen35Gguf.Q5KType, 256, 5, 1, false, true, false, false, false)]
    [InlineData(Qwen35Gguf.Q5KType, 256, 5, 2, true, true, false, false, false)]
    [InlineData(Qwen35Gguf.Q5KType, 256, 5, 2, false, false, false, false, false)]
    [InlineData(Qwen35Gguf.Q5KType, 256, 5, 2, false, true, true, false, false)]
    [InlineData(Qwen35Gguf.Q5KType, 256, 5, 2, false, true, false, true, false)]
    public void EligibilityRejectsUnsupportedRoutesBeforePreparingPeerBuffers(uint type, int width,
        int vocabulary, int devices, bool training, bool subgroup, bool outputLora, bool prism, bool expected)
        => Assert.Equal(expected, Qwen35QuantizedModel.CanSplitOutputHead(type, width,
            vocabulary, devices, training, subgroup, outputLora, prism));

    [Fact]
    public void PeerBudgetIncludesWeightsInputBiasOutputAndWorkspaceReserve()
    {
        // Three Q5_K rows: 528 encoded bytes, 1,024 input bytes, and 12 bytes
        // each for the persistent bias and output, plus the model reserve.
        const long liveBytes = 528 + 1024 + 12 + 12;
        Assert.True(Qwen35QuantizedModel.CanAllocateSplitOutputHead(256, 3, 1024, WorkspaceReserve + liveBytes));
        Assert.False(Qwen35QuantizedModel.CanAllocateSplitOutputHead(256, 3, 1024, WorkspaceReserve + liveBytes - 1));
        Assert.False(Qwen35QuantizedModel.CanAllocateSplitOutputHead(256, 3, 1023, long.MaxValue));
        // Here the peer weight is the largest individual allocation:
        // 2 Q5 blocks/row * 176 bytes/block * 7 rows = 2,464 bytes.
        Assert.True(Qwen35QuantizedModel.CanAllocateSplitOutputHead(512, 7, 2464, WorkspaceReserve + 4568));
        Assert.False(Qwen35QuantizedModel.CanAllocateSplitOutputHead(512, 7, 2463, long.MaxValue));
    }

    [Theory]
    [InlineData(0, 3)]
    [InlineData(257, 3)]
    [InlineData(256, 0)]
    [InlineData(256, -1)]
    [InlineData(256, int.MaxValue)]
    [InlineData(int.MaxValue, int.MaxValue)]
    public void InvalidOrUnrepresentablePeerAllocationsReturnFallback(int width, int peerRows)
        => Assert.False(Qwen35QuantizedModel.CanAllocateSplitOutputHead(width, peerRows, ulong.MaxValue, long.MaxValue));

    [Fact]
    public void UnavailablePeerHeadroomReturnsFallback()
    {
        Assert.False(Qwen35QuantizedModel.CanAllocateSplitOutputHead(256, 3, 1024, -1));
        Assert.False(Qwen35QuantizedModel.CanAllocateSplitOutputHead(256, 3, 1024, WorkspaceReserve));
        Assert.False(Qwen35QuantizedModel.CanAllocateSplitOutputHead(256, 3, 0, long.MaxValue));
    }

    [Theory]
    [InlineData(512, 7, 1)]
    [InlineData(5120, 17, 7)]
    [InlineData(5120, 17, 9)]
    public void RangeKernelPreservesLongInputBitsBiasOffsetsAndOutputGuards(int width, int vocabulary, int peerRows)
    {
        RequireSubgroupDevices(1);
        using var lane = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true });
        byte[] weights = QuantizedRows(Qwen35Gguf.Q5KType, width / 256 * vocabulary, 277);
        var random = new Random(991);
        float[] input = Enumerable.Range(0, width).Select(_ => random.NextSingle() * 2 - 1).ToArray();
        float[] biases = Enumerable.Range(0, vocabulary).Select(i => (i % 7 - 3) * .017f).ToArray();
        using ArcBuffer x = lane.Upload(input), w = lane.UploadRaw(weights), bias = lane.Upload(biases);
        using ArcBuffer expected = lane.Allocate(vocabulary);
        using ArcBuffer guarded = lane.Upload(Enumerable.Repeat(float.NaN, vocabulary + 4).ToArray());
        using ArcBuffer peerWeight = lane.UploadRaw(weights.AsSpan(0, peerRows * (width / 256) * 176).ToArray());
        using ArcBuffer peerBias = lane.Upload(biases[..peerRows]);
        lane.Run("q35l_q5_k_sg16", ((long)vocabulary + 1) / 2 * 32, 32,
            x, w, bias, expected, 1, width, vocabulary);
        // Owner reads from its original complete matrix/bias with nonzero
        // offsets; peer reads a compact prefix. Both leave two guard elements
        // before and after the merged output and preserve the original sums.
        int ownerRows = vocabulary - peerRows;
        lane.Run(RangeKernel, ((long)ownerRows + 1) / 2 * 32, 32,
            x, w, bias, guarded, width, ownerRows, peerRows, peerRows, peerRows + 2);
        lane.Run(RangeKernel, ((long)peerRows + 1) / 2 * 32, 32,
            x, peerWeight, peerBias, guarded, width, peerRows, 0, 0, 2);
        var expectedValues = new float[vocabulary];
        var guardedValues = new float[vocabulary + 4];
        lane.Read(expected, expectedValues);
        lane.Read(guarded, guardedValues);
        AssertBits(expectedValues, guardedValues.AsSpan(2, vocabulary).ToArray());
        foreach (int index in new[] { 0, 1, vocabulary + 2, vocabulary + 3 })
            Assert.True(float.IsNaN(guardedValues[index]), $"Output guard {index} was overwritten.");
    }

    [Theory]
    [InlineData(5, false)]
    [InlineData(7, true)]
    [InlineData(17, false)]
    public void OddUnequalSplitsPreserveEveryLogitAndGeneratedToken(int vocabulary, bool reverseDevices)
    {
        RequireSubgroupDevices(2);
        using TemporaryQwenGguf file = CreateFixture(vocabulary);
        int[] devices = reverseDevices ? [1, 0] : [0, 1];
        var options = new Qwen35ExecutionOptions { CollectKernelTimings = true };
        using Qwen35QuantizedModel original = Qwen35QuantizedModel.Load(file.Path, devices, options: options);
        using Qwen35QuantizedModel split = Qwen35QuantizedModel.Load(file.Path, devices,
            options: options with { InferenceSplitOutputHead = true });
        Assert.False(original.SplitOutputHeadAvailable);
        Assert.True(split.SplitOutputHeadAvailable);
        Assert.Equal(vocabulary, split.Descriptor.VocabularySize);
        // Odd vocabulary sizes put floor(N/2) rows on the peer and the larger
        // remainder on the owner, covering both sides of the split boundary.
        foreach (int token in new[] { 1, vocabulary - 1, 0, 2, vocabulary / 2, 3 })
        {
            float[] expected = original.ForwardToken(token), actual = split.ForwardToken(token);
            Assert.Equal(vocabulary, actual.Length);
            Assert.True(expected.Max() - expected.Min() > 1e-5f);
            AssertBits(expected, actual);
        }
        Assert.All(Lanes(split), lane => Assert.Contains(RangeKernel, lane.KernelTimings.Keys));
        Assert.DoesNotContain(RangeKernel, original.KernelMilliseconds.Keys);
        Assert.Equal(original.ResidentStateBytes, split.ResidentStateBytes);

        int[] prompt = [1, 3, 0, 2];
        Assert.Equal(original.GenerateTokenIds(prompt, 6), split.GenerateTokenIds(prompt, 6));
        AssertBits(original.ForwardToken(vocabulary - 1), split.ForwardToken(vocabulary - 1));
        original.Reset(); split.Reset();
        AssertBits(original.ForwardToken(1), split.ForwardToken(1));

        // Prefix restoration exercises the same output route after recurrent
        // snapshots and after growth of the full-attention cache past 16 rows.
        int[] history = Enumerable.Range(0, 17).Select(i => (i * 3 + 1) % vocabulary).ToArray();
        Assert.Equal(original.PrimePromptPrefix(history, TestContext.Current.CancellationToken),
            split.PrimePromptPrefix(history, TestContext.Current.CancellationToken));
        int[] continuation = [.. history, 2, 1];
        Assert.Equal(original.GenerateTokenIdsWithPrefixReuse(continuation, 3),
            split.GenerateTokenIdsWithPrefixReuse(continuation, 3));
        Assert.Equal(history.Length, split.LastReusedPromptTokens);
        AssertBits(original.ForwardToken(0), split.ForwardToken(0));
    }

    [Theory]
    [InlineData(1, Qwen35Gguf.Q5KType, false)]
    [InlineData(2, Qwen2Gguf.Q6KType, false)]
    [InlineData(2, Qwen35Gguf.Q5KType, true)]
    public void SingleDeviceUnsupportedStorageAndTiedOutputRetainExistingPath(int deviceCount,
        uint outputType, bool tiedOutput)
    {
        RequireDevices(deviceCount);
        using TemporaryQwenGguf file = CreateFixture(7, outputType, tiedOutput);
        int[] devices = Enumerable.Range(0, deviceCount).ToArray();
        var options = new Qwen35ExecutionOptions { CollectKernelTimings = true };
        using Qwen35QuantizedModel original = Qwen35QuantizedModel.Load(file.Path, devices, options: options);
        using Qwen35QuantizedModel requested = Qwen35QuantizedModel.Load(file.Path, devices,
            options: options with { InferenceSplitOutputHead = true });
        Assert.False(requested.SplitOutputHeadAvailable);
        Assert.Equal(original.LiveDeviceBytes, requested.LiveDeviceBytes);
        foreach (int token in new[] { 1, 6, 2, 0 })
            AssertBits(original.ForwardToken(token), requested.ForwardToken(token));
        Assert.DoesNotContain(RangeKernel, requested.KernelMilliseconds.Keys);
        Assert.Equal(original.GenerateTokenIds([1, 3, 0, 2], 4), requested.GenerateTokenIds([1, 3, 0, 2], 4));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AttachingNonzeroOutputLoraDisablesSplitAndPreservesAdapterLogits(bool fusedLora)
    {
        RequireSubgroupDevices(2);
        using TemporaryQwenGguf file = CreateFixture(7);
        var options = new Qwen35ExecutionOptions { CollectKernelTimings = true, InferenceFusedLora = fusedLora };
        using Qwen35QuantizedModel original = Qwen35QuantizedModel.Load(file.Path, [0, 1], options: options);
        using Qwen35QuantizedModel split = Qwen35QuantizedModel.Load(file.Path, [0, 1],
            options: options with { InferenceSplitOutputHead = true });
        Assert.True(split.SplitOutputHeadAvailable);
        float[] unadapted = original.ForwardToken(1);
        AssertBits(unadapted, split.ForwardToken(1));
        // Complete profile collection for the previous split invocation before
        // asserting that attaching LoRA causes no additional range dispatches.
        foreach (ArcExecutionLane lane in Lanes(split)) lane.Synchronize();
        Assert.Contains(RangeKernel, split.KernelMilliseconds.Keys);
        double rangeBefore = split.KernelMilliseconds[RangeKernel];

        var lora = new Qwen35LoraOptions { Rank = 2, Alpha = 4, Targets = [], IncludeOutput = true, Seed = 53 };
        original.AttachLora(lora); split.AttachLora(lora);
        float[][] state = original.LoraMatrices["output.weight"].ReadState();
        var random = new Random(811);
        foreach (int part in new[] { 0, 1 })
            for (int i = 0; i < state[part].Length; i++) state[part][i] = random.NextSingle() * .2f - .1f;
        original.LoraMatrices["output.weight"].RestoreState(state, training: false);
        split.LoraMatrices["output.weight"].RestoreState(state, training: false);
        Assert.False(split.SplitOutputHeadAvailable);
        float[] adapted = original.ForwardToken(1);
        Assert.Contains(adapted.Zip(unadapted, (a, b) => MathF.Abs(a - b)), change => change > 1e-6f);
        AssertBits(adapted, split.ForwardToken(1));
        foreach (int token in new[] { 6, 2, 0 }) AssertBits(original.ForwardToken(token), split.ForwardToken(token));
        Assert.Equal(original.GenerateTokenIds([1, 3, 0, 2], 4), split.GenerateTokenIds([1, 3, 0, 2], 4));
        Assert.Equal(rangeBefore, split.KernelMilliseconds[RangeKernel]);
    }

    [Fact]
    public void ReducedOwnerBudgetUsesOriginalHeadAndCanResumeSplitting()
    {
        RequireSubgroupDevices(2);
        using TemporaryQwenGguf file = CreateFixture(7);
        var options = new Qwen35ExecutionOptions { CollectKernelTimings = true };
        using Qwen35QuantizedModel original = Qwen35QuantizedModel.Load(file.Path, [0, 1], options: options);
        using Qwen35QuantizedModel split = Qwen35QuantizedModel.Load(file.Path, [0, 1],
            options: options with { InferenceSplitOutputHead = true });
        Assert.True(split.SplitOutputHeadAvailable);
        ArcExecutionLane owner = Lanes(split)[1];
        // The tiny model's normal forward fits comfortably in 64 MiB. The
        // split head requires that reserve in addition to its output, so this
        // reservation forces its fallback without inducing an allocation error.
        long reservation = owner.EffectivePhysicalBufferBudgetBytes - owner.AllocatedBytes - WorkspaceReserve;
        Assert.True(reservation >= 0);
        split.SetExternalDeviceMemoryReservation(1, reservation);
        AssertBits(original.ForwardToken(1), split.ForwardToken(1));
        Assert.DoesNotContain(RangeKernel, split.KernelMilliseconds.Keys);
        split.SetExternalDeviceMemoryReservation(1, 0);
        AssertBits(original.ForwardToken(6), split.ForwardToken(6));
        Assert.Contains(RangeKernel, split.KernelMilliseconds.Keys);
    }

    [Fact]
    public void PeerReservationReleasesAllAuxiliaryHeadBuffersAndKeepsOriginalRoute()
    {
        RequireSubgroupDevices(2);
        using TemporaryQwenGguf file = CreateFixture(7);
        var options = new Qwen35ExecutionOptions { CollectKernelTimings = true };
        using Qwen35QuantizedModel original = Qwen35QuantizedModel.Load(file.Path, [0, 1], options: options);
        using Qwen35QuantizedModel split = Qwen35QuantizedModel.Load(file.Path, [0, 1],
            options: options with { InferenceSplitOutputHead = true });
        ArcExecutionLane originalPeer = Lanes(original)[0], splitPeer = Lanes(split)[0];
        Assert.True(split.SplitOutputHeadAvailable);
        long[] splitWeights = split.ResidentWeightBytes.ToArray();
        long splitLive = splitPeer.AllocatedBytes;
        long normalBudget = splitPeer.EffectivePhysicalBufferBudgetBytes + splitPeer.ExternalMemoryReservationBytes;
        const long peerWeightBytes = 3 * 176;
        const long peerBufferBytes = peerWeightBytes + 256 * sizeof(float) + 2 * 3 * sizeof(float);
        Assert.Equal(peerWeightBytes, splitWeights[0] - original.ResidentWeightBytes[0]);
        Assert.Equal(peerBufferBytes, splitLive - originalPeer.AllocatedBytes);

        // Invalid API arguments must retain the prepared head and all its
        // weights/buffers; they are not legitimate memory-pressure fallbacks.
        foreach (long invalid in new[] { -1L, checked(normalBudget + 1) })
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => split.SetExternalDeviceMemoryReservation(0, invalid));
            Assert.True(split.SplitOutputHeadAvailable);
            Assert.Equal(splitWeights, split.ResidentWeightBytes);
            Assert.Equal(splitLive, splitPeer.AllocatedBytes);
            Assert.Equal(0, splitPeer.ExternalMemoryReservationBytes);
        }

        // Leave exactly enough room for the original model's live buffers.
        // This is valid for the original model but exceeds the split model's
        // live budget until its weight, input, bias, and output are all freed.
        long baselineLive = originalPeer.AllocatedBytes;
        long reservation = normalBudget - baselineLive;
        Assert.True(reservation > 0);
        Assert.Equal(normalBudget, originalPeer.EffectivePhysicalBufferBudgetBytes);
        original.SetExternalDeviceMemoryReservation(0, reservation);
        split.SetExternalDeviceMemoryReservation(0, reservation);
        Assert.Equal(reservation, splitPeer.ExternalMemoryReservationBytes);
        Assert.Equal(baselineLive, splitPeer.EffectivePhysicalBufferBudgetBytes);
        Assert.Equal(baselineLive, splitPeer.AllocatedBytes);
        Assert.Equal(baselineLive, split.TotalNativeDeviceBytes[0]);
        Assert.Equal(original.ResidentWeightBytes, split.ResidentWeightBytes);
        Assert.False(split.SplitOutputHeadAvailable);

        // Return transient workspace before forwarding. Releasing the external
        // reservation must not silently recreate the discarded auxiliary head.
        original.SetExternalDeviceMemoryReservation(0, 0);
        split.SetExternalDeviceMemoryReservation(0, 0);
        Assert.False(split.SplitOutputHeadAvailable);
        foreach (int token in new[] { 1, 6, 2, 0, 3 })
            AssertBits(original.ForwardToken(token), split.ForwardToken(token));
        Assert.Equal(original.ResidentWeightBytes, split.ResidentWeightBytes);
        Assert.False(split.SplitOutputHeadAvailable);
        Assert.DoesNotContain(RangeKernel, split.KernelMilliseconds.Keys);
    }

    [Fact]
    public void DisposingSplitModelReleasesPeerAllocationsAndTheModelFile()
    {
        RequireSubgroupDevices(2);
        using TemporaryQwenGguf file = CreateFixture(17);
        var model = Qwen35QuantizedModel.Load(file.Path, [0, 1], options: new()
        {
            InferenceSplitOutputHead = true
        });
        ArcExecutionLane[] lanes = Lanes(model);
        float[] expected;
        try
        {
            Assert.True(model.SplitOutputHeadAvailable);
            expected = model.ForwardToken(1);
            Assert.All(expected, value => Assert.True(float.IsFinite(value)));
            Assert.All(lanes, lane => Assert.True(lane.AllocatedBytes > 0));
        }
        finally { model.Dispose(); }
        model.Dispose(); // Owners and persistent peer buffers must be idempotent.
        Assert.All(lanes, lane =>
        {
            Assert.Equal(0, lane.AllocatedBytes);
            Assert.Equal(0, lane.CachedBytes);
            Assert.Equal(0, lane.RetiredBytes);
        });
        Assert.Throws<ObjectDisposedException>(() => model.ForwardToken(1));
        using (var exclusive = new FileStream(file.Path, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
            Assert.True(exclusive.Length > 0);
        using Qwen35QuantizedModel reloaded = Qwen35QuantizedModel.Load(file.Path, [0, 1], options: new()
        {
            InferenceSplitOutputHead = true
        });
        Assert.True(reloaded.SplitOutputHeadAvailable);
        AssertBits(expected, reloaded.ForwardToken(1));
    }

    [Theory]
    [InlineData(5)]
    [InlineData(7)]
    [InlineData(17)]
    public void OddVocabularyFixtureContainsDistinctQ5OutputRows(int vocabulary)
    {
        using TemporaryQwenGguf file = CreateFixture(vocabulary);
        Qwen35GgufDescriptor descriptor = Qwen35Gguf.Inspect(file.Path);
        Assert.Equal(vocabulary, descriptor.VocabularySize);
        using var reader = new GgufReader(file.Path);
        GgufTensorInfo head = reader.GetTensor("output.weight");
        Assert.Equal(Qwen35Gguf.Q5KType, head.Type);
        Assert.Equal(new ulong[] { 256, (ulong)vocabulary }, head.Shape);
        byte[] payload = reader.ReadTensorBytes(head, vocabulary * 176);
        Assert.Equal(vocabulary, Enumerable.Range(0, vocabulary)
            .Select(row => Convert.ToHexString(payload.AsSpan(row * 176, 176))).Distinct().Count());
    }

    // Reuse the established hybrid-transformer fixture, replacing only its
    // embedding vocabulary and output head. Each Q5_K row has a distinct
    // payload so an incorrect split offset cannot pass by repeating a row.
    private static TemporaryQwenGguf CreateFixture(int vocabulary, uint outputType = Qwen35Gguf.Q5KType,
        bool tiedOutput = false)
    {
        using TemporaryQwenGguf source = Qwen35ResidentModelTests.CreateFixture(false, contextLength: 48);
        using var reader = new GgufReader(source.Path);
        var metadata = reader.Metadata.ToDictionary(pair => pair.Key, pair => pair.Value);
        metadata["tokenizer.ggml.tokens"] = Enumerable.Range(0, vocabulary).Select(i => $"token-{i}").ToArray();
        using var payload = new MemoryStream();
        var tensors = new List<GgufTensorInfo>();
        foreach (GgufTensorInfo original in reader.Tensors)
        {
            if (tiedOutput && original.Name == "output.weight") continue;
            GgufTensorInfo tensor = original;
            byte[] bytes;
            if (original.Name == "output.weight")
            {
                tensor = original with { Shape = new ulong[] { 256, (ulong)vocabulary }, Type = outputType };
                bytes = QuantizedRows(outputType, vocabulary, 397);
            }
            else if (original.Name == "token_embd.weight")
            {
                tensor = original with { Shape = new ulong[] { 256, (ulong)vocabulary } };
                bytes = QuantizedRows(original.Type, vocabulary, 211);
            }
            else
            {
                long elements = original.Shape.Aggregate(1L, (total, width) => checked(total * (long)width));
                long byteCount = original.Type == Qwen2Gguf.F32Type ? elements * sizeof(float)
                    : elements / Qwen35Gguf.QuantizedBlockElements(original.Type)
                        * Qwen35Gguf.QuantizedBlockBytes(original.Type);
                bytes = reader.ReadTensorBytes(original, checked((int)byteCount));
            }
            while (payload.Position % 32 != 0) payload.WriteByte(0);
            tensors.Add(tensor with { Offset = (ulong)payload.Position });
            payload.Write(bytes);
        }
        return new TemporaryQwenGguf(metadata, tensors, payload.ToArray());
    }

    private static byte[] QuantizedRows(uint type, int rows, int seed)
    {
        int blockBytes = Qwen35Gguf.QuantizedBlockBytes(type);
        var bytes = new byte[checked(rows * blockBytes)];
        var random = new Random(seed);
        for (int row = 0; row < rows; row++)
        {
            Span<byte> block = bytes.AsSpan(row * blockBytes, blockBytes);
            random.NextBytes(block);
            if (type is Qwen2Gguf.Q4KType or Qwen35Gguf.Q5KType)
            {
                BitConverter.TryWriteBytes(block, BitConverter.HalfToUInt16Bits((Half)((row % 5 + 1) / 4096f)));
                BitConverter.TryWriteBytes(block[2..], BitConverter.HalfToUInt16Bits((Half)((row % 3 + 1) / 4096f)));
                block.Slice(4, 8).Fill(1);
                block.Slice(12, 4).Fill(0x11);
            }
            else if (type == Qwen2Gguf.Q6KType)
            {
                for (int i = 0; i < 16; i++) block[192 + i] = unchecked((byte)(sbyte)(i % 2 == 0 ? 1 : -1));
                BitConverter.TryWriteBytes(block[208..], BitConverter.HalfToUInt16Bits((Half)((row % 5 + 1) / 4096f)));
            }
            else throw new ArgumentOutOfRangeException(nameof(type));
        }
        return bytes;
    }

    private static ArcExecutionLane[] Lanes(Qwen35QuantizedModel model)
        => ((IReadOnlyList<ArcExecutionLane>)typeof(Qwen35QuantizedModel)
            .GetField("_lanes", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(model)!).ToArray();

    private static void RequireDevices(int count)
        => Assert.SkipWhen(ArcDevices.Enumerate().Count < count, $"{count} Intel Arc GPU(s) required.");

    private static void RequireSubgroupDevices(int count)
    {
        RequireDevices(count);
        Assert.SkipWhen(ArcDevices.Enumerate().Take(count).Any(device => !device.SupportsXmx
            || device.MinimumSubgroupSize != 16 || !device.Extensions.Split(' ').Contains("cl_intel_subgroups")),
            $"{count} Intel SG16 Arc GPU(s) required.");
    }

    private static void AssertBits(float[] expected, float[] actual)
    {
        Assert.All(expected, value => Assert.True(float.IsFinite(value)));
        Assert.All(actual, value => Assert.True(float.IsFinite(value)));
        Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits).ToArray(),
            actual.Select(BitConverter.SingleToInt32Bits).ToArray());
    }
}
