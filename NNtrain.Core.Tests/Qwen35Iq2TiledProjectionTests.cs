using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35Iq2TiledProjectionTests(ITestOutputHelper output)
{
    private const int Guard = 7;

    [Theory]
    [InlineData(17, 256)]
    [InlineData(129, 512)]
    [InlineData(129, 5120)]
    public void ForwardTilesPreserveBiasedProjectionAndSplitPanelTails(int rows, int inputWidth)
    {
        using var lane = CreateLane();
        const int outputWidth = 65;
        int count = rows * outputWidth;
        float[] input = Values(rows * inputWidth, 920 + inputWidth);
        // Include a broad dynamic range within each row in addition to the
        // separate all-tiny-row test below.
        for (int i = 0; i < input.Length; i += 31) input[i] = (i % 2 == 0 ? 1 : -1) * 1e-12f;
        float[] bias = Values(outputWidth, 319);
        byte[] encoded = Encoded(inputWidth, outputWidth);
        using ArcBuffer x = lane.Upload(input), w = lane.UploadRaw(encoded), b = lane.Upload(bias);
        using ArcBuffer expectedBuffer = lane.Upload(Poison(count));
        lane.Run("q35l_iq2_s_sg16", ((long)count + 1) / 2 * 32, 32,
            x, w, b, expectedBuffer, rows, inputWidth, outputWidth);
        float[] expected = Read(lane, expectedBuffer, count + Guard);
        var variants = new[] { (Rows: 8, Columns: 16, Unroll: false),
            (Rows: 16, Columns: 16, Unroll: false), (Rows: 16, Columns: 32, Unroll: false),
            (Rows: 32, Columns: 32, Unroll: false), (Rows: 16, Columns: 32, Unroll: true),
            (Rows: 32, Columns: 32, Unroll: true), (Rows: 128, Columns: 64, Unroll: false) };
        foreach (var variant in variants)
        foreach (int panel in new[] { 0, 16, 32 })
        {
            using ArcBuffer result = lane.Upload(Poison(count));
            long uploaded = lane.H2DBytes, downloaded = lane.D2HBytes;
            lane.Iq2TiledForward(x, w, b, result, rows, inputWidth, outputWidth,
                variant.Rows, variant.Columns, panel, variant.Unroll);
            Assert.Equal(uploaded, lane.H2DBytes);
            Assert.Equal(downloaded, lane.D2HBytes);
            float[] actual = Read(lane, result, count + Guard);
            AssertProjection(expected, actual, count, 5e-5,
                $"forward {rows}x{outputWidth}x{inputWidth}, tile={variant}, panel={panel}");
        }
        Assert.Equal(input, Read(lane, x, input.Length));
        Assert.Equal(encoded.Length, w.ByteLength);
    }

    [Theory]
    [InlineData(100_000f)]
    [InlineData(-100_000f)]
    public void WideDynamicRangePreservesTinyActivationRows(float extreme)
    {
        using var lane = CreateLane();
        const int rows = 17, inputWidth = 512, outputWidth = 65;
        int count = rows * outputWidth;
        float[] input = Values(rows * inputWidth, 440);
        for (int i = inputWidth; i < input.Length; ++i) input[i] *= 1e-12f;
        input[0] = extreme;
        using ArcBuffer x = lane.Upload(input), w = lane.UploadRaw(Encoded(inputWidth, outputWidth));
        // Zero bias here ensures that tiny-row errors are not hidden by bias.
        using ArcBuffer b = lane.Upload(new float[outputWidth]);
        using ArcBuffer expectedBuffer = lane.Upload(Poison(count));
        lane.Run("q35l_iq2_s_sg16", ((long)count + 1) / 2 * 32, 32,
            x, w, b, expectedBuffer, rows, inputWidth, outputWidth);
        float[] expected = Read(lane, expectedBuffer, count + Guard);
        foreach (int panel in new[] { 16, 32, 0 })
        {
            using ArcBuffer result = lane.Upload(Poison(count));
            lane.Iq2TiledForward(x, w, b, result, rows, inputWidth, outputWidth,
                tileRows: 32, tileColumns: 32, panelColumns: panel);
            float[] actual = Read(lane, result, count + Guard);
            AssertProjection(expected, actual, count, 5e-5, $"wide range extreme={extreme}, panel={panel}");
            for (int row = 1; row < rows; ++row)
                AssertRelative(expected.AsSpan(row * outputWidth, outputWidth),
                    actual.AsSpan(row * outputWidth, outputWidth), 5e-5, $"tiny row={row}");
        }
    }

    [Fact]
    public void NonFiniteWeightScalesRetainReferenceOutputClassification()
    {
        using var lane = CreateLane();
        const int rows = 17, inputWidth = 512, outputWidth = 65;
        int count = rows * outputWidth;
        byte[] encoded = Encoded(inputWidth, outputWidth);
        // Put exceptional scales in the second K256 block, so the direct
        // kernel must discard an already accumulated finite block on fallback.
        foreach (var item in new[] { (Column: 0, Scale: (ushort)0x7c00, Mixed: false),
            (Column: 1, Scale: (ushort)0xfc00, Mixed: false),
            (Column: 17, Scale: (ushort)0x7c00, Mixed: true),
            (Column: 64, Scale: (ushort)0x7e00, Mixed: false) })
        {
            int offset = (item.Column * 2 + 1) * 82;
            Array.Clear(encoded, offset, 82);
            encoded[offset] = (byte)item.Scale; encoded[offset + 1] = (byte)(item.Scale >> 8);
            if (item.Mixed) encoded[offset + 34] = 1;
        }
        float[] values = Enumerable.Repeat(1f, rows * inputWidth).ToArray();
        Array.Clear(values, 7 * inputWidth, inputWidth); // Zero times infinity must remain NaN.
        using ArcBuffer x = lane.Upload(values), w = lane.UploadRaw(encoded), bias = lane.Upload(new float[outputWidth]);
        using ArcBuffer reference = lane.Upload(Poison(count));
        lane.Run("q35l_iq2_s_sg16", ((long)count + 1) / 2 * 32, 32,
            x, w, bias, reference, rows, inputWidth, outputWidth);
        float[] expected = Read(lane, reference, count + Guard);
        Assert.True(float.IsPositiveInfinity(expected[0]));
        Assert.True(float.IsNegativeInfinity(expected[1]));
        Assert.True(float.IsNaN(expected[17]) && float.IsNaN(expected[64]));
        Assert.True(float.IsNaN(expected[7 * outputWidth]));
        foreach (var tile in new[] { (Rows: 16, Columns: 32), (Rows: 128, Columns: 64) })
        {
            using ArcBuffer result = lane.Upload(Poison(count));
            lane.Iq2TiledForward(x, w, bias, result, rows, inputWidth, outputWidth,
                tile.Rows, tile.Columns, panelColumns: 32);
            float[] actual = Read(lane, result, count + Guard);
            for (int i = 0; i < count; ++i)
            {
                if (float.IsNaN(expected[i])) Assert.True(float.IsNaN(actual[i]), $"tile={tile}, NaN index={i}");
                else if (float.IsInfinity(expected[i])) Assert.Equal(expected[i], actual[i]);
                else AssertClose(expected[i], actual[i], 5e-5, $"tile={tile}, finite index={i}");
            }
            for (int i = count; i < count + Guard; ++i) Assert.True(float.IsNaN(actual[i]));
        }
    }

    [Theory]
    [InlineData(17, 256, 80, false)]
    [InlineData(129, 512, 256, true)]
    public void TransposePanelsRetainGradientAddAndRowTails(int rows, int inputWidth, int outputWidth, bool add)
    {
        using var lane = CreateLane();
        int count = rows * inputWidth;
        float[] upstream = Values(rows * outputWidth, 333 + rows);
        float[] initial = add ? Values(count, 971) : new float[count];
        using ArcBuffer dy = lane.Upload(upstream), w = lane.UploadRaw(Encoded(inputWidth, outputWidth));
        using ArcBuffer partial = lane.Upload(Poison(count));
        using ArcBuffer reference = lane.Upload(initial.Concat(Enumerable.Repeat(float.NaN, Guard)).ToArray());
        lane.Run("q35t_xpose_iq2_s", count, 128,
            dy, w, partial, rows, inputWidth, outputWidth, 1, outputWidth);
        lane.Run("q35t_xpose_reduce", count, 128, partial, reference, rows, inputWidth, 1);
        float[] expected = Read(lane, reference, count + Guard);
        foreach (int panel in new[] { 16, 32, inputWidth })
        foreach (var tile in new[] { (Rows: 8, Columns: 16), (Rows: 16, Columns: 32), (Rows: 32, Columns: 32) })
        {
            using ArcBuffer result = lane.Upload(add
                ? initial.Concat(Enumerable.Repeat(float.NaN, Guard)).ToArray() : Poison(count));
            lane.Iq2TiledBackward(dy, w, result, rows, inputWidth, outputWidth,
                tile.Rows, tile.Columns, panel, addToOutput: add);
            AssertProjection(expected, Read(lane, result, count + Guard), count, 5e-5,
                $"transpose {rows}x{inputWidth}x{outputWidth}, add={add}, tile={tile}, panel={panel}");
        }
    }

    [Theory]
    [InlineData(1e-9f)]
    [InlineData(1e-20f)]
    public void TinyNormalRowsPreserveRelativePrecisionInBothDirections(float amplitude)
    {
        using var lane = CreateLane();
        const int rows = 17, inputWidth = 512, outputWidth = 80;
        float[] activation = Values(rows * inputWidth, 588).Select(value => value * amplitude).ToArray();
        float[] upstream = Values(rows * outputWidth, 891).Select(value => value * amplitude).ToArray();
        using ArcBuffer x = lane.Upload(activation), dy = lane.Upload(upstream);
        using ArcBuffer w = lane.UploadRaw(Encoded(inputWidth, outputWidth));
        using ArcBuffer zeroBias = lane.Upload(new float[outputWidth]);
        using ArcBuffer forwardReference = lane.Upload(Poison(rows * outputWidth));
        using ArcBuffer forward = lane.Upload(Poison(rows * outputWidth));
        lane.Run("q35l_iq2_s_sg16", ((long)rows * outputWidth + 1) / 2 * 32, 32,
            x, w, zeroBias, forwardReference, rows, inputWidth, outputWidth);
        lane.Iq2TiledForward(x, w, zeroBias, forward, rows, inputWidth, outputWidth, panelColumns: 32);
        AssertProjection(Read(lane, forwardReference, rows * outputWidth + Guard),
            Read(lane, forward, rows * outputWidth + Guard), rows * outputWidth, 5e-5, $"tiny forward amplitude={amplitude:E}");
        using ArcBuffer directForward = lane.Upload(Poison(rows * outputWidth));
        lane.Iq2TiledForward(x, w, zeroBias, directForward, rows, inputWidth, outputWidth,
            tileRows: 128, tileColumns: 64);
        AssertProjection(Read(lane, forwardReference, rows * outputWidth + Guard),
            Read(lane, directForward, rows * outputWidth + Guard), rows * outputWidth, 5e-5, $"tiny BSLM forward amplitude={amplitude:E}");
        using ArcBuffer transposeReference = lane.Upload(Poison(rows * inputWidth));
        using ArcBuffer transpose = lane.Upload(Poison(rows * inputWidth));
        lane.Run("q35t_xpose_iq2_s", rows * inputWidth, 128,
            dy, w, transposeReference, rows, inputWidth, outputWidth, 1, outputWidth);
        lane.Iq2TiledBackward(dy, w, transpose, rows, inputWidth, outputWidth, panelColumns: 32, addToOutput: false);
        AssertProjection(Read(lane, transposeReference, rows * inputWidth + Guard),
            Read(lane, transpose, rows * inputWidth + Guard), rows * inputWidth, 5e-5, $"tiny transpose amplitude={amplitude:E}");
    }

    [Fact]
    public void BoundedBackwardRowChunksPreserveAccumulationAndGuards()
    {
        using var lane = CreateLane();
        const int rows = 129, inputWidth = 512, outputWidth = 256, panelColumns = 32;
        const long chunkBudget = 100_000;
        int count = rows * inputWidth;
        float[] input = Values(rows * outputWidth, 709), initial = Values(count, 879);
        using ArcBuffer dy = lane.Upload(input), w = lane.UploadRaw(Encoded(inputWidth, outputWidth));
        using ArcBuffer full = lane.Upload(initial.Concat(Enumerable.Repeat(float.NaN, Guard)).ToArray());
        using ArcBuffer chunked = lane.Upload(initial.Concat(Enumerable.Repeat(float.NaN, Guard)).ToArray());
        lane.Iq2TiledBackward(dy, w, full, rows, inputWidth, outputWidth,
            panelColumns: panelColumns, workspaceBudgetBytes: 0);
        long uploaded = lane.H2DBytes, downloaded = lane.D2HBytes;
        lane.Iq2TiledBackward(dy, w, chunked, rows, inputWidth, outputWidth,
            panelColumns: panelColumns, workspaceBudgetBytes: chunkBudget);
        Assert.Equal(uploaded, lane.H2DBytes);
        Assert.Equal(downloaded, lane.D2HBytes);
        AssertProjection(Read(lane, full, count + Guard), Read(lane, chunked, count + Guard),
            count, 1e-6, "bounded backward chunks versus full rows");
        Assert.Equal(input, Read(lane, dy, input.Length));
    }

    [Fact]
    public void ForwardRowChunksPreserveBiasTailsAndGuardWithoutHostTransfers()
    {
        using var lane = CreateLane();
        const int rows = 129, inputWidth = 512, outputWidth = 65;
        int count = rows * outputWidth;
        float[] input = Values(rows * inputWidth, 715), biases = Values(outputWidth, 933);
        using ArcBuffer x = lane.Upload(input), w = lane.UploadRaw(Encoded(inputWidth, outputWidth));
        using ArcBuffer bias = lane.Upload(biases);
        using ArcBuffer full = lane.Upload(Poison(count)), chunked = lane.Upload(Poison(count));
        // This small matrix stays unchunked under the automatic default.
        lane.Iq2TiledForward(x, w, bias, full, rows, inputWidth, outputWidth, panelColumns: 32);
        long uploaded = lane.H2DBytes, downloaded = lane.D2HBytes;
        lane.Iq2TiledForward(x, w, bias, chunked, rows, inputWidth, outputWidth,
            panelColumns: 32, rowChunkRows: 16);
        Assert.Equal(uploaded, lane.H2DBytes);
        Assert.Equal(downloaded, lane.D2HBytes);
        AssertProjection(Read(lane, full, count + Guard), Read(lane, chunked, count + Guard),
            count, 1e-6, "forward row chunks with nonzero bias and final single-row chunk");
        Assert.Equal(input, Read(lane, x, input.Length));
        Assert.Equal(biases, Read(lane, bias, biases.Length));
    }

    [Theory]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void ModelPrefillAndLoraGradientsRemainCloseWithIndependentSwitches(bool forward, bool backward)
    {
        RequireArc();
        using TemporaryQwenGguf fixture = Qwen35ResidentModelTests.CreateFixture(
            tiedOutput: false, contextLength: 160, iq2Qkv: true, iq2Gate: true);
        var execution = new Qwen35ExecutionOptions
        {
            LoraTraining = true, CollectKernelTimings = true, TrainingGpuCheckpoints = true,
            InferencePrefillChunkTokens = 128, IQ2TiledForward = false, IQ2TiledBackward = false
        };
        using var reference = Qwen35QuantizedModel.Load(fixture.Path, [0], options: execution);
        using var candidate = Qwen35QuantizedModel.Load(fixture.Path, [0], options: execution with
        {
            IQ2TiledForward = forward, IQ2TiledBackward = backward
        });
        var adapter = new Qwen35LoraOptions
        {
            Rank = 2, Alpha = 4, IncludeOutput = true, Seed = 94, LearningRate = .001f
        };
        reference.AttachLora(adapter); candidate.AttachLora(adapter);
        // Nonzero B exercises both low-rank adjoints rather than just the
        // zero-initialized adapter's B gradient.
        var random = new Random(659);
        foreach (string name in reference.LoraMatrices.Keys)
        {
            float[][] state = reference.LoraMatrices[name].ReadState();
            foreach (int field in new[] { 0, 1 })
                for (int i = 0; i < state[field].Length; ++i) state[field][i] = (random.NextSingle() * 2 - 1) * .015f;
            reference.LoraMatrices[name].RestoreState(state, training: true);
            candidate.LoraMatrices[name].RestoreState(state, training: true);
        }
        reference.LoraCheckpointThresholdRows = 0; candidate.LoraCheckpointThresholdRows = 0;
        int[] tokens = Enumerable.Range(0, 129).Select(i => (i * 3 + i / 7) % 4).ToArray();
        reference.PrimePromptPrefix(tokens[..128], TestContext.Current.CancellationToken);
        candidate.PrimePromptPrefix(tokens[..128], TestContext.Current.CancellationToken);
        AssertRelative(reference.ForwardToken(tokens[^1]), candidate.ForwardToken(tokens[^1]), 1e-4, "prefill logits");
        reference.Reset(); candidate.Reset();
        Qwen35LoraStepResult expected = reference.ComputeLoraGradients(tokens, 96);
        Qwen35LoraStepResult actual = candidate.ComputeLoraGradients(tokens, 96);
        Assert.Equal(expected.SupervisedTokens, actual.SupervisedTokens);
        AssertClose(expected.Loss, actual.Loss, 1e-4, "loss");
        AssertClose(expected.GradientNorm, actual.GradientNorm, 1e-4, "gradient norm");
        double error = 0, scale = 0;
        foreach (var pair in reference.LoraMatrices)
        {
            float[][] left = pair.Value.ReadGradients(), right = candidate.LoraMatrices[pair.Key].ReadGradients();
            for (int field = 0; field < left.Length; ++field)
            for (int i = 0; i < left[field].Length; ++i)
            {
                Assert.True(float.IsFinite(right[field][i]));
                double delta = right[field][i] - left[field][i];
                error += delta * delta; scale += (double)left[field][i] * left[field][i];
            }
        }
        double relative = Math.Sqrt(error / Math.Max(scale, 1e-300));
        output.WriteLine($"model forward={forward}, backward={backward}: loss={actual.Loss:R}, gradient L2={relative:E6}");
        Assert.InRange(relative, 0, 1e-4);
        Assert.Equal(reference.ResidentWeightBytes, candidate.ResidentWeightBytes);
        if (forward) Assert.Contains(candidate.KernelMilliseconds.Keys, name => name.StartsWith("q35s_forward_", StringComparison.Ordinal));
        if (backward) Assert.Contains(candidate.KernelMilliseconds.Keys, name => name.StartsWith("q35s_transpose_", StringComparison.Ordinal));
        Assert.DoesNotContain(reference.KernelMilliseconds.Keys, name => name.StartsWith("q35s_", StringComparison.Ordinal));
    }

    private static ArcExecutionLane CreateLane()
    {
        RequireArc();
        return new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, Qwen35TrainingKernels = true,
            BufferPoolBytes = 32 * 1024 * 1024
        });
    }

    private static void RequireArc()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        var device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_khr_fp16")
            || !device.Extensions.Split(' ').Contains("cl_intel_subgroups_short"), "Intel SG16/F16 XMX is required.");
    }

    private static byte[] Encoded(int inputWidth, int outputWidth)
    {
        byte[] payload = new byte[(inputWidth / 256) * outputWidth * 82];
        new Random(inputWidth + outputWidth * 37).NextBytes(payload);
        for (int block = 0; block < payload.Length / 82; ++block)
        {
            ushort d = block % 19 == 0 ? (ushort)0x8001
                : BitConverter.HalfToUInt16Bits((Half)((block % 2 == 0 ? 1 : -1) * (block % 7 + 1) / 8192f));
            payload[block * 82] = (byte)d; payload[block * 82 + 1] = (byte)(d >> 8);
        }
        return payload;
    }

    private static float[] Values(int count, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, count).Select(_ => (random.NextSingle() * 2 - 1) * .75f).ToArray();
    }

    private static float[] Poison(int count) => Enumerable.Repeat(float.NaN, count + Guard).ToArray();

    private static float[] Read(ArcExecutionLane lane, ArcBuffer buffer, int count)
    {
        var values = new float[count]; lane.Read(buffer, values); return values;
    }

    private void AssertProjection(float[] expected, float[] actual, int count, double tolerance, string description)
    {
        for (int i = count; i < count + Guard; ++i) Assert.True(float.IsNaN(expected[i]) && float.IsNaN(actual[i]), $"{description}: guard={i}");
        AssertRelative(expected.AsSpan(0, count), actual.AsSpan(0, count), tolerance, description);
    }

    private void AssertRelative(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, double tolerance, string description)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = 0, scale = 0, maximum = 0;
        for (int i = 0; i < expected.Length; ++i)
        {
            Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]), $"{description}: nonfinite at {i}");
            double delta = actual[i] - expected[i];
            error += delta * delta; scale += (double)expected[i] * expected[i]; maximum = Math.Max(maximum, Math.Abs(delta));
        }
        double relative = Math.Sqrt(error / Math.Max(scale, 1e-300));
        output.WriteLine($"{description}: relative L2={relative:E6}, maximum absolute={maximum:E6}");
        Assert.InRange(relative, 0, tolerance);
    }

    private static void AssertClose(double expected, double actual, double tolerance, string label)
        => Assert.True(double.IsFinite(actual) && Math.Abs(expected - actual) <= tolerance * (1 + Math.Abs(expected)),
            $"{label}: expected={expected:R}, actual={actual:R}, tolerance={tolerance:R}");
}
