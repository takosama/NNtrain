using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35FusedResidualRmsTests
{
    [Fact]
    public void FusionRemainsOptIn() => Assert.False(new Qwen35ExecutionOptions().InferenceFusedResidualRms);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FusedKernelsRetainRoundedResidualNormBitsAndGuardValues(bool fast)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU required.");
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true,
            CacheProgramBinary = true,
            Qwen35FastRmsNorm = fast
        });
        Assert.SkipWhen(fast && (!lane.Options.XmxMatrices || !lane.Device.SupportsXmx
            || lane.Device.MinimumSubgroupSize != 16
            || !lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups")), "Intel SG16 required.");
        string originalKernel = fast ? "q35a_rms_norm_sg16_exact" : "q35a_rms_norm";
        string fusedKernel = fast ? "q35a_add_rms_norm_sg16_exact" : "q35a_add_rms_norm";
        const int rows = 3, guard = 11;
        foreach (int width in new[] { 1, 17, 128, 129, 5120, 17408 })
        {
            int count = rows * width;
            var random = new Random(width + 137);
            float[] values = Enumerable.Repeat(float.NaN, count + guard).ToArray();
            float[] residual = Enumerable.Repeat(float.NaN, count + guard).ToArray();
            for (int i = 0; i < count; i++)
            {
                float value = MathF.ScaleB(random.NextSingle() * 2f - 1f, (i % 7 - 3) * 12);
                values[i] = i % 29 == 0 ? -0f : value;
                residual[i] = (i % 5) switch
                {
                    0 => -values[i],
                    // Include additions close to a half-ULP rounding boundary.
                    1 => MathF.ScaleB(values[i], -24),
                    2 => -0f,
                    _ => MathF.ScaleB(random.NextSingle() * 2f - 1f, (i % 7 - 3) * 12)
                };
            }
            float[] weights = Enumerable.Range(0, width)
                .Select(i => i % 13 == 0 ? -0f : random.NextSingle() * 4f - 2f).ToArray();
            using ArcBuffer source = lane.Upload(residual), weight = lane.Upload(weights);
            using ArcBuffer oldHidden = lane.Upload(values), newHidden = lane.Upload(values);
            using ArcBuffer oldNorm = lane.Upload(Enumerable.Repeat(float.NaN, count + guard).ToArray());
            using ArcBuffer newNorm = lane.Upload(Enumerable.Repeat(float.NaN, count + guard).ToArray());
            long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
            lane.Run("q35a_add_in_place", count, 128, oldHidden, source, count);
            lane.Run(originalKernel, rows * 128, 128, oldHidden, weight, oldNorm, width, width, 1e-6f);
            lane.Run(fusedKernel, rows * 128, 128, newHidden, source, weight, newNorm, width, width, 1e-6f);
            Assert.Equal(uploads, lane.H2DBytes);
            Assert.Equal(downloads, lane.D2HBytes);
            float[] expectedHidden = Read(lane, oldHidden, count + guard);
            float[] actualHidden = Read(lane, newHidden, count + guard);
            float[] expectedNorm = Read(lane, oldNorm, count + guard);
            float[] actualNorm = Read(lane, newNorm, count + guard);
            AssertBits(expectedHidden[..count], actualHidden[..count]);
            AssertBits(expectedNorm[..count], actualNorm[..count]);
            Assert.All(actualNorm[..count], value => Assert.True(float.IsFinite(value)));
            for (int i = count; i < count + guard; i++)
                Assert.True(float.IsNaN(expectedHidden[i]) && float.IsNaN(actualHidden[i])
                    && float.IsNaN(expectedNorm[i]) && float.IsNaN(actualNorm[i]));
            AssertBits(residual, Read(lane, source, residual.Length));
            AssertBits(weights, Read(lane, weight, width));
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void FusedDecodeMatchesOriginalLogitsTokensAndCacheContinuation(bool fast, bool lora)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU required.");
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(
            tiedOutput: false, contextLength: 48, iq2Qkv: true, iq2Gate: true);
        var options = new Qwen35ExecutionOptions
        {
            InferenceFastRmsNorm = fast,
            InferencePrefillChunkTokens = 0,
            InferenceFusedResidualRms = false,
            CollectKernelTimings = true
        };
        using Qwen35QuantizedModel previous = Qwen35QuantizedModel.Load(file.Path, [0], options: options);
        using Qwen35QuantizedModel fused = Qwen35QuantizedModel.Load(file.Path, [0],
            options: options with { InferenceFusedResidualRms = true });
        if (lora) AttachMatchingLora(previous, fused);
        long oldUploads = Assert.Single(previous.UploadedBytes), newUploads = Assert.Single(fused.UploadedBytes);
        long oldDownloads = Assert.Single(previous.DownloadedBytes), newDownloads = Assert.Single(fused.DownloadedBytes);
        for (int position = 0; position < 19; position++)
        {
            int token = (position * 3 + 1) % 4;
            // Silent steps cover the unfused final residual and its following
            // cached continuation; all other steps exercise final output norm.
            bool returnLogits = position % 4 != 0;
            AssertBits(previous.ForwardToken(token, returnLogits), fused.ForwardToken(token, returnLogits));
        }
        Assert.Equal(previous.ResidentStateBytes, fused.ResidentStateBytes);
        Assert.Equal(Assert.Single(previous.UploadedBytes) - oldUploads, Assert.Single(fused.UploadedBytes) - newUploads);
        Assert.Equal(Assert.Single(previous.DownloadedBytes) - oldDownloads, Assert.Single(fused.DownloadedBytes) - newDownloads);
        Assert.Contains(fused.KernelMilliseconds.Keys, name => name.StartsWith("q35a_add_rms_norm", StringComparison.Ordinal));
        Assert.DoesNotContain(previous.KernelMilliseconds.Keys, name => name.StartsWith("q35a_add_rms_norm", StringComparison.Ordinal));
        if (!fast) Assert.Contains("q35a_add_rms_norm", fused.KernelMilliseconds.Keys);
        int[] prompt = [1, 2, 0, 3, 1, 0, 2];
        Assert.Equal(previous.GenerateTokenIds(prompt, 8), fused.GenerateTokenIds(prompt, 8));
        AssertBits(previous.ForwardToken(3), fused.ForwardToken(3));
    }

    private static void AttachMatchingLora(Qwen35QuantizedModel previous, Qwen35QuantizedModel fused)
    {
        var options = new Qwen35LoraOptions { Rank = 2, Alpha = 4, IncludeOutput = true, Seed = 113 };
        previous.AttachLora(options);
        fused.AttachLora(options);
        var random = new Random(227);
        foreach (string name in previous.LoraMatrices.Keys)
        {
            float[][] state = previous.LoraMatrices[name].ReadState();
            foreach (int index in new[] { 0, 1 })
                for (int i = 0; i < state[index].Length; i++)
                    state[index][i] = random.NextSingle() * .2f - .1f;
            previous.LoraMatrices[name].RestoreState(state, training: false);
            fused.LoraMatrices[name].RestoreState(state, training: false);
        }
    }

    private static void AssertBits(float[] expected, float[] actual)
        => Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), actual.Select(BitConverter.SingleToInt32Bits));

    private static float[] Read(ArcExecutionLane lane, ArcBuffer buffer, int count)
    {
        var values = new float[count];
        lane.Read(buffer, values);
        return values;
    }
}
