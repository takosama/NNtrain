using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35LoraCheckpointParityTests
{
    [Fact]
    public void Iq2ProjectionCachePrioritizesSavedWorkPerHostByte()
    {
        (string Name, int InputWidth, int OutputWidth)[] projections =
        [
            ("blk.0.low.weight", 256, 512),
            ("blk.0.high.weight", 1024, 256),
            ("blk.0.tiny.weight", 128, 128)
        ];
        HashSet<string> chosen = Qwen35QuantizedModel.SelectIq2ProjectionCacheTargets(
            projections, rows: 5, byteLimit: 5L * 512 * sizeof(float));

        Assert.Contains("blk.0.high.weight", chosen);
        Assert.Contains("blk.0.tiny.weight", chosen);
        Assert.DoesNotContain("blk.0.low.weight", chosen);
    }

    [Fact]
    public void GpuIq2CacheRejectsExcessiveOrSimultaneousBudgetsBeforeDeviceUse()
    {
        using TemporaryQwenGguf fixture = Qwen35ResidentModelTests.CreateFixture(tiedOutput: false);
        using var reader = new GgufReader(fixture.Path);
        Assert.Throws<ArgumentOutOfRangeException>(() => Qwen35QuantizedModel.Load(
            reader, options: new() { TrainingIQ2GpuProjectionCacheMiB = 1025 }));
        Assert.Throws<ArgumentException>(() => Qwen35QuantizedModel.Load(
            reader, options: new()
            {
                TrainingIQ2GpuProjectionCacheMiB = 1,
                TrainingIQ2ProjectionCacheMiB = 1
            }));
        Assert.Throws<ArgumentException>(() => Qwen35QuantizedModel.Load(
            reader, options: new()
            {
                TrainingIQ2GpuProjectionCacheMiB = 1,
                TrainingGpuCheckpoints = false
            }));
    }

    [Fact]
    public void VocabularyHeadScratchChunksDoNotRepeatWeightTraversalForRowTails()
    {
        const int rows = 976, rowTile = 16;
        const long bytesPerRow = 243L * 5120 * sizeof(float);
        const long scratchBudget = 256L * 1024 * 1024;
        int chunkRows = Qwen35QuantizedModel.AlignedTransposeChunkRows(
            rows, bytesPerRow, scratchBudget, rowTile);

        Assert.Equal(48, chunkRows);
        Assert.True(chunkRows * bytesPerRow <= scratchBudget);
        int passes = Enumerable.Range(0, (rows + chunkRows - 1) / chunkRows)
            .Sum(chunk => (Math.Min(chunkRows, rows - chunk * chunkRows) + rowTile - 1) / rowTile);
        Assert.Equal((rows + rowTile - 1) / rowTile, passes);
        Assert.Equal(5, Qwen35QuantizedModel.AlignedTransposeChunkRows(
            rows, bytesPerRow, 5 * bytesPerRow, rowTile));
    }

    [Theory]
    [InlineData(1, true, 1, true)]
    [InlineData(1, true, 1, false)]
    [InlineData(3, true, 1, true)]
    [InlineData(3, false, 1, true)]
    [InlineData(3, true, 2, true)]
    [InlineData(3, true, 2, false)]
    public void RecomputedLayersMatchFullTapeLossGradientsAndUpdate(
        int responseStart, bool responseOnlyHead, int deviceCount, bool gpuCheckpoints)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count < deviceCount, "Required Intel Arc GPUs are unavailable.");
        using TemporaryQwenGguf fixture = Qwen35ResidentModelTests.CreateFixture(tiedOutput: false);
        int[] devices = Enumerable.Range(0, deviceCount).ToArray();
        var execution = new Qwen35ExecutionOptions
        {
            LoraTraining = true,
            TrainingResponseOnlyHead = responseOnlyHead,
            TrainingGpuCheckpoints = gpuCheckpoints
        };
        using Qwen35QuantizedModel reference = Qwen35QuantizedModel.Load(fixture.Path, devices, options: execution);
        using Qwen35QuantizedModel recomputed = Qwen35QuantizedModel.Load(fixture.Path, devices, options: execution);
        var adapterOptions = new Qwen35LoraOptions
        {
            Rank = 2, Alpha = 4, IncludeOutput = true, Seed = 94, LearningRate = .001f
        };
        reference.AttachLora(adapterOptions);
        recomputed.AttachLora(adapterOptions);
        recomputed.LoraCheckpointThresholdRows = 0;
        recomputed.LoraTransposeScratchBudgetBytes = 64;

        var random = new Random(821);
        foreach (var pair in reference.LoraMatrices)
        {
            float[][] state = pair.Value.ReadState();
            foreach (float[] values in state.Take(2))
                for (int i = 0; i < values.Length; i++)
                    values[i] = (float)(random.NextDouble() * .04 - .02);
            pair.Value.RestoreState(state, training: true);
            recomputed.LoraMatrices[pair.Key].RestoreState(state, training: true);
        }

        int[] tokens = [1, 2, 3, 0, 1, 2];
        Qwen35LoraStepResult baseline = reference.ComputeLoraGradients(tokens, responseStart);
        Qwen35LoraStepResult checkpointed = recomputed.ComputeLoraGradients(tokens, responseStart);
        Close(baseline.Loss, checkpointed.Loss);
        Close(baseline.GradientNorm, checkpointed.GradientNorm);
        Assert.Equal(baseline.SupervisedTokens, checkpointed.SupervisedTokens);
        foreach (var pair in reference.LoraMatrices)
        {
            float[][] a = pair.Value.ReadGradients();
            float[][] b = recomputed.LoraMatrices[pair.Key].ReadGradients();
            for (int field = 0; field < a.Length; field++)
                for (int i = 0; i < a[field].Length; i++) Close(a[field][i], b[field][i]);
        }

        Qwen35LoraStepResult baselineStep = reference.TrainLora(tokens, responseStart);
        Qwen35LoraStepResult checkpointedStep = recomputed.TrainLora(tokens, responseStart);
        Assert.Equal(1, baselineStep.Step);
        Assert.Equal(baselineStep.Step, checkpointedStep.Step);
        Close(baselineStep.Loss, checkpointedStep.Loss);
        Close(baselineStep.GradientNorm, checkpointedStep.GradientNorm);
        foreach (var pair in reference.LoraMatrices)
        {
            float[][] a = pair.Value.ReadState();
            float[][] b = recomputed.LoraMatrices[pair.Key].ReadState();
            for (int field = 0; field < a.Length; field++)
                for (int i = 0; i < a[field].Length; i++) Close(a[field][i], b[field][i]);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void GpuLayerCheckpointsAvoidPerLayerHostTransfers(int deviceCount)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count < deviceCount, "Required Intel Arc GPUs are unavailable.");
        using TemporaryQwenGguf fixture = Qwen35ResidentModelTests.CreateFixture(tiedOutput: false);
        int[] devices = Enumerable.Range(0, deviceCount).ToArray();
        using Qwen35QuantizedModel host = Qwen35QuantizedModel.Load(fixture.Path, devices,
            options: new() { LoraTraining = true, TrainingGpuCheckpoints = false });
        using Qwen35QuantizedModel gpu = Qwen35QuantizedModel.Load(fixture.Path, devices,
            options: new() { LoraTraining = true, TrainingGpuCheckpoints = true });
        var adapter = new Qwen35LoraOptions { Rank = 2, Alpha = 4, Seed = 94 };
        host.AttachLora(adapter);
        gpu.AttachLora(adapter);
        host.LoraCheckpointThresholdRows = 0;
        gpu.LoraCheckpointThresholdRows = 0;
        int[] tokens = [1, 2, 3, 0, 1, 2];
        long hostUpload = host.UploadedBytes.Sum(), hostDownload = host.DownloadedBytes.Sum();
        long gpuUpload = gpu.UploadedBytes.Sum(), gpuDownload = gpu.DownloadedBytes.Sum();

        Close(host.EvaluateLoraLoss(tokens, 3), gpu.EvaluateLoraLoss(tokens, 3));

        hostUpload = host.UploadedBytes.Sum() - hostUpload;
        hostDownload = host.DownloadedBytes.Sum() - hostDownload;
        gpuUpload = gpu.UploadedBytes.Sum() - gpuUpload;
        gpuDownload = gpu.DownloadedBytes.Sum() - gpuDownload;
        Assert.True(gpuUpload < hostUpload,
            $"Expected GPU checkpoints to reduce H2D: {gpuUpload} versus {hostUpload} bytes.");
        Assert.True(gpuDownload < hostDownload,
            $"Expected GPU checkpoints to reduce D2H: {gpuDownload} versus {hostDownload} bytes.");
    }

    [Theory]
    [InlineData(1, false, false)]
    [InlineData(1, true, false)]
    [InlineData(2, false, false)]
    [InlineData(2, true, false)]
    [InlineData(1, true, true)]
    [InlineData(2, true, true)]
    public void CachedIq2BaseProjectionMatchesRecomputedLoraUpdate(
        int deviceCount, bool cacheExhausted, bool prioritize)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count < deviceCount, "Required Intel Arc GPUs are unavailable.");
        using TemporaryQwenGguf fixture = Qwen35ResidentModelTests.CreateFixture(
            tiedOutput: false, iq2Qkv: true, iq2Gate: cacheExhausted);
        int[] devices = Enumerable.Range(0, deviceCount).ToArray();
        var execution = new Qwen35ExecutionOptions { LoraTraining = true, TrainingGpuCheckpoints = true };
        using Qwen35QuantizedModel recomputed = Qwen35QuantizedModel.Load(
            fixture.Path, devices, options: execution);
        using Qwen35QuantizedModel cached = Qwen35QuantizedModel.Load(
            fixture.Path, devices, options: execution with
            {
                TrainingIQ2ProjectionCacheMiB = 1,
                TrainingIQ2ProjectionCachePrioritize = prioritize
            });
        var adapter = new Qwen35LoraOptions
        {
            Rank = 2, Alpha = 4, IncludeOutput = true, Seed = 94, LearningRate = .001f
        };
        recomputed.AttachLora(adapter);
        cached.AttachLora(adapter);
        recomputed.LoraCheckpointThresholdRows = 0;
        cached.LoraCheckpointThresholdRows = 0;
        int[] tokens = [1, 2, 3, 0, 1, 2];
        if (cacheExhausted)
            cached.LoraIq2ProjectionCacheBudgetBytesOverride = (tokens.Length - 1) * 512 * sizeof(float);
        long recomputedUpload = recomputed.UploadedBytes.Sum();
        long recomputedDownload = recomputed.DownloadedBytes.Sum();
        long cachedUpload = cached.UploadedBytes.Sum();
        long cachedDownload = cached.DownloadedBytes.Sum();

        Qwen35LoraStepResult reference = recomputed.TrainLora(tokens, 3);
        Qwen35LoraStepResult actual = cached.TrainLora(tokens, 3);
        recomputedUpload = recomputed.UploadedBytes.Sum() - recomputedUpload;
        recomputedDownload = recomputed.DownloadedBytes.Sum() - recomputedDownload;
        cachedUpload = cached.UploadedBytes.Sum() - cachedUpload;
        cachedDownload = cached.DownloadedBytes.Sum() - cachedDownload;

        Assert.Equal(reference.Step, actual.Step);
        Close(reference.Loss, actual.Loss);
        Close(reference.GradientNorm, actual.GradientNorm);
        Assert.Equal(0, recomputed.LastIq2ProjectionCacheStats.Captured);
        Assert.Equal(1, cached.LastIq2ProjectionCacheStats.Captured);
        Assert.Equal(1, cached.LastIq2ProjectionCacheStats.Reused);
        Assert.InRange(cached.LastIq2ProjectionCacheStats.PeakBytes, 1, 1024 * 1024);
        if (cacheExhausted)
            Assert.Equal(cached.LoraIq2ProjectionCacheBudgetBytesOverride,
                cached.LastIq2ProjectionCacheStats.PeakBytes);
        Assert.Equal(cached.LastIq2ProjectionCacheStats.PeakBytes, cachedUpload - recomputedUpload);
        Assert.Equal(cached.LastIq2ProjectionCacheStats.PeakBytes, cachedDownload - recomputedDownload);
        foreach (var pair in recomputed.LoraMatrices)
        {
            float[][] expected = pair.Value.ReadState();
            float[][] observed = cached.LoraMatrices[pair.Key].ReadState();
            for (int field = 0; field < expected.Length; field++)
                for (int i = 0; i < expected[field].Length; i++) Close(expected[field][i], observed[field][i]);
        }
    }

    [Theory]
    [InlineData(1, false)]
    [InlineData(2, false)]
    [InlineData(1, true)]
    [InlineData(2, true)]
    public void GpuIq2BaseCacheMatchesRecomputationWithoutHostTransfers(int deviceCount, bool fp16Forward)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count < deviceCount, "Required Intel Arc GPUs are unavailable.");
        using TemporaryQwenGguf fixture = Qwen35ResidentModelTests.CreateFixture(
            tiedOutput: false, iq2Qkv: true);
        int[] devices = Enumerable.Range(0, deviceCount).ToArray();
        var execution = new Qwen35ExecutionOptions
        {
            LoraTraining = true, TrainingGpuCheckpoints = true,
            TrainingIQ2Fp16XmxForward = fp16Forward
        };
        using Qwen35QuantizedModel recomputed = Qwen35QuantizedModel.Load(
            fixture.Path, devices, options: execution);
        using Qwen35QuantizedModel cached = Qwen35QuantizedModel.Load(
            fixture.Path, devices, options: execution with { TrainingIQ2GpuProjectionCacheMiB = 1 });
        var adapter = new Qwen35LoraOptions
        {
            Rank = 2, Alpha = 4, IncludeOutput = true, Seed = 94, LearningRate = .001f
        };
        recomputed.AttachLora(adapter);
        cached.AttachLora(adapter);
        recomputed.LoraCheckpointThresholdRows = 0;
        cached.LoraCheckpointThresholdRows = 0;
        int[] tokens = [1, 2, 3, 0, 1, 2];
        long recomputedUpload = recomputed.UploadedBytes.Sum();
        long recomputedDownload = recomputed.DownloadedBytes.Sum();
        long cachedUpload = cached.UploadedBytes.Sum();
        long cachedDownload = cached.DownloadedBytes.Sum();

        Qwen35LoraStepResult reference = recomputed.TrainLora(tokens, 3);
        Qwen35LoraStepResult actual = cached.TrainLora(tokens, 3);

        Assert.Equal(reference.Step, actual.Step);
        Close(reference.Loss, actual.Loss);
        Close(reference.GradientNorm, actual.GradientNorm);
        Assert.Equal(recomputed.UploadedBytes.Sum() - recomputedUpload,
            cached.UploadedBytes.Sum() - cachedUpload);
        Assert.Equal(recomputed.DownloadedBytes.Sum() - recomputedDownload,
            cached.DownloadedBytes.Sum() - cachedDownload);
        Assert.Equal(1, cached.LastIq2GpuProjectionCacheStats.Captured);
        Assert.Equal(1, cached.LastIq2GpuProjectionCacheStats.Reused);
        Assert.Equal((long)(tokens.Length - 1) * 512 * sizeof(float),
            cached.LastIq2GpuProjectionCacheStats.PeakBytes);
        Assert.Equal(cached.LastIq2GpuProjectionCacheStats.PeakBytes,
            cached.LastIq2GpuProjectionCachePeakBytesByDevice.Sum());
        Assert.All(cached.LastIq2GpuProjectionCachePeakBytesByDevice,
            bytes => Assert.InRange(bytes, 0, 1024 * 1024));
        foreach (var pair in recomputed.LoraMatrices)
        {
            float[][] expected = pair.Value.ReadState();
            float[][] observed = cached.LoraMatrices[pair.Key].ReadState();
            for (int field = 0; field < expected.Length; field++)
                for (int i = 0; i < expected[field].Length; i++) Close(expected[field][i], observed[field][i]);
        }
    }

    private static void Close(double expected, double actual)
        => Assert.True(double.IsFinite(actual)
            && Math.Abs(actual - expected) <= 2e-5 * (1 + Math.Abs(expected)),
            $"Expected {expected:R}, actual {actual:R}.");
}
