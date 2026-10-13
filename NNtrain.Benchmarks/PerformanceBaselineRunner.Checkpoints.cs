using System.Diagnostics;
using System.Security.Cryptography;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Execution;
using NNtrain.Training.Metrics;
using NNtrain.Training.Optimization;
using NNtrain.Training.Persistence;

namespace NNtrain.Benchmarks;

internal static partial class PerformanceBaselineRunner
{
    private static string ComputeFileSha256(string path)
    {
        using var stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 4 * 1024 * 1024,
            FileOptions.SequentialScan);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static (bool Model, bool Optimizer)
        ValidateReloadedCheckpointPayloads(
            string checkpointPath,
            WikiLanguageModelCommand.WikiModelCheckpoint manifest,
            Module model,
            IOptimizer optimizer,
            string artifactDirectory)
    {
        string currentArtifact =
            WikiLanguageModelCommand.GetCurrentModelArtifactPath(
                checkpointPath, manifest.ArtifactSlot);
        string verificationModel = Path.Combine(
            artifactDirectory, "reload-model-verification.safetensors");
        var verificationOptimizerPaths = new List<string>();
        try
        {
            SafeTensorFile.SaveModel(
                model,
                verificationModel,
                artifactDTypeOverride: TensorDType.Float32);
            bool modelValid = new FileInfo(currentArtifact).Length
                    == new FileInfo(verificationModel).Length
                && string.Equals(
                    ComputeFileSha256(currentArtifact),
                    ComputeFileSha256(verificationModel),
                    StringComparison.Ordinal);

            IReadOnlyList<IOptimizer> leaves =
                OptimizerBundle.GetCheckpointLeafOptimizers(optimizer);
            bool optimizerValid =
                leaves.Count == manifest.OptimizerStateTypes?.Length;
            for (int index = 0; index < leaves.Count; index++)
            {
                string source = WikiLanguageModelCommand
                    .GetOptimizerBinaryArtifactPath(
                        checkpointPath, manifest.ArtifactSlot, index);
                string verification = Path.Combine(
                    artifactDirectory,
                    $"reload-optimizer-{index}-verification.bin");
                verificationOptimizerPaths.Add(verification);
                using (var stream = new FileStream(
                    verification,
                    FileMode.Create,
                    FileAccess.Write,
                    FileShare.None,
                    bufferSize: 4 * 1024 * 1024,
                    FileOptions.SequentialScan))
                {
                    OptimizerStateStream.SaveStateBinary(
                        leaves[index], stream);
                    stream.Flush(flushToDisk: true);
                }
                optimizerValid &= new FileInfo(source).Length
                        == new FileInfo(verification).Length
                    && string.Equals(
                        ComputeFileSha256(source),
                        ComputeFileSha256(verification),
                        StringComparison.Ordinal);
            }
            return (modelValid, optimizerValid);
        }
        finally
        {
            if (File.Exists(verificationModel))
                File.Delete(verificationModel);
            foreach (string path in verificationOptimizerPaths)
            {
                if (File.Exists(path))
                    File.Delete(path);
            }
        }
    }

    private static WikiTrainingConfiguration
        CreateSoakCheckpointConfiguration(
            BaselineModelConfiguration configuration,
            TensorPrecisionMode precision,
            string checkpointPath,
            IReadOnlyList<int> deviceIndices,
            bool resume)
        => new()
        {
            CheckpointPath = checkpointPath,
            ResumeFromCheckpoint = resume,
            AutoResume = false,
            VocabularySize = configuration.Vocabulary,
            Epochs = 1,
            BatchSize = configuration.Batch,
            ContextLength = configuration.Sequence,
            ModelWidth = configuration.Width,
            Heads = configuration.Heads,
            HiddenSize = configuration.Hidden,
            Layers = configuration.Layers,
            ModelArchitecture =
                WikiTrainingConfiguration.TransformerArchitecture,
            Precision = TensorPrecisionModeNames.Format(precision),
            Bfp8BlockSize = configuration.Bfp8BlockSize,
            TieWordEmbeddings = configuration.TieWordEmbeddings,
            Device = WikiTrainingConfiguration.CudaDevice,
            DeviceIndex = deviceIndices[0],
            DeviceIndices = deviceIndices.ToArray(),
            AdaptiveCudaSharding = configuration.AdaptiveCudaSharding,
            CudaShardEmaAlpha = configuration.CudaShardEmaAlpha,
            CudaMinimumRelativeShardSize =
                configuration.CudaMinimumRelativeShardSize,
            CudaMaximumBatchAdjustmentPerStep =
                configuration.CudaMaximumBatchAdjustmentPerStep,
            CudaGraphCacheBudgetMiB =
                configuration.CudaGraphCacheBudgetMiB,
            Dropout = configuration.Dropout,
            InitializationScale = configuration.InitializationScale,
            Optimizer = WikiTrainingConfiguration.NekoMuonOptimizer,
            LearningRate = configuration.LearningRate,
            AuxiliaryLearningRate = configuration.AuxiliaryLearningRate,
            NekoMuonNewtonSchulzInterval =
                configuration.NewtonSchulzInterval,
            NekoMuonNewtonSchulzDepthMode = "fixed",
            NekoMuonNewtonSchulzDepth = 5f,
            WarmupPercent = 0f,
            WeightDecay = configuration.WeightDecay,
            Seed = configuration.Seed,
            ValidationFraction = 0f,
        };

    private static CheckpointArtifactSnapshot CaptureCheckpointArtifacts(
        string checkpointPath)
    {
        WikiLanguageModelCommand.WikiModelCheckpoint manifest =
            torch.load<WikiLanguageModelCommand.WikiModelCheckpoint>(
                checkpointPath);
        if (manifest.FormatVersion != 8
            || manifest.ArtifactSlot is < 0 or > 1
            || manifest.BestArtifactSlot is < 0 or > 1
            || manifest.OptimizerStateTypes is not { Length: > 0 })
        {
            throw new InvalidDataException(
                "Soak restart requires a committed Wiki checkpoint v8 " +
                "manifest.");
        }

        var paths = new List<string>
        {
            Path.GetFullPath(checkpointPath),
            WikiLanguageModelCommand.GetCurrentModelArtifactPath(
                checkpointPath,
                manifest.ArtifactSlot),
            WikiLanguageModelCommand.GetBestModelArtifactPath(
                checkpointPath,
                manifest.BestArtifactSlot),
        };
        for (int index = 0;
            index < manifest.OptimizerStateTypes.Length;
            index++)
        {
            paths.Add(
                WikiLanguageModelCommand.GetOptimizerBinaryArtifactPath(
                    checkpointPath,
                    manifest.ArtifactSlot,
                    index));
        }
        foreach (string path in paths)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException(
                    "A checkpoint artifact referenced by the v8 manifest " +
                    "was not published.",
                    path);
            }
        }

        BaselineCheckpointArtifact[] artifacts = paths
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Select(path => new BaselineCheckpointArtifact(
                Path.GetFileName(path),
                new FileInfo(path).Length,
                ComputeFileSha256(path)))
            .ToArray();
        string directory = Path.GetDirectoryName(
            Path.GetFullPath(checkpointPath))!;
        string stem = Path.GetFileNameWithoutExtension(checkpointPath);
        bool transactionDebris = Directory.EnumerateFiles(
                directory,
                $"{stem}*")
            .Any(path => path.Contains(
                    ".transaction.",
                    StringComparison.OrdinalIgnoreCase)
                || path.EndsWith(".tmp", StringComparison.OrdinalIgnoreCase)
                || path.EndsWith(
                    ".backup",
                    StringComparison.OrdinalIgnoreCase));
        DateTime manifestWrite = File.GetLastWriteTimeUtc(checkpointPath);
        bool manifestPublishedLast = paths.Skip(1).All(path =>
            File.GetLastWriteTimeUtc(path) <= manifestWrite);
        return new CheckpointArtifactSnapshot(
            manifest,
            artifacts,
            artifacts.Sum(artifact => artifact.Bytes),
            !transactionDebris && manifestPublishedLast);
    }

    private static IReadOnlyList<BaselineCheckpointArtifact>
        CapturePartialCheckpointArtifacts(string checkpointPath)
    {
        string fullPath = Path.GetFullPath(checkpointPath);
        string directory = Path.GetDirectoryName(fullPath)!;
        if (!Directory.Exists(directory))
            return [];
        string stem = Path.GetFileNameWithoutExtension(fullPath);
        return Directory.EnumerateFiles(directory, $"{stem}*")
            .Order(StringComparer.OrdinalIgnoreCase)
            .Select(path => new BaselineCheckpointArtifact(
                Path.GetFileName(path),
                new FileInfo(path).Length,
                ComputeFileSha256(path)))
            .ToArray();
    }

    private static bool TrainingRandomStatesEqual(
        TrainingRandomState? expected,
        TrainingRandomState? actual)
        => expected is null || actual is null
            ? expected is null && actual is null
            : expected.FormatVersion == actual.FormatVersion
                && expected.RootSeed == actual.RootSeed
                && expected.HostState == actual.HostState
                && expected.DeviceStates.Length
                    == actual.DeviceStates.Length
                && expected.DeviceStates.Zip(
                    actual.DeviceStates,
                    (left, right) => left.DeviceIndex == right.DeviceIndex
                        && left.State == right.State)
                    .All(equal => equal);

    private static bool AdaptiveShardStatesEqual(
        CudaAdaptiveShardState expected,
        CudaAdaptiveShardState? actual)
        => actual is not null
            && expected.FormatVersion == actual.FormatVersion
            && expected.Devices.AsSpan().SequenceEqual(actual.Devices)
            && expected.LastAllocation.AsSpan()
                .SequenceEqual(actual.LastAllocation)
            && expected.ThroughputEma.AsSpan()
                .SequenceEqual(actual.ThroughputEma)
            && expected.HasObservation == actual.HasObservation
            && expected.ObservationCount == actual.ObservationCount
            && expected.LastAdjustmentObservation
                == actual.LastAdjustmentObservation
            && expected.PendingAllocation.AsSpan()
                .SequenceEqual(actual.PendingAllocation)
            && expected.PendingConfirmationCount
                == actual.PendingConfirmationCount
            && expected.LastCandidateObservation
                == actual.LastCandidateObservation
            && expected.OversizedGraphAllocation.AsSpan()
                .SequenceEqual(actual.OversizedGraphAllocation)
            && expected.OversizedGraphPinnedBytes
                == actual.OversizedGraphPinnedBytes;

    private static bool ValidateBfp8BlockSize(
        IReadOnlyList<Parameter> parameters,
        TensorPrecisionMode precision,
        int expectedBlockSize)
        => precision != TensorPrecisionMode.Mix8_32
            || parameters.All(parameter =>
                parameter.T.Bfp8Quantization is
                {
                    Granularity: Bfp8ScaleGranularity.Block,
                    BlockSize: var blockSize,
                }
                && blockSize == expectedBlockSize);

    private static bool ValidateCudaResidency(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> deviceIndices)
        => parameters.Count > 0
            && parameters.All(parameter =>
            {
                int[] residents = parameter.T.GetResidentCudaDeviceIndices();
                return deviceIndices.All(device =>
                    residents.Contains(device));
            });

    private static BaselineCheckpointResult EmptyCheckpointResult(
        string artifactDirectory)
        => new(
            FormatVersion: 0,
            Validated: false,
            SaveMilliseconds: null,
            LoadMilliseconds: null,
            TotalBytes: 0,
            Artifacts: [],
            ArtifactFirstManifestLastValidated: false,
            CursorValidated: false,
            TrainingRandomValidated: false,
            SchedulerValidated: false,
            AdaptiveShardStateValidated: false,
            ModelValidated: false,
            OptimizerValidated: false,
            PrecisionValidated: false,
            Bfp8BlockSizeValidated: false,
            DeviceResidencyValidated: false,
            OldFixtureDisposedBeforeReload: false,
            ArtifactsRetainedAfterFailure: false,
            ArtifactDirectory: artifactDirectory);

    private static void AddRuntimeErrors(
        Exception exception,
        ICollection<string> errors)
    {
        if (exception is AggregateException aggregate)
        {
            foreach (Exception inner in aggregate.Flatten().InnerExceptions)
                AddRuntimeErrors(inner, errors);
            return;
        }
        string status = exception is NativeCudaException cuda
            ? $" CUDA-status={cuda.Status}"
            : string.Empty;
        errors.Add($"{exception.GetType().Name}:{status} {exception.Message}");
        if (exception.InnerException is not null)
            AddRuntimeErrors(exception.InnerException, errors);
    }

    private static IReadOnlyList<BaselineDeviceMemorySummary>
        SummarizePostWarmupDeviceMemory(
            IReadOnlyList<BaselineStepMeasurement> measurements,
            int warmupSteps)
    {
        if (measurements.Count == 0)
            return [];
        int baselineIndex = Math.Clamp(
            warmupSteps - 1,
            0,
            measurements.Count - 1);
        var result = new List<BaselineDeviceMemorySummary>();
        foreach (BaselineDeviceMemoryObservation baseline in
            measurements[baselineIndex].DeviceMemory)
        {
            if (!baseline.TotalBytes.HasValue
                || !baseline.UsedBytes.HasValue)
            {
                continue;
            }
            BaselineDeviceMemoryObservation[] values = measurements
                .Skip(baselineIndex)
                .SelectMany(value => value.DeviceMemory)
                .Where(value => value.Device == baseline.Device
                    && value.UsedBytes.HasValue)
                .ToArray();
            if (values.Length == 0)
                continue;
            long start = baseline.UsedBytes.Value;
            long peak = values.Max(value => value.UsedBytes!.Value);
            result.Add(new BaselineDeviceMemorySummary(
                baseline.Device,
                baseline.TotalBytes.Value,
                start,
                peak,
                values[^1].UsedBytes!.Value,
                Math.Max(0, peak - start),
                values.Length));
        }
        return result.OrderBy(value => value.Device).ToArray();
    }

    private static IReadOnlyList<BaselineGateResult> CreateSoakGates(
        BaselineSoakConfiguration configuration,
        BaselineSoakResult result,
        int selectedGpuCount)
    {
        bool enoughTrendSamples = result.FirstWindow.Count
                == configuration.TrendWindowSteps
            && result.LastWindow.Count == configuration.TrendWindowSteps;
        bool vramAvailable = result.PostWarmupDeviceMemory.Count
            == selectedGpuCount;
        bool vramPassed = vramAvailable
            && result.PostWarmupDeviceMemory.All(value =>
                value.PeakGrowthBytes
                    <= configuration.MaximumPostWarmupVramGrowthBytes);
        return
        [
            new BaselineGateResult(
                "committed-step-count",
                result.CompletedCommittedSteps
                    == configuration.TotalCommittedSteps,
                result.CompletedCommittedSteps.ToString(),
                configuration.TotalCommittedSteps.ToString()),
            new BaselineGateResult(
                "step-time-trend",
                enoughTrendSamples
                    && result.LastToFirstP50Ratio
                        <= configuration.MaximumLastToFirstP50Ratio,
                enoughTrendSamples
                    ? $"ratio={result.LastToFirstP50Ratio:F4}, first=" +
                        $"{result.FirstWindow.P50:F3}ms, last=" +
                        $"{result.LastWindow.P50:F3}ms"
                    : "insufficient samples",
                $"last p50 / first p50 <= " +
                    $"{configuration.MaximumLastToFirstP50Ratio:F4}"),
            new BaselineGateResult(
                "post-warmup-vram-growth",
                vramPassed,
                vramAvailable
                    ? string.Join(
                        ", ",
                        result.PostWarmupDeviceMemory.Select(value =>
                            $"cuda:{value.Device} +" +
                            $"{value.PeakGrowthBytes / 1048576d:F1} MiB"))
                    : "cudaMemGetInfo unavailable",
                $"each GPU <= " +
                    $"{configuration.MaximumPostWarmupVramGrowthBytes
                        / 1048576d:F1} MiB"),
            new BaselineGateResult(
                "generation-event",
                result.GenerationObserved,
                result.GenerationObserved
                    ? $"{result.GeneratedTokens} token(s) at global step " +
                        $"{result.GenerationStep} in " +
                        $"{result.GenerationMilliseconds:F3} ms"
                    : "not observed",
                $"post-commit generation at global step " +
                    $"{configuration.GenerationStep}"),
            new BaselineGateResult(
                "wiki-v8-streaming-checkpoint",
                result.RestartObserved
                    && result.ResumeArtifactValidated
                    && result.Checkpoint.Validated,
                $"restart={result.RestartObserved}, " +
                    $"validated={result.Checkpoint.Validated}, " +
                    $"format=v{result.Checkpoint.FormatVersion}, " +
                    $"artifacts={result.Checkpoint.Artifacts.Count}, " +
                    $"bytes={result.Checkpoint.TotalBytes}, " +
                    $"save={result.Checkpoint.SaveMilliseconds:F3}ms, " +
                    $"load={result.Checkpoint.LoadMilliseconds:F3}ms",
                $"validated full Wiki v8 restart after global step " +
                    $"{configuration.RestartStep}",
                "Artifact-first/manifest-last, cursor, training RNG, " +
                "zero-warmup cosine scheduler, adaptive shards, model, " +
                "optimizer, precision/BFP8 metadata, and two-GPU residency " +
                "must all round-trip after the old fixture is disposed. " +
                string.Join(
                    ", ",
                    result.Checkpoint.Artifacts.Select(artifact =>
                        $"{artifact.Name}:{artifact.Bytes}:" +
                        artifact.Sha256))),
            new BaselineGateResult(
                "metrics-sidecar-continuity",
                result.SidecarContinuityValidated,
                $"before={result.SidecarEntriesBeforeRestart}, " +
                    $"final={result.SidecarEntriesAfterResume}",
                "strictly continuous JSONL entries through restart"),
            new BaselineGateResult(
                "html-continuity",
                result.HtmlContinuityChecked
                    ? result.HtmlContinuityValidated
                    : null,
                result.HtmlContinuityStatus,
                "production HTML projection retains pre/post-resume metrics"),
            new BaselineGateResult(
                "nonzero-gpu-shards",
                !result.ZeroShardObserved,
                result.ZeroShardObserved
                    ? "a zero/missing shard was observed"
                    : "all selected GPUs received positive shards",
                "no zero shard"),
            new BaselineGateResult(
                "runtime-errors",
                result.RuntimeErrors.Count == 0,
                result.RuntimeErrors.Count == 0
                    ? "none"
                    : string.Join(" | ", result.RuntimeErrors),
                "no OOM, CUDA 600/700, illegal access, or other runtime error"),
        ];
    }

    private sealed record CheckpointArtifactSnapshot(
        WikiLanguageModelCommand.WikiModelCheckpoint Manifest,
        IReadOnlyList<BaselineCheckpointArtifact> Artifacts,
        long TotalBytes,
        bool ArtifactFirstManifestLastValidated);

    private sealed class SoakCheckpointFaultInjector
        : ICheckpointFaultInjector
    {
        public void OnCheckpointFaultPoint(CheckpointFaultContext context)
        {
            if (context.Point == CheckpointFaultPoint.AfterArtifactPublish
                && context.ArtifactIndex == 0)
            {
                throw new IOException(
                    "Intentional soak checkpoint failure after publishing " +
                    "the first artifact.");
            }
        }
    }

    private static bool ShouldWriteSoakHtmlMetric(
        int committedIndex,
        long globalStep,
        BaselineSoakConfiguration soak)
        => committedIndex == 0
            || globalStep == soak.RestartStep
            || globalStep == soak.RestartStep + 1L
            || committedIndex + 1 == soak.TotalCommittedSteps
            || (committedIndex + 1) % 100 == 0;
}
