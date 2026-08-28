namespace NNtrain;

/// <summary>
/// Immutable managed CUDA dispatch switches. Legacy environment variables are
/// sampled once when the startup policy is first requested; hot kernels only
/// read this object. Tests and benchmarks can install an async-flow-local
/// override without mutating process-global environment state.
/// </summary>
internal sealed record CudaDispatchPolicy
{
    private static readonly Lazy<CudaDispatchPolicy> StartupPolicy = new(
        LoadStartupPolicy,
        LazyThreadSafetyMode.ExecutionAndPublication);

    internal bool DisableCublasLt { get; init; }
    internal bool DisableCublasLtBackward { get; init; }
    internal bool DisableNativeFlashAttention { get; init; }
    internal bool DisableTensorCoreFlashAttention { get; init; }
    internal bool DisableAsyncFlashAttention { get; init; }
    internal bool DisableParallelAttentionDkv { get; init; }
    internal bool DisableAsyncAttentionBackward { get; init; }
    internal bool DisableTensorCoreForgetMemory { get; init; }
    internal bool DisableTensorCoreNekoMuon { get; init; }
    internal bool DisableBatchedNekoMuon { get; init; }
    internal int NekoMuonBatchSize { get; init; } = 8;
    internal long NekoMuonScratchBudgetBytes { get; init; } =
        32L * 1024L * 1024L;
    internal int? GradientBucketElements { get; init; }
    internal int? GradientHostChunkElements { get; init; }
    internal bool DisableGradientHostPipeline { get; init; }
    internal bool DisableAsyncGradientHostPipeline { get; init; }
    internal bool DisableExternalGradientReadyEvents { get; init; }
    internal bool DisableDirectAttentionBFloat16Gradient { get; init; }
    internal bool DisableDirectLayerNormBFloat16BranchGradient { get; init; }
    internal bool EnableLayerNormOneScan512 { get; init; }
    internal bool DisableDirectLinearBFloat16Gradient { get; init; }
    internal bool DisableKvCache { get; init; }
    internal bool SynchronizeDataParallelPhases { get; init; }
    internal bool DisableBFloat16GradientBuckets { get; init; }
    internal bool EnableBlockBfp8OptimizerState { get; init; }

    internal static CudaDispatchPolicy Defaults { get; } = new();

    internal static CudaDispatchPolicy Startup => StartupPolicy.Value;

    internal static CudaDispatchPolicy Current
        => TensorExecutionContext.ActiveCudaDispatchPolicy ?? Startup;

    internal static IDisposable Push(CudaDispatchPolicy policy)
    {
        ArgumentNullException.ThrowIfNull(policy);
        return TensorExecutionContext.PushCudaDispatchPolicy(policy.Validate());
    }

    internal CudaDispatchPolicy Validate()
    {
        if (NekoMuonBatchSize is < 1 or > 32)
        {
            throw new ArgumentOutOfRangeException(
                nameof(NekoMuonBatchSize),
                NekoMuonBatchSize,
                "NekoMuon batch size must be between 1 and 32.");
        }
        if (NekoMuonScratchBudgetBytes < 1024L * 1024L
            || NekoMuonScratchBudgetBytes > 1024L * 1024L * 1024L)
        {
            throw new ArgumentOutOfRangeException(
                nameof(NekoMuonScratchBudgetBytes),
                NekoMuonScratchBudgetBytes,
                "NekoMuon scratch budget must be between 1 MiB and 1 GiB.");
        }
        if (GradientBucketElements is <= 0)
            throw new ArgumentOutOfRangeException(nameof(GradientBucketElements));
        if (GradientHostChunkElements is <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(GradientHostChunkElements));
        }
        return this;
    }

    private static CudaDispatchPolicy LoadStartupPolicy()
        => FromEnvironment(ReadEnvironment);

    internal static CudaDispatchPolicy FromEnvironment(
        Func<string, string?> readEnvironment)
    {
        ArgumentNullException.ThrowIfNull(readEnvironment);
        int batchSize = ReadPositiveInt(
            readEnvironment,
            "NNTRAIN_NEKOMUON_BATCH_SIZE",
            defaultValue: 8,
            minimum: 1,
            maximum: 32);
        int scratchMebibytes = ReadPositiveInt(
            readEnvironment,
            "NNTRAIN_NEKOMUON_SCRATCH_MIB",
            defaultValue: 32,
            minimum: 1,
            maximum: 1024);
        return new CudaDispatchPolicy
        {
            DisableCublasLt = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_CUBLASLT"),
            DisableCublasLtBackward = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_CUBLASLT_BACKWARD"),
            DisableNativeFlashAttention = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_NATIVE_FLASH_ATTENTION"),
            DisableTensorCoreFlashAttention = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_TENSOR_CORE_FLASH_ATTENTION"),
            DisableAsyncFlashAttention = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_ASYNC_FLASH_ATTENTION"),
            DisableParallelAttentionDkv = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_PARALLEL_ATTENTION_DKV"),
            DisableAsyncAttentionBackward = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_ASYNC_ATTENTION_BACKWARD"),
            DisableTensorCoreForgetMemory = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_TENSOR_CORE_FORGET_MEMORY"),
            DisableTensorCoreNekoMuon = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_TENSOR_CORE_NEKOMUON"),
            DisableBatchedNekoMuon = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_BATCHED_NEKOMUON"),
            NekoMuonBatchSize = batchSize,
            NekoMuonScratchBudgetBytes = checked(
                (long)scratchMebibytes * 1024L * 1024L),
            GradientBucketElements = ReadOptionalPositiveInt(
                readEnvironment,
                "NNTRAIN_GRADIENT_BUCKET_ELEMENTS"),
            GradientHostChunkElements = ReadOptionalPositiveInt(
                readEnvironment,
                "NNTRAIN_GRADIENT_HOST_CHUNK_ELEMENTS"),
            DisableGradientHostPipeline = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_GRADIENT_HOST_PIPELINE"),
            DisableAsyncGradientHostPipeline = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_ASYNC_GRADIENT_HOST_PIPELINE"),
            DisableExternalGradientReadyEvents = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_EXTERNAL_GRADIENT_READY_EVENTS"),
            DisableDirectAttentionBFloat16Gradient = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_DIRECT_ATTENTION_BF16_GRADIENT"),
            DisableDirectLayerNormBFloat16BranchGradient = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_DIRECT_LAYERNORM_BF16_BRANCH_GRADIENT"),
            EnableLayerNormOneScan512 = ReadFlag(
                readEnvironment,
                "NNTRAIN_ENABLE_LAYERNORM_ONE_SCAN_512"),
            DisableDirectLinearBFloat16Gradient = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_DIRECT_LINEAR_BF16_GRADIENT"),
            DisableKvCache = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_KV_CACHE"),
            SynchronizeDataParallelPhases = ReadFlag(
                readEnvironment,
                "NNTRAIN_CUDA_SYNC_PHASES"),
            DisableBFloat16GradientBuckets = ReadFlag(
                readEnvironment,
                "NNTRAIN_DISABLE_BF16_GRADIENT_BUCKETS"),
            EnableBlockBfp8OptimizerState = ReadFlag(
                readEnvironment,
                "NNTRAIN_ENABLE_BLOCK_BFP8_OPTIMIZER_STATE"),
        };
    }

    private static bool ReadFlag(
        Func<string, string?> readEnvironment,
        string name)
        => string.Equals(
            readEnvironment(name),
            "1",
            StringComparison.Ordinal);

    private static int ReadPositiveInt(
        Func<string, string?> readEnvironment,
        string name,
        int defaultValue,
        int minimum,
        int maximum)
    {
        string? configured = readEnvironment(name);
        return int.TryParse(configured, out int value)
            ? Math.Clamp(value, minimum, maximum)
            : defaultValue;
    }

    private static int? ReadOptionalPositiveInt(
        Func<string, string?> readEnvironment,
        string name)
    {
        string? configured = readEnvironment(name);
        return int.TryParse(configured, out int value) && value > 0
            ? value
            : null;
    }

    private static string? ReadEnvironment(string name)
    {
        CudaDispatchEnvironmentTelemetry.RecordRead();
        return Environment.GetEnvironmentVariable(name);
    }
}

internal static class CudaDispatchEnvironmentTelemetry
{
    private static long _environmentReads;

    internal static CudaDispatchEnvironmentTelemetrySnapshot Snapshot
        => new(Interlocked.Read(ref _environmentReads));

    internal static void RecordRead()
        => Interlocked.Increment(ref _environmentReads);
}

internal readonly record struct CudaDispatchEnvironmentTelemetrySnapshot(
    long EnvironmentReads)
{
    public static CudaDispatchEnvironmentTelemetrySnapshot operator -(
        CudaDispatchEnvironmentTelemetrySnapshot left,
        CudaDispatchEnvironmentTelemetrySnapshot right)
        => new(left.EnvironmentReads - right.EnvironmentReads);
}
