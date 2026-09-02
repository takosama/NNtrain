using System.Diagnostics;
using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

[Collection(TensorSimdCollection.Name)]
public sealed class CudaDispatchPolicyTests
{
    private static bool _benchmarkSink;

    [Fact]
    public void LegacyStartupSwitchesParseWithoutProcessGlobalMutation()
    {
        var values = new Dictionary<string, string?>
        {
            ["NNTRAIN_DISABLE_CUBLASLT"] = "1",
            ["NNTRAIN_DISABLE_CUBLASLT_BACKWARD"] = "1",
            ["NNTRAIN_DISABLE_NATIVE_FLASH_ATTENTION"] = "1",
            ["NNTRAIN_DISABLE_TENSOR_CORE_FLASH_ATTENTION"] = "1",
            ["NNTRAIN_DISABLE_ASYNC_FLASH_ATTENTION"] = "1",
            ["NNTRAIN_DISABLE_PARALLEL_ATTENTION_DKV"] = "1",
            ["NNTRAIN_DISABLE_ASYNC_ATTENTION_BACKWARD"] = "1",
            ["NNTRAIN_DISABLE_TENSOR_CORE_FORGET_MEMORY"] = "1",
            ["NNTRAIN_DISABLE_TENSOR_CORE_NEKOMUON"] = "1",
            ["NNTRAIN_DISABLE_BATCHED_NEKOMUON"] = "1",
            ["NNTRAIN_NEKOMUON_BATCH_SIZE"] = "12",
            ["NNTRAIN_NEKOMUON_SCRATCH_MIB"] = "48",
            ["NNTRAIN_GRADIENT_BUCKET_ELEMENTS"] = "12345",
            ["NNTRAIN_GRADIENT_HOST_CHUNK_ELEMENTS"] = "6789",
            ["NNTRAIN_DISABLE_GRADIENT_HOST_PIPELINE"] = "1",
            ["NNTRAIN_DISABLE_ASYNC_GRADIENT_HOST_PIPELINE"] = "1",
            ["NNTRAIN_DISABLE_EXTERNAL_GRADIENT_READY_EVENTS"] = "1",
            ["NNTRAIN_DISABLE_DIRECT_ATTENTION_BF16_GRADIENT"] = "1",
            ["NNTRAIN_DISABLE_DIRECT_LAYERNORM_BF16_BRANCH_GRADIENT"] = "1",
            ["NNTRAIN_DISABLE_DIRECT_LINEAR_BF16_GRADIENT"] = "1",
            ["NNTRAIN_DISABLE_DIRECT_BFP8_FFN_GRADIENT"] = "1",
            ["NNTRAIN_DISABLE_DIRECT_BFP8_LAYERNORM_BLOCK32X512"] = "1",
            ["NNTRAIN_DISABLE_DIRECT_BFP8_ATTENTION_OUTPUT"] = "1",
            ["NNTRAIN_DISABLE_KV_CACHE"] = "1",
            ["NNTRAIN_DISABLE_CUDA_GRAPHS"] = "1",
            ["NNTRAIN_CUDA_SYNC_PHASES"] = "1",
            ["NNTRAIN_DISABLE_BF16_GRADIENT_BUCKETS"] = "1",
            ["NNTRAIN_ENABLE_BLOCK_BFP8_OPTIMIZER_STATE"] = "1",
        };

        CudaDispatchPolicy policy = CudaDispatchPolicy.FromEnvironment(
            name => values.GetValueOrDefault(name));

        Assert.True(policy.DisableCublasLt);
        Assert.True(policy.DisableCublasLtBackward);
        Assert.True(policy.DisableNativeFlashAttention);
        Assert.True(policy.DisableTensorCoreFlashAttention);
        Assert.True(policy.DisableAsyncFlashAttention);
        Assert.True(policy.DisableParallelAttentionDkv);
        Assert.True(policy.DisableAsyncAttentionBackward);
        Assert.True(policy.DisableTensorCoreForgetMemory);
        Assert.True(policy.DisableTensorCoreNekoMuon);
        Assert.True(policy.DisableBatchedNekoMuon);
        Assert.Equal(12, policy.NekoMuonBatchSize);
        Assert.Equal(48L * 1024L * 1024L,
            policy.NekoMuonScratchBudgetBytes);
        Assert.Equal(12345, policy.GradientBucketElements);
        Assert.Equal(6789, policy.GradientHostChunkElements);
        Assert.True(policy.DisableGradientHostPipeline);
        Assert.True(policy.DisableAsyncGradientHostPipeline);
        Assert.True(policy.DisableExternalGradientReadyEvents);
        Assert.True(policy.DisableDirectAttentionBFloat16Gradient);
        Assert.True(policy.DisableDirectLayerNormBFloat16BranchGradient);
        Assert.True(policy.DisableDirectLinearBFloat16Gradient);
        Assert.True(policy.DisableDirectBfp8FfnGradient);
        Assert.True(policy.DisableDirectBfp8LayerNormBlock32x512);
        Assert.True(policy.DisableDirectBfp8AttentionOutput);
        Assert.True(policy.DisableKvCache);
        Assert.True(policy.DisableCudaGraphs);
        Assert.True(policy.SynchronizeDataParallelPhases);
        Assert.True(policy.DisableBFloat16GradientBuckets);
        Assert.True(policy.EnableBlockBfp8OptimizerState);
    }

    [Fact]
    public async Task ExplicitOverridesAreNestedAndAsyncFlowLocal()
    {
        CudaDispatchPolicy original = CudaDispatchPolicy.Current;
        CudaDispatchPolicy first = CudaDispatchPolicy.Defaults with
        {
            DisableCublasLt = true,
            DisableKvCache = false,
        };
        CudaDispatchPolicy second = CudaDispatchPolicy.Defaults with
        {
            DisableCublasLt = false,
            DisableKvCache = true,
        };

        using (CudaDispatchPolicy.Push(first))
        {
            Assert.Same(first, CudaDispatchPolicy.Current);
            using (CudaDispatchPolicy.Push(second))
                Assert.Same(second, CudaDispatchPolicy.Current);
            Assert.Same(first, CudaDispatchPolicy.Current);
        }
        Assert.Same(original, CudaDispatchPolicy.Current);

        async Task<(bool DisableBlas, bool DisableCache)> ReadAsync(
            CudaDispatchPolicy policy)
        {
            using IDisposable scope = CudaDispatchPolicy.Push(policy);
            await Task.Yield();
            return (
                CudaDispatchPolicy.Current.DisableCublasLt,
                CudaDispatchPolicy.Current.DisableKvCache);
        }

        (bool DisableBlas, bool DisableCache)[] results = await Task.WhenAll(
            Task.Run(() => ReadAsync(first)),
            Task.Run(() => ReadAsync(second)));
        Assert.Equal((true, false), results[0]);
        Assert.Equal((false, true), results[1]);
        Assert.Same(original, CudaDispatchPolicy.Current);
    }

    [Fact]
    public void StableDispatchReadsImmutablePolicyAndNeverTheEnvironment()
    {
        _ = CudaDispatchPolicy.Startup;
        using IDisposable scope = CudaDispatchPolicy.Push(
            CudaDispatchPolicy.Defaults with
            {
                DisableTensorCoreForgetMemory = true,
                DisableTensorCoreNekoMuon = true,
            });
        CudaDispatchEnvironmentTelemetrySnapshot before =
            CudaDispatchEnvironmentTelemetry.Snapshot;

        bool value = false;
        for (int iteration = 0; iteration < 100_000; iteration++)
        {
            CudaDispatchPolicy policy = CudaDispatchPolicy.Current;
            value ^= policy.DisableTensorCoreForgetMemory;
            value ^= policy.DisableTensorCoreNekoMuon;
            value ^= policy.DisableCublasLt;
        }
        _benchmarkSink = value;

        CudaDispatchEnvironmentTelemetrySnapshot delta =
            CudaDispatchEnvironmentTelemetry.Snapshot - before;
        Assert.Equal(0, delta.EnvironmentReads);
    }

    [Fact]
    public void ImmutableDispatchLookupIsFasterThanLegacyEnvironmentLookup()
    {
        const int iterations = 200_000;
        _ = CudaDispatchPolicy.Startup;
        using IDisposable scope = CudaDispatchPolicy.Push(
            CudaDispatchPolicy.Defaults);

        _ = Measure(iterations / 10, ReadLegacySwitch);
        _ = Measure(iterations / 10, ReadImmutableSwitch);
        long legacyTicks = Enumerable.Range(0, 3)
            .Select(_ => Measure(iterations, ReadLegacySwitch))
            .Min();
        long immutableTicks = Enumerable.Range(0, 3)
            .Select(_ => Measure(iterations, ReadImmutableSwitch))
            .Min();

        string result =
            $"dispatch lookup: environment=" +
            $"{TicksPerOperation(legacyTicks, iterations):F1} ns/op, " +
            $"immutable={TicksPerOperation(immutableTicks, iterations):F1} " +
            $"ns/op";
        Console.WriteLine(result);
        TestContext.Current.TestOutputHelper?.WriteLine(result);
        Assert.True(
            immutableTicks <= legacyTicks,
            $"Immutable dispatch lookup regressed: {immutableTicks} ticks " +
            $"> environment lookup {legacyTicks} ticks.");
    }

    private static long Measure(int iterations, Func<bool> read)
    {
        bool value = false;
        long start = Stopwatch.GetTimestamp();
        for (int iteration = 0; iteration < iterations; iteration++)
            value ^= read();
        long elapsed = Stopwatch.GetTimestamp() - start;
        _benchmarkSink = value;
        return elapsed;
    }

    private static bool ReadLegacySwitch()
        => string.Equals(
            Environment.GetEnvironmentVariable("NNTRAIN_DISABLE_CUBLASLT"),
            "1",
            StringComparison.Ordinal);

    private static bool ReadImmutableSwitch()
        => CudaDispatchPolicy.Current.DisableCublasLt;

    private static double TicksPerOperation(long ticks, int iterations)
        => ticks * (1_000_000_000d / Stopwatch.Frequency) / iterations;
}
