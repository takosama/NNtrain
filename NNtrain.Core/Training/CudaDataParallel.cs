using System.Runtime.CompilerServices;

namespace NNtrain;

/// <summary>
/// Backward-compatible static facade for CUDA data parallelism. New training
/// sessions should own a <see cref="CudaDataParallelEngine"/> explicitly.
/// </summary>
public static class CudaDataParallel
{
    private static readonly ConditionalWeakTable<LanguageModel, CompatibilitySlot>
        CompatibilityEngines = new();
    private static CompatibilityConfiguration _configuration = new(
        Version: 0,
        Options: new CudaAdaptiveShardingOptions());

    /// <summary>
    /// Configures EMA-based CUDA batch balancing for compatibility engines.
    /// Explicit engines receive their options in their constructor.
    /// </summary>
    public static void ConfigureAdaptiveSharding(
        CudaAdaptiveShardingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        CompatibilityConfiguration previous = Volatile.Read(ref _configuration);
        Volatile.Write(
            ref _configuration,
            new CompatibilityConfiguration(
                checked(previous.Version + 1),
                options));
    }

    /// <summary>Returns the last per-device batch allocation for a model.</summary>
    public static IReadOnlyList<int> GetLastShardBatchSizes(LanguageModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (!CompatibilityEngines.TryGetValue(model, out CompatibilitySlot? slot))
            return [];
        CompatibilityConfiguration configuration =
            Volatile.Read(ref _configuration);
        return slot.GetLastShardBatchSizes(
            configuration,
            Tensor.CudaDeviceIndices);
    }

    /// <summary>
    /// Deterministically releases the model-specific compatibility engine.
    /// Explicitly owned engines do not need this method.
    /// </summary>
    public static void Release(LanguageModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (CompatibilityEngines.TryGetValue(model, out CompatibilitySlot? slot)
            && CompatibilityEngines.Remove(model))
        {
            slot.Dispose();
        }
    }

    public static float ForwardBackward(
        LanguageModel model,
        int[] input,
        int[] target,
        int batchSize,
        int sequenceLength,
        int ignoreIndex = Tensor.DefaultCrossEntropyIgnoreIndex)
        => GetCompatibilityEngine(model).ForwardBackward(
            input,
            target,
            batchSize,
            sequenceLength,
            ignoreIndex);

    internal static CudaDataParallelProfile ForwardBackwardProfiled(
        LanguageModel model,
        int[] input,
        int[] target,
        int batchSize,
        int sequenceLength,
        int ignoreIndex = Tensor.DefaultCrossEntropyIgnoreIndex)
        => GetCompatibilityEngine(model).ForwardBackwardProfiled(
            input,
            target,
            batchSize,
            sequenceLength,
            ignoreIndex);

    private static CudaDataParallelEngine GetCompatibilityEngine(
        LanguageModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        CompatibilityConfiguration configuration =
            Volatile.Read(ref _configuration);
        int[] devices = Tensor.CudaDeviceIndices.ToArray();
        return CompatibilityEngines
            .GetValue(model, static _ => new CompatibilitySlot())
            .Get(model, configuration, devices);
    }

    private sealed record CompatibilityConfiguration(
        int Version,
        CudaAdaptiveShardingOptions Options);

    private sealed class CompatibilitySlot : IDisposable
    {
        private readonly object _sync = new();
        private CudaDataParallelEngine? _engine;
        private int _configurationVersion = -1;

        internal CudaDataParallelEngine Get(
            LanguageModel model,
            CompatibilityConfiguration configuration,
            IReadOnlyList<int> cudaDeviceIndices)
        {
            lock (_sync)
            {
                if (_engine is null
                    || _configurationVersion != configuration.Version
                    || !_engine.UsesCudaDevices(cudaDeviceIndices))
                {
                    _engine?.Dispose();
                    _engine = new CudaDataParallelEngine(
                        model,
                        cudaDeviceIndices,
                        configuration.Options);
                    _configurationVersion = configuration.Version;
                }
                return _engine;
            }
        }

        internal IReadOnlyList<int> GetLastShardBatchSizes(
            CompatibilityConfiguration configuration,
            IReadOnlyList<int> cudaDeviceIndices)
        {
            lock (_sync)
            {
                if (_engine is null)
                    return [];
                if (_configurationVersion != configuration.Version
                    || !_engine.UsesCudaDevices(cudaDeviceIndices))
                {
                    _engine.Dispose();
                    _engine = null;
                    _configurationVersion = configuration.Version;
                    return [];
                }
                return _engine.LastShardBatchSizes;
            }
        }

        public void Dispose()
        {
            lock (_sync)
            {
                _engine?.Dispose();
                _engine = null;
            }
        }
    }
}

internal readonly record struct CudaShardProfile(
    int Device,
    int BatchSize,
    double DataPreparationMilliseconds,
    double ForwardMilliseconds,
    double LossMilliseconds,
    double BackwardMilliseconds);

internal readonly record struct CudaDataParallelProfile(
    float Loss,
    double GradientPreparationMilliseconds,
    double AllReduceMilliseconds,
    double TotalMilliseconds,
    IReadOnlyList<CudaShardProfile> Shards);
