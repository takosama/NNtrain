using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain;

internal readonly record struct CudaTrainingGraphTelemetry(
    long CaptureCount,
    long ReplayCount,
    long FallbackCount,
    int CachedCompiledPlanCount,
    long GraphPinnedBytes,
    long CapturedReadyEventRecordCount,
    double CapturedReadyEventRecordMilliseconds);

/// <summary>
/// One language-model microbatch in a gradient-accumulation update.
/// Token and target arrays remain host-side batch inputs; model state,
/// activations, and accumulated gradients remain resident on their CUDA lane.
/// </summary>
public readonly record struct CudaLanguageModelMicroBatch(
    int[] Input,
    int[] Target,
    int BatchSize,
    int SequenceLength);

/// <summary>
/// Owns the reusable CUDA data-parallel resources for one language model.
/// Dispose the engine before ending a training session so gradient arenas,
/// reduction buffers, streams, and events are released deterministically.
/// </summary>
public sealed partial class CudaDataParallelEngine : IDisposable
{
    private readonly object _sync = new();
    private readonly LanguageModel _model;
    private readonly Parameter[] _parameters;
    private readonly int[] _cudaDeviceIndices;
    private readonly int[][] _devicePrefixes;
    private readonly CudaReplicaExecutor _replicaExecutor;
    private readonly CudaDispatchPolicy _dispatchPolicy;
    private readonly LinkedList<CudaTrainingShapePlan> _trainingShapePlans = [];
    private long _graphCacheBudgetBytes;
    private CudaAdaptiveShardScheduler _adaptiveShardScheduler;
    private TensorCudaKernels.FlatGradientPlan? _flatGradientPlan;
    private CudaBFloat16GradientAllReducePlan? _bfloat16GradientPlan;
    private CudaBfp8GradientAllReducePlan? _bfp8GradientPlan;
    private readonly Dictionary<int, NativeCudaBuffer<float>>
        _accumulatedLossBuffers = [];
    private int _flatGradientPlanBuildCount;
    private int _bfloat16GradientPlanBuildCount;
    private int _trainingShapePlanBuildCount;
    private long _trainingShapePlanEvictionCount;
    private long _implicitGlobalStep;
    private long _graphCaptureCount;
    private long _graphReplayCount;
    private long _graphFallbackCount;
    private Exception? _lastGraphFailure;
    private bool _parameterResidencyPrepared;
    private TensorPrecisionMode? _preflightPrecisionMode;
    private bool _preflightCompiledGraphRequested;
    private int _disposed;

    public CudaDataParallelEngine(
        LanguageModel model,
        CudaAdaptiveShardingOptions? adaptiveShardingOptions = null)
        : this(model, Tensor.CudaDeviceIndices, adaptiveShardingOptions)
    {
    }

    public CudaDataParallelEngine(
        LanguageModel model,
        IReadOnlyList<int> cudaDeviceIndices,
        CudaAdaptiveShardingOptions? adaptiveShardingOptions = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _dispatchPolicy = CudaDispatchPolicy.Current.Validate();
        _parameters = _model.Parameters().ToArray();
        ArgumentNullException.ThrowIfNull(cudaDeviceIndices);
        if (cudaDeviceIndices.Count == 0)
        {
            throw new ArgumentException(
                "At least one CUDA device must belong to the engine.",
                nameof(cudaDeviceIndices));
        }
        _cudaDeviceIndices = cudaDeviceIndices.ToArray();
        if (_cudaDeviceIndices.Any(index => index < 0)
            || _cudaDeviceIndices.Distinct().Count()
                != _cudaDeviceIndices.Length)
        {
            throw new ArgumentException(
                "CUDA device indices must be unique and non-negative.",
                nameof(cudaDeviceIndices));
        }
        _devicePrefixes = new int[_cudaDeviceIndices.Length][];
        for (int count = 1; count <= _cudaDeviceIndices.Length; count++)
            _devicePrefixes[count - 1] = _cudaDeviceIndices[..count];
        CudaAdaptiveShardingOptions options = adaptiveShardingOptions
            ?? new CudaAdaptiveShardingOptions();
        options.Validate();
        _graphCacheBudgetBytes = options.GraphCacheBudgetBytes;
        _adaptiveShardScheduler = new CudaAdaptiveShardScheduler(options);
        _replicaExecutor = new CudaReplicaExecutor(_cudaDeviceIndices);
    }

    private sealed class DeferredGradientReductionPlan
        : ICudaGradientReductionPlan
    {
        internal static DeferredGradientReductionPlan Instance { get; } =
            new();

        private DeferredGradientReductionPlan()
        {
        }

        public void NotifyGradientReady(
            Tensor tensor,
            int deviceIndex,
            long stepId)
        {
            // Earlier microbatches accumulate into the reducer-owned local
            // gradient arena. The final backward publishes every leaf once,
            // which starts the single reduction for the whole update.
        }
    }

    /// <summary>
    /// Reuses the managed part of a stable CUDA training shape.  CUDA tensor
    /// storage remains lane-owned; this plan removes per-step shard arrays and
    /// scratch result arrays while retaining only the three most recent
    /// adaptive shapes.
    /// </summary>
    private sealed partial class CudaTrainingShapePlan
        : ICudaReplicaWorkDescriptor, IDisposable
    {
        private readonly CudaDataParallelEngine _owner;
        private readonly CompiledDeviceGraph?[] _compiledGraphs;
        private readonly CudaGraphBatchInputs?[] _graphInputs;
        private readonly CudaGraphRngState?[] _graphRngStates;
        private readonly NativeCudaBuffer<float>?[] _graphLossSlots;
        private readonly CudaBfp8GraphGradientPublication?[]
            _graphBfp8Publications;
        private readonly CudaGraphCaptureScope?[] _captureMemoryScopes;
        private readonly long[] _capturePinnedBaselines;
        private CudaBfp8GradientAllReducePlan? _bfp8Plan;
        private CudaBFloat16GradientAllReducePlan? _bfloat16Plan;
        private ICudaGradientReductionPlan? _reductionPlan;
        private long _reductionStepId;
        private long _globalStep;
        private ulong _rootSeed;
        private int _totalValid;
        private int _ignoreIndex;
        private int _warmupCount;
        private bool _reuseCaptureInputs;
        private bool _useFixedInputs;
        private bool _readEagerLoss;
        private bool _beginReductionDeviceStep;
        private bool _publishGradientsAfterReplay;
        private int _rngSequence;
        private NativeCudaBuffer<float>[]? _accumulatedLossBuffers;
        private bool _completeAccumulatedLoss;
        private bool _graphDisabled;
        private int _compiledTotalValid;
        private int _disposed;
        private WorkPhase _phase;

        internal CudaTrainingShapePlan(
            CudaDataParallelEngine owner,
            int batchSize,
            int sequenceLength,
            IReadOnlyList<int> devices,
            IReadOnlyList<int> shardBatches)
        {
            _owner = owner;
            BatchSize = batchSize;
            SequenceLength = sequenceLength;
            Devices = devices.ToArray();
            ShardBatches = shardBatches.ToArray();
            ShardStarts = GetShardStarts(ShardBatches);
            ShardInputs = new int[Devices.Length][];
            ShardTargets = new int[Devices.Length][];
            ShardValidTargets = new int[Devices.Length];
            WeightedLosses = new double[Devices.Length];
            ShardElapsed = new double[Devices.Length];
            ShardStarted = new long[Devices.Length];
            _compiledGraphs = new CompiledDeviceGraph?[Devices.Length];
            _graphInputs = new CudaGraphBatchInputs?[Devices.Length];
            _graphRngStates = new CudaGraphRngState?[Devices.Length];
            _graphLossSlots = new NativeCudaBuffer<float>?[Devices.Length];
            _graphBfp8Publications =
                new CudaBfp8GraphGradientPublication?[Devices.Length];
            _captureMemoryScopes = new CudaGraphCaptureScope?[Devices.Length];
            _capturePinnedBaselines = new long[Devices.Length];
            for (int shard = 0; shard < Devices.Length; shard++)
            {
                int length = checked(ShardBatches[shard] * sequenceLength);
                ShardInputs[shard] = new int[length];
                ShardTargets[shard] = new int[length];
            }
        }

        internal int BatchSize { get; }
        internal int SequenceLength { get; }
        internal int[] Devices { get; }
        internal int[] ShardBatches { get; }
        internal int[] ShardStarts { get; }
        internal int[][] ShardInputs { get; }
        internal int[][] ShardTargets { get; }
        internal int[] ShardValidTargets { get; }
        internal double[] WeightedLosses { get; }
        internal double[] ShardElapsed { get; }
        internal long[] ShardStarted { get; }

        internal bool IsCompiled
            => !_graphDisabled
                && _compiledGraphs.All(static graph => graph is not null);

        internal bool IsCompiledFor(int totalValid)
            => IsCompiled && _compiledTotalValid == totalValid;

        internal long GraphPinnedBytes
            => _compiledGraphs.Sum(static graph => graph?.GraphPinnedBytes ?? 0);

        internal bool GraphDisabled => _graphDisabled;

        internal bool ShouldCapture
            => !_graphDisabled && !IsCompiled && _warmupCount >= 2;

        internal int WarmupCount => _warmupCount;

        private static ulong DeriveCounter(
            ulong rootSeed,
            long globalStep,
            int deviceIndex,
            int rngSequence = 0)
            => Mix64(
                rootSeed
                ^ unchecked((ulong)globalStep * 0x9e3779b97f4a7c15UL)
                ^ unchecked((ulong)(uint)deviceIndex
                    * 0xbf58476d1ce4e5b9UL)
                ^ unchecked((ulong)(uint)rngSequence
                    * 0x94d049bb133111ebUL));

        private static ulong DeriveOperationSeed(
            ulong rootSeed,
            int deviceIndex)
            => Mix64(
                rootSeed
                ^ 0x4752_4150_4852_4e47UL
                ^ unchecked((ulong)(uint)deviceIndex
                    * 0x94d049bb133111ebUL));

        private static ulong Mix64(ulong value)
        {
            value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9UL;
            value = (value ^ (value >> 27)) * 0x94d049bb133111ebUL;
            return value ^ (value >> 31);
        }

        private sealed class CompiledDeviceGraph(
            CudaExecutionLane lane,
            CudaGraphExecutable graph,
            IDisposable memoryReservation,
            NativeCudaBuffer<float> lossSlot,
            CudaGraphRngState rng,
            CudaGraphBatchInputs inputs,
            CudaBfp8GraphGradientPublication? bfp8Publication,
            int[]? bfloat16BucketOrder,
            long graphPinnedBytes) : IDisposable
        {
            private int _disposed;

            internal CudaExecutionLane Lane { get; } = lane;
            internal CudaGraphExecutable Graph { get; } = graph;
            internal NativeCudaBuffer<float> LossSlot { get; } = lossSlot;
            internal CudaGraphRngState Rng { get; } = rng;
            internal CudaGraphBatchInputs Inputs { get; } = inputs;
            internal CudaBfp8GraphGradientPublication? Bfp8Publication
                { get; } = bfp8Publication;
            internal int[]? BFloat16BucketOrder { get; }
                = bfloat16BucketOrder;
            internal long GraphPinnedBytes { get; } = graphPinnedBytes;

            public void Dispose()
            {
                if (Interlocked.Exchange(ref _disposed, 1) != 0)
                    return;
                var failures = new List<Exception>();
                try
                {
                    Lane.SynchronizeComputeStream();
                }
                catch (Exception exception)
                {
                    failures.Add(exception);
                }
                TryDispose(Graph, failures);
                TryDispose(memoryReservation, failures);
                TryDispose(Bfp8Publication, failures);
                TryDispose(LossSlot, failures);
                TryDispose(Rng, failures);
                TryDispose(Inputs, failures);
                if (failures.Count != 0)
                {
                    throw new AggregateException(
                        "CUDA compiled training graph cleanup failed.",
                        failures);
                }
            }
        }

        private enum WorkPhase
        {
            None = 0,
            Eager = 1,
            Synchronize = 2,
            Capture = 3,
            Replay = 4,
            DisposeGraph = 5,
            InitializeGraphInputs = 6,
            ReservationPrewarm = 7,
        }
    }

}
