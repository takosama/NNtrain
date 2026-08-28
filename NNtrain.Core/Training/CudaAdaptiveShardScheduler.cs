namespace NNtrain;

/// <summary>
/// Controls runtime balancing of CUDA data-parallel batches.  The minimum is
/// expressed relative to an even shard: 0.5 keeps at least half of an even
/// share on every selected adapter.
/// </summary>
public sealed record CudaAdaptiveShardingOptions
{
    private const long DefaultGraphCacheBudget = 512L * 1024L * 1024L;

    public bool Enabled { get; init; } = true;
    public double EmaAlpha { get; init; } = 0.2d;
    public double MinimumRelativeShardSize { get; init; } = 0.5d;
    public int MaximumBatchAdjustmentPerStep { get; init; } = 1;

    /// <summary>
    /// Successful timing observations required before the first allocation
    /// change can be committed. This prevents a cold sample from selecting a
    /// new compiled training shape.
    /// </summary>
    public int MinimumObservationsBeforeAdjustment { get; init; } = 4;

    /// <summary>
    /// Consecutive observations that must request the same bounded allocation
    /// before it is committed.
    /// </summary>
    public int RequiredConsecutiveCandidateObservations { get; init; } = 3;

    /// <summary>
    /// Successful timing observations required between allocation changes.
    /// CUDA Graphs are shape-specific, so rapid oscillation is very expensive.
    /// </summary>
    public int MinimumStepsBetweenAdjustments { get; init; } = 64;

    /// <summary>
    /// Minimum predicted reduction in slowest-shard completion time for a
    /// cacheable shape change.
    /// </summary>
    public double MinimumPredictedStepTimeImprovement { get; init; } = 0.02d;

    /// <summary>
    /// Minimum predicted reduction when the active graph alone exceeds the
    /// cache budget and changing shape necessarily forces a full recapture.
    /// </summary>
    public double OversizedGraphMinimumPredictedImprovement { get; init; } =
        0.15d;

    /// <summary>
    /// Maximum device memory retained by compiled shape plans. The currently
    /// executing shape is always kept even when it alone exceeds the budget;
    /// older plans are retired behind their stream events before disposal.
    /// </summary>
    public long GraphCacheBudgetBytes { get; init; } =
        DefaultGraphCacheBudget;

    internal void Validate()
    {
        if (!double.IsFinite(EmaAlpha) || EmaAlpha <= 0d || EmaAlpha > 1d)
        {
            throw new ArgumentOutOfRangeException(
                nameof(EmaAlpha), "EMA alpha must be in (0, 1].");
        }
        if (!double.IsFinite(MinimumRelativeShardSize)
            || MinimumRelativeShardSize <= 0d
            || MinimumRelativeShardSize > 1d)
        {
            throw new ArgumentOutOfRangeException(
                nameof(MinimumRelativeShardSize),
                "Minimum relative shard size must be in (0, 1].");
        }
        if (MaximumBatchAdjustmentPerStep < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(MaximumBatchAdjustmentPerStep),
                "Maximum batch adjustment must be positive.");
        }
        if (MinimumObservationsBeforeAdjustment < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(MinimumObservationsBeforeAdjustment),
                "At least one timing observation is required.");
        }
        if (RequiredConsecutiveCandidateObservations < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(RequiredConsecutiveCandidateObservations),
                "Candidate confirmation count must be positive.");
        }
        if (MinimumStepsBetweenAdjustments < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(MinimumStepsBetweenAdjustments),
                "The adjustment interval cannot be negative.");
        }
        ValidateImprovementThreshold(
            MinimumPredictedStepTimeImprovement,
            nameof(MinimumPredictedStepTimeImprovement));
        ValidateImprovementThreshold(
            OversizedGraphMinimumPredictedImprovement,
            nameof(OversizedGraphMinimumPredictedImprovement));
        if (OversizedGraphMinimumPredictedImprovement
            < MinimumPredictedStepTimeImprovement)
        {
            throw new ArgumentOutOfRangeException(
                nameof(OversizedGraphMinimumPredictedImprovement),
                "The oversized-graph threshold cannot be lower than the " +
                "ordinary shape-change threshold.");
        }
        if (GraphCacheBudgetBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(GraphCacheBudgetBytes),
                "CUDA Graph cache budget must be positive.");
        }
    }

    private static void ValidateImprovementThreshold(
        double value,
        string parameterName)
    {
        if (!double.IsFinite(value) || value < 0d || value > 1d)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                "Predicted improvement thresholds must be in [0, 1].");
        }
    }
}

/// <summary>
/// Serializable runtime state for deterministic adaptive CUDA sharding.
/// Timing observations influence the next batch split, so they are part of a
/// resumable training cursor rather than disposable profiler telemetry.
/// </summary>
public sealed record CudaAdaptiveShardState(
    int FormatVersion,
    int[] Devices,
    int[] LastAllocation,
    double[] ThroughputEma,
    bool HasObservation)
{
    public const int CurrentFormatVersion = 2;

    public long ObservationCount { get; init; }
    public long LastAdjustmentObservation { get; init; } = -1;
    public int[] PendingAllocation { get; init; } = [];
    public int PendingConfirmationCount { get; init; }
    public long LastCandidateObservation { get; init; } = -1;
    public int[] OversizedGraphAllocation { get; init; } = [];
    public long OversizedGraphPinnedBytes { get; init; }
}

/// <summary>
/// Deterministic, lock-protected throughput scheduler.  It is deliberately
/// independent of CUDA so its safety bounds and convergence can be unit tested.
/// </summary>
internal sealed class CudaAdaptiveShardScheduler(
    CudaAdaptiveShardingOptions options)
{
    private readonly object _sync = new();
    private readonly CudaAdaptiveShardingOptions _options = options;
    private int[] _devices = [];
    private int[] _last = [];
    private double[] _throughputEma = [];
    private bool _hasObservation;
    private long _observationCount;
    private long _lastAdjustmentObservation = -1;
    private int[] _pendingAllocation = [];
    private int _pendingConfirmationCount;
    private long _lastCandidateObservation = -1;
    private int[] _oversizedGraphAllocation = [];
    private long _oversizedGraphPinnedBytes;

    internal int[] Allocate(int batchSize, IReadOnlyList<int> devices)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        ArgumentNullException.ThrowIfNull(devices);
        if (devices.Count == 0 || devices.Count > batchSize)
            throw new ArgumentException("Shard devices must fit the batch.");

        lock (_sync)
        {
            if (!_devices.SequenceEqual(devices))
                Reset(devices);
            int[] even = EvenAllocation(batchSize, devices.Count);
            if (!_options.Enabled)
                return Remember(even);
            if (!_hasObservation)
            {
                return _last.Length == devices.Count
                        && _last.Sum() == batchSize
                    ? (int[])_last.Clone()
                    : Remember(even);
            }

            int[] target = WeightedAllocation(batchSize, _throughputEma);
            if (_last.Length != devices.Count || _last.Sum() != batchSize)
                return Remember(target);

            int[] limited = LimitAdjustment(_last, target);
            if (limited.SequenceEqual(_last))
            {
                ResetCandidate();
                return (int[])_last.Clone();
            }

            double threshold = _oversizedGraphAllocation
                    .SequenceEqual(_last)
                ? _options.OversizedGraphMinimumPredictedImprovement
                : _options.MinimumPredictedStepTimeImprovement;
            double predictedImprovement = PredictStepTimeImprovement(
                _last,
                target,
                _throughputEma);
            bool enoughHistory = _observationCount
                >= _options.MinimumObservationsBeforeAdjustment;
            bool intervalElapsed = _lastAdjustmentObservation < 0
                || _observationCount - _lastAdjustmentObservation
                    >= _options.MinimumStepsBetweenAdjustments;
            if (!enoughHistory
                || !intervalElapsed
                || predictedImprovement < threshold)
            {
                ResetCandidate();
                return (int[])_last.Clone();
            }

            // Allocate may be queried multiple times between observations.
            // Count no more than one confirmation per successful timing sample.
            if (_lastCandidateObservation != _observationCount)
            {
                if (_pendingAllocation.SequenceEqual(limited))
                {
                    _pendingConfirmationCount++;
                }
                else
                {
                    _pendingAllocation = (int[])limited.Clone();
                    _pendingConfirmationCount = 1;
                }
                _lastCandidateObservation = _observationCount;
            }
            if (_pendingConfirmationCount
                < _options.RequiredConsecutiveCandidateObservations)
            {
                return (int[])_last.Clone();
            }

            _lastAdjustmentObservation = _observationCount;
            ResetCandidate();
            return Remember(limited);
        }
    }

    internal void Observe(
        IReadOnlyList<int> shardBatchSizes,
        IReadOnlyList<double> elapsedMilliseconds)
    {
        ArgumentNullException.ThrowIfNull(shardBatchSizes);
        ArgumentNullException.ThrowIfNull(elapsedMilliseconds);
        if (shardBatchSizes.Count != elapsedMilliseconds.Count)
            throw new ArgumentException("Shard timing lengths must match.");
        lock (_sync)
        {
            if (_throughputEma.Length != shardBatchSizes.Count)
                return;
            for (int index = 0; index < shardBatchSizes.Count; index++)
            {
                double elapsed = elapsedMilliseconds[index];
                int samples = shardBatchSizes[index];
                if (samples <= 0 || !double.IsFinite(elapsed) || elapsed <= 0d)
                {
                    // A corrupted timer must never produce an unsafe shard.
                    _hasObservation = false;
                    Array.Clear(_throughputEma);
                    _observationCount = 0;
                    _lastAdjustmentObservation = -1;
                    ResetCandidate();
                    return;
                }
                double throughput = samples / elapsed;
                _throughputEma[index] = _hasObservation
                    ? _options.EmaAlpha * throughput
                        + (1d - _options.EmaAlpha) * _throughputEma[index]
                    : throughput;
            }
            _hasObservation = _throughputEma.All(value =>
                double.IsFinite(value) && value > 0d);
            if (_hasObservation)
                _observationCount++;
        }
    }

    /// <summary>
    /// Records the allocation of a successfully compiled graph. If it alone
    /// exceeds the cache budget, only a strongly justified and confirmed
    /// allocation change is subsequently accepted.
    /// </summary>
    internal void ObserveCompiledGraph(
        IReadOnlyList<int> allocation,
        long graphPinnedBytes,
        long graphCacheBudgetBytes)
    {
        lock (_sync)
        {
            ValidateCompiledGraphObservation(
                allocation,
                graphPinnedBytes,
                graphCacheBudgetBytes);
            ApplyCompiledGraphObservation(
                allocation,
                graphPinnedBytes,
                graphCacheBudgetBytes);
        }
    }

    /// <summary>
    /// Rebinds a compiled graph after the scheduler itself was reconfigured.
    /// The active physical allocation becomes the zero-history starting point,
    /// avoiding a needless recapture merely because EMA settings changed.
    /// </summary>
    internal void ObserveCompiledGraph(
        IReadOnlyList<int> devices,
        IReadOnlyList<int> allocation,
        long graphPinnedBytes,
        long graphCacheBudgetBytes)
    {
        ArgumentNullException.ThrowIfNull(devices);
        lock (_sync)
        {
            ValidateCompiledGraphObservation(
                allocation,
                graphPinnedBytes,
                graphCacheBudgetBytes);
            if (devices.Count != allocation.Count)
            {
                throw new ArgumentException(
                    "Compiled graph devices and shards must have equal length.",
                    nameof(devices));
            }
            if (_devices.Length == 0)
            {
                _devices = devices.ToArray();
                _last = allocation.ToArray();
                _throughputEma = new double[devices.Count];
                _hasObservation = false;
                _observationCount = 0;
                _lastAdjustmentObservation = -1;
                ResetCandidate();
            }
            if (!_devices.SequenceEqual(devices)
                || !_last.SequenceEqual(allocation))
            {
                return;
            }
            ApplyCompiledGraphObservation(
                allocation,
                graphPinnedBytes,
                graphCacheBudgetBytes);
        }
    }

    private static void ValidateCompiledGraphObservation(
        IReadOnlyList<int> allocation,
        long graphPinnedBytes,
        long graphCacheBudgetBytes)
    {
        ArgumentNullException.ThrowIfNull(allocation);
        ArgumentOutOfRangeException.ThrowIfNegative(graphPinnedBytes);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(
            graphCacheBudgetBytes);
        if (allocation.Count == 0
            || allocation.Any(static value => value <= 0))
        {
            throw new ArgumentException(
                "A compiled graph allocation must contain positive shards.",
                nameof(allocation));
        }
    }

    private void ApplyCompiledGraphObservation(
        IReadOnlyList<int> allocation,
        long graphPinnedBytes,
        long graphCacheBudgetBytes)
    {
        if (graphPinnedBytes > graphCacheBudgetBytes)
        {
            _oversizedGraphAllocation = allocation.ToArray();
            _oversizedGraphPinnedBytes = graphPinnedBytes;
        }
        else if (_oversizedGraphAllocation.SequenceEqual(allocation))
        {
            _oversizedGraphAllocation = [];
            _oversizedGraphPinnedBytes = 0;
        }
    }

    internal int[] LastAllocation
    {
        get
        {
            lock (_sync)
                return (int[])_last.Clone();
        }
    }

    internal CudaAdaptiveShardState CaptureState()
    {
        lock (_sync)
        {
            return new CudaAdaptiveShardState(
                CudaAdaptiveShardState.CurrentFormatVersion,
                (int[])_devices.Clone(),
                (int[])_last.Clone(),
                (double[])_throughputEma.Clone(),
                _hasObservation)
            {
                ObservationCount = _observationCount,
                LastAdjustmentObservation = _lastAdjustmentObservation,
                PendingAllocation = (int[])_pendingAllocation.Clone(),
                PendingConfirmationCount = _pendingConfirmationCount,
                LastCandidateObservation = _lastCandidateObservation,
                OversizedGraphAllocation =
                    (int[])_oversizedGraphAllocation.Clone(),
                OversizedGraphPinnedBytes = _oversizedGraphPinnedBytes,
            };
        }
    }

    internal void RestoreState(
        CudaAdaptiveShardState state,
        IReadOnlyList<int> expectedDevices)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(expectedDevices);
        if (state.FormatVersion is not 1
            && state.FormatVersion != CudaAdaptiveShardState.CurrentFormatVersion)
        {
            throw new ArgumentException(
                $"Unsupported adaptive CUDA shard state format " +
                $"'{state.FormatVersion}'. Expected " +
                $"'{CudaAdaptiveShardState.CurrentFormatVersion}'.",
                nameof(state));
        }
        if (state.Devices is null
            || state.LastAllocation is null
            || state.ThroughputEma is null
            || !state.Devices.SequenceEqual(expectedDevices))
        {
            throw new ArgumentException(
                "Adaptive CUDA shard state does not match the active devices.",
                nameof(state));
        }
        int count = expectedDevices.Count;
        if (state.ThroughputEma.Length != count
            || (state.LastAllocation.Length != 0
                && state.LastAllocation.Length != count)
            || state.LastAllocation.Any(value => value <= 0)
            || state.ThroughputEma.Any(value =>
                !double.IsFinite(value) || value < 0d)
            || (state.HasObservation
                && state.ThroughputEma.Any(value => value <= 0d)))
        {
            throw new ArgumentException(
                "Adaptive CUDA shard state contains invalid allocation or " +
                "throughput values.",
                nameof(state));
        }
        if (state.FormatVersion >= 2
            && (state.ObservationCount < 0
                || state.LastAdjustmentObservation < -1
                || state.LastAdjustmentObservation > state.ObservationCount
                || state.PendingAllocation is null
                || (state.PendingAllocation.Length != 0
                    && (state.PendingAllocation.Length != count
                        || state.PendingAllocation.Any(value => value <= 0)))
                || state.PendingConfirmationCount < 0
                || (state.PendingConfirmationCount != 0
                    && state.PendingAllocation.Length == 0)
                || state.LastCandidateObservation < -1
                || state.LastCandidateObservation > state.ObservationCount
                || state.OversizedGraphAllocation is null
                || (state.OversizedGraphAllocation.Length != 0
                    && (state.OversizedGraphAllocation.Length != count
                        || state.OversizedGraphAllocation.Any(
                            value => value <= 0)))
                || state.OversizedGraphPinnedBytes < 0
                || (state.OversizedGraphPinnedBytes != 0
                    && state.OversizedGraphAllocation.Length == 0)))
        {
            throw new ArgumentException(
                "Adaptive CUDA shard state contains invalid stabilization " +
                "state.",
                nameof(state));
        }

        lock (_sync)
        {
            _devices = (int[])state.Devices.Clone();
            _last = (int[])state.LastAllocation.Clone();
            _throughputEma = (double[])state.ThroughputEma.Clone();
            _hasObservation = state.HasObservation;
            if (state.FormatVersion >= 2)
            {
                _observationCount = state.ObservationCount;
                _lastAdjustmentObservation =
                    state.LastAdjustmentObservation;
                _pendingAllocation =
                    (int[])state.PendingAllocation.Clone();
                _pendingConfirmationCount =
                    state.PendingConfirmationCount;
                _lastCandidateObservation =
                    state.LastCandidateObservation;
                _oversizedGraphAllocation =
                    (int[])state.OversizedGraphAllocation.Clone();
                _oversizedGraphPinnedBytes =
                    state.OversizedGraphPinnedBytes;
            }
            else
            {
                _observationCount = state.HasObservation ? 1 : 0;
                _lastAdjustmentObservation = -1;
                ResetCandidate();
                _oversizedGraphAllocation = [];
                _oversizedGraphPinnedBytes = 0;
            }
        }
    }

    private void Reset(IReadOnlyList<int> devices)
    {
        _devices = devices.ToArray();
        _last = [];
        _throughputEma = new double[devices.Count];
        _hasObservation = false;
        _observationCount = 0;
        _lastAdjustmentObservation = -1;
        ResetCandidate();
        _oversizedGraphAllocation = [];
        _oversizedGraphPinnedBytes = 0;
    }

    private int[] LimitAdjustment(
        IReadOnlyList<int> current,
        IReadOnlyList<int> target)
    {
        int[] limited = current.ToArray();
        int maximum = _options.MaximumBatchAdjustmentPerStep;
        var movedOut = new int[current.Count];
        var movedIn = new int[current.Count];
        while (true)
        {
            int receiver = Enumerable.Range(0, current.Count)
                .Where(index => limited[index] < target[index]
                    && movedIn[index] < maximum)
                .OrderByDescending(index => target[index] - limited[index])
                .ThenBy(index => index)
                .DefaultIfEmpty(-1)
                .First();
            int donor = Enumerable.Range(0, current.Count)
                .Where(index => limited[index] > target[index]
                    && movedOut[index] < maximum)
                .OrderByDescending(index => limited[index] - target[index])
                .ThenBy(index => index)
                .DefaultIfEmpty(-1)
                .First();
            if (receiver < 0 || donor < 0)
                break;
            limited[receiver]++;
            limited[donor]--;
            movedIn[receiver]++;
            movedOut[donor]++;
        }
        return limited;
    }

    private static double PredictStepTimeImprovement(
        IReadOnlyList<int> current,
        IReadOnlyList<int> target,
        IReadOnlyList<double> throughput)
    {
        double currentMilliseconds = 0d;
        double targetMilliseconds = 0d;
        for (int index = 0; index < current.Count; index++)
        {
            currentMilliseconds = Math.Max(
                currentMilliseconds,
                current[index] / throughput[index]);
            targetMilliseconds = Math.Max(
                targetMilliseconds,
                target[index] / throughput[index]);
        }
        if (!double.IsFinite(currentMilliseconds)
            || currentMilliseconds <= 0d
            || !double.IsFinite(targetMilliseconds))
        {
            return 0d;
        }
        return Math.Max(
            0d,
            (currentMilliseconds - targetMilliseconds)
                / currentMilliseconds);
    }

    private void ResetCandidate()
    {
        _pendingAllocation = [];
        _pendingConfirmationCount = 0;
        _lastCandidateObservation = -1;
    }

    private int[] WeightedAllocation(int batchSize, double[] weights)
    {
        int deviceCount = weights.Length;
        int evenFloor = batchSize / deviceCount;
        int minimum = Math.Max(
            1,
            (int)Math.Floor(
                evenFloor * _options.MinimumRelativeShardSize));
        minimum = Math.Min(minimum, batchSize / deviceCount);
        double totalWeight = weights.Sum();
        if (!double.IsFinite(totalWeight) || totalWeight <= 0d)
            return EvenAllocation(batchSize, deviceCount);

        var allocation = new int[deviceCount];
        var fractions = new (int Index, double Fraction)[deviceCount];
        int assigned = 0;
        for (int index = 0; index < deviceCount; index++)
        {
            double exact = batchSize * weights[index] / totalWeight;
            int whole = (int)Math.Floor(exact);
            allocation[index] = whole;
            assigned += whole;
            fractions[index] = (index, exact - whole);
        }
        foreach ((int index, _) in fractions
            .OrderByDescending(value => value.Fraction)
            .ThenBy(value => value.Index)
            .Take(batchSize - assigned))
        {
            allocation[index]++;
        }
        for (int receiver = 0; receiver < deviceCount; receiver++)
        {
            while (allocation[receiver] < minimum)
            {
                int donor = Enumerable.Range(0, deviceCount)
                    .Where(index => allocation[index] > minimum)
                    .OrderByDescending(index => allocation[index] - minimum)
                    .ThenBy(index => index)
                    .DefaultIfEmpty(-1)
                    .First();
                if (donor < 0)
                    return EvenAllocation(batchSize, deviceCount);
                allocation[donor]--;
                allocation[receiver]++;
            }
        }
        return allocation;
    }

    private int[] Remember(int[] allocation)
    {
        _last = (int[])allocation.Clone();
        return allocation;
    }

    private static int[] EvenAllocation(int batchSize, int deviceCount)
    {
        int quotient = batchSize / deviceCount;
        int remainder = batchSize % deviceCount;
        return Enumerable.Range(0, deviceCount)
            .Select(index => quotient + (index < remainder ? 1 : 0))
            .ToArray();
    }
}
