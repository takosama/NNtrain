namespace NNtrain;

/// <summary>
/// Controls runtime balancing of CUDA data-parallel batches.  The minimum is
/// expressed relative to an even shard: 0.5 keeps at least half of an even
/// share on every selected adapter.
/// </summary>
public sealed record CudaAdaptiveShardingOptions
{
    public bool Enabled { get; init; } = true;
    public double EmaAlpha { get; init; } = 0.2d;
    public double MinimumRelativeShardSize { get; init; } = 0.5d;
    public int MaximumBatchAdjustmentPerStep { get; init; } = 1;

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
    }
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
            if (!_options.Enabled || !_hasObservation)
                return Remember(even);

            int[] target = WeightedAllocation(batchSize, _throughputEma);
            if (_last.Length != devices.Count || _last.Sum() != batchSize)
                return Remember(target);

            int[] limited = (int[])_last.Clone();
            int maximum = _options.MaximumBatchAdjustmentPerStep;
            var movedOut = new int[devices.Count];
            var movedIn = new int[devices.Count];
            while (true)
            {
                int receiver = Enumerable.Range(0, devices.Count)
                    .Where(index => limited[index] < target[index]
                        && movedIn[index] < maximum)
                    .OrderByDescending(index => target[index] - limited[index])
                    .ThenBy(index => index)
                    .DefaultIfEmpty(-1)
                    .First();
                int donor = Enumerable.Range(0, devices.Count)
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

    private void Reset(IReadOnlyList<int> devices)
    {
        _devices = devices.ToArray();
        _last = [];
        _throughputEma = new double[devices.Count];
        _hasObservation = false;
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
