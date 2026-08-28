namespace NNtrain;

/// <summary>
/// Process-wide diagnostic counters for the fixed-depth CUDA NekoMuon path.
/// A batched GEMM launch advances several logical matrices at once; keeping
/// both counts makes launch-amortization directly testable without profiling
/// private cuBLAS entry points.
/// </summary>
internal static class NekoMuonFixedNs5Telemetry
{
    private static long _scalarDispatchCount;
    private static long _batchedDispatchCount;
    private static long _logicalMatrixCount;
    private static long _gemmLaunchCount;
    private static long _kernelLaunchCount;

    internal static NekoMuonFixedNs5TelemetrySnapshot Snapshot => new(
        Interlocked.Read(ref _scalarDispatchCount),
        Interlocked.Read(ref _batchedDispatchCount),
        Interlocked.Read(ref _logicalMatrixCount),
        Interlocked.Read(ref _gemmLaunchCount),
        Interlocked.Read(ref _kernelLaunchCount));

    internal static void RecordScalar(int rows)
    {
        Interlocked.Increment(ref _scalarDispatchCount);
        Interlocked.Increment(ref _logicalMatrixCount);
        if (rows <= CudaOptimizerKernels.DirectNewtonSchulzRowLimit)
        {
            // symmetric Gram, squared Gram, polynomial update, five times.
            Interlocked.Add(ref _kernelLaunchCount, 15);
            return;
        }

        // Three cuBLAS GEMMs plus one coefficient-combine kernel per NS step.
        Interlocked.Add(ref _gemmLaunchCount, 15);
        Interlocked.Add(ref _kernelLaunchCount, 20);
    }

    internal static void RecordBatch(int logicalMatrixCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(logicalMatrixCount);
        Interlocked.Increment(ref _batchedDispatchCount);
        Interlocked.Add(ref _logicalMatrixCount, logicalMatrixCount);
        Interlocked.Add(ref _gemmLaunchCount, 15);
        Interlocked.Add(ref _kernelLaunchCount, 20);
    }
}

internal readonly record struct NekoMuonFixedNs5TelemetrySnapshot(
    long ScalarDispatchCount,
    long BatchedDispatchCount,
    long LogicalMatrixCount,
    long GemmLaunchCount,
    long KernelLaunchCount)
{
    public static NekoMuonFixedNs5TelemetrySnapshot operator -(
        NekoMuonFixedNs5TelemetrySnapshot left,
        NekoMuonFixedNs5TelemetrySnapshot right)
        => new(
            left.ScalarDispatchCount - right.ScalarDispatchCount,
            left.BatchedDispatchCount - right.BatchedDispatchCount,
            left.LogicalMatrixCount - right.LogicalMatrixCount,
            left.GemmLaunchCount - right.GemmLaunchCount,
            left.KernelLaunchCount - right.KernelLaunchCount);
}
