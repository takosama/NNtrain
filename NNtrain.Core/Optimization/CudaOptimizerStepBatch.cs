namespace NNtrain;

/// <summary>
/// Coalesces the completion barriers of disjoint CUDA optimizers. Kernels and
/// scalar status readbacks are first queued on each lane's compute stream;
/// every participating stream is then synchronized exactly once before the
/// individual optimizer states are finalized.
/// </summary>
internal static class CudaOptimizerStepBatch
{
    private static readonly AsyncLocal<BatchState?> CurrentState = new();

    internal static Scope Enter(IReadOnlyList<int> deviceIndices)
    {
        ArgumentNullException.ThrowIfNull(deviceIndices);
        BatchState? current = CurrentState.Value;
        if (current is not null)
            return new Scope(current, ownsState: false);

        var created = new BatchState(
            NormalizeDevices(deviceIndices),
            static (deviceIndex, _) =>
                NativeCudaRuntime.SynchronizeDeviceComputeStream(deviceIndex));
        CurrentState.Value = created;
        CudaOptimizerSynchronizationTelemetry.RecordBatchStarted();
        return new Scope(created, ownsState: true);
    }

    internal static Scope EnterForTesting(
        IReadOnlyList<int> deviceIndices,
        Action<int, string> synchronizeDevice)
    {
        ArgumentNullException.ThrowIfNull(synchronizeDevice);
        if (CurrentState.Value is not null)
        {
            throw new InvalidOperationException(
                "A CUDA optimizer completion batch is already active.");
        }
        var created = new BatchState(
            NormalizeDevices(deviceIndices),
            synchronizeDevice);
        CurrentState.Value = created;
        CudaOptimizerSynchronizationTelemetry.RecordBatchStarted();
        return new Scope(created, ownsState: true);
    }

    /// <summary>
    /// Completes an optimizer immediately when it is standalone, or appends
    /// its completion work to the active composite batch. <paramref
    /// name="queueReadback"/> must only enqueue work; it runs before the one
    /// consolidated stream barrier. <paramref name="finalize"/> runs after it.
    /// </summary>
    internal static void CompleteAfterSynchronization(
        IReadOnlyList<int> deviceIndices,
        string operation,
        Action? queueReadback,
        Action finalize)
    {
        ArgumentNullException.ThrowIfNull(deviceIndices);
        ArgumentException.ThrowIfNullOrWhiteSpace(operation);
        ArgumentNullException.ThrowIfNull(finalize);
        int[] devices = NormalizeDevices(deviceIndices);
        CudaOptimizerSynchronizationTelemetry.RecordBarrierRequested(
            devices.Length,
            deferred: CurrentState.Value is not null);

        BatchState? current = CurrentState.Value;
        if (current is not null)
        {
            current.Enqueue(devices, operation, queueReadback, finalize);
            return;
        }

        Exception? enqueueFailure = null;
        try
        {
            queueReadback?.Invoke();
        }
        catch (Exception exception)
        {
            enqueueFailure = exception;
        }

        Exception? synchronizationFailure = null;
        try
        {
            SynchronizeStandalone(devices, operation);
        }
        catch (Exception exception)
        {
            synchronizationFailure = exception;
        }
        if (enqueueFailure is not null || synchronizationFailure is not null)
        {
            if (enqueueFailure is not null
                && synchronizationFailure is not null)
            {
                throw new AggregateException(
                    $"{operation} status-readback enqueue and CUDA stream " +
                    "drain both reported failures.",
                    enqueueFailure,
                    synchronizationFailure);
            }
            Exception failure = enqueueFailure ?? synchronizationFailure!;
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(failure)
                .Throw();
        }
        finalize();
    }

    internal static void RecordClipScaleBarrierElided(int deviceCount)
        => CudaOptimizerSynchronizationTelemetry.RecordClipBarrierElided(
            deviceCount);

    private static void SynchronizeStandalone(
        IReadOnlyList<int> devices,
        string operation)
    {
        List<Exception>? failures = null;
        foreach (int deviceIndex in devices)
        {
            try
            {
                NativeCudaRuntime.SynchronizeDeviceComputeStream(deviceIndex);
                CudaOptimizerSynchronizationTelemetry
                    .RecordPhysicalSynchronization();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(new InvalidOperationException(
                    $"{operation} failed while synchronizing CUDA device " +
                    $"{deviceIndex}.",
                    exception));
            }
        }
        ThrowFailures(failures);
    }

    private static int[] NormalizeDevices(IReadOnlyList<int> deviceIndices)
    {
        if (deviceIndices.Count == 0)
        {
            throw new ArgumentException(
                "At least one CUDA device is required.",
                nameof(deviceIndices));
        }
        int[] devices = deviceIndices.Distinct().ToArray();
        if (devices.Any(deviceIndex => deviceIndex < 0))
        {
            throw new ArgumentOutOfRangeException(
                nameof(deviceIndices),
                "CUDA device indices must be non-negative.");
        }
        return devices;
    }

    private static void ThrowFailures(List<Exception>? failures)
    {
        if (failures is null || failures.Count == 0)
            return;
        if (failures.Count == 1)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(failures[0])
                .Throw();
        }
        throw new AggregateException(
            "CUDA optimizer completion failed.", failures);
    }

    internal sealed class Scope : IDisposable
    {
        private BatchState? _state;
        private readonly bool _ownsState;

        internal Scope(BatchState state, bool ownsState)
        {
            _state = state;
            _ownsState = ownsState;
        }

        internal void Complete()
        {
            BatchState? state = _state
                ?? throw new ObjectDisposedException(nameof(Scope));
            if (!_ownsState)
                return;
            try
            {
                Exception? failure = state.Drain(primaryFailure: null);
                if (failure is not null)
                {
                    System.Runtime.ExceptionServices.ExceptionDispatchInfo
                        .Capture(failure)
                        .Throw();
                }
            }
            finally
            {
                CurrentState.Value = null;
                _state = null;
            }
        }

        internal Exception DrainAfterFailure(Exception primaryFailure)
        {
            ArgumentNullException.ThrowIfNull(primaryFailure);
            BatchState? state = _state;
            if (state is null || !_ownsState)
                return primaryFailure;
            try
            {
                return state.Drain(primaryFailure) ?? primaryFailure;
            }
            finally
            {
                CurrentState.Value = null;
                _state = null;
            }
        }

        public void Dispose()
        {
            BatchState? state = _state;
            _state = null;
            if (state is null || !_ownsState)
                return;
            CurrentState.Value = null;
            List<Exception>? failures = null;
            try
            {
                Exception? failure = state.Drain(primaryFailure: null);
                if (failure is not null)
                    failures = [failure];
            }
            catch (Exception exception)
            {
                failures = [exception];
            }
            ThrowFailures(failures);
        }
    }

    internal sealed class BatchState(
        IReadOnlyList<int> initialDevices,
        Action<int, string> synchronizeDevice)
    {
        private readonly HashSet<int> _devices = [.. initialDevices];
        private readonly List<Completion> _completions = [];
        private int _drained;

        internal void Enqueue(
            IReadOnlyList<int> devices,
            string operation,
            Action? queueReadback,
            Action finalize)
        {
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _drained) != 0,
                this);
            foreach (int device in devices)
                _devices.Add(device);
            _completions.Add(new Completion(
                operation,
                queueReadback,
                finalize));
        }

        internal Exception? Drain(Exception? primaryFailure)
        {
            if (Interlocked.Exchange(ref _drained, 1) != 0)
                return primaryFailure;
            bool failureDrain = primaryFailure is not null;
            if (failureDrain)
                CudaOptimizerSynchronizationTelemetry.RecordFailureDrain();

            List<Exception>? failures = primaryFailure is null
                ? null
                : [primaryFailure];
            foreach (Completion completion in _completions)
            {
                if (completion.QueueReadback is null)
                    continue;
                try
                {
                    completion.QueueReadback();
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(new InvalidOperationException(
                        $"{completion.Operation} status readback could not " +
                        "be queued.",
                        exception));
                }
            }

            bool synchronizationFailed = false;
            string operationNames = string.Join(
                ", ",
                _completions.Select(value => value.Operation).Distinct());
            if (operationNames.Length == 0)
                operationNames = "CUDA optimizer failure drain";
            foreach (int deviceIndex in _devices)
            {
                try
                {
                    synchronizeDevice(deviceIndex, operationNames);
                    CudaOptimizerSynchronizationTelemetry
                        .RecordPhysicalSynchronization();
                }
                catch (Exception exception)
                {
                    synchronizationFailed = true;
                    (failures ??= []).Add(new InvalidOperationException(
                        $"{operationNames} failed while synchronizing CUDA " +
                        $"device {deviceIndex}.",
                        exception));
                }
            }

            // Finalizers may read pinned status values, so none can run if a
            // stream failed to synchronize. If the original child threw but
            // the drain succeeded, already-enqueued children are finalized to
            // keep tensor coherence accurate.
            if (!synchronizationFailed)
            {
                foreach (Completion completion in _completions)
                {
                    try
                    {
                        completion.Finalize();
                    }
                    catch (Exception exception)
                    {
                        (failures ??= []).Add(exception);
                    }
                }
            }
            CudaOptimizerSynchronizationTelemetry.RecordBatchCompleted();

            if (failures is null || failures.Count == 0)
                return null;
            if (failures.Count == 1)
                return failures[0];
            return new AggregateException(
                "CUDA optimizer batch failed after safely draining its " +
                "compute streams.",
                failures);
        }
    }

    private sealed record Completion(
        string Operation,
        Action? QueueReadback,
        Action Finalize);
}

/// <summary>Process-wide counters used by focused synchronization tests.</summary>
internal static class CudaOptimizerSynchronizationTelemetry
{
    private static long _logicalBarrierRequests;
    private static long _requestedDeviceSynchronizations;
    private static long _deferredBarrierRequests;
    private static long _physicalComputeStreamSynchronizations;
    private static long _batchStarts;
    private static long _batchCompletions;
    private static long _failureDrains;
    private static long _clipScaleBarriersElided;

    internal static CudaOptimizerSynchronizationTelemetrySnapshot Snapshot
        => new(
            Interlocked.Read(ref _logicalBarrierRequests),
            Interlocked.Read(ref _requestedDeviceSynchronizations),
            Interlocked.Read(ref _deferredBarrierRequests),
            Interlocked.Read(ref _physicalComputeStreamSynchronizations),
            Interlocked.Read(ref _batchStarts),
            Interlocked.Read(ref _batchCompletions),
            Interlocked.Read(ref _failureDrains),
            Interlocked.Read(ref _clipScaleBarriersElided));

    internal static void RecordBarrierRequested(
        int deviceCount,
        bool deferred)
    {
        Interlocked.Increment(ref _logicalBarrierRequests);
        Interlocked.Add(ref _requestedDeviceSynchronizations, deviceCount);
        if (deferred)
            Interlocked.Increment(ref _deferredBarrierRequests);
    }

    internal static void RecordPhysicalSynchronization()
        => Interlocked.Increment(
            ref _physicalComputeStreamSynchronizations);

    internal static void RecordBatchStarted()
        => Interlocked.Increment(ref _batchStarts);

    internal static void RecordBatchCompleted()
        => Interlocked.Increment(ref _batchCompletions);

    internal static void RecordFailureDrain()
        => Interlocked.Increment(ref _failureDrains);

    internal static void RecordClipBarrierElided(int deviceCount)
        => Interlocked.Add(ref _clipScaleBarriersElided, deviceCount);
}

internal readonly record struct CudaOptimizerSynchronizationTelemetrySnapshot(
    long LogicalBarrierRequests,
    long RequestedDeviceSynchronizations,
    long DeferredBarrierRequests,
    long PhysicalComputeStreamSynchronizations,
    long BatchStarts,
    long BatchCompletions,
    long FailureDrains,
    long ClipScaleBarriersElided)
{
    public static CudaOptimizerSynchronizationTelemetrySnapshot operator -(
        CudaOptimizerSynchronizationTelemetrySnapshot left,
        CudaOptimizerSynchronizationTelemetrySnapshot right)
        => new(
            left.LogicalBarrierRequests - right.LogicalBarrierRequests,
            left.RequestedDeviceSynchronizations
                - right.RequestedDeviceSynchronizations,
            left.DeferredBarrierRequests - right.DeferredBarrierRequests,
            left.PhysicalComputeStreamSynchronizations
                - right.PhysicalComputeStreamSynchronizations,
            left.BatchStarts - right.BatchStarts,
            left.BatchCompletions - right.BatchCompletions,
            left.FailureDrains - right.FailureDrains,
            left.ClipScaleBarriersElided - right.ClipScaleBarriersElided);
}
