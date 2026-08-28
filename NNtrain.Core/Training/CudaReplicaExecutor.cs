using NNtrain.Runtime.Execution;

namespace NNtrain;

/// <summary>
/// A reusable unit of per-replica work. Implementations are owned by a cached
/// shape plan and may be dispatched repeatedly after their mutable step fields
/// have been prepared by the engine lock.
/// </summary>
internal interface ICudaReplicaWorkDescriptor
{
    void Execute(int replicaIndex, CancellationToken cancellationToken);
}

internal readonly record struct CudaReplicaExecutorTelemetrySnapshot(
    long DispatchCount,
    long CompletedDispatchCount,
    long FailedDispatchCount,
    long ReplicaExecutionCount,
    int WorkerThreadCreationCount,
    int LiveWorkerCount,
    int ActiveReplicaCount,
    int MaxConcurrentReplicaCount,
    int[] DeviceIndices,
    int[] WorkerThreadIds,
    long[] WorkerExecutionCounts,
    long[] WorkerContextBindingCounts);

/// <summary>
/// Runs CUDA replicas on one persistent managed thread per device. A worker
/// retains its ExecutionSession, precision and device scopes between stable
/// steps, so native device/stream TLS and managed execution state remain bound
/// to the same thread without ThreadPool scheduling on every step.
/// </summary>
internal sealed class CudaReplicaExecutor : IDisposable
{
    private static int _nextExecutorId;

    private readonly object _dispatchGate = new();
    private readonly CountdownEvent _completion = new(0);
    private readonly Worker[] _workers;
    private readonly Exception?[] _workerFailures;
    private readonly int[] _deviceIndices;
    private readonly Action<int> _bindDevice;
    private ICudaReplicaWorkDescriptor? _work;
    private ExecutionSession? _session;
    private PrecisionPolicy? _precision;
    private DeviceTransferGuard.SharedContext? _transferContext;
    private CancellationToken _cancellationToken;
    private int _activeWorkerCount;
    private int _disposeRequested;
    private long _dispatchCount;
    private long _completedDispatchCount;
    private long _failedDispatchCount;
    private long _replicaExecutionCount;
    private int _workerThreadCreationCount;
    private int _liveWorkerCount;
    private int _activeReplicaCount;
    private int _maxConcurrentReplicaCount;
    private long _dispatchGeneration;

    internal CudaReplicaExecutor(
        IReadOnlyList<int> deviceIndices,
        Action<int>? bindDevice = null)
    {
        ArgumentNullException.ThrowIfNull(deviceIndices);
        if (deviceIndices.Count == 0)
        {
            throw new ArgumentException(
                "At least one CUDA replica worker is required.",
                nameof(deviceIndices));
        }
        _deviceIndices = deviceIndices.ToArray();
        if (_deviceIndices.Any(static index => index < 0)
            || _deviceIndices.Distinct().Count() != _deviceIndices.Length)
        {
            throw new ArgumentException(
                "CUDA replica worker device indices must be unique and " +
                "non-negative.",
                nameof(deviceIndices));
        }
        _bindDevice = bindDevice ?? BindNativeDevice;
        _workers = new Worker[_deviceIndices.Length];
        _workerFailures = new Exception?[_deviceIndices.Length];
        int executorId = Interlocked.Increment(ref _nextExecutorId);
        int started = 0;
        try
        {
            for (int index = 0; index < _workers.Length; index++)
            {
                _workers[index] = new Worker(
                    this,
                    index,
                    _deviceIndices[index],
                    $"NNtrain CUDA replica {executorId}:{_deviceIndices[index]}");
                _workers[index].Start();
                started++;
            }
        }
        catch (Exception startFailure)
        {
            var failures = new List<Exception> { startFailure };
            StopAndJoinWorkers(started, failures);
            try
            {
                _completion.Dispose();
            }
            catch (Exception cleanupFailure)
            {
                failures.Add(cleanupFailure);
            }
            throw new AggregateException(
                "CUDA replica workers could not be started.", failures);
        }
    }

    internal CudaReplicaExecutorTelemetrySnapshot Telemetry
    {
        get
        {
            var threadIds = new int[_workers.Length];
            var executions = new long[_workers.Length];
            var bindings = new long[_workers.Length];
            for (int index = 0; index < _workers.Length; index++)
            {
                Worker? worker = _workers[index];
                if (worker is null)
                    continue;
                threadIds[index] = Volatile.Read(ref worker.ThreadId);
                executions[index] = Interlocked.Read(
                    ref worker.ExecutionCount);
                bindings[index] = Interlocked.Read(
                    ref worker.ContextBindingCount);
            }
            return new CudaReplicaExecutorTelemetrySnapshot(
                Interlocked.Read(ref _dispatchCount),
                Interlocked.Read(ref _completedDispatchCount),
                Interlocked.Read(ref _failedDispatchCount),
                Interlocked.Read(ref _replicaExecutionCount),
                Volatile.Read(ref _workerThreadCreationCount),
                Volatile.Read(ref _liveWorkerCount),
                Volatile.Read(ref _activeReplicaCount),
                Volatile.Read(ref _maxConcurrentReplicaCount),
                (int[])_deviceIndices.Clone(),
                threadIds,
                executions,
                bindings);
        }
    }

    internal void Execute(
        ICudaReplicaWorkDescriptor work,
        int replicaCount,
        ExecutionSession? session,
        PrecisionPolicy precision,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(work);
        ArgumentNullException.ThrowIfNull(precision);
        if (replicaCount <= 0 || replicaCount > _workers.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(replicaCount), replicaCount,
                $"Replica count must be between 1 and {_workers.Length}.");
        }
        cancellationToken.ThrowIfCancellationRequested();

        lock (_dispatchGate)
        {
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _disposeRequested) != 0,
                this);
            Array.Clear(_workerFailures, 0, replicaCount);
            _work = work;
            _session = session;
            _precision = precision;
            _transferContext = DeviceTransferGuard.CaptureCurrentContext();
            _cancellationToken = cancellationToken;
            Volatile.Write(ref _activeWorkerCount, replicaCount);
            _completion.Reset(replicaCount);
            Interlocked.Increment(ref _dispatchCount);
            Interlocked.Increment(ref _dispatchGeneration);
            try
            {
                for (int index = 0; index < replicaCount; index++)
                    _workers[index].SignalWork();
                _completion.Wait();

                List<Exception>? failures = null;
                for (int index = 0; index < replicaCount; index++)
                {
                    if (_workerFailures[index] is Exception failure)
                        (failures ??= []).Add(failure);
                }
                if (failures is not null)
                {
                    throw new AggregateException(
                        "One or more CUDA replica workers failed. Inner " +
                        "exceptions are ordered by replica/device index.",
                        failures);
                }
                cancellationToken.ThrowIfCancellationRequested();
                Interlocked.Increment(ref _completedDispatchCount);
            }
            catch
            {
                Interlocked.Increment(ref _failedDispatchCount);
                throw;
            }
            finally
            {
                _work = null;
                _session = null;
                _precision = null;
                _transferContext = null;
                _cancellationToken = default;
                Volatile.Write(ref _activeWorkerCount, 0);
            }
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposeRequested, 1) != 0)
            return;

        var failures = new List<Exception>();
        lock (_dispatchGate)
        {
            StopAndJoinWorkers(_workers.Length, failures);
            try
            {
                _completion.Dispose();
            }
            catch (Exception exception)
            {
                failures.Add(exception);
            }
        }
        if (failures.Count != 0)
        {
            throw new AggregateException(
                "CUDA replica executor failed to stop cleanly.", failures);
        }
    }

    private void ExecuteWorker(Worker worker)
    {
        ICudaReplicaWorkDescriptor work = _work
            ?? throw new InvalidOperationException(
                "A CUDA replica worker was signaled without work.");
        PrecisionPolicy precision = _precision
            ?? throw new InvalidOperationException(
                "A CUDA replica worker was signaled without a precision policy.");
        DeviceTransferGuard.SharedContext? transferContext =
            _transferContext;
        worker.EnsureExecutionContext(_session, precision, _bindDevice);
        _cancellationToken.ThrowIfCancellationRequested();
        IDisposable? transferScope = transferContext is null
            ? null
            : DeviceTransferGuard.EnterSharedContext(transferContext);
        int active = Interlocked.Increment(ref _activeReplicaCount);
        UpdateMaximum(ref _maxConcurrentReplicaCount, active);
        try
        {
            work.Execute(worker.Index, _cancellationToken);
            Interlocked.Increment(ref worker.ExecutionCount);
            Interlocked.Increment(ref _replicaExecutionCount);
        }
        finally
        {
            Interlocked.Decrement(ref _activeReplicaCount);
            transferScope?.Dispose();
        }
    }

    private void StopAndJoinWorkers(
        int count,
        List<Exception> failures)
    {
        for (int index = 0; index < count; index++)
        {
            try
            {
                _workers[index]?.RequestStop();
            }
            catch (Exception exception)
            {
                failures.Add(exception);
            }
        }
        for (int index = 0; index < count; index++)
        {
            Worker? worker = _workers[index];
            if (worker is null)
                continue;
            try
            {
                worker.Join();
            }
            catch (Exception exception)
            {
                failures.Add(exception);
            }
            if (worker.ShutdownFailure is Exception shutdownFailure)
                failures.Add(shutdownFailure);
            try
            {
                worker.DisposeSignal();
            }
            catch (Exception exception)
            {
                failures.Add(exception);
            }
        }
    }

    private static void BindNativeDevice(int deviceIndex)
        => ForgetMemoryV2Cuda.GetAccelerator(deviceIndex).Bind();

    private static void UpdateMaximum(ref int location, int value)
    {
        int current = Volatile.Read(ref location);
        while (value > current)
        {
            int observed = Interlocked.CompareExchange(
                ref location, value, current);
            if (observed == current)
                return;
            current = observed;
        }
    }

    private sealed class Worker
    {
        private readonly CudaReplicaExecutor _owner;
        private readonly AutoResetEvent _workSignal = new(false);
        private readonly Thread _thread;
        private readonly int _deviceIndex;
        private ExecutionSession? _boundSession;
        private PrecisionPolicy? _boundPrecision;
        private IDisposable? _sessionScope;
        private IDisposable? _precisionScope;
        private IDisposable? _deviceScope;
        private int _stopRequested;

        internal Worker(
            CudaReplicaExecutor owner,
            int index,
            int deviceIndex,
            string name)
        {
            _owner = owner;
            Index = index;
            _deviceIndex = deviceIndex;
            _thread = new Thread(Run)
            {
                IsBackground = true,
                Name = name,
                Priority = ThreadPriority.AboveNormal,
            };
        }

        internal int Index { get; }
        internal int ThreadId;
        internal long ExecutionCount;
        internal long ContextBindingCount;
        internal Exception? ShutdownFailure { get; private set; }

        internal void Start()
        {
            if (ExecutionContext.IsFlowSuppressed())
            {
                _thread.Start();
                return;
            }
            using (ExecutionContext.SuppressFlow())
                _thread.Start();
        }

        internal void SignalWork() => _workSignal.Set();

        internal void RequestStop()
        {
            Volatile.Write(ref _stopRequested, 1);
            _workSignal.Set();
        }

        internal void Join() => _thread.Join();

        internal void DisposeSignal() => _workSignal.Dispose();

        internal void EnsureExecutionContext(
            ExecutionSession? session,
            PrecisionPolicy precision,
            Action<int> bindDevice)
        {
            bool sessionChanged = !ReferenceEquals(_boundSession, session);
            bool precisionChanged = !Equals(_boundPrecision, precision);
            if (!sessionChanged && !precisionChanged)
                return;

            DisposeContext();
            bindDevice(_deviceIndex);
            _boundSession = session;
            _boundPrecision = precision;
            try
            {
                _sessionScope = session?.Enter();
                _precisionScope =
                    TensorExecutionContext.PushPrecisionPolicy(precision);
                _deviceScope = TensorExecutionContext.Push(
                    new TorchDevice(TensorDevice.Cuda, _deviceIndex));
                Interlocked.Increment(ref ContextBindingCount);
            }
            catch
            {
                DisposeContext();
                throw;
            }
        }

        private void Run()
        {
            Volatile.Write(
                ref ThreadId, Environment.CurrentManagedThreadId);
            Interlocked.Increment(ref _owner._workerThreadCreationCount);
            Interlocked.Increment(ref _owner._liveWorkerCount);
            try
            {
                long observedGeneration = 0;
                bool completedWork = false;
                while (true)
                {
                    if (!TryObserveNextDispatch(
                            ref observedGeneration,
                            spinFirst: completedWork))
                    {
                        break;
                    }
                    if (Volatile.Read(ref _stopRequested) != 0)
                        break;
                    if (Index >= Volatile.Read(
                            ref _owner._activeWorkerCount))
                    {
                        continue;
                    }
                    try
                    {
                        _owner.ExecuteWorker(this);
                    }
                    catch (Exception exception)
                    {
                        _owner._workerFailures[Index] = exception;
                    }
                    finally
                    {
                        completedWork = true;
                        _owner._completion.Signal();
                    }
                }
            }
            finally
            {
                try
                {
                    DisposeContext();
                }
                catch (Exception exception)
                {
                    ShutdownFailure = exception;
                }
                Interlocked.Decrement(ref _owner._liveWorkerCount);
            }
        }

        private bool TryObserveNextDispatch(
            ref long observedGeneration,
            bool spinFirst)
        {
            if (spinFirst)
            {
                long started = System.Diagnostics.Stopwatch.GetTimestamp();
                while (System.Diagnostics.Stopwatch.GetElapsedTime(started)
                    < TimeSpan.FromMilliseconds(6))
                {
                    if (Volatile.Read(ref _stopRequested) != 0)
                        return false;
                    long generation = Interlocked.Read(
                        ref _owner._dispatchGeneration);
                    if (generation != observedGeneration)
                    {
                        // Consume the AutoResetEvent signal published for this
                        // generation so it cannot wake the following wait.
                        _workSignal.WaitOne(0);
                        observedGeneration = generation;
                        return true;
                    }
                    Thread.SpinWait(128);
                }
            }

            while (true)
            {
                _workSignal.WaitOne();
                if (Volatile.Read(ref _stopRequested) != 0)
                    return false;
                long generation = Interlocked.Read(
                    ref _owner._dispatchGeneration);
                if (generation == observedGeneration)
                    continue;
                observedGeneration = generation;
                return true;
            }
        }

        private void DisposeContext()
        {
            List<Exception>? failures = null;
            TryDispose(ref _deviceScope, ref failures);
            TryDispose(ref _precisionScope, ref failures);
            TryDispose(ref _sessionScope, ref failures);
            _boundSession = null;
            _boundPrecision = null;
            if (failures is [Exception failure])
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo
                    .Capture(failure)
                    .Throw();
            }
            if (failures is { Count: > 1 })
            {
                throw new AggregateException(
                    "A CUDA replica worker context failed to dispose.",
                    failures);
            }
        }

        private static void TryDispose(
            ref IDisposable? resource,
            ref List<Exception>? failures)
        {
            IDisposable? value = resource;
            resource = null;
            if (value is null)
                return;
            try
            {
                value.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
    }
}
