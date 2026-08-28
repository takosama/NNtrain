namespace NNtrain.Runtime.Execution;

/// <summary>
/// Owns the resources and ambient scope for one explicit execution context.
/// Nested and out-of-order scope disposal is safe and idempotent.
/// </summary>
public sealed class ExecutionSession : IDisposable
{
    private static readonly AsyncLocal<ScopeFrame?> CurrentFrame = new();
    private static long _nextGeneration;

    private readonly object _sync = new();
    private readonly Dictionary<(ExecutionDeviceKind Kind, int Index), IExecutionLane>
        _lanes = [];
    private readonly List<IDisposable> _ownedResources = [];
    private readonly Dictionary<long, BeforeDisposeCallback> _beforeDispose = [];
    private long _nextBeforeDisposeRegistration;
    private int _disposed;

    public ExecutionSession(
        ExecutionOptions options,
        IEnumerable<IExecutionLane>? lanes = null)
    {
        Generation = Interlocked.Increment(ref _nextGeneration);
        Options = (options ?? throw new ArgumentNullException(nameof(options)))
            .Validate();
        if (lanes is null)
            return;
        foreach (IExecutionLane lane in lanes)
            AttachLane(lane);
    }

    public ExecutionOptions Options { get; }

    /// <summary>
    /// Monotonically increasing identity for resources whose validity is
    /// bounded by this execution session. It is never reused in this process.
    /// </summary>
    public long Generation { get; }

    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    public static ExecutionSession? Current
    {
        get
        {
            ScopeFrame? frame = FindActive(CurrentFrame.Value);
            if (!ReferenceEquals(frame, CurrentFrame.Value))
                CurrentFrame.Value = frame;
            return frame?.Session;
        }
    }

    public IReadOnlyList<IExecutionLane> Lanes
    {
        get
        {
            lock (_sync)
                return _lanes.Values.ToArray();
        }
    }

    public void AttachLane(IExecutionLane lane)
    {
        ArgumentNullException.ThrowIfNull(lane);
        ThrowIfDisposed();
        if (lane.DeviceIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(lane));
        if (lane.DeviceKind == ExecutionDeviceKind.Cuda
            && !Options.CudaDevices.Contains(lane.DeviceIndex))
        {
            throw new ArgumentException(
                $"CUDA device {lane.DeviceIndex} is not part of this session's device set.",
                nameof(lane));
        }

        lock (_sync)
        {
            ThrowIfDisposed();
            var key = (lane.DeviceKind, lane.DeviceIndex);
            if (!_lanes.TryAdd(key, lane))
                throw new InvalidOperationException(
                    $"A {lane.DeviceKind}:{lane.DeviceIndex} lane is already attached.");
            _ownedResources.Add(lane);
        }
    }

    public IExecutionLane GetRequiredLane(
        ExecutionDeviceKind deviceKind,
        int deviceIndex)
    {
        ThrowIfDisposed();
        lock (_sync)
        {
            if (_lanes.TryGetValue((deviceKind, deviceIndex), out IExecutionLane? lane))
                return lane;
        }
        throw new InvalidOperationException(
            $"No execution lane is attached for {deviceKind}:{deviceIndex}.");
    }

    public bool TryGetLane(
        ExecutionDeviceKind deviceKind,
        int deviceIndex,
        out IExecutionLane? lane)
    {
        if (IsDisposed)
        {
            lane = null;
            return false;
        }
        lock (_sync)
            return _lanes.TryGetValue((deviceKind, deviceIndex), out lane);
    }

    public T Own<T>(T resource)
        where T : class, IDisposable
    {
        ArgumentNullException.ThrowIfNull(resource);
        lock (_sync)
        {
            ThrowIfDisposed();
            _ownedResources.Add(resource);
        }
        return resource;
    }

    /// <summary>
    /// Registers work which must run before any lane or memory manager owned
    /// by this session is released. The returned registration may be disposed
    /// when the participant no longer owns session-bounded resources.
    /// </summary>
    /// <remarks>
    /// This is primarily used by storage owners that need to preserve an
    /// authoritative device value before the lane closes its memory leases.
    /// Every callback is attempted even when an earlier callback fails.
    /// </remarks>
    public IDisposable RegisterBeforeDispose(Action callback)
    {
        ArgumentNullException.ThrowIfNull(callback);
        lock (_sync)
        {
            ThrowIfDisposed();
            long registrationId = checked(++_nextBeforeDisposeRegistration);
            _beforeDispose.Add(
                registrationId,
                BeforeDisposeCallback.Strong(callback));
            return new BeforeDisposeRegistration(this, registrationId);
        }
    }

    /// <summary>
    /// Registers a pre-disposal callback without making the session retain
    /// the participant. Dead transient owners are compacted as registrations
    /// are added and are ignored when the session ends.
    /// </summary>
    public IDisposable RegisterBeforeDispose(
        object owner,
        Action<object> callback)
    {
        ArgumentNullException.ThrowIfNull(owner);
        ArgumentNullException.ThrowIfNull(callback);
        lock (_sync)
        {
            ThrowIfDisposed();
            long registrationId = checked(++_nextBeforeDisposeRegistration);
            _beforeDispose.Add(
                registrationId,
                BeforeDisposeCallback.Weak(owner, callback));
            if ((registrationId & 63) == 0)
            {
                foreach (long stale in _beforeDispose
                    .Where(static pair => !pair.Value.IsAlive)
                    .Select(static pair => pair.Key)
                    .ToArray())
                {
                    _beforeDispose.Remove(stale);
                }
            }
            return new BeforeDisposeRegistration(this, registrationId);
        }
    }

    public IDisposable Enter()
    {
        ThrowIfDisposed();
        var frame = new ScopeFrame(this, CurrentFrame.Value);
        CurrentFrame.Value = frame;
        try
        {
            ActivateDefaultStreamLane();
            return new Scope(frame);
        }
        catch
        {
            frame.MarkDisposed();
            CurrentFrame.Value = FindActive(frame.Previous);
            TryActivateCurrentWithoutMaskingFailure();
            throw;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        BeforeDisposeCallback[] beforeDispose;
        IDisposable[] resources;
        IStreamExecutionLane[] streamLanes;
        lock (_sync)
        {
            beforeDispose = _beforeDispose.Values.ToArray();
            _beforeDispose.Clear();
            streamLanes = _lanes.Values
                .OfType<IStreamExecutionLane>()
                .ToArray();
            resources = _ownedResources.ToArray();
            _ownedResources.Clear();
            _lanes.Clear();
        }

        List<Exception>? failures = null;
        // A retirement callback may copy an authoritative value from device
        // storage. Complete both producer streams before any callback reads
        // those bytes; lane disposal happens later and must not be the first
        // synchronization point.
        foreach (IStreamExecutionLane lane in streamLanes)
        {
            try
            {
                lane.SynchronizeCommunicationStream();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
            try
            {
                lane.SynchronizeComputeStream();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        for (int index = beforeDispose.Length - 1; index >= 0; index--)
        {
            try
            {
                beforeDispose[index].Invoke();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        for (int index = resources.Length - 1; index >= 0; index--)
        {
            try
            {
                resources[index].Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }

        ScopeFrame? current = CurrentFrame.Value;
        ScopeFrame? active = FindActive(current);
        if (!ReferenceEquals(current, active))
        {
            CurrentFrame.Value = active;
            TryActivateCurrent(ref failures);
        }
        if (failures is not null)
            throw new AggregateException(
                "One or more execution resources failed to dispose.",
                failures);
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(IsDisposed, this);

    private void RemoveBeforeDisposeRegistration(long registrationId)
    {
        lock (_sync)
            _beforeDispose.Remove(registrationId);
    }

    private void ActivateDefaultStreamLane()
    {
        if (Options.Device != ExecutionDeviceKind.Cuda)
            return;
        int deviceIndex = Options.CudaDevices[0];
        if (TryGetLane(
                ExecutionDeviceKind.Cuda,
                deviceIndex,
                out IExecutionLane? lane)
            && lane is IStreamExecutionLane streamLane)
        {
            streamLane.ActivateComputeStream();
        }
    }

    private static void TryActivateCurrentWithoutMaskingFailure()
    {
        try
        {
            Current?.ActivateDefaultStreamLane();
        }
        catch
        {
            // The activation which caused Enter to fail remains authoritative.
        }
    }

    private static void TryActivateCurrent(ref List<Exception>? failures)
    {
        try
        {
            Current?.ActivateDefaultStreamLane();
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
    }

    private static ScopeFrame? FindActive(ScopeFrame? frame)
    {
        while (frame is not null
            && (frame.IsDisposed || frame.Session.IsDisposed))
        {
            frame = frame.Previous;
        }
        return frame;
    }

    private sealed class ScopeFrame(
        ExecutionSession session,
        ScopeFrame? previous)
    {
        private int _disposed;
        internal ExecutionSession Session { get; } = session;
        internal ScopeFrame? Previous { get; } = previous;
        internal bool IsDisposed => Volatile.Read(ref _disposed) != 0;
        internal void MarkDisposed() => Interlocked.Exchange(ref _disposed, 1);
    }

    private sealed class Scope(ScopeFrame frame) : IDisposable
    {
        private ScopeFrame? _frame = frame;

        public void Dispose()
        {
            ScopeFrame? value = Interlocked.Exchange(ref _frame, null);
            if (value is null)
                return;
            value.MarkDisposed();
            if (ReferenceEquals(CurrentFrame.Value, value))
            {
                CurrentFrame.Value = FindActive(value.Previous);
                ExecutionSession? active = CurrentFrame.Value?.Session;
                active?.ActivateDefaultStreamLane();
            }
        }
    }

    private sealed class BeforeDisposeRegistration(
        ExecutionSession session,
        long registrationId) : IDisposable
    {
        private ExecutionSession? _session = session;

        public void Dispose()
        {
            ExecutionSession? value = Interlocked.Exchange(
                ref _session,
                null);
            value?.RemoveBeforeDisposeRegistration(registrationId);
        }
    }

    private sealed class BeforeDisposeCallback
    {
        private readonly Action? _strongCallback;
        private readonly WeakReference<object>? _weakOwner;
        private readonly Action<object>? _weakCallback;

        private BeforeDisposeCallback(Action strongCallback)
            => _strongCallback = strongCallback;

        private BeforeDisposeCallback(
            object owner,
            Action<object> weakCallback)
        {
            _weakOwner = new WeakReference<object>(owner);
            _weakCallback = weakCallback;
        }

        internal bool IsAlive => _strongCallback is not null
            || (_weakOwner?.TryGetTarget(out _) ?? false);

        internal static BeforeDisposeCallback Strong(Action callback)
            => new(callback);

        internal static BeforeDisposeCallback Weak(
            object owner,
            Action<object> callback)
            => new(owner, callback);

        internal void Invoke()
        {
            if (_strongCallback is not null)
            {
                _strongCallback();
                return;
            }
            if (_weakOwner!.TryGetTarget(out object? owner))
                _weakCallback!(owner);
        }
    }
}
