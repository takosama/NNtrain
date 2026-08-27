namespace NNtrain.Runtime.Execution;

/// <summary>
/// Owns the resources and ambient scope for one explicit execution context.
/// Nested and out-of-order scope disposal is safe and idempotent.
/// </summary>
public sealed class ExecutionSession : IDisposable
{
    private static readonly AsyncLocal<ScopeFrame?> CurrentFrame = new();

    private readonly object _sync = new();
    private readonly Dictionary<(ExecutionDeviceKind Kind, int Index), IExecutionLane>
        _lanes = [];
    private readonly List<IDisposable> _ownedResources = [];
    private int _disposed;

    public ExecutionSession(
        ExecutionOptions options,
        IEnumerable<IExecutionLane>? lanes = null)
    {
        Options = (options ?? throw new ArgumentNullException(nameof(options)))
            .Validate();
        if (lanes is null)
            return;
        foreach (IExecutionLane lane in lanes)
            AttachLane(lane);
    }

    public ExecutionOptions Options { get; }

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

    public IDisposable Enter()
    {
        ThrowIfDisposed();
        var frame = new ScopeFrame(this, CurrentFrame.Value);
        CurrentFrame.Value = frame;
        return new Scope(frame);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        IDisposable[] resources;
        lock (_sync)
        {
            resources = _ownedResources.ToArray();
            _ownedResources.Clear();
            _lanes.Clear();
        }

        List<Exception>? failures = null;
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

        _ = Current;
        if (failures is not null)
            throw new AggregateException(
                "One or more execution resources failed to dispose.",
                failures);
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(IsDisposed, this);

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
                CurrentFrame.Value = FindActive(value.Previous);
        }
    }
}
