namespace NNtrain.Runtime.Execution;

/// <summary>Transfer counters observed inside one guarded training step.</summary>
public readonly record struct DeviceTransferSnapshot(
    long HostToDeviceCopyCount,
    long HostToDeviceBytes,
    long DeviceToHostCopyCount,
    long DeviceToHostBytes);

/// <summary>Explicit transport classes that may cross the host boundary.</summary>
public enum DeviceTransferTransportCategory
{
    GradientCollective = 0,
}

/// <summary>Physical transfer counters attributed to one CUDA device.</summary>
public readonly record struct DeviceTransferDeviceSnapshot(
    int DeviceIndex,
    long HostToDeviceCopyCount,
    long HostToDeviceBytes,
    long DeviceToHostCopyCount,
    long DeviceToHostBytes);

/// <summary>
/// Physical transfer counters for an explicitly authorized transport class.
/// These counters are separate from the four-field implicit/batch snapshot so
/// existing telemetry consumers retain their original contract.
/// </summary>
public sealed record DeviceTransferTransportSnapshot(
    DeviceTransferTransportCategory Category,
    DeviceTransferSnapshot Totals,
    IReadOnlyList<DeviceTransferDeviceSnapshot> Devices);

/// <summary>
/// Fails a CUDA training step before a non-scalar device-to-host transfer is
/// submitted. Batch H2D remains permitted and measured; D2H is restricted to
/// a device-count-dependent constant number of small metrics/status values.
/// </summary>
public static class DeviceTransferGuard
{
    private const nuint DefaultMaximumScalarBytes = 64;
    private static readonly AsyncLocal<Frame?> CurrentFrame = new();
    private static readonly AsyncLocal<HostToDeviceAuthorization?>
        CurrentHostToDeviceAuthorization = new();

    public static DeviceTransferSnapshot? CurrentSnapshot
        => FindActive(CurrentFrame.Value)?.Snapshot;

    public static DeviceTransferTransportSnapshot?
        GetCurrentTransportSnapshot(DeviceTransferTransportCategory category)
    {
        ValidateTransportCategory(category);
        return FindActive(CurrentFrame.Value)?.Budget
            .GetTransportSnapshot(category);
    }

    /// <summary>
    /// Explicit handle to one training-step transfer budget. This handle is
    /// safe to pass to dedicated workers whose <see cref="ExecutionContext"/>
    /// flow is intentionally suppressed. Attaching the handle never transfers
    /// ownership of the budget; only the scope returned by
    /// <see cref="EnterTrainingStep"/> can end it.
    /// </summary>
    public sealed class SharedContext
    {
        internal readonly Budget Budget;

        internal SharedContext(Budget budget)
        {
            Budget = budget;
        }

        public bool IsActive => Budget.IsActive;

        public DeviceTransferSnapshot Snapshot => Budget.Snapshot;

        public DeviceTransferTransportSnapshot GetTransportSnapshot(
            DeviceTransferTransportCategory category)
        {
            ValidateTransportCategory(category);
            return Budget.GetTransportSnapshot(category);
        }
    }

    /// <summary>
    /// Reservation for one successful native no-P2P BF16 host-pipeline call.
    /// A reservation is committed only after the native call reports success,
    /// so rejected or failed submissions do not inflate step telemetry.
    /// </summary>
    public sealed class GradientCollectiveTransportReservation
    {
        private readonly Budget _budget;
        private int _committed;

        internal GradientCollectiveTransportReservation(
            Budget budget,
            int sourceDeviceIndex,
            int destinationDeviceIndex,
            long copyCount,
            long byteLength)
        {
            _budget = budget;
            SourceDeviceIndex = sourceDeviceIndex;
            DestinationDeviceIndex = destinationDeviceIndex;
            CopyCount = copyCount;
            ByteLength = byteLength;
        }

        public int SourceDeviceIndex { get; }

        public int DestinationDeviceIndex { get; }

        public long CopyCount { get; }

        public long ByteLength { get; }

        public void Commit()
        {
            if (Interlocked.CompareExchange(ref _committed, 1, 0) != 0)
            {
                throw new InvalidOperationException(
                    "A gradient-collective transport reservation can only " +
                    "be committed once.");
            }
            _budget.RecordGradientCollectiveTransport(
                SourceDeviceIndex,
                DestinationDeviceIndex,
                CopyCount,
                ByteLength);
        }
    }

    /// <summary>
    /// Captures the active step budget for explicit propagation to a worker.
    /// Returns <see langword="null"/> outside a guarded training step.
    /// </summary>
    public static SharedContext? CaptureCurrentContext()
        => FindActive(CurrentFrame.Value)?.Context;

    /// <summary>
    /// Authorizes the physical D2H and H2D legs of one no-P2P gradient
    /// collective. The caller must commit the returned reservation only after
    /// the native pipeline successfully completes all planned chunks.
    /// </summary>
    public static GradientCollectiveTransportReservation?
        ReserveGradientCollectiveTransport(
            int sourceDeviceIndex,
            int destinationDeviceIndex,
            long copyCount,
            long byteLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(sourceDeviceIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(destinationDeviceIndex);
        if (sourceDeviceIndex == destinationDeviceIndex)
        {
            throw new ArgumentException(
                "Gradient-collective source and destination devices must " +
                "differ.",
                nameof(destinationDeviceIndex));
        }
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(copyCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(byteLength);

        Frame? frame = FindActive(CurrentFrame.Value);
        if (frame is null)
            return null;
        frame.Budget.ThrowIfDisposed();
        return new GradientCollectiveTransportReservation(
            frame.Budget,
            sourceDeviceIndex,
            destinationDeviceIndex,
            copyCount,
            byteLength);
    }

    /// <summary>
    /// Attaches a previously captured budget to the current execution flow.
    /// The returned scope only removes this attachment; it does not end the
    /// owner training step or affect attachments on other workers.
    /// </summary>
    public static IDisposable EnterSharedContext(SharedContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (!context.IsActive)
        {
            throw new InvalidOperationException(
                "A completed CUDA training-step transfer budget cannot be " +
                "attached to a worker.");
        }

        var frame = new Frame(
            context.Budget,
            context,
            CurrentFrame.Value,
            ownsBudget: false);
        CurrentFrame.Value = frame;
        return new Scope(frame);
    }

    public static IDisposable EnterTrainingStep(
        int cudaDeviceCount,
        nuint maximumScalarBytes = DefaultMaximumScalarBytes,
        int? maximumDeviceToHostCopies = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cudaDeviceCount);
        if (maximumScalarBytes == 0 || maximumScalarBytes > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(maximumScalarBytes));
        int maximumCopies = maximumDeviceToHostCopies
            ?? checked(8 + 8 * cudaDeviceCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maximumCopies);

        var budget = new Budget(
            maximumScalarBytes,
            maximumCopies);
        var context = new SharedContext(budget);
        var frame = new Frame(
            budget,
            context,
            CurrentFrame.Value,
            ownsBudget: true);
        CurrentFrame.Value = frame;
        return new Scope(frame);
    }

    /// <summary>
    /// Reserves one D2H operation before the native copy is submitted.
    /// A rejected copy never reaches CUDA.
    /// </summary>
    public static void BeforeDeviceToHost(nuint byteLength, string operation)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(operation);
        Frame? frame = FindActive(CurrentFrame.Value);
        if (frame is null)
            return;
        frame.Budget.ReserveDeviceToHost(byteLength, operation);
    }

    /// <summary>
    /// Fails before an H2D copy is submitted unless the caller has explicitly
    /// identified it as a batch input/target upload. Optimizer/model residency
    /// must be prepared before the training-step guard is entered.
    /// </summary>
    public static void BeforeHostToDevice(
        nuint byteLength,
        string operation)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(operation);
        Frame? frame = FindActive(CurrentFrame.Value);
        if (frame is null)
            return;
        if (byteLength == 0)
            return;
        frame.Budget.ThrowIfDisposed();

        HostToDeviceAuthorization? authorization = FindActive(
            CurrentHostToDeviceAuthorization.Value);
        if (authorization is null
            || !ReferenceEquals(authorization.Frame, frame))
        {
            throw new InvalidOperationException(
                $"CUDA training attempted an unclassified H2D transfer of " +
                $"{byteLength:N0} bytes during '{operation}'. Only explicit " +
                "batch input/target uploads are permitted after the " +
                "training-step guard begins; prewarm model and optimizer " +
                "residency first.");
        }
    }

    /// <summary>
    /// Authorizes the narrow scope which uploads one batch's dynamic inputs
    /// or targets. The scope is bound to the current guard frame, so it cannot
    /// leak permission into a nested or later training step.
    /// </summary>
    public static IDisposable AllowBatchHostToDevice()
    {
        Frame? frame = FindActive(CurrentFrame.Value);
        var authorization = new HostToDeviceAuthorization(
            frame,
            CurrentHostToDeviceAuthorization.Value);
        CurrentHostToDeviceAuthorization.Value = authorization;
        return new HostToDeviceAuthorizationScope(authorization);
    }

    public static void RecordHostToDevice(nuint byteLength)
    {
        Frame? frame = FindActive(CurrentFrame.Value);
        frame?.Budget.RecordHostToDevice(byteLength);
    }

    private static Frame? FindActive(Frame? frame)
    {
        while (frame is not null && frame.IsDisposed)
            frame = frame.Previous;
        return frame;
    }

    private static HostToDeviceAuthorization? FindActive(
        HostToDeviceAuthorization? authorization)
    {
        while (authorization is not null && authorization.IsDisposed)
            authorization = authorization.Previous;
        return authorization;
    }

    private sealed class Frame(
        Budget budget,
        SharedContext context,
        Frame? previous,
        bool ownsBudget)
    {
        private int _disposed;

        internal Budget Budget { get; } = budget;
        internal SharedContext Context { get; } = context;
        internal Frame? Previous { get; } = previous;
        internal bool IsDisposed => Volatile.Read(ref _disposed) != 0;
        internal DeviceTransferSnapshot Snapshot => Budget.Snapshot;

        internal void MarkDisposed()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0
                && ownsBudget)
            {
                Budget.MarkDisposed();
            }
        }
    }

    internal sealed class Budget(
        nuint maximumScalarBytes,
        int maximumDeviceToHostCopies)
    {
        private readonly TransportCounters _gradientCollective = new();
        private long _hostToDeviceCopyCount;
        private long _hostToDeviceBytes;
        private long _deviceToHostCopyCount;
        private long _deviceToHostBytes;
        private int _disposed;

        internal bool IsActive => Volatile.Read(ref _disposed) == 0;
        internal DeviceTransferSnapshot Snapshot => new(
            Interlocked.Read(ref _hostToDeviceCopyCount),
            Interlocked.Read(ref _hostToDeviceBytes),
            Interlocked.Read(ref _deviceToHostCopyCount),
            Interlocked.Read(ref _deviceToHostBytes));

        internal void ReserveDeviceToHost(
            nuint byteLength,
            string operation)
        {
            ThrowIfDisposed();
            if (byteLength > maximumScalarBytes)
            {
                throw new InvalidOperationException(
                    $"CUDA training attempted an implicit D2H transfer of " +
                    $"{byteLength:N0} bytes during '{operation}'. Only " +
                    $"scalar/status transfers up to " +
                    $"{maximumScalarBytes:N0} bytes are permitted in a step.");
            }

            long copyCount;
            while (true)
            {
                long current = Interlocked.Read(
                    ref _deviceToHostCopyCount);
                if (current >= maximumDeviceToHostCopies)
                {
                    throw new InvalidOperationException(
                        $"CUDA training exceeded its constant D2H transfer " +
                        $"budget during '{operation}': {current + 1} copies, " +
                        $"maximum {maximumDeviceToHostCopies}. Aggregate " +
                        "metrics on the device instead of copying per tensor " +
                        "or shard.");
                }
                copyCount = current + 1;
                if (Interlocked.CompareExchange(
                        ref _deviceToHostCopyCount,
                        copyCount,
                        current) == current)
                {
                    break;
                }
            }
            Interlocked.Add(
                ref _deviceToHostBytes,
                checked((long)byteLength));
        }

        internal void RecordHostToDevice(nuint byteLength)
        {
            ThrowIfDisposed();
            long bytes = checked((long)byteLength);
            Interlocked.Increment(ref _hostToDeviceCopyCount);
            Interlocked.Add(
                ref _hostToDeviceBytes,
                bytes);
        }

        internal void RecordGradientCollectiveTransport(
            int sourceDeviceIndex,
            int destinationDeviceIndex,
            long copyCount,
            long byteLength)
        {
            ThrowIfDisposed();
            _gradientCollective.RecordRoundTrip(
                sourceDeviceIndex,
                destinationDeviceIndex,
                copyCount,
                byteLength);
        }

        internal DeviceTransferTransportSnapshot GetTransportSnapshot(
            DeviceTransferTransportCategory category)
            => category switch
            {
                DeviceTransferTransportCategory.GradientCollective =>
                    _gradientCollective.Snapshot(category),
                _ => throw new ArgumentOutOfRangeException(
                    nameof(category), category, "Unknown transport category."),
            };

        internal void ThrowIfDisposed()
        {
            if (!IsActive)
            {
                throw new InvalidOperationException(
                    "The CUDA training-step transfer budget has already " +
                    "completed.");
            }
        }

        internal void MarkDisposed()
            => Interlocked.Exchange(ref _disposed, 1);
    }

    internal sealed class TransportCounters
    {
        private readonly object _sync = new();
        private readonly Dictionary<int, MutableDeviceCounters> _devices = [];
        private long _hostToDeviceCopyCount;
        private long _hostToDeviceBytes;
        private long _deviceToHostCopyCount;
        private long _deviceToHostBytes;

        internal void RecordRoundTrip(
            int sourceDeviceIndex,
            int destinationDeviceIndex,
            long copyCount,
            long byteLength)
        {
            lock (_sync)
            {
                checked
                {
                    _deviceToHostCopyCount += copyCount;
                    _deviceToHostBytes += byteLength;
                    _hostToDeviceCopyCount += copyCount;
                    _hostToDeviceBytes += byteLength;

                    MutableDeviceCounters source = GetDevice(
                        sourceDeviceIndex);
                    source.DeviceToHostCopyCount += copyCount;
                    source.DeviceToHostBytes += byteLength;
                    MutableDeviceCounters destination = GetDevice(
                        destinationDeviceIndex);
                    destination.HostToDeviceCopyCount += copyCount;
                    destination.HostToDeviceBytes += byteLength;
                }
            }
        }

        internal DeviceTransferTransportSnapshot Snapshot(
            DeviceTransferTransportCategory category)
        {
            lock (_sync)
            {
                DeviceTransferDeviceSnapshot[] devices = _devices
                    .OrderBy(static pair => pair.Key)
                    .Select(static pair => pair.Value.Snapshot(pair.Key))
                    .ToArray();
                return new DeviceTransferTransportSnapshot(
                    category,
                    new DeviceTransferSnapshot(
                        _hostToDeviceCopyCount,
                        _hostToDeviceBytes,
                        _deviceToHostCopyCount,
                        _deviceToHostBytes),
                    Array.AsReadOnly(devices));
            }
        }

        private MutableDeviceCounters GetDevice(int deviceIndex)
        {
            if (!_devices.TryGetValue(
                    deviceIndex,
                    out MutableDeviceCounters? counters))
            {
                counters = new MutableDeviceCounters();
                _devices.Add(deviceIndex, counters);
            }
            return counters;
        }

        private sealed class MutableDeviceCounters
        {
            internal long HostToDeviceCopyCount;
            internal long HostToDeviceBytes;
            internal long DeviceToHostCopyCount;
            internal long DeviceToHostBytes;

            internal DeviceTransferDeviceSnapshot Snapshot(int deviceIndex)
                => new(
                    deviceIndex,
                    HostToDeviceCopyCount,
                    HostToDeviceBytes,
                    DeviceToHostCopyCount,
                    DeviceToHostBytes);
        }
    }

    private static void ValidateTransportCategory(
        DeviceTransferTransportCategory category)
    {
        if (!Enum.IsDefined(category))
        {
            throw new ArgumentOutOfRangeException(
                nameof(category), category, "Unknown transport category.");
        }
    }

    private sealed class Scope(Frame frame) : IDisposable
    {
        private Frame? _frame = frame;

        public void Dispose()
        {
            Frame? value = Interlocked.Exchange(ref _frame, null);
            if (value is null)
                return;
            value.MarkDisposed();
            if (ReferenceEquals(CurrentFrame.Value, value))
                CurrentFrame.Value = FindActive(value.Previous);
        }
    }

    private sealed class HostToDeviceAuthorization(
        Frame? frame,
        HostToDeviceAuthorization? previous)
    {
        private int _disposed;

        internal Frame? Frame { get; } = frame;
        internal HostToDeviceAuthorization? Previous { get; } = previous;
        internal bool IsDisposed => Volatile.Read(ref _disposed) != 0;

        internal void MarkDisposed()
            => Interlocked.Exchange(ref _disposed, 1);
    }

    private sealed class HostToDeviceAuthorizationScope(
        HostToDeviceAuthorization authorization) : IDisposable
    {
        private HostToDeviceAuthorization? _authorization = authorization;

        public void Dispose()
        {
            HostToDeviceAuthorization? value = Interlocked.Exchange(
                ref _authorization,
                null);
            if (value is null)
                return;
            value.MarkDisposed();
            if (ReferenceEquals(
                    CurrentHostToDeviceAuthorization.Value,
                    value))
            {
                CurrentHostToDeviceAuthorization.Value = FindActive(
                    value.Previous);
            }
        }
    }
}
