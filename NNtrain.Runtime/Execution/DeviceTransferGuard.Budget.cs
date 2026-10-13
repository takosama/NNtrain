namespace NNtrain.Runtime.Execution;

public static partial class DeviceTransferGuard
{
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
}
