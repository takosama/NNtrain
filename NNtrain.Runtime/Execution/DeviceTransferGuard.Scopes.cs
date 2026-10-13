namespace NNtrain.Runtime.Execution;

public static partial class DeviceTransferGuard
{
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
