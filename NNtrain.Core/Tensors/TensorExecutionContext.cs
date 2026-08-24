namespace NNtrain;

internal static class TensorExecutionContext
{
    private static readonly AsyncLocal<State?> CurrentState = new();

    internal static TorchDevice Device
    {
        get => StateValue.Device;
        set => CurrentState.Value = StateValue with { Device = value };
    }

    internal static IReadOnlyList<int> CudaDevices
    {
        get => Array.AsReadOnly(StateValue.CudaDevices);
        set
        {
            ArgumentNullException.ThrowIfNull(value);
            if (value.Count == 0)
                throw new ArgumentException("At least one CUDA device is required.");
            int[] indices = value.ToArray();
            if (indices.Any(index => index < 0)
                || indices.Distinct().Count() != indices.Length)
            {
                throw new ArgumentException(
                    "CUDA device indices must be unique and non-negative.");
            }
            CurrentState.Value = new State(
                new TorchDevice(TensorDevice.Cuda, indices[0]),
                indices);
        }
    }

    internal static IDisposable Push(TorchDevice device)
    {
        State? previous = CurrentState.Value;
        CurrentState.Value = new State(
            device,
            device.IsCuda ? [device.Index] : StateValue.CudaDevices);
        return new Scope(previous);
    }

    private static State StateValue
        => CurrentState.Value ?? new State(
            new TorchDevice(TensorDevice.Cpu),
            [0]);

    private sealed record State(TorchDevice Device, int[] CudaDevices);

    private sealed class Scope(State? previous) : IDisposable
    {
        private State? _previous = previous;

        public void Dispose()
        {
            State? value = Interlocked.Exchange(ref _previous, null);
            CurrentState.Value = value;
        }
    }
}
