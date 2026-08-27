using NNtrain.Runtime.Execution;

namespace NNtrain;

internal static class TensorExecutionContext
{
    private static readonly AsyncLocal<State?> CurrentState = new();
    private static readonly AsyncLocal<Scope?> CurrentScope = new();

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
            State current = StateValue;
            // Selecting the CUDA replica set must not opt a CPU execution
            // context into CUDA.  Preserve the historical convention that,
            // while CUDA is already selected, the first replica is the
            // default device for non-partitioned kernels.
            TorchDevice computeDevice = current.Device.IsCuda
                ? new TorchDevice(TensorDevice.Cuda, indices[0])
                : current.Device;
            CurrentState.Value = current with
            {
                Device = computeDevice,
                CudaDevices = indices,
            };
        }
    }

    /// <summary>
    /// Gets the model-level numeric contract for the current asynchronous
    /// execution flow. A missing policy preserves the legacy tensor-dtype
    /// behavior used by direct Tensor APIs.
    /// </summary>
    internal static PrecisionPolicy? ActivePrecisionPolicy
        => StateValue.PrecisionPolicy;

    internal static IDisposable Push(TorchDevice device)
    {
        State? previous = CurrentState.Value;
        var scope = new Scope(previous, CurrentScope.Value);
        CurrentState.Value = StateValue with { Device = device };
        CurrentScope.Value = scope;
        return scope;
    }

    internal static IDisposable PushPrecisionPolicy(
        PrecisionPolicy precisionPolicy)
    {
        ArgumentNullException.ThrowIfNull(precisionPolicy);
        State? previous = CurrentState.Value;
        var scope = new Scope(previous, CurrentScope.Value);
        CurrentState.Value = StateValue with
        {
            PrecisionPolicy = precisionPolicy,
        };
        CurrentScope.Value = scope;
        return scope;
    }

    private static State StateValue
        => CurrentState.Value ?? new State(
            new TorchDevice(TensorDevice.Cpu),
            [0],
            PrecisionPolicy: null);

    private sealed record State(
        TorchDevice Device,
        int[] CudaDevices,
        PrecisionPolicy? PrecisionPolicy);

    private sealed class Scope(
        State? previousState,
        Scope? previousScope) : IDisposable
    {
        private int _disposed;

        public void Dispose()
        {
            if (Volatile.Read(ref _disposed) != 0)
                return;
            if (!ReferenceEquals(CurrentScope.Value, this))
            {
                throw new InvalidOperationException(
                    "Tensor execution scopes must be disposed in LIFO order.");
            }
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            CurrentState.Value = previousState;
            CurrentScope.Value = previousScope;
        }
    }
}
