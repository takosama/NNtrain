using NNtrain.Runtime.Execution;

namespace NNtrain;

internal static class TensorExecutionContext
{
    private static readonly AsyncLocal<State?> LegacyState = new();
    private static readonly AsyncLocal<ScopeFrame?> CurrentFrame = new();

    internal static TorchDevice Device
    {
        get => EffectiveState.Device;
        set => SetAmbientState(AmbientState with { Device = value });
    }

    internal static IReadOnlyList<int> CudaDevices
    {
        get => Array.AsReadOnly(EffectiveState.CudaDevices);
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
            State current = AmbientState;
            // Configuring replicas never opts a CPU context into CUDA. During
            // a session these legacy writes cannot replace its authority.
            TorchDevice computeDevice = current.Device.IsCuda
                ? new TorchDevice(TensorDevice.Cuda, indices[0])
                : current.Device;
            SetAmbientState(current with
            {
                Device = computeDevice,
                CudaDevices = indices,
            });
        }
    }

    /// <summary>
    /// Gets the model-level numeric contract for the current asynchronous
    /// execution flow. A missing policy preserves the legacy tensor-dtype
    /// behavior used by direct Tensor APIs.
    /// </summary>
    internal static PrecisionPolicy? ActivePrecisionPolicy
        => EffectiveState.PrecisionPolicy;

    /// <summary>
    /// True when backward values are physically stored as BF16. Direct
    /// Tensor calls preserve the historical all-BF16 contract; an active
    /// model policy makes the distinction explicit so pure bfloat16 does not
    /// accidentally inherit mix16_32's FP32-gradient semantics.
    /// </summary>
    internal static bool UsesBFloat16GradientStorage
        => ActivePrecisionPolicy?.Gradient is null or NumericFormat.BFloat16;

    /// <summary>
    /// Gets the immutable CUDA dispatch switches installed for this
    /// asynchronous execution flow. A missing override uses the startup
    /// environment snapshot in <see cref="CudaDispatchPolicy"/>.
    /// </summary>
    internal static CudaDispatchPolicy? ActiveCudaDispatchPolicy
        // Dispatch policy is independent of the selected device lane. Reading
        // AmbientState directly avoids materializing the session device list
        // for every GEMM/attention/kernel dispatch.
        => AmbientState.CudaDispatchPolicy;

    internal static IDisposable Push(TorchDevice device)
    {
        ExecutionSession? session = ExecutionSession.Current;
        if (device.IsCuda
            && session is not null
            && session.Options.Device == ExecutionDeviceKind.Cuda
            && !session.Options.CudaDevices.Contains(device.Index))
        {
            throw new InvalidOperationException(
                $"CUDA device {device.Index} is not part of the active execution session.");
        }

        var frame = new ScopeFrame(
            AmbientState with { Device = device },
            CurrentFrame.Value);
        CurrentFrame.Value = frame;
        try
        {
            ActivateEffectiveSessionLane();
            return new Scope(frame);
        }
        catch
        {
            frame.MarkDisposed();
            CurrentFrame.Value = FindActive(frame.Previous);
            TryRestoreEffectiveSessionLane();
            throw;
        }
    }

    internal static IDisposable PushPrecisionPolicy(
        PrecisionPolicy precisionPolicy)
    {
        ArgumentNullException.ThrowIfNull(precisionPolicy);
        var frame = new ScopeFrame(
            AmbientState with { PrecisionPolicy = precisionPolicy },
            CurrentFrame.Value);
        CurrentFrame.Value = frame;
        return new Scope(frame);
    }

    internal static IDisposable PushCudaDispatchPolicy(
        CudaDispatchPolicy cudaDispatchPolicy)
    {
        ArgumentNullException.ThrowIfNull(cudaDispatchPolicy);
        var frame = new ScopeFrame(
            AmbientState with
            {
                CudaDispatchPolicy = cudaDispatchPolicy,
            },
            CurrentFrame.Value);
        CurrentFrame.Value = frame;
        return new Scope(frame);
    }

    internal static bool TryGetCudaStreamLane(
        int deviceIndex,
        out IStreamExecutionLane streamLane)
    {
        ExecutionSession? session = ExecutionSession.Current;
        if (session is null
            || session.Options.Device != ExecutionDeviceKind.Cuda
            || !session.Options.CudaDevices.Contains(deviceIndex)
            || !session.TryGetLane(
                ExecutionDeviceKind.Cuda,
                deviceIndex,
                out IExecutionLane? lane)
            || lane is not IStreamExecutionLane value)
        {
            streamLane = null!;
            return false;
        }
        streamLane = value;
        return true;
    }

    internal static bool TryGetActiveCudaStreamLane(
        out IStreamExecutionLane streamLane)
    {
        TorchDevice device = EffectiveState.Device;
        if (!device.IsCuda)
        {
            streamLane = null!;
            return false;
        }
        return TryGetCudaStreamLane(device.Index, out streamLane);
    }

    private static State EffectiveState
    {
        get
        {
            State ambient = AmbientState;
            ExecutionSession? session = ExecutionSession.Current;
            if (session is null)
                return ambient;

            ExecutionOptions options = session.Options;
            int[] cudaDevices = options.CudaDevices.ToArray();
            if (options.Device == ExecutionDeviceKind.Cpu)
            {
                return ambient with
                {
                    Device = new TorchDevice(TensorDevice.Cpu),
                    CudaDevices = cudaDevices,
                    PrecisionPolicy = options.Precision,
                };
            }

            int deviceIndex = ambient.Device.IsCuda
                && options.CudaDevices.Contains(ambient.Device.Index)
                    ? ambient.Device.Index
                    : options.CudaDevices[0];
            return ambient with
            {
                Device = new TorchDevice(TensorDevice.Cuda, deviceIndex),
                CudaDevices = cudaDevices,
                PrecisionPolicy = options.Precision,
            };
        }
    }

    private static State AmbientState
    {
        get
        {
            ScopeFrame? frame = FindActive(CurrentFrame.Value);
            if (!ReferenceEquals(frame, CurrentFrame.Value))
                CurrentFrame.Value = frame;
            return frame?.State
                ?? LegacyState.Value
                ?? State.Default;
        }
    }

    private static void SetAmbientState(State state)
    {
        ScopeFrame? frame = FindActive(CurrentFrame.Value);
        if (!ReferenceEquals(frame, CurrentFrame.Value))
            CurrentFrame.Value = frame;
        if (frame is null)
            LegacyState.Value = state;
        else
            frame.State = state;
    }

    private static void ActivateEffectiveSessionLane()
    {
        if (TryGetActiveCudaStreamLane(out IStreamExecutionLane lane))
            NativeCudaRuntime.BindExecutionLane(lane);
    }

    private static void TryRestoreEffectiveSessionLane()
    {
        try
        {
            ActivateEffectiveSessionLane();
        }
        catch
        {
            // Preserve the activation failure which caused scope entry to fail.
        }
    }

    private static ScopeFrame? FindActive(ScopeFrame? frame)
    {
        while (frame is not null && frame.IsDisposed)
            frame = frame.Previous;
        return frame;
    }

    private sealed record State(
        TorchDevice Device,
        int[] CudaDevices,
        PrecisionPolicy? PrecisionPolicy,
        CudaDispatchPolicy? CudaDispatchPolicy)
    {
        internal static State Default { get; } = new(
            new TorchDevice(TensorDevice.Cpu),
            [0],
            PrecisionPolicy: null,
            CudaDispatchPolicy: null);
    }

    private sealed class ScopeFrame(State state, ScopeFrame? previous)
    {
        private int _disposed;
        internal State State { get; set; } = state;
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
            ActivateEffectiveSessionLane();
        }
    }
}
