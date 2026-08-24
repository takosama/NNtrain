namespace NNtrain;

/// <summary>
/// Owns transient CUDA tensors created by one no-grad inference step.
/// Parameter buffers are not tracked because they are not operation results.
/// </summary>
internal sealed class CudaInferenceScope : IDisposable
{
    private static readonly AsyncLocal<CudaInferenceScope?> Current = new();
    private readonly CudaInferenceScope? _previous;
    private readonly List<Tensor> _tensors = [];
    private readonly List<IDisposable> _resources = [];
    private readonly HashSet<int> _devices = [];
    private readonly int? _poolDeviceToClear;
    private readonly bool _clearPoolOnDispose;
    private int _disposed;

    private CudaInferenceScope(bool resetPool, bool clearPoolOnDispose)
    {
        _previous = Current.Value;
        Current.Value = this;
        _clearPoolOnDispose = clearPoolOnDispose;
        if (Tensor.ExecutionDevice == TensorDevice.Cuda
            && (resetPool || clearPoolOnDispose))
        {
            _poolDeviceToClear = Tensor.CudaDeviceIndex;
            if (resetPool)
                Tensor.ClearCudaFloatBufferPool(_poolDeviceToClear.Value);
        }
    }

    internal static CudaInferenceScope Begin(
        bool resetPool = false,
        bool clearPoolOnDispose = false)
        => new(resetPool, clearPoolOnDispose);

    internal static void Track(Tensor tensor, int deviceIndex)
    {
        CudaInferenceScope? scope = Current.Value;
        if (scope is null)
            return;
        scope._tensors.Add(tensor);
        scope._devices.Add(deviceIndex);
    }

    internal static bool TrackResource(IDisposable resource)
    {
        CudaInferenceScope? scope = Current.Value;
        if (scope is null)
            return false;
        scope._resources.Add(resource);
        return true;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        try
        {
            foreach (int device in _devices)
                ForgetMemoryV2Cuda.GetAccelerator(device).Synchronize();
            foreach (IDisposable resource in _resources)
                resource.Dispose();
            foreach (Tensor tensor in _tensors)
                tensor.ReleaseCudaInferenceBuffers();
            if (_clearPoolOnDispose && _poolDeviceToClear.HasValue)
            {
                int deviceIndex = _poolDeviceToClear.Value;
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex).Synchronize();
                Tensor.ClearCudaFloatBufferPool(deviceIndex);
            }
        }
        finally
        {
            _tensors.Clear();
            _resources.Clear();
            _devices.Clear();
            Current.Value = _previous;
        }
    }
}
