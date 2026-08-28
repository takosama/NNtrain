namespace NNtrain;

/// <summary>
/// Owns transient CUDA tensors created by one no-grad inference step.
/// Parameter buffers are not tracked because they are not operation results.
/// </summary>
internal sealed class CudaInferenceScope : IDisposable
{
    private static readonly AsyncLocal<CudaInferenceScope?> Current = new();
    private readonly object _sync = new();
    private readonly CudaInferenceScope? _previous;
    private readonly List<Tensor> _tensors = [];
    private readonly List<IDisposable> _resources = [];
    private readonly HashSet<int> _devices = [];
    private readonly Dictionary<int, NNtrain.Runtime.Execution.IStreamExecutionLane>
        _capturedLanes = [];
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
            if (TensorExecutionContext.TryGetCudaStreamLane(
                    _poolDeviceToClear.Value,
                    out NNtrain.Runtime.Execution.IStreamExecutionLane lane))
            {
                _capturedLanes.Add(_poolDeviceToClear.Value, lane);
            }
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
        CudaInferenceScope? scope = FindActive(Current.Value);
        if (!ReferenceEquals(scope, Current.Value))
            Current.Value = scope;
        if (scope is null)
            return;
        scope.TrackTensor(tensor, deviceIndex);
    }

    internal static bool TrackResource(IDisposable resource)
    {
        CudaInferenceScope? scope = FindActive(Current.Value);
        if (!ReferenceEquals(scope, Current.Value))
            Current.Value = scope;
        if (scope is null)
            return false;
        lock (scope._sync)
        {
            if (Volatile.Read(ref scope._disposed) != 0)
                return false;
            scope._resources.Add(resource);
            return true;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Tensor[] tensors;
        IDisposable[] resources;
        int[] devices;
        Dictionary<int, NNtrain.Runtime.Execution.IStreamExecutionLane> lanes;
        lock (_sync)
        {
            tensors = _tensors.ToArray();
            resources = _resources.ToArray();
            devices = _devices.ToArray();
            lanes = new(_capturedLanes);
            _tensors.Clear();
            _resources.Clear();
            _devices.Clear();
            _capturedLanes.Clear();
        }
        List<Exception>? failures = null;

        void TryCleanup(Action cleanup)
        {
            try
            {
                cleanup();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }

        try
        {
            foreach (int device in devices)
                TryCleanup(() => Synchronize(device, lanes));
            foreach (IDisposable resource in resources)
                TryCleanup(resource.Dispose);
            foreach (Tensor tensor in tensors)
                TryCleanup(tensor.ReleaseCudaInferenceBuffers);
            if (_clearPoolOnDispose && _poolDeviceToClear.HasValue)
            {
                var poolDevices = new HashSet<int>(devices)
                {
                    _poolDeviceToClear.Value,
                };
                foreach (int deviceIndex in poolDevices)
                {
                    TryCleanup(
                        () => Synchronize(deviceIndex, lanes));
                    TryCleanup(
                        () => Tensor.ClearCudaFloatBufferPool(deviceIndex));
                }
            }
        }
        finally
        {
            if (ReferenceEquals(Current.Value, this))
                Current.Value = FindActive(_previous);
        }

        if (failures is [Exception failure])
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(failure)
                .Throw();
        }
        if (failures is { Count: > 1 })
            throw new AggregateException("CUDA inference cleanup failed.", failures);
    }

    private void TrackTensor(Tensor tensor, int deviceIndex)
    {
        lock (_sync)
        {
            if (Volatile.Read(ref _disposed) != 0)
                return;
            _tensors.Add(tensor);
            _devices.Add(deviceIndex);
            if (!_capturedLanes.ContainsKey(deviceIndex)
                && TensorExecutionContext.TryGetCudaStreamLane(
                    deviceIndex,
                    out NNtrain.Runtime.Execution.IStreamExecutionLane lane))
            {
                _capturedLanes.Add(deviceIndex, lane);
            }
        }
    }

    private static void Synchronize(
        int deviceIndex,
        IReadOnlyDictionary<
            int,
            NNtrain.Runtime.Execution.IStreamExecutionLane> lanes)
    {
        if (lanes.TryGetValue(deviceIndex, out var lane))
            lane.SynchronizeComputeStream();
        else
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex).Synchronize();
    }

    private static CudaInferenceScope? FindActive(CudaInferenceScope? scope)
    {
        while (scope is not null
            && Volatile.Read(ref scope._disposed) != 0)
        {
            scope = scope._previous;
        }
        return scope;
    }
}
