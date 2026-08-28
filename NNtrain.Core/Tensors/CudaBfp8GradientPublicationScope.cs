using NNtrain.Runtime.Execution;

namespace NNtrain;

/// <summary>
/// Owns one finite-status scalar for an entire single-GPU autograd graph.
/// Every pure-BFP8 leaf is quantized asynchronously on the lane compute
/// stream; the graph boundary reads one finite-status scalar and one norm
/// scalar regardless of leaf count, then makes the encoded replicas
/// authoritative.
/// </summary>
internal sealed class CudaBfp8GradientPublicationScope : IDisposable
{
    private static readonly AsyncLocal<CudaBfp8GradientPublicationScope?>
        Current = new();
    private static readonly AsyncLocal<CudaBfp8GraphGradientPublication?>
        GraphRecording = new();

    private readonly CudaBfp8GradientPublicationScope? _previous;
    private readonly CudaBfp8GraphGradientPublication? _graphPublication;
    private readonly int _deviceIndex;
    private readonly NativeCudaDevice _accelerator;
    private readonly nint _computeStream;
    private readonly NativeCudaBuffer<int> _finiteStatus;
    private readonly NativeCudaBuffer<double> _squaredSum;
    private readonly List<Tensor> _published = [];
    private readonly HashSet<Tensor> _seen = new(
        ReferenceEqualityComparer.Instance);
    private int _completed;
    private int _disposed;

    private CudaBfp8GradientPublicationScope(int deviceIndex)
    {
        _previous = Current.Value;
        _deviceIndex = deviceIndex;
        _accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        _computeStream = _accelerator.DefaultStream;
        _finiteStatus = _accelerator.Allocate1D<int>(
            1,
            Cuda.Memory.CudaMemoryKind.Workspace);
        NativeCudaBuffer<double>? squaredSum = null;
        try
        {
            squaredSum = _accelerator.Allocate1D<double>(
                1,
                Cuda.Memory.CudaMemoryKind.Workspace);
            _finiteStatus.MemSetToZero();
            squaredSum.MemSetToZero();
            _squaredSum = squaredSum;
            Current.Value = this;
        }
        catch
        {
            squaredSum?.Dispose();
            _finiteStatus.Dispose();
            throw;
        }
    }

    private CudaBfp8GradientPublicationScope(
        CudaBfp8GraphGradientPublication graphPublication)
    {
        ArgumentNullException.ThrowIfNull(graphPublication);
        _previous = Current.Value;
        if (_previous is not null)
        {
            throw new InvalidOperationException(
                "Pure BFP8 gradient publication scopes cannot be nested.");
        }
        _graphPublication = graphPublication;
        _deviceIndex = graphPublication.DeviceIndex;
        _accelerator = ForgetMemoryV2Cuda.GetAccelerator(_deviceIndex);
        _computeStream = _accelerator.DefaultStream;
        _finiteStatus = graphPublication.FiniteStatus;
        _squaredSum = graphPublication.SquaredSum;
        graphPublication.BeginBackwardRecording();
        Current.Value = this;
    }

    internal static IDisposable PushGraphRecording(
        CudaBfp8GraphGradientPublication graphPublication)
    {
        ArgumentNullException.ThrowIfNull(graphPublication);
        if (GraphRecording.Value is not null)
        {
            throw new InvalidOperationException(
                "Pure BFP8 CUDA Graph publication recording cannot be nested.");
        }
        if (Tensor.CudaDeviceIndex != graphPublication.DeviceIndex)
        {
            throw new InvalidOperationException(
                $"Pure BFP8 CUDA Graph publication belongs to device " +
                $"{graphPublication.DeviceIndex}, not active device " +
                $"{Tensor.CudaDeviceIndex}.");
        }
        GraphRecording.Value = graphPublication;
        return new GraphRecordingScope(graphPublication);
    }

    internal static CudaBfp8GradientPublicationScope? TryCreate(
        IReadOnlyList<Tensor> topologicalOrder)
    {
        ArgumentNullException.ThrowIfNull(topologicalOrder);
        PrecisionPolicy? policy =
            TensorExecutionContext.ActivePrecisionPolicy;
        bool pureBfp8 = policy?.Gradient == NumericFormat.Bfp8
            || policy is null;
        if (Tensor.ExecutionDevice != TensorDevice.Cuda
            || !pureBfp8
            || CudaGradientReductionContext.HasActivePlan)
        {
            return null;
        }
        bool hasPureBfp8Leaf = topologicalOrder.Any(tensor =>
            tensor.Node.IsLeaf
            && tensor.DType == TensorDType.Bfp8
            && tensor.Bfp8Quantization
                == Bfp8QuantizationDescriptor.TensorWide);
        if (!hasPureBfp8Leaf)
            return null;
        CudaBfp8GraphGradientPublication? graphPublication =
            GraphRecording.Value;
        return graphPublication is null
            ? new CudaBfp8GradientPublicationScope(Tensor.CudaDeviceIndex)
            : new CudaBfp8GradientPublicationScope(graphPublication);
    }

    internal static bool TryPublish(Tensor tensor)
    {
        CudaBfp8GradientPublicationScope? scope = Current.Value;
        if (scope is null
            || tensor.DType != TensorDType.Bfp8
            || tensor.Bfp8Quantization
                != Bfp8QuantizationDescriptor.TensorWide
            || !tensor.HasGradientBuffer)
        {
            return false;
        }
        if (!scope._seen.Add(tensor))
        {
            throw new InvalidOperationException(
                $"Pure BFP8 leaf '{tensor.Name}' was published twice in " +
                "one autograd graph.");
        }
        tensor.QuantizeCudaBfp8Gradient(
            scope._deviceIndex,
            scope._finiteStatus,
            scope._computeStream,
            scope._squaredSum);
        scope._published.Add(tensor);
        return true;
    }

    internal void Complete(bool publish)
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0, this);
        if (Interlocked.Exchange(ref _completed, 1) != 0)
            return;
        if (_graphPublication is not null)
        {
            _graphPublication.CompleteBackwardRecording(_published, publish);
            return;
        }
        if (_published.Count == 0)
            return;

        CudaGradientBuckets.Synchronize(
            _accelerator,
            _deviceIndex,
            _computeStream);
        var finite = new int[1];
        var squaredSum = new double[1];
        _finiteStatus.CopyToCPU(finite);
        _squaredSum.CopyToCPU(squaredSum);
        if (finite[0] != 0)
        {
            string names = string.Join(
                ", ",
                _published.Take(4).Select(tensor => tensor.Name));
            if (_published.Count > 4)
                names += $", ... ({_published.Count} leaves)";
            throw new InvalidOperationException(
                $"Non-finite CUDA gradient detected in pure BFP8 graph " +
                $"on device {_deviceIndex}; published leaves: {names}.");
        }

        if (publish)
        {
            foreach (Tensor tensor in _published)
                tensor.CommitCudaBfp8Gradient(_deviceIndex);
            TensorCudaKernels.PublishGradientSquaredSum(
                _published,
                [_deviceIndex],
                squaredSum[0]);
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        if (ReferenceEquals(Current.Value, this))
            Current.Value = _previous;
        if (_graphPublication is not null)
        {
            if (Interlocked.Exchange(ref _completed, 1) == 0)
                _graphPublication.AbortBackwardRecording();
            return;
        }
        List<Exception>? failures = null;
        try
        {
            _finiteStatus.Dispose();
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
        try
        {
            _squaredSum.Dispose();
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
        if (failures is not null)
        {
            throw new AggregateException(
                "BFP8 graph publication scalar cleanup failed.",
                failures);
        }
    }

    private sealed class GraphRecordingScope(
        CudaBfp8GraphGradientPublication graphPublication) : IDisposable
    {
        private int _disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            if (!ReferenceEquals(GraphRecording.Value, graphPublication))
            {
                throw new InvalidOperationException(
                    "Pure BFP8 CUDA Graph publication recording scope was " +
                    "disposed out of order.");
            }
            GraphRecording.Value = null;
        }
    }
}

/// <summary>
/// Session-shape-owned publication state for a captured single-device pure
/// BFP8 backward. Parameter payloads and their tensor-wide scales remain at
/// stable addresses, while the finite and norm scalars are persistent graph
/// operands. Managed gradient authority is published only after replay.
/// </summary>
internal sealed class CudaBfp8GraphGradientPublication : IDisposable
{
    private readonly int _ownerThreadId;
    private readonly NativeCudaDevice _accelerator;
    private readonly nint _computeStream;
    private readonly Tensor[] _slotTensors;
    private readonly GradientSlot[] _slots;
    private readonly HashSet<Tensor> _slotSet;
    private readonly int[] _finiteHost = new int[1];
    private readonly double[] _squaredSumHost = new double[1];
    private Tensor[]? _recordedTensors;
    private int _backwardRecording;
    private int _disposed;

    internal CudaBfp8GraphGradientPublication(
        int deviceIndex,
        IReadOnlyList<Parameter> parameters)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceIndex);
        ArgumentNullException.ThrowIfNull(parameters);
        DeviceIndex = deviceIndex;
        _ownerThreadId = Environment.CurrentManagedThreadId;
        _accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        _computeStream = _accelerator.DefaultStream;
        _slotTensors = parameters
            .Select(parameter => parameter.T)
            .Where(tensor => tensor.DType == TensorDType.Bfp8
                && tensor.Bfp8Quantization
                    == Bfp8QuantizationDescriptor.TensorWide)
            .Distinct((IEqualityComparer<Tensor>)
                ReferenceEqualityComparer.Instance)
            .ToArray();
        if (_slotTensors.Length == 0)
        {
            throw new ArgumentException(
                "Pure BFP8 CUDA Graph publication requires BFP8 parameters.",
                nameof(parameters));
        }
        _slotSet = new HashSet<Tensor>(
            _slotTensors,
            ReferenceEqualityComparer.Instance);
        _slots = new GradientSlot[_slotTensors.Length];
        NativeCudaBuffer<int>? finiteStatus = null;
        NativeCudaBuffer<double>? squaredSum = null;
        try
        {
            for (int index = 0; index < _slotTensors.Length; index++)
            {
                CudaBfp8BufferView view = _slotTensors[index]
                    .PrepareCudaBfp8GradientReplica(deviceIndex);
                _slots[index] = new GradientSlot(
                    view.Payload.NativePtr,
                    view.Scales.NativePtr,
                    checked((int)view.Payload.Length));
            }
            finiteStatus = _accelerator.Allocate1D<int>(
                1,
                Cuda.Memory.CudaMemoryKind.Persistent);
            squaredSum = _accelerator.Allocate1D<double>(
                1,
                Cuda.Memory.CudaMemoryKind.Persistent);
            FiniteStatus = finiteStatus;
            SquaredSum = squaredSum;
        }
        catch
        {
            squaredSum?.Dispose();
            finiteStatus?.Dispose();
            throw;
        }
    }

    internal int DeviceIndex { get; }
    internal NativeCudaBuffer<int> FiniteStatus { get; }
    internal NativeCudaBuffer<double> SquaredSum { get; }

    internal IDisposable PushRecording()
    {
        ValidateOwnerThread();
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
        ValidateStableSlots();
        return CudaBfp8GradientPublicationScope.PushGraphRecording(this);
    }

    internal void BeginBackwardRecording()
    {
        ValidateOwnerThread();
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
        if (Interlocked.CompareExchange(ref _backwardRecording, 1, 0) != 0)
        {
            throw new InvalidOperationException(
                "Pure BFP8 CUDA Graph backward recording was begun twice.");
        }
        FiniteStatus.MemSetToZero();
        SquaredSum.MemSetToZero();
    }

    internal void CompleteBackwardRecording(
        IReadOnlyList<Tensor> published,
        bool backwardSucceeded)
    {
        ArgumentNullException.ThrowIfNull(published);
        ValidateOwnerThread();
        if (Interlocked.Exchange(ref _backwardRecording, 0) != 1)
        {
            throw new InvalidOperationException(
                "Pure BFP8 CUDA Graph backward recording was not active.");
        }
        if (!backwardSucceeded)
            return;

        var seen = new HashSet<Tensor>(ReferenceEqualityComparer.Instance);
        foreach (Tensor tensor in published)
        {
            if (!_slotSet.Contains(tensor) || !seen.Add(tensor))
            {
                throw new InvalidOperationException(
                    $"Pure BFP8 CUDA Graph published an unexpected or " +
                    $"duplicate gradient '{tensor.Name}'.");
            }
        }
        if (published.Count == 0)
        {
            throw new InvalidOperationException(
                "Pure BFP8 CUDA Graph backward published no leaf gradients.");
        }
        if (_recordedTensors is null)
        {
            _recordedTensors = published.ToArray();
            return;
        }
        if (_recordedTensors.Length != published.Count
            || !_recordedTensors.All(seen.Contains))
        {
            throw new InvalidOperationException(
                "Pure BFP8 CUDA Graph leaf publication changed between " +
                "prewarm and capture.");
        }
    }

    internal void AbortBackwardRecording()
    {
        ValidateOwnerThread();
        Interlocked.Exchange(ref _backwardRecording, 0);
    }

    internal void PublishAfterReplay()
    {
        ValidateOwnerThread();
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
        if (Volatile.Read(ref _backwardRecording) != 0)
        {
            throw new InvalidOperationException(
                "Cannot publish pure BFP8 Graph gradients while recording.");
        }
        Tensor[] published = _recordedTensors
            ?? throw new InvalidOperationException(
                "Pure BFP8 CUDA Graph captured no gradient publication.");
        ValidateStableSlots();
        CudaGradientBuckets.Synchronize(
            _accelerator,
            DeviceIndex,
            _computeStream);
        FiniteStatus.CopyToCPU(_finiteHost);
        SquaredSum.CopyToCPU(_squaredSumHost);
        if (_finiteHost[0] != 0)
        {
            string names = string.Join(
                ", ",
                published.Take(4).Select(tensor => tensor.Name));
            if (published.Length > 4)
                names += $", ... ({published.Length} leaves)";
            throw new InvalidOperationException(
                $"Non-finite CUDA gradient detected in captured pure BFP8 " +
                $"backward on device {DeviceIndex}; published leaves: " +
                $"{names}.");
        }
        foreach (Tensor tensor in published)
            tensor.CommitCudaBfp8Gradient(DeviceIndex);
        TensorCudaKernels.PublishGradientSquaredSum(
            published,
            [DeviceIndex],
            _squaredSumHost[0]);
    }

    private void ValidateStableSlots()
    {
        for (int index = 0; index < _slotTensors.Length; index++)
        {
            CudaBfp8BufferView current = _slotTensors[index]
                .PrepareCudaBfp8GradientReplica(DeviceIndex);
            GradientSlot expected = _slots[index];
            if (current.Payload.NativePtr != expected.Payload
                || current.Scales.NativePtr != expected.Scale
                || current.Payload.Length != expected.Length)
            {
                throw new InvalidOperationException(
                    $"Pure BFP8 CUDA Graph gradient slot for " +
                    $"'{_slotTensors[index].Name}' changed address or shape.");
            }
        }
    }

    private void ValidateOwnerThread()
    {
        if (Environment.CurrentManagedThreadId != _ownerThreadId)
        {
            throw new InvalidOperationException(
                $"Pure BFP8 CUDA Graph publication for device " +
                $"{DeviceIndex} is thread-affine.");
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        List<Exception>? failures = null;
        try
        {
            CudaGradientBuckets.Synchronize(
                _accelerator,
                DeviceIndex,
                _computeStream);
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
        foreach (IDisposable resource in new IDisposable[]
            { FiniteStatus, SquaredSum })
        {
            try
            {
                resource.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        if (failures is not null)
        {
            throw new AggregateException(
                "Pure BFP8 CUDA Graph publication cleanup failed.",
                failures);
        }
    }

    private readonly record struct GradientSlot(
        nint Payload,
        nint Scale,
        int Length);
}
