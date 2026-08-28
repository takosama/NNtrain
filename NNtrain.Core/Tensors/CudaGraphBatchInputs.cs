using NNtrain.Runtime.Execution;

namespace NNtrain;

/// <summary>
/// Lifecycle and transfer counters for one fixed-shape CUDA Graph batch input
/// reservation. Logical borrows never allocate or transfer device memory.
/// </summary>
internal readonly record struct CudaGraphBatchInputsTelemetry(
    long UpdateCount,
    long HostToDeviceCopyCount,
    long HostToDeviceBytes,
    long BorrowCount,
    long ReturnCount);

/// <summary>
/// Owns stable token/target device addresses and their two independent pinned
/// upload slots for one lane and one fixed batch-token count. Update is issued
/// before graph launch; graph capture only borrows the already-populated
/// buffers, so memcpy nodes are not accidentally captured into the graph.
/// </summary>
internal sealed class CudaGraphBatchInputs : IDisposable
{
    [ThreadStatic]
    private static CaptureState? _currentCapture;

    private readonly object _sync = new();
    private readonly IStreamExecutionLane _lane;
    private readonly NativeCudaBuffer<int> _inputBuffer;
    private readonly NativeCudaBuffer<int> _targetBuffer;
    private readonly NativeCudaPinnedUpload<int> _inputUpload;
    private readonly NativeCudaPinnedUpload<int> _targetUpload;
    private readonly int _length;
    private readonly int _vocabularySize;
    private readonly int _ignoreIndex;
    private int[]? _latestInput;
    private int[]? _latestTarget;
    private CaptureState? _activeCapture;
    private long _updateCount;
    private long _hostToDeviceCopyCount;
    private long _hostToDeviceBytes;
    private long _borrowCount;
    private long _returnCount;
    private int _disposed;

    private CudaGraphBatchInputs(
        IStreamExecutionLane lane,
        int length,
        int vocabularySize,
        int ignoreIndex)
    {
        ArgumentNullException.ThrowIfNull(lane);
        if (lane.DeviceKind != ExecutionDeviceKind.Cuda)
            throw new ArgumentException("A CUDA execution lane is required.", nameof(lane));
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length));
        if (vocabularySize <= 0)
            throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        if (!TensorExecutionContext.TryGetCudaStreamLane(
                lane.DeviceIndex,
                out IStreamExecutionLane activeLane)
            || !ReferenceEquals(activeLane, lane))
        {
            throw new InvalidOperationException(
                "CUDA Graph batch inputs must be created inside the owning " +
                "lane's active execution session.");
        }

        _lane = lane;
        _length = length;
        _vocabularySize = vocabularySize;
        _ignoreIndex = ignoreIndex;
        lane.ActivateComputeStream();
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(lane.DeviceIndex);

        NativeCudaBuffer<int>? inputBuffer = null;
        NativeCudaBuffer<int>? targetBuffer = null;
        NativeCudaPinnedUpload<int>? inputUpload = null;
        NativeCudaPinnedUpload<int>? targetUpload = null;
        try
        {
            inputBuffer = accelerator.Allocate1D<int>(
                length,
                NNtrain.Cuda.Memory.CudaMemoryKind.Persistent);
            targetBuffer = accelerator.Allocate1D<int>(
                length,
                NNtrain.Cuda.Memory.CudaMemoryKind.Persistent);
            inputUpload = new NativeCudaPinnedUpload<int>(
                lane.DeviceIndex,
                length);
            targetUpload = new NativeCudaPinnedUpload<int>(
                lane.DeviceIndex,
                length);
        }
        catch (Exception creationFailure)
        {
            List<Exception> failures = [creationFailure];
            TryDispose(targetUpload, failures);
            TryDispose(inputUpload, failures);
            TryDispose(targetBuffer, failures);
            TryDispose(inputBuffer, failures);
            if (failures.Count == 1)
                throw;
            throw new AggregateException(
                "CUDA Graph batch input construction and rollback failed.",
                failures);
        }

        _inputBuffer = inputBuffer;
        _targetBuffer = targetBuffer;
        _inputUpload = inputUpload;
        _targetUpload = targetUpload;
    }

    internal static CudaGraphBatchInputs Create(
        IStreamExecutionLane lane,
        int length,
        int vocabularySize,
        int ignoreIndex = Tensor.DefaultCrossEntropyIgnoreIndex)
    {
        var reservation = new CudaGraphBatchInputs(
            lane,
            length,
            vocabularySize,
            ignoreIndex);
        return ExecutionLaneResources.Attach(lane, reservation);
    }

    internal int DeviceIndex => _lane.DeviceIndex;

    internal int Length => _length;

    internal nint InputPointer => _inputBuffer.NativePtr;

    internal nint TargetPointer => _targetBuffer.NativePtr;

    internal NativeCudaBuffer<int> InputBuffer => _inputBuffer;

    internal NativeCudaBuffer<int> TargetBuffer => _targetBuffer;

    internal CudaGraphBatchInputsTelemetry Telemetry => new(
        Interlocked.Read(ref _updateCount),
        Interlocked.Read(ref _hostToDeviceCopyCount),
        Interlocked.Read(ref _hostToDeviceBytes),
        Interlocked.Read(ref _borrowCount),
        Interlocked.Read(ref _returnCount));

    /// <summary>
    /// Validates and enqueues exactly two authorized H2D copies on the lane's
    /// compute stream. Source arrays are retained only until the next update so
    /// the subsequent capture scope can preserve their identity.
    /// </summary>
    internal void Update(int[] input, int[] target, nint stream)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(target);
        if (input.Length != _length)
        {
            throw new ArgumentException(
                $"Expected {_length} input tokens, but received {input.Length}.",
                nameof(input));
        }
        if (target.Length != _length)
        {
            throw new ArgumentException(
                $"Expected {_length} target tokens, but received {target.Length}.",
                nameof(target));
        }
        if (ReferenceEquals(input, target))
        {
            throw new ArgumentException(
                "Input and target arrays must be distinct so their fixed CUDA " +
                "buffers can be identified without call-order assumptions.",
                nameof(target));
        }
        if (stream != _lane.ComputeStreamHandle)
        {
            throw new ArgumentException(
                "Batch updates must use the reservation's compute stream.",
                nameof(stream));
        }

        ValidateValues(input, target);
        lock (_sync)
        {
            ThrowIfDisposed();
            if (_activeCapture is not null)
            {
                throw new InvalidOperationException(
                    "Batch inputs cannot be updated while their capture borrow " +
                    "scope is active.");
            }

            _lane.ActivateComputeStream();
            using (DeviceTransferGuard.AllowBatchHostToDevice())
            {
                _inputUpload.Upload(input, _inputBuffer, stream);
                _targetUpload.Upload(target, _targetBuffer, stream);
            }
            _latestInput = input;
            _latestTarget = target;
            Interlocked.Increment(ref _updateCount);
            Interlocked.Add(ref _hostToDeviceCopyCount, 2);
            Interlocked.Add(
                ref _hostToDeviceBytes,
                checked(2L * _length * sizeof(int)));
        }
    }

    /// <summary>
    /// Pushes the typed, thread-affine borrow scope used while recording one
    /// forward/backward graph. Every successful borrow must be returned before
    /// this scope is disposed.
    /// </summary>
    internal CudaGraphBatchInputCaptureScope PushCaptureScope()
    {
        lock (_sync)
        {
            ThrowIfDisposed();
            if (_latestInput is null || _latestTarget is null)
            {
                throw new InvalidOperationException(
                    "Update must succeed before a graph capture scope is pushed.");
            }
            CaptureState? ambient = GetCurrentCapture();
            if (ambient is not null || _activeCapture is not null)
            {
                throw new InvalidOperationException(
                    "CUDA Graph batch input capture scopes cannot be nested.");
            }
            if (!TensorExecutionContext.TryGetCudaStreamLane(
                    DeviceIndex,
                    out IStreamExecutionLane activeLane)
                || !ReferenceEquals(activeLane, _lane))
            {
                throw new InvalidOperationException(
                    "The owning CUDA execution lane is not active.");
            }

            var state = new CaptureState(
                this,
                _latestInput,
                _latestTarget,
                Environment.CurrentManagedThreadId);
            _activeCapture = state;
            _currentCapture = state;
            return new CudaGraphBatchInputCaptureScope(state);
        }
    }

    /// <summary>
    /// Preserves a registered token/target array only during graph capture.
    /// Every ordinary CPU/CUDA call receives the historical defensive clone.
    /// </summary>
    internal static int[] RetainOrClone(int[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        CaptureState? capture = GetCurrentCapture();
        return capture is not null && capture.Matches(values)
            ? values
            : (int[])values.Clone();
    }

    internal static bool TryBorrow(
        int deviceIndex,
        int[] source,
        out NativeCudaBuffer<int> buffer)
    {
        CaptureState? capture = GetCurrentCapture();
        if (capture is null || capture.Owner.DeviceIndex != deviceIndex)
        {
            buffer = null!;
            return false;
        }
        return capture.TryBorrow(source, out buffer);
    }

    internal static bool TryReturn(
        int deviceIndex,
        NativeCudaBuffer<int> buffer)
    {
        CaptureState? capture = GetCurrentCapture();
        return capture is not null
            && capture.Owner.DeviceIndex == deviceIndex
            && capture.TryReturn(buffer);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        CaptureState? capture;
        lock (_sync)
        {
            capture = _activeCapture;
            _activeCapture = null;
            _latestInput = null;
            _latestTarget = null;
        }
        capture?.Invalidate();
        if (ReferenceEquals(_currentCapture, capture))
            _currentCapture = null;

        var failures = new List<Exception>();
        TryRun(_lane.SynchronizeComputeStream, failures);
        TryDispose(_targetUpload, failures);
        TryDispose(_inputUpload, failures);
        TryDispose(_targetBuffer, failures);
        TryDispose(_inputBuffer, failures);
        if (failures.Count != 0)
        {
            throw new AggregateException(
                "CUDA Graph batch input cleanup failed.",
                failures);
        }
    }

    private void ValidateValues(int[] input, int[] target)
    {
        for (int index = 0; index < input.Length; index++)
        {
            if ((uint)input[index] >= (uint)_vocabularySize)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(input),
                    input[index],
                    $"Input token at position {index} must be between 0 and " +
                    $"{_vocabularySize - 1}.");
            }
        }

        int validTargets = 0;
        for (int index = 0; index < target.Length; index++)
        {
            int value = target[index];
            if (value == _ignoreIndex)
                continue;
            if ((uint)value >= (uint)_vocabularySize)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(target),
                    value,
                    $"Target token at position {index} must equal ignoreIndex " +
                    $"({_ignoreIndex}) or be between 0 and " +
                    $"{_vocabularySize - 1}.");
            }
            validTargets++;
        }
        if (validTargets == 0)
        {
            throw new ArgumentException(
                "At least one target token must not equal ignoreIndex.",
                nameof(target));
        }
    }

    private void EndCapture(CaptureState state, bool hadOutstandingBorrow)
    {
        lock (_sync)
        {
            if (ReferenceEquals(_activeCapture, state))
                _activeCapture = null;
        }
        if (ReferenceEquals(_currentCapture, state))
            _currentCapture = null;
        if (hadOutstandingBorrow)
        {
            throw new InvalidOperationException(
                "CUDA Graph batch input capture ended with an unreturned " +
                "input or target buffer.");
        }
    }

    private void RecordBorrow()
        => Interlocked.Increment(ref _borrowCount);

    private void RecordReturn()
        => Interlocked.Increment(ref _returnCount);

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);

    private static CaptureState? GetCurrentCapture()
    {
        CaptureState? capture = _currentCapture;
        if (capture is not null && !capture.IsActive)
        {
            _currentCapture = null;
            return null;
        }
        return capture;
    }

    private static void TryRun(Action action, List<Exception> failures)
    {
        try
        {
            action();
        }
        catch (Exception exception)
        {
            failures.Add(exception);
        }
    }

    private static void TryDispose(
        IDisposable? resource,
        List<Exception> failures)
    {
        if (resource is not null)
            TryRun(resource.Dispose, failures);
    }

    internal sealed class CaptureState(
        CudaGraphBatchInputs owner,
        int[] input,
        int[] target,
        int ownerThreadId)
    {
        private readonly object _sync = new();
        private bool _inputBorrowed;
        private bool _targetBorrowed;
        private bool _active = true;

        internal CudaGraphBatchInputs Owner { get; } = owner;

        internal int OwnerThreadId { get; } = ownerThreadId;

        internal bool IsActive
        {
            get
            {
                lock (_sync)
                    return _active;
            }
        }

        internal bool Matches(int[] source)
        {
            EnsureOwnerThread();
            lock (_sync)
            {
                return _active
                    && (ReferenceEquals(source, input)
                        || ReferenceEquals(source, target));
            }
        }

        internal bool TryBorrow(
            int[] source,
            out NativeCudaBuffer<int> buffer)
        {
            EnsureOwnerThread();
            lock (_sync)
            {
                EnsureActive();
                if (ReferenceEquals(source, input))
                {
                    if (_inputBorrowed)
                        throw new InvalidOperationException(
                            "The fixed CUDA input buffer is already borrowed.");
                    _inputBorrowed = true;
                    buffer = Owner._inputBuffer;
                }
                else if (ReferenceEquals(source, target))
                {
                    if (_targetBorrowed)
                        throw new InvalidOperationException(
                            "The fixed CUDA target buffer is already borrowed.");
                    _targetBorrowed = true;
                    buffer = Owner._targetBuffer;
                }
                else
                {
                    buffer = null!;
                    return false;
                }
            }
            Owner.RecordBorrow();
            return true;
        }

        internal bool TryReturn(NativeCudaBuffer<int> buffer)
        {
            EnsureOwnerThread();
            lock (_sync)
            {
                EnsureActive();
                if (ReferenceEquals(buffer, Owner._inputBuffer))
                {
                    if (!_inputBorrowed)
                        throw new InvalidOperationException(
                            "The fixed CUDA input buffer was returned twice.");
                    _inputBorrowed = false;
                }
                else if (ReferenceEquals(buffer, Owner._targetBuffer))
                {
                    if (!_targetBorrowed)
                        throw new InvalidOperationException(
                            "The fixed CUDA target buffer was returned twice.");
                    _targetBorrowed = false;
                }
                else
                {
                    return false;
                }
            }
            Owner.RecordReturn();
            return true;
        }

        internal bool Close()
        {
            EnsureOwnerThread();
            lock (_sync)
            {
                if (!_active)
                    return false;
                bool outstanding = _inputBorrowed || _targetBorrowed;
                _inputBorrowed = false;
                _targetBorrowed = false;
                _active = false;
                return outstanding;
            }
        }

        internal void Invalidate()
        {
            lock (_sync)
            {
                _inputBorrowed = false;
                _targetBorrowed = false;
                _active = false;
            }
        }

        private void EnsureOwnerThread()
        {
            if (Environment.CurrentManagedThreadId != OwnerThreadId)
            {
                throw new InvalidOperationException(
                    "CUDA Graph batch input capture scopes are thread-affine.");
            }
        }

        private void EnsureActive()
        {
            if (!_active)
                throw new ObjectDisposedException(
                    nameof(CudaGraphBatchInputCaptureScope));
        }
    }

    internal sealed class CudaGraphBatchInputCaptureScope : IDisposable
    {
        private CaptureState? _state;

        internal CudaGraphBatchInputCaptureScope(CaptureState state)
        {
            _state = state;
        }

        public void Dispose()
        {
            CaptureState? state = Volatile.Read(ref _state);
            if (state is null)
                return;
            if (Environment.CurrentManagedThreadId != state.OwnerThreadId)
            {
                throw new InvalidOperationException(
                    "CUDA Graph batch input capture scopes must be disposed " +
                    "on the thread that pushed them.");
            }
            state = Interlocked.Exchange(ref _state, null);
            if (state is null)
                return;
            bool outstanding = state.Close();
            state.Owner.EndCapture(state, outstanding);
        }
    }
}
