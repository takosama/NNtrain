using System.Collections.Concurrent;
using NNtrain.Cuda.Interop;
using NNtrain.Runtime.Execution;

namespace NNtrain;

/// <summary>
/// Launch boundary for resident classification accuracy. All row logits stay
/// on CUDA; the only host-visible value is one Int32 correct-count scalar.
/// </summary>
internal static class CudaClassificationAccuracyNative
{
    internal static void CorrectFloat32(
        int deviceIndex,
        NativeCudaBuffer<float> logits,
        NativeCudaBuffer<int> targets,
        NativeCudaBuffer<int> correctCount,
        int sampleCount,
        int classCount,
        nint stream)
        => Check(
            CudaNativeGateway.ClassificationCorrectFloat32(
                deviceIndex,
                logits.NativePtr,
                targets.NativePtr,
                correctCount.NativePtr,
                sampleCount,
                classCount,
                stream),
            "CUDA classification correct-count(float32)");

    internal static void CorrectBFloat16(
        int deviceIndex,
        NativeCudaBuffer<ushort> logits,
        NativeCudaBuffer<int> targets,
        NativeCudaBuffer<int> correctCount,
        int sampleCount,
        int classCount,
        nint stream)
        => Check(
            CudaNativeGateway.ClassificationCorrectBFloat16(
                deviceIndex,
                logits.NativePtr,
                targets.NativePtr,
                correctCount.NativePtr,
                sampleCount,
                classCount,
                stream),
            "CUDA classification correct-count(bfloat16)");

    internal static void CorrectBfp8(
        int deviceIndex,
        CudaBfp8BufferView logits,
        NativeCudaBuffer<int> targets,
        NativeCudaBuffer<int> correctCount,
        int sampleCount,
        int classCount,
        nint stream)
        => Check(
            CudaNativeGateway.ClassificationCorrectBfp8(
                deviceIndex,
                logits.Payload.NativePtr,
                logits.Scales.NativePtr,
                logits.Descriptor.GetEffectiveBlockSize(
                    logits.Payload.Length),
                targets.NativePtr,
                correctCount.NativePtr,
                sampleCount,
                classCount,
                stream),
            "CUDA classification correct-count(BFP8)");

    private static void Check(int status, string operation)
        => NativeCudaRuntime.Check(status, operation);
}

/// <summary>
/// Reusable pinned Int32 readback. Begin queues the scalar copy immediately
/// after the reduction; Complete is deliberately delayed until the caller has
/// queued backward, so accuracy introduces no pre-backward host barrier.
/// </summary>
internal sealed unsafe class NativeCudaIntScalarReadback
{
    private const int IdleSlotCapacity = 8;
    private const int FallbackPoolCapacity = 4;
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<
        IStreamExecutionLane,
        Lazy<NativeCudaIntScalarReadbackPool>> LanePools = new();
    private static readonly ResettableBoundedDisposableLeaseCache<
        int,
        NativeCudaIntScalarReadbackPool> FallbackPools =
            new(FallbackPoolCapacity);
    private static int _activeLanePoolCount;
    private static int _liveSlotCount;

    private readonly NativeCudaIntScalarReadbackPool _owner;
    private readonly int _device;
    private readonly nint _host;
    private readonly nint _event;
    private BoundedDisposableLeaseCache<
        int,
        NativeCudaIntScalarReadbackPool>.Lease? _fallbackLease;
    private bool _pending;
    private int _checkedOut;
    private int _disposed;

    private NativeCudaIntScalarReadback(
        NativeCudaIntScalarReadbackPool owner,
        int device)
    {
        _owner = owner;
        _device = device;
        NativeCudaRuntime.Check(
            NativeCudaRuntime.HostAllocateNative(sizeof(int), out _host),
            "cudaMallocHost(classification correct-count)");
        try
        {
            NativeCudaRuntime.Check(
                NativeCudaRuntime.EventCreateNative(device, out _event),
                "cudaEventCreate(classification correct-count)");
        }
        catch
        {
            _ = NativeCudaRuntime.HostFreeNative(_host);
            throw;
        }
        Interlocked.Increment(ref _liveSlotCount);
    }

    internal static int ActiveLanePoolCount =>
        Volatile.Read(ref _activeLanePoolCount);
    internal static int FallbackPoolCount => FallbackPools.Count;
    internal static int LiveSlotCount => Volatile.Read(ref _liveSlotCount);

    internal static void DisposeFallbackResources()
        => FallbackPools.Dispose();

    internal static NativeCudaIntScalarReadback Rent(int device)
    {
        NativeCudaIntScalarReadbackPool? pool = null;
        BoundedDisposableLeaseCache<
            int,
            NativeCudaIntScalarReadbackPool>.Lease? fallbackLease = null;
        if (TensorExecutionContext.TryGetCudaStreamLane(
                device,
                out IStreamExecutionLane lane))
        {
            pool = LanePools.GetValue(
                lane,
                static owner => new Lazy<NativeCudaIntScalarReadbackPool>(
                    () => ExecutionLaneResources.Attach(
                        owner,
                        new NativeCudaIntScalarReadbackPool(
                            owner.DeviceIndex,
                            laneOwned: true)),
                    LazyThreadSafetyMode.ExecutionAndPublication)).Value;
        }
        else
        {
            fallbackLease = FallbackPools.Acquire(
                device,
                static value => new NativeCudaIntScalarReadbackPool(
                    value,
                    laneOwned: false));
            pool = fallbackLease?.Value;
        }

        if (pool is null)
        {
            fallbackLease?.Dispose();
            throw new InvalidOperationException(
                "A CUDA Int32 scalar readback pool could not be created.");
        }
        try
        {
            NativeCudaIntScalarReadback readback = pool.Rent();
            readback._fallbackLease = fallbackLease;
            return readback;
        }
        catch
        {
            fallbackLease?.Dispose();
            throw;
        }
    }

    internal void Begin(nint deviceSource, nint stream)
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
        if (Volatile.Read(ref _checkedOut) == 0)
        {
            throw new InvalidOperationException(
                "Classification correct-count readback is not rented.");
        }
        if (_pending)
        {
            throw new InvalidOperationException(
                "Classification correct-count readback is already pending.");
        }
        try
        {
            NativeCudaRuntime.Check(
                NativeCudaRuntime.CopyDeviceToHostAsyncRecordNative(
                    _device,
                    _host,
                    deviceSource,
                    sizeof(int),
                    stream,
                    _event),
                "cudaMemcpyAsync(D2H classification correct-count)");
            _pending = true;
        }
        catch
        {
            ReturnToOwner(reusable: true);
            throw;
        }
    }

    internal int CompleteAndReturn()
    {
        if (!_pending)
        {
            throw new InvalidOperationException(
                "Classification correct-count readback was not started.");
        }
        Exception? failure = null;
        int value = 0;
        try
        {
            NativeCudaRuntime.Check(
                NativeCudaRuntime.EventSynchronizeNative(_device, _event),
                "cudaEventSynchronize(classification correct-count)");
            value = *(int*)_host;
        }
        catch (Exception exception)
        {
            failure = exception;
        }
        finally
        {
            _pending = false;
            try
            {
                ReturnToOwner(reusable: failure is null);
            }
            catch (Exception returnFailure)
            {
                failure = failure is null
                    ? returnFailure
                    : new AggregateException(failure, returnFailure);
            }
        }
        if (failure is not null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(failure)
                .Throw();
        }
        return value;
    }

    internal void ReturnUnused()
    {
        if (_pending)
        {
            throw new InvalidOperationException(
                "A pending classification readback cannot be returned unused.");
        }
        ReturnToOwner(reusable: true);
    }

    private void ReturnToOwner(bool reusable)
    {
        if (Interlocked.Exchange(ref _checkedOut, 0) == 0)
            return;
        BoundedDisposableLeaseCache<
            int,
            NativeCudaIntScalarReadbackPool>.Lease? fallback =
                Interlocked.Exchange(ref _fallbackLease, null);
        try
        {
            _owner.Return(this, reusable);
        }
        finally
        {
            fallback?.Dispose();
        }
    }

    private void MarkRented()
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
        if (Interlocked.Exchange(ref _checkedOut, 1) != 0)
        {
            throw new InvalidOperationException(
                "Classification correct-count readback is already rented.");
        }
    }

    private void DisposeNative()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        List<Exception>? failures = null;
        int eventStatus = NativeCudaRuntime.EventDestroyNative(
            _device,
            _event);
        if (eventStatus != 0)
        {
            (failures ??= []).Add(new NativeCudaException(
                "cudaEventDestroy(classification correct-count)",
                eventStatus));
        }
        int hostStatus = NativeCudaRuntime.HostFreeNative(_host);
        if (hostStatus != 0)
        {
            (failures ??= []).Add(new NativeCudaException(
                "cudaFreeHost(classification correct-count)",
                hostStatus));
        }
        Interlocked.Decrement(ref _liveSlotCount);
        if (failures is not null)
        {
            throw new AggregateException(
                "CUDA Int32 scalar readback cleanup failed.",
                failures);
        }
    }

    private sealed class NativeCudaIntScalarReadbackPool : IDisposable
    {
        private readonly object _sync = new();
        private readonly Stack<NativeCudaIntScalarReadback> _idle = [];
        private readonly bool _laneOwned;
        private bool _disposed;

        internal NativeCudaIntScalarReadbackPool(
            int device,
            bool laneOwned)
        {
            Device = device;
            _laneOwned = laneOwned;
            if (laneOwned)
                Interlocked.Increment(ref _activeLanePoolCount);
        }

        internal int Device { get; }

        internal NativeCudaIntScalarReadback Rent()
        {
            lock (_sync)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                NativeCudaIntScalarReadback value = _idle.Count == 0
                    ? new NativeCudaIntScalarReadback(this, Device)
                    : _idle.Pop();
                value.MarkRented();
                return value;
            }
        }

        internal void Return(
            NativeCudaIntScalarReadback readback,
            bool reusable)
        {
            bool dispose;
            lock (_sync)
            {
                dispose = _disposed
                    || !reusable
                    || _idle.Count >= IdleSlotCapacity;
                if (!dispose)
                    _idle.Push(readback);
            }
            if (dispose)
                readback.DisposeNative();
        }

        public void Dispose()
        {
            NativeCudaIntScalarReadback[] idle;
            lock (_sync)
            {
                if (_disposed)
                    return;
                _disposed = true;
                idle = _idle.ToArray();
                _idle.Clear();
            }
            List<Exception>? failures = null;
            foreach (NativeCudaIntScalarReadback readback in idle)
            {
                try
                {
                    readback.DisposeNative();
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
            }
            if (_laneOwned)
                Interlocked.Decrement(ref _activeLanePoolCount);
            if (failures is not null)
            {
                throw new AggregateException(
                    $"CUDA Int32 scalar readback pool cleanup failed on " +
                    $"device {Device}.",
                    failures);
            }
        }
    }
}

/// <summary>
/// Owns the target/result leases until the queued scalar readback completes.
/// </summary>
internal sealed class CudaClassificationCorrectCountReadback : IDisposable
{
    private readonly NativeCudaDevice _accelerator;
    private NativeCudaBuffer<int>? _targets;
    private NativeCudaBuffer<int>? _correctCount;
    private NativeCudaIntScalarReadback? _readback;
    private bool _completed;

    internal CudaClassificationCorrectCountReadback(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<int> targets,
        NativeCudaBuffer<int> correctCount,
        NativeCudaIntScalarReadback readback)
    {
        _accelerator = accelerator;
        _targets = targets;
        _correctCount = correctCount;
        _readback = readback;
    }

    internal int CompleteAndReturn()
    {
        if (_completed)
        {
            throw new InvalidOperationException(
                "Classification correct-count was already completed.");
        }

        // Do not recycle device leases when event synchronization fails: the
        // kernel or D2H copy may still own them. A fatal CUDA failure may leak
        // this tiny lease, which is safer than handing live memory to a later
        // operation.
        int result = _readback!.CompleteAndReturn();
        _completed = true;
        _readback = null;
        ReturnBuffers();
        return result;
    }

    public void Dispose()
    {
        if (_completed)
            return;
        _ = _readback!.CompleteAndReturn();
        _completed = true;
        _readback = null;
        ReturnBuffers();
    }

    private void ReturnBuffers()
    {
        NativeCudaBuffer<int>? correctCount = Interlocked.Exchange(
            ref _correctCount,
            null);
        NativeCudaBuffer<int>? targets = Interlocked.Exchange(
            ref _targets,
            null);
        List<Exception>? failures = null;
        TryReturn(correctCount, ref failures);
        TryReturn(targets, ref failures);
        if (failures is not null)
        {
            throw new AggregateException(
                "CUDA classification accuracy buffer cleanup failed.",
                failures);
        }
    }

    private void TryReturn(
        NativeCudaBuffer<int>? buffer,
        ref List<Exception>? failures)
    {
        if (buffer is null)
            return;
        try
        {
            Tensor.ReturnCudaIntBuffer(_accelerator, buffer);
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
    }
}

public partial class Tensor
{
    /// <summary>
    /// Queues row-wise argmax, target comparison, block reduction, and a
    /// single Int32 D2H readback for a CUDA-resident classification batch.
    /// </summary>
    internal CudaClassificationCorrectCountReadback
        BeginCudaClassificationCorrectCount(
            int[] targets,
            int classCount,
            int deviceIndex = -1)
    {
        ArgumentNullException.ThrowIfNull(targets);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(classCount);
        if (targets.Length == 0)
        {
            throw new ArgumentException(
                "Classification targets cannot be empty.",
                nameof(targets));
        }
        if (Numel != checked(targets.Length * classCount))
        {
            throw new ArgumentException(
                $"Expected {Numel / classCount} targets for logits shape " +
                $"[{targets.Length}, {classCount}], but received " +
                $"{targets.Length}.",
                nameof(targets));
        }
        if (Device != TensorDevice.Cuda)
        {
            throw new InvalidOperationException(
                "CUDA classification accuracy requires resident logits.");
        }
        if (DType == TensorDType.Float16)
        {
            throw new NotSupportedException(
                "CUDA classification accuracy supports Float32, BFloat16, " +
                "BFP8, and block-scaled Mix8 storage.");
        }

        int resolvedDevice = deviceIndex >= 0
            ? ResolveCudaDeviceIndex(deviceIndex)
            : _cudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(resolvedDevice);
        accelerator.Bind();
        nint stream = accelerator.DefaultStream;

        NativeCudaBuffer<int>? targetBuffer = null;
        NativeCudaBuffer<int>? correctCount = null;
        NativeCudaIntScalarReadback? readback = null;
        bool kernelWasSubmitted = false;
        try
        {
            targetBuffer = RentCudaIntBuffer(resolvedDevice, targets);
            correctCount = RentCudaIntBuffer(resolvedDevice, 1);
            switch (DType)
            {
                case TensorDType.Float32:
                    CudaClassificationAccuracyNative.CorrectFloat32(
                        resolvedDevice,
                        EnsureCudaFloat32Buffer(resolvedDevice),
                        targetBuffer,
                        correctCount,
                        targets.Length,
                        classCount,
                        stream);
                    break;
                case TensorDType.BFloat16:
                    CudaClassificationAccuracyNative.CorrectBFloat16(
                        resolvedDevice,
                        EnsureCudaBFloat16Buffer(resolvedDevice),
                        targetBuffer,
                        correctCount,
                        targets.Length,
                        classCount,
                        stream);
                    break;
                case TensorDType.Bfp8:
                    ValidateBfp8PrecisionContract();
                    CudaClassificationAccuracyNative.CorrectBfp8(
                        resolvedDevice,
                        EnsureCudaBfp8Buffer(resolvedDevice),
                        targetBuffer,
                        correctCount,
                        targets.Length,
                        classCount,
                        stream);
                    break;
                default:
                    throw new NotSupportedException(
                        $"CUDA classification accuracy does not support " +
                        $"physical dtype {DType}.");
            }
            kernelWasSubmitted = true;

            readback = NativeCudaIntScalarReadback.Rent(resolvedDevice);
            readback.Begin(correctCount.NativePtr, stream);
            var operation = new CudaClassificationCorrectCountReadback(
                accelerator,
                targetBuffer,
                correctCount,
                readback);
            targetBuffer = null;
            correctCount = null;
            readback = null;
            return operation;
        }
        catch
        {
            if (kernelWasSubmitted)
                accelerator.Synchronize(
                    "CUDA classification accuracy rollback synchronization");
            readback?.ReturnUnused();
            if (correctCount is not null)
                ReturnCudaIntBuffer(accelerator, correctCount);
            if (targetBuffer is not null)
                ReturnCudaIntBuffer(accelerator, targetBuffer);
            throw;
        }
    }
}
