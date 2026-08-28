using System.Text;
using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Interop;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain;

internal sealed class NativeCudaException : InvalidOperationException
{
    internal NativeCudaException(string operation, int status)
        : base($"{operation} failed with CUDA error {status}: " +
            NativeCudaRuntime.GetErrorString(status))
    {
        Status = status;
        NativeError = CudaNativeGateway.TakeCapturedFailure(status);
    }

    internal int Status { get; }

    internal CudaNativeErrorInfo? NativeError { get; }
}

internal static class NativeCudaRuntime
{
    // cudaErrorNotReady. CUDA allocation APIs may surface this status from a
    // previously queued asynchronous operation even though allocation itself
    // is valid after the device reaches the synchronization point.
    internal const int NotReadyStatus = 600;
    private const int OutOfMemoryStatus = 2;
    private static readonly Lazy<int> CachedDeviceCount = new(
        QueryDeviceCount,
        LazyThreadSafetyMode.ExecutionAndPublication);
    private static long _allocationCount;
    private static long _allocationBytes;
    private static long _freeCount;
    private static long _freeBytes;
    private static long _hostToDeviceCopyCount;
    private static long _hostToDeviceBytes;
    private static long _deviceToHostCopyCount;
    private static long _deviceToHostBytes;
    private static long _gradientCollectiveHostToDeviceCopyCount;
    private static long _gradientCollectiveHostToDeviceBytes;
    private static long _gradientCollectiveDeviceToHostCopyCount;
    private static long _gradientCollectiveDeviceToHostBytes;
    private static long _memsetLaunchCount;
    private static long _memsetBytes;
    // CUDA stream selection is native-thread-local. This managed value caches
    // only which stream binding has already been installed. Physical device
    // selection remains authoritative in cuda_runtime_bridge.cu and is
    // revalidated even when this stream binding is unchanged.
    [ThreadStatic]
    private static PreparedStreamState? _preparedStream;

    internal static int DeviceCount => CachedDeviceCount.Value;

    internal static NativeCudaAllocationTelemetry AllocationTelemetry
    {
        get
        {
            CudaNativeMemoryTelemetry lane =
                CudaMemoryManager.NativeTelemetry;
            return new NativeCudaAllocationTelemetry(
                checked(Interlocked.Read(ref _allocationCount)
                    + lane.AllocationCount),
                checked(Interlocked.Read(ref _allocationBytes)
                    + lane.AllocationBytes),
                checked(Interlocked.Read(ref _freeCount)
                    + lane.ReleaseCount),
                checked(Interlocked.Read(ref _freeBytes)
                    + lane.ReleaseBytes));
        }
    }

    internal static NativeCudaTransferTelemetry TransferTelemetry
        => new(
            Interlocked.Read(ref _hostToDeviceCopyCount),
            Interlocked.Read(ref _hostToDeviceBytes),
            Interlocked.Read(ref _deviceToHostCopyCount),
            Interlocked.Read(ref _deviceToHostBytes));

    /// <summary>
    /// Physical host transfers performed by the no-P2P gradient collective.
    /// These values are also included in <see cref="TransferTelemetry"/> so
    /// existing benchmark totals describe every physical host crossing.
    /// </summary>
    internal static NativeCudaTransferTelemetry
        GradientCollectiveTransferTelemetry
        => new(
            Interlocked.Read(
                ref _gradientCollectiveHostToDeviceCopyCount),
            Interlocked.Read(
                ref _gradientCollectiveHostToDeviceBytes),
            Interlocked.Read(
                ref _gradientCollectiveDeviceToHostCopyCount),
            Interlocked.Read(
                ref _gradientCollectiveDeviceToHostBytes));

    internal static NativeCudaMemsetTelemetry MemsetTelemetry
        => new(
            Interlocked.Read(ref _memsetLaunchCount),
            Interlocked.Read(ref _memsetBytes));

    internal static NativeCudaFallbackResourceTelemetry
        FallbackResourceTelemetry => new(
            CudaBlas.FallbackHandleCount,
            CudaBlasLt.FallbackResourceCount,
            CudaBlasLtInt8.FallbackResourceCount,
            CudaLayerNorm.FallbackScratchResourceCount,
            NativeCudaScalarReadback.FallbackPoolCount,
            NativeCudaIntScalarReadback.FallbackPoolCount,
            TensorCudaKernels.FallbackGradientNormScratchCount,
            NativeCudaScalarReadback.LiveSlotCount,
            NativeCudaIntScalarReadback.LiveSlotCount,
            TensorCudaKernels.LiveGradientNormScratchBufferCount);

    /// <summary>
    /// Retires every compatibility resource created without an execution
    /// session. Each cache immediately installs a fresh bounded generation;
    /// outstanding leases defer only their own values until they return.
    /// </summary>
    internal static void DisposeFallbackResources()
    {
        List<Exception>? failures = null;
        TryDisposeFallback(CudaBlas.DisposeFallbackResources, ref failures);
        TryDisposeFallback(CudaBlasLt.DisposeFallbackResources, ref failures);
        TryDisposeFallback(
            CudaBlasLtInt8.DisposeFallbackResources,
            ref failures);
        TryDisposeFallback(
            CudaLayerNorm.DisposeFallbackResources,
            ref failures);
        TryDisposeFallback(
            NativeCudaScalarReadback.DisposeFallbackResources,
            ref failures);
        TryDisposeFallback(
            NativeCudaIntScalarReadback.DisposeFallbackResources,
            ref failures);
        TryDisposeFallback(
            TensorCudaKernels.DisposeFallbackResources,
            ref failures);
        if (failures is not null)
        {
            throw new AggregateException(
                "One or more legacy CUDA fallback resources failed to dispose.",
                failures);
        }
    }

    private static void TryDisposeFallback(
        Action dispose,
        ref List<Exception>? failures)
    {
        try
        {
            dispose();
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
    }

    internal static void RecordAllocation(nuint bytes)
    {
        Interlocked.Increment(ref _allocationCount);
        Interlocked.Add(ref _allocationBytes, checked((long)bytes));
    }

    internal static void RecordFree(nuint bytes)
    {
        Interlocked.Increment(ref _freeCount);
        Interlocked.Add(ref _freeBytes, checked((long)bytes));
    }

    internal static bool CanAccessPeer(int device, int peerDevice)
    {
        Check(CanAccessPeerNative(device, peerDevice, out int canAccess),
            "cudaDeviceCanAccessPeer");
        return canAccess != 0;
    }

    private static int QueryDeviceCount()
    {
        Check(DeviceCountNative(out int count), "cudaGetDeviceCount");
        return count;
    }

    internal static NativeCudaDevice GetDevice(int index)
    {
        int count = DeviceCount;
        if ((uint)index >= (uint)count)
            throw new ArgumentOutOfRangeException(nameof(index));
        return NativeCudaDevice.GetOrCreate(index);
    }

    internal static string GetErrorString(int status)
        => CudaNativeGateway.ErrorString(status);

    internal static void Check(int status, string operation)
    {
        if (status != 0)
            throw new NativeCudaException(operation, status);
    }

    internal static nint AllocateWithNotReadyRetry(
        NativeCudaDevice device,
        nuint bytes)
    {
        ArgumentNullException.ThrowIfNull(device);
        int status = AllocateNative(device.Index, bytes, out nint pointer);
        if (status == NotReadyStatus)
        {
            // cudaErrorNotReady is not an OOM and does not indicate an
            // invalid pointer. Wait for all queued work once, then retry the
            // allocation. A real asynchronous kernel failure is returned by
            // cudaDeviceSynchronize and is deliberately not hidden.
            Check(
                SynchronizeNative(device.Index),
                $"cudaDeviceSynchronize before cudaMalloc retry " +
                $"(device {device.Index})");
            status = AllocateNative(device.Index, bytes, out pointer);
        }
        if (status == OutOfMemoryStatus)
        {
            // Exact-shape activation pools contain only idle allocations and
            // are recoverable. Persistent optimizer state, cuBLAS workspaces,
            // and other direct allocations also pass through this method; do
            // not fail them while several GiB of reusable cache can be
            // reclaimed. Pool disposal calls cudaFree only, so this recovery
            // path cannot recurse back into allocation.
            Tensor.ClearCudaFloatBufferPool(device.Index);
            status = AllocateNative(device.Index, bytes, out pointer);
        }
        Check(
            status,
            $"cudaMalloc (device {device.Index}, {bytes:N0} bytes)");
        return pointer;
    }

    private static int DeviceCountNative(out int count)
        => CudaNativeGateway.DeviceCount(out count);

    internal static int DeviceName(
        int device,
        StringBuilder destination,
        int capacity)
        => CudaNativeGateway.DeviceName(device, destination, capacity);

    internal static int SetDeviceNative(int device)
        => CudaNativeGateway.SetDevice(device);

    internal static int UseExternalStreamNative(nint stream)
        => CudaNativeGateway.UseExternalStream(stream);

    internal static void BindExecutionLane(IStreamExecutionLane lane)
    {
        ArgumentNullException.ThrowIfNull(lane);
        PreparedStreamState? prepared = _preparedStream;
        if (prepared is { IsLegacyDefault: false } current
            && ReferenceEquals(current.Lane, lane)
            && current.DeviceIndex == lane.DeviceIndex
            && PreparedNativeContextMatches(
                current,
                lane.DeviceIndex,
                lane.ComputeStreamHandle))
        {
            return;
        }
        lane.ActivateComputeStream();
        _preparedStream = PreparedStreamState.ForLane(
            lane,
            CaptureNativeContextGeneration(lane));
    }

    internal static void BindDeviceAndComputeStream(int deviceIndex)
    {
        if (TensorExecutionContext.TryGetCudaStreamLane(
                deviceIndex,
                out IStreamExecutionLane lane))
        {
            BindExecutionLane(lane);
            return;
        }

        if (_preparedStream is { IsLegacyDefault: true } prepared
            && prepared.DeviceIndex == deviceIndex
            && PreparedNativeContextMatches(
                prepared,
                deviceIndex,
                nint.Zero))
            return;

        Check(
            CudaNativeGateway.SetDevice(deviceIndex),
            $"cudaSetDevice (device {deviceIndex})");
        Check(
            CudaNativeGateway.UseExternalStream(nint.Zero),
            $"restore legacy CUDA stream (device {deviceIndex})");
        _preparedStream = PreparedStreamState.ForLegacyDefault(
            deviceIndex,
            CudaNativeGateway.CurrentThreadContext.Generation);
    }

    internal static nint ResolveComputeStream(int deviceIndex)
    {
        if (TensorExecutionContext.TryGetCudaStreamLane(
                deviceIndex,
                out IStreamExecutionLane lane))
        {
            BindExecutionLane(lane);
            return lane.ComputeStreamHandle;
        }

        BindDeviceAndComputeStream(deviceIndex);
        return nint.Zero;
    }

    internal static bool TryResolveCommunicationStream(
        int deviceIndex,
        out nint stream)
    {
        if (TensorExecutionContext.TryGetCudaStreamLane(
                deviceIndex,
                out IStreamExecutionLane lane))
        {
            stream = lane.CommunicationStreamHandle;
            return true;
        }
        stream = nint.Zero;
        return false;
    }

    private static bool PreparedNativeContextMatches(
        PreparedStreamState prepared,
        int deviceIndex,
        nint stream)
    {
        if (prepared.NativeContextGeneration < 0)
            return true;
        CudaNativeThreadContextSnapshot context =
            CudaNativeGateway.CurrentThreadContext;
        return context.Generation == prepared.NativeContextGeneration
            && context.HasSelectedDevice
            && context.SelectedDevice == deviceIndex
            && context.HasExternalStream
            && context.ExternalStream == stream;
    }

    private static long CaptureNativeContextGeneration(
        IStreamExecutionLane lane)
        => lane is CudaExecutionLane
            ? CudaNativeGateway.CurrentThreadContext.Generation
            : -1;

    private readonly record struct PreparedStreamState(
        IStreamExecutionLane? Lane,
        int DeviceIndex,
        bool IsLegacyDefault,
        long NativeContextGeneration)
    {
        internal static PreparedStreamState ForLane(
            IStreamExecutionLane lane,
            long nativeContextGeneration)
            => new(
                lane,
                lane.DeviceIndex,
                IsLegacyDefault: false,
                nativeContextGeneration);

        internal static PreparedStreamState ForLegacyDefault(
            int deviceIndex,
            long nativeContextGeneration)
            => new(
                null,
                deviceIndex,
                IsLegacyDefault: true,
                nativeContextGeneration);
    }

    internal static int SynchronizeNative(int device)
        => CudaNativeGateway.Synchronize(device);

    internal static int MemoryInfoNative(
        int device,
        out nuint freeBytes,
        out nuint totalBytes)
        => CudaNativeGateway.MemoryInfo(
            device,
            out freeBytes,
            out totalBytes);

    internal static int AllocateNative(
        int device,
        nuint bytes,
        out nint pointer)
        => CudaNativeGateway.Allocate(device, bytes, out pointer);

    internal static int FreeNative(int device, nint pointer)
        => CudaNativeGateway.Free(device, pointer);

    internal static int MemsetNative(
        int device,
        nint destination,
        int value,
        nuint bytes)
        => CudaNativeGateway.Memset(
            device,
            destination,
            value,
            bytes);

    internal static int MemsetAsyncNative(
        int device,
        nint destination,
        int value,
        nuint bytes,
        nint stream)
    {
        int status = CudaNativeGateway.MemsetAsync(
            device,
            destination,
            value,
            bytes,
            stream);
        if (status == 0)
        {
            Interlocked.Increment(ref _memsetLaunchCount);
            Interlocked.Add(ref _memsetBytes, checked((long)bytes));
        }
        return status;
    }

    internal static int CopyHostToDeviceNative(
        int device,
        nint destination,
        nint source,
        nuint bytes)
    {
        DeviceTransferGuard.BeforeHostToDevice(
            bytes,
            "cudaMemcpy(H2D)");
        int status = CudaNativeGateway.CopyHostToDevice(
            device,
            destination,
            source,
            bytes);
        if (status == 0)
            RecordHostToDevice(bytes);
        return status;
    }

    internal static int CopyHostToDeviceAsyncNative(
        int device,
        nint destination,
        nint source,
        nuint bytes,
        nint stream)
    {
        DeviceTransferGuard.BeforeHostToDevice(
            bytes,
            "cudaMemcpyAsync(H2D)");
        int status = CudaNativeGateway.CopyHostToDeviceAsync(
            device,
            destination,
            source,
            bytes,
            stream);
        if (status == 0)
            RecordHostToDevice(bytes);
        return status;
    }

    internal static int CopyDeviceToHostNative(
        int device,
        nint destination,
        nint source,
        nuint bytes)
    {
        DeviceTransferGuard.BeforeDeviceToHost(
            bytes,
            "cudaMemcpy(D2H)");
        int status = CudaNativeGateway.CopyDeviceToHost(
            device,
            destination,
            source,
            bytes);
        if (status == 0)
            RecordDeviceToHost(bytes);
        return status;
    }

    internal static int CopyDeviceToHostAsyncNative(
        int device,
        nint destination,
        nint source,
        nuint bytes,
        nint stream)
    {
        DeviceTransferGuard.BeforeDeviceToHost(
            bytes,
            "cudaMemcpyAsync(D2H)");
        int status = CudaNativeGateway.CopyDeviceToHostAsync(
            device,
            destination,
            source,
            bytes,
            stream);
        if (status == 0)
            RecordDeviceToHost(bytes);
        return status;
    }

    internal static int HostAllocateNative(nuint bytes, out nint pointer)
        => CudaNativeGateway.HostAllocate(bytes, out pointer);

    internal static int HostFreeNative(nint pointer)
        => CudaNativeGateway.HostFree(pointer);

    internal static int EventCreateNative(int device, out nint cudaEvent)
        => CudaNativeGateway.EventCreate(device, out cudaEvent);

    internal static int EventDestroyNative(int device, nint cudaEvent)
        => CudaNativeGateway.EventDestroy(device, cudaEvent);

    internal static int CopyDeviceToHostAsyncRecordNative(
        int device,
        nint destination,
        nint source,
        nuint bytes,
        nint stream,
        nint cudaEvent)
    {
        DeviceTransferGuard.BeforeDeviceToHost(
            bytes,
            "cudaMemcpyAsyncRecord(D2H)");
        int status = CudaNativeGateway.CopyDeviceToHostAsyncRecord(
            device,
            destination,
            source,
            bytes,
            stream,
            cudaEvent);
        if (status == 0)
            RecordDeviceToHost(bytes);
        return status;
    }

    internal static int CopyHostToDeviceAsyncRecordNative(
        int device,
        nint destination,
        nint source,
        nuint bytes,
        nint stream,
        nint cudaEvent)
    {
        DeviceTransferGuard.BeforeHostToDevice(
            bytes,
            "cudaMemcpyAsyncRecord(H2D)");
        int status = CudaNativeGateway.CopyHostToDeviceAsyncRecord(
            device,
            destination,
            source,
            bytes,
            stream,
            cudaEvent);
        if (status == 0)
            RecordHostToDevice(bytes);
        return status;
    }

    private static void RecordHostToDevice(nuint bytes)
    {
        DeviceTransferGuard.RecordHostToDevice(bytes);
        Interlocked.Increment(ref _hostToDeviceCopyCount);
        Interlocked.Add(ref _hostToDeviceBytes, checked((long)bytes));
    }

    private static void RecordDeviceToHost(nuint bytes)
    {
        Interlocked.Increment(ref _deviceToHostCopyCount);
        Interlocked.Add(ref _deviceToHostBytes, checked((long)bytes));
    }

    /// <summary>
    /// Records one successfully completed native no-P2P BF16 host pipeline.
    /// The native implementation performs the same number and byte volume of
    /// D2H and H2D chunk copies. This method is called only after that native
    /// operation returns success, avoiding telemetry for rejected launches.
    /// </summary>
    internal static void RecordGradientCollectiveHostPipeline(
        long copyCount,
        long byteLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(copyCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(byteLength);

        Interlocked.Add(ref _hostToDeviceCopyCount, copyCount);
        Interlocked.Add(ref _hostToDeviceBytes, byteLength);
        Interlocked.Add(ref _deviceToHostCopyCount, copyCount);
        Interlocked.Add(ref _deviceToHostBytes, byteLength);
        Interlocked.Add(
            ref _gradientCollectiveHostToDeviceCopyCount,
            copyCount);
        Interlocked.Add(
            ref _gradientCollectiveHostToDeviceBytes,
            byteLength);
        Interlocked.Add(
            ref _gradientCollectiveDeviceToHostCopyCount,
            copyCount);
        Interlocked.Add(
            ref _gradientCollectiveDeviceToHostBytes,
            byteLength);
    }

    internal static int EventSynchronizeNative(int device, nint cudaEvent)
        => CudaNativeGateway.EventSynchronize(device, cudaEvent);

    internal static int CopyDeviceToDeviceNative(
        int destinationDevice,
        nint destination,
        int sourceDevice,
        nint source,
        nuint bytes)
        => CudaNativeGateway.CopyDeviceToDevice(
            destinationDevice,
            destination,
            sourceDevice,
            source,
            bytes);

    internal static int CopyDeviceToDeviceAsyncNative(
        int destinationDevice,
        nint destination,
        int sourceDevice,
        nint source,
        nuint bytes,
        nint stream)
        => CudaNativeGateway.CopyDeviceToDeviceAsync(
            destinationDevice,
            destination,
            sourceDevice,
            source,
            bytes,
            stream);

    internal static void SynchronizeComputeStream(
        NativeCudaDevice accelerator,
        nint computeStream)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        if (TensorExecutionContext.TryGetCudaStreamLane(
                accelerator.Index,
                out IStreamExecutionLane lane)
            && lane.ComputeStreamHandle == computeStream)
        {
            lane.SynchronizeComputeStream();
            return;
        }
        Check(
            CudaNativeGateway.StreamSynchronize(
                accelerator.Index,
                computeStream),
            $"cudaStreamSynchronize (device {accelerator.Index})");
    }

    internal static void SynchronizeDeviceComputeStream(int deviceIndex)
    {
        NativeCudaDevice accelerator = GetDevice(deviceIndex);
        accelerator.Bind();
        nint stream = accelerator.DefaultStream;
        SynchronizeComputeStream(accelerator, stream);
    }

    /// <summary>
    /// Retires a stream-visible native resource only after all preceding work
    /// has crossed a CUDA event. A direct stream synchronization is used only
    /// as recovery when event creation/record/wait fails; cleanup failures are
    /// aggregated without skipping an independently safe release.
    /// </summary>
    internal static void DisposeAfterStreamFence(
        int deviceIndex,
        nint computeStream,
        Action release)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceIndex);
        ArgumentNullException.ThrowIfNull(release);

        List<Exception>? failures = null;
        bool safeToRelease = false;
        CudaEventCompletionFence? fence = null;
        try
        {
            fence = CudaEventCompletionFence.Record(
                deviceIndex,
                computeStream);
            fence.Wait();
            safeToRelease = true;
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
            try
            {
                Check(
                    CudaNativeGateway.StreamSynchronize(
                        deviceIndex,
                        computeStream),
                    $"cudaStreamSynchronize(resource retirement, device " +
                    $"{deviceIndex})");
                safeToRelease = true;
            }
            catch (Exception synchronizationFailure)
            {
                failures.Add(synchronizationFailure);
            }
        }
        finally
        {
            if (fence is not null)
            {
                try
                {
                    fence.Dispose();
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
            }
        }

        if (safeToRelease)
        {
            try
            {
                release();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }

        if (failures is null)
            return;
        if (failures.Count == 1)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(failures[0])
                .Throw();
        }
        throw new AggregateException(
            "A CUDA stream resource failed to retire cleanly.",
            failures);
    }

    private static int CanAccessPeerNative(
        int device,
        int peerDevice,
        out int canAccess)
        => CudaNativeGateway.CanAccessPeer(
            device,
            peerDevice,
            out canAccess);

    internal static CudaKernelCapabilities GetKernelCapabilities(int device)
    {
        Check(
            CudaNativeGateway.KernelCapabilities(
                device,
                out CudaKernelCapabilities capabilities),
            "cudaGetDeviceProperties(capabilities)");
        return capabilities;
    }
}

/// <summary>
/// Reusable pinned scalar readback. The copy and its event are queued before
/// backward; the CPU waits only after all backward kernels have been queued.
/// </summary>
internal sealed unsafe class NativeCudaScalarReadback
{
    private const int IdleSlotCapacity = 8;
    private const int FallbackPoolCapacity = 4;
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<
        IStreamExecutionLane,
        Lazy<NativeCudaScalarReadbackPool>> LanePools = new();
    private static readonly ResettableBoundedDisposableLeaseCache<
        int,
        NativeCudaScalarReadbackPool> FallbackPools =
            new(FallbackPoolCapacity);
    private static int _activeLanePoolCount;
    private static int _liveSlotCount;

    private readonly NativeCudaScalarReadbackPool _owner;
    private readonly int _device;
    private readonly nint _host;
    private readonly nint _event;
    private BoundedDisposableLeaseCache<
        int,
        NativeCudaScalarReadbackPool>.Lease? _fallbackLease;
    private bool _pending;
    private int _checkedOut;
    private int _disposed;

    private NativeCudaScalarReadback(
        NativeCudaScalarReadbackPool owner,
        int device)
    {
        _owner = owner;
        _device = device;
        NativeCudaRuntime.Check(
            NativeCudaRuntime.HostAllocateNative(sizeof(float), out _host),
            "cudaMallocHost(loss scalar)");
        try
        {
            NativeCudaRuntime.Check(
                NativeCudaRuntime.EventCreateNative(device, out _event),
                "cudaEventCreate(loss scalar)");
        }
        catch
        {
            NativeCudaRuntime.HostFreeNative(_host);
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

    internal static NativeCudaScalarReadback Rent(int device)
    {
        NativeCudaScalarReadbackPool? pool = null;
        BoundedDisposableLeaseCache<
            int,
            NativeCudaScalarReadbackPool>.Lease? fallbackLease = null;
        if (TensorExecutionContext.TryGetCudaStreamLane(
                device,
                out IStreamExecutionLane lane))
        {
            pool = LanePools.GetValue(
                lane,
                static owner => new Lazy<NativeCudaScalarReadbackPool>(
                    () => ExecutionLaneResources.Attach(
                        owner,
                        new NativeCudaScalarReadbackPool(
                            owner.DeviceIndex,
                            laneOwned: true)),
                    LazyThreadSafetyMode.ExecutionAndPublication)).Value;
        }
        else
        {
            fallbackLease = FallbackPools.Acquire(
                device,
                static value => new NativeCudaScalarReadbackPool(
                    value,
                    laneOwned: false));
            pool = fallbackLease?.Value;
        }

        if (pool is null)
        {
            fallbackLease?.Dispose();
            throw new InvalidOperationException(
                "A CUDA scalar readback pool could not be created.");
        }
        try
        {
            NativeCudaScalarReadback readback = pool.Rent();
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
            throw new InvalidOperationException("Scalar readback is not rented.");
        if (_pending)
            throw new InvalidOperationException("Scalar readback is pending.");
        try
        {
            NativeCudaRuntime.Check(
                NativeCudaRuntime.CopyDeviceToHostAsyncRecordNative(
                    _device,
                    _host,
                    deviceSource,
                    sizeof(float),
                    stream,
                    _event),
                "cudaMemcpyAsync(D2H loss scalar)");
            _pending = true;
        }
        catch
        {
            ReturnToOwner(reusable: true);
            throw;
        }
    }

    internal float CompleteAndReturn()
    {
        if (!_pending)
            throw new InvalidOperationException("Scalar readback was not started.");
        Exception? failure = null;
        float value = 0f;
        try
        {
            NativeCudaRuntime.Check(
                NativeCudaRuntime.EventSynchronizeNative(_device, _event),
                "cudaEventSynchronize(loss scalar)");
            value = *(float*)_host;
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

    private void ReturnToOwner(bool reusable)
    {
        if (Interlocked.Exchange(ref _checkedOut, 0) == 0)
            return;
        BoundedDisposableLeaseCache<
            int,
            NativeCudaScalarReadbackPool>.Lease? fallback =
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
            throw new InvalidOperationException("Scalar readback is already rented.");
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
                "cudaEventDestroy(loss scalar)",
                eventStatus));
        }
        int hostStatus = NativeCudaRuntime.HostFreeNative(_host);
        if (hostStatus != 0)
        {
            (failures ??= []).Add(new NativeCudaException(
                "cudaFreeHost(loss scalar)",
                hostStatus));
        }
        Interlocked.Decrement(ref _liveSlotCount);
        if (failures is not null)
        {
            throw new AggregateException(
                "CUDA scalar readback cleanup failed.",
                failures);
        }
    }

    private sealed class NativeCudaScalarReadbackPool : IDisposable
    {
        private readonly object _sync = new();
        private readonly Stack<NativeCudaScalarReadback> _idle = [];
        private readonly bool _laneOwned;
        private bool _disposed;

        internal NativeCudaScalarReadbackPool(int device, bool laneOwned)
        {
            Device = device;
            _laneOwned = laneOwned;
            if (laneOwned)
                Interlocked.Increment(ref _activeLanePoolCount);
        }

        internal int Device { get; }

        internal NativeCudaScalarReadback Rent()
        {
            lock (_sync)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                NativeCudaScalarReadback value = _idle.Count == 0
                    ? new NativeCudaScalarReadback(this, Device)
                    : _idle.Pop();
                value.MarkRented();
                return value;
            }
        }

        internal void Return(
            NativeCudaScalarReadback readback,
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
            NativeCudaScalarReadback[] idle;
            lock (_sync)
            {
                if (_disposed)
                    return;
                _disposed = true;
                idle = _idle.ToArray();
                _idle.Clear();
            }
            List<Exception>? failures = null;
            foreach (NativeCudaScalarReadback readback in idle)
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
                    $"CUDA scalar readback pool cleanup failed on device {Device}.",
                    failures);
            }
        }
    }
}

internal sealed unsafe class NativeCudaPinnedUpload<T> : IDisposable
    where T : unmanaged
{
    private readonly int _device;
    private readonly nint _host;
    private readonly nint _event;
    private readonly int _length;
    private readonly nuint _bytes;
    private bool _pending;
    private int _disposed;

    internal NativeCudaPinnedUpload(int device, int length)
    {
        _device = device;
        _length = length;
        _bytes = checked((nuint)length * (nuint)sizeof(T));
        NativeCudaRuntime.Check(
            NativeCudaRuntime.HostAllocateNative(_bytes, out _host),
            "cudaMallocHost(input staging)");
        NativeCudaPinnedUploadTracker.RecordHostAllocation(_bytes);
        try
        {
            NativeCudaRuntime.Check(
                NativeCudaRuntime.EventCreateNative(device, out _event),
                "cudaEventCreate(input staging)");
            NativeCudaPinnedUploadTracker.RecordEventCreate();
        }
        catch (Exception creationFailure)
        {
            int freeStatus = NativeCudaRuntime.HostFreeNative(_host);
            if (freeStatus == 0)
            {
                NativeCudaPinnedUploadTracker.RecordHostFree(_bytes);
                throw;
            }
            throw new AggregateException(
                "Pinned upload construction and rollback both failed.",
                creationFailure,
                new NativeCudaException(
                    "cudaFreeHost(input staging rollback)",
                    freeStatus));
        }
        NativeCudaPinnedUploadTracker.RecordSlotCreated();
    }

    internal void Upload(
        ReadOnlySpan<T> source,
        NativeCudaBuffer<T> destination,
        nint stream)
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0, this);
        if (source.Length != _length || destination.Length != _length)
            throw new ArgumentException("Pinned upload length mismatch.");
        if (_pending)
        {
            NativeCudaPinnedUploadTracker.RecordReuseSynchronization();
            NativeCudaRuntime.Check(
                NativeCudaRuntime.EventSynchronizeNative(_device, _event),
                "cudaEventSynchronize(input staging reuse)");
            _pending = false;
        }
        source.CopyTo(new Span<T>((void*)_host, _length));
        NativeCudaRuntime.Check(
            NativeCudaRuntime.CopyHostToDeviceAsyncRecordNative(
                _device,
                destination.NativePtr,
                _host,
                _bytes,
                stream,
                _event),
            "cudaMemcpyAsync(H2D input)");
        _pending = true;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        List<Exception>? failures = null;
        if (_pending)
        {
            int status = NativeCudaRuntime.EventSynchronizeNative(
                _device,
                _event);
            if (status != 0)
            {
                (failures ??= []).Add(new NativeCudaException(
                    "cudaEventSynchronize(input staging disposal)",
                    status));
            }
            _pending = false;
        }
        int eventStatus = NativeCudaRuntime.EventDestroyNative(
            _device,
            _event);
        if (eventStatus == 0)
        {
            NativeCudaPinnedUploadTracker.RecordEventDestroy();
        }
        else
        {
            (failures ??= []).Add(new NativeCudaException(
                "cudaEventDestroy(input staging)",
                eventStatus));
        }

        int hostStatus = NativeCudaRuntime.HostFreeNative(_host);
        if (hostStatus == 0)
        {
            NativeCudaPinnedUploadTracker.RecordHostFree(_bytes);
        }
        else
        {
            (failures ??= []).Add(new NativeCudaException(
                "cudaFreeHost(input staging)",
                hostStatus));
        }
        NativeCudaPinnedUploadTracker.RecordSlotDisposed();
        if (failures is not null)
        {
            throw new AggregateException(
                "Pinned upload slot cleanup failed.",
                failures);
        }
    }
}

internal sealed class NativeCudaDevice
{
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<
        int,
        NativeCudaDevice> Devices = new();
    private string? _name;
    private long _memorySize;

    private NativeCudaDevice(int index) => Index = index;

    internal int Index { get; }
    internal nint DefaultStream =>
        NativeCudaRuntime.ResolveComputeStream(Index);

    internal string Name
    {
        get
        {
            if (_name is not null)
                return _name;
            var builder = new StringBuilder(256);
            NativeCudaRuntime.Check(
                NativeCudaRuntime.DeviceName(Index, builder, builder.Capacity),
                "cudaGetDeviceProperties");
            return _name = builder.ToString();
        }
    }

    internal long MemorySize
    {
        get
        {
            long cached = Volatile.Read(ref _memorySize);
            if (cached != 0)
                return cached;
            GetMemoryInfo(out _, out long total);
            Interlocked.CompareExchange(ref _memorySize, total, 0);
            return Volatile.Read(ref _memorySize);
        }
    }

    internal static NativeCudaDevice GetOrCreate(int index)
        => Devices.GetOrAdd(index, static value => new NativeCudaDevice(value));

    internal void Bind()
        => NativeCudaRuntime.BindDeviceAndComputeStream(Index);

    internal void Synchronize()
        => Synchronize("cudaDeviceSynchronize");

    internal void Synchronize(string operation)
        => NativeCudaRuntime.Check(
            NativeCudaRuntime.SynchronizeNative(Index),
            operation);

    internal long GetFreeMemory()
    {
        GetMemoryInfo(out long free, out _);
        return free;
    }

    internal NativeCudaBuffer<T> Allocate<T>(
        int length,
        CudaMemoryKind kind = CudaMemoryKind.Persistent)
        where T : unmanaged
        => new(this, length, kind);

    internal NativeCudaBuffer<T> Allocate1D<T>(
        int length,
        CudaMemoryKind kind = CudaMemoryKind.Persistent)
        where T : unmanaged
        => Allocate<T>(length, kind);

    internal NativeCudaBuffer<T> Allocate1D<T>(T[] values) where T : unmanaged
        => Allocate(values.AsSpan());

    internal NativeCudaBuffer<T> Allocate1D<T>(ReadOnlySpan<T> values)
        where T : unmanaged
        => Allocate(values);

    internal NativeCudaBuffer<T> Allocate<T>(ReadOnlySpan<T> values)
        where T : unmanaged
    {
        var buffer = new NativeCudaBuffer<T>(this, values.Length);
        try
        {
            buffer.CopyFromCPU(values);
            return buffer;
        }
        catch
        {
            buffer.Dispose();
            throw;
        }
    }

    private void GetMemoryInfo(out long free, out long total)
    {
        NativeCudaRuntime.Check(
            NativeCudaRuntime.MemoryInfoNative(
                Index, out nuint freeBytes, out nuint totalBytes),
            "cudaMemGetInfo");
        free = checked((long)freeBytes);
        total = checked((long)totalBytes);
    }
}

internal sealed unsafe class NativeCudaBuffer<T> : IDisposable
    where T : unmanaged
{
    private const nuint MaximumPinnedTransferBytes = 16u * 1024u * 1024u;
    private nint _pointer;
    private CudaMemoryLease? _laneMemoryLease;
    private readonly bool _ownsMemory;
    private readonly NativeCudaArena<T>? _arena;
    private readonly long _sessionGeneration;
    private readonly ExecutionSession? _ownerSession;

    internal NativeCudaBuffer(
        NativeCudaDevice device,
        int length,
        CudaMemoryKind kind = CudaMemoryKind.Persistent)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        if (!Enum.IsDefined(kind))
            throw new ArgumentOutOfRangeException(nameof(kind));
        Device = device;
        Length = length;
        AllocationKind = kind;
        _ownsMemory = true;
        nuint bytes = checked((nuint)length * (nuint)sizeof(T));
        if (bytes != 0
            && TensorExecutionContext.TryGetCudaStreamLane(
                device.Index,
                out IStreamExecutionLane lane)
            && lane.MemoryManager is CudaMemoryManager memory)
        {
            ExecutionSession session = ExecutionSession.Current
                ?? throw new InvalidOperationException(
                    "A CUDA execution lane cannot exist without its session authority.");
            _sessionGeneration = session.Generation;
            _ownerSession = session;
            _laneMemoryLease = kind is CudaMemoryKind.Transient
                or CudaMemoryKind.Workspace
                    ? memory.Rent(bytes, kind)
                    : memory.Allocate(bytes, kind);
            _pointer = _laneMemoryLease.Pointer;
        }
        else
        {
            _pointer = NativeCudaRuntime.AllocateWithNotReadyRetry(
                device,
                bytes);
        }
        // Lane-backed allocations are counted by CudaMemoryManager only when
        // its native allocator is actually called.  A reusable cache hit must
        // not appear as another cudaMalloc in benchmark telemetry.
        if (_laneMemoryLease is null)
            NativeCudaRuntime.RecordAllocation(bytes);
    }

    internal NativeCudaBuffer(
        NativeCudaArena<T> arena,
        int offset,
        int length)
    {
        ArgumentNullException.ThrowIfNull(arena);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        if (offset > arena.Length - length)
            throw new ArgumentOutOfRangeException(nameof(length));
        Device = arena.Device;
        Length = length;
        AllocationKind = arena.Buffer.AllocationKind;
        _ownsMemory = false;
        _arena = arena;
        _sessionGeneration = arena.Buffer.SessionGeneration;
        if (arena.Buffer.TryGetOwnerSession(out ExecutionSession? session))
            _ownerSession = session;
        _pointer = arena.NativePtr + checked(offset * sizeof(T));
    }

    internal NativeCudaDevice Device { get; }
    internal int Length { get; }
    internal CudaMemoryKind AllocationKind { get; }
    internal long SessionGeneration => _sessionGeneration;
    internal bool IsAlive
        => Volatile.Read(ref _pointer) != 0
            && (_laneMemoryLease is null || !_laneMemoryLease.IsClosed)
            && (_arena is null || _arena.Buffer.IsAlive);
    internal bool IsLaneManagedReusable
        => _laneMemoryLease is not null
            && AllocationKind is CudaMemoryKind.Transient
                or CudaMemoryKind.Workspace;
    internal NativeCudaView<T> View => new(Device, NativePtr, Length);
    internal nint NativePtr
        => IsAlive
            ? _pointer
            : throw new ObjectDisposedException(nameof(NativeCudaBuffer<T>));
    internal NativeCudaArena<T>? Arena => _arena;

    internal bool TryGetOwnerSession(out ExecutionSession? session)
    {
        if (_ownerSession is not null)
        {
            session = _ownerSession;
            return true;
        }
        session = null;
        return false;
    }

    internal void MemSetToZero()
    {
        Device.Bind();
        nint stream = Device.DefaultStream;
        NativeCudaRuntime.Check(
            NativeCudaRuntime.MemsetAsyncNative(
                Device.Index,
                NativePtr,
                0,
                ByteLength,
                stream),
            "cudaMemsetAsync");
    }

    internal void ClearGradientStorage()
        // Tensor.ZeroGrad() is a per-tensor operation.  An arena-backed
        // buffer is a slice, so clearing its arena would also erase adjacent
        // parameters.  Bulk gradient plans explicitly call
        // NativeCudaArena.ClearIfDirty() when an arena-wide clear is wanted.
        => MemSetToZero();

    internal void MarkGradientStorageDirty() => _arena?.MarkDirty();

    internal void CopyFromCPU(ReadOnlySpan<T> values)
    {
        if (values.Length != Length)
            throw new ArgumentException("Source length must match the CUDA buffer.",
                nameof(values));
        nuint bytes = ByteLength;
        if (bytes <= MaximumPinnedTransferBytes)
        {
            fixed (T* source = values)
            {
                CopyHostToDevice(source, NativePtr, bytes);
            }
            return;
        }

        CopyFromCpuPinnedChunks(values);
    }

    internal void CopyToCPU(Span<T> values)
    {
        if (values.Length != Length)
            throw new ArgumentException(
                "Destination length must match the CUDA buffer.",
                nameof(values));
        nuint bytes = ByteLength;
        if (bytes <= MaximumPinnedTransferBytes)
        {
            fixed (T* destination = values)
            {
                CopyDeviceToHost(NativePtr, destination, bytes);
            }
            return;
        }

        CopyToCpuPinnedChunks(values);
    }

    private void CopyFromCpuPinnedChunks(ReadOnlySpan<T> values)
    {
        int elementsPerChunk = checked((int)Math.Max(
            1u,
            MaximumPinnedTransferBytes / (nuint)sizeof(T)));
        int capacity = Math.Min(values.Length, elementsPerChunk);
        nint staging = 0;
        NativeCudaRuntime.Check(
            NativeCudaRuntime.HostAllocateNative(
                checked((nuint)capacity * (nuint)sizeof(T)),
                out staging),
            "cudaMallocHost(H2D chunk staging)");
        Exception? failure = null;
        try
        {
            Span<T> host = new((void*)staging, capacity);
            for (int offset = 0; offset < values.Length; offset += capacity)
            {
                int count = Math.Min(capacity, values.Length - offset);
                values.Slice(offset, count).CopyTo(host[..count]);
                CopyHostToDevice(
                    (T*)staging,
                    NativePtr + checked(offset * sizeof(T)),
                    checked((nuint)count * (nuint)sizeof(T)));
            }
        }
        catch (Exception exception)
        {
            failure = exception;
        }
        finally
        {
            int status = NativeCudaRuntime.HostFreeNative(staging);
            if (status != 0)
            {
                var cleanup = new NativeCudaException(
                    $"cudaFreeHost(H2D chunk staging) failed with CUDA " +
                    $"error {status}.",
                    status);
                failure = failure is null
                    ? cleanup
                    : new AggregateException(failure, cleanup);
            }
        }
        if (failure is not null)
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(failure).Throw();
    }

    private void CopyToCpuPinnedChunks(Span<T> values)
    {
        int elementsPerChunk = checked((int)Math.Max(
            1u,
            MaximumPinnedTransferBytes / (nuint)sizeof(T)));
        int capacity = Math.Min(values.Length, elementsPerChunk);
        nint staging = 0;
        NativeCudaRuntime.Check(
            NativeCudaRuntime.HostAllocateNative(
                checked((nuint)capacity * (nuint)sizeof(T)),
                out staging),
            "cudaMallocHost(D2H chunk staging)");
        Exception? failure = null;
        try
        {
            Span<T> host = new((void*)staging, capacity);
            for (int offset = 0; offset < values.Length; offset += capacity)
            {
                int count = Math.Min(capacity, values.Length - offset);
                CopyDeviceToHost(
                    NativePtr + checked(offset * sizeof(T)),
                    (T*)staging,
                    checked((nuint)count * (nuint)sizeof(T)));
                host[..count].CopyTo(values.Slice(offset, count));
            }
        }
        catch (Exception exception)
        {
            failure = exception;
        }
        finally
        {
            int status = NativeCudaRuntime.HostFreeNative(staging);
            if (status != 0)
            {
                var cleanup = new NativeCudaException(
                    $"cudaFreeHost(D2H chunk staging) failed with CUDA " +
                    $"error {status}.",
                    status);
                failure = failure is null
                    ? cleanup
                    : new AggregateException(failure, cleanup);
            }
        }
        if (failure is not null)
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(failure).Throw();
    }

    private void CopyHostToDevice(T* source, nint destination, nuint bytes)
    {
        Device.Bind();
        nint stream = Device.DefaultStream;
        NativeCudaRuntime.Check(
            NativeCudaRuntime.CopyHostToDeviceAsyncNative(
                Device.Index,
                destination,
                (nint)source,
                bytes,
                stream),
            "cudaMemcpyAsync(H2D)");
        NativeCudaRuntime.SynchronizeComputeStream(Device, stream);
    }

    private void CopyDeviceToHost(nint source, T* destination, nuint bytes)
    {
        Device.Bind();
        nint stream = Device.DefaultStream;
        NativeCudaRuntime.Check(
            NativeCudaRuntime.CopyDeviceToHostAsyncNative(
                Device.Index,
                (nint)destination,
                source,
                bytes,
                stream),
            "cudaMemcpyAsync(D2H)");
        NativeCudaRuntime.SynchronizeComputeStream(Device, stream);
    }

    internal void CopyTo(NativeCudaBuffer<T> destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (destination.Length != Length)
            throw new ArgumentException(
                "Destination length must match the source buffer.",
                nameof(destination));
        if (Device.Index != destination.Device.Index)
            NativeCudaRuntime.SynchronizeDeviceComputeStream(Device.Index);
        destination.Device.Bind();
        nint stream = destination.Device.DefaultStream;
        NativeCudaRuntime.Check(
            NativeCudaRuntime.CopyDeviceToDeviceAsyncNative(
                destination.Device.Index,
                destination.NativePtr,
                Device.Index,
                NativePtr,
                ByteLength,
                stream),
            "cudaMemcpyAsync(D2D/Peer)");
        NativeCudaRuntime.SynchronizeComputeStream(
            destination.Device,
            stream);
    }

    internal NativeCudaView<T> SubView(int offset, int length)
        => View.SubView(offset, length);

    public void Dispose()
    {
        nint pointer = Interlocked.Exchange(ref _pointer, 0);
        if (pointer == 0)
            return;
        if (_ownsMemory)
        {
            CudaMemoryLease? laneLease = Interlocked.Exchange(
                ref _laneMemoryLease,
                null);
            if (laneLease is not null)
            {
                laneLease.Dispose();
            }
            else
            {
                NativeCudaRuntime.Check(
                    NativeCudaRuntime.FreeNative(Device.Index, pointer),
                    "cudaFree");
                NativeCudaRuntime.RecordFree(ByteLength);
            }
        }
        GC.SuppressFinalize(this);
    }

    private nuint ByteLength => checked((nuint)Length * (nuint)sizeof(T));
}

internal readonly record struct NativeCudaAllocationTelemetry(
    long AllocationCount,
    long AllocationBytes,
    long FreeCount,
    long FreeBytes)
{
    public static NativeCudaAllocationTelemetry operator -(
        NativeCudaAllocationTelemetry left,
        NativeCudaAllocationTelemetry right)
        => new(
            left.AllocationCount - right.AllocationCount,
            left.AllocationBytes - right.AllocationBytes,
            left.FreeCount - right.FreeCount,
            left.FreeBytes - right.FreeBytes);
}

internal readonly record struct NativeCudaTransferTelemetry(
    long HostToDeviceCopyCount,
    long HostToDeviceBytes,
    long DeviceToHostCopyCount,
    long DeviceToHostBytes)
{
    public static NativeCudaTransferTelemetry operator -(
        NativeCudaTransferTelemetry left,
        NativeCudaTransferTelemetry right)
        => new(
            left.HostToDeviceCopyCount - right.HostToDeviceCopyCount,
            left.HostToDeviceBytes - right.HostToDeviceBytes,
            left.DeviceToHostCopyCount - right.DeviceToHostCopyCount,
            left.DeviceToHostBytes - right.DeviceToHostBytes);
}

internal readonly record struct NativeCudaMemsetTelemetry(
    long LaunchCount,
    long Bytes)
{
    public static NativeCudaMemsetTelemetry operator -(
        NativeCudaMemsetTelemetry left,
        NativeCudaMemsetTelemetry right)
        => new(
            left.LaunchCount - right.LaunchCount,
            left.Bytes - right.Bytes);
}

internal readonly record struct NativeCudaFallbackResourceTelemetry(
    int CublasHandleCount,
    int CublasLtResourceCount,
    int CublasLtInt8ResourceCount,
    int LayerNormScratchResourceCount,
    int FloatScalarPoolCount,
    int IntScalarPoolCount,
    int GradientNormScratchResourceCount,
    int FloatScalarSlotCount,
    int IntScalarSlotCount,
    int GradientNormScratchBufferCount)
{
    internal int CachedOwnerCount => checked(
        CublasHandleCount
        + CublasLtResourceCount
        + CublasLtInt8ResourceCount
        + LayerNormScratchResourceCount
        + FloatScalarPoolCount
        + IntScalarPoolCount
        + GradientNormScratchResourceCount);

    internal int LiveSlotCount => checked(
        FloatScalarSlotCount
        + IntScalarSlotCount
        + GradientNormScratchBufferCount);
}

/// <summary>
/// Owns one contiguous CUDA allocation and lends non-owning typed slices to
/// tensors. Reducer-owned arenas use the dirty gate to issue at most one
/// worker-stream memset per accumulation window.
/// </summary>
internal sealed class NativeCudaArena<T> : IDisposable where T : unmanaged
{
    private readonly NativeCudaBuffer<T> _buffer;
    private int _dirty = 1;
    private int _disposed;

    internal NativeCudaArena(NativeCudaDevice device, int length)
    {
        _buffer = device.Allocate1D<T>(length);
    }

    internal NativeCudaDevice Device => _buffer.Device;
    internal int Length => _buffer.Length;
    internal nint NativePtr => _buffer.NativePtr;
    internal NativeCudaBuffer<T> Buffer => _buffer;

    internal NativeCudaBuffer<T> Slice(int offset, int length)
        => new(this, offset, length);

    internal bool ClearIfDirty()
    {
        if (Interlocked.Exchange(ref _dirty, 0) == 0)
            return false;
        _buffer.MemSetToZero();
        return true;
    }

    internal void MarkDirty() => Volatile.Write(ref _dirty, 1);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _buffer.Dispose();
    }
}

internal readonly unsafe struct NativeCudaView<T> where T : unmanaged
{
    internal NativeCudaView(NativeCudaDevice device, nint pointer, int length)
    {
        Device = device;
        NativePtr = pointer;
        Length = length;
    }

    internal NativeCudaDevice Device { get; }
    internal nint NativePtr { get; }
    internal int Length { get; }

    internal NativeCudaView<T> SubView(int offset, int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        if (offset > Length - length)
            throw new ArgumentOutOfRangeException(nameof(length));
        return new NativeCudaView<T>(
            Device,
            NativePtr + checked(offset * sizeof(T)),
            length);
    }

    internal void CopyTo(NativeCudaView<T> destination)
    {
        if (destination.Length != Length)
            throw new ArgumentException(
                "Destination length must match the source view.",
                nameof(destination));
        if (Device.Index != destination.Device.Index)
            NativeCudaRuntime.SynchronizeDeviceComputeStream(Device.Index);
        destination.Device.Bind();
        nint stream = destination.Device.DefaultStream;
        CopyTo(stream, destination);
        NativeCudaRuntime.SynchronizeComputeStream(
            destination.Device,
            stream);
    }

    internal void CopyTo(nint stream, NativeCudaView<T> destination)
    {
        if (destination.Length != Length)
            throw new ArgumentException(
                "Destination length must match the source view.",
                nameof(destination));
        NativeCudaRuntime.Check(
            NativeCudaRuntime.CopyDeviceToDeviceAsyncNative(
                destination.Device.Index,
                destination.NativePtr,
                Device.Index,
                NativePtr,
                checked((nuint)Length * (nuint)sizeof(T)),
                stream),
            "cudaMemcpyAsync(D2D/Peer)");
    }
}
