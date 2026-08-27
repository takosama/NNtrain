using System.Text;
using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Interop;

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

    internal static int DeviceCount => CachedDeviceCount.Value;

    internal static NativeCudaAllocationTelemetry AllocationTelemetry
        => new(
            Interlocked.Read(ref _allocationCount),
            Interlocked.Read(ref _allocationBytes),
            Interlocked.Read(ref _freeCount),
            Interlocked.Read(ref _freeBytes));

    internal static NativeCudaTransferTelemetry TransferTelemetry
        => new(
            Interlocked.Read(ref _hostToDeviceCopyCount),
            Interlocked.Read(ref _hostToDeviceBytes),
            Interlocked.Read(ref _deviceToHostCopyCount),
            Interlocked.Read(ref _deviceToHostBytes));

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

    internal static int CopyHostToDeviceNative(
        int device,
        nint destination,
        nint source,
        nuint bytes)
    {
        int status = CudaNativeGateway.CopyHostToDevice(
            device,
            destination,
            source,
            bytes);
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
        int status = CudaNativeGateway.CopyDeviceToHost(
            device,
            destination,
            source,
            bytes);
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
        Interlocked.Increment(ref _hostToDeviceCopyCount);
        Interlocked.Add(ref _hostToDeviceBytes, checked((long)bytes));
    }

    private static void RecordDeviceToHost(nuint bytes)
    {
        Interlocked.Increment(ref _deviceToHostCopyCount);
        Interlocked.Add(ref _deviceToHostBytes, checked((long)bytes));
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
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<
        int,
        System.Collections.Concurrent.ConcurrentBag<NativeCudaScalarReadback>>
        Pools = new();
    private readonly int _device;
    private readonly nint _host;
    private readonly nint _event;
    private bool _pending;

    private NativeCudaScalarReadback(int device)
    {
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
    }

    internal static NativeCudaScalarReadback Rent(int device)
    {
        var pool = Pools.GetOrAdd(device, static _ => []);
        return pool.TryTake(out NativeCudaScalarReadback? readback)
            ? readback
            : new NativeCudaScalarReadback(device);
    }

    internal void Begin(nint deviceSource, nint stream)
    {
        if (_pending)
            throw new InvalidOperationException("Scalar readback is pending.");
        NativeCudaRuntime.Check(
            NativeCudaRuntime.CopyDeviceToHostAsyncRecordNative(
                _device, _host, deviceSource, sizeof(float), stream, _event),
            "cudaMemcpyAsync(D2H loss scalar)");
        _pending = true;
    }

    internal float CompleteAndReturn()
    {
        if (!_pending)
            throw new InvalidOperationException("Scalar readback was not started.");
        NativeCudaRuntime.Check(
            NativeCudaRuntime.EventSynchronizeNative(_device, _event),
            "cudaEventSynchronize(loss scalar)");
        _pending = false;
        float value = *(float*)_host;
        Pools.GetOrAdd(_device, static _ => []).Add(this);
        return value;
    }
}

internal sealed unsafe class NativeCudaPinnedUpload<T> : IDisposable
    where T : unmanaged
{
    private readonly int _device;
    private readonly nint _host;
    private readonly nint _event;
    private readonly int _length;
    private bool _pending;
    private int _disposed;

    internal NativeCudaPinnedUpload(int device, int length)
    {
        _device = device;
        _length = length;
        nuint bytes = checked((nuint)length * (nuint)sizeof(T));
        NativeCudaRuntime.Check(
            NativeCudaRuntime.HostAllocateNative(bytes, out _host),
            "cudaMallocHost(input staging)");
        try
        {
            NativeCudaRuntime.Check(
                NativeCudaRuntime.EventCreateNative(device, out _event),
                "cudaEventCreate(input staging)");
        }
        catch
        {
            NativeCudaRuntime.HostFreeNative(_host);
            throw;
        }
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
            NativeCudaRuntime.Check(
                NativeCudaRuntime.EventSynchronizeNative(_device, _event),
                "cudaEventSynchronize(input staging reuse)");
        }
        source.CopyTo(new Span<T>((void*)_host, _length));
        NativeCudaRuntime.Check(
            NativeCudaRuntime.CopyHostToDeviceAsyncRecordNative(
                _device,
                destination.NativePtr,
                _host,
                checked((nuint)_length * (nuint)sizeof(T)),
                stream,
                _event),
            "cudaMemcpyAsync(H2D input)");
        _pending = true;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        if (_pending)
        {
            _ = NativeCudaRuntime.EventSynchronizeNative(_device, _event);
            _pending = false;
        }
        _ = NativeCudaRuntime.EventDestroyNative(_device, _event);
        _ = NativeCudaRuntime.HostFreeNative(_host);
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
    internal nint DefaultStream => 0;

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
        => NativeCudaRuntime.Check(
            NativeCudaRuntime.SetDeviceNative(Index), "cudaSetDevice");

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

    internal NativeCudaBuffer<T> Allocate<T>(int length) where T : unmanaged
        => new(this, length);

    internal NativeCudaBuffer<T> Allocate1D<T>(int length) where T : unmanaged
        => Allocate<T>(length);

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
    private nint _pointer;
    private readonly bool _ownsMemory;
    private readonly NativeCudaArena<T>? _arena;

    internal NativeCudaBuffer(NativeCudaDevice device, int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        Device = device;
        Length = length;
        _ownsMemory = true;
        nuint bytes = checked((nuint)length * (nuint)sizeof(T));
        _pointer = NativeCudaRuntime.AllocateWithNotReadyRetry(device, bytes);
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
        _ownsMemory = false;
        _arena = arena;
        _pointer = arena.NativePtr + checked(offset * sizeof(T));
    }

    internal NativeCudaDevice Device { get; }
    internal int Length { get; }
    internal NativeCudaView<T> View => new(Device, NativePtr, Length);
    internal nint NativePtr
        => _pointer != 0
            ? _pointer
            : throw new ObjectDisposedException(nameof(NativeCudaBuffer<T>));
    internal NativeCudaArena<T>? Arena => _arena;

    internal void MemSetToZero()
        => NativeCudaRuntime.Check(
            NativeCudaRuntime.MemsetNative(
                Device.Index, NativePtr, 0, ByteLength),
            "cudaMemset");

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
        fixed (T* source = values)
        {
            NativeCudaRuntime.Check(
                NativeCudaRuntime.CopyHostToDeviceNative(
                    Device.Index, NativePtr, (nint)source, ByteLength),
                "cudaMemcpy(H2D)");
        }
    }

    internal void CopyToCPU(Span<T> values)
    {
        if (values.Length != Length)
            throw new ArgumentException(
                "Destination length must match the CUDA buffer.",
                nameof(values));
        fixed (T* destination = values)
        {
            NativeCudaRuntime.Check(
                NativeCudaRuntime.CopyDeviceToHostNative(
                    Device.Index, (nint)destination, NativePtr, ByteLength),
                "cudaMemcpy(D2H)");
        }
    }

    internal void CopyTo(NativeCudaBuffer<T> destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (destination.Length != Length)
            throw new ArgumentException(
                "Destination length must match the source buffer.",
                nameof(destination));
        NativeCudaRuntime.Check(
            NativeCudaRuntime.CopyDeviceToDeviceNative(
                destination.Device.Index, destination.NativePtr,
                Device.Index, NativePtr, ByteLength),
            "cudaMemcpy(D2D/Peer)");
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
            NativeCudaRuntime.Check(
                NativeCudaRuntime.FreeNative(Device.Index, pointer), "cudaFree");
            NativeCudaRuntime.RecordFree(ByteLength);
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

/// <summary>
/// Owns one contiguous CUDA allocation and lends non-owning typed slices to
/// tensors. The dirty gate turns parameter-by-parameter zero_grad calls into
/// one memset for the entire arena.
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

    internal void ClearIfDirty()
    {
        if (Interlocked.Exchange(ref _dirty, 0) != 0)
            _buffer.MemSetToZero();
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
        NativeCudaRuntime.Check(
            NativeCudaRuntime.CopyDeviceToDeviceNative(
                destination.Device.Index,
                destination.NativePtr,
                Device.Index,
                NativePtr,
                checked((nuint)Length * (nuint)sizeof(T))),
            "cudaMemcpy(D2D/Peer)");
    }

    internal void CopyTo(nint stream, NativeCudaView<T> destination)
        => CopyTo(destination);
}
