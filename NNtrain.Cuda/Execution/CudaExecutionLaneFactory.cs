using NNtrain.Cuda.Interop;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain.Cuda.Execution;

/// <summary>
/// Narrow native contract used to construct a CUDA execution lane. Keeping it
/// injectable makes ownership and partial-construction cleanup verifiable on
/// machines without a CUDA adapter.
/// </summary>
public interface ICudaExecutionRuntime : ICudaMemoryAllocator
{
    nint CreateStream(int deviceIndex);
    void DestroyStream(int deviceIndex, nint stream);
    void ActivateStream(int deviceIndex, nint stream);
    void SynchronizeStream(int deviceIndex, nint stream);
    CudaKernelCapabilities GetKernelCapabilities(int deviceIndex);
}

/// <summary>Creates a fully-owned execution lane for one CUDA device.</summary>
public static class CudaExecutionLaneFactory
{
    public static CudaExecutionLane Create(
        int deviceIndex,
        IExecutionProfiler? profiler = null)
        => Create(
            deviceIndex,
            CudaGatewayExecutionRuntime.Instance,
            profiler);

    public static CudaExecutionLane Create(
        int deviceIndex,
        ICudaExecutionRuntime runtime,
        IExecutionProfiler? profiler = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceIndex);
        ArgumentNullException.ThrowIfNull(runtime);

        CudaStreamHandle? compute = null;
        CudaStreamHandle? communication = null;
        CudaMemoryManager? memory = null;
        try
        {
            compute = CreateOwnedStream(deviceIndex, runtime);
            communication = CreateOwnedStream(deviceIndex, runtime);
            memory = new CudaMemoryManager(deviceIndex, runtime);
            CudaKernelCapabilities capabilities =
                runtime.GetKernelCapabilities(deviceIndex);

            var lane = new CudaExecutionLane(
                deviceIndex,
                compute,
                communication,
                memory,
                capabilities,
                profiler,
                runtime.ActivateStream,
                runtime.SynchronizeStream);
            compute = null;
            communication = null;
            memory = null;
            return lane;
        }
        catch (Exception creationFailure)
        {
            List<Exception> failures = [creationFailure];
            if (memory is not null)
                TryCleanup(memory.Dispose, failures);
            if (communication is not null)
                TryCleanup(communication.DisposeChecked, failures);
            if (compute is not null)
                TryCleanup(compute.DisposeChecked, failures);
            if (profiler is not null
                && !ReferenceEquals(profiler, NullExecutionProfiler.Instance))
            {
                TryCleanup(profiler.Dispose, failures);
            }

            if (failures.Count == 1)
                System.Runtime.ExceptionServices.ExceptionDispatchInfo
                    .Capture(creationFailure)
                    .Throw();
            throw new AggregateException(
                $"CUDA lane {deviceIndex} construction and cleanup failed.",
                failures);
        }
    }

    private static CudaStreamHandle CreateOwnedStream(
        int deviceIndex,
        ICudaExecutionRuntime runtime)
    {
        nint stream = runtime.CreateStream(deviceIndex);
        if (stream == nint.Zero)
        {
            throw new InvalidOperationException(
                $"CUDA stream creation returned a null handle for device {deviceIndex}.");
        }
        return new CudaStreamHandle(
            deviceIndex,
            stream,
            runtime.DestroyStream);
    }

    private static void TryCleanup(Action? cleanup, List<Exception> failures)
    {
        if (cleanup is null)
            return;
        try
        {
            cleanup();
        }
        catch (Exception exception)
        {
            failures.Add(exception);
        }
    }
}

internal sealed class CudaGatewayExecutionRuntime : ICudaExecutionRuntime
{
    private const int OutOfMemoryStatus = 2;
    private const int NotReadyStatus = 600;

    private CudaGatewayExecutionRuntime()
    {
    }

    internal static CudaGatewayExecutionRuntime Instance { get; } = new();

    public nint CreateStream(int deviceIndex)
    {
        Check(
            CudaNativeGateway.StreamCreate(deviceIndex, out nint stream),
            $"cudaStreamCreate (device {deviceIndex})");
        return stream;
    }

    public void DestroyStream(int deviceIndex, nint stream)
        => Check(
            CudaNativeGateway.StreamDestroy(deviceIndex, stream),
            $"cudaStreamDestroy (device {deviceIndex})");

    public void ActivateStream(int deviceIndex, nint stream)
    {
        Check(
            CudaNativeGateway.SetDevice(deviceIndex),
            $"cudaSetDevice (device {deviceIndex})");
        Check(
            CudaNativeGateway.UseExternalStream(stream),
            $"use external CUDA stream (device {deviceIndex})");
    }

    public void SynchronizeStream(int deviceIndex, nint stream)
        => Check(
            CudaNativeGateway.StreamSynchronize(deviceIndex, stream),
            $"cudaStreamSynchronize (device {deviceIndex})");

    public CudaKernelCapabilities GetKernelCapabilities(int deviceIndex)
    {
        Check(
            CudaNativeGateway.KernelCapabilities(
                deviceIndex,
                out CudaKernelCapabilities capabilities),
            $"CUDA capability query (device {deviceIndex})");
        return capabilities;
    }

    public nint Allocate(
        int deviceIndex,
        nuint byteLength,
        CudaMemoryKind kind)
    {
        int status = CudaNativeGateway.Allocate(
            deviceIndex,
            byteLength,
            out nint pointer);
        if (status == NotReadyStatus)
        {
            Check(
                CudaNativeGateway.Synchronize(deviceIndex),
                $"cudaDeviceSynchronize before allocation retry (device {deviceIndex})");
            status = CudaNativeGateway.Allocate(
                deviceIndex,
                byteLength,
                out pointer);
        }
        if (status == OutOfMemoryStatus)
        {
            throw new OutOfMemoryException(
                $"CUDA allocation failed on device {deviceIndex} for " +
                $"{byteLength:N0} bytes ({kind}).");
        }
        Check(
            status,
            $"cudaMalloc (device {deviceIndex}, {byteLength:N0} bytes, {kind})");
        return pointer;
    }

    public void Release(
        int deviceIndex,
        nint pointer,
        nuint byteLength,
        CudaMemoryKind kind)
        => Check(
            CudaNativeGateway.Free(deviceIndex, pointer),
            $"cudaFree (device {deviceIndex}, {byteLength:N0} bytes, {kind})");

    private static void Check(int status, string operation)
    {
        if (status == 0)
            return;
        throw new InvalidOperationException(
            $"{operation} failed with CUDA error {status}: " +
            CudaNativeGateway.ErrorString(status));
    }
}
