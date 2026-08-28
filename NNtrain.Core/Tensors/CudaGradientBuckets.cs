using NNtrain.Cuda.Interop;
using NNtrain.Runtime.Execution;

namespace NNtrain;

internal static class CudaGradientBuckets
{
    internal static nint CreateCommunicationStream(
        NativeCudaDevice accelerator,
        int device)
    {
        accelerator.Bind();
        ThrowIfFailed(
            CudaNativeGateway.GradientCommunicationStreamCreate(
                device, out nint stream),
            "create gradient communication stream");
        return stream;
    }

    internal static nint CreateReadyEvent(
        NativeCudaDevice accelerator,
        int device)
    {
        accelerator.Bind();
        ThrowIfFailed(
            CudaNativeGateway.GradientReadyEventCreate(
                device, out nint readyEvent),
            "create gradient ready event");
        return readyEvent;
    }

    internal static void Pack(
        int device,
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> source,
        NativeCudaBuffer<ushort> destination,
        int destinationOffset,
        int length)
    {
        accelerator.Bind();
        ThrowIfFailed(CudaNativeGateway.GradientPackBFloat16(
            device,
            source.NativePtr,
            destination.NativePtr,
            destinationOffset,
            length,
            accelerator.DefaultStream),
            "pack BF16 gradient bucket");
    }

    internal static void RecordReady(
        int device,
        NativeCudaDevice accelerator,
        nint readyEvent)
    {
        accelerator.Bind();
        ThrowIfFailed(CudaNativeGateway.GradientRecordReady(
            device,
            readyEvent,
            accelerator.DefaultStream),
            "record gradient ready event");
    }

    internal static void RecordReadyExternal(
        int device,
        NativeCudaDevice accelerator,
        nint readyEvent)
    {
        accelerator.Bind();
        ThrowIfFailed(CudaNativeGateway.GradientRecordReadyExternal(
            device,
            readyEvent,
            accelerator.DefaultStream),
            "record external CUDA Graph gradient ready event");
    }

    internal static void Exchange(
        NativeCudaDevice destinationAccelerator,
        int destinationDevice,
        int sourceDevice,
        NativeCudaBuffer<ushort> local,
        NativeCudaBuffer<ushort> remoteSource,
        NativeCudaBuffer<ushort> remoteStaging,
        NativeCudaBuffer<float> reduced,
        int length,
        nint squaredSum,
        nint communicationStream,
        nint localReadyEvent,
        nint remoteReadyEvent)
    {
        destinationAccelerator.Bind();
        ThrowIfFailed(CudaNativeGateway.GradientExchangeBFloat16(
            destinationDevice,
            sourceDevice,
            local.NativePtr,
            remoteSource.NativePtr,
            remoteStaging.NativePtr,
            reduced.NativePtr,
            length,
            squaredSum,
            communicationStream,
            localReadyEvent,
            remoteReadyEvent),
            "exchange BF16 gradient bucket");
    }

    internal static nint CreateHostPipeline(
        int sourceDevice,
        int destinationDevice,
        int chunkElements)
    {
        NativeCudaRuntime.Check(
            NativeCudaRuntime.SetDeviceNative(destinationDevice),
            "cudaSetDevice(gradient host pipeline)");
        ThrowIfFailed(CudaNativeGateway.GradientHostPipelineCreate(
            sourceDevice,
            destinationDevice,
            chunkElements,
            out nint pipeline),
            "create gradient host pipeline");
        return pipeline;
    }

    internal static void HostPipelineExchange(
        NativeCudaDevice destinationAccelerator,
        int sourceDevice,
        int chunkElements,
        nint pipeline,
        NativeCudaBuffer<ushort> local,
        NativeCudaBuffer<ushort> remoteSource,
        NativeCudaBuffer<float> reduced,
        int length,
        nint squaredSum,
        nint localReadyEvent,
        nint remoteReadyEvent)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(sourceDevice);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(chunkElements);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        long physicalCopyCount = checked(
            ((long)length - 1L) / chunkElements + 1L);
        long physicalBytes = checked((long)length * sizeof(ushort));
        DeviceTransferGuard.GradientCollectiveTransportReservation?
            transport = DeviceTransferGuard
                .ReserveGradientCollectiveTransport(
                    sourceDevice,
                    destinationAccelerator.Index,
                    physicalCopyCount,
                    physicalBytes);

        // The native host pipeline switches between the source and
        // destination device. Bind first so older native binaries that used
        // direct cudaSetDevice calls still leave the managed bridge cache and
        // the CUDA runtime on the same destination device when they return.
        destinationAccelerator.Bind();
        ThrowIfFailed(CudaNativeGateway.GradientHostPipelineExchangeBFloat16(
            destinationAccelerator.Index,
            pipeline,
            local.NativePtr,
            remoteSource.NativePtr,
            reduced.NativePtr,
            length,
            squaredSum,
            localReadyEvent,
            remoteReadyEvent),
            "exchange BF16 gradient host pipeline");
        NativeCudaRuntime.RecordGradientCollectiveHostPipeline(
            physicalCopyCount,
            physicalBytes);
        transport?.Commit();
    }

    internal static void DestroyHostPipeline(
        NativeCudaDevice destinationAccelerator,
        nint pipeline)
    {
        if (pipeline != 0)
        {
            destinationAccelerator.Bind();
            int status = CudaNativeGateway.GradientHostPipelineDestroy(
                destinationAccelerator.Index,
                pipeline);
            _ = CudaNativeGateway.TakeCapturedFailure(status);
        }
    }

    internal static void Unpack(
        NativeCudaDevice accelerator,
        int device,
        NativeCudaBuffer<float> source,
        int sourceOffset,
        NativeCudaBuffer<float> destination,
        int length,
        nint communicationStream)
    {
        accelerator.Bind();
        ThrowIfFailed(CudaNativeGateway.GradientUnpackFloat32(
            device,
            source.NativePtr,
            sourceOffset,
            destination.NativePtr,
            length,
            communicationStream),
            "unpack reduced gradient bucket");
    }

    internal static void Synchronize(
        NativeCudaDevice accelerator,
        int device,
        nint communicationStream)
    {
        accelerator.Bind();
        ThrowIfFailed(
            CudaNativeGateway.GradientCommunicationSynchronize(
                device, communicationStream),
            "synchronize gradient communication stream");
    }

    internal static void DestroyEvent(
        NativeCudaDevice accelerator,
        int device,
        nint readyEvent)
    {
        if (readyEvent != 0)
        {
            accelerator.Bind();
            int status = CudaNativeGateway.GradientReadyEventDestroy(
                device,
                readyEvent);
            _ = CudaNativeGateway.TakeCapturedFailure(status);
        }
    }

    internal static void DestroyCommunicationStream(
        NativeCudaDevice accelerator,
        int device,
        nint communicationStream)
    {
        if (communicationStream != 0)
        {
            accelerator.Bind();
            int status = CudaNativeGateway.GradientCommunicationStreamDestroy(
                device,
                communicationStream);
            _ = CudaNativeGateway.TakeCapturedFailure(status);
        }
    }

    private static void ThrowIfFailed(int status, string operation)
        => NativeCudaRuntime.Check(status, operation);

}
