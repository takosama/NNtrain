using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// Versioned gateway for asynchronous gradient-bucket communication.
/// </summary>
public static partial class CudaNativeGateway
{
    public static int GradientCommunicationStreamCreate(
        int device, out nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            GradientCollectiveNativeMethods.CreateCommunicationStream(
                device, out stream),
            CudaNativeOperation.GradientCollective,
            device);
    }

    public static int GradientReadyEventCreate(
        int device, out nint readyEvent)
    {
        EnsureCompatibleAbi();
        return Complete(
            GradientCollectiveNativeMethods.CreateReadyEvent(
                device, out readyEvent),
            CudaNativeOperation.GradientCollective,
            device);
    }

    public static int GradientPackBFloat16(
        int device, nint source, nint destination, int destinationOffset,
        int length, nint computeStream)
    {
        EnsureCompatibleAbi();
        return Complete(
            GradientCollectiveNativeMethods.PackBFloat16(
                device, source, destination, destinationOffset, length,
                computeStream),
            CudaNativeOperation.GradientCollectiveBFloat16,
            device);
    }

    public static int GradientPackBfp8Block(
        int device, nint source, nint destination, nint scales,
        int length, nint computeStream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.BlockBfp8GradientTransportMinor,
            "block-scaled BFP8 gradient transport");
        return Complete(
            GradientCollectiveNativeMethods.PackBfp8Block(
                device, source, destination, scales, length, computeStream),
            CudaNativeOperation.GradientCollective,
            device);
    }

    public static int GradientRecordReady(
        int device, nint readyEvent, nint computeStream)
    {
        EnsureCompatibleAbi();
        return Complete(
            GradientCollectiveNativeMethods.RecordReady(
                device, readyEvent, computeStream),
            CudaNativeOperation.GradientCollective,
            device);
    }

    /// <summary>
    /// Records an externally observable event node while the compute stream is
    /// being captured.  Unlike a default captured event, this event may be
    /// synchronized by the non-peer host pipeline after graph launch, allowing
    /// bucket exchange to begin at its real backward boundary.
    /// </summary>
    public static int GradientRecordReadyExternal(
        int device, nint readyEvent, nint computeStream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.ExternalGradientReadyEventMinor,
            "external CUDA Graph gradient ready events");
        return Complete(
            GradientCollectiveNativeMethods.RecordReadyExternal(
                device, readyEvent, computeStream),
            CudaNativeOperation.GradientCollective,
            device);
    }

    public static int GradientExchangeBFloat16(
        int destinationDevice, int sourceDevice, nint local,
        nint remoteSource, nint remoteStaging, nint reduced, int length,
        nint squaredSum, nint communicationStream, nint localReadyEvent,
        nint remoteReadyEvent)
    {
        EnsureCompatibleAbi();
        return Complete(
            GradientCollectiveNativeMethods.ExchangeBFloat16(
                destinationDevice, sourceDevice, local, remoteSource,
                remoteStaging, reduced, length, squaredSum,
                communicationStream, localReadyEvent, remoteReadyEvent),
            CudaNativeOperation.GradientCollectiveBFloat16,
            destinationDevice);
    }

    public static int GradientExchangeBfp8Block(
        int destinationDevice, int sourceDevice, nint local,
        nint localScales, nint remoteSource, nint remoteSourceScales,
        nint remoteStaging, nint remoteStagingScales, nint reduced,
        int length, nint squaredSum, nint communicationStream,
        nint localReadyEvent, nint remoteReadyEvent)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.BlockBfp8GradientTransportMinor,
            "block-scaled BFP8 gradient transport");
        return Complete(
            GradientCollectiveNativeMethods.ExchangeBfp8Block(
                destinationDevice, sourceDevice, local, localScales,
                remoteSource, remoteSourceScales, remoteStaging,
                remoteStagingScales, reduced, length, squaredSum,
                communicationStream, localReadyEvent, remoteReadyEvent),
            CudaNativeOperation.GradientCollective,
            destinationDevice);
    }

    public static int GradientHostPipelineCreate(
        int sourceDevice, int destinationDevice, int chunkElements,
        out nint pipeline)
    {
        EnsureCompatibleAbi();
        return CompleteSelectingDevice(
            GradientCollectiveNativeMethods.CreateHostPipeline(
                sourceDevice, destinationDevice, chunkElements, out pipeline),
            CudaNativeOperation.GradientCollective,
            destinationDevice);
    }

    public static int GradientHostPipelineCreateBfp8Block(
        int sourceDevice, int destinationDevice, int chunkElements,
        out nint pipeline)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.BlockBfp8GradientTransportMinor,
            "block-scaled BFP8 gradient host pipeline");
        return CompleteSelectingDevice(
            GradientCollectiveNativeMethods.CreateHostPipelineBfp8Block(
                sourceDevice, destinationDevice, chunkElements, out pipeline),
            CudaNativeOperation.GradientCollective,
            destinationDevice);
    }

    public static int GradientHostPipelineExchangeBFloat16(
        int destinationDevice, nint pipeline, nint local, nint remoteSource,
        nint reduced, int length, nint squaredSum, nint localReadyEvent,
        nint remoteReadyEvent)
    {
        EnsureCompatibleAbi();
        return CompleteSelectingDevice(
            GradientCollectiveNativeMethods.HostPipelineExchangeBFloat16(
                pipeline, local, remoteSource, reduced, length, squaredSum,
                localReadyEvent, remoteReadyEvent),
            CudaNativeOperation.GradientCollectiveBFloat16,
            destinationDevice);
    }

    public static int GradientHostPipelineExchangeBfp8Block(
        int destinationDevice, nint pipeline, nint local, nint localScales,
        nint remoteSource, nint remoteSourceScales, nint reduced, int length,
        nint squaredSum, nint localReadyEvent, nint remoteReadyEvent)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.BlockBfp8GradientTransportMinor,
            "block-scaled BFP8 gradient host pipeline");
        return CompleteSelectingDevice(
            GradientCollectiveNativeMethods.HostPipelineExchangeBfp8Block(
                pipeline, local, localScales, remoteSource,
                remoteSourceScales, reduced, length, squaredSum,
                localReadyEvent, remoteReadyEvent),
            CudaNativeOperation.GradientCollective,
            destinationDevice);
    }

    public static int GradientHostPipelineDestroy(
        int destinationDevice, nint pipeline)
    {
        EnsureCompatibleAbi();
        return CompleteSelectingDevice(
            GradientCollectiveNativeMethods.DestroyHostPipeline(pipeline),
            CudaNativeOperation.GradientCollective,
            destinationDevice);
    }

    public static int GradientUnpackFloat32(
        int device, nint source, int sourceOffset, nint destination,
        int length, nint communicationStream)
    {
        EnsureCompatibleAbi();
        return Complete(
            GradientCollectiveNativeMethods.UnpackFloat32(
                device, source, sourceOffset, destination, length,
                communicationStream),
            CudaNativeOperation.GradientCollective,
            device);
    }

    public static int GradientCommunicationSynchronize(
        int device, nint communicationStream)
    {
        EnsureCompatibleAbi();
        return Complete(
            GradientCollectiveNativeMethods.Synchronize(
                device, communicationStream),
            CudaNativeOperation.GradientCollective,
            device);
    }

    public static int GradientReadyEventDestroy(
        int device, nint readyEvent)
    {
        EnsureCompatibleAbi();
        return Complete(
            GradientCollectiveNativeMethods.DestroyEvent(device, readyEvent),
            CudaNativeOperation.GradientCollective,
            device);
    }

    public static int GradientCommunicationStreamDestroy(
        int device, nint communicationStream)
    {
        EnsureCompatibleAbi();
        return Complete(
            GradientCollectiveNativeMethods.DestroyCommunicationStream(
                device, communicationStream),
            CudaNativeOperation.GradientCollective,
            device);
    }

    private static class GradientCollectiveNativeMethods
    {
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_comm_create", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CreateCommunicationStream(int device, out nint stream);
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_event_create", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CreateReadyEvent(int device, out nint readyEvent);
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_pack_bf16", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int PackBFloat16(int device, nint source, nint destination, int destinationOffset, int length, nint computeStream);
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_pack_bfp8_block", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int PackBfp8Block(int device, nint source, nint destination, nint scales, int length, nint computeStream);
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_record_ready", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int RecordReady(int device, nint readyEvent, nint computeStream);
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_record_ready_external", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int RecordReadyExternal(int device, nint readyEvent, nint computeStream);
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_exchange_bf16", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ExchangeBFloat16(int destinationDevice, int sourceDevice, nint local, nint remoteSource, nint remoteStaging, nint reduced, int length, nint squaredSum, nint communicationStream, nint localReadyEvent, nint remoteReadyEvent);
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_exchange_bfp8_block", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ExchangeBfp8Block(int destinationDevice, int sourceDevice, nint local, nint localScales, nint remoteSource, nint remoteSourceScales, nint remoteStaging, nint remoteStagingScales, nint reduced, int length, nint squaredSum, nint communicationStream, nint localReadyEvent, nint remoteReadyEvent);
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_host_pipeline_create", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CreateHostPipeline(int sourceDevice, int destinationDevice, int chunkElements, out nint pipeline);
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_host_pipeline_create_bfp8_block", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CreateHostPipelineBfp8Block(int sourceDevice, int destinationDevice, int chunkElements, out nint pipeline);
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_host_pipeline_exchange_bf16", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int HostPipelineExchangeBFloat16(nint pipeline, nint local, nint remoteSource, nint reduced, int length, nint squaredSum, nint localReadyEvent, nint remoteReadyEvent);
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_host_pipeline_exchange_bfp8_block", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int HostPipelineExchangeBfp8Block(nint pipeline, nint local, nint localScales, nint remoteSource, nint remoteSourceScales, nint reduced, int length, nint squaredSum, nint localReadyEvent, nint remoteReadyEvent);
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_host_pipeline_destroy", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DestroyHostPipeline(nint pipeline);
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_unpack_float", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int UnpackFloat32(int device, nint source, int sourceOffset, nint destination, int length, nint communicationStream);
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_comm_synchronize", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Synchronize(int device, nint communicationStream);
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_event_destroy", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DestroyEvent(int device, nint readyEvent);
        [DllImport(LibraryName, EntryPoint = "nntrain_gradient_comm_destroy", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DestroyCommunicationStream(int device, nint communicationStream);
    }
}
