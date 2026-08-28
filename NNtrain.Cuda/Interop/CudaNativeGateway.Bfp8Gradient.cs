using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// Versioned gateway for pure tensor-wide BFP8 gradient publication,
/// reduction, norm accumulation, and scale-only clipping.
/// </summary>
public static partial class CudaNativeGateway
{
    public static int Bfp8GradientQuantize(
        int device,
        nint source,
        nint payload,
        nint scale,
        int length,
        nint finiteStatus,
        nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            Bfp8GradientNativeMethods.Quantize(
                device, source, payload, scale, length, finiteStatus, stream),
            CudaNativeOperation.Bfp8GradientQuantize,
            device);
    }

    public static int Bfp8GradientQuantizeAccumulate(
        int device,
        nint source,
        nint payload,
        nint scale,
        int length,
        nint finiteStatus,
        nint squaredSum,
        nint stream)
    {
        EnsureScaleAwareGradientAbi();
        return Complete(
            Bfp8GradientNativeMethods.QuantizeAccumulate(
                device,
                source,
                payload,
                scale,
                length,
                finiteStatus,
                squaredSum,
                stream),
            CudaNativeOperation.Bfp8GradientQuantizeAccumulate,
            device);
    }

    public static int Bfp8GradientSquaredSum(
        int device,
        nint payload,
        nint scale,
        int length,
        nint squaredSum,
        nint finiteStatus,
        nint stream)
    {
        EnsureScaleAwareGradientAbi();
        return Complete(
            Bfp8GradientNativeMethods.SquaredSum(
                device,
                payload,
                scale,
                length,
                squaredSum,
                finiteStatus,
                stream),
            CudaNativeOperation.Bfp8GradientSquaredSum,
            device);
    }

    public static int Bfp8GradientScale(
        int device,
        nint scale,
        float multiplier,
        nint stream)
    {
        EnsureScaleAwareGradientAbi();
        return Complete(
            Bfp8GradientNativeMethods.Scale(
                device, scale, multiplier, stream),
            CudaNativeOperation.Bfp8GradientScale,
            device);
    }

    public static int Bfp8GradientReduce(
        int primaryDevice,
        int secondaryDevice,
        nint localPayload,
        nint localScale,
        nint remotePayload,
        nint remoteScale,
        nint remotePayloadStaging,
        nint remoteScaleStaging,
        nint reduced,
        nint outputPayload,
        nint outputScale,
        int length,
        float reductionScale,
        nint finiteStatus,
        nint remoteFiniteStatus,
        nint remoteStatusStaging,
        nint squaredSum,
        nint communicationStream,
        nint localReady,
        nint remoteReady,
        nint reducedReady)
    {
        EnsureCompatibleAbi();
        return Complete(
            Bfp8GradientNativeMethods.Reduce(
                primaryDevice,
                secondaryDevice,
                localPayload,
                localScale,
                remotePayload,
                remoteScale,
                remotePayloadStaging,
                remoteScaleStaging,
                reduced,
                outputPayload,
                outputScale,
                length,
                reductionScale,
                finiteStatus,
                remoteFiniteStatus,
                remoteStatusStaging,
                squaredSum,
                communicationStream,
                localReady,
                remoteReady,
                reducedReady),
            CudaNativeOperation.Bfp8GradientReduce,
            primaryDevice);
    }

    public static int Bfp8GradientBroadcast(
        int destinationDevice,
        int sourceDevice,
        nint sourcePayload,
        nint sourceScale,
        nint destinationPayload,
        nint destinationScale,
        nint destinationFloat,
        int length,
        nint destinationFiniteStatus,
        nint destinationStream,
        nint sourceReady)
    {
        EnsureCompatibleAbi();
        return Complete(
            Bfp8GradientNativeMethods.Broadcast(
                destinationDevice,
                sourceDevice,
                sourcePayload,
                sourceScale,
                destinationPayload,
                destinationScale,
                destinationFloat,
                length,
                destinationFiniteStatus,
                destinationStream,
                sourceReady),
            CudaNativeOperation.Bfp8GradientBroadcast,
            destinationDevice);
    }

    private static void EnsureScaleAwareGradientAbi()
        => EnsureMinimumAbiMinor(
            CudaAbiVersion.Bfp8ScaleAwareGradientMinor,
            "scale-aware resident BFP8 gradient clipping");

    private static class Bfp8GradientNativeMethods
    {
        [DllImport(LibraryName,
            EntryPoint = "nntrain_bfp8_gradient_quantize",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Quantize(
            int device,
            nint source,
            nint payload,
            nint scale,
            int length,
            nint finiteStatus,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_bfp8_gradient_quantize_accumulate",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int QuantizeAccumulate(
            int device,
            nint source,
            nint payload,
            nint scale,
            int length,
            nint finiteStatus,
            nint squaredSum,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_bfp8_gradient_squared_sum",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int SquaredSum(
            int device,
            nint payload,
            nint scale,
            int length,
            nint squaredSum,
            nint finiteStatus,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_bfp8_gradient_scale",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Scale(
            int device,
            nint scale,
            float multiplier,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_bfp8_gradient_reduce",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Reduce(
            int primaryDevice,
            int secondaryDevice,
            nint localPayload,
            nint localScale,
            nint remotePayload,
            nint remoteScale,
            nint remotePayloadStaging,
            nint remoteScaleStaging,
            nint reduced,
            nint outputPayload,
            nint outputScale,
            int length,
            float reductionScale,
            nint finiteStatus,
            nint remoteFiniteStatus,
            nint remoteStatusStaging,
            nint squaredSum,
            nint communicationStream,
            nint localReady,
            nint remoteReady,
            nint reducedReady);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_bfp8_gradient_broadcast",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Broadcast(
            int destinationDevice,
            int sourceDevice,
            nint sourcePayload,
            nint sourceScale,
            nint destinationPayload,
            nint destinationScale,
            nint destinationFloat,
            int length,
            nint destinationFiniteStatus,
            nint destinationStream,
            nint sourceReady);
    }
}
