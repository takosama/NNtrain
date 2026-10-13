using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

public static partial class CudaNativeGateway
{

    public static int Bfp8QuantizeFloat32(
        int device,
        nint source,
        nint payload,
        nint scales,
        int length,
        int blockSize,
        nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Bfp8QuantizeFloat32(
                device,
                source,
                payload,
                scales,
                length,
                blockSize,
                stream),
            CudaNativeOperation.Bfp8Quantize,
            device);
    }

    public static int Bfp8DequantizeFloat32(
        int device,
        nint payload,
        nint scales,
        nint destination,
        int length,
        int blockSize,
        nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Bfp8DequantizeFloat32(
                device,
                payload,
                scales,
                destination,
                length,
                blockSize,
                stream),
            CudaNativeOperation.Bfp8DequantizeFloat32,
            device);
    }

    public static int Bfp8QuantizeFloat32Roundtrip(
        int device,
        nint source,
        nint payload,
        nint scales,
        int length,
        int blockSize,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.BlockBfp8OptimizerStateMinor,
            "block-BFP8 optimizer state roundtrip");
        return Complete(
            NativeMethods.Bfp8QuantizeFloat32Roundtrip(
                device,
                source,
                payload,
                scales,
                length,
                blockSize,
                stream),
            CudaNativeOperation.Bfp8Quantize,
            device);
    }

    public static int AdamWBlockBfp8State(
        int device,
        nint data,
        nint gradient,
        nint firstPayload,
        nint firstScales,
        nint secondPayload,
        nint secondScales,
        int length,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        float updateScale,
        float scaledEpsilon,
        bool applyWeightDecay,
        nint finiteStatus,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.BlockBfp8OptimizerStateMinor,
            "fused block-BFP8 AdamW state");
        return Complete(
            NativeMethods.AdamWBlockBfp8State(
                device,
                data,
                gradient,
                firstPayload,
                firstScales,
                secondPayload,
                secondScales,
                length,
                beta1,
                beta2,
                learningRate,
                weightDecay,
                updateScale,
                scaledEpsilon,
                applyWeightDecay ? 1 : 0,
                finiteStatus,
                stream),
            CudaNativeOperation.Bfp8Quantize,
            device);
    }

    public static int NekoMuonBlockBfp8Moments(
        int device,
        nint gradient,
        nint fastPayload,
        nint fastScales,
        nint slowPayload,
        nint slowScales,
        nint fastRoundtrip,
        nint slowRoundtrip,
        int length,
        float betaFast,
        float betaSlow,
        nint finiteStatus,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.BlockBfp8OptimizerStateMinor,
            "fused block-BFP8 NekoMuon moments");
        return Complete(
            NativeMethods.NekoMuonBlockBfp8Moments(
                device,
                gradient,
                fastPayload,
                fastScales,
                slowPayload,
                slowScales,
                fastRoundtrip,
                slowRoundtrip,
                length,
                betaFast,
                betaSlow,
                finiteStatus,
                stream),
            CudaNativeOperation.Bfp8Quantize,
            device);
    }

    public static int MuonBlockBfp8Moments(
        int device, nint gradient, nint fastPayload, nint fastScales,
        nint directionPayload, nint directionScales, nint fastRoundtrip,
        nint directionRoundtrip, int length, float beta,
        nint finiteStatus, nint stream)
    {
        EnsureMinimumAbiMinor(CudaAbiVersion.OrdinaryMuonNesterovMinor,
            "fused block-BFP8 reference Nesterov Muon moments");
        return Complete(NativeMethods.MuonBlockBfp8Moments(
                device, gradient, fastPayload, fastScales,
                directionPayload, directionScales, fastRoundtrip,
                directionRoundtrip, length, beta, finiteStatus, stream),
            CudaNativeOperation.Bfp8Quantize, device);
    }

    public static int Bfp8DequantizeBFloat16(
        int device,
        nint payload,
        nint scales,
        nint destination,
        int length,
        int blockSize,
        nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Bfp8DequantizeBFloat16(
                device,
                payload,
                scales,
                destination,
                length,
                blockSize,
                stream),
            CudaNativeOperation.Bfp8DequantizeBFloat16,
            device);
    }

    public static int Bfp8QuantizeBFloat16(
        int device,
        nint source,
        nint payload,
        nint scales,
        int length,
        int blockSize,
        nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Bfp8QuantizeBFloat16(
                device,
                source,
                payload,
                scales,
                length,
                blockSize,
                stream),
            CudaNativeOperation.Bfp8QuantizeBFloat16,
            device);
    }

    public static int Bfp8RequantizeInt32(
        int device,
        nint source,
        nint leftScales,
        nint rightScales,
        nint biasPayload,
        nint biasScales,
        nint payload,
        nint scales,
        int length,
        int outputWidth,
        int blockSize,
        int biasBlockSize,
        bool applyRelu,
        nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Bfp8RequantizeInt32(
                device,
                source,
                leftScales,
                rightScales,
                biasPayload,
                biasScales,
                payload,
                scales,
                length,
                outputWidth,
                blockSize,
                biasBlockSize,
                applyRelu ? 1 : 0,
                stream),
            CudaNativeOperation.Bfp8RequantizeInt32,
            device);
    }

    public static int Bfp8TransposeInt8(
        int device,
        nint source,
        nint destination,
        int rows,
        int columns,
        nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Bfp8TransposeInt8(
                device,
                source,
                destination,
                rows,
                columns,
                stream),
            CudaNativeOperation.Bfp8TransposeInt8,
            device);
    }

    private static partial class NativeMethods
    {

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_bfp8_quantize_f32",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Bfp8QuantizeFloat32(
            int device,
            nint source,
            nint payload,
            nint scales,
            int length,
            int blockSize,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_bfp8_quantize_f32_roundtrip",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Bfp8QuantizeFloat32Roundtrip(
            int device,
            nint source,
            nint payload,
            nint scales,
            int length,
            int blockSize,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_adamw_block_bfp8_state",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AdamWBlockBfp8State(
            int device,
            nint data,
            nint gradient,
            nint firstPayload,
            nint firstScales,
            nint secondPayload,
            nint secondScales,
            int length,
            float beta1,
            float beta2,
            float learningRate,
            float weightDecay,
            float updateScale,
            float scaledEpsilon,
            int applyWeightDecay,
            nint finiteStatus,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_nekomuon_block_bfp8_moments",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoMuonBlockBfp8Moments(
            int device,
            nint gradient,
            nint fastPayload,
            nint fastScales,
            nint slowPayload,
            nint slowScales,
            nint fastRoundtrip,
            nint slowRoundtrip,
            int length,
            float betaFast,
            float betaSlow,
            nint finiteStatus,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_muon_block_bfp8_moments",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int MuonBlockBfp8Moments(
            int device, nint gradient, nint fastPayload, nint fastScales,
            nint directionPayload, nint directionScales,
            nint fastRoundtrip, nint directionRoundtrip, int length,
            float beta, nint finiteStatus, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_bfp8_dequantize_f32",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Bfp8DequantizeFloat32(
            int device,
            nint payload,
            nint scales,
            nint destination,
            int length,
            int blockSize,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_bfp8_dequantize_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Bfp8DequantizeBFloat16(
            int device,
            nint payload,
            nint scales,
            nint destination,
            int length,
            int blockSize,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_bfp8_quantize_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Bfp8QuantizeBFloat16(
            int device,
            nint source,
            nint payload,
            nint scales,
            int length,
            int blockSize,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_bfp8_requantize_i32",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Bfp8RequantizeInt32(
            int device,
            nint source,
            nint leftScales,
            nint rightScales,
            nint biasPayload,
            nint biasScales,
            nint payload,
            nint scales,
            int length,
            int outputWidth,
            int blockSize,
            int biasBlockSize,
            int applyRelu,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_bfp8_transpose_i8",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Bfp8TransposeInt8(
            int device,
            nint source,
            nint destination,
            int rows,
            int columns,
            nint stream);
    }
}
