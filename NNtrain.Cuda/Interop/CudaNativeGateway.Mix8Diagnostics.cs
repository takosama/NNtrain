using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// Constant-size device aggregate produced by the optional mix8 diagnostic
/// kernels. Sums are squared ratios measured in units of the applicable BFP8
/// block quantum; counts describe the publication pass.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CudaMix8DiagnosticAccumulator
{
    public double UpdateStepRatioSquaredSum;
    public double ResidualStepRatioSquaredSum;
    public ulong ChangedCodeCount;
    public ulong ElementCount;
}

/// <summary>
/// ABI v1.30 AdamW chunk descriptor. The first 64 bytes mirror the stable
/// AdamW descriptor; the suffix supplies the parameter's current BFP8 scale.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct CudaAdamWMix8DiagnosticChunkDescriptor(
    nint data,
    nint gradient,
    nint firstMoment,
    nint secondMoment,
    nint compute,
    int offset,
    int length,
    int applyWeightDecay,
    int physicalBFloat16,
    int momentFormatFlags,
    int pureBFloat16,
    nint currentScales,
    int blockSize)
{
    public readonly nint Data = data;
    public readonly nint Gradient = gradient;
    public readonly nint FirstMoment = firstMoment;
    public readonly nint SecondMoment = secondMoment;
    public readonly nint Compute = compute;
    public readonly int Offset = offset;
    public readonly int Length = length;
    public readonly int ApplyWeightDecay = applyWeightDecay;
    public readonly int PhysicalBFloat16 = physicalBFloat16;
    public readonly int MomentFormatFlags = momentFormatFlags;
    public readonly int PureBFloat16 = pureBFloat16;
    public readonly nint CurrentScales = currentScales;
    public readonly int BlockSize = blockSize;
    public readonly int Reserved = 0;
}

public static partial class CudaNativeGateway
{
    public static int Mix8DiagnosticsReset(
        int device,
        nint metrics,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.Mix8DiagnosticsMinor,
            "mix8 optimizer diagnostics");
        return Complete(
            Mix8DiagnosticNativeMethods.Reset(device, metrics, stream),
            CudaNativeOperation.MemsetAsync,
            device);
    }

    public static int Bfp8QuantizeFloat32Diagnostic(
        int device,
        nint source,
        nint payload,
        nint scales,
        int length,
        int blockSize,
        nint metrics,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.Mix8DiagnosticsMinor,
            "mix8 BFP8 publication diagnostics");
        return Complete(
            Mix8DiagnosticNativeMethods.QuantizeFloat32(
                device,
                source,
                payload,
                scales,
                length,
                blockSize,
                metrics,
                stream),
            CudaNativeOperation.Bfp8Quantize,
            device);
    }

    public static int OptimizerNekoMuonApplyMix8Diagnostic(
        int device,
        nint data,
        nint update,
        nint currentScales,
        int blockSize,
        int length,
        float learningRate,
        float finalScale,
        float weightDecay,
        bool applyWeightDecay,
        nint metrics)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.Mix8DiagnosticsMinor,
            "mix8 NekoMuon update diagnostics");
        return Complete(
            Mix8DiagnosticNativeMethods.NekoApply(
                data,
                update,
                currentScales,
                blockSize,
                length,
                learningRate,
                finalScale,
                weightDecay,
                applyWeightDecay ? 1 : 0,
                metrics),
            CudaNativeOperation.OptimizerNekoMuon,
            device);
    }

    public static int OptimizerAdamWMultiTensorMix8Diagnostic(
        int device,
        nint chunks,
        int chunkCount,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        float updateScale,
        float scaledEpsilon,
        nint metrics)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.Mix8DiagnosticsMinor,
            "mix8 AdamW update diagnostics");
        return Complete(
            Mix8DiagnosticNativeMethods.AdamWMultiTensor(
                chunks,
                chunkCount,
                beta1,
                beta2,
                learningRate,
                weightDecay,
                updateScale,
                scaledEpsilon,
                metrics),
            CudaNativeOperation.OptimizerBfp8,
            device);
    }

    private static class Mix8DiagnosticNativeMethods
    {
        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_mix8_diagnostics_reset",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Reset(
            int device,
            nint metrics,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_bfp8_quantize_f32_diagnostic",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int QuantizeFloat32(
            int device,
            nint source,
            nint payload,
            nint scales,
            int length,
            int blockSize,
            nint metrics,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_optimizer_neko_apply_mix8_diagnostic",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoApply(
            nint data,
            nint update,
            nint currentScales,
            int blockSize,
            int length,
            float learningRate,
            float finalScale,
            float weightDecay,
            int applyWeightDecay,
            nint metrics);

        [DllImport(
            LibraryName,
            EntryPoint =
                "nntrain_optimizer_adamw_multi_tensor_mix8_diagnostic_v2",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AdamWMultiTensor(
            nint chunks,
            int chunkCount,
            float beta1,
            float beta2,
            float learningRate,
            float weightDecay,
            float updateScale,
            float scaledEpsilon,
            nint metrics);
    }
}
