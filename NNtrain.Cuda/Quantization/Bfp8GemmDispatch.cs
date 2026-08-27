using NNtrain.Cuda.Execution;

namespace NNtrain.Cuda.Quantization;

/// <summary>CUDA-only execution routes for a BFP8 matrix multiply.</summary>
public enum CudaBfp8GemmBackend
{
    /// <summary>Aligned signed Int8 operands consumed by cuBLASLt.</summary>
    CublasLtInt8TensorCore = 0,

    /// <summary>Device-side BFP8-to-BF16 conversion followed by BF16 GEMM.</summary>
    BFloat16Dequantize = 1,
}

/// <summary>
/// Scale topology presented to a CUDA BFP8 GEMM. This type deliberately lives
/// below Core so CUDA dispatch does not depend on the public Tensor facade.
/// </summary>
public enum CudaBfp8ScaleGranularity
{
    /// <summary>Exactly one scale is shared by the complete operand.</summary>
    TensorWide = 0,

    /// <summary>Contiguous blocks own independent scales.</summary>
    Block = 1,
}

/// <summary>
/// A validated CUDA GEMM boundary. It intentionally has no CPU route: shapes
/// that are not eligible for the Int8 Tensor Core path remain resident and
/// use the BF16 CUDA fallback.
/// </summary>
public readonly record struct CudaBfp8GemmPlan(
    CudaBfp8GemmBackend Backend,
    int M,
    int N,
    int K,
    int QuantizationBlockSize,
    CudaBfp8ScaleGranularity ScaleGranularity =
        CudaBfp8ScaleGranularity.Block)
{
    public bool RequiresBFloat16Dequantization
        => Backend == CudaBfp8GemmBackend.BFloat16Dequantize;
}

public static class CudaBfp8GemmDispatch
{
    public static CudaBfp8GemmPlan Preflight(
        CudaKernelCapabilities capabilities,
        int m,
        int n,
        int k,
        int quantizationBlockSize)
        // The legacy overload cannot distinguish a whole-tensor scale from a
        // fixed block that happens to have the same size. Conservatively stay
        // on CUDA BF16 rather than applying one incorrect Int8 alpha.
        => Preflight(
            capabilities,
            m,
            n,
            k,
            quantizationBlockSize,
            CudaBfp8ScaleGranularity.Block);

    public static CudaBfp8GemmPlan Preflight(
        CudaKernelCapabilities capabilities,
        int m,
        int n,
        int k,
        int quantizationBlockSize,
        CudaBfp8ScaleGranularity scaleGranularity)
    {
        ArgumentNullException.ThrowIfNull(capabilities);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(m);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(n);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(
            quantizationBlockSize);
        if (!Enum.IsDefined(scaleGranularity))
        {
            throw new ArgumentOutOfRangeException(
                nameof(scaleGranularity),
                scaleGranularity,
                "Unknown BFP8 scale granularity.");
        }

        if (!capabilities.Supports(CudaKernelFeature.Bfp8Quantization))
        {
            throw new NotSupportedException(
                "This CUDA device/runtime does not provide resident BFP8 " +
                "quantization. CPU fallback is forbidden.");
        }

        // Conservative cuBLASLt layout contract for Ampere and newer. A
        // future plan cache can relax this after querying concrete Lt layouts.
        bool tensorCoreEligible =
            scaleGranularity == CudaBfp8ScaleGranularity.TensorWide
            && capabilities.Supports(CudaKernelFeature.Int8TensorCores)
            && m % 8 == 0
            && n % 8 == 0
            && k % 32 == 0
            && quantizationBlockSize % 32 == 0;
        if (tensorCoreEligible)
        {
            return new CudaBfp8GemmPlan(
                CudaBfp8GemmBackend.CublasLtInt8TensorCore,
                m,
                n,
                k,
                quantizationBlockSize,
                scaleGranularity);
        }

        if (!capabilities.Supports(CudaKernelFeature.BFloat16))
        {
            throw new NotSupportedException(
                "This BFP8 shape needs the CUDA BF16 dequantization fallback, " +
                "but the device does not support BF16. CPU fallback is forbidden.");
        }

        return new CudaBfp8GemmPlan(
            CudaBfp8GemmBackend.BFloat16Dequantize,
            m,
            n,
            k,
            quantizationBlockSize,
            scaleGranularity);
    }
}
