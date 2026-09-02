using NNtrain.Runtime.Execution;

namespace NNtrain.Cuda.Execution;

[Flags]
public enum CudaKernelFeature
{
    None = 0,
    TensorCores = 1 << 0,
    BFloat16 = 1 << 1,
    FlashAttention = 1 << 2,
    FusedLayerNorm = 1 << 3,
    ForgetMemory = 1 << 4,
    BlockReducedMuon = 1 << 5,
    AsynchronousGradientReduction = 1 << 6,
    CudaGraphs = 1 << 7,
    Bfp8Quantization = 1 << 8,
    Int8TensorCores = 1 << 9,
    /// <summary>
    /// Resident fused CUDA kernels for elementwise first-order optimizers
    /// such as Lion and the reduction/apply stages of GainShareAdamW.
    /// </summary>
    FusedFirstOrderOptimizers = 1 << 10,
}

/// <summary>Immutable capability snapshot for one CUDA lane.</summary>
public sealed record CudaKernelCapabilities(
    int ComputeCapabilityMajor,
    int ComputeCapabilityMinor,
    CudaKernelFeature Features) : IKernelCapabilitySet
{
    public bool Supports(string feature)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(feature);
        return Enum.TryParse(feature, ignoreCase: true, out CudaKernelFeature parsed)
            && parsed != CudaKernelFeature.None
            && Features.HasFlag(parsed);
    }

    public bool Supports(CudaKernelFeature feature)
        => feature != CudaKernelFeature.None && Features.HasFlag(feature);
}
