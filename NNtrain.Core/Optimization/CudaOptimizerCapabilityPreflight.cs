using NNtrain.Cuda.Execution;

namespace NNtrain;

/// <summary>
/// Stable optimizer family used by configuration-time CUDA preflight. It is
/// deliberately independent from optimizer construction so unsupported
/// hardware is rejected before moment arrays or parameter replicas exist.
/// </summary>
internal enum CudaOptimizerKind
{
    AdamW,
    NekoMuon,
    Lion,
    GainShareAdamW,
}

internal static class CudaOptimizerCapabilityPreflight
{
    internal static CudaKernelFeature ResolveRequiredCudaFeatures(
        CudaOptimizerKind optimizer,
        TensorPrecisionMode precisionMode)
    {
        CudaKernelFeature required = optimizer == CudaOptimizerKind.NekoMuon
            ? CudaKernelFeature.TensorCores
                | CudaKernelFeature.BlockReducedMuon
            : CudaKernelFeature.None;

        switch (precisionMode)
        {
            case TensorPrecisionMode.Float32:
                break;
            case TensorPrecisionMode.BFloat16:
            case TensorPrecisionMode.Mix16_32:
                required |= CudaKernelFeature.BFloat16;
                break;
            case TensorPrecisionMode.Bfp8:
            case TensorPrecisionMode.Mix8_32:
                required |= CudaKernelFeature.BFloat16
                    | CudaKernelFeature.Bfp8Quantization;
                break;
            default:
                throw new ArgumentOutOfRangeException(
                    nameof(precisionMode),
                    precisionMode,
                    null);
        }

        return required;
    }

    /// <summary>
    /// Performs optimizer capability validation without constructing the
    /// optimizer or touching model parameters. The injectable provider keeps
    /// ordering and failure behavior directly testable without CUDA hardware.
    /// </summary>
    internal static void EnsureBeforeAllocation(
        CudaOptimizerKind optimizer,
        TensorPrecisionMode precisionMode,
        IReadOnlyList<int> cudaDeviceIndices,
        Func<int, CudaKernelCapabilities>? capabilityProvider = null)
    {
        ArgumentNullException.ThrowIfNull(cudaDeviceIndices);
        if (cudaDeviceIndices.Count == 0)
        {
            throw new ArgumentException(
                "CUDA optimizer preflight requires at least one device.",
                nameof(cudaDeviceIndices));
        }
        if (cudaDeviceIndices.Any(index => index < 0)
            || cudaDeviceIndices.Distinct().Count()
                != cudaDeviceIndices.Count)
        {
            throw new ArgumentException(
                "CUDA device indices must be unique and non-negative.",
                nameof(cudaDeviceIndices));
        }

        CudaKernelFeature required = ResolveRequiredCudaFeatures(
            optimizer,
            precisionMode);
        capabilityProvider ??= NativeCudaRuntime.GetKernelCapabilities;
        foreach (int deviceIndex in cudaDeviceIndices)
        {
            CudaKernelCapabilities capabilities =
                capabilityProvider(deviceIndex)
                ?? throw new InvalidOperationException(
                    $"CUDA capability provider returned null for device " +
                    $"{deviceIndex}.");
            EnsureDeviceCapabilities(
                optimizer,
                precisionMode,
                deviceIndex,
                required,
                capabilities);
        }
    }

    internal static void EnsureDeviceCapabilities(
        CudaOptimizerKind optimizer,
        TensorPrecisionMode precisionMode,
        int deviceIndex,
        CudaKernelFeature required,
        CudaKernelCapabilities capabilities)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceIndex);
        ArgumentNullException.ThrowIfNull(capabilities);
        CudaKernelFeature missing = required & ~capabilities.Features;
        if (missing == CudaKernelFeature.None)
            return;

        throw new NotSupportedException(
            $"CUDA optimizer capability preflight failed for device " +
            $"{deviceIndex} (SM {capabilities.ComputeCapabilityMajor}." +
            $"{capabilities.ComputeCapabilityMinor}, optimizer " +
            $"{optimizer}, precision " +
            $"{TensorPrecisionModeNames.Format(precisionMode)}). Missing " +
            $"required CUDA kernel capabilities: " +
            $"{CudaDataParallelEngine.FormatCudaFeatures(missing)}. " +
            "CPU fallback is forbidden.");
    }
}
