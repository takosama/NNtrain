using NNtrain;
using NNtrain.Cuda.Execution;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class CudaTrainingCapabilityPreflightTests
{
    public static TheoryData<
        int,
        TensorPrecisionMode,
        CudaKernelFeature> OptimizerRequirements => new()
        {
            {
                (int)CudaOptimizerKind.AdamW,
                TensorPrecisionMode.Float32,
                CudaKernelFeature.None
            },
            {
                (int)CudaOptimizerKind.AdamW,
                TensorPrecisionMode.Mix16_32,
                CudaKernelFeature.BFloat16
            },
            {
                (int)CudaOptimizerKind.AdamW,
                TensorPrecisionMode.Mix8_32,
                CudaKernelFeature.BFloat16
                    | CudaKernelFeature.Bfp8Quantization
            },
            {
                (int)CudaOptimizerKind.NekoMuon,
                TensorPrecisionMode.Float32,
                CudaKernelFeature.TensorCores
                    | CudaKernelFeature.BlockReducedMuon
            },
            {
                (int)CudaOptimizerKind.NekoMuon,
                TensorPrecisionMode.Bfp8,
                CudaKernelFeature.TensorCores
                    | CudaKernelFeature.BlockReducedMuon
                    | CudaKernelFeature.BFloat16
                    | CudaKernelFeature.Bfp8Quantization
            },
            {
                (int)CudaOptimizerKind.Lion,
                TensorPrecisionMode.Float32,
                CudaKernelFeature.FusedFirstOrderOptimizers
            },
            {
                (int)CudaOptimizerKind.GainShareAdamW,
                TensorPrecisionMode.Bfp8,
                CudaKernelFeature.FusedFirstOrderOptimizers
                    | CudaKernelFeature.BFloat16
                    | CudaKernelFeature.Bfp8Quantization
            },
        };

    public static TheoryData<
        Type,
        TensorPrecisionMode,
        int,
        bool,
        CudaKernelFeature> Requirements => new()
        {
            {
                typeof(GptRinWikiJp),
                TensorPrecisionMode.Float32,
                1,
                false,
                CudaKernelFeature.FlashAttention
                    | CudaKernelFeature.FusedLayerNorm
            },
            {
                typeof(GptRinWikiJp),
                TensorPrecisionMode.BFloat16,
                1,
                false,
                CudaKernelFeature.TensorCores
                    | CudaKernelFeature.BFloat16
                    | CudaKernelFeature.FlashAttention
                    | CudaKernelFeature.FusedLayerNorm
            },
            {
                typeof(GptRinWikiJp),
                TensorPrecisionMode.Mix16_32,
                2,
                true,
                CudaKernelFeature.TensorCores
                    | CudaKernelFeature.BFloat16
                    | CudaKernelFeature.FlashAttention
                    | CudaKernelFeature.FusedLayerNorm
                    | CudaKernelFeature.AsynchronousGradientReduction
                    | CudaKernelFeature.CudaGraphs
            },
            {
                typeof(GptRinWikiJp),
                TensorPrecisionMode.Bfp8,
                1,
                false,
                CudaKernelFeature.TensorCores
                    | CudaKernelFeature.BFloat16
                    | CudaKernelFeature.Bfp8Quantization
                    | CudaKernelFeature.FlashAttention
                    | CudaKernelFeature.FusedLayerNorm
            },
            {
                typeof(ForgetMemoryV3Gpt),
                TensorPrecisionMode.Mix8_32,
                2,
                true,
                CudaKernelFeature.TensorCores
                    | CudaKernelFeature.BFloat16
                    | CudaKernelFeature.Bfp8Quantization
                    | CudaKernelFeature.ForgetMemory
                    | CudaKernelFeature.AsynchronousGradientReduction
                    | CudaKernelFeature.CudaGraphs
            },
            {
                typeof(ForgetScanGpt),
                TensorPrecisionMode.Float32,
                1,
                false,
                CudaKernelFeature.ForgetMemory
            },
            {
                typeof(HyenaGpt),
                TensorPrecisionMode.Float32,
                2,
                false,
                CudaKernelFeature.AsynchronousGradientReduction
            },
        };

    [Theory]
    [MemberData(nameof(Requirements))]
    public void RequiredFeaturesArePurelyResolvedFromModelPrecisionAndPlan(
        Type modelType,
        TensorPrecisionMode precisionMode,
        int deviceCount,
        bool compiledGraphRequested,
        CudaKernelFeature expected)
    {
        CudaKernelFeature actual =
            CudaDataParallelEngine.ResolveRequiredCudaFeatures(
                modelType,
                precisionMode,
                deviceCount,
                compiledGraphRequested);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void FeatureFormattingListsEachMissingCapability()
    {
        string formatted = CudaDataParallelEngine.FormatCudaFeatures(
            CudaKernelFeature.TensorCores
                | CudaKernelFeature.BFloat16
                | CudaKernelFeature.CudaGraphs);

        Assert.Equal("TensorCores, BFloat16, CudaGraphs", formatted);
    }

    [Fact]
    public void MissingFeaturesFailBeforeTrainingWithDeviceSpecificMessage()
    {
        CudaKernelFeature required =
            CudaDataParallelEngine.ResolveRequiredCudaFeatures(
                typeof(GptRinWikiJp),
                TensorPrecisionMode.Mix8_32,
                2,
                true);
        var capabilities = new CudaKernelCapabilities(
            8,
            6,
            CudaKernelFeature.TensorCores
                | CudaKernelFeature.BFloat16
                | CudaKernelFeature.FusedLayerNorm);

        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => CudaDataParallelEngine.EnsureCudaCapabilities(
                typeof(GptRinWikiJp),
                TensorPrecisionMode.Mix8_32,
                1,
                required,
                capabilities));

        Assert.Contains("device 1 (SM 8.6", error.Message);
        Assert.Contains("GptRinWikiJp", error.Message);
        Assert.Contains("mix8_32", error.Message);
        Assert.Contains("FlashAttention", error.Message);
        Assert.Contains("AsynchronousGradientReduction", error.Message);
        Assert.Contains("CudaGraphs", error.Message);
        Assert.Contains("Bfp8Quantization", error.Message);
        Assert.Contains("CPU fallback is forbidden", error.Message);
    }

    [Fact]
    public void ResolverRejectsInvalidModelAndDeviceCount()
    {
        Assert.Throws<ArgumentException>(() =>
            CudaDataParallelEngine.ResolveRequiredCudaFeatures(
                typeof(string),
                TensorPrecisionMode.Float32,
                1,
                false));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            CudaDataParallelEngine.ResolveRequiredCudaFeatures(
                typeof(GptRinWikiJp),
                TensorPrecisionMode.Float32,
                0,
                false));
    }

    [Theory]
    [MemberData(nameof(OptimizerRequirements))]
    public void OptimizerRequirementsAreResolvedBeforeConstruction(
        int optimizer,
        TensorPrecisionMode precisionMode,
        CudaKernelFeature expected)
    {
        Assert.Equal(
                expected,
                CudaOptimizerCapabilityPreflight.ResolveRequiredCudaFeatures(
                (CudaOptimizerKind)optimizer,
                precisionMode));
    }

    [Fact]
    public void MissingNekoMuonCapabilityFailsBeforeAllocatorWork()
    {
        bool allocatorWorkStarted = false;
        var capabilities = new CudaKernelCapabilities(
            8,
            6,
            CudaKernelFeature.TensorCores
                | CudaKernelFeature.BFloat16);

        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () =>
            {
                CudaOptimizerCapabilityPreflight.EnsureBeforeAllocation(
                    CudaOptimizerKind.NekoMuon,
                    TensorPrecisionMode.Mix16_32,
                    [0, 1],
                    _ => capabilities);
                allocatorWorkStarted = true;
            });

        Assert.False(allocatorWorkStarted);
        Assert.Contains("optimizer NekoMuon", error.Message);
        Assert.Contains("BlockReducedMuon", error.Message);
        Assert.Contains("CPU fallback is forbidden", error.Message);
    }

    [Fact]
    public void MissingFirstOrderKernelFailsBeforeLionConstruction()
    {
        var capabilities = new CudaKernelCapabilities(
            8,
            6,
            CudaKernelFeature.BFloat16
                | CudaKernelFeature.Bfp8Quantization);

        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => CudaOptimizerCapabilityPreflight.EnsureBeforeAllocation(
                CudaOptimizerKind.Lion,
                TensorPrecisionMode.Mix8_32,
                [0],
                _ => capabilities));

        Assert.Contains("optimizer Lion", error.Message);
        Assert.Contains("FusedFirstOrderOptimizers", error.Message);
        Assert.Contains("CPU fallback is forbidden", error.Message);
    }
}
