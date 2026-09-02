using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// Versioned gateway for CUDA optimizer kernels. Core owns policy and state;
/// this type owns the native ABI and immutable failure snapshots.
/// </summary>
public static partial class CudaNativeGateway
{
    public static int OptimizerAdamW(
        int device, nint data, nint gradient, nint first, nint second,
        int length, float beta1, float beta2, float learningRate,
        float weightDecay, float updateScale, float scaledEpsilon,
        bool applyWeightDecay, bool bfloat16State)
    {
        EnsureCompatibleAbi();
        return Complete(
            bfloat16State
                ? OptimizerNativeMethods.AdamWBFloat16State(
                    data, gradient, first, second, length, beta1, beta2,
                    learningRate, weightDecay, updateScale, scaledEpsilon,
                    applyWeightDecay ? 1 : 0)
                : OptimizerNativeMethods.AdamW(
                    data, gradient, first, second, length, beta1, beta2,
                    learningRate, weightDecay, updateScale, scaledEpsilon,
                    applyWeightDecay ? 1 : 0),
            bfloat16State
                ? CudaNativeOperation.OptimizerBFloat16
                : CudaNativeOperation.Optimizer,
            device);
    }

    public static int OptimizerAdamWBfp8Moments(
        int device, nint gradient, nint first, nint second, int length,
        float beta1, float beta2, nint finiteStatus)
    {
        EnsureCompatibleAbi();
        return Complete(
            OptimizerNativeMethods.AdamWBfp8Moments(
                gradient, first, second, length, beta1, beta2, finiteStatus),
            CudaNativeOperation.OptimizerBfp8,
            device);
    }

    public static int OptimizerAdamWBfp8Apply(
        int device, nint data, nint first, nint second, nint secondScale,
        int secondScaleBlockSize, int length, float learningRate,
        float weightDecay, float updateScale,
        float scaledEpsilon, bool applyWeightDecay, nint finiteStatus)
    {
        EnsureCompatibleAbi();
        return Complete(
            OptimizerNativeMethods.AdamWBfp8Apply(
                data, first, second, secondScale, secondScaleBlockSize,
                length, learningRate,
                weightDecay, updateScale, scaledEpsilon,
                applyWeightDecay ? 1 : 0, finiteStatus),
            CudaNativeOperation.OptimizerBfp8,
            device);
    }

    public static int OptimizerAdamWMultiTensorBfp8(
        int device,
        nint tensors,
        int tensorCount,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        float updateScale,
        float scaledEpsilon,
        nint reduction,
        int maximumChunks,
        nint finiteStatus,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.FusedFirstOrderOptimizerMinor,
            "multi-tensor BFP8 AdamW");
        return Complete(
            OptimizerNativeMethods.AdamWMultiTensorBfp8(
                device,
                tensors,
                tensorCount,
                beta1,
                beta2,
                learningRate,
                weightDecay,
                updateScale,
                scaledEpsilon,
                reduction,
                maximumChunks,
                finiteStatus,
                stream),
            CudaNativeOperation.OptimizerBfp8,
            device);
    }

    public static int OptimizerAdamWAndPublish(
        int device, nint data, nint gradient, nint first, nint second,
        nint compute, bool physicalBFloat16, int length, float beta1,
        float beta2, float learningRate, float weightDecay, float updateScale,
        float scaledEpsilon, bool applyWeightDecay, bool bfloat16State)
    {
        EnsureCompatibleAbi();
        return Complete(
            bfloat16State
                ? OptimizerNativeMethods.AdamWBFloat16StatePublish(
                    data, gradient, first, second, compute,
                    physicalBFloat16 ? 1 : 0, length, beta1, beta2,
                    learningRate, weightDecay, updateScale, scaledEpsilon,
                    applyWeightDecay ? 1 : 0)
                : OptimizerNativeMethods.AdamWPublish(
                    data, gradient, first, second, compute,
                    physicalBFloat16 ? 1 : 0, length, beta1, beta2,
                    learningRate, weightDecay, updateScale, scaledEpsilon,
                    applyWeightDecay ? 1 : 0),
            bfloat16State
                ? CudaNativeOperation.OptimizerBFloat16
                : CudaNativeOperation.Optimizer,
            device);
    }

    public static int OptimizerPublishBFloat16(
        int device, nint master, nint compute, int length, bool physical)
    {
        EnsureCompatibleAbi();
        return Complete(
            OptimizerNativeMethods.PublishBFloat16(
                master, compute, length, physical ? 1 : 0),
            CudaNativeOperation.OptimizerBFloat16,
            device);
    }

    public static int OptimizerGatherStats(
        int device, nint sources, nint destination, int count)
    {
        EnsureCompatibleAbi();
        return Complete(
            OptimizerNativeMethods.GatherStats(sources, destination, count),
            CudaNativeOperation.Optimizer,
            device);
    }

    public static int OptimizerAdamWMultiTensor(
        int device, nint chunks, int chunkCount, float beta1, float beta2,
        float learningRate, float weightDecay, float updateScale,
        float scaledEpsilon)
    {
        EnsureCompatibleAbi();
        return Complete(
            OptimizerNativeMethods.AdamWMultiTensor(
                chunks, chunkCount, beta1, beta2, learningRate, weightDecay,
                updateScale, scaledEpsilon),
            CudaNativeOperation.Optimizer,
            device);
    }

    public static int OptimizerAccumulateFiniteStatus(
        int device, nint values, int length, nint finiteStatus)
    {
        EnsureCompatibleAbi();
        return Complete(
            OptimizerNativeMethods.AccumulateFiniteStatus(
                values, length, finiteStatus),
            CudaNativeOperation.Optimizer,
            device);
    }

    public static int OptimizerNekoMuonMoments(
        int device, nint gradient, nint fast, nint slow, nint fastHat,
        nint slowHat, int length, float betaFast, float betaSlow,
        float fastCorrection, float slowCorrection)
    {
        EnsureCompatibleAbi();
        return Complete(
            OptimizerNativeMethods.NekoMoments(
                gradient, fast, slow, fastHat, slowHat, length, betaFast,
                betaSlow, fastCorrection, slowCorrection),
            CudaNativeOperation.OptimizerNekoMuon,
            device);
    }

    public static int OptimizerNekoMuonInitializeLegacy(
        int device, nint source, nint destination, int length,
        int originalRows, int originalColumns, bool transpose,
        float inverseNorm)
    {
        EnsureCompatibleAbi();
        return Complete(
            OptimizerNativeMethods.NekoInitialize(
                source, destination, length, originalRows, originalColumns,
                transpose ? 1 : 0, inverseNorm),
            CudaNativeOperation.OptimizerNekoMuon,
            device);
    }

    public static int OptimizerNekoMuonInitialize(
        int device, nint source, nint destination, int length,
        int originalRows, int originalColumns, bool transpose,
        float inverseFastCorrection, float inverseNorm)
    {
        EnsureCompatibleAbi();
        return Complete(
            OptimizerNativeMethods.NekoInitializeCorrected(
                source, destination, length, originalRows, originalColumns,
                transpose ? 1 : 0, inverseFastCorrection, inverseNorm),
            CudaNativeOperation.OptimizerNekoMuon,
            device);
    }

    public static int OptimizerNekoMuonUpdateDeviceControl(
        int device, nint stats, nint confidence, nint finiteStatus,
        float epsilon, float rho)
    {
        EnsureCompatibleAbi();
        return Complete(
            OptimizerNativeMethods.NekoUpdateDeviceControl(
                stats, confidence, finiteStatus, epsilon, rho),
            CudaNativeOperation.OptimizerNekoMuon,
            device);
    }

    public static int OptimizerNekoMuonInitializeFromDeviceStats(
        int device, nint source, nint destination, int length,
        int originalRows, int originalColumns, bool transpose,
        float inverseFastCorrection, nint stats, float epsilon,
        nint finiteStatus)
    {
        EnsureCompatibleAbi();
        return Complete(
            OptimizerNativeMethods.NekoInitializeFromDeviceStats(
                source, destination, length, originalRows, originalColumns,
                transpose ? 1 : 0, inverseFastCorrection, stats, epsilon,
                finiteStatus),
            CudaNativeOperation.OptimizerNekoMuon,
            device);
    }

    public static int OptimizerNekoMuonInterpolate(
        int device, nint current, nint next, int length, float fraction)
    {
        EnsureCompatibleAbi();
        return Complete(
            OptimizerNativeMethods.NekoInterpolate(
                current, next, length, fraction),
            CudaNativeOperation.OptimizerNekoMuon,
            device);
    }

    public static int OptimizerNekoMuonAdaptiveAcceptBatched(
        int device, nint current, nint candidate, nint confidences,
        int matrixLength, int batch, int step, int maxSteps,
        int depthMode, float configuredDepth)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.DeviceAdaptiveNekoMuonMinor,
            "device-resident adaptive NekoMuon");
        return Complete(
            OptimizerNativeMethods.NekoAdaptiveAcceptBatched(
                current, candidate, confidences, matrixLength, batch, step,
                maxSteps, depthMode, configuredDepth),
            CudaNativeOperation.OptimizerNekoMuon,
            device);
    }

    public static int OptimizerNekoMuonConfidenceSummary(
        int device, nint confidences, int count, int maxSteps,
        int depthMode, float configuredDepth, bool runNewtonSchulz,
        bool forceFullDepth, nint summary)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.DeviceAdaptiveNekoMuonMinor,
            "device-resident NekoMuon confidence summary");
        return Complete(
            OptimizerNativeMethods.NekoConfidenceSummary(
                confidences, count, maxSteps, depthMode, configuredDepth,
                runNewtonSchulz ? 1 : 0, forceFullDepth ? 1 : 0, summary),
            CudaNativeOperation.OptimizerNekoMuon,
            device);
    }

    public static int OptimizerNekoMuonTransposeBack(
        int device, nint source, nint destination, int length,
        int originalRows, int originalColumns)
    {
        EnsureCompatibleAbi();
        return Complete(
            OptimizerNativeMethods.NekoTransposeBack(
                source, destination, length, originalRows, originalColumns),
            CudaNativeOperation.OptimizerNekoMuon,
            device);
    }

    public static int OptimizerNekoMuonApply(
        int device, nint data, nint update, int length, float learningRate,
        float finalScale, float weightDecay, bool applyWeightDecay)
    {
        EnsureCompatibleAbi();
        return Complete(
            OptimizerNativeMethods.NekoApply(
                data, update, length, learningRate, finalScale, weightDecay,
                applyWeightDecay ? 1 : 0),
            CudaNativeOperation.OptimizerNekoMuon,
            device);
    }

    public static int OptimizerNekoMuonCombine(
        int device, nint gram, nint gramSquared, int length, int rows,
        float a, float b, float c)
    {
        EnsureCompatibleAbi();
        return Complete(
            OptimizerNativeMethods.NekoCombine(
                gram, gramSquared, length, rows, a, b, c),
            CudaNativeOperation.OptimizerNekoMuon,
            device);
    }

    public static int OptimizerNekoMuonCombineBatched(
        int device, nint gram, nint gramSquared, int matrixLength, int batch,
        int rows, float a, float b, float c)
    {
        EnsureCompatibleAbi();
        return Complete(
            OptimizerNativeMethods.NekoCombineBatched(
                gram, gramSquared, matrixLength, batch, rows, a, b, c),
            CudaNativeOperation.OptimizerNekoMuon,
            device);
    }

    public static int OptimizerNekoMuonSymmetricGram(
        int device, nint source, nint destination, int rows, int columns,
        bool bfloat16Operands)
    {
        EnsureCompatibleAbi();
        return Complete(
            bfloat16Operands
                ? OptimizerNativeMethods.SymmetricGramBFloat16Operands(
                    source, destination, rows, columns)
                : OptimizerNativeMethods.SymmetricGram(
                    source, destination, rows, columns),
            bfloat16Operands
                ? CudaNativeOperation.OptimizerNekoMuonBFloat16
                : CudaNativeOperation.OptimizerNekoMuon,
            device);
    }

    public static int OptimizerNekoMuonNewtonSchulz(
        int device, nint source, nint gram, nint gramSquared,
        nint destination, int rows, int columns, float a, float b, float c,
        bool bfloat16Operands)
    {
        EnsureCompatibleAbi();
        return Complete(
            bfloat16Operands
                ? OptimizerNativeMethods.NewtonSchulzBFloat16Operands(
                    source, gram, gramSquared, destination, rows, columns,
                    a, b, c)
                : OptimizerNativeMethods.NewtonSchulz(
                    source, gram, gramSquared, destination, rows, columns,
                    a, b, c),
            bfloat16Operands
                ? CudaNativeOperation.OptimizerNekoMuonBFloat16
                : CudaNativeOperation.OptimizerNekoMuon,
            device);
    }

    private static class OptimizerNativeMethods
    {
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_adamw", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AdamW(nint data, nint gradient, nint first, nint second, int length, float beta1, float beta2, float learningRate, float weightDecay, float updateScale, float scaledEpsilon, int applyWeightDecay);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_adamw_bfp8_moments", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AdamWBfp8Moments(nint gradient, nint first, nint second, int length, float beta1, float beta2, nint finiteStatus);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_adamw_bfp8_apply", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AdamWBfp8Apply(nint data, nint first, nint second, nint secondScale, int secondScaleBlockSize, int length, float learningRate, float weightDecay, float updateScale, float scaledEpsilon, int applyWeightDecay, nint finiteStatus);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_adamw_multi_tensor_bfp8", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AdamWMultiTensorBfp8(int device, nint tensors, int tensorCount, float beta1, float beta2, float learningRate, float weightDecay, float updateScale, float scaledEpsilon, nint reduction, int maximumChunks, nint finiteStatus, nint stream);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_adamw_bf16_state", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AdamWBFloat16State(nint data, nint gradient, nint first, nint second, int length, float beta1, float beta2, float learningRate, float weightDecay, float updateScale, float scaledEpsilon, int applyWeightDecay);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_adamw_publish", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AdamWPublish(nint data, nint gradient, nint first, nint second, nint compute, int physicalBFloat16, int length, float beta1, float beta2, float learningRate, float weightDecay, float updateScale, float scaledEpsilon, int applyWeightDecay);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_adamw_bf16_state_publish", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AdamWBFloat16StatePublish(nint data, nint gradient, nint first, nint second, nint compute, int physicalBFloat16, int length, float beta1, float beta2, float learningRate, float weightDecay, float updateScale, float scaledEpsilon, int applyWeightDecay);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_publish_bf16", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int PublishBFloat16(nint master, nint compute, int length, int physical);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_gather_stats", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int GatherStats(nint sources, nint destination, int count);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_adamw_multi_tensor", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AdamWMultiTensor(nint chunks, int chunkCount, float beta1, float beta2, float learningRate, float weightDecay, float updateScale, float scaledEpsilon);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_accumulate_finite_status", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AccumulateFiniteStatus(nint values, int length, nint finiteStatus);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_neko_moments", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoMoments(nint gradient, nint fast, nint slow, nint fastHat, nint slowHat, int length, float betaFast, float betaSlow, float fastCorrection, float slowCorrection);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_neko_initialize", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoInitialize(nint source, nint destination, int length, int originalRows, int originalColumns, int transpose, float inverseNorm);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_neko_initialize_corrected", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoInitializeCorrected(nint source, nint destination, int length, int originalRows, int originalColumns, int transpose, float inverseFastCorrection, float inverseNorm);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_neko_update_device_control", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoUpdateDeviceControl(nint stats, nint confidence, nint finiteStatus, float epsilon, float rho);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_neko_initialize_device_stats", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoInitializeFromDeviceStats(nint source, nint destination, int length, int originalRows, int originalColumns, int transpose, float inverseFastCorrection, nint stats, float epsilon, nint finiteStatus);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_neko_interpolate", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoInterpolate(nint current, nint next, int length, float fraction);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_neko_adaptive_accept_batched", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoAdaptiveAcceptBatched(nint current, nint candidate, nint confidences, int matrixLength, int batch, int step, int maxSteps, int depthMode, float configuredDepth);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_neko_confidence_summary", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoConfidenceSummary(nint confidences, int count, int maxSteps, int depthMode, float configuredDepth, int runNewtonSchulz, int forceFullDepth, nint summary);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_neko_transpose_back", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoTransposeBack(nint source, nint destination, int length, int originalRows, int originalColumns);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_neko_apply", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoApply(nint data, nint update, int length, float learningRate, float finalScale, float weightDecay, int applyWeightDecay);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_neko_combine", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoCombine(nint gram, nint gramSquared, int length, int rows, float a, float b, float c);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_neko_combine_batched", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoCombineBatched(nint gram, nint gramSquared, int matrixLength, int batch, int rows, float a, float b, float c);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_symmetric_gram", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int SymmetricGram(nint source, nint destination, int rows, int columns);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_symmetric_gram_bf16_operands", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int SymmetricGramBFloat16Operands(nint source, nint destination, int rows, int columns);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_newton_schulz", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NewtonSchulz(nint source, nint gram, nint gramSquared, nint destination, int rows, int columns, float a, float b, float c);
        [DllImport(LibraryName, EntryPoint = "nntrain_optimizer_newton_schulz_bf16_operands", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NewtonSchulzBFloat16Operands(nint source, nint gram, nint gramSquared, nint destination, int rows, int columns, float a, float b, float c);
    }
}
