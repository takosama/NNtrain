using System.Runtime.InteropServices;
using NNtrain.Cuda.Interop;

namespace NNtrain;

internal static class CudaOptimizerNative
{
    internal static void AdamW(int device, nint data, nint gradient,
        nint first, nint second, int length, float beta1, float beta2,
        float learningRate, float weightDecay, float updateScale,
        float scaledEpsilon, bool applyWeightDecay, bool bfloat16State)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerAdamW(
            device, data, gradient, first, second, length, beta1, beta2,
            learningRate, weightDecay, updateScale, scaledEpsilon,
            applyWeightDecay, bfloat16State), "AdamW update");
    }

    internal static void AdamWBfp8Moments(
        int device,
        nint gradient,
        nint first,
        nint second,
        int length,
        float beta1,
        float beta2,
        nint finiteStatus)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerAdamWBfp8Moments(
            device,
            gradient,
            first,
            second,
            length,
            beta1,
            beta2,
            finiteStatus), "pure BFP8 AdamW moment update");
    }

    internal static void AdamWBfp8Apply(
        int device,
        nint data,
        nint first,
        nint second,
        nint secondScale,
        int length,
        float learningRate,
        float weightDecay,
        float updateScale,
        float scaledEpsilon,
        bool applyWeightDecay,
        nint finiteStatus)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerAdamWBfp8Apply(
            device,
            data,
            first,
            second,
            secondScale,
            length,
            learningRate,
            weightDecay,
            updateScale,
            scaledEpsilon,
            applyWeightDecay,
            finiteStatus), "scale-aware pure BFP8 AdamW update");
    }

    internal static void AdamWAndPublish(int device, nint data, nint gradient,
        nint first, nint second, nint compute, int length, float beta1,
        float beta2, float learningRate, float weightDecay,
        float updateScale, float scaledEpsilon, bool applyWeightDecay,
        bool bfloat16State, bool physicalBFloat16)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerAdamWAndPublish(
            device, data, gradient, first, second, compute, physicalBFloat16,
            length, beta1, beta2, learningRate, weightDecay, updateScale,
            scaledEpsilon, applyWeightDecay, bfloat16State),
            "AdamW update and BF16 publish");
    }

    internal static void AdamWPureBFloat16(
        int device, nint data, nint gradient, nint first, nint second,
        int length, float beta1, float beta2, float learningRate,
        float weightDecay, float updateScale, float scaledEpsilon,
        bool applyWeightDecay)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerAdamWPureBFloat16(
            device, data, gradient, first, second, length, beta1, beta2,
            learningRate, weightDecay, updateScale, scaledEpsilon,
            applyWeightDecay), "pure BF16 AdamW update");
    }

    internal static void PublishBFloat16(int device, nint master,
        nint compute, int length, bool physical)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerPublishBFloat16(
            device, master, compute, length, physical),
            "publish BF16 master weights");
    }

    internal static void AccumulateFiniteStatus(
        int device,
        nint values,
        int length,
        nint finiteStatus)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerAccumulateFiniteStatus(
            device, values, length, finiteStatus),
            "accumulate optimizer finite status");
    }

    internal static void GatherStats(int device, nint sources,
        nint destination, int count)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerGatherStats(
            device, sources, destination, count),
            "gather optimizer statistics");
    }

    [StructLayout(LayoutKind.Sequential)]
    internal readonly struct AdamWChunkDescriptor(
        nint data,
        nint gradient,
        nint firstMoment,
        nint secondMoment,
        nint compute,
        int offset,
        int length,
        int applyWeightDecay,
        int physicalBFloat16,
        int bfloat16State,
        int pureBFloat16)
    {
        internal readonly nint Data = data;
        internal readonly nint Gradient = gradient;
        internal readonly nint FirstMoment = firstMoment;
        internal readonly nint SecondMoment = secondMoment;
        internal readonly nint Compute = compute;
        internal readonly int Offset = offset;
        internal readonly int Length = length;
        internal readonly int ApplyWeightDecay = applyWeightDecay;
        internal readonly int PhysicalBFloat16 = physicalBFloat16;
        internal readonly int BFloat16State = bfloat16State;
        internal readonly int PureBFloat16 = pureBFloat16;
    }

    internal static void AdamWMultiTensor(
        int device,
        nint chunks,
        int chunkCount,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        float updateScale,
        float scaledEpsilon)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerAdamWMultiTensor(
            device, chunks, chunkCount, beta1, beta2, learningRate,
            weightDecay, updateScale, scaledEpsilon),
            "AdamW multi-tensor update");
    }

    internal static void NekoMoments(int device, nint gradient, nint fast,
        nint slow, nint fastHat, nint slowHat, int length, float betaFast,
        float betaSlow, float fastCorrection, float slowCorrection)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerNekoMuonMoments(
            device, gradient, fast, slow, fastHat, slowHat, length, betaFast,
            betaSlow, fastCorrection, slowCorrection),
            "NekoMuon moments");
    }

    internal static void NekoInitialize(int device, nint source,
        nint destination, int length, int originalRows, int originalColumns,
        bool transpose, float inverseFastCorrection, float inverseNorm)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerNekoMuonInitialize(
            device, source, destination, length, originalRows,
            originalColumns, transpose,
            inverseFastCorrection, inverseNorm),
            "NekoMuon initialize");
    }

    internal static void NekoInitializeBFloat16(
        int device, nint source, nint destination, int length,
        int originalRows, int originalColumns, bool transpose,
        float inverseFastCorrection, float inverseNorm)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerNekoMuonInitializeBFloat16(
            device, source, destination, length, originalRows,
            originalColumns, transpose, inverseFastCorrection, inverseNorm),
            "pure BF16 NekoMuon initialize");
    }

    internal static void NekoUpdateDeviceControl(
        int device,
        nint stats,
        nint confidence,
        nint finiteStatus,
        float epsilon,
        float rho)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerNekoMuonUpdateDeviceControl(
            device, stats, confidence, finiteStatus, epsilon, rho),
            "NekoMuon device control update");
    }

    internal static void NekoInitializeFromDeviceStats(
        int device,
        nint source,
        nint destination,
        int length,
        int originalRows,
        int originalColumns,
        bool transpose,
        float inverseFastCorrection,
        nint stats,
        float epsilon,
        nint finiteStatus)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerNekoMuonInitializeFromDeviceStats(
            device,
            source,
            destination,
            length,
            originalRows,
            originalColumns,
            transpose,
            inverseFastCorrection,
            stats,
            epsilon,
            finiteStatus),
            "NekoMuon initialize from device statistics");
    }

    internal static void NekoInitializeBFloat16FromDeviceStats(
        int device, nint source, nint destination, int length,
        int originalRows, int originalColumns, bool transpose,
        float inverseFastCorrection, nint stats, float epsilon,
        nint finiteStatus)
    {
        Select(device);
        Check(CudaNativeGateway
            .OptimizerNekoMuonInitializeBFloat16FromDeviceStats(
                device, source, destination, length, originalRows,
                originalColumns, transpose, inverseFastCorrection, stats,
                epsilon, finiteStatus),
            "pure BF16 NekoMuon initialize from device statistics");
    }

    internal static void NekoInterpolate(int device, nint current, nint next,
        int length, float fraction)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerNekoMuonInterpolate(
            device, current, next, length, fraction),
            "NekoMuon interpolate");
    }

    internal static void NekoTransposeBack(int device, nint source,
        nint destination, int length, int originalRows, int originalColumns)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerNekoMuonTransposeBack(
            device, source, destination, length, originalRows,
            originalColumns), "NekoMuon transpose");
    }

    internal static void NekoApply(int device, nint data, nint update,
        int length, float learningRate, float finalScale, float weightDecay,
        bool applyWeightDecay)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerNekoMuonApply(
            device, data, update, length, learningRate, finalScale,
            weightDecay, applyWeightDecay), "NekoMuon update");
    }

    internal static void NekoApplyBFloat16(
        int device, nint data, nint update, int length, float learningRate,
        float finalScale, float weightDecay, bool applyWeightDecay)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerNekoMuonApplyBFloat16(
            device, data, update, length, learningRate, finalScale,
            weightDecay, applyWeightDecay), "pure BF16 NekoMuon update");
    }

    internal static void NekoCombine(int device, nint gram,
        nint gramSquared, int length, int rows, float a, float b, float c)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerNekoMuonCombine(
            device, gram, gramSquared, length, rows, a, b, c),
            "NekoMuon polynomial");
    }

    internal static void NekoCombineBatched(
        int device,
        nint gram,
        nint gramSquared,
        int matrixLength,
        int batch,
        int rows,
        float a,
        float b,
        float c)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerNekoMuonCombineBatched(
            device, gram, gramSquared, matrixLength, batch, rows, a, b, c),
            "NekoMuon batched polynomial");
    }

    internal static void SymmetricGram(int device, nint source,
        nint destination, int rows, int columns)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerNekoMuonSymmetricGram(
            device, source, destination, rows, columns,
            bfloat16Operands: false),
            "NekoMuon Gram");
    }

    internal static void SymmetricGramBFloat16Operands(
        int device, nint source, nint destination, int rows, int columns)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerNekoMuonSymmetricGram(
            device, source, destination, rows, columns,
            bfloat16Operands: true),
            "NekoMuon BF16-operand Gram");
    }

    internal static void NewtonSchulz(int device, nint source, nint gram,
        nint gramSquared, nint destination, int rows, int columns, float a,
        float b, float c)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerNekoMuonNewtonSchulz(
            device, source, gram, gramSquared, destination, rows, columns,
            a, b, c, bfloat16Operands: false),
            "NekoMuon Newton-Schulz");
    }

    internal static void NewtonSchulzBFloat16Operands(
        int device, nint source, nint gram, nint gramSquared,
        nint destination, int rows, int columns, float a, float b, float c)
    {
        Select(device);
        Check(CudaNativeGateway.OptimizerNekoMuonNewtonSchulz(
            device, source, gram, gramSquared, destination, rows, columns,
            a, b, c, bfloat16Operands: true),
            "NekoMuon BF16-operand Newton-Schulz");
    }

    private static void Select(int device)
        => NativeCudaRuntime.BindDeviceAndComputeStream(device);

    private static void Check(int status, string operation)
        => NativeCudaRuntime.Check(status, operation);

}
