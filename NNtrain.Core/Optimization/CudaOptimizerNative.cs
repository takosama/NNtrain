using System.Runtime.InteropServices;

namespace NNtrain;

internal static class CudaOptimizerNative
{
    private const string Library = "NNtrain.CudaKernels.dll";

    internal static void AdamW(int device, nint data, nint gradient,
        nint first, nint second, int length, float beta1, float beta2,
        float learningRate, float weightDecay, float updateScale,
        float scaledEpsilon, bool applyWeightDecay, bool bfloat16State)
    {
        Select(device);
        Check(bfloat16State
            ? AdamWBFloat16StateNative(data, gradient, first, second, length,
                beta1, beta2, learningRate, weightDecay, updateScale,
                scaledEpsilon, applyWeightDecay ? 1 : 0)
            : AdamWNative(data, gradient, first, second, length, beta1, beta2,
                learningRate, weightDecay, updateScale, scaledEpsilon,
                applyWeightDecay ? 1 : 0), "AdamW update");
    }

    internal static void AdamWAndPublish(int device, nint data, nint gradient,
        nint first, nint second, nint compute, int length, float beta1,
        float beta2, float learningRate, float weightDecay,
        float updateScale, float scaledEpsilon, bool applyWeightDecay,
        bool bfloat16State, bool physicalBFloat16)
    {
        Select(device);
        Check(bfloat16State
            ? AdamWBFloat16StatePublishNative(data, gradient, first, second,
                compute, physicalBFloat16 ? 1 : 0, length, beta1, beta2,
                learningRate, weightDecay, updateScale, scaledEpsilon,
                applyWeightDecay ? 1 : 0)
            : AdamWPublishNative(data, gradient, first, second, compute,
                physicalBFloat16 ? 1 : 0, length, beta1, beta2,
                learningRate, weightDecay, updateScale, scaledEpsilon,
                applyWeightDecay ? 1 : 0), "AdamW update and BF16 publish");
    }

    internal static void PublishBFloat16(int device, nint master,
        nint compute, int length, bool physical)
    {
        Select(device);
        Check(PublishBFloat16Native(master, compute, length,
            physical ? 1 : 0), "publish BF16 master weights");
    }

    internal static void GatherStats(int device, nint sources,
        nint destination, int count)
    {
        Select(device);
        Check(GatherStatsNative(sources, destination, count),
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
        int bfloat16State)
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
        Check(AdamWMultiTensorNative(
            chunks, chunkCount, beta1, beta2, learningRate, weightDecay,
            updateScale, scaledEpsilon), "AdamW multi-tensor update");
    }

    internal static void NekoMoments(int device, nint gradient, nint fast,
        nint slow, nint fastHat, nint slowHat, int length, float betaFast,
        float betaSlow, float fastCorrection, float slowCorrection)
    {
        Select(device);
        Check(NekoMomentsNative(gradient, fast, slow, fastHat, slowHat,
            length, betaFast, betaSlow, fastCorrection, slowCorrection),
            "NekoMuon moments");
    }

    internal static void NekoInitialize(int device, nint source,
        nint destination, int length, int originalRows, int originalColumns,
        bool transpose, float inverseFastCorrection, float inverseNorm)
    {
        Select(device);
        Check(NekoInitializeCorrectedNative(
            source, destination, length, originalRows,
            originalColumns, transpose ? 1 : 0,
            inverseFastCorrection, inverseNorm),
            "NekoMuon initialize");
    }

    internal static void NekoInterpolate(int device, nint current, nint next,
        int length, float fraction)
    {
        Select(device);
        Check(NekoInterpolateNative(current, next, length, fraction),
            "NekoMuon interpolate");
    }

    internal static void NekoTransposeBack(int device, nint source,
        nint destination, int length, int originalRows, int originalColumns)
    {
        Select(device);
        Check(NekoTransposeBackNative(source, destination, length,
            originalRows, originalColumns), "NekoMuon transpose");
    }

    internal static void NekoApply(int device, nint data, nint update,
        int length, float learningRate, float finalScale, float weightDecay,
        bool applyWeightDecay)
    {
        Select(device);
        Check(NekoApplyNative(data, update, length, learningRate, finalScale,
            weightDecay, applyWeightDecay ? 1 : 0), "NekoMuon update");
    }

    internal static void NekoCombine(int device, nint gram,
        nint gramSquared, int length, int rows, float a, float b, float c)
    {
        Select(device);
        Check(NekoCombineNative(gram, gramSquared, length, rows, a, b, c),
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
        Check(NekoCombineBatchedNative(
            gram, gramSquared, matrixLength, batch, rows, a, b, c),
            "NekoMuon batched polynomial");
    }

    internal static void SymmetricGram(int device, nint source,
        nint destination, int rows, int columns)
    {
        Select(device);
        Check(SymmetricGramNative(source, destination, rows, columns),
            "NekoMuon Gram");
    }

    internal static void NewtonSchulz(int device, nint source, nint gram,
        nint gramSquared, nint destination, int rows, int columns, float a,
        float b, float c)
    {
        Select(device);
        Check(NewtonSchulzNative(source, gram, gramSquared, destination, rows,
            columns, a, b, c), "NekoMuon Newton-Schulz");
    }

    private static void Select(int device)
    {
        var accelerator = ForgetMemoryV2Cuda.GetAccelerator(device);
        accelerator.Bind();
        NativeCudaRuntime.Check(
            NativeCudaRuntime.UseExternalStreamNative(accelerator.DefaultStream),
            "select CUDA stream");
    }

    private static void Check(int status, string operation)
        => NativeCudaRuntime.Check(status, operation);

    [DllImport(Library, EntryPoint = "nntrain_optimizer_adamw", CallingConvention = CallingConvention.Cdecl)]
    private static extern int AdamWNative(nint data, nint gradient, nint first, nint second, int length, float beta1, float beta2, float learningRate, float weightDecay, float updateScale, float scaledEpsilon, int applyWeightDecay);
    [DllImport(Library, EntryPoint = "nntrain_optimizer_adamw_bf16_state", CallingConvention = CallingConvention.Cdecl)]
    private static extern int AdamWBFloat16StateNative(nint data, nint gradient, nint first, nint second, int length, float beta1, float beta2, float learningRate, float weightDecay, float updateScale, float scaledEpsilon, int applyWeightDecay);
    [DllImport(Library, EntryPoint = "nntrain_optimizer_adamw_publish", CallingConvention = CallingConvention.Cdecl)]
    private static extern int AdamWPublishNative(nint data, nint gradient, nint first, nint second, nint compute, int physicalBFloat16, int length, float beta1, float beta2, float learningRate, float weightDecay, float updateScale, float scaledEpsilon, int applyWeightDecay);
    [DllImport(Library, EntryPoint = "nntrain_optimizer_adamw_bf16_state_publish", CallingConvention = CallingConvention.Cdecl)]
    private static extern int AdamWBFloat16StatePublishNative(nint data, nint gradient, nint first, nint second, nint compute, int physicalBFloat16, int length, float beta1, float beta2, float learningRate, float weightDecay, float updateScale, float scaledEpsilon, int applyWeightDecay);
    [DllImport(Library, EntryPoint = "nntrain_optimizer_publish_bf16", CallingConvention = CallingConvention.Cdecl)]
    private static extern int PublishBFloat16Native(nint master, nint compute, int length, int physical);
    [DllImport(Library, EntryPoint = "nntrain_optimizer_gather_stats", CallingConvention = CallingConvention.Cdecl)]
    private static extern int GatherStatsNative(nint sources, nint destination, int count);
    [DllImport(Library, EntryPoint = "nntrain_optimizer_adamw_multi_tensor", CallingConvention = CallingConvention.Cdecl)]
    private static extern int AdamWMultiTensorNative(nint chunks, int chunkCount, float beta1, float beta2, float learningRate, float weightDecay, float updateScale, float scaledEpsilon);
    [DllImport(Library, EntryPoint = "nntrain_optimizer_neko_moments", CallingConvention = CallingConvention.Cdecl)]
    private static extern int NekoMomentsNative(nint gradient, nint fast, nint slow, nint fastHat, nint slowHat, int length, float betaFast, float betaSlow, float fastCorrection, float slowCorrection);
    [DllImport(Library, EntryPoint = "nntrain_optimizer_neko_initialize", CallingConvention = CallingConvention.Cdecl)]
    private static extern int NekoInitializeNative(nint source, nint destination, int length, int originalRows, int originalColumns, int transpose, float inverseNorm);
    [DllImport(Library, EntryPoint = "nntrain_optimizer_neko_initialize_corrected", CallingConvention = CallingConvention.Cdecl)]
    private static extern int NekoInitializeCorrectedNative(nint source, nint destination, int length, int originalRows, int originalColumns, int transpose, float inverseFastCorrection, float inverseNorm);
    [DllImport(Library, EntryPoint = "nntrain_optimizer_neko_interpolate", CallingConvention = CallingConvention.Cdecl)]
    private static extern int NekoInterpolateNative(nint current, nint next, int length, float fraction);
    [DllImport(Library, EntryPoint = "nntrain_optimizer_neko_transpose_back", CallingConvention = CallingConvention.Cdecl)]
    private static extern int NekoTransposeBackNative(nint source, nint destination, int length, int originalRows, int originalColumns);
    [DllImport(Library, EntryPoint = "nntrain_optimizer_neko_apply", CallingConvention = CallingConvention.Cdecl)]
    private static extern int NekoApplyNative(nint data, nint update, int length, float learningRate, float finalScale, float weightDecay, int applyWeightDecay);
    [DllImport(Library, EntryPoint = "nntrain_optimizer_neko_combine", CallingConvention = CallingConvention.Cdecl)]
    private static extern int NekoCombineNative(nint gram, nint gramSquared, int length, int rows, float a, float b, float c);
    [DllImport(Library, EntryPoint = "nntrain_optimizer_neko_combine_batched", CallingConvention = CallingConvention.Cdecl)]
    private static extern int NekoCombineBatchedNative(nint gram, nint gramSquared, int matrixLength, int batch, int rows, float a, float b, float c);
    [DllImport(Library, EntryPoint = "nntrain_optimizer_symmetric_gram", CallingConvention = CallingConvention.Cdecl)]
    private static extern int SymmetricGramNative(nint source, nint destination, int rows, int columns);
    [DllImport(Library, EntryPoint = "nntrain_optimizer_newton_schulz", CallingConvention = CallingConvention.Cdecl)]
    private static extern int NewtonSchulzNative(nint source, nint gram, nint gramSquared, nint destination, int rows, int columns, float a, float b, float c);
}
