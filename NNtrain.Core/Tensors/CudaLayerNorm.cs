using System.Runtime.InteropServices;

namespace NNtrain;

internal static class CudaLayerNorm
{
    private const string Library = "NNtrain.CudaKernels";
    private const int ParameterRowsPerTile = 1024;
    private static int _availability;
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<
        (int Device, int Length),
        Lazy<NativeCudaBuffer<float>>> ParameterScratch = new();

    internal static bool TryForward(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> input,
        NativeCudaBuffer<float> gamma,
        NativeCudaBuffer<float> beta,
        NativeCudaBuffer<float> output,
        NativeCudaBuffer<float> normalized,
        NativeCudaBuffer<float> inverses,
        int rows,
        int columns,
        float epsilon)
    {
        if (Volatile.Read(ref _availability) < 0)
            return false;
        try
        {
            accelerator.Bind();
            int status = Forward(
                input.NativePtr, gamma.NativePtr, beta.NativePtr,
                output.NativePtr, normalized.NativePtr, inverses.NativePtr,
                rows, columns, epsilon, Stream(accelerator));
            return Complete(status);
        }
        catch (Exception exception) when (IsUnavailable(exception))
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }

    internal static bool TryForwardBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> input,
        NativeCudaBuffer<ushort> gamma,
        NativeCudaBuffer<ushort> beta,
        NativeCudaBuffer<ushort> output,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        int rows,
        int columns,
        float epsilon)
    {
        if (Volatile.Read(ref _availability) < 0)
            return false;
        try
        {
            accelerator.Bind();
            int status = ForwardBFloat16(
                input.NativePtr, gamma.NativePtr, beta.NativePtr,
                output.NativePtr, means.NativePtr, inverses.NativePtr,
                rows, columns, epsilon, Stream(accelerator));
            return Complete(status);
        }
        catch (Exception exception) when (IsUnavailable(exception))
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }

    internal static bool TryFusedForward(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> residual,
        NativeCudaBuffer<float> branch,
        NativeCudaBuffer<float> gamma,
        NativeCudaBuffer<float> beta,
        NativeCudaBuffer<float> output,
        NativeCudaBuffer<float> normalized,
        NativeCudaBuffer<float> inverses,
        int rows,
        int columns,
        uint seed,
        uint dropThreshold,
        float dropoutScale,
        float epsilon)
    {
        if (Volatile.Read(ref _availability) < 0)
            return false;
        try
        {
            accelerator.Bind();
            int status = FusedForward(
                residual.NativePtr, branch.NativePtr, gamma.NativePtr,
                beta.NativePtr, output.NativePtr, normalized.NativePtr,
                inverses.NativePtr, rows, columns, seed, dropThreshold,
                dropoutScale, epsilon, Stream(accelerator));
            return Complete(status);
        }
        catch (Exception exception) when (IsUnavailable(exception))
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }

    internal static bool TryFusedForwardBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> residual,
        NativeCudaBuffer<ushort> branch,
        NativeCudaBuffer<ushort> gamma,
        NativeCudaBuffer<ushort> beta,
        NativeCudaBuffer<ushort> output,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        int rows,
        int columns,
        uint seed,
        uint dropThreshold,
        float dropoutScale,
        float epsilon)
    {
        if (Volatile.Read(ref _availability) < 0)
            return false;
        try
        {
            accelerator.Bind();
            int status = FusedForwardBFloat16(
                residual.NativePtr, branch.NativePtr, gamma.NativePtr,
                beta.NativePtr, output.NativePtr, means.NativePtr,
                inverses.NativePtr, rows, columns, seed, dropThreshold,
                dropoutScale, epsilon, Stream(accelerator));
            return Complete(status);
        }
        catch (Exception exception) when (IsUnavailable(exception))
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }

    internal static void Backward(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> input,
        NativeCudaBuffer<float> gamma,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> inputGradient,
        NativeCudaBuffer<float> gammaGradient,
        NativeCudaBuffer<float> betaGradient,
        int rows,
        int columns)
    {
        accelerator.Bind();
        NativeCudaBuffer<float> parameterScratch = GetParameterScratch(
            accelerator, rows, columns);
        ThrowIfFailed(BackwardNative(
            input.NativePtr, gamma.NativePtr, means.NativePtr,
            inverses.NativePtr,
            outputGradient.NativePtr, inputGradient.NativePtr,
            gammaGradient.NativePtr, betaGradient.NativePtr,
            parameterScratch.NativePtr,
            rows, columns, Stream(accelerator)));
    }

    internal static void BackwardBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> input,
        NativeCudaBuffer<ushort> gamma,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> inputGradient,
        NativeCudaBuffer<float> gammaGradient,
        NativeCudaBuffer<float> betaGradient,
        int rows,
        int columns)
    {
        accelerator.Bind();
        NativeCudaBuffer<float> parameterScratch = GetParameterScratch(
            accelerator, rows, columns);
        ThrowIfFailed(BackwardBFloat16Native(
            input.NativePtr, gamma.NativePtr, means.NativePtr,
            inverses.NativePtr,
            outputGradient.NativePtr, inputGradient.NativePtr,
            gammaGradient.NativePtr, betaGradient.NativePtr,
            parameterScratch.NativePtr,
            rows, columns, Stream(accelerator)));
    }

    internal static void FusedBackward(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> residual,
        NativeCudaBuffer<float> branch,
        NativeCudaBuffer<float> gamma,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> residualGradient,
        NativeCudaBuffer<float> branchGradient,
        NativeCudaBuffer<float> gammaGradient,
        NativeCudaBuffer<float> betaGradient,
        int rows,
        int columns,
        bool sameParent,
        uint seed,
        uint dropThreshold,
        float dropoutScale)
    {
        accelerator.Bind();
        NativeCudaBuffer<float> parameterScratch = GetParameterScratch(
            accelerator, rows, columns);
        ThrowIfFailed(FusedBackwardNative(
            residual.NativePtr, branch.NativePtr, gamma.NativePtr,
            means.NativePtr, inverses.NativePtr,
            outputGradient.NativePtr, residualGradient.NativePtr,
            branchGradient.NativePtr, gammaGradient.NativePtr,
            betaGradient.NativePtr, parameterScratch.NativePtr,
            rows, columns, sameParent ? 1 : 0,
            seed, dropThreshold, dropoutScale, Stream(accelerator)));
    }

    internal static void FusedBackwardBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> residual,
        NativeCudaBuffer<ushort> branch,
        NativeCudaBuffer<ushort> gamma,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> residualGradient,
        NativeCudaBuffer<float> branchGradient,
        NativeCudaBuffer<float> gammaGradient,
        NativeCudaBuffer<float> betaGradient,
        int rows,
        int columns,
        bool sameParent,
        uint seed,
        uint dropThreshold,
        float dropoutScale)
    {
        accelerator.Bind();
        NativeCudaBuffer<float> parameterScratch = GetParameterScratch(
            accelerator, rows, columns);
        ThrowIfFailed(FusedBackwardBFloat16Native(
            residual.NativePtr, branch.NativePtr, gamma.NativePtr,
            means.NativePtr, inverses.NativePtr,
            outputGradient.NativePtr, residualGradient.NativePtr,
            branchGradient.NativePtr, gammaGradient.NativePtr,
            betaGradient.NativePtr, parameterScratch.NativePtr,
            rows, columns, sameParent ? 1 : 0,
            seed, dropThreshold, dropoutScale, Stream(accelerator)));
    }

    internal static void FusedBackwardBFloat16DirectBranch(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> residual,
        NativeCudaBuffer<ushort> branch,
        NativeCudaBuffer<ushort> gamma,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        NativeCudaBuffer<float>? outputGradient,
        NativeCudaBuffer<ushort>? outputGradientBFloat16,
        NativeCudaBuffer<float> residualGradient,
        NativeCudaBuffer<ushort> branchGradientBFloat16,
        NativeCudaBuffer<float> gammaGradient,
        NativeCudaBuffer<float> betaGradient,
        int rows,
        int columns,
        uint seed,
        uint dropThreshold,
        float dropoutScale)
    {
        accelerator.Bind();
        NativeCudaBuffer<float> parameterScratch = GetParameterScratch(
            accelerator, rows, columns);
        int status;
        if (outputGradientBFloat16 is not null)
        {
            status = FusedBackwardBFloat16IoGradientNative(
                residual.NativePtr, branch.NativePtr, gamma.NativePtr,
                means.NativePtr, inverses.NativePtr,
                outputGradientBFloat16.NativePtr, residualGradient.NativePtr,
                branchGradientBFloat16.NativePtr, gammaGradient.NativePtr,
                betaGradient.NativePtr, parameterScratch.NativePtr,
                rows, columns, seed, dropThreshold, dropoutScale,
                Stream(accelerator));
        }
        else
        {
            if (outputGradient is null)
            {
                throw new ArgumentNullException(
                    nameof(outputGradient),
                    "LayerNorm backward requires an output gradient.");
            }
            status = FusedBackwardBFloat16BranchGradientNative(
                residual.NativePtr, branch.NativePtr, gamma.NativePtr,
                means.NativePtr, inverses.NativePtr,
                outputGradient.NativePtr, residualGradient.NativePtr,
                branchGradientBFloat16.NativePtr, gammaGradient.NativePtr,
                betaGradient.NativePtr, parameterScratch.NativePtr,
                rows, columns, seed, dropThreshold, dropoutScale,
                Stream(accelerator));
        }
        ThrowIfFailed(status);
    }

    private static nint Stream(NativeCudaDevice accelerator) =>
        accelerator.DefaultStream;

    private static NativeCudaBuffer<float> GetParameterScratch(
        NativeCudaDevice accelerator,
        int rows,
        int columns)
    {
        int rowTiles = checked(
            (rows + ParameterRowsPerTile - 1) / ParameterRowsPerTile);
        int length = checked(2 * rowTiles * columns);
        return ParameterScratch.GetOrAdd(
            (accelerator.Index, length),
            static key => new Lazy<NativeCudaBuffer<float>>(
                () => ForgetMemoryV2Cuda.GetAccelerator(key.Device)
                    .Allocate1D<float>(key.Length),
                LazyThreadSafetyMode.ExecutionAndPublication)).Value;
    }

    private static bool Complete(int status)
    {
        ThrowIfFailed(status);
        Volatile.Write(ref _availability, 1);
        return true;
    }

    private static void ThrowIfFailed(int status)
    {
        if (status != 0)
            throw new InvalidOperationException($"LayerNorm CUDA error {status}.");
    }

    private static bool IsUnavailable(Exception exception) =>
        exception is DllNotFoundException
        or EntryPointNotFoundException
        or BadImageFormatException;

    [DllImport(Library, EntryPoint = "nntrain_layer_norm_forward",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int Forward(
        nint input, nint gamma, nint beta, nint output, nint normalized,
        nint inverses, int rows, int columns, float epsilon, nint stream);

    [DllImport(Library, EntryPoint = "nntrain_layer_norm_forward_bf16",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int ForwardBFloat16(
        nint input, nint gamma, nint beta, nint output, nint normalized,
        nint inverses, int rows, int columns, float epsilon, nint stream);

    [DllImport(Library, EntryPoint = "nntrain_layer_norm_backward",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int BackwardNative(
        nint input, nint gamma, nint means, nint inverses, nint outputGradient,
        nint inputGradient, nint gammaGradient, nint betaGradient,
        nint parameterScratch,
        int rows, int columns, nint stream);

    [DllImport(Library, EntryPoint = "nntrain_layer_norm_backward_bf16",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int BackwardBFloat16Native(
        nint input, nint gamma, nint means, nint inverses, nint outputGradient,
        nint inputGradient, nint gammaGradient, nint betaGradient,
        nint parameterScratch,
        int rows, int columns, nint stream);

    [DllImport(Library, EntryPoint = "nntrain_residual_dropout_layer_norm_forward",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int FusedForward(
        nint residual, nint branch, nint gamma, nint beta, nint output,
        nint normalized, nint inverses, int rows, int columns, uint seed,
        uint dropThreshold, float dropoutScale, float epsilon, nint stream);

    [DllImport(Library, EntryPoint = "nntrain_residual_dropout_layer_norm_forward_bf16",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int FusedForwardBFloat16(
        nint residual, nint branch, nint gamma, nint beta, nint output,
        nint normalized, nint inverses, int rows, int columns, uint seed,
        uint dropThreshold, float dropoutScale, float epsilon, nint stream);

    [DllImport(Library, EntryPoint = "nntrain_residual_dropout_layer_norm_backward",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int FusedBackwardNative(
        nint residual, nint branch, nint gamma, nint means, nint inverses,
        nint outputGradient,
        nint residualGradient, nint branchGradient, nint gammaGradient,
        nint betaGradient, nint parameterScratch,
        int rows, int columns, int sameParent, uint seed,
        uint dropThreshold, float dropoutScale, nint stream);

    [DllImport(Library, EntryPoint = "nntrain_residual_dropout_layer_norm_backward_bf16",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int FusedBackwardBFloat16Native(
        nint residual, nint branch, nint gamma, nint means, nint inverses,
        nint outputGradient,
        nint residualGradient, nint branchGradient, nint gammaGradient,
        nint betaGradient, nint parameterScratch,
        int rows, int columns, int sameParent, uint seed,
        uint dropThreshold, float dropoutScale, nint stream);

    [DllImport(
        Library,
        EntryPoint = "nntrain_residual_dropout_layer_norm_backward_bf16_branch_gradient",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int FusedBackwardBFloat16BranchGradientNative(
        nint residual, nint branch, nint gamma, nint means, nint inverses,
        nint outputGradient,
        nint residualGradient, nint branchGradient, nint gammaGradient,
        nint betaGradient, nint parameterScratch,
        int rows, int columns, uint seed,
        uint dropThreshold, float dropoutScale, nint stream);

    [DllImport(
        Library,
        EntryPoint = "nntrain_residual_dropout_layer_norm_backward_bf16_io_gradient",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int FusedBackwardBFloat16IoGradientNative(
        nint residual, nint branch, nint gamma, nint means, nint inverses,
        nint outputGradient,
        nint residualGradient, nint branchGradient, nint gammaGradient,
        nint betaGradient, nint parameterScratch,
        int rows, int columns, uint seed,
        uint dropThreshold, float dropoutScale, nint stream);
}
