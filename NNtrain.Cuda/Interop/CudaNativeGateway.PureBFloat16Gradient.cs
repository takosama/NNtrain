using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// CUDA-resident pure-BF16 gradient primitives. Reductions accumulate in
/// FP32/FP64 internally, while every externally visible gradient remains in
/// BF16 storage. All launches accept the execution lane's explicit stream.
/// </summary>
public static partial class CudaNativeGateway
{
    public static int EmbeddingBackwardReducedBFloat16Gradient(
        int device,
        nint indices,
        nint outputGradient,
        nint tableGradient,
        nint workspace,
        int workspaceInts,
        int length,
        int width,
        nint stream)
    {
        EnsurePureBFloat16GradientAbi();
        return Complete(
            PureBFloat16GradientNativeMethods
                .EmbeddingBackwardReducedBFloat16Gradient(
                    indices,
                    outputGradient,
                    tableGradient,
                    workspace,
                    workspaceInts,
                    length,
                    width,
                    stream),
            CudaNativeOperation.EmbeddingBackwardBFloat16Gradient,
            device);
    }

    public static int EmbeddingPositionsBackwardReducedBFloat16Gradient(
        int device,
        nint indices,
        nint outputGradient,
        nint tokenGradient,
        nint positionGradient,
        nint workspace,
        int workspaceInts,
        int length,
        int sequence,
        int width,
        nint stream)
    {
        EnsurePureBFloat16GradientAbi();
        return Complete(
            PureBFloat16GradientNativeMethods
                .EmbeddingPositionsBackwardReducedBFloat16Gradient(
                    indices,
                    outputGradient,
                    tokenGradient,
                    positionGradient,
                    workspace,
                    workspaceInts,
                    length,
                    sequence,
                    width,
                    stream),
            CudaNativeOperation.EmbeddingPositionsBackwardBFloat16Gradient,
            device);
    }

    public static int DropoutBackwardBFloat16Gradient(
        int device,
        nint outputGradient,
        nint inputGradient,
        int length,
        uint seed,
        uint threshold,
        float scale,
        nint stream)
    {
        EnsurePureBFloat16GradientAbi();
        return Complete(
            PureBFloat16GradientNativeMethods.DropoutBackwardBFloat16Gradient(
                outputGradient,
                inputGradient,
                length,
                seed,
                threshold,
                scale,
                stream),
            CudaNativeOperation.DropoutBackwardBFloat16Gradient,
            device);
    }

    public static int AddDropoutBackwardBFloat16Gradient(
        int device,
        nint outputGradient,
        nint residualGradient,
        nint branchGradient,
        int length,
        bool sameParent,
        uint seed,
        uint threshold,
        float scale,
        nint stream)
    {
        EnsurePureBFloat16GradientAbi();
        return Complete(
            PureBFloat16GradientNativeMethods
                .AddDropoutBackwardBFloat16Gradient(
                    outputGradient,
                    residualGradient,
                    branchGradient,
                    length,
                    sameParent ? 1 : 0,
                    seed,
                    threshold,
                    scale,
                    stream),
            CudaNativeOperation.AddDropoutBackwardBFloat16Gradient,
            device);
    }

    public static int LinearBiasBackwardBFloat16Gradient(
        int device,
        nint outputGradient,
        nint biasGradient,
        int rows,
        int width,
        nint stream)
    {
        EnsurePureBFloat16GradientAbi();
        return Complete(
            PureBFloat16GradientNativeMethods
                .LinearBiasBackwardBFloat16Gradient(
                    outputGradient,
                    biasGradient,
                    rows,
                    width,
                    stream),
            CudaNativeOperation.LinearBiasBackwardBFloat16Gradient,
            device);
    }

    public static int BFloat16GradientSquaredSum(
        int device,
        nint values,
        int length,
        nint result,
        nint stream)
    {
        EnsurePureBFloat16GradientAbi();
        return Complete(
            PureBFloat16GradientNativeMethods.BFloat16GradientSquaredSum(
                values,
                length,
                result,
                stream),
            CudaNativeOperation.BFloat16GradientSquaredSum,
            device);
    }

    public static int BFloat16GradientScale(
        int device,
        nint values,
        int length,
        float multiplier,
        nint stream)
    {
        EnsurePureBFloat16GradientAbi();
        return Complete(
            PureBFloat16GradientNativeMethods.BFloat16GradientScale(
                values,
                length,
                multiplier,
                stream),
            CudaNativeOperation.BFloat16GradientScale,
            device);
    }

    public static int GraphDropoutBackwardBFloat16Gradient(
        int device,
        nint stepCounter,
        ulong operationSeed,
        float dropoutProbability,
        nint outputGradient,
        nint inputGradient,
        int length,
        nint stream)
    {
        EnsurePureBFloat16GradientAbi();
        return Complete(
            PureBFloat16GradientNativeMethods
                .GraphDropoutBackwardBFloat16Gradient(
                    stepCounter,
                    operationSeed,
                    dropoutProbability,
                    outputGradient,
                    inputGradient,
                    length,
                    stream),
            CudaNativeOperation.GraphDropoutBackward,
            device);
    }

    public static int GraphAddDropoutBackwardBFloat16Gradient(
        int device,
        nint stepCounter,
        ulong operationSeed,
        float dropoutProbability,
        nint outputGradient,
        nint residualGradient,
        nint branchGradient,
        int length,
        bool sameParent,
        nint stream)
    {
        EnsurePureBFloat16GradientAbi();
        return Complete(
            PureBFloat16GradientNativeMethods
                .GraphAddDropoutBackwardBFloat16Gradient(
                    stepCounter,
                    operationSeed,
                    dropoutProbability,
                    outputGradient,
                    residualGradient,
                    branchGradient,
                    length,
                    sameParent ? 1 : 0,
                    stream),
            CudaNativeOperation.GraphAddDropoutBackward,
            device);
    }

    private static void EnsurePureBFloat16GradientAbi()
        => EnsureMinimumAbiMinor(
            CudaAbiVersion.PureBFloat16GradientMinor,
            "CUDA-resident pure-BF16 gradients");

    private static class PureBFloat16GradientNativeMethods
    {
        [DllImport(
            LibraryName,
            EntryPoint =
                "nntrain_tensor_embedding_backward_reduced_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int
            EmbeddingBackwardReducedBFloat16Gradient(
                nint indices,
                nint outputGradient,
                nint tableGradient,
                nint workspace,
                int workspaceInts,
                int length,
                int width,
                nint stream);

        [DllImport(
            LibraryName,
            EntryPoint =
                "nntrain_tensor_embedding_positions_backward_reduced_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int
            EmbeddingPositionsBackwardReducedBFloat16Gradient(
                nint indices,
                nint outputGradient,
                nint tokenGradient,
                nint positionGradient,
                nint workspace,
                int workspaceInts,
                int length,
                int sequence,
                int width,
                nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_tensor_dropout_backward_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DropoutBackwardBFloat16Gradient(
            nint outputGradient,
            nint inputGradient,
            int length,
            uint seed,
            uint threshold,
            float scale,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint =
                "nntrain_tensor_add_dropout_backward_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AddDropoutBackwardBFloat16Gradient(
            nint outputGradient,
            nint residualGradient,
            nint branchGradient,
            int length,
            int sameParent,
            uint seed,
            uint threshold,
            float scale,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint =
                "nntrain_tensor_linear_bias_backward_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int LinearBiasBackwardBFloat16Gradient(
            nint outputGradient,
            nint biasGradient,
            int rows,
            int width,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_tensor_bf16_gradient_squared_sum",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BFloat16GradientSquaredSum(
            nint values,
            int length,
            nint result,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_tensor_bf16_gradient_scale",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BFloat16GradientScale(
            nint values,
            int length,
            float multiplier,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint =
                "nntrain_cuda_graph_dropout_backward_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int GraphDropoutBackwardBFloat16Gradient(
            nint stepCounter,
            ulong operationSeed,
            float dropoutProbability,
            nint outputGradient,
            nint inputGradient,
            int length,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint =
                "nntrain_cuda_graph_add_dropout_backward_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int GraphAddDropoutBackwardBFloat16Gradient(
            nint stepCounter,
            ulong operationSeed,
            float dropoutProbability,
            nint outputGradient,
            nint residualGradient,
            nint branchGradient,
            int length,
            int sameParent,
            nint stream);
    }
}
