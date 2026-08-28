using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// Versioned LayerNorm and fused residual/dropout/LayerNorm entry points.
/// Device is managed gateway metadata; the native kernel ABI remains intact.
/// </summary>
public static partial class CudaNativeGateway
{
    public static int LayerNormForward(
        int device, nint input, nint gamma, nint beta, nint output,
        nint normalized, nint inverses, int rows, int columns, float epsilon,
        nint stream)
    {
        EnsureTrainingKernelAbi("CUDA LayerNorm");
        return Complete(
            LayerNormNativeMethods.Forward(
                input, gamma, beta, output, normalized, inverses,
                rows, columns, epsilon, stream),
            CudaNativeOperation.LayerNormForward,
            device);
    }

    public static int LayerNormForwardBFloat16(
        int device, nint input, nint gamma, nint beta, nint output,
        nint means, nint inverses, int rows, int columns, float epsilon,
        nint stream)
    {
        EnsureTrainingKernelAbi("BF16 CUDA LayerNorm");
        return Complete(
            LayerNormNativeMethods.ForwardBFloat16(
                input, gamma, beta, output, means, inverses,
                rows, columns, epsilon, stream),
            CudaNativeOperation.LayerNormForwardBFloat16,
            device);
    }

    public static int LayerNormBackward(
        int device, nint input, nint gamma, nint means, nint inverses,
        nint outputGradient, nint inputGradient, nint gammaGradient,
        nint betaGradient, nint parameterScratch, int rows, int columns,
        nint stream)
    {
        EnsureTrainingKernelAbi("CUDA LayerNorm backward");
        return Complete(
            LayerNormNativeMethods.Backward(
                input, gamma, means, inverses, outputGradient, inputGradient,
                gammaGradient, betaGradient, parameterScratch, rows, columns,
                stream),
            CudaNativeOperation.LayerNormBackward,
            device);
    }

    public static int LayerNormBackwardBFloat16(
        int device, nint input, nint gamma, nint means, nint inverses,
        nint outputGradient, nint inputGradient, nint gammaGradient,
        nint betaGradient, nint parameterScratch, int rows, int columns,
        nint stream)
    {
        EnsureTrainingKernelAbi("BF16 CUDA LayerNorm backward");
        return Complete(
            LayerNormNativeMethods.BackwardBFloat16(
                input, gamma, means, inverses, outputGradient, inputGradient,
                gammaGradient, betaGradient, parameterScratch, rows, columns,
                stream),
            CudaNativeOperation.LayerNormBackwardBFloat16,
            device);
    }

    public static int ResidualDropoutLayerNormForward(
        int device, nint residual, nint branch, nint gamma, nint beta,
        nint output, nint normalized, nint inverses, int rows, int columns,
        uint seed, uint dropThreshold, float dropoutScale, float epsilon,
        nint stream)
    {
        EnsureTrainingKernelAbi("fused CUDA residual/dropout/LayerNorm");
        return Complete(
            LayerNormNativeMethods.ResidualDropoutForward(
                residual, branch, gamma, beta, output, normalized, inverses,
                rows, columns, seed, dropThreshold, dropoutScale, epsilon,
                stream),
            CudaNativeOperation.ResidualDropoutLayerNormForward,
            device);
    }

    public static int ResidualDropoutLayerNormForwardBFloat16(
        int device, nint residual, nint branch, nint gamma, nint beta,
        nint output, nint means, nint inverses, int rows, int columns,
        uint seed, uint dropThreshold, float dropoutScale, float epsilon,
        nint stream)
    {
        EnsureTrainingKernelAbi("fused BF16 CUDA residual/dropout/LayerNorm");
        return Complete(
            LayerNormNativeMethods.ResidualDropoutForwardBFloat16(
                residual, branch, gamma, beta, output, means, inverses,
                rows, columns, seed, dropThreshold, dropoutScale, epsilon,
                stream),
            CudaNativeOperation.ResidualDropoutLayerNormForwardBFloat16,
            device);
    }

    public static int ResidualDropoutLayerNormForwardBfp8Block128x512(
        int device,
        nint residualPayload, nint residualScales,
        nint branchPayload, nint branchScales,
        nint gammaPayload, nint gammaScales,
        nint betaPayload, nint betaScales,
        nint outputPayload, nint outputScales,
        nint means, nint inverses, int rows, int columns, int blockSize,
        uint seed, uint dropThreshold, float dropoutScale, float epsilon,
        nint stream)
    {
        EnsureDirectBfp8LayerNormAbi();
        return Complete(
            LayerNormNativeMethods.ResidualDropoutForwardBfp8Block128x512(
                residualPayload, residualScales, branchPayload, branchScales,
                gammaPayload, gammaScales, betaPayload, betaScales,
                outputPayload, outputScales, means, inverses, rows, columns,
                blockSize, seed, dropThreshold, dropoutScale, epsilon,
                stream),
            CudaNativeOperation.ResidualDropoutLayerNormForwardBFloat16,
            device);
    }

    public static int GraphResidualDropoutLayerNormForward(
        int device, nint residual, nint branch, nint gamma, nint beta,
        nint output, nint means, nint inverses, int rows, int columns,
        nint stepCounter, ulong operationSeed, uint dropThreshold,
        float dropoutScale, float epsilon, nint stream)
    {
        EnsureGraphFusedLayerNormAbi();
        return Complete(
            LayerNormNativeMethods.GraphResidualDropoutForward(
                residual, branch, gamma, beta, output, means, inverses,
                rows, columns, stepCounter, operationSeed, dropThreshold,
                dropoutScale, epsilon, stream),
            CudaNativeOperation.ResidualDropoutLayerNormForward,
            device);
    }

    public static int GraphResidualDropoutLayerNormForwardBFloat16(
        int device, nint residual, nint branch, nint gamma, nint beta,
        nint output, nint means, nint inverses, int rows, int columns,
        nint stepCounter, ulong operationSeed, uint dropThreshold,
        float dropoutScale, float epsilon, nint stream)
    {
        EnsureGraphFusedLayerNormAbi();
        return Complete(
            LayerNormNativeMethods.GraphResidualDropoutForwardBFloat16(
                residual, branch, gamma, beta, output, means, inverses,
                rows, columns, stepCounter, operationSeed, dropThreshold,
                dropoutScale, epsilon, stream),
            CudaNativeOperation.ResidualDropoutLayerNormForwardBFloat16,
            device);
    }

    public static int GraphResidualDropoutLayerNormForwardBfp8Block128x512(
        int device,
        nint residualPayload, nint residualScales,
        nint branchPayload, nint branchScales,
        nint gammaPayload, nint gammaScales,
        nint betaPayload, nint betaScales,
        nint outputPayload, nint outputScales,
        nint means, nint inverses, int rows, int columns, int blockSize,
        nint stepCounter, ulong operationSeed, uint dropThreshold,
        float dropoutScale, float epsilon, nint stream)
    {
        EnsureDirectBfp8LayerNormAbi();
        return Complete(
            LayerNormNativeMethods.GraphResidualDropoutForwardBfp8Block128x512(
                residualPayload, residualScales, branchPayload, branchScales,
                gammaPayload, gammaScales, betaPayload, betaScales,
                outputPayload, outputScales, means, inverses, rows, columns,
                blockSize, stepCounter, operationSeed, dropThreshold,
                dropoutScale, epsilon, stream),
            CudaNativeOperation.ResidualDropoutLayerNormForwardBFloat16,
            device);
    }

    public static int ResidualDropoutLayerNormBackward(
        int device, nint residual, nint branch, nint gamma, nint means,
        nint inverses, nint outputGradient, nint residualGradient,
        nint branchGradient, nint gammaGradient, nint betaGradient,
        nint parameterScratch, int rows, int columns, int sameParent,
        uint seed, uint dropThreshold, float dropoutScale, nint stream)
    {
        EnsureTrainingKernelAbi(
            "fused CUDA residual/dropout/LayerNorm backward");
        return Complete(
            LayerNormNativeMethods.ResidualDropoutBackward(
                residual, branch, gamma, means, inverses, outputGradient,
                residualGradient, branchGradient, gammaGradient, betaGradient,
                parameterScratch, rows, columns, sameParent, seed,
                dropThreshold, dropoutScale, stream),
            CudaNativeOperation.ResidualDropoutLayerNormBackward,
            device);
    }

    public static int ResidualDropoutLayerNormBackwardBFloat16(
        int device, nint residual, nint branch, nint gamma, nint means,
        nint inverses, nint outputGradient, nint residualGradient,
        nint branchGradient, nint gammaGradient, nint betaGradient,
        nint parameterScratch, int rows, int columns, int sameParent,
        uint seed, uint dropThreshold, float dropoutScale, nint stream)
    {
        EnsureTrainingKernelAbi(
            "fused BF16 CUDA residual/dropout/LayerNorm backward");
        return Complete(
            LayerNormNativeMethods.ResidualDropoutBackwardBFloat16(
                residual, branch, gamma, means, inverses, outputGradient,
                residualGradient, branchGradient, gammaGradient, betaGradient,
                parameterScratch, rows, columns, sameParent, seed,
                dropThreshold, dropoutScale, stream),
            CudaNativeOperation.ResidualDropoutLayerNormBackwardBFloat16,
            device);
    }

    public static int ResidualDropoutLayerNormBackwardBfp8Block128x512(
        int device,
        nint residualPayload, nint residualScales,
        nint branchPayload, nint branchScales,
        nint gammaPayload, nint gammaScales,
        nint means, nint inverses, nint outputGradient,
        nint residualGradient, nint branchGradient,
        nint gammaGradient, nint betaGradient, nint parameterScratch,
        int rows, int columns, int blockSize, int sameParent,
        uint seed, uint dropThreshold, float dropoutScale, nint stream)
    {
        EnsureDirectBfp8LayerNormAbi();
        return Complete(
            LayerNormNativeMethods.ResidualDropoutBackwardBfp8Block128x512(
                residualPayload, residualScales, branchPayload, branchScales,
                gammaPayload, gammaScales, means, inverses, outputGradient,
                residualGradient, branchGradient, gammaGradient,
                betaGradient, parameterScratch, rows, columns, blockSize,
                sameParent, seed, dropThreshold, dropoutScale, stream),
            CudaNativeOperation.ResidualDropoutLayerNormBackwardBFloat16,
            device);
    }

    public static int ResidualDropoutLayerNormBackwardBFloat16OneScan512(
        int device, nint residual, nint branch, nint gamma, nint means,
        nint inverses, nint outputGradient, nint residualGradient,
        nint branchGradient, nint gammaGradient, nint betaGradient,
        nint parameterScratch, int rows, int columns, int sameParent,
        uint seed, uint dropThreshold, float dropoutScale, nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.LayerNormOneScanMinor,
            "one-scan BF16 LayerNorm backward");
        return Complete(
            LayerNormNativeMethods.ResidualDropoutBackwardBFloat16OneScan512(
                residual, branch, gamma, means, inverses, outputGradient,
                residualGradient, branchGradient, gammaGradient, betaGradient,
                parameterScratch, rows, columns, sameParent, seed,
                dropThreshold, dropoutScale, stream),
            CudaNativeOperation.ResidualDropoutLayerNormBackwardBFloat16,
            device);
    }

    public static int GraphResidualDropoutLayerNormBackward(
        int device, nint residual, nint branch, nint gamma, nint means,
        nint inverses, nint outputGradient, nint residualGradient,
        nint branchGradient, nint gammaGradient, nint betaGradient,
        nint parameterScratch, int rows, int columns, int sameParent,
        nint stepCounter, ulong operationSeed, uint dropThreshold,
        float dropoutScale, nint stream)
    {
        EnsureGraphFusedLayerNormAbi();
        return Complete(
            LayerNormNativeMethods.GraphResidualDropoutBackward(
                residual, branch, gamma, means, inverses, outputGradient,
                residualGradient, branchGradient, gammaGradient, betaGradient,
                parameterScratch, rows, columns, sameParent, stepCounter,
                operationSeed, dropThreshold, dropoutScale, stream),
            CudaNativeOperation.ResidualDropoutLayerNormBackward,
            device);
    }

    public static int GraphResidualDropoutLayerNormBackwardBFloat16(
        int device, nint residual, nint branch, nint gamma, nint means,
        nint inverses, nint outputGradient, nint residualGradient,
        nint branchGradient, nint gammaGradient, nint betaGradient,
        nint parameterScratch, int rows, int columns, int sameParent,
        nint stepCounter, ulong operationSeed, uint dropThreshold,
        float dropoutScale, nint stream)
    {
        EnsureGraphFusedLayerNormAbi();
        return Complete(
            LayerNormNativeMethods.GraphResidualDropoutBackwardBFloat16(
                residual, branch, gamma, means, inverses, outputGradient,
                residualGradient, branchGradient, gammaGradient, betaGradient,
                parameterScratch, rows, columns, sameParent, stepCounter,
                operationSeed, dropThreshold, dropoutScale, stream),
            CudaNativeOperation.ResidualDropoutLayerNormBackwardBFloat16,
            device);
    }

    public static int GraphResidualDropoutLayerNormBackwardBfp8Block128x512(
        int device,
        nint residualPayload, nint residualScales,
        nint branchPayload, nint branchScales,
        nint gammaPayload, nint gammaScales,
        nint means, nint inverses, nint outputGradient,
        nint residualGradient, nint branchGradient,
        nint gammaGradient, nint betaGradient, nint parameterScratch,
        int rows, int columns, int blockSize, int sameParent,
        nint stepCounter, ulong operationSeed, uint dropThreshold,
        float dropoutScale, nint stream)
    {
        EnsureDirectBfp8LayerNormAbi();
        return Complete(
            LayerNormNativeMethods.GraphResidualDropoutBackwardBfp8Block128x512(
                residualPayload, residualScales, branchPayload, branchScales,
                gammaPayload, gammaScales, means, inverses, outputGradient,
                residualGradient, branchGradient, gammaGradient,
                betaGradient, parameterScratch, rows, columns, blockSize,
                sameParent, stepCounter, operationSeed, dropThreshold,
                dropoutScale, stream),
            CudaNativeOperation.ResidualDropoutLayerNormBackwardBFloat16,
            device);
    }

    public static int GraphResidualDropoutLayerNormBackwardBFloat16OneScan512(
        int device, nint residual, nint branch, nint gamma, nint means,
        nint inverses, nint outputGradient, nint residualGradient,
        nint branchGradient, nint gammaGradient, nint betaGradient,
        nint parameterScratch, int rows, int columns, int sameParent,
        nint stepCounter, ulong operationSeed, uint dropThreshold,
        float dropoutScale, nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.LayerNormOneScanMinor,
            "CUDA Graph one-scan BF16 LayerNorm backward");
        return Complete(
            LayerNormNativeMethods
                .GraphResidualDropoutBackwardBFloat16OneScan512(
                    residual, branch, gamma, means, inverses, outputGradient,
                    residualGradient, branchGradient, gammaGradient,
                    betaGradient, parameterScratch, rows, columns, sameParent,
                    stepCounter, operationSeed, dropThreshold, dropoutScale,
                    stream),
            CudaNativeOperation.ResidualDropoutLayerNormBackwardBFloat16,
            device);
    }

    public static int ResidualDropoutLayerNormBackwardBFloat16BranchGradient(
        int device, nint residual, nint branch, nint gamma, nint means,
        nint inverses, nint outputGradient, nint residualGradient,
        nint branchGradient, nint gammaGradient, nint betaGradient,
        nint parameterScratch, int rows, int columns, uint seed,
        uint dropThreshold, float dropoutScale, nint stream)
    {
        EnsureTrainingKernelAbi(
            "fused BF16 CUDA LayerNorm branch-gradient backward");
        return Complete(
            LayerNormNativeMethods.ResidualDropoutBackwardBFloat16BranchGradient(
                residual, branch, gamma, means, inverses, outputGradient,
                residualGradient, branchGradient, gammaGradient, betaGradient,
                parameterScratch, rows, columns, seed, dropThreshold,
                dropoutScale, stream),
            CudaNativeOperation
                .ResidualDropoutLayerNormBackwardBFloat16BranchGradient,
            device);
    }

    public static int ResidualDropoutLayerNormBackwardBFloat16IoGradient(
        int device, nint residual, nint branch, nint gamma, nint means,
        nint inverses, nint outputGradient, nint residualGradient,
        nint branchGradient, nint gammaGradient, nint betaGradient,
        nint parameterScratch, int rows, int columns, uint seed,
        uint dropThreshold, float dropoutScale, nint stream)
    {
        EnsureTrainingKernelAbi(
            "fused BF16 CUDA LayerNorm IO-gradient backward");
        return Complete(
            LayerNormNativeMethods.ResidualDropoutBackwardBFloat16IoGradient(
                residual, branch, gamma, means, inverses, outputGradient,
                residualGradient, branchGradient, gammaGradient, betaGradient,
                parameterScratch, rows, columns, seed, dropThreshold,
                dropoutScale, stream),
            CudaNativeOperation
                .ResidualDropoutLayerNormBackwardBFloat16IoGradient,
            device);
    }

    public static int
        GraphResidualDropoutLayerNormBackwardBFloat16BranchGradient(
            int device, nint residual, nint branch, nint gamma, nint means,
            nint inverses, nint outputGradient, nint residualGradient,
            nint branchGradient, nint gammaGradient, nint betaGradient,
            nint parameterScratch, int rows, int columns, nint stepCounter,
            ulong operationSeed, uint dropThreshold, float dropoutScale,
            nint stream)
    {
        EnsureGraphFusedLayerNormAbi();
        return Complete(
            LayerNormNativeMethods
                .GraphResidualDropoutBackwardBFloat16BranchGradient(
                    residual, branch, gamma, means, inverses, outputGradient,
                    residualGradient, branchGradient, gammaGradient,
                    betaGradient, parameterScratch, rows, columns,
                    stepCounter, operationSeed, dropThreshold, dropoutScale,
                    stream),
            CudaNativeOperation
                .ResidualDropoutLayerNormBackwardBFloat16BranchGradient,
            device);
    }

    public static int
        GraphResidualDropoutLayerNormBackwardBFloat16IoGradient(
            int device, nint residual, nint branch, nint gamma, nint means,
            nint inverses, nint outputGradient, nint residualGradient,
            nint branchGradient, nint gammaGradient, nint betaGradient,
            nint parameterScratch, int rows, int columns, nint stepCounter,
            ulong operationSeed, uint dropThreshold, float dropoutScale,
            nint stream)
    {
        EnsureGraphFusedLayerNormAbi();
        return Complete(
            LayerNormNativeMethods
                .GraphResidualDropoutBackwardBFloat16IoGradient(
                    residual, branch, gamma, means, inverses, outputGradient,
                    residualGradient, branchGradient, gammaGradient,
                    betaGradient, parameterScratch, rows, columns,
                    stepCounter, operationSeed, dropThreshold, dropoutScale,
                    stream),
            CudaNativeOperation
                .ResidualDropoutLayerNormBackwardBFloat16IoGradient,
            device);
    }

    private static void EnsureTrainingKernelAbi(string feature) =>
        EnsureMinimumAbiMinor(
            CudaAbiVersion.TrainingKernelGatewayMinor,
            feature);

    private static void EnsureGraphFusedLayerNormAbi() =>
        EnsureMinimumAbiMinor(
            CudaAbiVersion.GraphFusedLayerNormMinor,
            "CUDA Graph fused residual/dropout/LayerNorm");

    private static void EnsureDirectBfp8LayerNormAbi() =>
        EnsureMinimumAbiMinor(
            CudaAbiVersion.DirectBfp8LayerNormMinor,
            "direct block-BFP8 residual/dropout/LayerNorm");

    private static class LayerNormNativeMethods
    {
        [DllImport(LibraryName, EntryPoint = "nntrain_layer_norm_forward",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Forward(
            nint input, nint gamma, nint beta, nint output, nint normalized,
            nint inverses, int rows, int columns, float epsilon, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_layer_norm_forward_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ForwardBFloat16(
            nint input, nint gamma, nint beta, nint output, nint means,
            nint inverses, int rows, int columns, float epsilon, nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_layer_norm_backward",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Backward(
            nint input, nint gamma, nint means, nint inverses,
            nint outputGradient, nint inputGradient, nint gammaGradient,
            nint betaGradient, nint parameterScratch, int rows, int columns,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_layer_norm_backward_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BackwardBFloat16(
            nint input, nint gamma, nint means, nint inverses,
            nint outputGradient, nint inputGradient, nint gammaGradient,
            nint betaGradient, nint parameterScratch, int rows, int columns,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_residual_dropout_layer_norm_forward",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ResidualDropoutForward(
            nint residual, nint branch, nint gamma, nint beta, nint output,
            nint normalized, nint inverses, int rows, int columns, uint seed,
            uint dropThreshold, float dropoutScale, float epsilon, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_residual_dropout_layer_norm_forward_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ResidualDropoutForwardBFloat16(
            nint residual, nint branch, nint gamma, nint beta, nint output,
            nint means, nint inverses, int rows, int columns, uint seed,
            uint dropThreshold, float dropoutScale, float epsilon, nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_residual_dropout_layer_norm_forward_bfp8_block128_512",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ResidualDropoutForwardBfp8Block128x512(
            nint residualPayload, nint residualScales,
            nint branchPayload, nint branchScales,
            nint gammaPayload, nint gammaScales,
            nint betaPayload, nint betaScales,
            nint outputPayload, nint outputScales,
            nint means, nint inverses, int rows, int columns, int blockSize,
            uint seed, uint dropThreshold, float dropoutScale, float epsilon,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_residual_dropout_layer_norm_forward",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int GraphResidualDropoutForward(
            nint residual, nint branch, nint gamma, nint beta, nint output,
            nint means, nint inverses, int rows, int columns, nint stepCounter,
            ulong operationSeed, uint dropThreshold, float dropoutScale,
            float epsilon, nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_residual_dropout_layer_norm_forward_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int GraphResidualDropoutForwardBFloat16(
            nint residual, nint branch, nint gamma, nint beta, nint output,
            nint means, nint inverses, int rows, int columns, nint stepCounter,
            ulong operationSeed, uint dropThreshold, float dropoutScale,
            float epsilon, nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_residual_dropout_layer_norm_forward_bfp8_block128_512",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int GraphResidualDropoutForwardBfp8Block128x512(
            nint residualPayload, nint residualScales,
            nint branchPayload, nint branchScales,
            nint gammaPayload, nint gammaScales,
            nint betaPayload, nint betaScales,
            nint outputPayload, nint outputScales,
            nint means, nint inverses, int rows, int columns, int blockSize,
            nint stepCounter, ulong operationSeed, uint dropThreshold,
            float dropoutScale, float epsilon, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_residual_dropout_layer_norm_backward",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ResidualDropoutBackward(
            nint residual, nint branch, nint gamma, nint means, nint inverses,
            nint outputGradient, nint residualGradient, nint branchGradient,
            nint gammaGradient, nint betaGradient, nint parameterScratch,
            int rows, int columns, int sameParent, uint seed,
            uint dropThreshold, float dropoutScale, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_residual_dropout_layer_norm_backward_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ResidualDropoutBackwardBFloat16(
            nint residual, nint branch, nint gamma, nint means, nint inverses,
            nint outputGradient, nint residualGradient, nint branchGradient,
            nint gammaGradient, nint betaGradient, nint parameterScratch,
            int rows, int columns, int sameParent, uint seed,
            uint dropThreshold, float dropoutScale, nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_residual_dropout_layer_norm_backward_bfp8_block128_512",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ResidualDropoutBackwardBfp8Block128x512(
            nint residualPayload, nint residualScales,
            nint branchPayload, nint branchScales,
            nint gammaPayload, nint gammaScales,
            nint means, nint inverses, nint outputGradient,
            nint residualGradient, nint branchGradient,
            nint gammaGradient, nint betaGradient, nint parameterScratch,
            int rows, int columns, int blockSize, int sameParent,
            uint seed, uint dropThreshold, float dropoutScale, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_residual_dropout_layer_norm_backward_bf16_one_scan_512",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ResidualDropoutBackwardBFloat16OneScan512(
            nint residual, nint branch, nint gamma, nint means, nint inverses,
            nint outputGradient, nint residualGradient, nint branchGradient,
            nint gammaGradient, nint betaGradient, nint parameterScratch,
            int rows, int columns, int sameParent, uint seed,
            uint dropThreshold, float dropoutScale, nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_residual_dropout_layer_norm_backward",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int GraphResidualDropoutBackward(
            nint residual, nint branch, nint gamma, nint means, nint inverses,
            nint outputGradient, nint residualGradient, nint branchGradient,
            nint gammaGradient, nint betaGradient, nint parameterScratch,
            int rows, int columns, int sameParent, nint stepCounter,
            ulong operationSeed, uint dropThreshold, float dropoutScale,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_residual_dropout_layer_norm_backward_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int GraphResidualDropoutBackwardBFloat16(
            nint residual, nint branch, nint gamma, nint means, nint inverses,
            nint outputGradient, nint residualGradient, nint branchGradient,
            nint gammaGradient, nint betaGradient, nint parameterScratch,
            int rows, int columns, int sameParent, nint stepCounter,
            ulong operationSeed, uint dropThreshold, float dropoutScale,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_graph_residual_dropout_layer_norm_backward_bf16_one_scan_512",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int
            GraphResidualDropoutBackwardBFloat16OneScan512(
                nint residual, nint branch, nint gamma, nint means,
                nint inverses, nint outputGradient, nint residualGradient,
                nint branchGradient, nint gammaGradient, nint betaGradient,
                nint parameterScratch, int rows, int columns, int sameParent,
                nint stepCounter, ulong operationSeed, uint dropThreshold,
                float dropoutScale, nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_residual_dropout_layer_norm_backward_bf16_branch_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int
            ResidualDropoutBackwardBFloat16BranchGradient(
                nint residual, nint branch, nint gamma, nint means,
                nint inverses, nint outputGradient, nint residualGradient,
                nint branchGradient, nint gammaGradient, nint betaGradient,
                nint parameterScratch, int rows, int columns, uint seed,
                uint dropThreshold, float dropoutScale, nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_residual_dropout_layer_norm_backward_bf16_io_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ResidualDropoutBackwardBFloat16IoGradient(
            nint residual, nint branch, nint gamma, nint means, nint inverses,
            nint outputGradient, nint residualGradient, nint branchGradient,
            nint gammaGradient, nint betaGradient, nint parameterScratch,
            int rows, int columns, uint seed, uint dropThreshold,
            float dropoutScale, nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_residual_dropout_layer_norm_backward_bfp8_block128_512",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int GraphResidualDropoutBackwardBfp8Block128x512(
            nint residualPayload, nint residualScales,
            nint branchPayload, nint branchScales,
            nint gammaPayload, nint gammaScales,
            nint means, nint inverses, nint outputGradient,
            nint residualGradient, nint branchGradient,
            nint gammaGradient, nint betaGradient, nint parameterScratch,
            int rows, int columns, int blockSize, int sameParent,
            nint stepCounter, ulong operationSeed, uint dropThreshold,
            float dropoutScale, nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_residual_dropout_layer_norm_backward_bf16_branch_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int
            GraphResidualDropoutBackwardBFloat16BranchGradient(
                nint residual, nint branch, nint gamma, nint means,
                nint inverses, nint outputGradient, nint residualGradient,
                nint branchGradient, nint gammaGradient, nint betaGradient,
                nint parameterScratch, int rows, int columns,
                nint stepCounter, ulong operationSeed, uint dropThreshold,
                float dropoutScale, nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_cuda_graph_residual_dropout_layer_norm_backward_bf16_io_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int
            GraphResidualDropoutBackwardBFloat16IoGradient(
                nint residual, nint branch, nint gamma, nint means,
                nint inverses, nint outputGradient, nint residualGradient,
                nint branchGradient, nint gammaGradient, nint betaGradient,
                nint parameterScratch, int rows, int columns,
                nint stepCounter, ulong operationSeed, uint dropThreshold,
                float dropoutScale, nint stream);
    }
}
