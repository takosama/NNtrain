using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// Versioned gateway for the CUDA tensor kernels used by the Core facade.
/// All failures are captured before control returns to Core.
/// </summary>
public static partial class CudaNativeGateway
{
    public static int TensorAdd(
        int device, nint left, nint right, nint output, int length,
        bool bfloat16)
    {
        EnsureCompatibleAbi();
        return Complete(
            bfloat16
                ? TensorPrimitiveNativeMethods.AddBFloat16(
                    left, right, output, length)
                : TensorPrimitiveNativeMethods.AddFloat32(
                    left, right, output, length),
            TensorOperation(bfloat16),
            device);
    }

    public static int TensorAddBackward(
        int device, nint outputGradient, nint leftGradient,
        nint rightGradient, int length, bool sameParent)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.AddBackward(
                outputGradient, leftGradient, rightGradient, length,
                sameParent ? 1 : 0),
            CudaNativeOperation.TensorPrimitiveFloat32,
            device);
    }

    public static int TensorEmbedding(
        int device, nint table, nint indices, nint output, int length,
        int width, bool bfloat16)
    {
        EnsureCompatibleAbi();
        return Complete(
            bfloat16
                ? TensorPrimitiveNativeMethods.EmbeddingBFloat16(
                    table, indices, output, length, width)
                : TensorPrimitiveNativeMethods.EmbeddingFloat32(
                    table, indices, output, length, width),
            TensorOperation(bfloat16),
            device);
    }

    public static int TensorEmbeddingBackward(
        int device, nint indices, nint outputGradient, nint tableGradient,
        int length, int width)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.EmbeddingBackward(
                indices, outputGradient, tableGradient, length, width),
            CudaNativeOperation.TensorPrimitiveFloat32,
            device);
    }

    public static int TensorEmbeddingPositions(
        int device, nint tokens, nint positions, nint indices, nint output,
        int length, int sequence, int width, bool bfloat16)
    {
        EnsureCompatibleAbi();
        return Complete(
            bfloat16
                ? TensorPrimitiveNativeMethods.EmbeddingPositionsBFloat16(
                    tokens, positions, indices, output, length, sequence,
                    width)
                : TensorPrimitiveNativeMethods.EmbeddingPositionsFloat32(
                    tokens, positions, indices, output, length, sequence,
                    width),
            TensorOperation(bfloat16),
            device);
    }

    public static int TensorEmbeddingPositionsBackward(
        int device, nint indices, nint outputGradient, nint tokenGradient,
        nint positionGradient, int length, int sequence, int width)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.EmbeddingPositionsBackward(
                indices, outputGradient, tokenGradient, positionGradient,
                length, sequence, width),
            CudaNativeOperation.TensorPrimitiveFloat32,
            device);
    }

    public static int TensorDropout(
        int device, nint input, nint output, int length, uint seed,
        uint threshold, float scale, bool bfloat16)
    {
        EnsureCompatibleAbi();
        return Complete(
            bfloat16
                ? TensorPrimitiveNativeMethods.DropoutBFloat16(
                    input, output, length, seed, threshold, scale)
                : TensorPrimitiveNativeMethods.DropoutFloat32(
                    input, output, length, seed, threshold, scale),
            TensorOperation(bfloat16),
            device);
    }

    public static int TensorDropoutBackward(
        int device, nint outputGradient, nint inputGradient, int length,
        uint seed, uint threshold, float scale)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.DropoutBackward(
                outputGradient, inputGradient, length, seed, threshold,
                scale),
            CudaNativeOperation.TensorPrimitiveFloat32,
            device);
    }

    public static int TensorAddDropout(
        int device, nint residual, nint branch, nint output, int length,
        uint seed, uint threshold, float scale, bool bfloat16)
    {
        EnsureCompatibleAbi();
        return Complete(
            bfloat16
                ? TensorPrimitiveNativeMethods.AddDropoutBFloat16(
                    residual, branch, output, length, seed, threshold, scale)
                : TensorPrimitiveNativeMethods.AddDropoutFloat32(
                    residual, branch, output, length, seed, threshold, scale),
            TensorOperation(bfloat16),
            device);
    }

    public static int TensorAddDropoutBackward(
        int device, nint outputGradient, nint residualGradient,
        nint branchGradient, int length, bool sameParent, uint seed,
        uint threshold, float scale)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.AddDropoutBackward(
                outputGradient, residualGradient, branchGradient, length,
                sameParent ? 1 : 0, seed, threshold, scale),
            CudaNativeOperation.TensorPrimitiveFloat32,
            device);
    }

    public static int TensorLinearBias(
        int device, nint output, nint bias, int length, int width, bool relu,
        bool bfloat16)
    {
        EnsureCompatibleAbi();
        return Complete(
            bfloat16
                ? TensorPrimitiveNativeMethods.LinearBiasBFloat16(
                    output, bias, length, width, relu ? 1 : 0)
                : TensorPrimitiveNativeMethods.LinearBiasFloat32(
                    output, bias, length, width, relu ? 1 : 0),
            TensorOperation(bfloat16),
            device);
    }

    public static int TensorLinearMask(
        int device, nint output, nint outputGradient, int length, bool relu)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.LinearMaskFloat32(
                output, outputGradient, length, relu ? 1 : 0),
            CudaNativeOperation.TensorPrimitiveFloat32,
            device);
    }

    public static int TensorLinearEncodeBFloat16(
        int device, nint outputGradient, nint output, nint encoded,
        int length, bool relu)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.LinearEncodeBFloat16(
                outputGradient, output, encoded, length, relu ? 1 : 0),
            CudaNativeOperation.TensorPrimitiveBFloat16,
            device);
    }

    public static int TensorLinearEncodeBfp8Relu(
        int device,
        nint outputGradient,
        nint outputPayload,
        nint encoded,
        int length)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.LinearEncodeBfp8Relu(
                outputGradient, outputPayload, encoded, length),
            CudaNativeOperation.TensorPrimitiveBFloat16,
            device);
    }

    public static int TensorLinearMaskBFloat16Gradient(
        int device, nint outputGradient, nint output, nint masked, int length)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.LinearMaskBFloat16Gradient(
                outputGradient, output, masked, length),
            CudaNativeOperation.TensorPrimitiveBFloat16,
            device);
    }

    public static int TensorLinearBiasBackward(
        int device, nint outputGradient, nint biasGradient, int rows,
        int width, bool bfloat16)
    {
        EnsureCompatibleAbi();
        return Complete(
            bfloat16
                ? TensorPrimitiveNativeMethods.LinearBiasBackwardBFloat16(
                    outputGradient, biasGradient, rows, width)
                : TensorPrimitiveNativeMethods.LinearBiasBackwardFloat32(
                    outputGradient, biasGradient, rows, width),
            TensorOperation(bfloat16),
            device);
    }

    public static int TensorScale(
        int device, nint values, int length, float scale)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.Scale(values, length, scale),
            CudaNativeOperation.TensorPrimitiveFloat32,
            device);
    }

    public static int TensorAccumulate(
        int device, nint source, nint destination, int length,
        int sourceOffset, int destinationOffset)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.Accumulate(
                source, destination, length, sourceOffset, destinationOffset),
            CudaNativeOperation.TensorPrimitiveFloat32,
            device);
    }

    public static int TensorCopy(
        int device, nint source, nint destination, int length,
        int sourceOffset, int destinationOffset)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.Copy(
                source, destination, length, sourceOffset, destinationOffset),
            CudaNativeOperation.TensorPrimitiveFloat32,
            device);
    }

    public static int TensorEncodeBFloat16(
        int device, nint source, nint destination, int length)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.EncodeBFloat16(
                source, destination, length),
            CudaNativeOperation.TensorPrimitiveBFloat16,
            device);
    }

    public static int TensorDecodeBFloat16(
        int device, nint source, nint destination, int length)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.DecodeBFloat16(
                source, destination, length),
            CudaNativeOperation.TensorPrimitiveBFloat16,
            device);
    }

    public static int TensorSoftmaxProbabilities(
        int device, nint logits, nint maxima, nint inverseSums,
        nint probabilities, int length, int columns)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.SoftmaxProbabilities(
                logits, maxima, inverseSums, probabilities, length, columns),
            CudaNativeOperation.TensorPrimitiveFloat32,
            device);
    }

    public static int TensorCrossEntropyProbabilitiesBackward(
        int device, nint probabilities, nint labels, nint gradient,
        int length, int columns, int ignoreIndex, int validRows,
        float smoothing, float upstream)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.CrossEntropyProbabilitiesBackward(
                probabilities, labels, gradient, length, columns,
                ignoreIndex, validRows, smoothing, upstream),
            CudaNativeOperation.TensorPrimitiveFloat32,
            device);
    }

    public static int TensorSquaredSum(
        int device, nint values, int length, nint result)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.SquaredSum(values, length, result),
            CudaNativeOperation.TensorPrimitiveFloat32,
            device);
    }

    public static int TensorCrossEntropy(
        int device, nint logits, nint labels, nint maxima, nint inverseSums,
        nint rowLosses, nint loss, int rows, int columns, int ignoreIndex,
        int validRows, float smoothing, bool bfloat16)
    {
        EnsureCompatibleAbi();
        return Complete(
            bfloat16
                ? TensorPrimitiveNativeMethods.CrossEntropyBFloat16(
                    logits, labels, maxima, inverseSums, rowLosses, loss,
                    rows, columns, ignoreIndex, validRows, smoothing)
                : TensorPrimitiveNativeMethods.CrossEntropyFloat32(
                    logits, labels, maxima, inverseSums, rowLosses, loss,
                    rows, columns, ignoreIndex, validRows, smoothing),
            TensorOperation(bfloat16),
            device);
    }

    public static int TensorCrossEntropyBackward(
        int device, nint logits, nint maxima, nint inverseSums, nint labels,
        nint gradient, nint upstream, int length, int columns,
        int ignoreIndex, int validRows, float smoothing, bool bfloat16)
    {
        EnsureCompatibleAbi();
        return Complete(
            bfloat16
                ? TensorPrimitiveNativeMethods.CrossEntropyBackwardBFloat16(
                    logits, maxima, inverseSums, labels, gradient, upstream,
                    length, columns, ignoreIndex, validRows, smoothing)
                : TensorPrimitiveNativeMethods.CrossEntropyBackwardFloat32(
                    logits, maxima, inverseSums, labels, gradient, upstream,
                    length, columns, ignoreIndex, validRows, smoothing),
            TensorOperation(bfloat16),
            device);
    }

    public static int TensorCrossEntropyBackwardBFloat16Output(
        int device, nint logits, nint maxima, nint inverseSums, nint labels,
        nint gradient, nint upstream, int length, int columns,
        int ignoreIndex, int validRows, float smoothing)
    {
        EnsureCompatibleAbi();
        return Complete(
            TensorPrimitiveNativeMethods.CrossEntropyBackwardBFloat16Output(
                logits, maxima, inverseSums, labels, gradient, upstream,
                length, columns, ignoreIndex, validRows, smoothing),
            CudaNativeOperation.TensorPrimitiveBFloat16,
            device);
    }

    private static CudaNativeOperation TensorOperation(bool bfloat16)
        => bfloat16
            ? CudaNativeOperation.TensorPrimitiveBFloat16
            : CudaNativeOperation.TensorPrimitiveFloat32;

    private static class TensorPrimitiveNativeMethods
    {
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_add_float", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AddFloat32(nint left, nint right, nint output, int length);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_add_bf16", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AddBFloat16(nint left, nint right, nint output, int length);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_add_backward", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AddBackward(nint outputGradient, nint leftGradient, nint rightGradient, int length, int sameParent);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_embedding_float", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EmbeddingFloat32(nint table, nint indices, nint output, int length, int width);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_embedding_bf16", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EmbeddingBFloat16(nint table, nint indices, nint output, int length, int width);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_embedding_backward", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EmbeddingBackward(nint indices, nint outputGradient, nint tableGradient, int length, int width);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_embedding_positions_float", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EmbeddingPositionsFloat32(nint tokens, nint positions, nint indices, nint output, int length, int sequence, int width);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_embedding_positions_bf16", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EmbeddingPositionsBFloat16(nint tokens, nint positions, nint indices, nint output, int length, int sequence, int width);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_embedding_positions_backward", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EmbeddingPositionsBackward(nint indices, nint outputGradient, nint tokenGradient, nint positionGradient, int length, int sequence, int width);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_dropout_float", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DropoutFloat32(nint input, nint output, int length, uint seed, uint threshold, float scale);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_dropout_bf16", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DropoutBFloat16(nint input, nint output, int length, uint seed, uint threshold, float scale);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_dropout_backward", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DropoutBackward(nint outputGradient, nint inputGradient, int length, uint seed, uint threshold, float scale);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_add_dropout_float", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AddDropoutFloat32(nint residual, nint branch, nint output, int length, uint seed, uint threshold, float scale);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_add_dropout_bf16", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AddDropoutBFloat16(nint residual, nint branch, nint output, int length, uint seed, uint threshold, float scale);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_add_dropout_backward", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AddDropoutBackward(nint outputGradient, nint residualGradient, nint branchGradient, int length, int sameParent, uint seed, uint threshold, float scale);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_linear_bias_float", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int LinearBiasFloat32(nint output, nint bias, int length, int width, int relu);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_linear_bias_bf16", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int LinearBiasBFloat16(nint output, nint bias, int length, int width, int relu);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_linear_mask_float", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int LinearMaskFloat32(nint output, nint outputGradient, int length, int relu);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_linear_encode_bf16", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int LinearEncodeBFloat16(nint outputGradient, nint output, nint encoded, int length, int relu);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_linear_encode_bfp8_relu", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int LinearEncodeBfp8Relu(nint outputGradient, nint outputPayload, nint encoded, int length);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_linear_mask_bf16_gradient", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int LinearMaskBFloat16Gradient(nint outputGradient, nint output, nint masked, int length);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_linear_bias_backward_float", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int LinearBiasBackwardFloat32(nint outputGradient, nint biasGradient, int rows, int width);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_linear_bias_backward_bf16", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int LinearBiasBackwardBFloat16(nint outputGradient, nint biasGradient, int rows, int width);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_scale", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Scale(nint values, int length, float scale);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_accumulate", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Accumulate(nint source, nint destination, int length, int sourceOffset, int destinationOffset);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_copy", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Copy(nint source, nint destination, int length, int sourceOffset, int destinationOffset);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_encode_bf16", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EncodeBFloat16(nint source, nint destination, int length);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_decode_bf16", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DecodeBFloat16(nint source, nint destination, int length);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_softmax_probabilities", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int SoftmaxProbabilities(nint logits, nint maxima, nint inverseSums, nint probabilities, int length, int columns);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_cross_entropy_probabilities_backward", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CrossEntropyProbabilitiesBackward(nint probabilities, nint labels, nint gradient, int length, int columns, int ignoreIndex, int validRows, float smoothing, float upstream);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_squared_sum", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int SquaredSum(nint values, int length, nint result);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_cross_entropy_float", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CrossEntropyFloat32(nint logits, nint labels, nint maxima, nint inverseSums, nint rowLosses, nint loss, int rows, int columns, int ignoreIndex, int validRows, float smoothing);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_cross_entropy_bf16", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CrossEntropyBFloat16(nint logits, nint labels, nint maxima, nint inverseSums, nint rowLosses, nint loss, int rows, int columns, int ignoreIndex, int validRows, float smoothing);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_cross_entropy_backward_float", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CrossEntropyBackwardFloat32(nint logits, nint maxima, nint inverseSums, nint labels, nint gradient, nint upstream, int length, int columns, int ignoreIndex, int validRows, float smoothing);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_cross_entropy_backward_bf16", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CrossEntropyBackwardBFloat16(nint logits, nint maxima, nint inverseSums, nint labels, nint gradient, nint upstream, int length, int columns, int ignoreIndex, int validRows, float smoothing);
        [DllImport(LibraryName, EntryPoint = "nntrain_tensor_cross_entropy_backward_bf16_output", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CrossEntropyBackwardBFloat16Output(nint logits, nint maxima, nint inverseSums, nint labels, nint gradient, nint upstream, int length, int columns, int ignoreIndex, int validRows, float smoothing);
    }
}
