using System.Runtime.InteropServices;

namespace NNtrain;

/// <summary>
/// Native CUDA launch surface for tensor operations that do not map to GEMM.
/// Matrix operations stay on cuBLAS/cuBLASLt/WMMA Tensor Core paths.
/// </summary>
internal static class CudaTensorNative
{
    private const string Library = "NNtrain.CudaKernels.dll";
    [ThreadStatic]
    private static int _streamDevice;
    [ThreadStatic]
    private static bool _streamSelected;

    internal static void Add(int device, nint left, nint right, nint output,
        int length, bool bfloat16)
    {
        Select(device);
        Check(bfloat16
            ? AddBFloat16(left, right, output, length)
            : AddFloat(left, right, output, length), "tensor add");
    }

    internal static void AddBackward(int device, nint outputGradient,
        nint leftGradient, nint rightGradient, int length, bool sameParent)
    {
        Select(device);
        Check(AddBackwardNative(outputGradient, leftGradient, rightGradient,
            length, sameParent ? 1 : 0), "tensor add backward");
    }

    internal static void Embedding(int device, nint table, nint indices,
        nint output, int length, int width, bool bfloat16)
    {
        Select(device);
        Check(bfloat16
            ? EmbeddingBFloat16(table, indices, output, length, width)
            : EmbeddingFloat(table, indices, output, length, width),
            "embedding forward");
    }

    internal static void EmbeddingBackward(int device, nint indices,
        nint outputGradient, nint tableGradient, int length, int width)
    {
        Select(device);
        Check(EmbeddingBackwardNative(indices, outputGradient, tableGradient,
            length, width), "embedding backward");
    }

    internal static void EmbeddingPositions(int device, nint tokens,
        nint positions, nint indices, nint output, int length, int sequence,
        int width, bool bfloat16)
    {
        Select(device);
        Check(bfloat16
            ? EmbeddingPositionsBFloat16(tokens, positions, indices, output,
                length, sequence, width)
            : EmbeddingPositionsFloat(tokens, positions, indices, output,
                length, sequence, width), "embedding positions forward");
    }

    internal static void EmbeddingPositionsBackward(int device, nint indices,
        nint outputGradient, nint tokenGradient, nint positionGradient,
        int length, int sequence, int width)
    {
        Select(device);
        Check(EmbeddingPositionsBackwardNative(indices, outputGradient,
            tokenGradient, positionGradient, length, sequence, width),
            "embedding positions backward");
    }

    internal static void Dropout(int device, nint input, nint output,
        int length, uint seed, uint threshold, float scale, bool bfloat16)
    {
        Select(device);
        Check(bfloat16
            ? DropoutBFloat16(input, output, length, seed, threshold, scale)
            : DropoutFloat(input, output, length, seed, threshold, scale),
            "dropout forward");
    }

    internal static void DropoutBackward(int device, nint outputGradient,
        nint inputGradient, int length, uint seed, uint threshold,
        float scale)
    {
        Select(device);
        Check(DropoutBackwardNative(outputGradient, inputGradient, length,
            seed, threshold, scale), "dropout backward");
    }

    internal static void AddDropout(int device, nint residual, nint branch,
        nint output, int length, uint seed, uint threshold, float scale,
        bool bfloat16)
    {
        Select(device);
        Check(bfloat16
            ? AddDropoutBFloat16(residual, branch, output, length, seed,
                threshold, scale)
            : AddDropoutFloat(residual, branch, output, length, seed,
                threshold, scale), "residual dropout forward");
    }

    internal static void AddDropoutBackward(int device, nint outputGradient,
        nint residualGradient, nint branchGradient, int length,
        bool sameParent, uint seed, uint threshold, float scale)
    {
        Select(device);
        Check(AddDropoutBackwardNative(outputGradient, residualGradient,
            branchGradient, length, sameParent ? 1 : 0, seed, threshold,
            scale), "residual dropout backward");
    }

    internal static void LinearBias(int device, nint output, nint bias,
        int length, int width, bool relu, bool bfloat16)
    {
        Select(device);
        Check(bfloat16
            ? LinearBiasBFloat16(output, bias, length, width, relu ? 1 : 0)
            : LinearBiasFloat(output, bias, length, width, relu ? 1 : 0),
            "linear bias epilogue");
    }

    internal static void LinearMask(int device, nint output,
        nint outputGradient, int length, bool relu)
    {
        Select(device);
        Check(LinearMaskFloat(output, outputGradient, length, relu ? 1 : 0),
            "linear activation backward");
    }

    internal static void LinearEncodeBFloat16(int device,
        nint outputGradient, nint output, nint encoded, int length, bool relu)
    {
        Select(device);
        Check(LinearEncodeBFloat16Native(outputGradient, output, encoded,
            length, relu ? 1 : 0), "linear BF16 gradient encode");
    }

    internal static void LinearBiasBackward(int device, nint outputGradient,
        nint biasGradient, int rows, int width, bool bfloat16)
    {
        Select(device);
        Check(bfloat16
            ? LinearBiasBackwardBFloat16(outputGradient, biasGradient, rows,
                width)
            : LinearBiasBackwardFloat(outputGradient, biasGradient, rows,
                width), "linear bias backward");
    }

    internal static void Scale(int device, nint values, int length,
        float scale)
    {
        Select(device);
        Check(ScaleNative(values, length, scale), "tensor scale");
    }

    internal static void Accumulate(int device, nint source,
        nint destination, int length, int sourceOffset = 0,
        int destinationOffset = 0)
    {
        Select(device);
        Check(AccumulateNative(source, destination, length, sourceOffset,
            destinationOffset), "tensor accumulate");
    }

    internal static void Copy(int device, nint source, nint destination,
        int length, int sourceOffset = 0, int destinationOffset = 0)
    {
        Select(device);
        Check(CopyNative(source, destination, length, sourceOffset,
            destinationOffset), "tensor copy");
    }

    internal static void EncodeBFloat16(int device, nint source,
        nint destination, int length)
    {
        Select(device);
        Check(EncodeBFloat16Native(source, destination, length),
            "encode BF16");
    }

    internal static void DecodeBFloat16(int device, nint source,
        nint destination, int length)
    {
        Select(device);
        Check(DecodeBFloat16Native(source, destination, length),
            "decode BF16");
    }

    internal static void SoftmaxProbabilities(int device, nint logits,
        nint maxima, nint inverseSums, nint probabilities, int length,
        int columns)
    {
        Select(device);
        Check(SoftmaxProbabilitiesNative(logits, maxima, inverseSums,
            probabilities, length, columns), "softmax probabilities");
    }

    internal static void CrossEntropyProbabilitiesBackward(int device,
        nint probabilities, nint labels, nint gradient, int length,
        int columns, int ignoreIndex, int validRows, float smoothing,
        float upstream)
    {
        Select(device);
        Check(CrossEntropyProbabilitiesBackwardNative(probabilities, labels,
            gradient, length, columns, ignoreIndex, validRows, smoothing,
            upstream), "cross entropy probabilities backward");
    }

    internal static void SquaredSum(int device, nint values, int length,
        nint result)
    {
        Select(device);
        Check(SquaredSumNative(values, length, result),
            "gradient squared sum");
    }

    internal static void CrossEntropy(int device, nint logits, nint labels,
        nint maxima, nint inverseSums, nint rowLosses, nint loss,
        int rows, int columns,
        int ignoreIndex, int validRows, float smoothing, bool bfloat16)
    {
        Select(device);
        Check(bfloat16
            ? CrossEntropyBFloat16(logits, labels, maxima, inverseSums,
                rowLosses, loss, rows, columns, ignoreIndex, validRows,
                smoothing)
            : CrossEntropyFloat(logits, labels, maxima, inverseSums,
                rowLosses, loss, rows, columns, ignoreIndex, validRows,
                smoothing),
            "cross entropy forward");
    }

    internal static void CrossEntropyBackwardBFloat16Output(
        int device, nint logits, nint maxima, nint inverseSums, nint labels,
        nint gradient, nint upstream, int length, int columns,
        int ignoreIndex, int validRows, float smoothing)
    {
        Select(device);
        Check(CrossEntropyBackwardBFloat16OutputNative(
            logits, maxima, inverseSums, labels, gradient, upstream, length,
            columns, ignoreIndex, validRows, smoothing),
            "cross entropy BF16 gradient backward");
    }

    internal static void CrossEntropyBackward(int device, nint logits,
        nint maxima, nint inverseSums, nint labels, nint gradient,
        nint upstream, int length, int columns, int ignoreIndex,
        int validRows, float smoothing, bool bfloat16)
    {
        Select(device);
        Check(bfloat16
            ? CrossEntropyBackwardBFloat16(logits, maxima, inverseSums,
                labels, gradient, upstream, length, columns, ignoreIndex,
                validRows, smoothing)
            : CrossEntropyBackwardFloat(logits, maxima, inverseSums, labels,
                gradient, upstream, length, columns, ignoreIndex, validRows,
                smoothing), "cross entropy backward");
    }

    private static void Select(int device)
    {
        var accelerator = ForgetMemoryV2Cuda.GetAccelerator(device);
        if (_streamSelected && _streamDevice == device)
            return;
        NativeCudaRuntime.Check(
            NativeCudaRuntime.UseExternalStreamNative(accelerator.DefaultStream),
            "select CUDA stream");
        _streamDevice = device;
        _streamSelected = true;
    }

    private static void Check(int status, string operation)
        => NativeCudaRuntime.Check(status, operation);

    [DllImport(Library, EntryPoint = "nntrain_tensor_add_float", CallingConvention = CallingConvention.Cdecl)]
    private static extern int AddFloat(nint left, nint right, nint output, int length);
    [DllImport(Library, EntryPoint = "nntrain_tensor_add_bf16", CallingConvention = CallingConvention.Cdecl)]
    private static extern int AddBFloat16(nint left, nint right, nint output, int length);
    [DllImport(Library, EntryPoint = "nntrain_tensor_add_backward", CallingConvention = CallingConvention.Cdecl)]
    private static extern int AddBackwardNative(nint outputGradient, nint leftGradient, nint rightGradient, int length, int sameParent);
    [DllImport(Library, EntryPoint = "nntrain_tensor_embedding_float", CallingConvention = CallingConvention.Cdecl)]
    private static extern int EmbeddingFloat(nint table, nint indices, nint output, int length, int width);
    [DllImport(Library, EntryPoint = "nntrain_tensor_embedding_bf16", CallingConvention = CallingConvention.Cdecl)]
    private static extern int EmbeddingBFloat16(nint table, nint indices, nint output, int length, int width);
    [DllImport(Library, EntryPoint = "nntrain_tensor_embedding_backward", CallingConvention = CallingConvention.Cdecl)]
    private static extern int EmbeddingBackwardNative(nint indices, nint outputGradient, nint tableGradient, int length, int width);
    [DllImport(Library, EntryPoint = "nntrain_tensor_embedding_positions_float", CallingConvention = CallingConvention.Cdecl)]
    private static extern int EmbeddingPositionsFloat(nint tokens, nint positions, nint indices, nint output, int length, int sequence, int width);
    [DllImport(Library, EntryPoint = "nntrain_tensor_embedding_positions_bf16", CallingConvention = CallingConvention.Cdecl)]
    private static extern int EmbeddingPositionsBFloat16(nint tokens, nint positions, nint indices, nint output, int length, int sequence, int width);
    [DllImport(Library, EntryPoint = "nntrain_tensor_embedding_positions_backward", CallingConvention = CallingConvention.Cdecl)]
    private static extern int EmbeddingPositionsBackwardNative(nint indices, nint outputGradient, nint tokenGradient, nint positionGradient, int length, int sequence, int width);
    [DllImport(Library, EntryPoint = "nntrain_tensor_dropout_float", CallingConvention = CallingConvention.Cdecl)]
    private static extern int DropoutFloat(nint input, nint output, int length, uint seed, uint threshold, float scale);
    [DllImport(Library, EntryPoint = "nntrain_tensor_dropout_bf16", CallingConvention = CallingConvention.Cdecl)]
    private static extern int DropoutBFloat16(nint input, nint output, int length, uint seed, uint threshold, float scale);
    [DllImport(Library, EntryPoint = "nntrain_tensor_dropout_backward", CallingConvention = CallingConvention.Cdecl)]
    private static extern int DropoutBackwardNative(nint outputGradient, nint inputGradient, int length, uint seed, uint threshold, float scale);
    [DllImport(Library, EntryPoint = "nntrain_tensor_add_dropout_float", CallingConvention = CallingConvention.Cdecl)]
    private static extern int AddDropoutFloat(nint residual, nint branch, nint output, int length, uint seed, uint threshold, float scale);
    [DllImport(Library, EntryPoint = "nntrain_tensor_add_dropout_bf16", CallingConvention = CallingConvention.Cdecl)]
    private static extern int AddDropoutBFloat16(nint residual, nint branch, nint output, int length, uint seed, uint threshold, float scale);
    [DllImport(Library, EntryPoint = "nntrain_tensor_add_dropout_backward", CallingConvention = CallingConvention.Cdecl)]
    private static extern int AddDropoutBackwardNative(nint outputGradient, nint residualGradient, nint branchGradient, int length, int sameParent, uint seed, uint threshold, float scale);
    [DllImport(Library, EntryPoint = "nntrain_tensor_linear_bias_float", CallingConvention = CallingConvention.Cdecl)]
    private static extern int LinearBiasFloat(nint output, nint bias, int length, int width, int relu);
    [DllImport(Library, EntryPoint = "nntrain_tensor_linear_bias_bf16", CallingConvention = CallingConvention.Cdecl)]
    private static extern int LinearBiasBFloat16(nint output, nint bias, int length, int width, int relu);
    [DllImport(Library, EntryPoint = "nntrain_tensor_linear_mask_float", CallingConvention = CallingConvention.Cdecl)]
    private static extern int LinearMaskFloat(nint output, nint outputGradient, int length, int relu);
    [DllImport(Library, EntryPoint = "nntrain_tensor_linear_encode_bf16", CallingConvention = CallingConvention.Cdecl)]
    private static extern int LinearEncodeBFloat16Native(nint outputGradient, nint output, nint encoded, int length, int relu);
    [DllImport(Library, EntryPoint = "nntrain_tensor_linear_bias_backward_float", CallingConvention = CallingConvention.Cdecl)]
    private static extern int LinearBiasBackwardFloat(nint outputGradient, nint biasGradient, int rows, int width);
    [DllImport(Library, EntryPoint = "nntrain_tensor_linear_bias_backward_bf16", CallingConvention = CallingConvention.Cdecl)]
    private static extern int LinearBiasBackwardBFloat16(nint outputGradient, nint biasGradient, int rows, int width);
    [DllImport(Library, EntryPoint = "nntrain_tensor_scale", CallingConvention = CallingConvention.Cdecl)]
    private static extern int ScaleNative(nint values, int length, float scale);
    [DllImport(Library, EntryPoint = "nntrain_tensor_accumulate", CallingConvention = CallingConvention.Cdecl)]
    private static extern int AccumulateNative(nint source, nint destination, int length, int sourceOffset, int destinationOffset);
    [DllImport(Library, EntryPoint = "nntrain_tensor_copy", CallingConvention = CallingConvention.Cdecl)]
    private static extern int CopyNative(nint source, nint destination, int length, int sourceOffset, int destinationOffset);
    [DllImport(Library, EntryPoint = "nntrain_tensor_encode_bf16", CallingConvention = CallingConvention.Cdecl)]
    private static extern int EncodeBFloat16Native(nint source, nint destination, int length);
    [DllImport(Library, EntryPoint = "nntrain_tensor_decode_bf16", CallingConvention = CallingConvention.Cdecl)]
    private static extern int DecodeBFloat16Native(nint source, nint destination, int length);
    [DllImport(Library, EntryPoint = "nntrain_tensor_softmax_probabilities", CallingConvention = CallingConvention.Cdecl)]
    private static extern int SoftmaxProbabilitiesNative(nint logits, nint maxima, nint inverseSums, nint probabilities, int length, int columns);
    [DllImport(Library, EntryPoint = "nntrain_tensor_cross_entropy_probabilities_backward", CallingConvention = CallingConvention.Cdecl)]
    private static extern int CrossEntropyProbabilitiesBackwardNative(nint probabilities, nint labels, nint gradient, int length, int columns, int ignoreIndex, int validRows, float smoothing, float upstream);
    [DllImport(Library, EntryPoint = "nntrain_tensor_squared_sum", CallingConvention = CallingConvention.Cdecl)]
    private static extern int SquaredSumNative(nint values, int length, nint result);
    [DllImport(Library, EntryPoint = "nntrain_tensor_cross_entropy_float", CallingConvention = CallingConvention.Cdecl)]
    private static extern int CrossEntropyFloat(nint logits, nint labels, nint maxima, nint inverseSums, nint rowLosses, nint loss, int rows, int columns, int ignoreIndex, int validRows, float smoothing);
    [DllImport(Library, EntryPoint = "nntrain_tensor_cross_entropy_bf16", CallingConvention = CallingConvention.Cdecl)]
    private static extern int CrossEntropyBFloat16(nint logits, nint labels, nint maxima, nint inverseSums, nint rowLosses, nint loss, int rows, int columns, int ignoreIndex, int validRows, float smoothing);
    [DllImport(Library, EntryPoint = "nntrain_tensor_cross_entropy_backward_float", CallingConvention = CallingConvention.Cdecl)]
    private static extern int CrossEntropyBackwardFloat(nint logits, nint maxima, nint inverseSums, nint labels, nint gradient, nint upstream, int length, int columns, int ignoreIndex, int validRows, float smoothing);
    [DllImport(Library, EntryPoint = "nntrain_tensor_cross_entropy_backward_bf16", CallingConvention = CallingConvention.Cdecl)]
    private static extern int CrossEntropyBackwardBFloat16(nint logits, nint maxima, nint inverseSums, nint labels, nint gradient, nint upstream, int length, int columns, int ignoreIndex, int validRows, float smoothing);
    [DllImport(Library, EntryPoint = "nntrain_tensor_cross_entropy_backward_bf16_output", CallingConvention = CallingConvention.Cdecl)]
    private static extern int CrossEntropyBackwardBFloat16OutputNative(nint logits, nint maxima, nint inverseSums, nint labels, nint gradient, nint upstream, int length, int columns, int ignoreIndex, int validRows, float smoothing);
}
