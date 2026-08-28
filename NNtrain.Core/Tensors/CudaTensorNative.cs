using NNtrain.Cuda.Interop;

namespace NNtrain;

/// <summary>
/// Native CUDA launch surface for tensor operations that do not map to GEMM.
/// Matrix operations stay on cuBLAS/cuBLASLt/WMMA Tensor Core paths.
/// </summary>
internal static class CudaTensorNative
{
    internal static void Add(int device, nint left, nint right, nint output,
        int length, bool bfloat16)
    {
        Select(device);
        Check(CudaNativeGateway.TensorAdd(
            device, left, right, output, length, bfloat16), "tensor add");
    }

    internal static void AddBackward(int device, nint outputGradient,
        nint leftGradient, nint rightGradient, int length, bool sameParent)
    {
        Select(device);
        Check(CudaNativeGateway.TensorAddBackward(
            device, outputGradient, leftGradient, rightGradient, length,
            sameParent), "tensor add backward");
    }

    internal static void Embedding(int device, nint table, nint indices,
        nint output, int length, int width, bool bfloat16)
    {
        Select(device);
        Check(CudaNativeGateway.TensorEmbedding(
            device, table, indices, output, length, width, bfloat16),
            "embedding forward");
    }

    internal static void EmbeddingBackward(int device, nint indices,
        nint outputGradient, nint tableGradient, int length, int width)
    {
        Select(device);
        Check(CudaNativeGateway.TensorEmbeddingBackward(
            device, indices, outputGradient, tableGradient, length, width),
            "embedding backward");
    }

    internal static void EmbeddingBackwardReduced(
        int device,
        nint indices,
        nint outputGradient,
        nint tableGradient,
        nint workspace,
        int workspaceInts,
        int length,
        int width)
    {
        Select(device);
        Check(
            CudaNativeGateway.EmbeddingBackwardReduced(
                device,
                indices,
                outputGradient,
                tableGradient,
                workspace,
                workspaceInts,
                length,
                width),
            "owner-reduced embedding backward");
    }

    internal static void EmbeddingPositions(int device, nint tokens,
        nint positions, nint indices, nint output, int length, int sequence,
        int width, bool bfloat16)
    {
        Select(device);
        Check(CudaNativeGateway.TensorEmbeddingPositions(
            device, tokens, positions, indices, output, length, sequence,
            width, bfloat16), "embedding positions forward");
    }

    internal static void EmbeddingPositionsBackward(int device, nint indices,
        nint outputGradient, nint tokenGradient, nint positionGradient,
        int length, int sequence, int width)
    {
        Select(device);
        Check(CudaNativeGateway.TensorEmbeddingPositionsBackward(
            device, indices, outputGradient, tokenGradient, positionGradient,
            length, sequence, width),
            "embedding positions backward");
    }

    internal static void EmbeddingPositionsBackwardReduced(
        int device,
        nint indices,
        nint outputGradient,
        nint tokenGradient,
        nint positionGradient,
        nint workspace,
        int workspaceInts,
        int length,
        int sequence,
        int width)
    {
        Select(device);
        Check(
            CudaNativeGateway.EmbeddingPositionsBackwardReduced(
                device,
                indices,
                outputGradient,
                tokenGradient,
                positionGradient,
                workspace,
                workspaceInts,
                length,
                sequence,
                width),
            "owner-reduced embedding positions backward");
    }

    internal static void Dropout(int device, nint input, nint output,
        int length, uint seed, uint threshold, float scale, bool bfloat16)
    {
        Select(device);
        Check(CudaNativeGateway.TensorDropout(
            device, input, output, length, seed, threshold, scale, bfloat16),
            "dropout forward");
    }

    internal static void DropoutBackward(int device, nint outputGradient,
        nint inputGradient, int length, uint seed, uint threshold,
        float scale)
    {
        Select(device);
        Check(CudaNativeGateway.TensorDropoutBackward(
            device, outputGradient, inputGradient, length, seed, threshold,
            scale), "dropout backward");
    }

    internal static void AddDropout(int device, nint residual, nint branch,
        nint output, int length, uint seed, uint threshold, float scale,
        bool bfloat16)
    {
        Select(device);
        Check(CudaNativeGateway.TensorAddDropout(
            device, residual, branch, output, length, seed, threshold, scale,
            bfloat16), "residual dropout forward");
    }

    internal static void AddDropoutBackward(int device, nint outputGradient,
        nint residualGradient, nint branchGradient, int length,
        bool sameParent, uint seed, uint threshold, float scale)
    {
        Select(device);
        Check(CudaNativeGateway.TensorAddDropoutBackward(
            device, outputGradient, residualGradient, branchGradient, length,
            sameParent, seed, threshold, scale), "residual dropout backward");
    }

    internal static void LinearBias(int device, nint output, nint bias,
        int length, int width, bool relu, bool bfloat16)
    {
        Select(device);
        Check(CudaNativeGateway.TensorLinearBias(
            device, output, bias, length, width, relu, bfloat16),
            "linear bias epilogue");
    }

    internal static void LinearMask(int device, nint output,
        nint outputGradient, int length, bool relu)
    {
        Select(device);
        Check(CudaNativeGateway.TensorLinearMask(
            device, output, outputGradient, length, relu),
            "linear activation backward");
    }

    internal static void LinearEncodeBFloat16(int device,
        nint outputGradient, nint output, nint encoded, int length, bool relu)
    {
        Select(device);
        Check(CudaNativeGateway.TensorLinearEncodeBFloat16(
            device, outputGradient, output, encoded, length, relu),
            "linear BF16 gradient encode");
    }

    internal static void LinearEncodeBfp8Relu(
        int device,
        nint outputGradient,
        nint outputPayload,
        nint encoded,
        int length)
    {
        Select(device);
        Check(CudaNativeGateway.TensorLinearEncodeBfp8Relu(
            device, outputGradient, outputPayload, encoded, length),
            "linear BFP8 ReLU gradient encode");
    }

    internal static void LinearMaskBFloat16Gradient(
        int device,
        nint outputGradient,
        nint output,
        nint masked,
        int length)
    {
        Select(device);
        Check(CudaNativeGateway.TensorLinearMaskBFloat16Gradient(
            device,
            outputGradient,
            output,
            masked,
            length),
            "linear BF16 gradient mask");
    }

    internal static void LinearBiasBackward(int device, nint outputGradient,
        nint biasGradient, int rows, int width, bool bfloat16)
    {
        Select(device);
        Check(CudaNativeGateway.TensorLinearBiasBackward(
            device, outputGradient, biasGradient, rows, width, bfloat16),
            "linear bias backward");
    }

    internal static void Scale(int device, nint values, int length,
        float scale)
    {
        Select(device);
        Check(CudaNativeGateway.TensorScale(device, values, length, scale),
            "tensor scale");
    }

    internal static void AccumulateScalar(
        int device,
        nint destination,
        float value,
        bool accumulate)
    {
        Select(device);
        Check(
            CudaNativeGateway.TensorAccumulateScalar(
                device,
                destination,
                value,
                accumulate),
            "CUDA output-gradient seed");
    }

    internal static void Accumulate(int device, nint source,
        nint destination, int length, int sourceOffset = 0,
        int destinationOffset = 0)
    {
        Select(device);
        Check(CudaNativeGateway.TensorAccumulate(
            device, source, destination, length, sourceOffset,
            destinationOffset), "tensor accumulate");
    }

    internal static void Copy(int device, nint source, nint destination,
        int length, int sourceOffset = 0, int destinationOffset = 0)
    {
        Select(device);
        Check(CudaNativeGateway.TensorCopy(
            device, source, destination, length, sourceOffset,
            destinationOffset), "tensor copy");
    }

    internal static void EncodeBFloat16(int device, nint source,
        nint destination, int length)
    {
        Select(device);
        Check(CudaNativeGateway.TensorEncodeBFloat16(
            device, source, destination, length),
            "encode BF16");
    }

    internal static void DecodeBFloat16(int device, nint source,
        nint destination, int length)
    {
        Select(device);
        Check(CudaNativeGateway.TensorDecodeBFloat16(
            device, source, destination, length),
            "decode BF16");
    }

    internal static void SoftmaxProbabilities(int device, nint logits,
        nint maxima, nint inverseSums, nint probabilities, int length,
        int columns)
    {
        Select(device);
        Check(CudaNativeGateway.TensorSoftmaxProbabilities(
            device, logits, maxima, inverseSums, probabilities, length,
            columns), "softmax probabilities");
    }

    internal static void CrossEntropyProbabilitiesBackward(int device,
        nint probabilities, nint labels, nint gradient, int length,
        int columns, int ignoreIndex, int validRows, float smoothing,
        float upstream)
    {
        Select(device);
        Check(CudaNativeGateway.TensorCrossEntropyProbabilitiesBackward(
            device, probabilities, labels, gradient, length, columns,
            ignoreIndex, validRows, smoothing, upstream),
            "cross entropy probabilities backward");
    }

    internal static void SquaredSum(int device, nint values, int length,
        nint result)
    {
        Select(device);
        Check(CudaNativeGateway.TensorSquaredSum(
            device, values, length, result),
            "gradient squared sum");
    }

    internal static void CrossEntropy(int device, nint logits, nint labels,
        nint maxima, nint inverseSums, nint rowLosses, nint loss,
        int rows, int columns,
        int ignoreIndex, int validRows, float smoothing, bool bfloat16)
    {
        Select(device);
        Check(CudaNativeGateway.TensorCrossEntropy(
            device, logits, labels, maxima, inverseSums, rowLosses, loss,
            rows, columns, ignoreIndex, validRows, smoothing, bfloat16),
            "cross entropy forward");
    }

    internal static void CrossEntropyBackwardBFloat16Output(
        int device, nint logits, nint maxima, nint inverseSums, nint labels,
        nint gradient, nint upstream, int length, int columns,
        int ignoreIndex, int validRows, float smoothing)
    {
        Select(device);
        Check(CudaNativeGateway.TensorCrossEntropyBackwardBFloat16Output(
            device, logits, maxima, inverseSums, labels, gradient, upstream,
            length, columns, ignoreIndex, validRows, smoothing),
            "cross entropy BF16 gradient backward");
    }

    internal static void CrossEntropyBackward(int device, nint logits,
        nint maxima, nint inverseSums, nint labels, nint gradient,
        nint upstream, int length, int columns, int ignoreIndex,
        int validRows, float smoothing, bool bfloat16)
    {
        Select(device);
        Check(CudaNativeGateway.TensorCrossEntropyBackward(
            device, logits, maxima, inverseSums, labels, gradient, upstream,
            length, columns, ignoreIndex, validRows, smoothing, bfloat16),
            "cross entropy backward");
    }

    private static void Select(int device)
        => NativeCudaRuntime.BindDeviceAndComputeStream(device);

    private static void Check(int status, string operation)
        => NativeCudaRuntime.Check(status, operation);

}
