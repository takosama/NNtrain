namespace NNtrain;

public partial class Tensor
{
    internal static Tensor ArcQwenQuantizedEmbedding(
        NNtrain.Arc.ArcExecutionLane.ArcBuffer encodedWeight,
        int[] tokenIds,
        int batch,
        int sequence,
        int width,
        uint ggmlType)
    {
        if (ExecutionDevice != TensorDevice.Arc)
            throw new NotSupportedException("Qwen quantized embedding requires Intel Arc.");
        var lane = ArcLane;
        using var ids = lane.UploadRaw(tokenIds);
        using var output = lane.Allocate(checked(tokenIds.Length * width));
        string kernel = ggmlType switch
        {
            Qwen2Gguf.Q4KType => "qwen_embedding_q4_k",
            Qwen2Gguf.Q6KType => "qwen_embedding_q6_k",
            _ => throw new NotSupportedException($"GGML type {ggmlType} is not supported by quantized embedding.")
        };
        lane.Run(kernel, checked((long)tokenIds.Length * width), 0,
            encodedWeight, ids, output, width);
        return ArcDeviceResult(output, [batch, sequence, width], [], TensorDType.Float32);
    }

    internal Tensor ArcQwenQuantizedLinear(
        NNtrain.Arc.ArcExecutionLane.ArcBuffer encodedWeight,
        Tensor bias,
        uint ggmlType,
        int outputWidth,
        int inputWidth)
    {
        if (ExecutionDevice != TensorDevice.Arc)
            throw new NotSupportedException("Qwen quantized linear requires Intel Arc.");
        if (_shape[^1] != inputWidth)
            throw new ArgumentException("Quantized linear input width mismatch.");
        int rows = Numel / inputWidth;
        int[] outputShape = (int[])_shape.Clone();
        outputShape[^1] = outputWidth;

        var lane = ArcLane;
        using var input = ArcUploadValues(true);
        using var biasValues = bias.ArcUploadValues();
        using var output = lane.Allocate(checked(rows * outputWidth));
        string kernel = ggmlType switch
        {
            Qwen2Gguf.Q4KType => "qwen_linear_q4_k",
            Qwen2Gguf.Q6KType => "qwen_linear_q6_k",
            _ => throw new NotSupportedException($"GGML type {ggmlType} is not supported by quantized linear.")
        };
        lane.Run(kernel, checked((long)rows * outputWidth), 0,
            input, encodedWeight, biasValues, output,
            rows, inputWidth, outputWidth);
        // Frozen inference weight is intentionally not an autograd parent.
        return ArcDeviceResult(output, outputShape, [this, bias]);
    }
}
