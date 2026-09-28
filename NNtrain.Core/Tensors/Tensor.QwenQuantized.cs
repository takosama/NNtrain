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
        bool subgroupDecode = lane.Options.QwenQuantizedLinearFast
            && lane.Options.QwenQuantizedLinearSubgroup
            && (ggmlType == Qwen2Gguf.Q4KType || ggmlType == Qwen2Gguf.Q6KType)
            && (rows == 1 || lane.Options.QwenQuantizedSubgroupPrefill)
            && lane.Options.XmxMatrices && lane.Device.SupportsXmx
            && lane.Device.MinimumSubgroupSize == 16
            && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups");
        if (subgroupDecode)
        {
            string kernel = ggmlType == Qwen2Gguf.Q4KType
                ? "qwen_linear_q4_k_sg16_r1" : "qwen_linear_q6_k_sg16_r1";
            lane.Run(kernel,
                checked((((long)rows * outputWidth + 1) / 2) * 32), 32,
                input, encodedWeight, biasValues, output,
                rows, inputWidth, outputWidth);
        }
        else if (lane.Options.QwenQuantizedLinearFast)
        {
            int rowTile = rows == 1 ? 1 : 4;
            string kernel = ggmlType switch
            {
                Qwen2Gguf.Q4KType => rowTile == 1
                    ? "qwen_linear_q4_k_fast_r1" : "qwen_linear_q4_k_fast_r4",
                Qwen2Gguf.Q6KType => rowTile == 1
                    ? "qwen_linear_q6_k_fast_r1" : "qwen_linear_q6_k_fast_r4",
                _ => throw new NotSupportedException(
                    $"GGML type {ggmlType} is not supported by quantized linear.")
            };
            lane.Run2D(kernel, checked((long)outputWidth * 64),
                checked((rows + rowTile - 1) / rowTile), 64, 1,
                input, encodedWeight, biasValues, output,
                rows, inputWidth, outputWidth);
        }
        else
        {
            string kernel = ggmlType switch
            {
                Qwen2Gguf.Q4KType => "qwen_linear_q4_k",
                Qwen2Gguf.Q6KType => "qwen_linear_q6_k",
                _ => throw new NotSupportedException(
                    $"GGML type {ggmlType} is not supported by quantized linear.")
            };
            lane.Run(kernel, checked((long)rows * outputWidth), 0,
                input, encodedWeight, biasValues, output,
                rows, inputWidth, outputWidth);
        }
        // Frozen inference weight is intentionally not an autograd parent.
        return ArcDeviceResult(output, outputShape, [this, bias]);
    }
}
