namespace NNtrain;

public partial class Tensor
{
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
