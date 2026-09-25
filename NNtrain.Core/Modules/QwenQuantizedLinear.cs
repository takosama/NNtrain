namespace NNtrain;

/// <summary>
/// Inference-only linear whose matrix remains in its original GGUF Q4_K/Q6_K
/// representation.  Bias stays an ordinary NNtrain parameter because Qwen
/// attention projection biases are small.
/// </summary>
internal sealed class QwenQuantizedLinear : Module, IDisposable
{
    private readonly ArcQuantizedMatrix _weight;

    internal QwenQuantizedLinear(
        byte[] payload,
        uint ggmlType,
        int inputWidth,
        int outputWidth,
        float[]? bias,
        TensorDType dtype)
        : base(dtype)
    {
        _weight = new ArcQuantizedMatrix(
            payload, ggmlType, outputWidth, inputWidth);
        Bias = RegisterParameter(new Parameter(
            bias ?? new float[outputWidth],
            [outputWidth],
            "bias",
            WeightDecayPolicy.Exclude,
            dtype));
    }

    internal Parameter Bias { get; }
    internal int StorageBytes => _weight.StorageBytes + Bias.T.StorageByteLength;

    internal Tensor Forward(Tensor input) => _weight.Forward(input, Bias.T);

    public void Dispose() => _weight.Dispose();
}
