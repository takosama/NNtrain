namespace NNtrain;

/// <summary>Unmerged low-rank additive branch. Only A and B are trainable.</summary>
internal sealed class LoraAdapter : Module
{
    private readonly Tensor _aBias;
    private readonly Tensor _bBias;
    private readonly Tensor _scale;
    internal LoraAdapter(int input, int output, int rank, float alpha, Random random, TensorDType dtype, int bfp8BlockSize = 32)
        : base(dtype == TensorDType.Bfp8 ? TensorDType.Float32 : dtype)
    {
        TensorDType initialDtype = dtype == TensorDType.Bfp8 ? TensorDType.Float32 : dtype;
        float bound = 1f / MathF.Sqrt(input);
        A = RegisterParameter(new Parameter(Enumerable.Range(0, rank * input)
            .Select(_ => ((float)random.NextDouble() * 2 - 1) * bound).ToArray(),
            [rank, input], "lora_A", WeightDecayPolicy.Apply, initialDtype));
        B = RegisterParameter(new Parameter(new float[output * rank], [output, rank],
            "lora_B", WeightDecayPolicy.Apply, initialDtype));
        Tensor Constant(float[] values) => dtype == TensorDType.Bfp8
            ? Tensor.FromBfp8(values, [values.Length], Bfp8QuantizationDescriptor.Block(bfp8BlockSize))
            : new Tensor(values, [values.Length], dtype: dtype);
        _aBias = Constant(new float[rank]);
        _bBias = Constant(new float[output]);
        _scale = Constant([alpha / rank]);
        if (dtype == TensorDType.Bfp8) to(TensorPrecisionMode.Mix8_32, bfp8BlockSize);
    }
    internal Parameter A { get; }
    internal bool Enabled { get; set; } = true;
    internal Parameter B { get; }
    internal Tensor Forward(Tensor input)
        => input.LinearLastDim(A.T, _aBias, false).LinearLastDim(B.T, _bBias, false) * _scale;
}

/// <summary>Separate adapter state/optimizer view; does not contain base weights.</summary>
public sealed class LoraAdapterSet : Module
{
    internal LoraAdapterSet(TensorDType dtype) : base(dtype) { }
    private readonly List<string> _targets = [];
    private readonly List<LoraAdapter> _adapters = [];
    internal void SetEnabled(bool enabled)
    {
        foreach (var adapter in _adapters) adapter.Enabled = enabled;
    }
    public IReadOnlyList<string> Targets => _targets.AsReadOnly();
    internal void Add(string name, LoraAdapter adapter)
    {
        _targets.Add(name);
        _adapters.Add(adapter);
        RegisterModule(adapter);
    }
}
