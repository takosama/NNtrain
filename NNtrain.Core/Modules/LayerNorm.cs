namespace NNtrain;

class LayerNorm : Module
{
    public Parameter Gamma { get; } // (dim)
    public Parameter Beta { get; }  // (dim)
    private readonly float _eps;

    public LayerNorm(
        int dim,
        float eps = 1e-5f,
        TensorDType dtype = TensorDType.Float32)
        : base(dtype)
    {
        float[] g = Enumerable.Repeat(1f, dim).ToArray();
        float[] b = new float[dim];

        Gamma = RegisterParameter(
            new Parameter(
                g,
                new[] { dim },
                "Gamma",
                WeightDecayPolicy.Exclude,
                dtype));
        Beta = RegisterParameter(
            new Parameter(
                b,
                new[] { dim },
                "Beta",
                WeightDecayPolicy.Exclude,
                dtype));
        _eps = eps;
    }

    public Tensor Forward(Tensor x)
    {
        return x.LayerNormLastDim(Gamma.T, Beta.T, _eps);
    }

    public Tensor ForwardResidual(Tensor x, Tensor residual)
    {
        return x.AddLayerNormLastDim(
            residual,
            Gamma.T,
            Beta.T,
            _eps);
    }

}
