namespace NNtrain;

class Linear : Module
{
    public Parameter W { get; } // (out, in)
    public Parameter B { get; } // (out)

    public Linear(
        int inFeatures,
        int outFeatures,
        Random? rng = null,
        float initScale = 0.02f,
        TensorDType dtype = TensorDType.Float32)
        : base(dtype)
    {
        rng ??= new Random(1);

        float[] w = new float[outFeatures * inFeatures];
        for (int i = 0; i < w.Length; i++)
            w[i] = ((float)rng.NextDouble() * 2f - 1f) * initScale;

        W = RegisterParameter(
            new Parameter(
                w,
                new[] { outFeatures, inFeatures },
                "W",
                WeightDecayPolicy.Apply,
                dtype));
        B = RegisterParameter(
            new Parameter(
                new float[outFeatures],
                new[] { outFeatures },
                "B",
                WeightDecayPolicy.Exclude,
            dtype));
    }

    /// <summary>
    /// Creates an output projection whose matrix is shared with another
    /// module (normally the token embedding). The shared parameter remains
    /// registered only at its original owner; this module owns only the bias.
    /// </summary>
    internal Linear(Parameter sharedWeight, int inFeatures, int outFeatures)
        : base(sharedWeight?.T.DType ?? throw new ArgumentNullException(
            nameof(sharedWeight)))
    {
        if (sharedWeight.T.Rank != 2
            || sharedWeight.T.Shape[0] != outFeatures
            || sharedWeight.T.Shape[1] != inFeatures)
        {
            throw new ArgumentException(
                "Shared linear weight must have shape [outFeatures, inFeatures].",
                nameof(sharedWeight));
        }

        W = sharedWeight;
        B = RegisterParameter(
            new Parameter(
                new float[outFeatures],
                [outFeatures],
                "B",
                WeightDecayPolicy.Exclude,
                sharedWeight.T.DType));
    }

    public Tensor Forward(Tensor x) // x: (in)
    {
        return W.T.MatMul(x) + B.T;
    }

    public Tensor ForwardBatch(Tensor x) // x: (..., in)
    {
        ArgumentNullException.ThrowIfNull(x);
        return x.LinearLastDim(W.T, B.T, applyRelu: false);
    }

    public Tensor ForwardBatchRelu(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);
        return x.LinearLastDim(W.T, B.T, applyRelu: true);
    }

}
