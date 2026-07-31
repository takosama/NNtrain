namespace NNtrain;

class Linear : Module
{
    public Parameter W { get; } // (out, in)
    public Parameter B { get; } // (out)

    public Linear(int inFeatures, int outFeatures, Random? rng = null, float initScale = 0.02f)
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
                WeightDecayPolicy.Apply));
        B = RegisterParameter(
            new Parameter(
                new float[outFeatures],
                new[] { outFeatures },
                "B",
                WeightDecayPolicy.Exclude));
    }

    public Tensor Forward(Tensor x) // x: (in)
    {
        return W.T.MatMul(x) + B.T;
    }

    public Tensor ForwardBatch(Tensor x) // x: (batch, in)
    {
        return x.MatMul(W.T.Transpose()).AddRowWise(B.T);
    }

}
