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

    public Tensor ForwardBatch(Tensor x) // x: (..., in)
    {
        ArgumentNullException.ThrowIfNull(x);

        if (x.Rank == 3)
        {
            int batch = x.Shape[0];
            int rows = x.Shape[1];
            int inputFeatures = x.Shape[2];
            Tensor flattened = x.Reshape(batch * rows, inputFeatures);
            Tensor projected = flattened.MatMulTransposedRightAddRow(W.T, B.T);
            return projected.Reshape(batch, rows, W.T.Shape[0]);
        }

        return x.MatMulTransposedRightAddRow(W.T, B.T);
    }

    public Tensor ForwardBatchRelu(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);

        if (x.Rank == 3)
        {
            int batch = x.Shape[0];
            int rows = x.Shape[1];
            int inputFeatures = x.Shape[2];
            Tensor flattened = x.Reshape(batch * rows, inputFeatures);
            Tensor projected = flattened
                .MatMulTransposedRightAddRowRelu(W.T, B.T);
            return projected.Reshape(batch, rows, W.T.Shape[0]);
        }

        return x.MatMulTransposedRightAddRowRelu(W.T, B.T);
    }

}
