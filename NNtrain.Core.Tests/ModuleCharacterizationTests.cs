using Xunit;
using NNtrain;
using static TensorCharacterizationTests;

public sealed class ModuleCharacterizationTests
{
    [Fact]
    public void TransformerCanContinueFromItsExposedEmbedding()
    {
        var model = new TransformerClassifier(
            2,
            4,
            2,
            8,
            1,
            3,
            new Random(11));
        Tensor input = Tensor.From2D(new float[,]
        {
            { 0.1f, 0.2f, 0.3f, 0.4f },
            { 0.5f, 0.6f, 0.7f, 0.8f },
        });

        Tensor direct = model.Forward(input);
        Tensor embedding = model.Embed(input);
        Tensor continued = model.ForwardFromEmbedding(embedding);

        Assert.Equal([2, 4], embedding.Shape);
        AssertClose(direct.Data, continued.Data);
    }

    [Fact]
    public void TransformerSeparatesHiddenMatrixWeightsFromAuxiliaryParameters()
    {
        var model = new TransformerClassifier(
            seqLen: 2,
            dModel: 4,
            numHeads: 2,
            dHidden: 8,
            numLayers: 1,
            numClasses: 3,
            rng: new Random(11));

        Parameter[] all = model.Parameters().ToArray();
        Parameter[] hidden = model.HiddenWeightParameters.ToArray();
        Parameter[] auxiliary = model.AuxiliaryParameters.ToArray();

        Assert.Equal(4, hidden.Length);
        Assert.Equal(11, auxiliary.Length);
        Assert.All(hidden, parameter => Assert.Equal(2, parameter.T.Rank));
        Assert.DoesNotContain(model.Pos, hidden);
        Assert.DoesNotContain(model.Head.W, hidden);
        Assert.DoesNotContain(model.Head.B, hidden);
        Assert.Contains(model.Pos, auxiliary);
        Assert.Contains(model.Head.W, auxiliary);
        Assert.Contains(model.Head.B, auxiliary);
        Assert.Empty(hidden.Intersect(auxiliary, ReferenceEqualityComparer.Instance));
        Assert.Equal(
            all.ToHashSet(ReferenceEqualityComparer.Instance),
            hidden.Concat(auxiliary)
                .ToHashSet(ReferenceEqualityComparer.Instance));
    }

    [Fact]
    public void LinearForwardAndBackwardAreDeterministic()
    {
        var linear = new Linear(2, 2, new Random(123), initScale: 0.1f);
        var input = Tensor.From1D([0.5f, -1f]);

        var output = linear.Forward(input);
        AssertClose([-0.03310738f, -0.03797377f], output.Data, 1e-5f);

        output.Sum().Backward();
        AssertClose([0.5f, -1f, 0.5f, -1f], linear.W.T.Grad);
        AssertClose([1f, 1f], linear.B.T.Grad);
    }

    [Fact]
    public void AttentionAndTransformerPreserveDocumentedShapes()
    {
        var input = new Tensor(
            [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f],
            [2, 4]);

        var attention = new MultiHeadAttention(4, 2, causal: true, new Random(7));
        var attentionOutput = attention.Forward(input);
        Assert.Equal([2, 4], attentionOutput.Shape);
        Assert.All(attentionOutput.Data, value => Assert.True(float.IsFinite(value)));

        var block = new TransformerBlock(4, 2, 8, rng: new Random(7));
        var blockOutput = block.Forward(input);
        Assert.Equal([2, 4], blockOutput.Shape);

        var classifier = new TransformerClassifier(2, 4, 2, 8, 2, 3, new Random(7));
        var logits = classifier.Forward(input);
        Assert.Equal([3], logits.Shape);
        Assert.All(logits.Data, value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void MultiHeadAttentionUsesOneFusedQkvProjection()
    {
        var attention = new MultiHeadAttention(
            dModel: 8,
            numHeads: 2,
            rng: new Random(9));

        Assert.Equal<int>([24, 8], attention.Qkv.W.T.Shape);
        Assert.Equal<int>([24], attention.Qkv.B.T.Shape);
        Assert.Equal(4, attention.Parameters().Count());
    }

    [Fact]
    public void BatchedTransformerMatchesIndependentSamplesAndBackpropagates()
    {
        var model = new TransformerClassifier(
            2,
            4,
            2,
            8,
            1,
            3,
            new Random(13));
        float[] values =
        [
            0.1f, 0.2f, 0.3f, 0.4f,
            0.5f, 0.6f, 0.7f, 0.8f,
            -0.1f, -0.2f, -0.3f, -0.4f,
            -0.5f, -0.6f, -0.7f, -0.8f,
        ];

        float[] expected;
        using (AutogradContext.NoGrad())
        {
            expected = new float[6];
            for (int batch = 0; batch < 2; batch++)
            {
                var sample = new Tensor(
                    values.Skip(batch * 8).Take(8).ToArray(),
                    [2, 4]);
                Tensor logits = model.Forward(sample);
                logits.Data.ToArray().CopyTo(expected, batch * 3);
            }
        }

        var input = new Tensor(values, [2, 2, 4]);
        Tensor batched = model.ForwardBatch(input);

        Assert.Equal<int>([2, 3], batched.Shape);
        AssertClose(expected, batched.Data, 2e-5f);

        batched.Sum().Backward();
        Assert.All(input.Grad, value => Assert.True(float.IsFinite(value)));
        Assert.All(
            model.Parameters().SelectMany(parameter => parameter.T.Grad),
            value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void BatchedTransformerBackwardMatchesAccumulatedSampleGradients()
    {
        var batchedModel = new TransformerClassifier(
            2,
            4,
            2,
            8,
            1,
            3,
            new Random(17));
        var referenceModel = new TransformerClassifier(
            2,
            4,
            2,
            8,
            1,
            3,
            new Random(17));
        float[] values =
        [
            0.1f, 0.2f, 0.3f, 0.4f,
            0.5f, 0.6f, 0.7f, 0.8f,
            -0.1f, -0.2f, -0.3f, -0.4f,
            -0.5f, -0.6f, -0.7f, -0.8f,
        ];
        float[] outputGradient =
        [
            0.1f, -0.2f, 0.3f,
            -0.4f, 0.5f, -0.6f,
        ];
        var batchedInput = new Tensor(values, [2, 2, 4]);

        Tensor batchedOutput = batchedModel.ForwardBatch(batchedInput);
        batchedOutput.Backward(outputGradient);

        var expectedOutput = new List<float>();
        var expectedInputGradient = new List<float>();
        for (int batch = 0; batch < 2; batch++)
        {
            var sampleInput = new Tensor(
                values.Skip(batch * 8).Take(8).ToArray(),
                [2, 4]);
            Tensor sampleOutput = referenceModel.Forward(sampleInput);
            sampleOutput.Backward(
                outputGradient.Skip(batch * 3).Take(3).ToArray());
            expectedOutput.AddRange(sampleOutput.Data);
            expectedInputGradient.AddRange(sampleInput.Grad);
        }

        AssertClose(expectedOutput, batchedOutput.Data, 2e-5f);
        AssertClose(expectedInputGradient, batchedInput.Grad, 3e-4f);

        Parameter[] batchedParameters = batchedModel.Parameters().ToArray();
        Parameter[] referenceParameters = referenceModel.Parameters().ToArray();
        Assert.Equal(referenceParameters.Length, batchedParameters.Length);
        for (int index = 0; index < batchedParameters.Length; index++)
        {
            AssertClose(
                referenceParameters[index].T.Grad,
                batchedParameters[index].T.Grad,
                3e-4f);
        }
    }

    [Fact]
    public void TransformerBackwardProducesFiniteGradientsForEveryParameter()
    {
        var model = new TransformerClassifier(2, 4, 2, 8, 1, 3, new Random(11));
        var input = new Tensor(
            [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f],
            [2, 4]);

        model.Forward(input).Sum().Backward();

        var parameters = model.Parameters().ToArray();
        Assert.NotEmpty(parameters);
        Assert.All(parameters, parameter =>
            Assert.All(parameter.T.Grad, value => Assert.True(float.IsFinite(value))));
        Assert.Contains(parameters.SelectMany(parameter => parameter.T.Grad), value => value != 0f);
    }

    [Fact]
    public void AdamWOneStepMatchesCurrentUpdateRule()
    {
        var parameter = new Parameter(
            [1f, -2f],
            [2],
            "p",
            WeightDecayPolicy.Exclude);
        parameter.T.MutableGrad[0] = 0.5f;
        parameter.T.MutableGrad[1] = -0.25f;
        var optimizer = new AdamW(
            [parameter],
            new AdamWOptions
            {
                LearningRate = 0.01f,
                Beta1 = 0.9f,
                Beta2 = 0.999f,
                Epsilon = 1e-8f,
                WeightDecay = 0.1f,
                Decay1D = true,
            });

        optimizer.Step();

        AssertClose([0.989f, -1.988f], parameter.T.Data, 2e-6f);
    }

    [Fact]
    public void AdamWDoesNotDecayOneDimensionalParametersByDefault()
    {
        var parameter = new Parameter(
            [1f],
            [1],
            "p",
            WeightDecayPolicy.Exclude);
        parameter.T.MutableGrad[0] = 0f;
        var optimizer = new AdamW(
            [parameter],
            new AdamWOptions
            {
                LearningRate = 0.01f,
                WeightDecay = 0.1f,
            });

        optimizer.Step();

        AssertClose([1f], parameter.T.Data);
    }
}
