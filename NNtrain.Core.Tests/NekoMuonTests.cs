using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class NekoMuonTests
{
    [Fact]
    public void ConfidenceControlsFractionalNewtonSchulzDepth()
    {
        Parameter parameter = CreateParameter(
            [0f, 0f, 0f, 0f],
            [2, 2],
            WeightDecayPolicy.Exclude);
        new float[] { 1f, 0f, 0f, 1f }
            .AsSpan()
            .CopyTo(parameter.T.MutableGrad);
        var optimizer = new NekoMuon(
            [parameter],
            new NekoMuonOptions
            {
                LearningRate = 1f,
                BetaFast = 0f,
                BetaSlow = 0f,
                Rho = 0.5f,
                Epsilon = 1e-12f,
                MaxNewtonSchulzSteps = 1,
                WeightDecay = 0f,
            });

        optimizer.Step();

        AssertClose(
            [-0.9068203f, 0f, 0f, -0.9068203f],
            parameter.T.Data,
            2e-5f);
        Assert.Equal(0.5f, optimizer.CaptureState()
            .ParameterStates[0].Confidence);
    }

    [Fact]
    public void TallMatricesAreTransposedAndReceiveMuonFinalScale()
    {
        Parameter parameter = CreateParameter(
            [0f, 0f],
            [2, 1],
            WeightDecayPolicy.Exclude);
        new float[] { 1f, 0f }
            .AsSpan()
            .CopyTo(parameter.T.MutableGrad);
        var optimizer = new NekoMuon(
            [parameter],
            new NekoMuonOptions
            {
                LearningRate = 1f,
                BetaFast = 0f,
                BetaSlow = 0f,
                Rho = 0f,
                Epsilon = 1e-12f,
                MaxNewtonSchulzSteps = 1,
                WeightDecay = 0f,
            });

        optimizer.Step();

        AssertClose([-0.9913637f, 0f], parameter.T.Data, 2e-5f);
    }

    [Fact]
    public void FastAndSlowBiasCorrectionProduceTheFirstGradient()
    {
        Parameter parameter = CreateParameter(
            [0f, 0f],
            [1, 2],
            WeightDecayPolicy.Exclude);
        new float[] { 3f, 4f }
            .AsSpan()
            .CopyTo(parameter.T.MutableGrad);
        var optimizer = new NekoMuon(
            [parameter],
            new NekoMuonOptions
            {
                LearningRate = 0.1f,
                BetaFast = 0.5f,
                BetaSlow = 0.75f,
                Rho = 0.9f,
                MaxNewtonSchulzSteps = 1,
                WeightDecay = 0f,
            });

        optimizer.Step();
        NekoMuonParameterState state =
            optimizer.CaptureState().ParameterStates[0];

        AssertClose([1.5f, 2f], state.FastMoment);
        AssertClose([0.75f, 1f], state.SlowMoment);
        Assert.InRange(state.Confidence, 0.09999f, 0.10001f);
    }

    [Fact]
    public void PersistenceReducesConfidenceWhenFastDepartsFromSlow()
    {
        Parameter parameter = CreateParameter(
            [0f],
            [1],
            WeightDecayPolicy.Exclude);
        var optimizer = new NekoMuon(
            [parameter],
            new NekoMuonOptions
            {
                LearningRate = 0.01f,
                BetaFast = 0.5f,
                BetaSlow = 0.75f,
                Rho = 0.9f,
                Epsilon = 1e-12f,
                MaxNewtonSchulzSteps = 1,
                WeightDecay = 0f,
            });

        parameter.T.MutableGrad[0] = 1f;
        optimizer.Step();
        parameter.T.MutableGrad[0] = -1f;
        optimizer.Step();

        float confidence = optimizer.CaptureState()
            .ParameterStates[0].Confidence;
        Assert.InRange(confidence, 0.12599f, 0.12601f);
    }

    [Fact]
    public void WeightDecayUsesParameterMetadata()
    {
        Parameter decayed = CreateParameter(
            [2f],
            [1],
            WeightDecayPolicy.Apply);
        Parameter excluded = CreateParameter(
            [2f],
            [1],
            WeightDecayPolicy.Exclude);
        var optimizer = new NekoMuon(
            [decayed, excluded],
            new NekoMuonOptions
            {
                LearningRate = 0.1f,
                WeightDecay = 0.2f,
            });

        optimizer.Step();

        AssertClose([1.96f], decayed.T.Data);
        AssertClose([2f], excluded.T.Data);
    }

    private static Parameter CreateParameter(
        float[] data,
        int[] shape,
        WeightDecayPolicy weightDecay)
    {
        return new Parameter(data, shape, "weight", weightDecay);
    }
}
