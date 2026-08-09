using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class LionTests
{
    [Fact]
    public void StepUsesSignedInterpolatedMomentumThenUpdatesTheEma()
    {
        Parameter parameter = CreateParameter(
            [1f],
            "weight",
            WeightDecayPolicy.Exclude);
        var optimizer = new Lion(
            [parameter],
            new LionOptions
            {
                LearningRate = 0.1f,
                Beta1 = 0.5f,
                Beta2 = 0.5f,
                WeightDecay = 0f,
            });

        parameter.T.MutableGrad[0] = 1f;
        optimizer.Step();
        parameter.T.MutableGrad[0] = -0.25f;
        optimizer.Step();

        AssertClose([0.8f], parameter.T.Data);
        AssertClose(
            [0.125f],
            optimizer.CaptureState().ParameterStates[0].Momentum);
    }

    [Fact]
    public void WeightDecayUsesParameterMetadata()
    {
        Parameter decayed = CreateParameter(
            [2f],
            "weight",
            WeightDecayPolicy.Apply);
        Parameter excluded = CreateParameter(
            [2f],
            "bias",
            WeightDecayPolicy.Exclude);
        var optimizer = new Lion(
            [decayed, excluded],
            new LionOptions
            {
                LearningRate = 0.1f,
                WeightDecay = 0.2f,
            });

        optimizer.Step();

        AssertClose([1.96f], decayed.T.Data);
        AssertClose([2f], excluded.T.Data);
    }

    [Fact]
    public void Decay1DOverridesExcludedMetadataForCompatibility()
    {
        Parameter parameter = CreateParameter(
            [2f],
            "bias",
            WeightDecayPolicy.Exclude);
        var optimizer = new Lion(
            [parameter],
            new LionOptions
            {
                LearningRate = 0.1f,
                WeightDecay = 0.2f,
                Decay1D = true,
            });

        optimizer.Step();

        AssertClose([1.96f], parameter.T.Data);
    }

    [Fact]
    public void StepAdvancesTheTensorDataVersion()
    {
        Parameter parameter = CreateParameter(
            [1f],
            "weight",
            WeightDecayPolicy.Exclude);
        parameter.T.MutableGrad[0] = 1f;
        long originalVersion = parameter.T.DataVersion;
        var optimizer = new Lion(
            [parameter],
            new LionOptions { WeightDecay = 0f });

        optimizer.Step();

        Assert.Equal(originalVersion + 1, parameter.T.DataVersion);
    }

    private static Parameter CreateParameter(
        float[] data,
        string name,
        WeightDecayPolicy weightDecay)
    {
        return new Parameter(data, [data.Length], name, weightDecay);
    }
}
