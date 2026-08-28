using NNtrain;
using Xunit;

public sealed class NekoMuonOptionsTests
{
    public static TheoryData<NekoMuonOptions> InvalidOptions => new()
    {
        new NekoMuonOptions { LearningRate = 0f },
        new NekoMuonOptions { LearningRate = float.NaN },
        new NekoMuonOptions { BetaFast = -0.1f },
        new NekoMuonOptions { BetaFast = 1f },
        new NekoMuonOptions { BetaSlow = float.NaN },
        new NekoMuonOptions { BetaSlow = 1f },
        new NekoMuonOptions { Rho = -0.1f },
        new NekoMuonOptions { Rho = 1f },
        new NekoMuonOptions { Epsilon = 0f },
        new NekoMuonOptions { Epsilon = float.PositiveInfinity },
        new NekoMuonOptions { MaxNewtonSchulzSteps = 0 },
        new NekoMuonOptions { NewtonSchulzInterval = 0 },
        new NekoMuonOptions
        {
            NewtonSchulzDepthMode =
                (NekoMuonNewtonSchulzDepthMode)int.MaxValue,
        },
        new NekoMuonOptions
        {
            NewtonSchulzDepthMode =
                NekoMuonNewtonSchulzDepthMode.Minimum,
            NewtonSchulzDepth = float.NaN,
        },
        new NekoMuonOptions
        {
            NewtonSchulzDepthMode =
                NekoMuonNewtonSchulzDepthMode.Minimum,
            NewtonSchulzDepth = -0.25f,
        },
        new NekoMuonOptions
        {
            MaxNewtonSchulzSteps = 2,
            NewtonSchulzDepthMode =
                NekoMuonNewtonSchulzDepthMode.Fixed,
            NewtonSchulzDepth = 2.25f,
        },
        new NekoMuonOptions
        {
            NewtonSchulzDepthMode =
                NekoMuonNewtonSchulzDepthMode.Fixed,
            NewtonSchulzDepth = float.PositiveInfinity,
        },
        new NekoMuonOptions { WeightDecay = -0.1f },
    };

    [Fact]
    public void DefaultsMatchTheConfiguredNekoMuonHyperparameters()
    {
        var options = new NekoMuonOptions();

        Assert.Equal(3e-4f, options.LearningRate);
        Assert.Equal(0.9f, options.BetaFast);
        Assert.Equal(0.99f, options.BetaSlow);
        Assert.Equal(0.9f, options.Rho);
        Assert.Equal(1e-7f, options.Epsilon);
        Assert.Equal(5, options.MaxNewtonSchulzSteps);
        Assert.Equal(5, options.NewtonSchulzInterval);
        Assert.Equal(
            NekoMuonNewtonSchulzDepthMode.Adaptive,
            options.NewtonSchulzDepthMode);
        Assert.Equal(0f, options.NewtonSchulzDepth);
        Assert.Equal(1e-2f, options.WeightDecay);
        Assert.False(options.Decay1D);
    }

    [Theory]
    [MemberData(nameof(InvalidOptions))]
    public void ConstructorRejectsInvalidOptions(NekoMuonOptions options)
    {
        var parameter = new Parameter(
            [1f],
            [1],
            "weight",
            WeightDecayPolicy.Exclude);

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            () => new NekoMuon([parameter], options));

        Assert.Equal("options", exception.ParamName);
    }

    [Fact]
    public void ConstructorRejectsConfiguredDepthInAdaptiveMode()
    {
        var parameter = new Parameter(
            [1f],
            [1],
            "weight",
            WeightDecayPolicy.Exclude);
        var options = new NekoMuonOptions
        {
            NewtonSchulzDepthMode =
                NekoMuonNewtonSchulzDepthMode.Adaptive,
            NewtonSchulzDepth = 0.5f,
        };

        var exception = Assert.Throws<ArgumentException>(
            () => new NekoMuon([parameter], options));

        Assert.Equal("options", exception.ParamName);
    }
}
