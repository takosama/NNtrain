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
}
