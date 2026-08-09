using NNtrain;
using Xunit;

public sealed class LionOptionsTests
{
    public static TheoryData<LionOptions> InvalidOptions => new()
    {
        new LionOptions { LearningRate = 0f },
        new LionOptions { LearningRate = float.NaN },
        new LionOptions { Beta1 = -0.1f },
        new LionOptions { Beta1 = 1f },
        new LionOptions { Beta2 = -0.1f },
        new LionOptions { Beta2 = 1f },
        new LionOptions { WeightDecay = -0.1f },
        new LionOptions { WeightDecay = float.PositiveInfinity },
    };

    [Fact]
    public void DefaultsMatchTheConfiguredLionHyperparameters()
    {
        var options = new LionOptions();

        Assert.Equal(3e-4f, options.LearningRate);
        Assert.Equal(0.9f, options.Beta1);
        Assert.Equal(0.99f, options.Beta2);
        Assert.Equal(1e-2f, options.WeightDecay);
        Assert.False(options.Decay1D);
    }

    [Theory]
    [MemberData(nameof(InvalidOptions))]
    public void ConstructorRejectsOptionsThatCouldCorruptParameters(
        LionOptions options)
    {
        Parameter parameter = CreateParameter([1f], "weight");

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            () => new Lion([parameter], options));

        Assert.Equal("options", exception.ParamName);
    }

    private static Parameter CreateParameter(float[] data, string name)
    {
        return new Parameter(
            data,
            [data.Length],
            name,
            WeightDecayPolicy.Exclude);
    }
}
