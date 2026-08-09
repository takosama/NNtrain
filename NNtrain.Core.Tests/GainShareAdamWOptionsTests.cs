using NNtrain;
using Xunit;

public sealed class GainShareAdamWOptionsTests
{
    public static TheoryData<GainShareAdamWOptions> InvalidOptions => new()
    {
        new GainShareAdamWOptions { LearningRate = 0f },
        new GainShareAdamWOptions { Beta1 = 1f },
        new GainShareAdamWOptions { Beta2 = -0.1f },
        new GainShareAdamWOptions { Epsilon = 0f },
        new GainShareAdamWOptions { Rho = 1f },
        new GainShareAdamWOptions { Gamma = -0.1f },
        new GainShareAdamWOptions { MinScale = 0f },
        new GainShareAdamWOptions { MinScale = 2f, MaxScale = 1f },
        new GainShareAdamWOptions { WeightDecay = -0.1f },
    };

    [Fact]
    public void DefaultsMatchTheRequestedConfiguration()
    {
        var options = new GainShareAdamWOptions();

        Assert.Equal(3e-4f, options.LearningRate);
        Assert.Equal(0.9f, options.Beta1);
        Assert.Equal(0.999f, options.Beta2);
        Assert.Equal(1e-8f, options.Epsilon);
        Assert.Equal(0.95f, options.Rho);
        Assert.Equal(1f, options.Gamma);
        Assert.Equal(0.5f, options.MinScale);
        Assert.Equal(2f, options.MaxScale);
        Assert.Equal(5e-4f, options.WeightDecay);
    }

    [Theory]
    [MemberData(nameof(InvalidOptions))]
    public void ConstructorRejectsInvalidOptions(
        GainShareAdamWOptions options)
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new GainShareAdamW([[CreateParameter()]], options));
    }

    private static Parameter CreateParameter()
        => new(
            [1f],
            [1],
            "weight",
            WeightDecayPolicy.Apply);
}
