using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class AdamWOptionsTests
{
    public static TheoryData<AdamWOptions> InvalidOptions => new()
    {
        new AdamWOptions { LearningRate = 0f },
        new AdamWOptions { LearningRate = float.NaN },
        new AdamWOptions { Beta1 = -0.1f },
        new AdamWOptions { Beta1 = 1f },
        new AdamWOptions { Beta2 = -0.1f },
        new AdamWOptions { Beta2 = 1f },
        new AdamWOptions { Epsilon = 0f },
        new AdamWOptions { Epsilon = float.PositiveInfinity },
        new AdamWOptions { WeightDecay = -0.1f },
        new AdamWOptions { WeightDecay = float.NaN },
    };

    [Fact]
    public void DefaultsMatchThePreviousAdamWConstructorDefaults()
    {
        var options = new AdamWOptions();

        Assert.Equal(1e-3f, options.LearningRate);
        Assert.Equal(0.9f, options.Beta1);
        Assert.Equal(0.999f, options.Beta2);
        Assert.Equal(1e-8f, options.Epsilon);
        Assert.Equal(1e-2f, options.WeightDecay);
        Assert.False(options.Decay1D);
    }

    [Fact]
    public void ConstructorAcceptsOnlyParametersAndOptions()
    {
        var constructor = Assert.Single(typeof(AdamW).GetConstructors());
        var constructorParameters = constructor.GetParameters();

        Assert.Equal(
            [typeof(IEnumerable<Parameter>), typeof(AdamWOptions)],
            constructorParameters.Select(parameter => parameter.ParameterType));
        Assert.True(constructorParameters[1].HasDefaultValue);
        Assert.Null(constructorParameters[1].DefaultValue);
    }

    [Fact]
    public void CustomLearningRateAndEpsilonControlTheUpdate()
    {
        var parameter = new Parameter(
            [1f],
            [1],
            "weight",
            WeightDecayPolicy.Exclude);
        parameter.T.MutableGrad[0] = 2f;
        var optimizer = new AdamW(
            [parameter],
            new AdamWOptions
            {
                LearningRate = 0.4f,
                Epsilon = 2f,
                WeightDecay = 0f,
            });

        optimizer.Step();

        AssertClose([0.8f], parameter.T.Data, 2e-6f);
    }

    [Theory]
    [MemberData(nameof(InvalidOptions))]
    public void ConstructorRejectsOptionsThatCouldCorruptParameters(
        AdamWOptions options)
    {
        var parameter = new Parameter(
            [1f],
            [1],
            "weight",
            WeightDecayPolicy.Exclude);

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            () => new AdamW([parameter], options));

        Assert.Equal("options", exception.ParamName);
    }
}
