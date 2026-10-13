using NNtrain;
using Xunit;

public sealed class OptimizerParameterValidationTests
{
    [Theory]
    [InlineData("AdamW")]
    [InlineData("NekoMuon")]
    public void NullSequenceIsRejectedWithOriginalParameterName(string name)
    {
        ArgumentNullException error = Assert.Throws<ArgumentNullException>(
            () => Create(name, null!));

        Assert.Equal("parameters", error.ParamName);
    }

    [Theory]
    [InlineData("AdamW")]
    [InlineData("NekoMuon")]
    public void NullElementIsRejectedWithOriginalMessage(string name)
    {
        Parameter first = new([1f], [1], "weight", WeightDecayPolicy.Apply);
        ArgumentException error = Assert.Throws<ArgumentException>(
            () => Create(name, [first, null!]));

        Assert.Equal("parameters", error.ParamName);
        Assert.Equal(
            new ArgumentException(
                "Optimizer parameters cannot contain null.", "parameters").Message,
            error.Message);
    }

    [Theory]
    [InlineData("AdamW")]
    [InlineData("NekoMuon")]
    public void RepeatedReferenceIsRejectedWithOptimizerName(string name)
    {
        Parameter parameter = new([1f], [1], "weight", WeightDecayPolicy.Apply);
        ArgumentException error = Assert.Throws<ArgumentException>(
            () => Create(name, [parameter, parameter]));

        Assert.Equal("parameters", error.ParamName);
        Assert.Equal(
            new ArgumentException(
                $"Parameter 'weight' was supplied to {name} more than once.",
                "parameters").Message,
            error.Message);
    }

    [Theory]
    [InlineData("AdamW")]
    [InlineData("NekoMuon")]
    public void DistinctSameNameParametersKeepTheirSingleEnumerationOrder(string name)
    {
        Parameter first = new([1f], [1], "weight", WeightDecayPolicy.Apply);
        Parameter second = new([2f, 3f], [2], "weight", WeightDecayPolicy.Apply);
        int enumerations = 0;
        IEnumerable<Parameter> Enumerate()
        {
            enumerations++;
            yield return second;
            yield return first;
        }

        IOptimizer optimizer = Create(name, Enumerate());
        IReadOnlyList<Parameter> parameters = optimizer switch
        {
            AdamW adam => adam.Parameters,
            NekoMuon neko => neko.Parameters,
            _ => throw new InvalidOperationException(),
        };

        Assert.Equal(1, enumerations);
        Assert.Equal(2, parameters.Count);
        Assert.Same(second, parameters[0]);
        Assert.Same(first, parameters[1]);
    }

    private static IOptimizer Create(string name, IEnumerable<Parameter> parameters)
        => name switch
        {
            "AdamW" => new AdamW(parameters),
            "NekoMuon" => new NekoMuon(parameters),
            _ => throw new ArgumentOutOfRangeException(nameof(name)),
        };
}
