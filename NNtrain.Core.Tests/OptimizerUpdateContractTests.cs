using System.Reflection;
using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class OptimizerUpdateContractTests
{
    [Fact]
    public void ModuleAndParameterDoNotExposeStep()
    {
        const BindingFlags publicInstance =
            BindingFlags.Instance | BindingFlags.Public;

        Assert.Null(typeof(NNtrain.Module).GetMethod("Step", publicInstance));
        Assert.Null(typeof(Parameter).GetMethod("Step", publicInstance));
    }

    [Fact]
    public void AdamWOwnsTheVersionedParameterUpdate()
    {
        var parameter = new Parameter(
            [1f],
            [1],
            "weight",
            WeightDecayPolicy.Exclude);
        parameter.T.MutableGrad[0] = 1f;
        long originalVersion = parameter.T.DataVersion;
        var optimizer = new AdamW(
            [parameter],
            new AdamWOptions
            {
                LearningRate = 0.1f,
                WeightDecay = 0f,
            });

        optimizer.Step();

        AssertClose([0.9f], parameter.T.Data, 2e-5f);
        Assert.Equal(originalVersion + 1, parameter.T.DataVersion);
    }

    [Fact]
    public void ZeroGradDoesNotUpdateDataOrDataVersion()
    {
        var parameter = new Parameter(
            [1f],
            [1],
            "weight",
            WeightDecayPolicy.Exclude);
        parameter.T.MutableGrad[0] = 3f;
        float[] originalData = parameter.T.Data.ToArray();
        long originalVersion = parameter.T.DataVersion;

        parameter.ZeroGrad();

        AssertClose(originalData, parameter.T.Data);
        AssertClose([0f], parameter.T.Grad);
        Assert.Equal(originalVersion, parameter.T.DataVersion);
    }
}
