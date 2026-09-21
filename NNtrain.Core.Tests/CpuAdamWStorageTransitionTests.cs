using NNtrain;
using Xunit;

public sealed class CpuAdamWStorageTransitionTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ExistingOptimizerUpdatesCurrentMasterAfterMix8RestoreOrConversion(bool convertStorage)
    {
        using var scope = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu));
        var parameter = new Parameter([1f, -.75f, .125f], [3], "weight", WeightDecayPolicy.Apply);
        if (!convertStorage)
            parameter.T.ConvertStorageInPlace(TensorDType.Bfp8, Bfp8QuantizationDescriptor.Block(32), true);
        var options = new AdamWOptions { LearningRate = .01f, WeightDecay = 0, Beta1 = .7f, Beta2 = .8f };
        var optimizer = new AdamW([parameter], options);
        parameter.T.MutableGrad[0] = .4f;
        parameter.T.MutableGrad[1] = -.2f;
        parameter.T.MutableGrad[2] = .1f;
        optimizer.step();
        AdamWState savedOptimizer = optimizer.CaptureState();
        float[] savedMaster = parameter.T.CaptureData(preferMaster: true);

        if (convertStorage)
            parameter.T.ConvertStorageInPlace(TensorDType.Bfp8, Bfp8QuantizationDescriptor.Block(32), true);
        else
            parameter.T.RestoreBfp8ValuesInPlace(savedMaster, preserveFloat32Master: true);
        optimizer.RestoreState(savedOptimizer);

        var reference = new Parameter(savedMaster, [3], "weight", WeightDecayPolicy.Apply);
        reference.T.ConvertStorageInPlace(TensorDType.Bfp8, Bfp8QuantizationDescriptor.Block(32), true);
        var referenceOptimizer = new AdamW([reference], options);
        referenceOptimizer.RestoreState(savedOptimizer);
        float[] gradient = [-.3f, .15f, .6f];
        gradient.CopyTo(parameter.T.MutableGrad);
        gradient.CopyTo(reference.T.MutableGrad);
        optimizer.step();
        referenceOptimizer.step();

        float[] updatedMaster = parameter.T.CaptureData(preferMaster: true);
        Assert.NotEqual(savedMaster, updatedMaster);
        Assert.Equal(reference.T.CaptureData(preferMaster: true), updatedMaster);
        Assert.Equal(reference.T.Data, parameter.T.Data);
        AdamWState actual = optimizer.CaptureState();
        AdamWState expected = referenceOptimizer.CaptureState();
        Assert.Equal(2, actual.Step);
        Assert.Equal(expected.ParameterStates[0].FirstMoment, actual.ParameterStates[0].FirstMoment);
        Assert.Equal(expected.ParameterStates[0].SecondMoment, actual.ParameterStates[0].SecondMoment);
    }
}
