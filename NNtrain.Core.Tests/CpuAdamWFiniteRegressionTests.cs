using System.Runtime.Intrinsics;
using NNtrain;
using Xunit;

public sealed class CpuAdamWFiniteRegressionTests
{
    [Fact]
    public void SimdUpdateKeepsSubnormalSecondMomentFinite()
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var parameter = new Parameter(
                new float[Vector256<float>.Count],
                [Vector256<float>.Count],
                "subnormal-second-moment",
                WeightDecayPolicy.Exclude);
            parameter.T.MutableGrad.Fill(1e-3f);
            parameter.T.MutableGrad[4] = -1.73472348e-18f;
            var optimizer = new AdamW(
                [parameter],
                new AdamWOptions { LearningRate = 0.001f });

            optimizer.Step();

            Assert.All(
                parameter.T.Data,
                value => Assert.True(float.IsFinite(value)));
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
        }
    }
}
