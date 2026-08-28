using NNtrain;
using Xunit;

public sealed class ClassificationAccuracyCpuTests
{
    [Fact]
    public void CpuMetricsPathRetainsStrictArgmaxWithoutCudaTransfers()
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var logits = new Tensor(
                [
                    1f, 9f, 9f, -1f,
                    float.NaN, 30f, 40f, 50f,
                    -4f, -3f, -2f, -1f,
                ],
                [3, 4]);
            int[] targets = [1, 0, 2];

            int correct = Program.CountCorrectForMetrics(
                logits,
                targets,
                classCount: 4);

            Assert.Equal(2, correct);
            Assert.Equal(TensorDevice.Cpu, logits.Device);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
        }
    }
}
