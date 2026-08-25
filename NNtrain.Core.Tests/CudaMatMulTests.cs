using NNtrain;
using Xunit;

public sealed class CudaMatMulTests
{
    [Theory]
    [InlineData(TensorDType.Float32, 2e-5f)]
    [InlineData(TensorDType.BFloat16, 2e-2f)]
    public void MatMulAndBatchedMatMulMatchCpuForwardBackward(
        TensorDType dtype,
        float tolerance)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previous = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            static float[] Values(int length, int offset)
                => Enumerable.Range(0, length)
                    .Select(i => MathF.Sin((i + offset) * 0.37f) * 0.5f)
                    .ToArray();

            (float[] Output, float[] LeftGradient, float[] RightGradient) Run(
                TensorDevice device,
                bool batched)
            {
                Tensor.ExecutionDevice = device;
                Tensor.CudaDeviceIndices = [0];
                int batch = batched ? 3 : 1;
                int m = 5;
                int k = 7;
                int n = 4;
                var left = new Tensor(
                    Values(batch * m * k, 1),
                    batched ? [batch, m, k] : [m, k],
                    dtype: dtype);
                var right = new Tensor(
                    Values(batch * k * n, 11),
                    batched ? [batch, k, n] : [k, n],
                    dtype: dtype);
                Tensor output = batched
                    ? left.BatchedMatMul(right)
                    : left.MatMul(right);
                output.Backward(Values(batch * m * n, 23));
                return (
                    output.Data.ToArray(),
                    left.Grad.ToArray(),
                    right.Grad.ToArray());
            }

            foreach (bool batched in new[] { false, true })
            {
                var cpu = Run(TensorDevice.Cpu, batched);
                var cuda = Run(TensorDevice.Cuda, batched);
                AssertClose(cpu.Output, cuda.Output, tolerance);
                AssertClose(cpu.LeftGradient, cuda.LeftGradient, tolerance);
                AssertClose(cpu.RightGradient, cuda.RightGradient, tolerance);
            }
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.InRange(
                MathF.Abs(expected[i] - actual[i]),
                0f,
                tolerance);
        }
    }
}
