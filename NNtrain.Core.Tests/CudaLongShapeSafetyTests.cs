using NNtrain;
using Xunit;

public sealed class CudaLongShapeSafetyTests
{
    [Fact]
    public void Head32AttentionAndFusedLayerNormBackwardMatchTrainingShape()
    {
        if (Environment.GetEnvironmentVariable(
                "NNTRAIN_RUN_LARGE_CUDA_SAFETY_TEST") != "1"
            || !Tensor.IsCudaAvailable())
        {
            return;
        }

        const int defaultBatch = 36;
        const int sequence = 512;
        const int width = 512;
        const int heads = 16;
        int batch = int.TryParse(
            Environment.GetEnvironmentVariable(
                "NNTRAIN_CUDA_SAFETY_TEST_BATCH"),
            out int configuredBatch)
                ? configuredBatch
                : defaultBatch;
        int iterations = int.TryParse(
            Environment.GetEnvironmentVariable(
                "NNTRAIN_CUDA_SAFETY_TEST_ITERATIONS"),
            out int configuredIterations)
                ? configuredIterations
                : 1;
        ArgumentOutOfRangeException.ThrowIfLessThan(batch, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(iterations, 1);

        float[] qkvValues = Pattern(
            checked(batch * sequence * 3 * width),
            modulus: 31,
            scale: 0.001f);
        float[] residualValues = Pattern(
            checked(batch * sequence * width),
            modulus: 29,
            scale: 0.0015f);
        float[] gammaValues = Enumerable.Range(0, width)
            .Select(index => 0.9f + (index % 17) * 0.01f)
            .ToArray();
        float[] betaValues = Pattern(width, modulus: 13, scale: 0.002f);
        float[] outputGradient = Pattern(
            checked(batch * sequence * width),
            modulus: 23,
            scale: 0.0005f);

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            var qkv = new Tensor(
                qkvValues,
                [batch, sequence, 3 * width],
                dtype: TensorDType.BFloat16);
            var residual = new Tensor(
                residualValues,
                [batch, sequence, width],
                dtype: TensorDType.BFloat16);
            var gamma = new Tensor(
                gammaValues, [width], dtype: TensorDType.BFloat16);
            var beta = new Tensor(
                betaValues, [width], dtype: TensorDType.BFloat16);

            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(0);
            for (int iteration = 0; iteration < iterations; iteration++)
            {
                Tensor attention = qkv.FusedMultiHeadAttention(
                    heads, causal: true);
                Tensor output = residual.AddDropoutLayerNormLastDim(
                    attention,
                    gamma,
                    beta,
                    probability: 0.1f,
                    random: new Random(719 + iteration));
                output.BackwardAndRelease(outputGradient);
                accelerator.Synchronize();
            }
            Assert.True(CudaFlashAttention.TensorCoreBackendActive);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private static float[] Pattern(int length, int modulus, float scale)
        => Enumerable.Range(0, length)
            .Select(index => (index % modulus - modulus / 2) * scale)
            .ToArray();
}
