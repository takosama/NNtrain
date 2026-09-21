using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaFrozenLinearTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Mix8FrozenLinearMatchesInputGradientsWithoutBaseGradients(bool relu)
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA unavailable.");
        using var session = new ExecutionSession(new ExecutionOptions
        {
            Device = ExecutionDeviceKind.Cuda,
            CudaDevices = new DeviceSet([0]),
            Precision = PrecisionPolicy.Mix8_32,
        }, [CudaExecutionLaneFactory.Create(0)]);
        using IDisposable scope = session.Enter();

        // Non-aligned widths cover block-scale boundaries and output tails.
        const int rows = 6, inputWidth = 17, outputWidth = 19;
        Tensor input = Tensor.FromBfp8(
            Enumerable.Range(0, rows * inputWidth)
                .Select(i => MathF.Sin(i * 0.17f)).ToArray(),
            [2, 3, inputWidth], Bfp8QuantizationDescriptor.Mix8_32);
        Tensor weight = Tensor.FromBfp8(
            Enumerable.Range(0, inputWidth * outputWidth)
                .Select(i => MathF.Cos(i * 0.23f) * 0.2f).ToArray(),
            [outputWidth, inputWidth], Bfp8QuantizationDescriptor.Mix8_32);
        Tensor bias = Tensor.FromBfp8(
            Enumerable.Range(0, outputWidth)
                .Select(i => i % 2 == 0 ? 0.2f : -0.2f).ToArray(),
            [outputWidth], Bfp8QuantizationDescriptor.Mix8_32);
        float[] upstream = Enumerable.Range(0, rows * outputWidth)
            .Select(i => MathF.Sin(i * 0.11f) * 0.7f).ToArray();
        input.to(TensorDevice.Cuda);
        weight.to(TensorDevice.Cuda);
        bias.to(TensorDevice.Cuda);
        try
        {
            using (DeviceTransferGuard.EnterTrainingStep(1))
            {
                Tensor output = input.LinearLastDimFrozen(weight, bias, relu);
                Assert.Equal(TensorDType.Bfp8, output.DType);
                Assert.Equal(Bfp8QuantizationDescriptor.Mix8_32, output.Bfp8Quantization);
                // The test supplies an explicit host seed; classify that one
                // upload and assert that no other H2D traffic was introduced.
                using (DeviceTransferGuard.AllowBatchHostToDevice())
                    output.BackwardAndRelease(upstream);
                Assert.False(weight.HasGradientBuffer);
                Assert.False(bias.HasGradientBuffer);
                Assert.True(input.HasGradientBuffer);
                Assert.False(input.TryGetCudaBFloat16GradientBuffer(0, out _));
                Assert.False(input.TryGetCudaBfp8GradientBuffer(0, out _));
                DeviceTransferSnapshot transfers = Assert.NotNull(DeviceTransferGuard.CurrentSnapshot);
                Assert.Equal(1, transfers.HostToDeviceCopyCount);
                Assert.Equal(upstream.Length * sizeof(float), transfers.HostToDeviceBytes);
                Assert.Equal(0, transfers.DeviceToHostBytes);
            }
            float[] frozenGradient = input.Grad.ToArray();
            Assert.Contains(frozenGradient, g => MathF.Abs(g) > 1e-5f);

            // A second branch must add to the FP32 accumulator, as the LoRA
            // adapter and the frozen base both contribute to the same input.
            input.LinearLastDimFrozen(weight, bias, relu).BackwardAndRelease(upstream);
            AssertClose(frozenGradient.Select(g => 2 * g).ToArray(), input.Grad.ToArray());
            Assert.False(weight.HasGradientBuffer);
            Assert.False(bias.HasGradientBuffer);

            input.ZeroGrad();
            input.LinearLastDim(weight, bias, relu).BackwardAndRelease(upstream);
            Assert.True(weight.HasGradientBuffer);
            Assert.True(bias.HasGradientBuffer);
            AssertClose(frozenGradient, input.Grad.ToArray());
        }
        finally
        {
            input.InvalidateCudaBuffers();
            weight.InvalidateCudaBuffers();
            bias.InvalidateCudaBuffers();
        }
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(MathF.Abs(expected[i] - actual[i]), 0f, 2e-6f);
    }
}
