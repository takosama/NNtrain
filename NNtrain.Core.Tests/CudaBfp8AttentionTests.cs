using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class CudaBfp8AttentionTests
{
    [Theory]
    [InlineData(false, 2, true, 8, 7)]
    [InlineData(true, 3, false, 24, 5)]
    [InlineData(true, 2, true, 40, 9)]
    public void ResidentAttentionMatchesCpuForwardBackwardForTensorAndBlockScales(
        bool blockScaled,
        int rank,
        bool causal,
        int headWidth,
        int sequence)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int heads = 2;
        int batch = rank == 3 ? 2 : 1;
        int modelWidth = checked(heads * headWidth);
        int length = checked(batch * sequence * 3 * modelWidth);
        int outputLength = checked(batch * sequence * modelWidth);
        int[] shape = rank == 3
            ? [batch, sequence, 3 * modelWidth]
            : [sequence, 3 * modelWidth];
        Bfp8QuantizationDescriptor descriptor = blockScaled
            ? new Bfp8QuantizationDescriptor(
                Bfp8ScaleGranularity.Block,
                128)
            : Bfp8QuantizationDescriptor.TensorWide;
        float[] values = Values(length, headWidth + sequence, 0.075f);
        float[] seed = Values(outputLength, headWidth * 3 + sequence, 0.031f);
        AttentionRun expected = CpuRun(
            values,
            shape,
            descriptor,
            heads,
            causal,
            seed);

        WithCuda(() =>
        {
            Tensor input = Tensor.FromBfp8(values, shape, descriptor);
            input.to(new TorchDevice(TensorDevice.Cuda, 0));
            _ = input.EnsureCudaBfp8Buffer(0);
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;
            _ = CudaDispatchPolicy.Startup;
            CudaDispatchEnvironmentTelemetrySnapshot environmentBefore =
                CudaDispatchEnvironmentTelemetry.Snapshot;

            Tensor output = input.FusedMultiHeadAttention(heads, causal);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaTransferTelemetry forwardTransfers =
                NativeCudaRuntime.TransferTelemetry - before;
            Assert.Equal(0, forwardTransfers.HostToDeviceBytes);
            Assert.Equal(0, forwardTransfers.DeviceToHostBytes);
            Assert.Equal(TensorDType.Bfp8, output.DType);
            Assert.Equal(descriptor, output.Bfp8Quantization);

            float[] actualOutput = output.Data.ToArray();
            output.BackwardAndRelease(seed);
            float[] actualGradient = input.Grad.ToArray();
            CudaDispatchEnvironmentTelemetrySnapshot environmentDelta =
                CudaDispatchEnvironmentTelemetry.Snapshot
                - environmentBefore;
            Assert.Equal(0, environmentDelta.EnvironmentReads);

            AssertClose(expected.Output, actualOutput, 6e-2f);
            AssertClose(expected.Gradient, actualGradient, 8e-2f);
            output.InvalidateCudaBuffers();
            input.InvalidateCudaBuffers();
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ResidentAttentionIsBf16FlashFollowedByDeviceRequantization(
        bool blockScaled)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int batch = 2;
        const int sequence = 11;
        const int heads = 2;
        const int headWidth = 24;
        const int modelWidth = heads * headWidth;
        int[] shape = [batch, sequence, 3 * modelWidth];
        Bfp8QuantizationDescriptor descriptor = blockScaled
            ? Bfp8QuantizationDescriptor.Mix8_32
            : Bfp8QuantizationDescriptor.TensorWide;
        float[] source = Values(
            batch * sequence * 3 * modelWidth,
            blockScaled ? 71 : 53,
            0.08f);

        WithCuda(() =>
        {
            Tensor bfp8Input = Tensor.FromBfp8(source, shape, descriptor);
            // Decode before moving the tensor. Both CUDA routes therefore see
            // precisely the same BF16 QKV operand.
            float[] decoded = bfp8Input.Data.ToArray();
            bfp8Input.to(new TorchDevice(TensorDevice.Cuda, 0));
            _ = bfp8Input.EnsureCudaBfp8Buffer(0);
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;

            Tensor bfp8Output = bfp8Input.FusedMultiHeadAttention(
                heads,
                causal: false);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - before;
            Assert.Equal(0, transfers.HostToDeviceBytes);
            Assert.Equal(0, transfers.DeviceToHostBytes);

            Tensor bf16Input = new(
                decoded,
                shape,
                dtype: TensorDType.BFloat16);
            bf16Input.to(new TorchDevice(TensorDevice.Cuda, 0));
            Tensor bf16Output = bf16Input.FusedMultiHeadAttention(
                heads,
                causal: false);
            float[] bf16Values = bf16Output.Data.ToArray();
            Bfp8EncodedStorage expectedEncoded =
                Bfp8QuantizationCodec.Default.Encode(
                    bf16Values,
                    descriptor);
            float[] expected = new float[bf16Values.Length];
            Bfp8QuantizationCodec.Default.Decode(
                expectedEncoded.Payload.Span,
                expectedEncoded.Scales.Span,
                descriptor,
                expected);

            AssertClose(expected, bfp8Output.Data, 3e-4f);
            bfp8Output.InvalidateCudaBuffers();
            bf16Output.InvalidateCudaBuffers();
            bfp8Input.InvalidateCudaBuffers();
            bf16Input.InvalidateCudaBuffers();
        });
    }

    [Fact]
    public void RepeatedNoGradAttentionReleasesEveryTransientCudaAllocation()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int batch = 2;
        const int sequence = 13;
        const int heads = 2;
        const int modelWidth = 48;
        int[] shape = [batch, sequence, 3 * modelWidth];

        WithCuda(() =>
        {
            Tensor input = Tensor.FromBfp8(
                Values(batch * sequence * 3 * modelWidth, 97, 0.06f),
                shape,
                Bfp8QuantizationDescriptor.Mix8_32);
            input.to(new TorchDevice(TensorDevice.Cuda, 0));
            _ = input.EnsureCudaBfp8Buffer(0);

            // Warm all native FlashAttention state before measuring managed
            // allocation ownership.
            RunInferenceIteration(input, heads);
            NativeCudaAllocationTelemetry before =
                NativeCudaRuntime.AllocationTelemetry;

            for (int iteration = 0; iteration < 24; iteration++)
                RunInferenceIteration(input, heads);

            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
            NativeCudaAllocationTelemetry delta =
                NativeCudaRuntime.AllocationTelemetry - before;
            Assert.Equal(delta.AllocationCount, delta.FreeCount);
            Assert.Equal(delta.AllocationBytes, delta.FreeBytes);
            input.InvalidateCudaBuffers();
        });
    }

    private static void RunInferenceIteration(Tensor input, int heads)
    {
        using IDisposable noGrad = AutogradContext.NoGrad();
        using CudaInferenceScope inference = CudaInferenceScope.Begin(
            resetPool: true,
            clearPoolOnDispose: true);
        _ = input.FusedMultiHeadAttention(heads, causal: true);
    }

    private static AttentionRun CpuRun(
        float[] values,
        int[] shape,
        Bfp8QuantizationDescriptor descriptor,
        int heads,
        bool causal,
        float[] seed)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            Tensor input = Tensor.FromBfp8(values, shape, descriptor);
            Tensor output = input.FusedMultiHeadAttention(heads, causal);
            output.Backward(seed);
            return new AttentionRun(
                output.Data.ToArray(),
                input.Grad.ToArray());
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index =>
                MathF.Sin((index + offset) * 0.137f) * scale
                + MathF.Cos((index + offset) * 0.047f) * scale * 0.41f)
            .ToArray();

    private static void WithCuda(Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.CudaDeviceIndex = 0;
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private sealed record AttentionRun(float[] Output, float[] Gradient);
}
