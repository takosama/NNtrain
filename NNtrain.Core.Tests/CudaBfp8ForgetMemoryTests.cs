using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaBfp8ForgetMemoryTests
{
    [Theory]
    [InlineData(false, 16, 16, 5)]
    [InlineData(true, 17, 13, 7)]
    public void ResidentForwardBackwardMatchesBf16ReferenceWithoutTransfers(
        bool blockScaled,
        int keyWidth,
        int valueWidth,
        int sequence)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int batch = 2;
        int projectionWidth = checked(2 * keyWidth + 3 * valueWidth);
        int[] shape = [batch, sequence, projectionWidth];
        Bfp8QuantizationDescriptor descriptor = blockScaled
            ? Bfp8QuantizationDescriptor.Mix8_32
            : Bfp8QuantizationDescriptor.TensorWide;
        float[] values = Values(
            batch * sequence * projectionWidth,
            keyWidth + valueWidth,
            0.19f);
        float[] seed = Values(
            batch * sequence * valueWidth,
            sequence * 7,
            0.071f);
        ForgetMemoryRun expected = Bf16Reference(
            values,
            shape,
            descriptor,
            keyWidth,
            valueWidth,
            retentionFloor: 0.37f,
            seed);

        WithCuda(descriptor, () =>
        {
            Tensor input = Tensor.FromBfp8(values, shape, descriptor);
            input.to(new TorchDevice(TensorDevice.Cuda, 0));
            _ = input.EnsureCudaBfp8Buffer(0);
            NativeCudaTransferTelemetry transferBefore =
                NativeCudaRuntime.TransferTelemetry;
            CudaBfp8ForgetMemoryTelemetrySnapshot backendBefore =
                CudaBfp8ForgetMemoryTelemetry.Snapshot;
            _ = CudaDispatchPolicy.Startup;
            CudaDispatchEnvironmentTelemetrySnapshot environmentBefore =
                CudaDispatchEnvironmentTelemetry.Snapshot;

            Tensor output = input.ForgetMemoryV2(
                keyWidth,
                valueWidth,
                retentionFloor: 0.37f);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - transferBefore;
            CudaBfp8ForgetMemoryTelemetrySnapshot backend =
                CudaBfp8ForgetMemoryTelemetry.Snapshot - backendBefore;
            Assert.Equal(0, transfers.HostToDeviceBytes);
            Assert.Equal(0, transfers.DeviceToHostBytes);
            Assert.Equal(1,
                backend.TensorCoreForwardExecutions
                + backend.GenericCudaForwardExecutions);
            if (keyWidth % 16 != 0 || valueWidth % 16 != 0)
            {
                Assert.Equal(0, backend.TensorCoreForwardExecutions);
                Assert.Equal(1, backend.GenericCudaForwardExecutions);
            }
            else if (!CudaDispatchPolicy.Current
                .DisableTensorCoreForgetMemory)
            {
                Assert.Equal(1, backend.TensorCoreForwardExecutions);
            }
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
            AssertClose(expected.Gradient, actualGradient, 1.1e-1f);
            if (descriptor == Bfp8QuantizationDescriptor.TensorWide)
            {
                Assert.True(input.HasAuthoritativeCudaBfp8Gradient);
                CudaBfp8BufferView gradient =
                    input.EnsureCudaBfp8GradientBuffer(0);
                Assert.Equal(
                    Bfp8QuantizationDescriptor.TensorWide,
                    gradient.Descriptor);
                Assert.Equal(1, gradient.Scales.Length);
            }

            output.InvalidateCudaBuffers();
            input.InvalidateCudaBuffers();
        });
    }

    [Fact]
    public void RecurrentContinuationTransfersOnlyExplicitFp32State()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int keyWidth = 17;
        const int valueWidth = 13;
        const int firstSequence = 3;
        const int secondSequence = 4;
        int projectionWidth = 2 * keyWidth + 3 * valueWidth;
        int matrixSize = keyWidth * valueWidth;
        Bfp8QuantizationDescriptor descriptor =
            Bfp8QuantizationDescriptor.Mix8_32;
        float[] firstValues = Values(
            firstSequence * projectionWidth,
            101,
            0.16f);
        float[] secondValues = Values(
            secondSequence * projectionWidth,
            173,
            0.14f);
        float[] initialState = Values(matrixSize, 211, 0.023f);
        ContinueRun expected = Bf16ContinuationReference(
            firstValues,
            secondValues,
            initialState,
            descriptor,
            firstSequence,
            secondSequence,
            projectionWidth,
            keyWidth,
            valueWidth);

        WithCuda(descriptor, () =>
        {
            using IDisposable noGrad = AutogradContext.NoGrad();
            using CudaInferenceScope inference = CudaInferenceScope.Begin(
                resetPool: true,
                clearPoolOnDispose: true);
            Tensor first = Tensor.FromBfp8(
                firstValues,
                [1, firstSequence, projectionWidth],
                descriptor);
            Tensor second = Tensor.FromBfp8(
                secondValues,
                [1, secondSequence, projectionWidth],
                descriptor);
            first.to(new TorchDevice(TensorDevice.Cuda, 0));
            second.to(new TorchDevice(TensorDevice.Cuda, 0));
            _ = first.EnsureCudaBfp8Buffer(0);
            _ = second.EnsureCudaBfp8Buffer(0);
            float[] actualState = initialState.ToArray();
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;

            Tensor firstOutput = first.ForgetMemoryV2Continue(
                keyWidth,
                valueWidth,
                retentionFloor: 0.42f,
                actualState);
            Tensor secondOutput = second.ForgetMemoryV2Continue(
                keyWidth,
                valueWidth,
                retentionFloor: 0.42f,
                actualState);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - before;
            long explicitStateBytes = checked(2L * matrixSize * sizeof(float));
            Assert.Equal(explicitStateBytes, transfers.HostToDeviceBytes);
            Assert.Equal(explicitStateBytes, transfers.DeviceToHostBytes);
            Assert.Equal(descriptor, firstOutput.Bfp8Quantization);
            Assert.Equal(descriptor, secondOutput.Bfp8Quantization);

            AssertClose(expected.FirstOutput, firstOutput.Data, 6e-2f);
            AssertClose(expected.SecondOutput, secondOutput.Data, 6e-2f);
            AssertClose(expected.FinalState, actualState, 4e-3f);
            first.InvalidateCudaBuffers();
            second.InvalidateCudaBuffers();
        });
    }

    [Fact]
    public void RecurrentStateStaysResidentAcrossBfp8ContinuationCalls()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int keyWidth = 17;
        const int valueWidth = 13;
        const int firstSequence = 3;
        const int secondSequence = 4;
        int projectionWidth = 2 * keyWidth + 3 * valueWidth;
        int matrixSize = keyWidth * valueWidth;
        Bfp8QuantizationDescriptor descriptor =
            Bfp8QuantizationDescriptor.Mix8_32;
        float[] firstValues = Values(
            firstSequence * projectionWidth,
            301,
            0.16f);
        float[] secondValues = Values(
            secondSequence * projectionWidth,
            373,
            0.14f);
        ContinueRun expected = Bf16ContinuationReference(
            firstValues,
            secondValues,
            new float[matrixSize],
            descriptor,
            firstSequence,
            secondSequence,
            projectionWidth,
            keyWidth,
            valueWidth);

        WithCuda(descriptor, () =>
        {
            using IDisposable noGrad = AutogradContext.NoGrad();
            using CudaInferenceScope inference = CudaInferenceScope.Begin(
                resetPool: true,
                clearPoolOnDispose: true);
            using var state = new ForgetMemoryRecurrentMemory(matrixSize);
            Tensor first = Tensor.FromBfp8(
                firstValues,
                [1, firstSequence, projectionWidth],
                descriptor);
            Tensor second = Tensor.FromBfp8(
                secondValues,
                [1, secondSequence, projectionWidth],
                descriptor);
            first.to(new TorchDevice(TensorDevice.Cuda, 0));
            second.to(new TorchDevice(TensorDevice.Cuda, 0));
            _ = first.EnsureCudaBfp8Buffer(0);
            _ = second.EnsureCudaBfp8Buffer(0);
            _ = state.EnsureCudaBuffer(0);
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;

            Tensor firstOutput = first.ForgetMemoryV2Continue(
                keyWidth,
                valueWidth,
                retentionFloor: 0.42f,
                state);
            Tensor secondOutput = second.ForgetMemoryV2Continue(
                keyWidth,
                valueWidth,
                retentionFloor: 0.42f,
                state);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - before;
            Assert.Equal(0, transfers.HostToDeviceBytes);
            Assert.Equal(0, transfers.DeviceToHostBytes);
            AssertClose(expected.FirstOutput, firstOutput.Data, 6e-2f);
            AssertClose(expected.SecondOutput, secondOutput.Data, 6e-2f);
            AssertClose(expected.FinalState, state.HostSnapshot(), 4e-3f);
            first.InvalidateCudaBuffers();
            second.InvalidateCudaBuffers();
        });
    }

    [Fact]
    public void RepeatedNoGradTailForwardReleasesEveryTransientAllocation()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int batch = 2;
        const int sequence = 7;
        const int keyWidth = 17;
        const int valueWidth = 13;
        int projectionWidth = 2 * keyWidth + 3 * valueWidth;
        Bfp8QuantizationDescriptor descriptor =
            Bfp8QuantizationDescriptor.Mix8_32;

        WithCuda(descriptor, () =>
        {
            Tensor input = Tensor.FromBfp8(
                Values(
                    batch * sequence * projectionWidth,
                    257,
                    0.17f),
                [batch, sequence, projectionWidth],
                descriptor);
            input.to(new TorchDevice(TensorDevice.Cuda, 0));
            _ = input.EnsureCudaBfp8Buffer(0);

            RunInference(input, keyWidth, valueWidth);
            NativeCudaAllocationTelemetry before =
                NativeCudaRuntime.AllocationTelemetry;
            for (int iteration = 0; iteration < 20; iteration++)
                RunInference(input, keyWidth, valueWidth);

            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
            NativeCudaAllocationTelemetry delta =
                NativeCudaRuntime.AllocationTelemetry - before;
            Assert.Equal(delta.AllocationCount, delta.FreeCount);
            Assert.Equal(delta.AllocationBytes, delta.FreeBytes);
            input.InvalidateCudaBuffers();
        });
    }

    private static void RunInference(
        Tensor input,
        int keyWidth,
        int valueWidth)
    {
        using IDisposable noGrad = AutogradContext.NoGrad();
        using CudaInferenceScope inference = CudaInferenceScope.Begin(
            resetPool: true,
            clearPoolOnDispose: true);
        _ = input.ForgetMemoryV2(
            keyWidth,
            valueWidth,
            retentionFloor: 0.31f);
    }

    private static ForgetMemoryRun Bf16Reference(
        float[] source,
        int[] shape,
        Bfp8QuantizationDescriptor descriptor,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        float[] seed)
        => WithCpu(() =>
        {
            Tensor input = Bf16FromBfp8(source, shape, descriptor);
            Tensor output = input.ForgetMemoryV2(
                keyWidth,
                valueWidth,
                retentionFloor);
            output.Backward(seed);
            return new ForgetMemoryRun(
                Quantize(output.Data, descriptor),
                descriptor == Bfp8QuantizationDescriptor.TensorWide
                    ? Quantize(
                        input.Grad,
                        Bfp8QuantizationDescriptor.TensorWide)
                    : input.Grad.ToArray());
        });

    private static ContinueRun Bf16ContinuationReference(
        float[] firstValues,
        float[] secondValues,
        float[] initialState,
        Bfp8QuantizationDescriptor descriptor,
        int firstSequence,
        int secondSequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth)
        => WithCpu(() =>
        {
            using IDisposable noGrad = AutogradContext.NoGrad();
            float[] state = initialState.ToArray();
            Tensor first = Bf16FromBfp8(
                firstValues,
                [1, firstSequence, projectionWidth],
                descriptor);
            Tensor second = Bf16FromBfp8(
                secondValues,
                [1, secondSequence, projectionWidth],
                descriptor);
            Tensor firstOutput = first.ForgetMemoryV2Continue(
                keyWidth,
                valueWidth,
                retentionFloor: 0.42f,
                state);
            Tensor secondOutput = second.ForgetMemoryV2Continue(
                keyWidth,
                valueWidth,
                retentionFloor: 0.42f,
                state);
            return new ContinueRun(
                Quantize(firstOutput.Data, descriptor),
                Quantize(secondOutput.Data, descriptor),
                state);
        });

    private static Tensor Bf16FromBfp8(
        float[] values,
        int[] shape,
        Bfp8QuantizationDescriptor descriptor)
    {
        Tensor encoded = Tensor.FromBfp8(values, shape, descriptor);
        return new Tensor(
            encoded.Data.ToArray(),
            shape,
            dtype: TensorDType.BFloat16);
    }

    private static float[] Quantize(
        IReadOnlyList<float> values,
        Bfp8QuantizationDescriptor descriptor)
    {
        float[] source = values.ToArray();
        Bfp8EncodedStorage encoded =
            Bfp8QuantizationCodec.Default.Encode(source, descriptor);
        var decoded = new float[source.Length];
        Bfp8QuantizationCodec.Default.Decode(
            encoded.Payload.Span,
            encoded.Scales.Span,
            descriptor,
            decoded);
        return decoded;
    }

    private static T WithCpu<T>(Func<T> action)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            return action();
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static void WithCuda(
        Bfp8QuantizationDescriptor descriptor,
        Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int previousDeviceIndex = Tensor.CudaDeviceIndex;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.CudaDeviceIndex = 0;
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(
                    descriptor.Granularity == Bfp8ScaleGranularity.Tensor
                        ? PrecisionPolicy.Bfp8
                        : PrecisionPolicy.Mix8_32);
            action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndex = previousDeviceIndex;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index =>
                MathF.Sin((index + offset) * 0.113f) * scale
                + MathF.Cos((index + offset) * 0.037f) * scale * 0.43f)
            .ToArray();

    private static void AssertClose(
        IReadOnlyList<float> expected,
        IReadOnlyList<float> actual,
        float tolerance)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int index = 0; index < expected.Count; index++)
        {
            Assert.InRange(
                MathF.Abs(expected[index] - actual[index]),
                0f,
                tolerance);
        }
    }

    private sealed record ForgetMemoryRun(float[] Output, float[] Gradient);

    private sealed record ContinueRun(
        float[] FirstOutput,
        float[] SecondOutput,
        float[] FinalState);
}
