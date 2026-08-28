using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaBfp8LossDropoutTests
{
    [Theory]
    [InlineData(false, 1, 257)]
    [InlineData(true, 3, 515)]
    public void ResidentCrossEntropyMatchesQuantizedBf16ReferenceWithTailRows(
        bool blockScaled,
        int rows,
        int columns)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        Bfp8QuantizationDescriptor descriptor = Descriptor(blockScaled);
        int[] shape = rows == 1 ? [columns] : [rows, columns];
        float[] values = Values(rows * columns, 17, 2.1f);
        int[] labels = Enumerable.Range(0, rows)
            .Select(row => row == 1 ? -1 : (row * 97 + 13) % columns)
            .ToArray();
        const float smoothing = 0.13f;
        LossRun expected = CpuCrossEntropy(
            values,
            shape,
            descriptor,
            labels,
            smoothing);

        WithCuda(descriptor, () =>
        {
            Tensor logits = Tensor.FromBfp8(values, shape, descriptor);
            MakeResident(logits);
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;
            CudaBfp8GemmTelemetrySnapshot gemmBefore =
                CudaBfp8GemmTelemetry.Snapshot;

            Tensor loss = logits.CrossEntropyWithLogits(
                labels,
                smoothing);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - before;
            CudaBfp8GemmTelemetrySnapshot gemm =
                CudaBfp8GemmTelemetry.Snapshot - gemmBefore;
            Assert.Equal(labels.Length * sizeof(int),
                transfers.HostToDeviceBytes);
            Assert.Equal(0, transfers.DeviceToHostBytes);
            Assert.Equal(0, gemm.BFloat16DecodeCacheMisses);
            Assert.Equal(TensorDType.Float32, loss.DType);
            Assert.InRange(
                MathF.Abs(expected.Loss - loss.item()),
                0f,
                4e-3f);

            loss.BackwardAndRelease();
            AssertClose(expected.Gradient, logits.Grad, 4e-3f);
            Release(loss, logits);
        });
    }

    [Theory]
    [InlineData(false, 257)]
    [InlineData(true, 515)]
    public void ResidentDropoutReusesMaskAndPreservesBfp8Descriptor(
        bool blockScaled,
        int columns)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int rows = 3;
        const float probability = 0.31f;
        const int randomSeed = 73;
        Bfp8QuantizationDescriptor descriptor = Descriptor(blockScaled);
        float[] values = Values(rows * columns, 31, 0.9f);
        float[] seed = Values(rows * columns, 43, 0.071f);
        UnaryRun expected = CpuDropout(
            values,
            [rows, columns],
            descriptor,
            probability,
            randomSeed,
            seed);

        WithCuda(descriptor, () =>
        {
            Tensor input = Tensor.FromBfp8(
                values, [rows, columns], descriptor);
            MakeResident(input);
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;
            CudaBfp8GemmTelemetrySnapshot gemmBefore =
                CudaBfp8GemmTelemetry.Snapshot;

            Tensor output = input.Dropout(
                probability,
                new Random(randomSeed));
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - before;
            CudaBfp8GemmTelemetrySnapshot gemm =
                CudaBfp8GemmTelemetry.Snapshot - gemmBefore;
            Assert.Equal(0, transfers.HostToDeviceBytes);
            Assert.Equal(0, transfers.DeviceToHostBytes);
            Assert.Equal(0, gemm.BFloat16DecodeCacheMisses);
            Assert.Equal(TensorDType.Bfp8, output.DType);
            Assert.Equal(descriptor, output.Bfp8Quantization);
            AssertClose(expected.Output, output.Data, 2e-3f);

            output.BackwardAndRelease(seed);
            AssertClose(expected.Gradient, input.Grad, 3e-3f);
            Release(output, input);
        });
    }

    [Theory]
    [InlineData(false, false, 257)]
    [InlineData(true, false, 515)]
    [InlineData(false, true, 129)]
    [InlineData(true, true, 384)]
    public void ResidentResidualDropoutMatchesBf16ForDistinctAndSharedParents(
        bool blockScaled,
        bool sameParent,
        int columns)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int rows = 3;
        const float probability = 0.27f;
        const int randomSeed = 109;
        Bfp8QuantizationDescriptor descriptor = Descriptor(blockScaled);
        float[] residualValues = Values(rows * columns, 61, 0.72f);
        float[] branchValues = sameParent
            ? residualValues
            : Values(rows * columns, 83, 0.57f);
        float[] seed = Values(rows * columns, 101, 0.043f);
        BinaryRun expected = CpuAddDropout(
            residualValues,
            branchValues,
            [rows, columns],
            descriptor,
            probability,
            randomSeed,
            sameParent,
            seed);

        WithCuda(descriptor, () =>
        {
            Tensor residual = Tensor.FromBfp8(
                residualValues, [rows, columns], descriptor);
            Tensor branch = sameParent
                ? residual
                : Tensor.FromBfp8(
                    branchValues, [rows, columns], descriptor);
            MakeResident(residual, branch);
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;

            Tensor output = residual.AddDropout(
                branch,
                probability,
                new Random(randomSeed));
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - before;
            Assert.Equal(0, transfers.HostToDeviceBytes);
            Assert.Equal(0, transfers.DeviceToHostBytes);
            Assert.Equal(descriptor, output.Bfp8Quantization);
            AssertClose(expected.Output, output.Data, 3e-3f);

            output.BackwardAndRelease(seed);
            AssertClose(
                expected.LeftGradient,
                residual.Grad,
                4e-3f);
            if (!sameParent)
            {
                AssertClose(
                    expected.RightGradient,
                    branch.Grad,
                    4e-3f);
            }
            Release(output, residual, branch);
        });
    }

    [Theory]
    [InlineData(false, 257)]
    [InlineData(true, 515)]
    public void ResidentPlainResidualAddHasNoHostFallback(
        bool blockScaled,
        int columns)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int rows = 3;
        Bfp8QuantizationDescriptor descriptor = Descriptor(blockScaled);
        float[] leftValues = Values(rows * columns, 127, 0.81f);
        float[] rightValues = Values(rows * columns, 149, 0.64f);
        float[] seed = Values(rows * columns, 167, 0.039f);
        BinaryRun expected = CpuAdd(
            leftValues,
            rightValues,
            [rows, columns],
            descriptor,
            seed);

        WithCuda(descriptor, () =>
        {
            Tensor left = Tensor.FromBfp8(
                leftValues, [rows, columns], descriptor);
            Tensor right = Tensor.FromBfp8(
                rightValues, [rows, columns], descriptor);
            MakeResident(left, right);
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;

            Tensor output = left + right;
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - before;
            Assert.Equal(0, transfers.HostToDeviceBytes);
            Assert.Equal(0, transfers.DeviceToHostBytes);
            Assert.Equal(descriptor, output.Bfp8Quantization);
            AssertClose(expected.Output, output.Data, 3e-3f);

            output.BackwardAndRelease(seed);
            AssertClose(expected.LeftGradient, left.Grad, 3e-3f);
            AssertClose(expected.RightGradient, right.Grad, 3e-3f);
            Release(output, left, right);
        });
    }

    [Fact]
    public void RepeatedNoGradLossAndDropoutReleaseEveryTransientAllocation()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int rows = 3;
        const int columns = 515;
        int[] labels = [17, -1, 503];
        Bfp8QuantizationDescriptor descriptor =
            Bfp8QuantizationDescriptor.Mix8_32;

        WithCuda(descriptor, () =>
        {
            Tensor left = Tensor.FromBfp8(
                Values(rows * columns, 181, 0.73f),
                [rows, columns],
                descriptor);
            Tensor right = Tensor.FromBfp8(
                Values(rows * columns, 193, 0.51f),
                [rows, columns],
                descriptor);
            MakeResident(left, right);

            RunInference(left, right, labels);
            NativeCudaAllocationTelemetry before =
                NativeCudaRuntime.AllocationTelemetry;

            for (int iteration = 0; iteration < 16; iteration++)
                RunInference(left, right, labels);

            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
            NativeCudaAllocationTelemetry delta =
                NativeCudaRuntime.AllocationTelemetry - before;
            Assert.Equal(delta.AllocationCount, delta.FreeCount);
            Assert.Equal(delta.AllocationBytes, delta.FreeBytes);
            Release(left, right);
        });
    }

    [Fact]
    public void RepeatedDropoutLossBackwardReleasesSavedContexts()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int rows = 3;
        const int columns = 257;
        int[] labels = [19, 127, 251];
        Bfp8QuantizationDescriptor descriptor =
            Bfp8QuantizationDescriptor.Mix8_32;

        WithCuda(descriptor, () =>
        {
            Tensor input = Tensor.FromBfp8(
                Values(rows * columns, 229, 0.67f),
                [rows, columns],
                descriptor);
            MakeResident(input);

            RunTrainingGraph(input, labels, 241);
            NativeCudaAllocationTelemetry before =
                NativeCudaRuntime.AllocationTelemetry;

            for (int iteration = 0; iteration < 12; iteration++)
            {
                input.ZeroGrad();
                RunTrainingGraph(input, labels, 251 + iteration);
            }

            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
            NativeCudaAllocationTelemetry delta =
                NativeCudaRuntime.AllocationTelemetry - before;
            Assert.Equal(delta.AllocationCount, delta.FreeCount);
            Assert.Equal(delta.AllocationBytes, delta.FreeBytes);
            Release(input);
        });
    }

    private static void RunInference(
        Tensor left,
        Tensor right,
        int[] labels)
    {
        using IDisposable noGrad = AutogradContext.NoGrad();
        using CudaInferenceScope scope = CudaInferenceScope.Begin(
            resetPool: true,
            clearPoolOnDispose: true);
        _ = left.Dropout(0.23f, new Random(211));
        _ = left.AddDropout(right, 0.19f, new Random(223));
        _ = left + right;
        _ = left.CrossEntropyWithLogits(labels, labelSmoothing: 0.07f);
    }

    private static void RunTrainingGraph(
        Tensor input,
        int[] labels,
        int randomSeed)
    {
        Tensor dropped = input.Dropout(0.21f, new Random(randomSeed));
        Tensor loss = dropped.CrossEntropyWithLogits(
            labels,
            labelSmoothing: 0.09f);
        loss.BackwardAndRelease();
        loss.InvalidateCudaBuffers();
        dropped.InvalidateCudaBuffers();
    }

    private static LossRun CpuCrossEntropy(
        float[] values,
        int[] shape,
        Bfp8QuantizationDescriptor descriptor,
        int[] labels,
        float smoothing)
        => WithCpu(() =>
        {
            Tensor logits = Bf16FromBfp8(values, shape, descriptor);
            Tensor loss = logits.CrossEntropyWithLogits(labels, smoothing);
            float value = loss.item();
            loss.BackwardAndRelease();
            float[] gradient = logits.Grad.ToArray();
            if (descriptor.Granularity == Bfp8ScaleGranularity.Tensor)
            {
                gradient = Quantize(
                    gradient,
                    Bfp8QuantizationDescriptor.TensorWide);
            }
            return new LossRun(value, gradient);
        });

    private static UnaryRun CpuDropout(
        float[] values,
        int[] shape,
        Bfp8QuantizationDescriptor descriptor,
        float probability,
        int randomSeed,
        float[] seed)
        => WithCpu(() =>
        {
            Tensor input = Bf16FromBfp8(values, shape, descriptor);
            Tensor output = input.Dropout(
                probability,
                new Random(randomSeed));
            float[] expectedOutput = Quantize(output.Data, descriptor);
            output.BackwardAndRelease(seed);
            float[] gradient = input.Grad.ToArray();
            if (descriptor.Granularity == Bfp8ScaleGranularity.Tensor)
            {
                gradient = Quantize(
                    gradient,
                    Bfp8QuantizationDescriptor.TensorWide);
            }
            return new UnaryRun(expectedOutput, gradient);
        });

    private static BinaryRun CpuAddDropout(
        float[] leftValues,
        float[] rightValues,
        int[] shape,
        Bfp8QuantizationDescriptor descriptor,
        float probability,
        int randomSeed,
        bool sameParent,
        float[] seed)
        => WithCpu(() =>
        {
            Tensor left = Bf16FromBfp8(
                leftValues, shape, descriptor);
            Tensor right = sameParent
                ? left
                : Bf16FromBfp8(rightValues, shape, descriptor);
            Tensor output = left.AddDropout(
                right,
                probability,
                new Random(randomSeed));
            float[] expectedOutput = Quantize(output.Data, descriptor);
            output.BackwardAndRelease(seed);
            float[] leftGradient = left.Grad.ToArray();
            float[] rightGradient = sameParent
                ? leftGradient
                : right.Grad.ToArray();
            if (descriptor.Granularity == Bfp8ScaleGranularity.Tensor)
            {
                leftGradient = Quantize(
                    leftGradient,
                    Bfp8QuantizationDescriptor.TensorWide);
                rightGradient = sameParent
                    ? leftGradient
                    : Quantize(
                        rightGradient,
                        Bfp8QuantizationDescriptor.TensorWide);
            }
            return new BinaryRun(
                expectedOutput,
                leftGradient,
                rightGradient);
        });

    private static BinaryRun CpuAdd(
        float[] leftValues,
        float[] rightValues,
        int[] shape,
        Bfp8QuantizationDescriptor descriptor,
        float[] seed)
        => WithCpu(() =>
        {
            Tensor left = Bf16FromBfp8(
                leftValues, shape, descriptor);
            Tensor right = Bf16FromBfp8(
                rightValues, shape, descriptor);
            Tensor output = left + right;
            float[] expectedOutput = Quantize(output.Data, descriptor);
            output.BackwardAndRelease(seed);
            float[] leftGradient = left.Grad.ToArray();
            float[] rightGradient = right.Grad.ToArray();
            if (descriptor.Granularity == Bfp8ScaleGranularity.Tensor)
            {
                leftGradient = Quantize(
                    leftGradient,
                    Bfp8QuantizationDescriptor.TensorWide);
                rightGradient = Quantize(
                    rightGradient,
                    Bfp8QuantizationDescriptor.TensorWide);
            }
            return new BinaryRun(
                expectedOutput,
                leftGradient,
                rightGradient);
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
        Bfp8EncodedStorage encoded = Bfp8QuantizationCodec.Default.Encode(
            values.ToArray(), descriptor);
        float[] decoded = new float[values.Count];
        Bfp8QuantizationCodec.Default.Decode(
            encoded.Payload.Span,
            encoded.Scales.Span,
            descriptor,
            decoded);
        return decoded;
    }

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index =>
                MathF.Sin((index + offset) * 0.131f) * scale
                + MathF.Cos((index + offset) * 0.047f) * scale * 0.29f)
            .ToArray();

    private static Bfp8QuantizationDescriptor Descriptor(bool blockScaled)
        => blockScaled
            ? Bfp8QuantizationDescriptor.Mix8_32
            : Bfp8QuantizationDescriptor.TensorWide;

    private static void MakeResident(params Tensor[] tensors)
    {
        foreach (Tensor tensor in tensors.Distinct())
        {
            tensor.to(new TorchDevice(TensorDevice.Cuda, 0));
            _ = tensor.EnsureCudaBfp8Buffer(0);
        }
    }

    private static void Release(params Tensor[] tensors)
    {
        foreach (Tensor tensor in tensors.Distinct())
            tensor.InvalidateCudaBuffers();
    }

    private static T WithCpu<T>(Func<T> action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            return action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    private static void WithCuda(
        Bfp8QuantizationDescriptor descriptor,
        Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
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
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

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

    private sealed record LossRun(float Loss, float[] Gradient);

    private sealed record UnaryRun(float[] Output, float[] Gradient);

    private sealed record BinaryRun(
        float[] Output,
        float[] LeftGradient,
        float[] RightGradient);
}
