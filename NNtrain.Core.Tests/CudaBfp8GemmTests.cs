using NNtrain;
using NNtrain.Cuda.Quantization;
using Xunit;

public sealed class CudaBfp8GemmTests
{
    [Fact]
    public void TensorWideAlignedMatMulUsesRealInt8RouteAndScaleProduct()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            const int m = 64;
            const int k = 128;
            const int n = 96;
            float[] leftValues = Values(m * k, 3, 0.35f);
            float[] rightValues = Values(k * n, 17, 0.27f);
            float[] expected = CpuMatMul(
                leftValues,
                rightValues,
                m,
                k,
                n,
                Bfp8QuantizationDescriptor.TensorWide).Output;

            Tensor left = Bfp8(leftValues, [m, k], tensorWide: true);
            Tensor right = Bfp8(rightValues, [k, n], tensorWide: true);
            MoveToCuda(left, right);
            NativeCudaTransferTelemetry transfersBefore =
                NativeCudaRuntime.TransferTelemetry;
            CudaBfp8GemmTelemetrySnapshot routeBefore =
                CudaBfp8GemmTelemetry.Snapshot;

            Tensor output = left.MatMul(right);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - transfersBefore;
            CudaBfp8GemmTelemetrySnapshot route =
                CudaBfp8GemmTelemetry.Snapshot - routeBefore;
            Assert.Equal(1, route.Int8TensorCoreExecutions);
            Assert.Equal(0, route.BFloat16FallbackExecutions);
            Assert.Equal(
                CudaBfp8GemmBackend.CublasLtInt8TensorCore,
                route.LastBackend);
            Assert.True(CudaBlasLtInt8.LastExecutionUsedInt8TensorCores);
            Assert.Equal(1, route.Int8LayoutTransformCacheMisses);
            Assert.Equal(0, transfers.HostToDeviceBytes);
            Assert.Equal(0, transfers.DeviceToHostBytes);
            Assert.Equal(Bfp8ScaleGranularity.Tensor,
                output.Bfp8Quantization!.Granularity);
            AssertClose(expected, output.Data, 2e-4f);

            CudaBfp8GemmTelemetrySnapshot reuseBefore =
                CudaBfp8GemmTelemetry.Snapshot;
            Tensor second = left.MatMul(right);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
            CudaBfp8GemmTelemetrySnapshot reuse =
                CudaBfp8GemmTelemetry.Snapshot - reuseBefore;
            Assert.Equal(1, reuse.Int8TensorCoreExecutions);
            Assert.Equal(0, reuse.Int8LayoutTransformCacheMisses);
            AssertClose(expected, second.Data, 2e-4f);

            DisposeCuda(second, output, left, right);
        });
    }

    [Fact]
    public void TensorWideAlignedLinearPreservesBiasAndReluOnInt8Route()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            const int rows = 64;
            const int inputWidth = 128;
            const int outputWidth = 96;
            float[] inputValues = Values(rows * inputWidth, 5, 0.42f);
            float[] weightValues = Values(outputWidth * inputWidth, 29, 0.31f);
            float[] biasValues = Enumerable.Range(0, outputWidth)
                .Select(index => (index - 3) * 0.09f)
                .ToArray();
            float[] expected = CpuLinear(
                inputValues,
                weightValues,
                biasValues,
                rows,
                inputWidth,
                outputWidth,
                Bfp8QuantizationDescriptor.TensorWide,
                applyRelu: true).Output;

            Tensor input = Bfp8(inputValues, [rows, inputWidth], tensorWide: true);
            Tensor weight = Bfp8(
                weightValues, [outputWidth, inputWidth], tensorWide: true);
            Tensor bias = Bfp8(biasValues, [outputWidth], tensorWide: true);
            MoveToCuda(input, weight, bias);
            CudaBfp8GemmTelemetrySnapshot before =
                CudaBfp8GemmTelemetry.Snapshot;

            Tensor output = input.LinearLastDim(weight, bias, applyRelu: true);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            CudaBfp8GemmTelemetrySnapshot route =
                CudaBfp8GemmTelemetry.Snapshot - before;
            Assert.Equal(1, route.Int8TensorCoreExecutions);
            Assert.Equal(0, route.BFloat16FallbackExecutions);
            AssertClose(expected, output.Data, 2e-4f);
            Assert.All(output.Data, value => Assert.True(value >= 0f));

            DisposeCuda(output, input, weight, bias);
        });
    }

    [Fact]
    public void BlockScaledAlignedMatMulNeverUsesSingleAlphaInt8()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            const int m = 8;
            const int k = 32;
            const int n = 8;
            Tensor left = Bfp8(Values(m * k, 7, 0.4f), [m, k]);
            Tensor right = Bfp8(Values(k * n, 31, 0.33f), [k, n]);
            MoveToCuda(left, right);
            NativeCudaTransferTelemetry transferBefore =
                NativeCudaRuntime.TransferTelemetry;
            CudaBfp8GemmTelemetrySnapshot routeBefore =
                CudaBfp8GemmTelemetry.Snapshot;

            Tensor output = left.MatMul(right);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            CudaBfp8GemmTelemetrySnapshot route =
                CudaBfp8GemmTelemetry.Snapshot - routeBefore;
            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - transferBefore;
            Assert.Equal(0, route.Int8TensorCoreExecutions);
            Assert.Equal(1, route.BFloat16FallbackExecutions);
            Assert.Equal(
                CudaBfp8GemmBackend.BFloat16Dequantize,
                route.LastBackend);
            Assert.Equal(0, transfers.HostToDeviceBytes);
            Assert.Equal(0, transfers.DeviceToHostBytes);
            Assert.Equal(Bfp8ScaleGranularity.Block,
                output.Bfp8Quantization!.Granularity);
            Assert.Equal(128, output.Bfp8Quantization.BlockSize);

            DisposeCuda(output, left, right);
        });
    }

    [Fact]
    public void Mix8LinearAndMatMulBackwardMatchCpuReference()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int rows = 8;
        const int k = 32;
        const int n = 8;
        float[] leftValues = Values(rows * k, 11, 0.24f);
        float[] rightValues = Values(k * n, 41, 0.21f);
        float[] biasValues = Enumerable.Range(0, n)
            .Select(index => 0.2f + index * 0.03f)
            .ToArray();
        float[] seed = Values(rows * n, 73, 0.17f);

        MatrixRun cpuMatMul = CpuMatMul(
            leftValues,
            rightValues,
            rows,
            k,
            n,
            Bfp8QuantizationDescriptor.Mix8_32,
            seed);
        LinearRun cpuLinear = CpuLinear(
            leftValues,
            rightValues,
            biasValues,
            rows,
            k,
            n,
            Bfp8QuantizationDescriptor.Mix8_32,
            applyRelu: false,
            seed);

        WithCuda(() =>
        {
            Tensor left = Bfp8(leftValues, [rows, k]);
            Tensor right = Bfp8(rightValues, [k, n]);
            MoveToCuda(left, right);
            Tensor matMul = left.MatMul(right);
            matMul.Backward(seed);
            AssertClose(cpuMatMul.Output, matMul.Data, 0.08f);
            AssertClose(cpuMatMul.LeftGradient, left.Grad, 0.08f);
            AssertClose(cpuMatMul.RightGradient, right.Grad, 0.08f);
            DisposeCuda(matMul, left, right);

            Tensor input = Bfp8(leftValues, [rows, k]);
            Tensor weight = Bfp8(rightValues, [n, k]);
            Tensor bias = Bfp8(biasValues, [n]);
            MoveToCuda(input, weight, bias);
            Tensor linear = input.LinearLastDim(weight, bias, applyRelu: false);
            linear.Backward(seed);
            AssertClose(cpuLinear.Output, linear.Data, 0.08f);
            AssertClose(cpuLinear.InputGradient, input.Grad, 0.08f);
            AssertClose(cpuLinear.WeightGradient, weight.Grad, 0.08f);
            AssertClose(cpuLinear.BiasGradient, bias.Grad, 0.08f);
            DisposeCuda(linear, input, weight, bias);
        });
    }

    [Fact]
    public void BFloat16DecodeCacheIsReusedByVersionAndDisposedWithTensor()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            const int m = 7;
            const int k = 31;
            const int n = 9;
            Tensor left = Bfp8(Values(m * k, 13, 0.3f), [m, k]);
            Tensor right = Bfp8(Values(k * n, 47, 0.2f), [k, n]);
            MoveToCuda(left, right);
            CudaBfp8GemmTelemetrySnapshot before =
                CudaBfp8GemmTelemetry.Snapshot;

            Tensor first = left.MatMul(right);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
            CudaBfp8GemmTelemetrySnapshot afterFirst =
                CudaBfp8GemmTelemetry.Snapshot;
            Tensor second = left.MatMul(right);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
            CudaBfp8GemmTelemetrySnapshot afterSecond =
                CudaBfp8GemmTelemetry.Snapshot;

            Assert.Equal(2,
                (afterFirst - before).BFloat16DecodeCacheMisses);
            Assert.Equal(0,
                (afterSecond - afterFirst).BFloat16DecodeCacheMisses);
            DisposeCuda(first, second, left, right);
        });
    }

    [Fact]
    public void PureBfp8BackwardRefusesSilentFloat32GradientPublication()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            const int m = 8;
            const int k = 32;
            const int n = 8;
            Tensor left = Bfp8(
                Values(m * k, 19, 0.22f), [m, k], tensorWide: true);
            Tensor right = Bfp8(
                Values(k * n, 53, 0.19f), [k, n], tensorWide: true);
            MoveToCuda(left, right);
            Tensor output = left.MatMul(right);

            NotSupportedException exception = Assert.Throws<NotSupportedException>(
                () => output.Backward(Values(m * n, 89, 0.1f)));
            Assert.Contains("publishing", exception.Message);
            Assert.Contains("mix8_32", exception.Message);

            DisposeCuda(output, left, right);
        });
    }

    private static MatrixRun CpuMatMul(
        float[] leftValues,
        float[] rightValues,
        int m,
        int k,
        int n,
        Bfp8QuantizationDescriptor descriptor,
        float[]? seed = null)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            Tensor left = Tensor.FromBfp8(leftValues, [m, k], descriptor);
            Tensor right = Tensor.FromBfp8(rightValues, [k, n], descriptor);
            Tensor output = left.MatMul(right);
            if (seed is not null)
                output.Backward(seed);
            return new MatrixRun(
                output.Data.ToArray(),
                seed is null ? [] : left.Grad.ToArray(),
                seed is null ? [] : right.Grad.ToArray());
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static LinearRun CpuLinear(
        float[] inputValues,
        float[] weightValues,
        float[] biasValues,
        int rows,
        int inputWidth,
        int outputWidth,
        Bfp8QuantizationDescriptor descriptor,
        bool applyRelu,
        float[]? seed = null)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            Tensor input = Tensor.FromBfp8(
                inputValues, [rows, inputWidth], descriptor);
            Tensor weight = Tensor.FromBfp8(
                weightValues, [outputWidth, inputWidth], descriptor);
            Tensor bias = Tensor.FromBfp8(
                biasValues, [outputWidth], descriptor);
            Tensor output = input.LinearLastDim(weight, bias, applyRelu);
            if (seed is not null)
                output.Backward(seed);
            return new LinearRun(
                output.Data.ToArray(),
                seed is null ? [] : input.Grad.ToArray(),
                seed is null ? [] : weight.Grad.ToArray(),
                seed is null ? [] : bias.Grad.ToArray());
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static Tensor Bfp8(
        float[] values,
        int[] shape,
        bool tensorWide = false)
        => Tensor.FromBfp8(
            values,
            shape,
            tensorWide
                ? Bfp8QuantizationDescriptor.TensorWide
                : Bfp8QuantizationDescriptor.Mix8_32);

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index =>
                MathF.Sin((index + offset) * 0.173f) * scale
                + MathF.Cos((index + offset) * 0.071f) * scale * 0.37f)
            .ToArray();

    private static void MoveToCuda(params Tensor[] tensors)
    {
        foreach (Tensor tensor in tensors)
            tensor.to(new TorchDevice(TensorDevice.Cuda, 0));
    }

    private static void DisposeCuda(params Tensor[] tensors)
    {
        foreach (Tensor tensor in tensors)
            tensor.InvalidateCudaBuffers();
    }

    private static void WithCuda(Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
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

    private sealed record MatrixRun(
        float[] Output,
        float[] LeftGradient,
        float[] RightGradient);

    private sealed record LinearRun(
        float[] Output,
        float[] InputGradient,
        float[] WeightGradient,
        float[] BiasGradient);
}
