using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Quantization;
using NNtrain.Runtime.Execution;
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
            CudaBfp8GemmTelemetrySnapshot linearBefore =
                CudaBfp8GemmTelemetry.Snapshot;
            Tensor linear = input.LinearLastDim(weight, bias, applyRelu: false);
            linear.Backward(seed);
            CudaBfp8GemmTelemetrySnapshot linearTelemetry =
                CudaBfp8GemmTelemetry.Snapshot - linearBefore;
            // Forward decodes input, weight, and bias exactly once. A
            // non-ReLU backward must not decode the very large output merely
            // to encode its gradient.
            Assert.Equal(3, linearTelemetry.BFloat16DecodeCacheMisses);
            AssertClose(cpuLinear.Output, linear.Data, 0.08f);
            AssertClose(cpuLinear.InputGradient, input.Grad, 0.08f);
            AssertClose(cpuLinear.WeightGradient, weight.Grad, 0.08f);
            AssertClose(cpuLinear.BiasGradient, bias.Grad, 0.08f);
            DisposeCuda(linear, input, weight, bias);
        });
    }

    [Fact]
    public void Mix8ReluBackwardUsesPayloadMaskWithoutOutputDecode()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int rows = 8;
        const int inputWidth = 32;
        const int outputWidth = 8;
        float[] inputValues = Values(rows * inputWidth, 101, 0.31f);
        float[] weightValues = Values(
            outputWidth * inputWidth, 137, 0.27f);
        float[] biasValues = Values(outputWidth, 173, 0.04f);
        float[] seed = Values(rows * outputWidth, 191, 0.13f);
        LinearRun cpu = CpuLinear(
            inputValues,
            weightValues,
            biasValues,
            rows,
            inputWidth,
            outputWidth,
            Bfp8QuantizationDescriptor.Mix8_32,
            applyRelu: true,
            seed);

        WithCuda(() =>
        {
            Tensor input = Bfp8(inputValues, [rows, inputWidth]);
            Tensor weight = Bfp8(
                weightValues, [outputWidth, inputWidth]);
            Tensor bias = Bfp8(biasValues, [outputWidth]);
            MoveToCuda(input, weight, bias);
            CudaBfp8GemmTelemetrySnapshot decodeBefore =
                CudaBfp8GemmTelemetry.Snapshot;
            CudaBlasLtTelemetrySnapshot blasBefore = CudaBlasLt.Telemetry;

            Tensor output = input.LinearLastDim(
                weight, bias, applyRelu: true);
            output.Backward(seed);
            CudaBfp8GemmTelemetrySnapshot decode =
                CudaBfp8GemmTelemetry.Snapshot - decodeBefore;
            CudaBlasLtTelemetrySnapshot blas =
                CudaBlasLt.Telemetry - blasBefore;

            Assert.Equal(3, decode.BFloat16DecodeCacheMisses);
            Assert.Equal(2, blas.AccumulatingBackwardCublasExecutions);
            AssertClose(cpu.Output, output.Data, 0.08f);
            AssertClose(cpu.InputGradient, input.Grad, 0.08f);
            AssertClose(cpu.WeightGradient, weight.Grad, 0.08f);
            AssertClose(cpu.BiasGradient, bias.Grad, 0.08f);
            DisposeCuda(output, input, weight, bias);
        });
    }

    [Fact]
    public void ExclusiveMix8FfnDirectGradientMatchesLegacyAndKeepsLeavesFloat32()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int rows = 8;
        const int width = 32;
        const int hiddenWidth = 64;
        float[] inputValues = Values(rows * width, 1201, 0.13f);
        float[] firstWeightValues = Values(
            hiddenWidth * width, 1213, 0.035f);
        float[] firstBiasValues = Enumerable.Range(0, hiddenWidth)
            .Select(index => index % 2 == 0 ? -1f : 0.15f)
            .ToArray();
        float[] secondWeightValues = Values(
            width * hiddenWidth, 1229, 0.041f);
        float[] secondBiasValues = Values(width, 1237, 0.02f);
        float[] seed = Values(rows * width, 1249, 0.11f);

        FfnRun cpu = RunCpuFfn(
            inputValues,
            firstWeightValues,
            firstBiasValues,
            secondWeightValues,
            secondBiasValues,
            seed,
            rows,
            width,
            hiddenWidth);
        FfnRun legacy = WithCuda(() => RunCudaFfn(
            inputValues,
            firstWeightValues,
            firstBiasValues,
            secondWeightValues,
            secondBiasValues,
            seed,
            rows,
            width,
            hiddenWidth,
            exclusive: true,
            disableDirect: true));
        FfnRun direct = WithCuda(() => RunCudaFfn(
            inputValues,
            firstWeightValues,
            firstBiasValues,
            secondWeightValues,
            secondBiasValues,
            seed,
            rows,
            width,
            hiddenWidth,
            exclusive: true,
            disableDirect: false));
        FfnRun general = WithCuda(() => RunCudaFfn(
            inputValues,
            firstWeightValues,
            firstBiasValues,
            secondWeightValues,
            secondBiasValues,
            seed,
            rows,
            width,
            hiddenWidth,
            exclusive: false,
            disableDirect: false));

        Assert.Equal(0, legacy.Telemetry.DirectBFloat16FfnInputGradientExecutions);
        Assert.Equal(0, legacy.Telemetry.Bfp8ReluBFloat16MaskExecutions);
        Assert.Equal(1, direct.Telemetry.DirectBFloat16FfnInputGradientExecutions);
        Assert.Equal(1, direct.Telemetry.Bfp8ReluBFloat16MaskExecutions);
        Assert.Equal(0, general.Telemetry.DirectBFloat16FfnInputGradientExecutions);
        Assert.Equal(0, general.Telemetry.Bfp8ReluBFloat16MaskExecutions);
        Assert.True(direct.HiddenHasBFloat16Gradient);
        Assert.False(direct.InputHasBFloat16Gradient);
        Assert.False(direct.AnyParameterHasBFloat16Gradient);

        AssertFfnClose(legacy, direct, 0.025f);
        AssertFfnClose(cpu, direct, 0.10f);
        for (int hidden = 0; hidden < hiddenWidth; hidden += 2)
        {
            Assert.Equal(0f, direct.FirstBiasGradient[hidden]);
            int rowStart = hidden * width;
            for (int column = 0; column < width; column++)
                Assert.Equal(0f, direct.FirstWeightGradient[rowStart + column]);
        }
    }

    [Fact]
    public void ExclusiveMix8FfnFailureReturnsEveryTransientLaneLease()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            const int rows = 4;
            const int width = 32;
            const int hiddenWidth = 64;
            CudaExecutionLane lane = CudaExecutionLaneFactory.Create(0);
            using var execution = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet([0]),
                    Precision = PrecisionPolicy.Mix8_32,
                },
                [lane]);
            using IDisposable sessionScope = execution.Enter();
            Tensor input = Bfp8(
                Values(rows * width, 1301, 0.11f), [rows, width]);
            Tensor firstWeight = Bfp8(
                Values(hiddenWidth * width, 1307, 0.03f),
                [hiddenWidth, width]);
            Tensor firstBias = Bfp8(
                Values(hiddenWidth, 1319, 0.02f), [hiddenWidth]);
            Tensor secondWeight = Bfp8(
                Values(width * hiddenWidth, 1321, 0.03f),
                [width, hiddenWidth]);
            Tensor secondBias = Bfp8(
                Values(width, 1327, 0.02f), [width]);
            MoveToCuda(
                input,
                firstWeight,
                firstBias,
                secondWeight,
                secondBias);
            Tensor hidden =
                input.LinearLastDimReluExclusiveBfp8OutputGradient(
                    firstWeight,
                    firstBias);
            Tensor projected =
                hidden.LinearLastDimExclusiveBfp8InputGradient(
                    secondWeight,
                    secondBias);
            _ = projected.EnsureCudaGradientBuffer(0);
            var before = lane.Memory.Telemetry;
            using IDisposable policy = CudaDispatchPolicy.Push(
                CudaDispatchPolicy.Defaults with
                {
                    ThrowAfterDirectBfp8FfnGradientAllocationsForTest = true,
                });

            InvalidOperationException failure = Assert.Throws<
                InvalidOperationException>(() => projected.Backward(
                    Values(rows * width, 1331, 0.07f)));
            Assert.Contains("Injected failure", failure.Message);
            lane.SynchronizeComputeStream();
            var after = lane.Memory.Telemetry;
            // Backward owns one new persistent FP32 output-gradient seed.
            // Every BF16 encoded/direct/decode transient acquired before the
            // injected failure must already have left the active set.
            Assert.Equal(
                before.ActiveAllocationCount + 1,
                after.ActiveAllocationCount);
            Assert.Equal(
                before.ActiveBytes + checked((long)projected.Numel * sizeof(float)),
                after.ActiveBytes);

            DisposeCuda(
                projected,
                hidden,
                input,
                firstWeight,
                firstBias,
                secondWeight,
                secondBias);
        });
    }

    [Fact]
    public void Mix8ProductionHiddenTailUsesBFloat16HmmaPlan()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            const int rows = 8;
            const int inputWidth = 512;
            const int outputWidth = 1538;
            Tensor input = Bfp8(
                Values(rows * inputWidth, 211, 0.08f),
                [rows, inputWidth]);
            Tensor weight = Bfp8(
                Values(outputWidth * inputWidth, 223, 0.05f),
                [outputWidth, inputWidth]);
            Tensor bias = Bfp8(
                Values(outputWidth, 227, 0.02f),
                [outputWidth]);
            MoveToCuda(input, weight, bias);
            CudaBlasLtTelemetrySnapshot before = CudaBlasLt.Telemetry;

            Tensor output = input.LinearLastDim(
                weight, bias, applyRelu: true);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
            CudaBlasLtTelemetrySnapshot telemetry =
                CudaBlasLt.Telemetry - before;

            Assert.Equal(1, telemetry.ForwardTensorCoreExecutions);
            Assert.NotEqual(
                0UL,
                telemetry.LastForwardNumericalImplementationFlags & 0x02UL);
            Assert.Equal([rows, outputWidth], output.Shape);
            DisposeCuda(output, input, weight, bias);
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
    public void NonLeafBFloat16DecodeIsTransientWhileLeafOperandIsCached()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            const int m = 7;
            const int k = 31;
            const int n = 9;
            Tensor source = Bfp8(Values(m * k, 13, 0.3f), [m, k]);
            Tensor zero = Bfp8(new float[m * k], [m, k]);
            Tensor right = Bfp8(Values(k * n, 47, 0.2f), [k, n]);
            MoveToCuda(source, zero, right);
            Tensor activation = source + zero;
            CudaBfp8GemmTelemetrySnapshot before =
                CudaBfp8GemmTelemetry.Snapshot;

            Tensor first = activation.MatMul(right);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
            CudaBfp8GemmTelemetrySnapshot afterFirst =
                CudaBfp8GemmTelemetry.Snapshot;
            Tensor second = activation.MatMul(right);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
            CudaBfp8GemmTelemetrySnapshot afterSecond =
                CudaBfp8GemmTelemetry.Snapshot;

            // The first pass decodes the transient activation and the leaf
            // weight. Only the activation is decoded again on the second pass.
            Assert.Equal(2,
                (afterFirst - before).BFloat16DecodeCacheMisses);
            Assert.Equal(1,
                (afterSecond - afterFirst).BFloat16DecodeCacheMisses);
            DisposeCuda(first, second, activation, source, zero, right);
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PureBfp8BackwardPublishesLeavesWithFiniteAndNormScalars(
        bool releaseGraph)
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
            float[] seed = Values(m * n, 89, 0.1f);
            MatrixRun cpu = CpuMatMul(
                left.Data.ToArray(),
                right.Data.ToArray(),
                m,
                k,
                n,
                Bfp8QuantizationDescriptor.TensorWide,
                seed);
            MoveToCuda(left, right);
            Tensor output = left.MatMul(right);
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;

            if (releaseGraph)
                output.BackwardAndRelease(seed);
            else
                output.Backward(seed);

            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - before;
            Assert.Equal(
                sizeof(int) + sizeof(double),
                transfers.DeviceToHostBytes);
            Assert.True(left.HasAuthoritativeCudaBfp8Gradient);
            Assert.True(right.HasAuthoritativeCudaBfp8Gradient);
            AssertClose(cpu.LeftGradient, left.Grad, 0.01f);
            AssertClose(cpu.RightGradient, right.Grad, 0.01f);

            DisposeCuda(output, left, right);
        });
    }

    [Fact]
    public void PureBfp8BackwardRejectsDeviceNonFiniteOncePerGraph()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            const int m = 8;
            const int k = 32;
            const int n = 8;
            Tensor left = Bfp8(
                Values(m * k, 23, 0.2f), [m, k], tensorWide: true);
            Tensor right = Bfp8(
                Values(k * n, 61, 0.17f), [k, n], tensorWide: true);
            MoveToCuda(left, right);
            Tensor output = left.MatMul(right);
            float[] seed = Values(m * n, 97, 0.1f);
            seed[^1] = float.NaN;
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;

            InvalidOperationException exception =
                Assert.Throws<InvalidOperationException>(
                    () => output.Backward(seed));

            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - before;
            Assert.Equal(
                sizeof(int) + sizeof(double),
                transfers.DeviceToHostBytes);
            Assert.Contains("Non-finite CUDA gradient", exception.Message);
            Assert.Contains("device 0", exception.Message);
            Assert.False(left.HasAuthoritativeCudaBfp8Gradient);
            Assert.False(right.HasAuthoritativeCudaBfp8Gradient);

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

    private static FfnRun RunCpuFfn(
        float[] inputValues,
        float[] firstWeightValues,
        float[] firstBiasValues,
        float[] secondWeightValues,
        float[] secondBiasValues,
        float[] seed,
        int rows,
        int width,
        int hiddenWidth)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            Tensor input = Bfp8(inputValues, [rows, width]);
            Tensor firstWeight = Bfp8(
                firstWeightValues, [hiddenWidth, width]);
            Tensor firstBias = Bfp8(firstBiasValues, [hiddenWidth]);
            Tensor secondWeight = Bfp8(
                secondWeightValues, [width, hiddenWidth]);
            Tensor secondBias = Bfp8(secondBiasValues, [width]);
            Tensor hidden = input.LinearLastDim(
                firstWeight, firstBias, applyRelu: true);
            Tensor projected = hidden.LinearLastDim(
                secondWeight, secondBias, applyRelu: false);
            Tensor output = projected + input;
            output.Backward(seed);
            return SnapshotFfn(
                output,
                hidden,
                input,
                firstWeight,
                firstBias,
                secondWeight,
                secondBias,
                default);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static FfnRun RunCudaFfn(
        float[] inputValues,
        float[] firstWeightValues,
        float[] firstBiasValues,
        float[] secondWeightValues,
        float[] secondBiasValues,
        float[] seed,
        int rows,
        int width,
        int hiddenWidth,
        bool exclusive,
        bool disableDirect)
    {
        Tensor input = Bfp8(inputValues, [rows, width]);
        Tensor firstWeight = Bfp8(
            firstWeightValues, [hiddenWidth, width]);
        Tensor firstBias = Bfp8(firstBiasValues, [hiddenWidth]);
        Tensor secondWeight = Bfp8(
            secondWeightValues, [width, hiddenWidth]);
        Tensor secondBias = Bfp8(secondBiasValues, [width]);
        MoveToCuda(input, firstWeight, firstBias, secondWeight, secondBias);
        using IDisposable policy = CudaDispatchPolicy.Push(
            CudaDispatchPolicy.Defaults with
            {
                DisableDirectBfp8FfnGradient = disableDirect,
            });
        CudaBfp8GemmTelemetrySnapshot before =
            CudaBfp8GemmTelemetry.Snapshot;
        Tensor hidden = exclusive
            ? input.LinearLastDimReluExclusiveBfp8OutputGradient(
                firstWeight, firstBias)
            : input.LinearLastDim(
                firstWeight, firstBias, applyRelu: true);
        Tensor projected = exclusive
            ? hidden.LinearLastDimExclusiveBfp8InputGradient(
                secondWeight, secondBias)
            : hidden.LinearLastDim(
                secondWeight, secondBias, applyRelu: false);
        Tensor output = projected + input;
        output.Backward(seed);
        ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
        CudaBfp8GemmTelemetrySnapshot telemetry =
            CudaBfp8GemmTelemetry.Snapshot - before;
        FfnRun run = SnapshotFfn(
            output,
            hidden,
            input,
            firstWeight,
            firstBias,
            secondWeight,
            secondBias,
            telemetry);
        DisposeCuda(
            output,
            projected,
            hidden,
            input,
            firstWeight,
            firstBias,
            secondWeight,
            secondBias);
        return run;
    }

    private static FfnRun SnapshotFfn(
        Tensor output,
        Tensor hidden,
        Tensor input,
        Tensor firstWeight,
        Tensor firstBias,
        Tensor secondWeight,
        Tensor secondBias,
        CudaBfp8GemmTelemetrySnapshot telemetry)
    {
        bool hiddenBFloat16 = hidden.HasAuthoritativeCudaBFloat16Gradient;
        bool inputBFloat16 = input.HasAuthoritativeCudaBFloat16Gradient;
        bool parameterBFloat16 = new[]
        {
            firstWeight,
            firstBias,
            secondWeight,
            secondBias,
        }.Any(tensor => tensor.HasAuthoritativeCudaBFloat16Gradient);
        return new FfnRun(
            output.Data.ToArray(),
            input.Grad.ToArray(),
            firstWeight.Grad.ToArray(),
            firstBias.Grad.ToArray(),
            secondWeight.Grad.ToArray(),
            secondBias.Grad.ToArray(),
            hiddenBFloat16,
            inputBFloat16,
            parameterBFloat16,
            telemetry);
    }

    private static void AssertFfnClose(
        FfnRun expected,
        FfnRun actual,
        float tolerance)
    {
        AssertClose(expected.Output, actual.Output, tolerance);
        AssertClose(expected.InputGradient, actual.InputGradient, tolerance);
        AssertClose(
            expected.FirstWeightGradient,
            actual.FirstWeightGradient,
            tolerance);
        AssertClose(
            expected.FirstBiasGradient,
            actual.FirstBiasGradient,
            tolerance);
        AssertClose(
            expected.SecondWeightGradient,
            actual.SecondWeightGradient,
            tolerance);
        AssertClose(
            expected.SecondBiasGradient,
            actual.SecondBiasGradient,
            tolerance);
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

    private static T WithCuda<T>(Func<T> action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            return action();
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

    private sealed record FfnRun(
        float[] Output,
        float[] InputGradient,
        float[] FirstWeightGradient,
        float[] FirstBiasGradient,
        float[] SecondWeightGradient,
        float[] SecondBiasGradient,
        bool HiddenHasBFloat16Gradient,
        bool InputHasBFloat16Gradient,
        bool AnyParameterHasBFloat16Gradient,
        CudaBfp8GemmTelemetrySnapshot Telemetry);
}
