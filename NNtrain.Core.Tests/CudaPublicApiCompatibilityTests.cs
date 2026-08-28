using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaPublicApiCompatibilityTests
{
    [Fact]
    public void PureBFloat16SharedParentDagAccumulatesInResidentGradient()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndex = 0;
            float[] values = Pattern(257, 0.004f, 31);
            Tensor input = CreateTensor(values, [257], PrecisionMode.BFloat16);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(PrecisionMode.BFloat16));

            Tensor result = (input + input) + (input * input);
            result.Sum().BackwardAndRelease([1f]);

            Assert.True(input.TryGetCudaBFloat16GradientBuffer(0, out _));
            float[] expected = values
                .Select(static value => 2f + 2f * value)
                .ToArray();
            AssertClose(expected, input.Grad.ToArray(), 6e-2f);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    [Fact]
    public void PureBFloat16ShapeDagAndRepeatedConcatAccumulateInPlace()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndex = 0;
            Tensor input = CreateTensor(
                Pattern(8, 0.01f, 13), [2, 4], PrecisionMode.BFloat16);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(PrecisionMode.BFloat16));

            Tensor loss = Tensor.Concat(0, input, input).Sum()
                + input.Reshape(8).Sum()
                + input.Transpose().Sum()
                + input.Slice(1, 0, 4).Sum()
                + input.SelectLastSequenceToken().Sum();
            loss.BackwardAndRelease([1f]);

            Assert.True(input.TryGetCudaBFloat16GradientBuffer(0, out _));
            AssertClose(
                [5f, 5f, 5f, 5f, 6f, 6f, 6f, 6f],
                input.Grad.ToArray(),
                4e-2f);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    [Fact]
    public void PureBFloat16DoesNotOverwriteExistingFloatGradientAuthority()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndex = 0;
            Tensor input = CreateTensor(
                Pattern(64, 0.01f, 17), [64], PrecisionMode.BFloat16);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(PrecisionMode.BFloat16));
            _ = input.EnsureCudaGradientBuffer(0);

            (input + input).Sum().BackwardAndRelease([1f]);

            Assert.False(input.TryGetCudaBFloat16GradientBuffer(0, out _));
            AssertClose(
                Enumerable.Repeat(2f, 64).ToArray(),
                input.Grad.ToArray(),
                3e-2f);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    [Theory]
    [InlineData(PrecisionMode.Float32, TensorDType.BFloat16)]
    [InlineData(PrecisionMode.Float32, TensorDType.Bfp8)]
    [InlineData(PrecisionMode.BFloat16, TensorDType.Float32)]
    [InlineData(PrecisionMode.BFloat16, TensorDType.Bfp8)]
    [InlineData(PrecisionMode.Mix16_32, TensorDType.Float32)]
    [InlineData(PrecisionMode.Bfp8, TensorDType.Float32)]
    [InlineData(PrecisionMode.Bfp8, TensorDType.BFloat16)]
    [InlineData(PrecisionMode.Mix8_32, TensorDType.Float32)]
    [InlineData(PrecisionMode.Mix8_32, TensorDType.BFloat16)]
    public void DTypeConversionStaysResidentAndPreservesGradient(
        PrecisionMode mode,
        TensorDType target)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndex = 0;
            float[] values = Pattern(257, 0.006f, 37);
            Tensor input = CreateTensor(values, [257], mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Prewarm(input);
            Tensor converted;
            using (DeviceTransferGuard.EnterTrainingStep(1))
            {
                converted = input.To(target);
                DeviceTransferSnapshot snapshot = Assert.NotNull(
                    DeviceTransferGuard.CurrentSnapshot);
                Assert.Equal(0, snapshot.HostToDeviceCopyCount);
                Assert.Equal(0, snapshot.DeviceToHostCopyCount);
            }

            Assert.Equal(target, converted.DType);
            float[] convertedValues = converted.Data.ToArray();
            converted.Sum().BackwardAndRelease([1f]);
            float tolerance = target == TensorDType.Bfp8 ? 8e-2f : 3e-2f;
            AssertClose(values, convertedValues, tolerance);
            AssertClose(
                Enumerable.Repeat(1f, values.Length).ToArray(),
                input.Grad.ToArray(),
                3e-2f);
            if (mode == PrecisionMode.BFloat16)
                Assert.True(input.TryGetCudaBFloat16GradientBuffer(0, out _));
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    [Theory]
    [InlineData(PrecisionMode.Bfp8, Bfp8ScaleGranularity.Tensor)]
    [InlineData(PrecisionMode.Mix8_32, Bfp8ScaleGranularity.Block)]
    public void Float32ToBfp8UsesActiveScaleContract(
        PrecisionMode mode,
        Bfp8ScaleGranularity expectedGranularity)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndex = 0;
            Tensor input = new(
                Pattern(257, 0.006f, 41), [257], dtype: TensorDType.Float32);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Prewarm(input);
            using IDisposable guard = DeviceTransferGuard.EnterTrainingStep(1);
            Tensor converted = input.To(TensorDType.Bfp8);

            Assert.Equal(expectedGranularity,
                converted.Bfp8Quantization!.Granularity);
            DeviceTransferSnapshot snapshot = Assert.NotNull(
                DeviceTransferGuard.CurrentSnapshot);
            Assert.Equal(0, snapshot.HostToDeviceCopyCount);
            Assert.Equal(0, snapshot.DeviceToHostCopyCount);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    [Theory]
    [InlineData(PrecisionMode.Float32)]
    [InlineData(PrecisionMode.BFloat16)]
    [InlineData(PrecisionMode.Mix16_32)]
    [InlineData(PrecisionMode.Bfp8)]
    [InlineData(PrecisionMode.Mix8_32)]
    public void TransposeMatchesCpuForwardBackwardWithoutTransfers(
        PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        float[] values = Pattern(32 * 16, 0.007f, 41);
        float[] seed = Pattern(32 * 16, 0.009f, 37);
        ShapeEvaluation cpu = EvaluateTranspose(
            TensorDevice.Cpu, mode, values, seed, guarded: false);
        ShapeEvaluation cuda = EvaluateTranspose(
            TensorDevice.Cuda, mode, values, seed, guarded: false);
        float tolerance = mode == PrecisionMode.Float32 ? 1e-6f : 3e-2f;
        AssertClose(cpu.Output, cuda.Output, tolerance);
        AssertClose(cpu.Gradients[0], cuda.Gradients[0], tolerance);

        ShapeEvaluation guarded = EvaluateTranspose(
            TensorDevice.Cuda, mode, values, seed: null, guarded: true);
        Assert.Empty(guarded.Gradients);
    }

    [Theory]
    [InlineData(PrecisionMode.Float32)]
    [InlineData(PrecisionMode.BFloat16)]
    [InlineData(PrecisionMode.Mix16_32)]
    [InlineData(PrecisionMode.Bfp8)]
    [InlineData(PrecisionMode.Mix8_32)]
    public void NonContiguousSliceMatchesCpuForwardBackward(PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        float[] values = Pattern(2 * 3 * 128, 0.007f, 41);
        float[] seed = Pattern(2 * 3 * 96, 0.009f, 37);
        ShapeEvaluation cpu = EvaluateSlice(TensorDevice.Cpu, mode, values, seed);
        ShapeEvaluation cuda = EvaluateSlice(TensorDevice.Cuda, mode, values, seed);
        float tolerance = mode == PrecisionMode.Float32 ? 1e-6f : 3e-2f;
        AssertClose(cpu.Output, cuda.Output, tolerance);
        AssertClose(cpu.Gradients[0], cuda.Gradients[0], tolerance);
    }

    [Theory]
    [InlineData(PrecisionMode.Float32)]
    [InlineData(PrecisionMode.BFloat16)]
    [InlineData(PrecisionMode.Mix16_32)]
    [InlineData(PrecisionMode.Bfp8)]
    [InlineData(PrecisionMode.Mix8_32)]
    public void RankThreeConcatMatchesCpuForwardBackward(PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        float[] firstValues = Pattern(2 * 2 * 64, 0.007f, 43);
        float[] secondValues = Pattern(2 * 3 * 64, 0.011f, 47);
        float[] seed = Pattern(2 * 5 * 64, 0.009f, 53);
        ShapeEvaluation cpu = EvaluateConcat(
            TensorDevice.Cpu,
            mode,
            firstValues,
            secondValues,
            seed);
        ShapeEvaluation cuda = EvaluateConcat(
            TensorDevice.Cuda,
            mode,
            firstValues,
            secondValues,
            seed);
        float tolerance = mode == PrecisionMode.Float32 ? 1e-6f : 3e-2f;
        AssertClose(cpu.Output, cuda.Output, tolerance);
        AssertClose(cpu.Gradients[0], cuda.Gradients[0], tolerance);
        AssertClose(cpu.Gradients[1], cuda.Gradients[1], tolerance);
    }

    [Theory]
    [InlineData(PrecisionMode.Float32)]
    [InlineData(PrecisionMode.BFloat16)]
    [InlineData(PrecisionMode.Mix16_32)]
    [InlineData(PrecisionMode.Bfp8)]
    [InlineData(PrecisionMode.Mix8_32)]
    public void SliceAndConcatHaveNoGuardedTransfers(PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndex = 0;
            Tensor first = CreateTensor(
                Pattern(2 * 2 * 128, 0.007f, 41),
                [2, 2, 128],
                mode);
            Tensor second = CreateTensor(
                Pattern(2 * 3 * 128, 0.011f, 43),
                [2, 3, 128],
                mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Prewarm(first);
            Prewarm(second);
            using IDisposable noGrad = AutogradContext.NoGrad();
            using CudaInferenceScope inference = CudaInferenceScope.Begin();
            using IDisposable guard = DeviceTransferGuard.EnterTrainingStep(1);

            Tensor slice = second.Slice(2, 16, 96);
            Tensor concat = Tensor.Concat(1, first, second);

            Assert.Equal(ExpectedDType(mode), slice.DType);
            Assert.Equal(ExpectedDType(mode), concat.DType);
            DeviceTransferSnapshot snapshot = Assert.NotNull(
                DeviceTransferGuard.CurrentSnapshot);
            Assert.Equal(0, snapshot.HostToDeviceCopyCount);
            Assert.Equal(0, snapshot.DeviceToHostCopyCount);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    [Theory]
    [InlineData(PrecisionMode.Float32, false)]
    [InlineData(PrecisionMode.BFloat16, false)]
    [InlineData(PrecisionMode.Mix16_32, false)]
    [InlineData(PrecisionMode.Bfp8, false)]
    [InlineData(PrecisionMode.Mix8_32, false)]
    [InlineData(PrecisionMode.Float32, true)]
    [InlineData(PrecisionMode.BFloat16, true)]
    [InlineData(PrecisionMode.Mix16_32, true)]
    [InlineData(PrecisionMode.Bfp8, true)]
    [InlineData(PrecisionMode.Mix8_32, true)]
    public void TransposedRightGemmMatchesCpuForwardBackward(
        PrecisionMode mode,
        bool batched)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int batch = 2;
        const int rows = 16;
        const int inner = 16;
        const int columns = 16;
        int leftLength = (batched ? batch : 1) * rows * inner;
        int rightLength = (batched ? batch : 1) * columns * inner;
        int outputLength = (batched ? batch : 1) * rows * columns;
        float[] leftValues = Pattern(leftLength, 0.013f, 17);
        float[] rightValues = Pattern(rightLength, 0.017f, 23);
        float[] seed = Pattern(outputLength, 0.011f, 29);

        Evaluation cpu = Evaluate(
            TensorDevice.Cpu,
            mode,
            batched,
            leftValues,
            rightValues,
            seed);
        Evaluation cuda = Evaluate(
            TensorDevice.Cuda,
            mode,
            batched,
            leftValues,
            rightValues,
            seed);

        float tolerance = mode switch
        {
            PrecisionMode.Float32 => 4e-4f,
            PrecisionMode.BFloat16 or PrecisionMode.Mix16_32 => 4e-2f,
            _ => 1.2e-1f,
        };
        AssertClose(cpu.Output, cuda.Output, tolerance);
        AssertClose(cpu.LeftGradient, cuda.LeftGradient, tolerance);
        AssertClose(cpu.RightGradient, cuda.RightGradient, tolerance);
    }

    [Theory]
    [InlineData(PrecisionMode.Float32, false)]
    [InlineData(PrecisionMode.BFloat16, false)]
    [InlineData(PrecisionMode.Mix16_32, false)]
    [InlineData(PrecisionMode.Bfp8, false)]
    [InlineData(PrecisionMode.Mix8_32, false)]
    [InlineData(PrecisionMode.Float32, true)]
    [InlineData(PrecisionMode.BFloat16, true)]
    [InlineData(PrecisionMode.Mix16_32, true)]
    [InlineData(PrecisionMode.Bfp8, true)]
    [InlineData(PrecisionMode.Mix8_32, true)]
    public void TransposedRightGemmForwardHasNoGuardedTransfers(
        PrecisionMode mode,
        bool batched)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndex = 0;
            int count = (batched ? 2 : 1) * 16 * 16;
            Tensor left = CreateTensor(
                Pattern(count, 0.013f, 31),
                batched ? [2, 16, 16] : [16, 16],
                mode);
            Tensor right = CreateTensor(
                Pattern(count, 0.017f, 37),
                batched ? [2, 16, 16] : [16, 16],
                mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Prewarm(left);
            Prewarm(right);
            using IDisposable noGrad = AutogradContext.NoGrad();
            using CudaInferenceScope inference = CudaInferenceScope.Begin();
            using IDisposable guard = DeviceTransferGuard.EnterTrainingStep(1);

            Tensor result = batched
                ? left.BatchedMatMulTransposedRight(right)
                : left.MatMulTransposedRight(right);

            Assert.Equal(ExpectedDType(mode), result.DType);
            DeviceTransferSnapshot snapshot = Assert.NotNull(
                DeviceTransferGuard.CurrentSnapshot);
            Assert.Equal(0, snapshot.HostToDeviceCopyCount);
            Assert.Equal(0, snapshot.DeviceToHostCopyCount);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    [Theory]
    [InlineData(PrecisionMode.Float32)]
    [InlineData(PrecisionMode.BFloat16)]
    [InlineData(PrecisionMode.Mix16_32)]
    [InlineData(PrecisionMode.Bfp8)]
    [InlineData(PrecisionMode.Mix8_32)]
    public void ElementwiseAndScalarBroadcastMatchCpuForwardBackward(
        PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        float[] left = Pattern(513, 0.013f, 37)
            .Select(static value => value + 0.75f)
            .ToArray();
        float[] right = Pattern(513, 0.009f, 31)
            .Select(static value => value + 1.25f)
            .ToArray();
        float[] seed = Pattern(513, 0.007f, 29);
        foreach (string operation in new[]
        {
            "add", "subtract", "multiply", "divide", "scalar-multiply",
        })
        {
            ShapeEvaluation cpu = EvaluateElementwise(
                TensorDevice.Cpu, mode, operation, left, right, seed);
            ShapeEvaluation cuda = EvaluateElementwise(
                TensorDevice.Cuda, mode, operation, left, right, seed);
            float tolerance = mode switch
            {
                PrecisionMode.Float32 => 2e-5f,
                PrecisionMode.BFloat16 or PrecisionMode.Mix16_32 => 3e-2f,
                _ => 1.2e-1f,
            };
            AssertClose(cpu.Output, cuda.Output, tolerance);
            AssertClose(cpu.Gradients[0], cuda.Gradients[0], tolerance);
            AssertClose(cpu.Gradients[1], cuda.Gradients[1], tolerance);
        }
    }

    [Theory]
    [InlineData(PrecisionMode.Float32)]
    [InlineData(PrecisionMode.BFloat16)]
    [InlineData(PrecisionMode.Mix16_32)]
    [InlineData(PrecisionMode.Bfp8)]
    [InlineData(PrecisionMode.Mix8_32)]
    public void UnaryOperationsMatchCpuForwardBackward(PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        float[] signed = Pattern(513, 0.011f, 41);
        float[] positive = signed.Select(static value => value + 1.5f).ToArray();
        float[] seed = Pattern(513, 0.006f, 23);
        foreach (string operation in new[]
        {
            "negate", "relu", "gelu", "tanh", "exp", "log",
        })
        {
            float[] values = operation == "log" ? positive : signed;
            ShapeEvaluation cpu = EvaluateUnary(
                TensorDevice.Cpu, mode, operation, values, seed);
            ShapeEvaluation cuda = EvaluateUnary(
                TensorDevice.Cuda, mode, operation, values, seed);
            float tolerance = mode switch
            {
                PrecisionMode.Float32 => 3e-5f,
                PrecisionMode.BFloat16 or PrecisionMode.Mix16_32 => 4e-2f,
                _ => 1.5e-1f,
            };
            AssertClose(cpu.Output, cuda.Output, tolerance);
            AssertClose(cpu.Gradients[0], cuda.Gradients[0], tolerance);
        }
    }

    [Theory]
    [InlineData(PrecisionMode.Float32)]
    [InlineData(PrecisionMode.BFloat16)]
    [InlineData(PrecisionMode.Mix16_32)]
    [InlineData(PrecisionMode.Bfp8)]
    [InlineData(PrecisionMode.Mix8_32)]
    public void ReductionsMatchCpuForwardBackward(PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        float[] values = Pattern(513, 0.009f, 521);
        // Make the maximum unique so the public Max subgradient is stable.
        values[317] = 7f;
        foreach (string operation in new[] { "sum", "mean", "max" })
        {
            ShapeEvaluation cpu = EvaluateReduction(
                TensorDevice.Cpu, mode, operation, values);
            ShapeEvaluation cuda = EvaluateReduction(
                TensorDevice.Cuda, mode, operation, values);
            float tolerance = mode switch
            {
                PrecisionMode.Float32 => 2e-4f,
                PrecisionMode.BFloat16 or PrecisionMode.Mix16_32 => 5e-2f,
                _ => 2e-1f,
            };
            AssertClose(cpu.Output, cuda.Output, tolerance);
            AssertClose(cpu.Gradients[0], cuda.Gradients[0], tolerance);
        }
    }

    [Theory]
    [InlineData(PrecisionMode.Float32)]
    [InlineData(PrecisionMode.BFloat16)]
    [InlineData(PrecisionMode.Mix16_32)]
    [InlineData(PrecisionMode.Bfp8)]
    [InlineData(PrecisionMode.Mix8_32)]
    public void PublicElementwiseUnaryAndReductionsHaveNoGuardedTransfers(
        PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndex = 0;
            Tensor left = CreateTensor(
                Pattern(512, 0.009f, 37)
                    .Select(static value => value + 1f).ToArray(),
                [512],
                mode);
            Tensor right = CreateTensor(
                Pattern(512, 0.007f, 31)
                    .Select(static value => value + 1.5f).ToArray(),
                [512],
                mode);
            Tensor scalar = CreateTensor([0.75f], [1], mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Prewarm(left);
            Prewarm(right);
            Prewarm(scalar);
            using IDisposable noGrad = AutogradContext.NoGrad();
            using CudaInferenceScope inference = CudaInferenceScope.Begin();
            using IDisposable guard = DeviceTransferGuard.EnterTrainingStep(1);

            _ = left + right;
            _ = left - right;
            _ = left * right;
            _ = left / right;
            _ = left * scalar;
            _ = -left;
            _ = left.Relu();
            _ = left.Gelu();
            _ = left.Tanh();
            _ = left.Exp();
            _ = left.Log();
            _ = left.Sum();
            _ = left.Mean();
            _ = left.Max();

            DeviceTransferSnapshot snapshot = Assert.NotNull(
                DeviceTransferGuard.CurrentSnapshot);
            Assert.Equal(0, snapshot.HostToDeviceCopyCount);
            Assert.Equal(0, snapshot.DeviceToHostCopyCount);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    [Theory]
    [InlineData(PrecisionMode.Float32)]
    [InlineData(PrecisionMode.BFloat16)]
    [InlineData(PrecisionMode.Mix16_32)]
    [InlineData(PrecisionMode.Bfp8)]
    [InlineData(PrecisionMode.Mix8_32)]
    public void ForgetScanMatchesCpuForwardBackwardWithoutTransfers(
        PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        float[] projected = Pattern(2 * 7 * 3 * 16, 0.012f, 43);
        float[] seed = Pattern(2 * 7 * 16, 0.008f, 37);
        ShapeEvaluation cpu = EvaluateForgetScan(
            TensorDevice.Cpu, mode, projected, seed);
        ShapeEvaluation cuda = EvaluateForgetScan(
            TensorDevice.Cuda, mode, projected, seed);
        float tolerance = mode switch
        {
            PrecisionMode.Float32 => 4e-5f,
            PrecisionMode.BFloat16 or PrecisionMode.Mix16_32 => 5e-2f,
            _ => 2e-1f,
        };
        AssertClose(cpu.Output, cuda.Output, tolerance);
        AssertClose(cpu.Gradients[0], cuda.Gradients[0], tolerance);

        AssertForgetScanNoTransfers(mode, projected);
    }

    [Theory]
    [InlineData(PrecisionMode.Float32)]
    [InlineData(PrecisionMode.BFloat16)]
    [InlineData(PrecisionMode.Mix16_32)]
    [InlineData(PrecisionMode.Bfp8)]
    [InlineData(PrecisionMode.Mix8_32)]
    public void HyenaDirectMatchesCpuForwardBackwardWithoutTransfers(
        PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int batch = 2;
        const int sequence = 6;
        const int width = 8;
        int channels = 3 * width;
        float[] projected = Pattern(
            batch * sequence * channels, 0.01f, 41);
        float[] shortFilter = Pattern(3 * channels, 0.018f, 31);
        float[] longFilter = Pattern(sequence * width, 0.014f, 29);
        float[] diagonal = Pattern(width, 0.021f, 17)
            .Select(static value => value + 0.25f).ToArray();
        float[] seed = Pattern(batch * sequence * width, 0.009f, 37);
        ShapeEvaluation cpu = EvaluateHyena(
            TensorDevice.Cpu,
            mode,
            projected,
            shortFilter,
            longFilter,
            diagonal,
            seed);
        ShapeEvaluation cuda = EvaluateHyena(
            TensorDevice.Cuda,
            mode,
            projected,
            shortFilter,
            longFilter,
            diagonal,
            seed);
        float tolerance = mode switch
        {
            PrecisionMode.Float32 => 8e-5f,
            PrecisionMode.BFloat16 or PrecisionMode.Mix16_32 => 6e-2f,
            _ => 2.5e-1f,
        };
        AssertClose(cpu.Output, cuda.Output, tolerance);
        for (int index = 0; index < cpu.Gradients.Length; index++)
        {
            AssertClose(
                cpu.Gradients[index],
                cuda.Gradients[index],
                tolerance);
        }

        ShapeEvaluation cpuFft = EvaluateHyena(
            TensorDevice.Cpu,
            mode,
            projected,
            shortFilter,
            longFilter,
            diagonal,
            seed,
            HyenaConvolutionAlgorithm.Fft);
        ShapeEvaluation cudaParallelLong = EvaluateHyena(
            TensorDevice.Cuda,
            mode,
            projected,
            shortFilter,
            longFilter,
            diagonal,
            seed,
            HyenaConvolutionAlgorithm.Fft);
        AssertClose(cpuFft.Output, cudaParallelLong.Output, tolerance);
        for (int index = 0; index < cpuFft.Gradients.Length; index++)
        {
            AssertClose(
                cpuFft.Gradients[index],
                cudaParallelLong.Gradients[index],
                tolerance);
        }

        AssertHyenaNoTransfers(
            mode,
            projected,
            shortFilter,
            longFilter,
            diagonal);
    }

    [Theory]
    [InlineData(PrecisionMode.Float32)]
    [InlineData(PrecisionMode.BFloat16)]
    [InlineData(PrecisionMode.Mix16_32)]
    [InlineData(PrecisionMode.Bfp8)]
    [InlineData(PrecisionMode.Mix8_32)]
    public void RemainingUnaryAndStructuredOpsMatchCpuForwardBackward(
        PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        foreach (string operation in new[]
        {
            "sin", "pow", "add-row", "add-batch", "softmax",
            "log-softmax", "causal-mask",
        })
        {
            ShapeEvaluation cpu = EvaluateStructured(
                TensorDevice.Cpu, mode, operation);
            ShapeEvaluation cuda = EvaluateStructured(
                TensorDevice.Cuda, mode, operation);
            float tolerance = mode switch
            {
                PrecisionMode.Float32 => 4e-4f,
                PrecisionMode.BFloat16 or PrecisionMode.Mix16_32 => 7e-2f,
                _ => 3e-1f,
            };
            AssertClose(cpu.Output, cuda.Output, tolerance);
            Assert.Equal(cpu.Gradients.Length, cuda.Gradients.Length);
            for (int index = 0; index < cpu.Gradients.Length; index++)
            {
                AssertClose(
                    cpu.Gradients[index],
                    cuda.Gradients[index],
                    tolerance);
            }
        }
    }

    [Theory]
    [InlineData(PrecisionMode.Float32)]
    [InlineData(PrecisionMode.BFloat16)]
    [InlineData(PrecisionMode.Mix16_32)]
    [InlineData(PrecisionMode.Bfp8)]
    [InlineData(PrecisionMode.Mix8_32)]
    public void VectorMatMulVariantsMatchCpuForwardBackward(
        PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        foreach (bool dot in new[] { true, false })
        {
            ShapeEvaluation cpu = EvaluateVectorMatMul(
                TensorDevice.Cpu, mode, dot);
            ShapeEvaluation cuda = EvaluateVectorMatMul(
                TensorDevice.Cuda, mode, dot);
            float tolerance = mode switch
            {
                PrecisionMode.Float32 => 8e-4f,
                PrecisionMode.BFloat16 or PrecisionMode.Mix16_32 => 8e-2f,
                _ => 3e-1f,
            };
            AssertClose(cpu.Output, cuda.Output, tolerance);
            AssertClose(cpu.Gradients[0], cuda.Gradients[0], tolerance);
            AssertClose(cpu.Gradients[1], cuda.Gradients[1], tolerance);
        }
    }

    [Theory]
    [InlineData(PrecisionMode.Float32)]
    [InlineData(PrecisionMode.BFloat16)]
    [InlineData(PrecisionMode.Mix16_32)]
    [InlineData(PrecisionMode.Bfp8)]
    [InlineData(PrecisionMode.Mix8_32)]
    public void RemainingPublicOpsHaveNoGuardedTransfers(PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor vector = CreateTensor(
                Pattern(257, 0.006f, 37)
                    .Select(static value => value + 1.5f).ToArray(),
                [257], mode);
            Tensor otherVector = CreateTensor(
                Pattern(257, 0.007f, 41)
                    .Select(static value => value + 1f).ToArray(),
                [257], mode);
            Tensor matrix = CreateTensor(Pattern(17 * 33, 0.004f, 43),
                [17, 33], mode);
            Tensor matVector = CreateTensor(Pattern(33, 0.008f, 31),
                [33], mode);
            Tensor rows = CreateTensor(Pattern(5 * 73, 0.005f, 47),
                [5, 73], mode);
            Tensor row = CreateTensor(Pattern(73, 0.007f, 29), [73], mode);
            Tensor batched = CreateTensor(Pattern(3 * 5 * 73, 0.003f, 53),
                [3, 5, 73], mode);
            Tensor addMatrix = CreateTensor(Pattern(5 * 73, 0.004f, 59),
                [5, 73], mode);
            Tensor logits = CreateTensor(Pattern(3 * 257, 0.006f, 61),
                [3, 257], mode);
            Tensor mask = CreateTensor(Pattern(2 * 7 * 9, 0.005f, 31),
                [2, 7, 9], mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            foreach (Tensor tensor in new[]
            {
                vector, otherVector, matrix, matVector, rows, row, batched,
                addMatrix, logits, mask,
            })
            {
                Prewarm(tensor);
            }
            using IDisposable noGrad = AutogradContext.NoGrad();
            using CudaInferenceScope inference = CudaInferenceScope.Begin();
            using IDisposable guard = DeviceTransferGuard.EnterTrainingStep(1);
            _ = vector.Sin();
            _ = vector.Pow(1.7f);
            _ = rows.AddRowWise(row);
            _ = batched.AddBatchWise(addMatrix);
            _ = logits.SoftmaxLastDim();
            _ = logits.LogSoftmaxLastDim();
            _ = mask.CausalMask(-1f);
            _ = vector.MatMul(otherVector);
            _ = matrix.MatMul(matVector);

            DeviceTransferSnapshot snapshot = Assert.NotNull(
                DeviceTransferGuard.CurrentSnapshot);
            Assert.Equal(0, snapshot.HostToDeviceCopyCount);
            Assert.Equal(0, snapshot.DeviceToHostCopyCount);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static ShapeEvaluation EvaluateElementwise(
        TensorDevice device,
        PrecisionMode mode,
        string operation,
        float[] leftValues,
        float[] rightValues,
        float[] seed)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = device;
            Tensor.CudaDeviceIndex = 0;
            Tensor left = CreateTensor(leftValues, [leftValues.Length], mode);
            Tensor right = operation == "scalar-multiply"
                ? CreateTensor([0.75f], [1], mode)
                : CreateTensor(rightValues, [rightValues.Length], mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Tensor result = operation switch
            {
                "add" => left + right,
                "subtract" => left - right,
                "multiply" or "scalar-multiply" => left * right,
                "divide" => left / right,
                _ => throw new ArgumentOutOfRangeException(nameof(operation)),
            };
            float[] output = result.Data.ToArray();
            result.BackwardAndRelease(seed);
            return new ShapeEvaluation(
                output,
                [left.Grad.ToArray(), right.Grad.ToArray()]);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static ShapeEvaluation EvaluateUnary(
        TensorDevice device,
        PrecisionMode mode,
        string operation,
        float[] values,
        float[] seed)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = device;
            Tensor.CudaDeviceIndex = 0;
            Tensor input = CreateTensor(values, [values.Length], mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Tensor result = operation switch
            {
                "negate" => -input,
                "relu" => input.Relu(),
                "gelu" => input.Gelu(),
                "tanh" => input.Tanh(),
                "exp" => input.Exp(),
                "log" => input.Log(),
                _ => throw new ArgumentOutOfRangeException(nameof(operation)),
            };
            float[] output = result.Data.ToArray();
            result.BackwardAndRelease(seed);
            return new ShapeEvaluation(output, [input.Grad.ToArray()]);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static ShapeEvaluation EvaluateReduction(
        TensorDevice device,
        PrecisionMode mode,
        string operation,
        float[] values)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = device;
            Tensor.CudaDeviceIndex = 0;
            Tensor input = CreateTensor(values, [values.Length], mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Tensor result = operation switch
            {
                "sum" => input.Sum(),
                "mean" => input.Mean(),
                "max" => input.Max(),
                _ => throw new ArgumentOutOfRangeException(nameof(operation)),
            };
            float[] output = result.Data.ToArray();
            result.BackwardAndRelease([0.75f]);
            return new ShapeEvaluation(output, [input.Grad.ToArray()]);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static ShapeEvaluation EvaluateForgetScan(
        TensorDevice device,
        PrecisionMode mode,
        float[] projectedValues,
        float[] seed)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = device;
            Tensor.CudaDeviceIndex = 0;
            Tensor projected = CreateTensor(
                projectedValues,
                [2, 7, 3 * 16],
                mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Tensor result = projected.FusedForgetScan();
            float[] output = result.Data.ToArray();
            result.BackwardAndRelease(seed);
            return new ShapeEvaluation(output, [projected.Grad.ToArray()]);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static ShapeEvaluation EvaluateHyena(
        TensorDevice device,
        PrecisionMode mode,
        float[] projectedValues,
        float[] shortValues,
        float[] longValues,
        float[] diagonalValues,
        float[] seed,
        HyenaConvolutionAlgorithm algorithm =
            HyenaConvolutionAlgorithm.Direct)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = device;
            Tensor.CudaDeviceIndex = 0;
            Tensor projected = CreateTensor(projectedValues, [2, 6, 24], mode);
            Tensor shortFilter = CreateTensor(shortValues, [3, 24], mode);
            Tensor longFilter = CreateTensor(longValues, [6, 8], mode);
            Tensor diagonal = CreateTensor(diagonalValues, [8], mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Tensor result = projected.FusedCausalHyenaOrder2(
                shortFilter,
                longFilter,
                diagonal,
                algorithm);
            float[] output = result.Data.ToArray();
            result.BackwardAndRelease(seed);
            return new ShapeEvaluation(
                output,
                [
                    projected.Grad.ToArray(),
                    shortFilter.Grad.ToArray(),
                    longFilter.Grad.ToArray(),
                    diagonal.Grad.ToArray(),
                ]);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static ShapeEvaluation EvaluateStructured(
        TensorDevice device,
        PrecisionMode mode,
        string operation)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = device;
            Tensor.CudaDeviceIndex = 0;
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Tensor[] inputs;
            Tensor result;
            float[] seed;
            switch (operation)
            {
                case "sin":
                case "pow":
                {
                    Tensor input = CreateTensor(
                        Pattern(513, 0.004f, 37)
                            .Select(static value => value + 1.25f).ToArray(),
                        [513], mode);
                    inputs = [input];
                    result = operation == "sin"
                        ? input.Sin()
                        : input.Pow(1.7f);
                    seed = Pattern(513, 0.006f, 31);
                    break;
                }
                case "add-row":
                {
                    Tensor input = CreateTensor(
                        Pattern(5 * 73, 0.005f, 43), [5, 73], mode);
                    Tensor row = CreateTensor(
                        Pattern(73, 0.007f, 29), [73], mode);
                    inputs = [input, row];
                    result = input.AddRowWise(row);
                    seed = Pattern(5 * 73, 0.006f, 37);
                    break;
                }
                case "add-batch":
                {
                    Tensor input = CreateTensor(
                        Pattern(3 * 5 * 73, 0.003f, 47),
                        [3, 5, 73], mode);
                    Tensor matrix = CreateTensor(
                        Pattern(5 * 73, 0.004f, 53), [5, 73], mode);
                    inputs = [input, matrix];
                    result = input.AddBatchWise(matrix);
                    seed = Pattern(3 * 5 * 73, 0.005f, 41);
                    break;
                }
                case "softmax":
                case "log-softmax":
                {
                    Tensor input = CreateTensor(
                        Pattern(3 * 257, 0.006f, 61), [3, 257], mode);
                    inputs = [input];
                    result = operation == "softmax"
                        ? input.SoftmaxLastDim()
                        : input.LogSoftmaxLastDim();
                    seed = Pattern(3 * 257, 0.004f, 37);
                    break;
                }
                case "causal-mask":
                {
                    Tensor input = CreateTensor(
                        Pattern(2 * 7 * 9, 0.005f, 31), [2, 7, 9], mode);
                    inputs = [input];
                    result = input.CausalMask(-1f);
                    seed = Pattern(2 * 7 * 9, 0.006f, 29);
                    break;
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(operation));
            }
            float[] output = result.Data.ToArray();
            result.BackwardAndRelease(seed);
            return new ShapeEvaluation(
                output,
                inputs.Select(static input => input.Grad.ToArray()).ToArray());
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static ShapeEvaluation EvaluateVectorMatMul(
        TensorDevice device,
        PrecisionMode mode,
        bool dot)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = device;
            Tensor.CudaDeviceIndex = 0;
            Tensor left = dot
                ? CreateTensor(Pattern(257, 0.006f, 37), [257], mode)
                : CreateTensor(Pattern(17 * 33, 0.004f, 43), [17, 33], mode);
            Tensor right = dot
                ? CreateTensor(Pattern(257, 0.007f, 41), [257], mode)
                : CreateTensor(Pattern(33, 0.008f, 31), [33], mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Tensor result = left.MatMul(right);
            float[] output = result.Data.ToArray();
            result.BackwardAndRelease(Pattern(result.Numel, 0.01f, 17));
            return new ShapeEvaluation(
                output,
                [left.Grad.ToArray(), right.Grad.ToArray()]);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static void AssertForgetScanNoTransfers(
        PrecisionMode mode,
        float[] projectedValues)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor projected = CreateTensor(
                projectedValues,
                [2, 7, 3 * 16],
                mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Prewarm(projected);
            using IDisposable noGrad = AutogradContext.NoGrad();
            using CudaInferenceScope inference = CudaInferenceScope.Begin();
            using IDisposable guard = DeviceTransferGuard.EnterTrainingStep(1);
            Tensor result = projected.FusedForgetScan();
            Assert.Equal(ExpectedDType(mode), result.DType);
            DeviceTransferSnapshot snapshot = Assert.NotNull(
                DeviceTransferGuard.CurrentSnapshot);
            Assert.Equal(0, snapshot.HostToDeviceCopyCount);
            Assert.Equal(0, snapshot.DeviceToHostCopyCount);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static void AssertHyenaNoTransfers(
        PrecisionMode mode,
        float[] projectedValues,
        float[] shortValues,
        float[] longValues,
        float[] diagonalValues)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor projected = CreateTensor(projectedValues, [2, 6, 24], mode);
            Tensor shortFilter = CreateTensor(shortValues, [3, 24], mode);
            Tensor longFilter = CreateTensor(longValues, [6, 8], mode);
            Tensor diagonal = CreateTensor(diagonalValues, [8], mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Prewarm(projected);
            Prewarm(shortFilter);
            Prewarm(longFilter);
            Prewarm(diagonal);
            using IDisposable noGrad = AutogradContext.NoGrad();
            using CudaInferenceScope inference = CudaInferenceScope.Begin();
            using IDisposable guard = DeviceTransferGuard.EnterTrainingStep(1);
            Tensor result = projected.FusedCausalHyenaOrder2(
                shortFilter,
                longFilter,
                diagonal,
                HyenaConvolutionAlgorithm.Fft);
            Assert.Equal(ExpectedDType(mode), result.DType);
            DeviceTransferSnapshot snapshot = Assert.NotNull(
                DeviceTransferGuard.CurrentSnapshot);
            Assert.Equal(0, snapshot.HostToDeviceCopyCount);
            Assert.Equal(0, snapshot.DeviceToHostCopyCount);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static Evaluation Evaluate(
        TensorDevice device,
        PrecisionMode mode,
        bool batched,
        float[] leftValues,
        float[] rightValues,
        float[] seed)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = device;
            Tensor.CudaDeviceIndex = 0;
            Tensor left = CreateTensor(
                leftValues,
                batched ? [2, 16, 16] : [16, 16],
                mode);
            Tensor right = CreateTensor(
                rightValues,
                batched ? [2, 16, 16] : [16, 16],
                mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Tensor result = batched
                ? left.BatchedMatMulTransposedRight(right)
                : left.MatMulTransposedRight(right);
            float[] output = result.Data.ToArray();
            result.BackwardAndRelease(seed);
            return new Evaluation(
                output,
                left.Grad.ToArray(),
                right.Grad.ToArray());
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static ShapeEvaluation EvaluateSlice(
        TensorDevice device,
        PrecisionMode mode,
        float[] values,
        float[] seed)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = device;
            Tensor.CudaDeviceIndex = 0;
            Tensor input = CreateTensor(values, [2, 3, 128], mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Tensor result = input.Slice(2, 16, 96);
            float[] output = result.Data.ToArray();
            result.BackwardAndRelease(seed);
            return new ShapeEvaluation(output, [input.Grad.ToArray()]);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static ShapeEvaluation EvaluateConcat(
        TensorDevice device,
        PrecisionMode mode,
        float[] firstValues,
        float[] secondValues,
        float[] seed)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = device;
            Tensor.CudaDeviceIndex = 0;
            Tensor first = CreateTensor(firstValues, [2, 2, 64], mode);
            Tensor second = CreateTensor(secondValues, [2, 3, 64], mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            Tensor result = Tensor.Concat(1, first, second);
            float[] output = result.Data.ToArray();
            result.BackwardAndRelease(seed);
            return new ShapeEvaluation(
                output,
                [first.Grad.ToArray(), second.Grad.ToArray()]);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static ShapeEvaluation EvaluateTranspose(
        TensorDevice device,
        PrecisionMode mode,
        float[] values,
        float[]? seed,
        bool guarded)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = device;
            Tensor.CudaDeviceIndex = 0;
            Tensor input = CreateTensor(values, [32, 16], mode);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.For(mode));
            if (guarded)
            {
                Prewarm(input);
                using IDisposable noGrad = AutogradContext.NoGrad();
                using CudaInferenceScope inference = CudaInferenceScope.Begin();
                using IDisposable guard = DeviceTransferGuard.EnterTrainingStep(1);
                Tensor guardedResult = input.Transpose();
                Assert.Equal(ExpectedDType(mode), guardedResult.DType);
                DeviceTransferSnapshot snapshot = Assert.NotNull(
                    DeviceTransferGuard.CurrentSnapshot);
                Assert.Equal(0, snapshot.HostToDeviceCopyCount);
                Assert.Equal(0, snapshot.DeviceToHostCopyCount);
                return new ShapeEvaluation([], []);
            }

            Tensor result = input.Transpose();
            float[] output = result.Data.ToArray();
            result.BackwardAndRelease(seed!);
            return new ShapeEvaluation(output, [input.Grad.ToArray()]);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static Tensor CreateTensor(
        float[] values,
        int[] shape,
        PrecisionMode mode)
        => mode switch
        {
            PrecisionMode.Float32 => new Tensor(
                values, shape, dtype: TensorDType.Float32),
            PrecisionMode.BFloat16 or PrecisionMode.Mix16_32 => new Tensor(
                values, shape, dtype: TensorDType.BFloat16),
            PrecisionMode.Bfp8 => Tensor.FromBfp8(
                values, shape, Bfp8QuantizationDescriptor.TensorWide),
            PrecisionMode.Mix8_32 => Tensor.FromBfp8(
                values, shape, Bfp8QuantizationDescriptor.Mix8_32),
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };

    private static void Prewarm(Tensor tensor)
    {
        if (tensor.DType == TensorDType.Bfp8)
            _ = tensor.EnsureCudaBfp8Buffer(0);
        else if (tensor.DType == TensorDType.BFloat16)
            _ = tensor.EnsureCudaBFloat16Buffer(0);
        else
            _ = tensor.EnsureCudaFloat32Buffer(0);
    }

    private static TensorDType ExpectedDType(PrecisionMode mode)
        => mode switch
        {
            PrecisionMode.Float32 => TensorDType.Float32,
            PrecisionMode.BFloat16 or PrecisionMode.Mix16_32 =>
                TensorDType.BFloat16,
            _ => TensorDType.Bfp8,
        };

    private static float[] Pattern(int length, float scale, int period)
        => Enumerable.Range(0, length)
            .Select(index => ((index % period) - period / 2) * scale)
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

    private sealed record Evaluation(
        float[] Output,
        float[] LeftGradient,
        float[] RightGradient);

    private sealed record ShapeEvaluation(
        float[] Output,
        float[][] Gradients);
}
