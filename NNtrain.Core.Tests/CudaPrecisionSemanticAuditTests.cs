using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaPrecisionSemanticAuditTests
{
    [Fact]
    public void TransformerManifestCoversEveryPrecisionWithoutCpuFallback()
    {
        PrecisionMode[] modes = Enum.GetValues<PrecisionMode>();
        Assert.Equal(5, modes.Length);
        Assert.Equal(20, CudaPrecisionOperationManifest.Entries.Count);

        foreach (CudaPrecisionOperationEntry entry
            in CudaPrecisionOperationManifest.Entries)
        {
            Assert.Equal(modes.Length, entry.Routes.Count);
            Assert.Equal(
                modes.OrderBy(static mode => mode),
                entry.Routes.Select(static route => route.Mode)
                    .OrderBy(static mode => mode));
            Assert.All(entry.Routes, route =>
            {
                PrecisionPolicy policy = PrecisionPolicy.For(route.Mode);
                Assert.Equal(policy.ActivationStorage, route.Storage);
                NumericFormat expectedCompute = entry.ComputeContract switch
                {
                    CudaPrecisionComputeContract.MatrixOperand =>
                        policy.MatrixOperand,
                    CudaPrecisionComputeContract.Elementwise =>
                        policy.ElementwiseCompute,
                    CudaPrecisionComputeContract.Reduction =>
                        policy.Reduction,
                    CudaPrecisionComputeContract.Storage =>
                        policy.ActivationStorage,
                    _ => throw new InvalidOperationException(),
                };
                Assert.Equal(expectedCompute, route.Compute);
                Assert.Equal(policy.Accumulation, route.Accumulation);
                Assert.Equal(policy.Gradient, route.Gradient);
                Assert.False(route.AllowsCpuFallback);
                Assert.DoesNotContain(
                    "CPU",
                    route.Backend,
                    StringComparison.OrdinalIgnoreCase);
            });

            CudaPrecisionOperationRoute bfp8 = Assert.Single(
                entry.Routes,
                static route => route.Mode == PrecisionMode.Bfp8);
            CudaPrecisionOperationRoute mix8 = Assert.Single(
                entry.Routes,
                static route => route.Mode == PrecisionMode.Mix8_32);
            Assert.Equal(
                CudaBfp8ScaleContract.TensorWide,
                bfp8.ScaleContract);
            Assert.Equal(CudaBfp8ScaleContract.Block, mix8.ScaleContract);
        }

        foreach (string operation in new[]
        {
            "linear/GEMM",
            "attention",
            "ForgetMemory",
        })
        {
            CudaPrecisionOperationEntry entry = Assert.Single(
                CudaPrecisionOperationManifest.Entries,
                candidate => candidate.Operation == operation);
            Assert.All(
                entry.Routes.Where(static route =>
                    route.Mode != PrecisionMode.Float32),
                static route => Assert.True(route.UsesTensorCoreWhenEligible));
        }


        foreach (string operation in new[]
        {
            "elementwise/scalar",
            "activations",
            "sum/mean/max",
            "Slice/Concat/Transpose/Reshape/Select",
            "indexed broadcast",
            "softmax/logsoftmax/causal-mask",
            "rank/batched/transposed GEMM",
            "ForgetScan",
            "Hyena",
            "dtype conversion",
            "classification accuracy",
        })
        {
            Assert.Contains(
                CudaPrecisionOperationManifest.Entries,
                entry => entry.Operation == operation);
        }

        CudaPrecisionOperationEntry crossEntropy = Assert.Single(
            CudaPrecisionOperationManifest.Entries,
            static entry => entry.Operation == "cross-entropy loss");
        Assert.All(
            crossEntropy.Routes.Where(static route =>
                route.Mode is PrecisionMode.Bfp8 or PrecisionMode.Mix8_32),
            static route => Assert.Contains(
                "direct BF16 loss-head intermediate",
                route.Backend,
                StringComparison.Ordinal));
    }

    [Fact]
    public void ActiveBfp8PoliciesRejectTheWrongScaleGranularityBeforeCuda()
    {
        Tensor tensorWide = Tensor.FromBfp8(
            Enumerable.Range(0, 256).Select(static value => (float)value).ToArray(),
            [256],
            Bfp8QuantizationDescriptor.TensorWide);
        Tensor block = Tensor.FromBfp8(
            Enumerable.Range(0, 256).Select(static value => (float)value).ToArray(),
            [256],
            Bfp8QuantizationDescriptor.Mix8_32);

        using (TensorExecutionContext.PushPrecisionPolicy(PrecisionPolicy.Mix8_32))
        {
            InvalidOperationException failure = Assert.Throws<InvalidOperationException>(
                tensorWide.ValidateBfp8PrecisionContract);
            Assert.Contains("block-scaled", failure.Message);
        }
        using (TensorExecutionContext.PushPrecisionPolicy(PrecisionPolicy.Bfp8))
        {
            InvalidOperationException failure = Assert.Throws<InvalidOperationException>(
                block.ValidateBfp8PrecisionContract);
            Assert.Contains("tensor-wide", failure.Message);
        }
    }

    [Fact]
    public void Bfp8GemmRejectsMixedScaleDescriptorsBeforeKernelDispatch()
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor left = Tensor.FromBfp8(
                Enumerable.Range(0, 16).Select(static value => (float)value).ToArray(),
                [4, 4],
                Bfp8QuantizationDescriptor.TensorWide);
            Tensor right = Tensor.FromBfp8(
                Enumerable.Range(0, 16).Select(static value => (float)value).ToArray(),
                [4, 4],
                Bfp8QuantizationDescriptor.Block(4));

            InvalidOperationException failure = Assert.Throws<InvalidOperationException>(
                () => left.MatMul(right));
            Assert.Contains("cannot be mixed", failure.Message);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    [Fact]
    public void ResidentCudaPublicOpsPerformNoTrainingStepTransfers()
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            var left = new Tensor([1f, -2f, 3f, -4f], [2, 2]);
            var right = new Tensor([4f, 3f, 2f, 1f], [2, 2]);

            // Materialize the source before entering the training-step guard.
            // Slice is now a CUDA-resident public operation; invoking it below
            // must perform only device-to-device copies.
            _ = left.Slice(0, 0, 1);
            _ = left.Relu();
            _ = left * right;

            using IDisposable guard = DeviceTransferGuard.EnterTrainingStep(1);
            _ = left.Relu();
            _ = left * right;
            _ = left.Slice(0, 0, 1);

            DeviceTransferSnapshot snapshot = Assert.NotNull(
                DeviceTransferGuard.CurrentSnapshot);
            Assert.Equal(0, snapshot.HostToDeviceCopyCount);
            Assert.Equal(0, snapshot.HostToDeviceBytes);
            Assert.Equal(0, snapshot.DeviceToHostCopyCount);
            Assert.Equal(0, snapshot.DeviceToHostBytes);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    [Fact]
    public void PureBfloat16UsesBfloat16GradientsWhileMix16UsesFloat32()
    {
        Assert.True(TensorExecutionContext.UsesBFloat16GradientStorage);
        using (TensorExecutionContext.PushPrecisionPolicy(PrecisionPolicy.BFloat16))
            Assert.True(TensorExecutionContext.UsesBFloat16GradientStorage);
        using (TensorExecutionContext.PushPrecisionPolicy(PrecisionPolicy.Mix16_32))
            Assert.False(TensorExecutionContext.UsesBFloat16GradientStorage);

        if (!Tensor.IsCudaAvailable())
            return;

        Assert.True(RunBfloat16LinearLoss(PrecisionPolicy.BFloat16));
        Assert.False(RunBfloat16LinearLoss(PrecisionPolicy.Mix16_32));
    }

    [Fact]
    public void BatchedBfp8MatMulUsesResidentCudaRouteWithoutTransfers()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndex = 0;
            float[] values = Enumerable.Range(0, 2 * 16 * 16)
                .Select(static index => MathF.Sin(index * 0.03125f))
                .ToArray();
            Tensor left = Tensor.FromBfp8(
                values,
                [2, 16, 16],
                Bfp8QuantizationDescriptor.TensorWide);
            Tensor right = Tensor.FromBfp8(
                values,
                [2, 16, 16],
                Bfp8QuantizationDescriptor.TensorWide);

            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.Bfp8);
            _ = left.EnsureCudaBfp8Buffer(0);
            _ = right.EnsureCudaBfp8Buffer(0);
            using IDisposable noGrad = AutogradContext.NoGrad();
            using CudaInferenceScope inference = CudaInferenceScope.Begin();
            using IDisposable guard = DeviceTransferGuard.EnterTrainingStep(1);
            Tensor result = left.BatchedMatMul(right);

            Assert.Equal(TensorDType.Bfp8, result.DType);
            Assert.Equal(
                Bfp8QuantizationDescriptor.TensorWide,
                result.Bfp8Quantization);
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

    private static bool RunBfloat16LinearLoss(PrecisionPolicy precision)
    {
        const int width = 16;
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndex = 0;
            float[] matrix = Enumerable.Range(0, width * width)
                .Select(static index => MathF.Sin(index * 0.015625f))
                .ToArray();
            var input = new Tensor(matrix, [width, width], dtype: TensorDType.BFloat16);
            var weight = new Tensor(matrix, [width, width], dtype: TensorDType.BFloat16);
            var bias = new Tensor(new float[width], [width], dtype: TensorDType.BFloat16);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(precision);
            Tensor logits = input.LinearLastDim(weight, bias, applyRelu: false);
            Tensor loss = logits.CrossEntropyWithLogits(
                Enumerable.Range(0, width).ToArray());
            loss.BackwardAndRelease();
            return input.TryGetCudaBFloat16GradientBuffer(0, out _);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }
}
