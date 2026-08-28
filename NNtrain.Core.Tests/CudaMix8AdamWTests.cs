using NNtrain;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Execution;
using Xunit;

public sealed class CudaMix8AdamWTests
{
    [Fact]
    public void FirstGuardedStepPrewarmsAllMix8ResidencyOutsideGuard()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithCuda([0, 1], () =>
        {
            Parameter parameter = CreateParameter(
                Values(257, 5, 0.5f),
                [257],
                "weight",
                Bfp8QuantizationDescriptor.Block(96),
                [0, 1]);
            var optimizer = new AdamW([parameter], Options());
            using var execution = new ExecutionSession(new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(0, 1),
                Precision = PrecisionPolicy.Mix8_32,
            });
            using IDisposable executionScope = execution.Enter();
            using var session = new TrainingSession(execution);
            var executor = new TrainingStepExecutor(session);
            var operations = new OptimizerTrainingOperations(optimizer);
            try
            {
                SetSynchronizedGradient(
                    parameter,
                    Values(257, 31, 0.06f),
                    [0, 1]);

                executor.Execute(operations);

                DeviceTransferSnapshot snapshot = Assert.NotNull(
                    operations.GuardedSnapshot);
                Assert.Equal(0, snapshot.HostToDeviceCopyCount);
                Assert.Equal(0, snapshot.HostToDeviceBytes);
                Assert.Equal(2 * sizeof(int), snapshot.DeviceToHostBytes);
                Assert.True(parameter.T.HasCudaMasterFloat32Buffer(0));
                Assert.True(parameter.T.HasCudaMasterFloat32Buffer(1));
                Assert.Equal(2, optimizer.CudaMultiTensorPlanBuildCount);

                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;
                optimizer.prepare();
                NativeCudaTransferTelemetry repeated =
                    NativeCudaRuntime.TransferTelemetry - before;
                Assert.Equal(0, repeated.HostToDeviceBytes);
                Assert.Equal(2, optimizer.CudaMultiTensorPlanBuildCount);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void PureBfp8PrewarmDoesNotCreateFloat32Master()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCudaPolicy([0], PrecisionPolicy.Bfp8, () =>
        {
            var parameter = new Parameter(
                Values(129, 3, 0.4f),
                [129],
                "weight",
                WeightDecayPolicy.Apply);
            parameter.T.ConvertStorageInPlace(
                TensorDType.Bfp8,
                Bfp8QuantizationDescriptor.TensorWide,
                preserveFloat32Master: false);
            _ = parameter.T.EnsureCudaBfp8Buffer(0);
            parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
            var optimizer = new AdamW([parameter], Options());
            try
            {
                optimizer.prepare();

                Assert.False(parameter.T.HasCudaMasterFloat32Buffer(0));
                _ = optimizer.GetCudaBfp8Moments(0, 0);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void OptInBlockBfp8StateKeepsMomentsResidentAndFinite()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            using IDisposable dispatch = CudaDispatchPolicy.Push(
                CudaDispatchPolicy.Defaults with
                {
                    EnableBlockBfp8OptimizerState = true,
                });
            Parameter parameter = CreateParameter(
                Values(257, 13, 0.6f),
                [257],
                "weight",
                Bfp8QuantizationDescriptor.Mix8_32);
            var optimizer = new AdamW([parameter], Options());
            try
            {
                parameter.T.SetCudaGradient(Values(257, 41, 0.08f), 0);
                optimizer.step();

                var moments = optimizer.GetCudaBfp8Moments(0, 0);
                Assert.Equal(
                    Bfp8QuantizationDescriptor.Mix8_32,
                    moments.First.Descriptor);
                Assert.Equal(3, moments.First.Scales.Length);
                Assert.All(Read(moments.First).Scales.ToArray(),
                    scale => Assert.True(float.IsFinite(scale) && scale > 0f));
                Assert.All(Read(moments.Second).Scales.ToArray(),
                    scale => Assert.True(float.IsFinite(scale) && scale > 0f));
                Assert.All(Read(parameter.T.EnsureCudaMasterFloat32Buffer(0)),
                    AssertFinite);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Theory]
    [InlineData(1, 128)]
    [InlineData(257, 128)]
    [InlineData(515, 96)]
    public void OneGpuKeepsBlockParameterAndFp32MasterGradientAndState(
        int length,
        int blockSize)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            float[] values = Values(length, 7, 0.7f);
            float[] gradient = Values(length, 23, 0.11f);
            Bfp8QuantizationDescriptor descriptor =
                Bfp8QuantizationDescriptor.Block(blockSize);
            Parameter parameter = CreateParameter(
                values, [length], "weight", descriptor);
            AdamWOptions options = Options();
            var optimizer = new AdamW([parameter], options);
            try
            {
                parameter.T.SetCudaGradient(gradient, 0);
                optimizer.step();

                Assert.Equal(descriptor, parameter.T.Bfp8Quantization);
                Assert.True(parameter.T.HasCudaMasterFloat32Buffer(0));
                Assert.False(parameter.T.HasAuthoritativeCudaBfp8Gradient);
                Assert.Throws<InvalidOperationException>(
                    () => optimizer.GetCudaBfp8Moments(0, 0));

                float[] expectedData = (float[])values.Clone();
                float[] expectedFirst = new float[length];
                float[] expectedSecond = new float[length];
                AdamWReference(
                    expectedData,
                    gradient,
                    expectedFirst,
                    expectedSecond,
                    options,
                    step: 1);
                AssertClose(expectedData,
                    Read(parameter.T.EnsureCudaMasterFloat32Buffer(0)),
                    2e-6f);
                var moments = optimizer.GetCudaMix8Moments(0, 0);
                AssertClose(expectedFirst, Read(moments.First), 2e-6f);
                AssertClose(expectedSecond, Read(moments.Second), 2e-6f);

                Bfp8EncodedStorage expectedEncoded =
                    Bfp8QuantizationCodec.Default.Encode(
                        expectedData,
                        descriptor);
                AssertEncoded(expectedEncoded,
                    Read(parameter.T.EnsureCudaBfp8Buffer(0)));
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void WarmTwoGpuStepTransfersOnlyOneFiniteScalarPerDevice()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithCuda([0, 1], () =>
        {
            Bfp8QuantizationDescriptor descriptor =
                Bfp8QuantizationDescriptor.Block(96);
            Parameter parameter = CreateParameter(
                Values(257, 5, 0.6f),
                [257],
                "weight",
                descriptor,
                [0, 1]);
            var optimizer = new AdamW([parameter], Options());
            try
            {
                SetSynchronizedGradient(
                    parameter, Values(257, 31, 0.09f), [0, 1]);
                optimizer.step();
                optimizer.zero_grad();
                SetSynchronizedGradient(
                    parameter, Values(257, 47, 0.07f), [0, 1]);
                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;

                optimizer.step();

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - before;
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(2 * sizeof(int), transfer.DeviceToHostBytes);
                AssertEncoded(
                    Read(parameter.T.EnsureCudaBfp8Buffer(0)),
                    Read(parameter.T.EnsureCudaBfp8Buffer(1)));
                Assert.Equal(
                    Read(parameter.T.EnsureCudaMasterFloat32Buffer(0)),
                    Read(parameter.T.EnsureCudaMasterFloat32Buffer(1)));
                var primary = optimizer.GetCudaMix8Moments(0, 0);
                var secondary = optimizer.GetCudaMix8Moments(0, 1);
                Assert.Equal(Read(primary.First), Read(secondary.First));
                Assert.Equal(Read(primary.Second), Read(secondary.Second));
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void BinaryCheckpointResumePreservesFp32MasterAndState()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Bfp8QuantizationDescriptor descriptor =
                Bfp8QuantizationDescriptor.Block(128);
            Parameter source = CreateParameter(
                Values(257, 3, 0.55f), [257], "weight", descriptor);
            AdamWOptions options = Options();
            var uninterrupted = new AdamW([source], options);
            AdamW? resumed = null;
            Parameter? restored = null;
            try
            {
                source.T.SetCudaGradient(Values(257, 17, 0.08f), 0);
                uninterrupted.step();
                float[] savedMaster = Read(
                    source.T.EnsureCudaMasterFloat32Buffer(0));
                using var checkpoint = new MemoryStream();
                OptimizerStateStream.SaveStateBinary(
                    uninterrupted,
                    checkpoint);

                restored = CreateParameter(
                    savedMaster,
                    [257],
                    "weight",
                    descriptor);
                resumed = new AdamW([restored], options);
                checkpoint.Position = 0;
                OptimizerStateStream.LoadStateBinary(resumed, checkpoint);

                float[] gradient = Values(257, 41, 0.06f);
                uninterrupted.zero_grad();
                resumed.zero_grad();
                source.T.SetCudaGradient(gradient, 0);
                restored.T.SetCudaGradient(gradient, 0);
                uninterrupted.step();
                resumed.step();

                Assert.Equal(
                    Read(source.T.EnsureCudaMasterFloat32Buffer(0)),
                    Read(restored.T.EnsureCudaMasterFloat32Buffer(0)));
                AssertEncoded(
                    Read(source.T.EnsureCudaBfp8Buffer(0)),
                    Read(restored.T.EnsureCudaBfp8Buffer(0)));
                var expected = uninterrupted.GetCudaMix8Moments(0, 0);
                var actual = resumed.GetCudaMix8Moments(0, 0);
                Assert.Equal(Read(expected.First), Read(actual.First));
                Assert.Equal(Read(expected.Second), Read(actual.Second));
            }
            finally
            {
                resumed?.DisposeCudaResources();
                uninterrupted.DisposeCudaResources();
                restored?.T.InvalidateCudaBuffers();
                source.T.InvalidateCudaBuffers();
            }
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BFloat16MomentOptionsResumeFp32MixedState(
        bool binaryCheckpoint)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Bfp8QuantizationDescriptor descriptor =
                Bfp8QuantizationDescriptor.Block(96);
            Parameter source = CreateParameter(
                Values(257, 13, 0.45f), [257], "weight", descriptor);
            AdamWOptions options = Options() with
            {
                UseBFloat16FirstMoment = true,
                UseBFloat16SecondMoment = true,
            };
            var uninterrupted = new AdamW([source], options);
            AdamW? resumed = null;
            Parameter? restored = null;
            try
            {
                source.T.SetCudaGradient(Values(257, 29, 0.07f), 0);
                uninterrupted.step();
                float[] savedMaster = Read(
                    source.T.EnsureCudaMasterFloat32Buffer(0));
                OptimizerStateDictionary? json = null;
                using var binary = new MemoryStream();
                if (binaryCheckpoint)
                    OptimizerStateStream.SaveStateBinary(uninterrupted, binary);
                else
                    json = uninterrupted.state_dict();

                restored = CreateParameter(
                    savedMaster, [257], "weight", descriptor);
                resumed = new AdamW([restored], options);
                if (binaryCheckpoint)
                {
                    binary.Position = 0;
                    OptimizerStateStream.LoadStateBinary(resumed, binary);
                }
                else
                {
                    resumed.load_state_dict(json!);
                }

                float[] gradient = Values(257, 47, 0.05f);
                uninterrupted.zero_grad();
                resumed.zero_grad();
                source.T.SetCudaGradient(gradient, 0);
                restored.T.SetCudaGradient(gradient, 0);
                uninterrupted.step();
                resumed.step();

                Assert.Equal(
                    Read(source.T.EnsureCudaMasterFloat32Buffer(0)),
                    Read(restored.T.EnsureCudaMasterFloat32Buffer(0)));
                var expected = uninterrupted.GetCudaMix8Moments(0, 0);
                var actual = resumed.GetCudaMix8Moments(0, 0);
                Assert.Equal(Read(expected.First), Read(actual.First));
                Assert.Equal(Read(expected.Second), Read(actual.Second));
            }
            finally
            {
                resumed?.DisposeCudaResources();
                uninterrupted.DisposeCudaResources();
                restored?.T.InvalidateCudaBuffers();
                source.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void BlockDescriptorDispatchesOutsidePrecisionScope()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCudaWithoutPrecision([0], () =>
        {
            Parameter parameter = CreateParameter(
                Values(129, 3, 0.4f),
                [129],
                "weight",
                Bfp8QuantizationDescriptor.Block(64));
            var optimizer = new AdamW([parameter], Options());
            try
            {
                Assert.Null(TensorExecutionContext.ActivePrecisionPolicy);
                parameter.T.SetCudaGradient(Values(129, 17, 0.04f), 0);
                optimizer.step();
                _ = optimizer.GetCudaMix8Moments(0, 0);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void Mix8PlanRebuildsAfterGradientArenaRebind()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Parameter parameter = CreateParameter(
                Values(257, 7, 0.5f),
                [257],
                "weight",
                Bfp8QuantizationDescriptor.Block(96));
            var optimizer = new AdamW([parameter], Options());
            NativeCudaArena<float>? arena = null;
            try
            {
                parameter.T.SetCudaGradient(Values(257, 23, 0.06f), 0);
                optimizer.step();
                Assert.Equal(1, optimizer.CudaMultiTensorPlanBuildCount);

                optimizer.zero_grad();
                arena = new NativeCudaArena<float>(
                    ForgetMemoryV2Cuda.GetAccelerator(0),
                    parameter.T.Numel);
                parameter.T.BindCudaGradientArena(
                    0,
                    arena.Slice(0, parameter.T.Numel));
                parameter.T.SetCudaGradient(Values(257, 41, 0.05f), 0);
                optimizer.step();

                Assert.Equal(2, optimizer.CudaMultiTensorPlanBuildCount);
                Assert.All(
                    Read(parameter.T.EnsureCudaMasterFloat32Buffer(0)),
                    value => Assert.True(float.IsFinite(value)));
            }
            finally
            {
                optimizer.DisposeCudaResources();
                if (arena is not null)
                {
                    parameter.T.UnbindCudaGradientArena(0, arena);
                    arena.Dispose();
                }
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void Mix8PlanRebuildsAfterBlockDescriptorConversion()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Parameter parameter = CreateParameter(
                Values(257, 15, 0.5f),
                [257],
                "weight",
                Bfp8QuantizationDescriptor.Block(96));
            var optimizer = new AdamW([parameter], Options());
            try
            {
                parameter.T.SetCudaGradient(Values(257, 27, 0.06f), 0);
                optimizer.step();
                Assert.Equal(1, optimizer.CudaMultiTensorPlanBuildCount);

                _ = parameter.T.CaptureData(preferMaster: true);
                Bfp8QuantizationDescriptor descriptor =
                    Bfp8QuantizationDescriptor.Block(64);
                parameter.T.ConvertStorageInPlace(
                    TensorDType.Bfp8,
                    descriptor,
                    preserveFloat32Master: true);
                parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
                parameter.T.SetCudaGradient(Values(257, 45, 0.05f), 0);
                optimizer.step();

                Assert.Equal(2, optimizer.CudaMultiTensorPlanBuildCount);
                Assert.Equal(descriptor, parameter.T.Bfp8Quantization);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void GenericPlanRebuildsAfterGradientArenaRebind()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCudaPolicy([0], PrecisionPolicy.Float32, () =>
        {
            var parameter = new Parameter(
                Values(257, 9, 0.5f),
                [257],
                "weight",
                WeightDecayPolicy.Apply);
            parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
            var optimizer = new AdamW([parameter], Options());
            NativeCudaArena<float>? arena = null;
            try
            {
                parameter.T.SetCudaGradient(Values(257, 25, 0.06f), 0);
                optimizer.step();
                Assert.Equal(1, optimizer.CudaMultiTensorPlanBuildCount);

                optimizer.zero_grad();
                arena = new NativeCudaArena<float>(
                    ForgetMemoryV2Cuda.GetAccelerator(0),
                    parameter.T.Numel);
                parameter.T.BindCudaGradientArena(
                    0,
                    arena.Slice(0, parameter.T.Numel));
                parameter.T.SetCudaGradient(Values(257, 43, 0.05f), 0);
                optimizer.step();

                Assert.Equal(2, optimizer.CudaMultiTensorPlanBuildCount);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                if (arena is not null)
                {
                    parameter.T.UnbindCudaGradientArena(0, arena);
                    arena.Dispose();
                }
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void FiniteGradientWhoseSquareOverflowsIsRejected()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Parameter parameter = CreateParameter(
                Values(129, 5, 0.2f),
                [129],
                "weight",
                Bfp8QuantizationDescriptor.Block(64));
            var optimizer = new AdamW([parameter], Options());
            try
            {
                parameter.T.SetCudaGradient(
                    Enumerable.Repeat(1e30f, 129).ToArray(),
                    0);
                InvalidOperationException exception = Assert.Throws<
                    InvalidOperationException>(optimizer.step);
                Assert.Contains("Non-finite", exception.Message);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void MixedOptimizerAndTensorResourcesDisposeIdempotently()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Parameter parameter = CreateParameter(
                Values(129, 2, 0.4f),
                [129],
                "weight",
                Bfp8QuantizationDescriptor.Block(64));
            var optimizer = new AdamW([parameter], Options());
            parameter.T.SetCudaGradient(Values(129, 19, 0.05f), 0);
            optimizer.step();
            NativeCudaAllocationTelemetry before =
                NativeCudaRuntime.AllocationTelemetry;

            optimizer.DisposeCudaResources();
            parameter.T.InvalidateCudaBuffers();
            optimizer.DisposeCudaResources();
            parameter.T.InvalidateCudaBuffers();

            NativeCudaAllocationTelemetry released =
                NativeCudaRuntime.AllocationTelemetry - before;
            Assert.True(released.FreeCount >= 6);
            Assert.True(released.FreeBytes > 0);
        });
    }

    private static AdamWOptions Options() => new()
    {
        LearningRate = 0.003f,
        Beta1 = 0.8f,
        Beta2 = 0.91f,
        Epsilon = 1e-6f,
        WeightDecay = 0.02f,
        Decay1D = true,
    };

    private static void AdamWReference(
        float[] data,
        float[] gradient,
        float[] first,
        float[] second,
        AdamWOptions options,
        int step)
    {
        float bc1 = 1f - MathF.Pow(options.Beta1, step);
        float bc2 = 1f - MathF.Pow(options.Beta2, step);
        float sqrtBc2 = MathF.Sqrt(bc2);
        float updateScale = options.LearningRate * sqrtBc2 / bc1;
        float scaledEpsilon = options.Epsilon * sqrtBc2;
        for (int index = 0; index < data.Length; index++)
        {
            float g = gradient[index];
            first[index] = MathF.FusedMultiplyAdd(
                options.Beta1, first[index], (1f - options.Beta1) * g);
            second[index] = MathF.FusedMultiplyAdd(
                options.Beta2,
                second[index],
                (1f - options.Beta2) * g * g);
            data[index] *= 1f - options.LearningRate * options.WeightDecay;
            data[index] -= updateScale * first[index]
                / (MathF.Sqrt(second[index]) + scaledEpsilon);
        }
    }

    private static Parameter CreateParameter(
        float[] values,
        int[] shape,
        string name,
        Bfp8QuantizationDescriptor descriptor,
        int[]? devices = null)
    {
        var parameter = new Parameter(
            values,
            shape,
            name,
            WeightDecayPolicy.Apply);
        parameter.T.ConvertStorageInPlace(
            TensorDType.Bfp8,
            descriptor,
            preserveFloat32Master: true);
        foreach (int device in devices ?? [0])
            _ = parameter.T.EnsureCudaBfp8Buffer(device);
        parameter.T.to(new TorchDevice(
            TensorDevice.Cuda,
            devices?[0] ?? 0));
        return parameter;
    }

    private static void SetSynchronizedGradient(
        Parameter parameter,
        float[] values,
        int[] devices)
    {
        foreach (int device in devices)
            parameter.T.SetCudaGradient(values, device);
        parameter.T.MarkCudaGradientsSynchronized(devices);
    }

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index => index == length - 1
                ? scale * 2.75f
                : MathF.Sin((index + offset) * 0.173f) * scale)
            .ToArray();

    private static float[] Read(NativeCudaBuffer<float> buffer)
    {
        var result = new float[buffer.Length];
        buffer.CopyToCPU(result);
        return result;
    }

    private static Bfp8EncodedStorage Read(CudaBfp8BufferView view)
    {
        var payload = new sbyte[view.Payload.Length];
        var scales = new float[view.Scales.Length];
        view.Payload.CopyToCPU(payload);
        view.Scales.CopyToCPU(scales);
        return new Bfp8EncodedStorage(payload, scales, view.Descriptor);
    }

    private static void AssertEncoded(
        Bfp8EncodedStorage expected,
        Bfp8EncodedStorage actual)
    {
        Assert.Equal(expected.Descriptor, actual.Descriptor);
        Assert.Equal(expected.Payload.ToArray(), actual.Payload.ToArray());
        Assert.Equal(expected.Scales.ToArray(), actual.Scales.ToArray());
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

    private static void AssertFinite(float value)
        => Assert.True(float.IsFinite(value));

    private static void WithCuda(int[] devices, Action action)
        => WithCudaPolicy(devices, PrecisionPolicy.Mix8_32, action);

    private static void WithCudaPolicy(
        int[] devices,
        PrecisionPolicy policy,
        Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = devices;
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(policy);
            action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private static void WithCudaWithoutPrecision(
        int[] devices,
        Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = devices;
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private sealed class OptimizerTrainingOperations(IOptimizer optimizer)
        : ITrainingStepOperations
    {
        internal DeviceTransferSnapshot? GuardedSnapshot { get; private set; }

        public TrainingGradientExecutionMode GradientExecutionMode
            => TrainingGradientExecutionMode.Separate;

        public void Prepare() => optimizer.prepare();

        public void AcquireBatch()
        {
        }

        public void ClearGradients()
        {
        }

        public void Forward()
        {
        }

        public void Backward()
        {
        }

        public void ReduceGradients()
        {
        }

        public void ForwardBackwardReduced()
            => throw new InvalidOperationException();

        public void ClipGradients()
        {
        }

        public void ApplySchedule()
        {
        }

        public void CommitOptimizer() => optimizer.step();

        public void CommitMetrics()
            => GuardedSnapshot = DeviceTransferGuard.CurrentSnapshot;
    }
}
