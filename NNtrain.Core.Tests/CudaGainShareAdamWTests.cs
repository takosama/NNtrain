using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

[Collection(TensorSimdCollection.Name)]
public sealed class CudaGainShareAdamWTests
{
    public static TheoryData<PrecisionMode> Precisions => new()
    {
        PrecisionMode.Float32,
        PrecisionMode.BFloat16,
        PrecisionMode.Mix16_32,
        PrecisionMode.Bfp8,
        PrecisionMode.Mix8_32,
    };

    [Theory]
    [MemberData(nameof(Precisions))]
    public void OneGpuStepSupportsEveryPrecisionAndSynchronizesCheckpoint(
        PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], mode, () =>
        {
            Parameter first = CreateParameter(
                mode, Values(257, 7, 0.5f), [257], "first.weight", [0]);
            Parameter second = CreateParameter(
                mode, Values(129, 19, 0.35f), [3, 43], "second.weight", [0]);
            using var optimizer = new GainShareAdamW(
                [[first], [second]], Options());
            try
            {
                optimizer.prepare();
                PublishGradient(first, Values(257, 31, 0.08f), mode, [0]);
                PublishGradient(second, Values(129, 47, 0.06f), mode, [0]);

                optimizer.step();

                GainShareAdamWState state = optimizer.CaptureState();
                Assert.Equal(1, state.Step);
                Assert.All(state.ParameterStates, parameterState =>
                {
                    Assert.Contains(parameterState.FirstMoment,
                        value => value != 0f);
                    Assert.Contains(parameterState.SecondMoment,
                        value => value > 0f);
                    Assert.All(parameterState.FirstMoment,
                        value => Assert.True(float.IsFinite(value)));
                    Assert.All(parameterState.SecondMoment,
                        value => Assert.True(float.IsFinite(value)));
                });
                Assert.All(state.GroupStates, group =>
                    Assert.True(group.AlignmentEma is >= 0d));
                Assert.All(ReadParameter(first, mode, 0), value =>
                    Assert.True(float.IsFinite(value)));
                AssertPrecisionResidency(first.T, mode, 0);
            }
            finally
            {
                first.T.InvalidateCudaBuffers();
                second.T.InvalidateCudaBuffers();
            }
        });
    }

    [Theory]
    [MemberData(nameof(Precisions))]
    public void WarmStepHasNoPayloadTransferOrNativeAllocation(
        PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], mode, () =>
        {
            Parameter parameter = CreateParameter(
                mode, Values(513, 11, 0.4f), [513], "weight", [0]);
            using var optimizer = new GainShareAdamW(
                [[parameter]], Options());
            try
            {
                optimizer.prepare();
                PublishGradient(
                    parameter, Values(513, 23, 0.07f), mode, [0]);
                optimizer.step();
                optimizer.zero_grad();
                PublishGradient(
                    parameter, Values(513, 41, 0.05f), mode, [0]);
                NativeCudaTransferTelemetry transferBefore =
                    NativeCudaRuntime.TransferTelemetry;
                NativeCudaAllocationTelemetry allocationBefore =
                    NativeCudaRuntime.AllocationTelemetry;

                optimizer.step();

                NativeCudaTransferTelemetry transfers =
                    NativeCudaRuntime.TransferTelemetry - transferBefore;
                NativeCudaAllocationTelemetry allocations =
                    NativeCudaRuntime.AllocationTelemetry - allocationBefore;
                Assert.Equal(0, transfers.HostToDeviceBytes);
                Assert.Equal(
                    mode is PrecisionMode.Bfp8 or PrecisionMode.Mix8_32
                        ? sizeof(int)
                        : 0,
                    transfers.DeviceToHostBytes);
                Assert.Equal(0, allocations.AllocationCount);
                Assert.Equal(0, allocations.AllocationBytes);
            }
            finally
            {
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void Mix8StepRejectsNonFiniteGradientFromDeviceStatus()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], PrecisionMode.Mix8_32, () =>
        {
            Parameter parameter = CreateParameter(
                PrecisionMode.Mix8_32,
                Values(129, 5, 0.3f),
                [3, 43],
                "weight",
                [0]);
            using var optimizer = new GainShareAdamW(
                [[parameter]], Options());
            try
            {
                float[] gradient = Values(129, 17, 0.05f);
                PublishGradient(
                    parameter,
                    gradient,
                    PrecisionMode.Mix8_32,
                    [0]);
                gradient[73] = float.NaN;
                parameter.T.EnsureCudaGradientBuffer(0)
                    .CopyFromCPU(gradient);

                InvalidOperationException failure = Assert.Throws<
                    InvalidOperationException>(() => optimizer.step());

                Assert.Contains("Non-finite CUDA value", failure.Message);
            }
            finally
            {
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void Bfp8StepRejectsNonFiniteGradientScaleFromDeviceStatus()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], PrecisionMode.Bfp8, () =>
        {
            Parameter parameter = CreateParameter(
                PrecisionMode.Bfp8,
                Values(129, 7, 0.3f),
                [3, 43],
                "weight",
                [0]);
            using var optimizer = new GainShareAdamW(
                [[parameter]], Options());
            try
            {
                PublishGradient(
                    parameter,
                    Values(129, 23, 0.05f),
                    PrecisionMode.Bfp8,
                    [0]);
                CudaBfp8BufferView gradient =
                    parameter.T.EnsureCudaBfp8GradientBuffer(0);
                gradient.Scales.CopyFromCPU([float.NaN]);

                InvalidOperationException failure = Assert.Throws<
                    InvalidOperationException>(() => optimizer.step());

                Assert.Contains("Non-finite CUDA value", failure.Message);
            }
            finally
            {
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void CudaToCpuTransitionContinuesFromResidentMoments()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], PrecisionMode.Float32, () =>
        {
            float[] initial = Values(257, 3, 0.35f);
            float[] firstGradient = Values(257, 19, 0.065f);
            float[] secondGradient = Values(257, 41, 0.045f);
            Parameter transitionedParameter = CreateParameter(
                PrecisionMode.Float32,
                initial,
                [257],
                "weight",
                [0]);
            var referenceParameter = new Parameter(
                initial.ToArray(),
                [257],
                "weight",
                WeightDecayPolicy.Apply);
            using var transitioned = new GainShareAdamW(
                [[transitionedParameter]], Options());
            using var reference = new GainShareAdamW(
                [[referenceParameter]], Options());
            try
            {
                PublishGradient(
                    transitionedParameter,
                    firstGradient,
                    PrecisionMode.Float32,
                    [0]);
                transitioned.step();

                Tensor.ExecutionDevice = TensorDevice.Cpu;
                referenceParameter.T.BackwardAndRelease(firstGradient);
                reference.step();
                transitioned.zero_grad();
                reference.zero_grad();
                transitionedParameter.T.BackwardAndRelease(secondGradient);
                referenceParameter.T.BackwardAndRelease(secondGradient);

                transitioned.step();
                reference.step();

                GainShareAdamWState expected = reference.CaptureState();
                GainShareAdamWState actual = transitioned.CaptureState();
                Assert.Equal(2, actual.Step);
                AssertClose(
                    expected.ParameterStates[0].FirstMoment,
                    actual.ParameterStates[0].FirstMoment,
                    2e-5f);
                AssertClose(
                    expected.ParameterStates[0].SecondMoment,
                    actual.ParameterStates[0].SecondMoment,
                    2e-5f);
                AssertClose(
                    referenceParameter.T.Data.ToArray(),
                    transitionedParameter.T.Data.ToArray(),
                    5e-4f);
            }
            finally
            {
                transitionedParameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Theory]
    [MemberData(nameof(Precisions))]
    public void TwoGpuStepPublishesEqualReplicas(PrecisionMode mode)
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithCuda([0, 1], mode, () =>
        {
            Parameter parameter = CreateParameter(
                mode, Values(515, 13, 0.45f), [5, 103], "weight", [0, 1]);
            using var optimizer = new GainShareAdamW(
                [[parameter]], Options());
            try
            {
                optimizer.prepare();
                PublishGradient(
                    parameter, Values(515, 37, 0.065f), mode, [0, 1]);

                optimizer.step();

                float[] first = ReadParameter(parameter, mode, 0);
                float[] second = ReadParameter(parameter, mode, 1);
                AssertClose(first, second, mode == PrecisionMode.Float32
                    ? 1e-6f
                    : 2e-3f);
            }
            finally
            {
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Theory]
    [MemberData(nameof(Precisions))]
    public void RestoredCheckpointProducesSameNextStep(PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], mode, () =>
        {
            float[] initial = Values(259, 17, 0.42f);
            Parameter uninterruptedParameter = CreateParameter(
                mode, initial, [7, 37], "weight", [0]);
            using var uninterrupted = new GainShareAdamW(
                [[uninterruptedParameter]], Options());
            Parameter? resumedParameter = null;
            GainShareAdamW? resumed = null;
            try
            {
                PublishGradient(
                    uninterruptedParameter,
                    Values(259, 29, 0.07f),
                    mode,
                    [0]);
                uninterrupted.step();
                GainShareAdamWState checkpoint =
                    uninterrupted.CaptureState();
                float[] checkpointWeights =
                    ReadParameter(uninterruptedParameter, mode, 0);

                resumedParameter = CreateParameter(
                    mode, checkpointWeights, [7, 37], "weight", [0]);
                resumed = new GainShareAdamW(
                    [[resumedParameter]], Options());
                resumed.RestoreState(checkpoint);
                float[] nextGradient = Values(259, 53, 0.05f);
                uninterrupted.zero_grad();
                PublishGradient(
                    uninterruptedParameter, nextGradient, mode, [0]);
                PublishGradient(resumedParameter, nextGradient, mode, [0]);

                uninterrupted.step();
                resumed.step();

                AssertClose(
                    ReadParameter(uninterruptedParameter, mode, 0),
                    ReadParameter(resumedParameter, mode, 0),
                    mode == PrecisionMode.Float32 ? 2e-5f : 4e-3f);
                GainShareAdamWState expected = uninterrupted.CaptureState();
                GainShareAdamWState actual = resumed.CaptureState();
                AssertClose(
                    expected.ParameterStates[0].FirstMoment,
                    actual.ParameterStates[0].FirstMoment,
                    mode == PrecisionMode.Float32 ? 2e-6f : 4e-3f);
                AssertClose(
                    expected.ParameterStates[0].SecondMoment,
                    actual.ParameterStates[0].SecondMoment,
                    mode == PrecisionMode.Float32 ? 2e-6f : 4e-3f);
            }
            finally
            {
                resumed?.Dispose();
                resumedParameter?.T.InvalidateCudaBuffers();
                uninterruptedParameter.T.InvalidateCudaBuffers();
            }
        });
    }

    private static GainShareAdamWOptions Options() => new()
    {
        LearningRate = 3e-4f,
        Beta1 = 0.9f,
        Beta2 = 0.95f,
        Epsilon = 1e-8f,
        Rho = 0.9f,
        Gamma = 0.5f,
        MinScale = 0.5f,
        MaxScale = 2f,
        WeightDecay = 0.01f,
        Decay1D = true,
    };

    private static Parameter CreateParameter(
        PrecisionMode mode,
        float[] values,
        int[] shape,
        string name,
        int[] devices)
    {
        var parameter = new Parameter(
            values, shape, name, WeightDecayPolicy.Apply);
        switch (mode)
        {
            case PrecisionMode.BFloat16:
                parameter.T.ConvertStorageInPlace(
                    TensorDType.BFloat16,
                    preserveFloat32Master: false);
                break;
            case PrecisionMode.Mix16_32:
                parameter.T.ConvertStorageInPlace(
                    TensorDType.BFloat16,
                    preserveFloat32Master: true);
                break;
            case PrecisionMode.Bfp8:
                parameter.T.ConvertStorageInPlace(
                    TensorDType.Bfp8,
                    Bfp8QuantizationDescriptor.TensorWide,
                    preserveFloat32Master: false);
                break;
            case PrecisionMode.Mix8_32:
                parameter.T.ConvertStorageInPlace(
                    TensorDType.Bfp8,
                    Bfp8QuantizationDescriptor.Mix8_32,
                    preserveFloat32Master: true);
                break;
        }
        foreach (int device in devices)
        {
            switch (mode)
            {
                case PrecisionMode.BFloat16:
                case PrecisionMode.Mix16_32:
                    _ = parameter.T.EnsureCudaBFloat16Buffer(device);
                    break;
                case PrecisionMode.Bfp8:
                case PrecisionMode.Mix8_32:
                    _ = parameter.T.EnsureCudaBfp8Buffer(device);
                    break;
                default:
                    _ = parameter.T.EnsureCudaFloat32Buffer(device);
                    break;
            }
        }
        parameter.T.to(new TorchDevice(TensorDevice.Cuda, devices[0]));
        return parameter;
    }

    private static void PublishGradient(
        Parameter parameter,
        float[] values,
        PrecisionMode mode,
        int[] devices)
    {
        switch (mode)
        {
            case PrecisionMode.BFloat16:
            {
                var encoded = new ushort[values.Length];
                TensorStorageCodec.EncodeBFloat16(values, encoded);
                foreach (int device in devices)
                {
                    NativeCudaBuffer<ushort> buffer =
                        ForgetMemoryV2Cuda.GetAccelerator(device)
                            .Allocate1D(encoded);
                    parameter.T.AdoptCudaBFloat16GradientBuffer(
                        buffer, device);
                }
                parameter.T.MarkCudaBFloat16GradientsSynchronized(devices);
                break;
            }
            case PrecisionMode.Bfp8:
            {
                Bfp8EncodedStorage encoded =
                    Bfp8QuantizationCodec.Default.Encode(
                        values,
                        Bfp8QuantizationDescriptor.TensorWide);
                foreach (int device in devices)
                {
                    CudaBfp8BufferView target =
                        parameter.T.PrepareCudaBfp8GradientReplica(device);
                    target.Payload.CopyFromCPU(encoded.Payload.Span);
                    target.Scales.CopyFromCPU(encoded.Scales.Span);
                }
                parameter.T.MarkCudaBfp8GradientsSynchronized(devices);
                break;
            }
            default:
                foreach (int device in devices)
                    parameter.T.SetCudaGradient(values, device);
                parameter.T.MarkCudaGradientsSynchronized(devices);
                break;
        }
    }

    private static float[] ReadParameter(
        Parameter parameter,
        PrecisionMode mode,
        int device)
    {
        switch (mode)
        {
            case PrecisionMode.BFloat16:
            {
                NativeCudaBuffer<ushort> source =
                    parameter.T.EnsureCudaBFloat16Buffer(device);
                var encoded = new ushort[source.Length];
                var result = new float[source.Length];
                source.CopyToCPU(encoded);
                TensorStorageCodec.DecodeBFloat16(encoded, result);
                return result;
            }
            case PrecisionMode.Bfp8:
            case PrecisionMode.Mix8_32:
            {
                CudaBfp8BufferView source =
                    parameter.T.EnsureCudaBfp8Buffer(device);
                var payload = new sbyte[source.Payload.Length];
                var scales = new float[source.Scales.Length];
                var result = new float[source.Payload.Length];
                source.Payload.CopyToCPU(payload);
                source.Scales.CopyToCPU(scales);
                Bfp8QuantizationCodec.Default.Decode(
                    payload, scales, source.Descriptor, result);
                return result;
            }
            default:
            {
                NativeCudaBuffer<float> source =
                    parameter.T.EnsureCudaMasterFloat32Buffer(device);
                var result = new float[source.Length];
                source.CopyToCPU(result);
                return result;
            }
        }
    }

    private static void AssertPrecisionResidency(
        Tensor tensor,
        PrecisionMode mode,
        int device)
    {
        switch (mode)
        {
            case PrecisionMode.BFloat16:
            case PrecisionMode.Bfp8:
                Assert.False(tensor.HasCudaMasterFloat32Buffer(device));
                break;
            case PrecisionMode.Mix16_32:
            case PrecisionMode.Mix8_32:
                Assert.True(tensor.HasCudaMasterFloat32Buffer(device));
                break;
        }
    }

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index => MathF.Sin((index + offset) * 0.173f) * scale)
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

    private static void WithCuda(
        int[] devices,
        PrecisionMode mode,
        Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = devices;
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(
                    PrecisionPolicy.For(mode));
            action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }
}
