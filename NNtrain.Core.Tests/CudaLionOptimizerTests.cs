using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

[Collection(TensorSimdCollection.Name)]
public sealed class CudaLionOptimizerTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.BFloat16)]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Bfp8)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void OneStepMatchesStorageAwareReference(TensorPrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], Policy(mode), () =>
        {
            Parameter parameter = CreateParameter(mode, [0]);
            LionOptions options = Options();
            using var optimizer = new Lion([parameter], options);
            try
            {
                float[] initial = Values(257, 3, 0.55f);
                float[] gradient = Values(257, 29, 0.08f);
                (float[] expectedData, float[] expectedMomentum) =
                    ReferenceStep(initial, gradient, mode, options);
                PublishGradient(parameter, mode, gradient, [0]);

                optimizer.step();

                AssertClose(expectedData, parameter.T.Data, 1e-6f);
                AssertClose(
                    expectedMomentum,
                    optimizer.CaptureState().ParameterStates[0].Momentum,
                    1e-6f);
            }
            finally
            {
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.BFloat16)]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Bfp8)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void OneGpuKeepsStateResidentAndCheckpointCurrent(
        TensorPrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], Policy(mode), () =>
        {
            Parameter parameter = CreateParameter(mode, [0]);
            using var optimizer = new Lion([parameter], Options());
            try
            {
                optimizer.prepare();
                PublishGradient(
                    parameter,
                    mode,
                    Values(parameter.T.Numel, 31, 0.09f),
                    [0]);
                optimizer.step();
                optimizer.zero_grad();
                PublishGradient(
                    parameter,
                    mode,
                    Values(parameter.T.Numel, 67, 0.07f),
                    [0]);
                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;

                optimizer.step();

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - before;
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(
                    UsesQuantizedPublication(mode) ? sizeof(int) : 0,
                    transfer.DeviceToHostBytes);
                AssertMasterContract(parameter.T, mode, [0]);

                LionState checkpoint = optimizer.CaptureState();
                Assert.Equal(2, checkpoint.Step);
                Assert.Contains(
                    checkpoint.ParameterStates[0].Momentum,
                    value => value != 0f);
                Assert.All(
                    checkpoint.ParameterStates[0].Momentum,
                    value => Assert.True(float.IsFinite(value)));

                optimizer.RestoreState(checkpoint);
                LionState restored = optimizer.CaptureState();
                Assert.Equal(checkpoint.Step, restored.Step);
                Assert.Equal(
                    checkpoint.ParameterStates[0].Momentum,
                    restored.ParameterStates[0].Momentum);
            }
            finally
            {
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.BFloat16)]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Bfp8)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void TwoGpuReplicasRemainIdenticalWithoutHotPayloadTransfers(
        TensorPrecisionMode mode)
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        int[] devices = [0, 1];
        WithCuda(devices, Policy(mode), () =>
        {
            Parameter parameter = CreateParameter(mode, devices);
            using var optimizer = new Lion([parameter], Options());
            try
            {
                optimizer.prepare();
                PublishGradient(
                    parameter,
                    mode,
                    Values(parameter.T.Numel, 17, 0.08f),
                    devices);
                optimizer.step();
                optimizer.zero_grad();
                PublishGradient(
                    parameter,
                    mode,
                    Values(parameter.T.Numel, 53, 0.06f),
                    devices);
                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;

                optimizer.step();

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - before;
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(
                    UsesQuantizedPublication(mode)
                        ? devices.Length * sizeof(int)
                        : 0,
                    transfer.DeviceToHostBytes);
                AssertReplicaEqual(parameter.T, mode, 0, 1);
                AssertMasterContract(parameter.T, mode, devices);
                LionState checkpoint = optimizer.CaptureState();
                Assert.All(
                    checkpoint.ParameterStates[0].Momentum,
                    value => Assert.True(float.IsFinite(value)));
            }
            finally
            {
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void Mix8RejectsNonFiniteUpdateThroughFusedStatusPass()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], PrecisionPolicy.Mix8_32, () =>
        {
            Parameter parameter = CreateParameter(
                TensorPrecisionMode.Mix8_32,
                [0]);
            using var optimizer = new Lion([parameter], Options());
            try
            {
                optimizer.prepare();
                float[] gradient = Values(parameter.T.Numel, 7, 0.05f);
                PublishGradient(
                    parameter,
                    TensorPrecisionMode.Mix8_32,
                    gradient,
                    [0]);
                gradient[13] = float.NaN;
                parameter.T.EnsureCudaGradientBuffer(0).CopyFromCPU(gradient);

                InvalidOperationException error = Assert.Throws<
                    InvalidOperationException>(() => optimizer.step());

                Assert.Contains("Non-finite", error.Message);
            }
            finally
            {
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    private static LionOptions Options() => new()
    {
        LearningRate = 0.003f,
        Beta1 = 0.8f,
        Beta2 = 0.91f,
        WeightDecay = 0.02f,
        Decay1D = true,
    };

    private static Parameter CreateParameter(
        TensorPrecisionMode mode,
        int[] devices)
    {
        var parameter = new Parameter(
            Values(257, 3, 0.55f),
            [257],
            "lion.weight",
            WeightDecayPolicy.Exclude);
        switch (mode)
        {
            case TensorPrecisionMode.Float32:
                break;
            case TensorPrecisionMode.BFloat16:
                parameter.T.ConvertStorageInPlace(
                    TensorDType.BFloat16,
                    preserveFloat32Master: false);
                break;
            case TensorPrecisionMode.Mix16_32:
                parameter.T.ConvertStorageInPlace(
                    TensorDType.BFloat16,
                    preserveFloat32Master: true);
                break;
            case TensorPrecisionMode.Bfp8:
                parameter.T.ConvertStorageInPlace(
                    TensorDType.Bfp8,
                    Bfp8QuantizationDescriptor.TensorWide,
                    preserveFloat32Master: false);
                break;
            case TensorPrecisionMode.Mix8_32:
                parameter.T.ConvertStorageInPlace(
                    TensorDType.Bfp8,
                    Bfp8QuantizationDescriptor.Mix8_32,
                    preserveFloat32Master: true);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(mode));
        }

        foreach (int device in devices)
        {
            switch (mode)
            {
                case TensorPrecisionMode.Float32:
                    _ = parameter.T.EnsureCudaFloat32Buffer(device);
                    break;
                case TensorPrecisionMode.BFloat16:
                case TensorPrecisionMode.Mix16_32:
                    _ = parameter.T.EnsureCudaBFloat16Buffer(device);
                    break;
                case TensorPrecisionMode.Bfp8:
                case TensorPrecisionMode.Mix8_32:
                    _ = parameter.T.EnsureCudaBfp8Buffer(device);
                    break;
            }
        }
        parameter.T.to(new TorchDevice(TensorDevice.Cuda, devices[0]));
        return parameter;
    }

    private static void PublishGradient(
        Parameter parameter,
        TensorPrecisionMode mode,
        float[] values,
        int[] devices)
    {
        switch (mode)
        {
            case TensorPrecisionMode.BFloat16:
            {
                var encoded = new ushort[values.Length];
                TensorStorageCodec.EncodeBFloat16(values, encoded);
                foreach (int device in devices)
                {
                    if (!parameter.T.TryGetCudaBFloat16GradientBuffer(
                            device,
                            out NativeCudaBuffer<ushort>? buffer))
                    {
                        buffer = ForgetMemoryV2Cuda.GetAccelerator(device)
                            .Allocate1D<ushort>(encoded.Length);
                        parameter.T.AdoptCudaBFloat16GradientBuffer(
                            buffer,
                            device);
                    }
                    buffer!.CopyFromCPU(encoded);
                    buffer.MarkGradientStorageDirty();
                }
                parameter.T.MarkCudaBFloat16GradientsSynchronized(devices);
                break;
            }
            case TensorPrecisionMode.Bfp8:
            {
                Bfp8EncodedStorage encoded =
                    Bfp8QuantizationCodec.Default.Encode(
                        values,
                        Bfp8QuantizationDescriptor.TensorWide);
                foreach (int device in devices)
                {
                    CudaBfp8BufferView view =
                        parameter.T.PrepareCudaBfp8GradientReplica(device);
                    view.Payload.CopyFromCPU(encoded.Payload.Span);
                    view.Scales.CopyFromCPU(encoded.Scales.Span);
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

    private static void AssertReplicaEqual(
        Tensor tensor,
        TensorPrecisionMode mode,
        int first,
        int second)
    {
        switch (mode)
        {
            case TensorPrecisionMode.Float32:
                Assert.Equal(
                    Read(tensor.EnsureCudaFloat32Buffer(first)),
                    Read(tensor.EnsureCudaFloat32Buffer(second)));
                break;
            case TensorPrecisionMode.BFloat16:
            case TensorPrecisionMode.Mix16_32:
                Assert.Equal(
                    Read(tensor.EnsureCudaBFloat16Buffer(first)),
                    Read(tensor.EnsureCudaBFloat16Buffer(second)));
                break;
            case TensorPrecisionMode.Bfp8:
            case TensorPrecisionMode.Mix8_32:
                AssertEncodedEqual(
                    tensor.EnsureCudaBfp8Buffer(first),
                    tensor.EnsureCudaBfp8Buffer(second));
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(mode));
        }
    }

    private static void AssertMasterContract(
        Tensor tensor,
        TensorPrecisionMode mode,
        int[] devices)
    {
        bool expected = mode is TensorPrecisionMode.Mix16_32
            or TensorPrecisionMode.Mix8_32;
        foreach (int device in devices)
        {
            Assert.Equal(
                expected,
                tensor.HasCudaMasterFloat32Buffer(device));
        }
    }

    private static void AssertEncodedEqual(
        CudaBfp8BufferView expected,
        CudaBfp8BufferView actual)
    {
        Assert.Equal(Read(expected.Payload), Read(actual.Payload));
        Assert.Equal(Read(expected.Scales), Read(actual.Scales));
    }

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index => MathF.Sin((index + offset) * 0.173f) * scale)
            .ToArray();

    private static (float[] Data, float[] Momentum) ReferenceStep(
        float[] initial,
        float[] gradient,
        TensorPrecisionMode mode,
        LionOptions options)
    {
        float[] data = mode switch
        {
            TensorPrecisionMode.BFloat16 => RoundBFloat16(initial),
            TensorPrecisionMode.Bfp8 => RoundBfp8(initial),
            _ => initial.ToArray(),
        };
        float[] effectiveGradient = mode switch
        {
            TensorPrecisionMode.BFloat16 => RoundBFloat16(gradient),
            TensorPrecisionMode.Bfp8 => RoundBfp8(gradient),
            _ => gradient.ToArray(),
        };
        var momentum = new float[data.Length];
        for (int index = 0; index < data.Length; index++)
        {
            float direction = (1f - options.Beta1)
                * effectiveGradient[index];
            float sign = direction > 0f
                ? 1f
                : direction < 0f
                    ? -1f
                    : direction;
            data[index] -= options.LearningRate
                * options.WeightDecay
                * data[index];
            data[index] -= options.LearningRate * sign;
            momentum[index] = (1f - options.Beta2)
                * effectiveGradient[index];
        }
        return mode switch
        {
            TensorPrecisionMode.BFloat16 =>
                (RoundBFloat16(data), RoundBFloat16(momentum)),
            TensorPrecisionMode.Mix16_32 =>
                (RoundBFloat16(data), momentum),
            TensorPrecisionMode.Bfp8 =>
                (RoundBfp8(data), RoundBfp8(momentum)),
            TensorPrecisionMode.Mix8_32 =>
                (RoundBfp8(data, Bfp8QuantizationDescriptor.Mix8_32),
                    momentum),
            _ => (data, momentum),
        };
    }

    private static float[] RoundBFloat16(IReadOnlyList<float> values)
    {
        var result = new float[values.Count];
        for (int index = 0; index < result.Length; index++)
        {
            result[index] = TensorStorageCodec.DecodeBFloat16(
                TensorStorageCodec.EncodeBFloat16(values[index]));
        }
        return result;
    }

    private static float[] RoundBfp8(float[] values)
        => RoundBfp8(values, Bfp8QuantizationDescriptor.TensorWide);

    private static float[] RoundBfp8(
        float[] values,
        Bfp8QuantizationDescriptor descriptor)
    {
        Bfp8EncodedStorage encoded = Bfp8QuantizationCodec.Default.Encode(
            values,
            descriptor);
        var result = new float[values.Length];
        Bfp8QuantizationCodec.Default.Decode(
            encoded.Payload.Span,
            encoded.Scales.Span,
            encoded.Descriptor,
            result);
        return result;
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

    private static bool UsesQuantizedPublication(TensorPrecisionMode mode)
        => mode is TensorPrecisionMode.Bfp8
            or TensorPrecisionMode.Mix8_32;

    private static PrecisionPolicy Policy(TensorPrecisionMode mode)
        => mode switch
        {
            TensorPrecisionMode.Float32 => PrecisionPolicy.Float32,
            TensorPrecisionMode.BFloat16 => PrecisionPolicy.BFloat16,
            TensorPrecisionMode.Mix16_32 => PrecisionPolicy.Mix16_32,
            TensorPrecisionMode.Bfp8 => PrecisionPolicy.Bfp8,
            TensorPrecisionMode.Mix8_32 => PrecisionPolicy.Mix8_32,
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };

    private static T[] Read<T>(NativeCudaBuffer<T> buffer)
        where T : unmanaged
    {
        var values = new T[buffer.Length];
        buffer.CopyToCPU(values);
        return values;
    }

    private static void WithCuda(
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
}
