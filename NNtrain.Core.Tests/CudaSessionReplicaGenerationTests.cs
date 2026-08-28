using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaSessionReplicaGenerationTests
{
    [Fact]
    public void AuthoritativeWeightSurvivesLaneDisposalAndModelReuse()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        var model = new Linear(4, 3, new Random(907), initScale: 0.03f);
        float[] updated = Enumerable.Range(0, model.W.T.Numel)
            .Select(index => 0.125f + index * 0.03125f)
            .ToArray();
        NativeCudaTransferTelemetry beforeRetainedStep;
        NativeCudaTransferTelemetry afterRetainedStep;

        using var first = CreateSession(PrecisionPolicy.Float32);
        using (IDisposable scope = first.Enter())
        {
            model.to(new TorchDevice(TensorDevice.Cuda, 0));
            NativeCudaBuffer<float> weight =
                model.W.T.EnsureCudaFloat32Buffer(0);
            beforeRetainedStep = NativeCudaRuntime.TransferTelemetry;
            weight.CopyFromCPU(updated);
            model.W.T.MarkCudaDataMutated(0);
            afterRetainedStep = NativeCudaRuntime.TransferTelemetry;

            Assert.Equal(
                beforeRetainedStep.DeviceToHostCopyCount,
                afterRetainedStep.DeviceToHostCopyCount);
            Assert.Equal(
                beforeRetainedStep.DeviceToHostBytes,
                afterRetainedStep.DeviceToHostBytes);
            Assert.Equal(first.Generation, weight.SessionGeneration);
            Assert.True(weight.IsAlive);
        }

        // This used to close the lane memory lease while the tensor retained
        // the same dataVersion in its replica dictionary. The next to(cuda)
        // returned that closed pointer, and Data/to(cpu) could no longer
        // recover the optimizer-produced value.
        first.Dispose();

        Assert.Equal(TensorDevice.Cpu, model.W.T.Device);
        Assert.Equal(0, model.W.T.Value.Replicas.DataReplicaCount);
        Assert.Empty(model.W.T.Value.Replicas.SessionRegistrations);
        AssertClose(updated, model.W.T.Data, 0f);

        using var second = CreateSession(PrecisionPolicy.Float32);
        using (IDisposable scope = second.Enter())
        {
            model.to(new TorchDevice(TensorDevice.Cuda, 0));
            NativeCudaBuffer<float> recreated =
                model.W.T.EnsureCudaFloat32Buffer(0);
            var roundTrip = new float[updated.Length];
            recreated.CopyToCPU(roundTrip);

            Assert.NotEqual(first.Generation, recreated.SessionGeneration);
            Assert.Equal(second.Generation, recreated.SessionGeneration);
            AssertClose(updated, roundTrip, 0f);
        }
        second.Dispose();

        model.to(TensorDevice.Cpu);
        Assert.Empty(model.W.T.Value.Replicas.SessionRegistrations);
        AssertClose(updated, model.W.T.Data, 0f);
    }

    [Theory]
    [InlineData(PrecisionMode.BFloat16)]
    [InlineData(PrecisionMode.Mix16_32)]
    [InlineData(PrecisionMode.Bfp8)]
    [InlineData(PrecisionMode.Mix8_32)]
    public void SessionRetirementPreservesPhysicalOrMasterAuthority(
        PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        float[] initial = [0.25f, -0.5f, 0.75f, -1f];
        float[] updated = [1.25f, -1.5f, 1.75f, -2f];
        Tensor tensor = CreateTensor(initial, mode);
        PrecisionPolicy policy = PrecisionPolicy.For(mode);
        using var session = CreateSession(policy);
        using (IDisposable scope = session.Enter())
        {
            tensor.to(new TorchDevice(TensorDevice.Cuda, 0));
            if (mode == PrecisionMode.BFloat16)
            {
                var encoded = new ushort[updated.Length];
                TensorStorageCodec.EncodeBFloat16(updated, encoded);
                tensor.EnsureCudaBFloat16Buffer(0).CopyFromCPU(encoded);
            }
            else if (mode == PrecisionMode.Bfp8)
            {
                Bfp8EncodedStorage encoded = Bfp8QuantizationCodec.Default
                    .Encode(updated, Bfp8QuantizationDescriptor.TensorWide);
                CudaBfp8BufferView view = tensor.EnsureCudaBfp8Buffer(0);
                view.Payload.CopyFromCPU(encoded.Payload.Span);
                view.Scales.CopyFromCPU(encoded.Scales.Span);
            }
            else
            {
                tensor.EnsureCudaMasterFloat32Buffer(0).CopyFromCPU(updated);
            }
            tensor.MarkCudaDataMutated(0);
        }

        session.Dispose();

        float tolerance = mode is PrecisionMode.BFloat16
            or PrecisionMode.Bfp8
            or PrecisionMode.Mix8_32
            ? 0.02f
            : 0f;
        Assert.Equal(TensorDevice.Cpu, tensor.Device);
        AssertClose(updated, tensor.Data, tolerance);

        using ExecutionSession next = CreateSession(policy);
        using IDisposable nextScope = next.Enter();
        tensor.to(new TorchDevice(TensorDevice.Cuda, 0));
        Assert.Equal([0], tensor.GetResidentCudaDeviceIndices());
    }

    [Fact]
    public void AuthoritativeGradientSurvivesSessionRetirement()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        Tensor tensor = new([1f, 2f, 3f, 4f], [4]);
        float[] expectedGradient = [0.5f, -0.25f, 1.5f, -2f];
        using var session = CreateSession(PrecisionPolicy.Float32);
        using (IDisposable scope = session.Enter())
        {
            tensor.to(new TorchDevice(TensorDevice.Cuda, 0));
            tensor.SetCudaGradient(expectedGradient, 0);
            Assert.Equal(1, tensor.Value.Replicas.GradientReplicaCount);
        }

        session.Dispose();

        Assert.Equal(0, tensor.Value.Replicas.GradientReplicaCount);
        Assert.Empty(tensor.Value.Replicas.SessionRegistrations);
        AssertClose(expectedGradient, tensor.Grad, 0f);

        using ExecutionSession next = CreateSession(PrecisionPolicy.Float32);
        using IDisposable nextScope = next.Enter();
        NativeCudaBuffer<float> restored = tensor.EnsureCudaGradientBuffer(0);
        var roundTrip = new float[expectedGradient.Length];
        restored.CopyToCPU(roundTrip);
        AssertClose(expectedGradient, roundTrip, 0f);
    }

    [Fact]
    public void TwoGpuReplicaGenerationRetiresAsOneLogicalValue()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        Tensor tensor = new([1f, 2f, 3f, 4f], [4]);
        float[] updated = [5f, 6f, 7f, 8f];
        using var session = new ExecutionSession(
            new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(0, 1),
                Precision = PrecisionPolicy.Float32,
            },
            [
                CudaExecutionLaneFactory.Create(0),
                CudaExecutionLaneFactory.Create(1),
            ]);
        using (IDisposable scope = session.Enter())
        {
            tensor.EnsureCudaFloat32Buffer(0).CopyFromCPU(updated);
            tensor.EnsureCudaFloat32Buffer(1).CopyFromCPU(updated);
            tensor.MarkCudaDataReplicasSynchronized([0, 1]);
            Assert.Equal([0, 1], tensor.GetResidentCudaDeviceIndices());
        }

        session.Dispose();

        Assert.Empty(tensor.GetResidentCudaDeviceIndices());
        Assert.Equal(0, tensor.Value.Replicas.DataReplicaCount);
        Assert.Empty(tensor.Value.Replicas.SessionRegistrations);
        AssertClose(updated, tensor.Data, 0f);
    }

    private static Tensor CreateTensor(float[] values, PrecisionMode mode)
        => mode switch
        {
            PrecisionMode.BFloat16 or PrecisionMode.Mix16_32 => new Tensor(
                values,
                [values.Length],
                dtype: TensorDType.BFloat16),
            PrecisionMode.Mix8_32 => Tensor.FromBfp8(
                values,
                [values.Length],
                Bfp8QuantizationDescriptor.Mix8_32),
            PrecisionMode.Bfp8 => Tensor.FromBfp8(
                values,
                [values.Length],
                Bfp8QuantizationDescriptor.TensorWide),
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };

    private static ExecutionSession CreateSession(PrecisionPolicy precision)
        => new(
            new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(0),
                Precision = precision,
            },
            [CudaExecutionLaneFactory.Create(0)]);

    private static void AssertClose(
        IReadOnlyList<float> expected,
        IReadOnlyList<float> actual,
        float tolerance)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int index = 0; index < expected.Count; index++)
        {
            Assert.InRange(
                Math.Abs(expected[index] - actual[index]),
                0f,
                tolerance);
        }
    }
}
