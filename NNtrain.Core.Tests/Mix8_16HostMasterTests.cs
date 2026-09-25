using System.Buffers.Binary;
using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class Mix8_16HostMasterTests
{
    [Fact]
    public void ModuleConversionAndStateRestoreKeepOnlyPackedHostMasters()
    {
        var model = new Linear(4, 3, new Random(42));
        model.to(TensorPrecisionMode.Mix8_16, 32);

        ModuleState state = model.state_dict();
        foreach (Parameter parameter in model.parameters())
        {
            Assert.Equal(parameter.T.Numel * sizeof(ushort), parameter.T.HostMasterByteLength);
            Assert.True(parameter.T.HasBFloat16HostMaster);
            Assert.False(parameter.T.HasFloat32HostMaster);
            Assert.All(parameter.T.CaptureData(preferMaster: true), value =>
                Assert.Equal(value, TensorStorageCodec.RoundToBFloat16(value)));
            Assert.Equal(parameter.T.Numel * sizeof(ushort), parameter.T.HostMasterByteLength);
        }

        model.load_state_dict(state);
        foreach (Parameter parameter in model.parameters())
        {
            Assert.Equal(parameter.T.Numel * sizeof(ushort), parameter.T.HostMasterByteLength);
            Assert.True(parameter.T.HasBFloat16HostMaster);
            Assert.False(parameter.T.HasFloat32HostMaster);
        }
    }

    [Fact]
    public void CheckpointStreamRestoresBFloat16MasterWithoutRetainingFloatCopy()
    {
        var model = new Linear(4, 3, new Random(9));
        model.to(TensorPrecisionMode.Mix8_16, 32);
        Tensor tensor = model.parameters().First().T;
        float[] values = Enumerable.Range(0, tensor.Numel)
            .Select(index => 0.1234567f + index * 0.01731f).ToArray();

        using (Tensor.CheckpointRestoreWriter writer = tensor.BeginCheckpointRestore())
        {
            int middle = values.Length / 2;
            writer.WriteNext(values.AsSpan(0, middle));
            writer.WriteNext(values.AsSpan(middle));
            writer.Complete();
        }

        using var staging = new CheckpointFloatStagingBuffer();
        float[] restored = tensor.CopyCheckpointRangeTo(0, tensor.Numel, staging,
            preferMaster: true).ToArray();
        Assert.Equal(values.Select(TensorStorageCodec.RoundToBFloat16), restored);
        Assert.Equal(tensor.Numel * sizeof(ushort), tensor.HostMasterByteLength);
        Assert.True(tensor.HasBFloat16HostMaster);
        Assert.False(tensor.HasFloat32HostMaster);
    }

    [Fact]
    public void Float32CheckpointArtifactRoundTripsIntoPackedHostMaster()
    {
        var source = new Linear(4, 3, new Random(14));
        source.to(TensorPrecisionMode.Mix8_16, 32);
        var destination = new Linear(4, 3, new Random(29));
        destination.to(TensorPrecisionMode.Mix8_16, 32);
        string path = Path.Combine(Path.GetTempPath(), $"nntrain-mix8-16-host-{Guid.NewGuid():N}.safetensors");
        try
        {
            SafeTensorFile.SaveModel(source, path, artifactDTypeOverride: TensorDType.Float32);
            SafeTensorFile.LoadModel(path, destination);
            ModuleState expected = source.state_dict();
            ModuleState actual = destination.state_dict();
            Assert.Equal(expected.Parameters.Length, actual.Parameters.Length);
            for (int index = 0; index < expected.Parameters.Length; index++)
                Assert.Equal(expected.Parameters[index].Values, actual.Parameters[index].Values);
            foreach (Parameter parameter in destination.parameters())
            {
                Assert.True(parameter.T.HasBFloat16HostMaster);
                Assert.False(parameter.T.HasFloat32HostMaster);
            }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void BFloat16CheckpointArtifactUsesTwoBytesAndRoundTripsPackedMaster()
    {
        var source = new Linear(4, 3, new Random(51));
        source.to(TensorPrecisionMode.Mix8_16, 32);
        var destination = new Linear(4, 3, new Random(52));
        destination.to(TensorPrecisionMode.Mix8_16, 32);
        string path = Path.Combine(Path.GetTempPath(), $"nntrain-mix8-16-bf16-{Guid.NewGuid():N}.safetensors");
        try
        {
            var observedStagingBytes = new List<int>();
            SafeTensorFile.SaveModel(source, path, observedStagingBytes.Add,
                artifactDTypeOverride: TensorDType.BFloat16);
            long expectedPayloadBytes = source.parameters()
                .Sum(parameter => (long)parameter.T.Numel * sizeof(ushort));
            Assert.Equal(expectedPayloadBytes, observedStagingBytes.Sum(bytes => (long)bytes));
            using (var stream = File.OpenRead(path))
            {
                Span<byte> prefix = stackalloc byte[sizeof(ulong)];
                stream.ReadExactly(prefix);
                ulong headerLength = BinaryPrimitives.ReadUInt64LittleEndian(prefix);
                long payloadBytes = stream.Length - sizeof(ulong) - checked((long)headerLength);
                Assert.Equal(expectedPayloadBytes, payloadBytes);
            }
            ModuleState artifact = SafeTensorFile.Load(path);
            ModuleState sourceState = source.state_dict();
            Assert.All(artifact.Parameters, parameter => Assert.Equal(TensorDType.BFloat16, parameter.DType));
            for (int index = 0; index < artifact.Parameters.Length; index++)
                Assert.Equal(sourceState.Parameters[index].Values, artifact.Parameters[index].Values);

            SafeTensorFile.LoadModel(path, destination);
            ModuleState restored = destination.state_dict();
            for (int index = 0; index < restored.Parameters.Length; index++)
                Assert.Equal(sourceState.Parameters[index].Values, restored.Parameters[index].Values);
            foreach (Parameter parameter in destination.parameters())
            {
                Assert.True(parameter.T.HasBFloat16HostMaster);
                Assert.False(parameter.T.HasFloat32HostMaster);
            }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void ArcMasterUploadAndSessionClosePreservePackedHostAuthority()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");
        var model = new Linear(4, 3, new Random(36));
        model.to(TensorPrecisionMode.Mix8_16, 32);
        Tensor tensor = model.parameters().First().T;
        using (Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_16))
        {
            using ArcExecutionLane.ArcBuffer master = tensor.ArcBFloat16Master().Borrow();
            Assert.Equal(tensor.Numel * sizeof(ushort), master.ByteLength);
            Assert.True(tensor.HasBFloat16HostMaster);
            Assert.False(tensor.HasFloat32HostMaster);
            tensor.CompleteArcUpdate();
        }
        Assert.True(tensor.HasBFloat16HostMaster);
        Assert.False(tensor.HasFloat32HostMaster);
        Assert.Equal(tensor.Numel * sizeof(ushort), tensor.HostMasterByteLength);
    }
}
