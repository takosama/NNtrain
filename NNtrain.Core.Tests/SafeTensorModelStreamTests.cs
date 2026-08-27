using NNtrain;
using Xunit;

public sealed class SafeTensorModelStreamTests
{
    [Theory]
    [InlineData(TensorDType.Float32)]
    [InlineData(TensorDType.Float16)]
    [InlineData(TensorDType.BFloat16)]
    public void StreamedModelWriterIsByteIdenticalToLegacyWriter(
        TensorDType dtype)
    {
        var model = new StreamingModule(
            dtype,
            [0f, -0f, 0.1f, -2.5f, 31.125f],
            [MathF.PI, -0.00006103515625f, 7f]);
        string legacyPath = TemporaryPath("legacy");
        string streamedPath = TemporaryPath("streamed");
        var observedChunks = new List<int>();

        try
        {
            SafeTensorFile.Save(model.state_dict(), legacyPath);
            SafeTensorFile.SaveModel(
                model,
                streamedPath,
                observedChunks.Add);

            Assert.Equal(
                File.ReadAllBytes(legacyPath),
                File.ReadAllBytes(streamedPath));
            Assert.NotEmpty(observedChunks);
            Assert.All(
                observedChunks,
                bytes => Assert.InRange(
                    bytes,
                    1,
                    CheckpointFloatStagingBuffer.MaximumByteLength));
        }
        finally
        {
            DeleteIfExists(legacyPath);
            DeleteIfExists(streamedPath);
        }
    }

    [Theory]
    [InlineData(TensorDType.Float32)]
    [InlineData(TensorDType.Float16)]
    [InlineData(TensorDType.BFloat16)]
    public void StreamedModelReaderRestoresWithoutModuleState(
        TensorDType dtype)
    {
        var source = new StreamingModule(
            dtype,
            [0.25f, -2f, 18.75f, 0.00390625f],
            [1f, 2f, 3f]);
        var destination = new StreamingModule(
            dtype,
            [-100f, -100f, -100f, -100f],
            [-200f, -200f, -200f]);
        string path = TemporaryPath("reader");
        var observedChunks = new List<int>();

        try
        {
            SafeTensorFile.SaveModel(source, path);
            ModuleState expected = SafeTensorFile.Load(path);

            SafeTensorFile.LoadModel(
                path,
                destination,
                stagingChunkObserved: observedChunks.Add);

            ModuleState actual = destination.state_dict();
            Assert.Equal(expected.Parameters.Length, actual.Parameters.Length);
            for (int index = 0; index < expected.Parameters.Length; index++)
            {
                Assert.Equal(
                    expected.Parameters[index].Values,
                    actual.Parameters[index].Values);
            }
            Assert.NotEmpty(observedChunks);
            Assert.All(
                observedChunks,
                bytes => Assert.InRange(
                    bytes,
                    1,
                    CheckpointFloatStagingBuffer.MaximumByteLength));
        }
        finally
        {
            DeleteIfExists(path);
        }
    }

    [Fact]
    public void CheckpointStagingRejectsMoreThanSixteenMebibytes()
    {
        using var staging = new CheckpointFloatStagingBuffer();
        Assert.Throws<ArgumentOutOfRangeException>(
            () => staging.GetManagedSpan(
                CheckpointFloatStagingBuffer.MaximumElementCount + 1));
    }

    [Fact]
    public void Float32ArtifactOverrideMatchesRelabeledMixedState()
    {
        var model = new StreamingModule(
            TensorDType.BFloat16,
            [0.1f, -2.25f, MathF.PI],
            [7.5f]);
        ModuleState relabeled = model.state_dict() with
        {
            Parameters = model.state_dict().Parameters
                .Select(parameter => parameter with
                {
                    DType = TensorDType.Float32,
                })
                .ToArray(),
        };
        string legacyPath = TemporaryPath("override-legacy");
        string streamedPath = TemporaryPath("override-streamed");
        try
        {
            SafeTensorFile.Save(relabeled, legacyPath);
            SafeTensorFile.SaveModel(
                model,
                streamedPath,
                artifactDTypeOverride: TensorDType.Float32);

            Assert.Equal(
                File.ReadAllBytes(legacyPath),
                File.ReadAllBytes(streamedPath));
        }
        finally
        {
            DeleteIfExists(legacyPath);
            DeleteIfExists(streamedPath);
        }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Bfp8, 128)]
    [InlineData(TensorPrecisionMode.Mix8_32, 3)]
    public void Bfp8ModelsRoundTripThroughLogicalFloat32Artifacts(
        TensorPrecisionMode mode,
        int blockSize)
    {
        var source = new StreamingModule(
            TensorDType.Float32,
            [0.1f, -2.25f, MathF.PI, 0f, 7.5f, -0.75f, 1.25f],
            [9f, -4f, 0.125f, 0f]);
        var destination = new StreamingModule(
            TensorDType.Float32,
            Enumerable.Repeat(-10f, 7).ToArray(),
            Enumerable.Repeat(-20f, 4).ToArray());
        source.to(mode, blockSize);
        destination.to(mode, blockSize);
        ModuleState expected = source.state_dict();
        string path = TemporaryPath("bfp8-f32");
        var chunks = new List<int>();
        try
        {
            SafeTensorFile.SaveModel(
                source,
                path,
                artifactDTypeOverride: TensorDType.Float32);
            ModuleState artifact = SafeTensorFile.Load(path);
            Assert.All(
                artifact.Parameters,
                parameter => Assert.Equal(
                    TensorDType.Float32,
                    parameter.DType));

            SafeTensorFile.LoadModel(
                path,
                destination,
                stagingChunkObserved: chunks.Add);
            ModuleState actual = destination.state_dict();
            Assert.Equal(expected.Parameters.Length, actual.Parameters.Length);
            for (int index = 0; index < expected.Parameters.Length; index++)
            {
                Assert.Equal(
                    expected.Parameters[index].Values,
                    actual.Parameters[index].Values);
                Bfp8QuantizationDescriptor descriptor =
                    destination.parameters().ElementAt(index).T
                        .Bfp8Quantization!;
                Assert.Equal(
                    mode == TensorPrecisionMode.Bfp8
                        ? Bfp8ScaleGranularity.Tensor
                        : Bfp8ScaleGranularity.Block,
                    descriptor.Granularity);
                if (mode == TensorPrecisionMode.Mix8_32)
                    Assert.Equal(blockSize, descriptor.BlockSize);
            }
            Assert.NotEmpty(chunks);
            Assert.All(
                chunks,
                bytes => Assert.InRange(
                    bytes,
                    1,
                    CheckpointFloatStagingBuffer.MaximumByteLength));
        }
        finally
        {
            DeleteIfExists(path);
        }
    }

    private static string TemporaryPath(string prefix)
        => Path.Combine(
            Path.GetTempPath(),
            $"nntrain-{prefix}-{Guid.NewGuid():N}.safetensors");

    private static void DeleteIfExists(string path)
    {
        if (File.Exists(path))
            File.Delete(path);
    }

    private sealed class StreamingModule : Module
    {
        internal StreamingModule(
            TensorDType dtype,
            float[] first,
            float[] second)
            : base(dtype)
        {
            RegisterParameter(
                new Parameter(
                    first,
                    [first.Length],
                    "first",
                    WeightDecayPolicy.Apply,
                    dtype));
            RegisterParameter(
                new Parameter(
                    second,
                    [second.Length],
                    "second",
                    WeightDecayPolicy.Exclude,
                    dtype));
        }
    }
}
