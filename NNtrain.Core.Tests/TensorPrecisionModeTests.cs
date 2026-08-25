using NNtrain;
using Xunit;

public sealed class TensorPrecisionModeTests
{
    [Fact]
    public void BFloat16StorageUsesFloat32Accumulation()
    {
        var tensor = new Tensor([1f, -2f], [2], dtype: TensorDType.BFloat16);

        Assert.Equal(TensorDType.BFloat16, tensor.ComputeDType);
        Assert.Equal(TensorDType.Float32, tensor.AccumulationDType);
    }

    [Theory]
    [InlineData("float32", TensorPrecisionMode.Float32, TensorDType.Float32)]
    [InlineData("bfloat16", TensorPrecisionMode.BFloat16, TensorDType.BFloat16)]
    [InlineData("mix16_32", TensorPrecisionMode.Mix16_32, TensorDType.BFloat16)]
    public void CanonicalNamesMapToOneStorageContract(
        string name,
        TensorPrecisionMode expectedMode,
        TensorDType expectedStorage)
    {
        TensorPrecisionMode mode = TensorPrecisionModeNames.Parse(name);

        Assert.Equal(expectedMode, mode);
        Assert.Equal(expectedStorage, mode.ToStorageDType());
        Assert.Equal(name, TensorPrecisionModeNames.Format(mode));
    }

    [Theory]
    [InlineData(TensorDType.Float32, TensorPrecisionMode.Float32)]
    [InlineData(TensorDType.BFloat16, TensorPrecisionMode.BFloat16)]
    [InlineData(TensorDType.Float16, TensorPrecisionMode.Mix16_32)]
    public void RawStorageDTypesHaveStableDefaultModes(
        TensorDType dtype,
        TensorPrecisionMode expectedMode)
        => Assert.Equal(expectedMode, dtype.ToPrecisionMode());

    [Fact]
    public void PrecisionModeParserRejectsStorageAliases()
    {
        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => TensorPrecisionModeNames.Parse("float16"));

        Assert.Contains("mix16_32", exception.Message);
    }

    [Fact]
    public void ModulePrecisionModePropagatesAndRejectsWrongStorage()
    {
        var model = new GptRinWikiJp(
            vocabularySize: 32,
            contextLength: 2,
            dModel: 4,
            numHeads: 1,
            dHidden: 8,
            numLayers: 1,
            rng: new Random(7),
            dtype: TensorDType.BFloat16);

        model.SetPrecisionMode(TensorPrecisionMode.Mix16_32);

        Assert.Equal(TensorPrecisionMode.Mix16_32, model.PrecisionMode);
        Assert.All(
            model.Parameters(),
            parameter => Assert.Equal(
                TensorDType.BFloat16,
                parameter.T.DType));
        Assert.Throws<InvalidOperationException>(
            () => model.SetPrecisionMode(TensorPrecisionMode.Float32));
    }
}
