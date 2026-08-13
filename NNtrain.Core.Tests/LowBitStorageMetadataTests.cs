using System.Text.Json;
using NNtrain;
using Xunit;

public sealed class LowBitStorageMetadataTests
{
    [Fact]
    public void NativeFormatsAndReservedFloat8HaveExplicitPayloadSizes()
    {
        var float32 = new TensorStorageDescriptor(TensorDType.Float32);
        var float16 = new TensorStorageDescriptor(TensorDType.Float16);
        var float8 = new TensorStorageDescriptor(TensorDType.Float8E4M3Fn);

        Assert.Equal(28, float32.GetPayloadByteLength(7));
        Assert.Equal(14, float16.GetPayloadByteLength(7));
        Assert.Equal(7, float8.GetPayloadByteLength(7));
        Assert.True(float32.IsSupportedByCurrentRuntime);
        Assert.True(float16.IsSupportedByCurrentRuntime);
        Assert.False(float8.IsSupportedByCurrentRuntime);
    }

    [Fact]
    public void PackedBlockFloat4AccountsForValuesAndScaleSidecars()
    {
        var descriptor = new TensorStorageDescriptor(
            TensorDType.Float4,
            new TensorStorageMetadata(
                TensorStorageEncoding.PackedBlockQuantized,
                new TensorPackingMetadata(BitsPerValue: 4),
                new TensorQuantizationMetadata(
                    TensorQuantizationScheme.Symmetric,
                    BlockSize: 4,
                    Scales: [0.25f, 0.5f, 1f])));

        descriptor.Validate(10);

        Assert.Equal(5, descriptor.GetPayloadByteLength(10));
        Assert.Equal(3 * sizeof(float), descriptor.GetAuxiliaryByteLength(10));
        Assert.Equal(17, descriptor.GetTotalByteLength(10));
        Assert.False(descriptor.IsSupportedByCurrentRuntime);
    }

    [Fact]
    public void PackedAffineFloat2IncludesOneZeroPointPerBlock()
    {
        var descriptor = new TensorStorageDescriptor(
            TensorDType.Float2,
            new TensorStorageMetadata(
                TensorStorageEncoding.PackedBlockQuantized,
                new TensorPackingMetadata(BitsPerValue: 2),
                new TensorQuantizationMetadata(
                    TensorQuantizationScheme.Affine,
                    BlockSize: 4,
                    Scales: [0.25f, 0.5f, 1f],
                    ZeroPoints: [1, 2, 3])));

        Assert.Equal(3, descriptor.GetPayloadByteLength(10));
        Assert.Equal(
            3L * (sizeof(float) + sizeof(int)),
            descriptor.GetAuxiliaryByteLength(10));
        Assert.Equal(27, descriptor.GetTotalByteLength(10));
    }

    [Fact]
    public void TernaryUsesTwoBitCodesAndReportsLog2OfThreeTarget()
    {
        var descriptor = new TensorStorageDescriptor(
            TensorDType.Ternary1Bit58,
            new TensorStorageMetadata(
                TensorStorageEncoding.PackedBlockQuantized,
                new TensorPackingMetadata(
                    BitsPerValue: 2,
                    EffectiveBitsPerValue: Math.Log2(3d)),
                new TensorQuantizationMetadata(
                    TensorQuantizationScheme.Ternary,
                    BlockSize: 8,
                    Scales: [0.25f, 0.5f])));

        descriptor.Validate(9);

        Assert.Equal(3, descriptor.GetPayloadByteLength(9));
        Assert.Equal(
            Math.Log2(3d),
            descriptor.EffectiveMetadata.Packing!.LogicalBitsPerValue,
            precision: 12);
    }

    [Fact]
    public void InvalidLayoutsAreRejectedBeforeAnyFutureCodecIsSelected()
    {
        Assert.Throws<ArgumentException>(() => new TensorStorageDescriptor(
            TensorDType.Float4).Validate(1));

        var invalidAffine = new TensorStorageDescriptor(
            TensorDType.Float2,
            new TensorStorageMetadata(
                TensorStorageEncoding.PackedBlockQuantized,
                new TensorPackingMetadata(BitsPerValue: 2),
                new TensorQuantizationMetadata(
                    TensorQuantizationScheme.Affine,
                    BlockSize: 4,
                    Scales: [1f, 1f])));
        Assert.Throws<ArgumentException>(() => invalidAffine.Validate(5));

        var invalidTernary = new TensorStorageDescriptor(
            TensorDType.Ternary1Bit58,
            new TensorStorageMetadata(
                TensorStorageEncoding.Packed,
                new TensorPackingMetadata(BitsPerValue: 2)));
        Assert.Throws<ArgumentException>(() => invalidTernary.Validate(2));
    }

    [Fact]
    public void LegacyModuleParameterJsonDefaultsToRawStorage()
    {
        const string json =
            "{\"Index\":0,\"Name\":\"Weight\",\"Shape\":[1]," +
            "\"Values\":[1.25],\"DType\":0}";

        ModuleParameterState? state =
            JsonSerializer.Deserialize<ModuleParameterState>(json);

        Assert.NotNull(state);
        Assert.Null(state.StorageMetadata);
        Assert.True(new TensorStorageDescriptor(
            state.DType,
            state.StorageMetadata).IsSupportedByCurrentRuntime);
    }

    [Fact]
    public void SafeTensorsRejectsNonRawMetadataRatherThanDroppingIt()
    {
        ModuleState state = new(
            ModuleState.CurrentFormatVersion,
            [
                new ModuleParameterState(
                    0,
                    "Weight",
                    [2],
                    [1f, -2f],
                    TensorDType.Float32,
                    new TensorStorageMetadata(
                        TensorStorageEncoding.Packed,
                        new TensorPackingMetadata(BitsPerValue: 4))),
            ]);
        string path = Path.Combine(
            Path.GetTempPath(),
            $"nntrain-low-bit-{Guid.NewGuid():N}.safetensors");

        try
        {
            ArgumentException exception = Assert.Throws<ArgumentException>(
                () => safetensors.torch.save_file(state, path));
            Assert.Contains(
                "storage metadata",
                exception.Message,
                StringComparison.OrdinalIgnoreCase);
            Assert.False(File.Exists(path));
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }
}
