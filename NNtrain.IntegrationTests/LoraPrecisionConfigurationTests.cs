using System.Text.Json;
using NNtrain;
using Xunit;

public sealed class LoraPrecisionConfigurationTests
{
    [Fact]
    public void TrainingJsonOverridesPrecisionAndBlockSizeWithoutChangingAdapterDefinition()
    {
        var lora = JsonSerializer.Deserialize<LoraConfiguration>("""
            { "rank": 4, "alpha": 8, "precisionMode": "mix16_32", "bfp8BlockSize": 128 }
            """, LoraConfiguration.Json)!;
        var training = JsonSerializer.Deserialize<LoraTrainingConfiguration>("""
            { "precisionMode": "mix8_32", "bfp8BlockSize": 32, "allowPrecisionConversionOnResume": true }
            """, LoraConfiguration.Json)!;
        training.Validate();
        LoraConfiguration effective = lora.WithTrainingOverrides(training);
        Assert.Equal("mix8_32", effective.PrecisionMode);
        Assert.Equal(32, effective.Bfp8BlockSize);
        Assert.Equal(4, effective.Rank);
        Assert.Equal(8, effective.Alpha);
        Assert.Equal("mix16_32", lora.PrecisionMode);
        Assert.Equal(128, lora.Bfp8BlockSize);
        Assert.True(training.AllowPrecisionConversionOnResume);
    }

    [Fact]
    public void MissingTrainingOverridesInheritAdapterPrecisionAndBlockSize()
    {
        var lora = JsonSerializer.Deserialize<LoraConfiguration>("""
            { "precisionMode": "mix8_32", "bfp8BlockSize": 64 }
            """, LoraConfiguration.Json)!;
        var training = JsonSerializer.Deserialize<LoraTrainingConfiguration>("{}", LoraConfiguration.Json)!;
        LoraConfiguration effective = lora.WithTrainingOverrides(training);
        Assert.Equal("mix8_32", effective.PrecisionMode);
        Assert.Equal(64, effective.Bfp8BlockSize);
        Assert.False(training.AllowPrecisionConversionOnResume);
    }

    [Fact]
    public void PrecisionOnlyOverrideUsesAdapterBlockSize()
    {
        var lora = new LoraConfiguration { Bfp8BlockSize = 64 };
        var training = new LoraTrainingConfiguration { PrecisionMode = "mix8_32" };
        Assert.Equal(64, lora.WithTrainingOverrides(training).Bfp8BlockSize);
    }

    [Theory]
    [InlineData("float32")]
    [InlineData("mix16_32")]
    [InlineData("mix8_32")]
    public void SupportedPrecisionModesPassBothConfigurationContracts(string precision)
    {
        new LoraConfiguration { PrecisionMode = precision }.Validate();
        new LoraTrainingConfiguration { PrecisionMode = precision }.Validate();
    }

    [Theory]
    [InlineData("bfp8")]
    [InlineData("bfloat16")]
    [InlineData("unknown")]
    public void UnsupportedPrecisionModesFailBeforeModelLoading(string precision)
    {
        Assert.Throws<NotSupportedException>(() => new LoraConfiguration { PrecisionMode = precision }.Validate());
        Assert.Throws<ArgumentException>(() => new LoraTrainingConfiguration { PrecisionMode = precision }.Validate());
        Assert.Throws<NotSupportedException>(() => new LoraConfiguration().WithTrainingOverrides(
            new LoraTrainingConfiguration { PrecisionMode = precision }));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void NonpositiveBlockSizesFailBeforeModelLoading(int blockSize)
    {
        Assert.Throws<ArgumentException>(() => new LoraConfiguration { Bfp8BlockSize = blockSize }.Validate());
        Assert.Throws<ArgumentException>(() => new LoraTrainingConfiguration { Bfp8BlockSize = blockSize }.Validate());
        Assert.Throws<ArgumentException>(() => new LoraConfiguration().WithTrainingOverrides(
            new LoraTrainingConfiguration { Bfp8BlockSize = blockSize }));
    }
}
