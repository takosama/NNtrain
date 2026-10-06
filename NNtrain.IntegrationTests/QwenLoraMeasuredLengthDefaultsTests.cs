using System.Text.Json;
using NNtrain;
using Xunit;

public sealed class QwenLoraMeasuredLengthDefaultsTests
{
    [Theory]
    [InlineData(4096, true, false, 0, false)]
    [InlineData(8192, true, true, 512, true)]
    [InlineData(3840, false, false, 0, false)]
    [InlineData(4095, false, false, 0, false)]
    [InlineData(8193, false, false, 0, false)]
    public void OnlyMeasuredHomogeneousLengthsSelectDefaults(int count, bool fused, bool handoff, int tile, bool rolling)
    {
        var config = new QwenLoraTrainingConfiguration { ContextLength = count };
        var options = config.ExecutionOptions(new[] { count, count }, false);
        Assert.Equal(fused, options.TrainingFusedAttentionRows);
        Assert.Equal(handoff, options.TrainingHostCheckpointBufferHandoff);
        Assert.Equal(tile, options.TrainingStreamedAttentionTileRows);
        Assert.Equal(rolling, options.TrainingIQ2RollingTranspose);
        Assert.False(options.TrainingIQ2Fp16XmxForward);
        Assert.False(options.TrainingFusedAttentionOutput);
        Assert.False(options.TrainingHostCheckpointForwardCopyHandoff);
        Assert.Equal(8, options.TrainingTransposeOctetRows);
    }

    [Theory]
    [InlineData(4096)]
    [InlineData(8192)]
    public void MixedShortEmptyResumeAndDisabledProfileKeepOmittedModesOff(int context)
    {
        var config = new QwenLoraTrainingConfiguration { ContextLength = context };
        AssertOff(config.ExecutionOptions(new[] { context, 100 }, false));
        AssertOff(config.ExecutionOptions(Array.Empty<int>(), false));
        AssertOff(config.ExecutionOptions(new[] { context }, true));
        AssertOff((config with { UseMeasuredLengthDefaults = false }).ExecutionOptions(new[] { context }, false));
    }

    [Theory]
    [InlineData(4096)]
    [InlineData(8192)]
    public void ExplicitFalseAndZeroWinAfterSaveReload(int context)
    {
        var config = Load($"{{\"contextLength\":{context},\"fusedAttentionRows\":false,\"hostCheckpointBufferHandoff\":false,\"streamedAttentionTileRows\":0,\"iq2RollingTranspose\":false}}");
        config.Validate();
        AssertOff(config.ExecutionOptions(new[] { context }, false));
        var saved = Load(JsonSerializer.Serialize(config, LoraConfiguration.Json));
        AssertOff(saved.ExecutionOptions(new[] { context }, false));
        Assert.False(saved.FusedAttentionRows);
        Assert.False(saved.HostCheckpointBufferHandoff);
        Assert.Equal(0, saved.StreamedAttentionTileRows);
    }

    [Fact]
    public void OmittedModesRemainUnspecifiedOnSaveAndExplicitModesPersistOnResume()
    {
        var omitted = Load("{\"contextLength\":8192}");
        var saved = Load(JsonSerializer.Serialize(omitted, LoraConfiguration.Json));
        Assert.Null(saved.FusedAttentionRows);
        Assert.Null(saved.HostCheckpointBufferHandoff);
        Assert.Null(saved.StreamedAttentionTileRows);
        Assert.Null(saved.Iq2RollingTranspose);
        Assert.True(saved.ExecutionOptions(new[] { 8192 }, false).TrainingFusedAttentionRows);
        AssertOff(saved.ExecutionOptions(new[] { 8192 }, true));
        var explicitConfig = saved with { FusedAttentionRows = true, HostCheckpointBufferHandoff = true,
            StreamedAttentionTileRows = 512, Iq2RollingTranspose = true, Iq2ForwardPrecision = "fp16" };
        var result = Load(JsonSerializer.Serialize(explicitConfig, LoraConfiguration.Json)).ExecutionOptions(new[] { 8192 }, true);
        Assert.True(result.TrainingFusedAttentionRows);
        Assert.True(result.TrainingHostCheckpointBufferHandoff);
        Assert.True(result.TrainingIQ2RollingTranspose);
        Assert.True(result.TrainingIQ2Fp16XmxForward);
        Assert.Equal(512, result.TrainingStreamedAttentionTileRows);
    }

    [Fact]
    public void PackedOffDisablesAutomaticCombinationAndExplicitIncompatibilityIsRejected()
    {
        var config = new QwenLoraTrainingConfiguration { ContextLength = 8192, PackedAttentionScores = false };
        AssertOff(config.ExecutionOptions(new[] { 8192 }, false));
        Assert.Throws<ArgumentException>(() => (config with { FusedAttentionRows = true }).Validate());
    }

    private static void AssertOff(Qwen35ExecutionOptions options)
    {
        Assert.False(options.TrainingFusedAttentionRows);
        Assert.False(options.TrainingHostCheckpointBufferHandoff);
        Assert.False(options.TrainingIQ2RollingTranspose);
        Assert.Equal(0, options.TrainingStreamedAttentionTileRows);
    }
    private static QwenLoraTrainingConfiguration Load(string json)
    {
        string path = Path.GetTempFileName();
        try { File.WriteAllText(path, json); return QwenLoraTrainingConfiguration.Load(path); }
        finally { File.Delete(path); }
    }
}
