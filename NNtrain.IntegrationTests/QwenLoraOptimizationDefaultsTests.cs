using System.Text.Json;
using NNtrain;
using Xunit;

public sealed class QwenLoraOptimizationDefaultsTests
{
    [Theory]
    [InlineData("{}", true, true)]
    [InlineData("{\"hostCheckpointGpuGradients\":false,\"packedAttentionScores\":false}", false, false)]
    [InlineData("{\"hostCheckpointGpuGradients\":false}", false, true)]
    [InlineData("{\"packedAttentionScores\":false}", true, false)]
    [InlineData("{\"hostCheckpointGpuGradients\":true,\"packedAttentionScores\":true}", true, true)]
    public void SavedCliConfigurationHonorsExplicitValuesAndDefaults(string json, bool gradients, bool packed)
    {
        string path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".qwen-lora.json");
        try
        {
            File.WriteAllText(path, json);
            var config = QwenLoraTrainingConfiguration.Load(path);
            config.Validate();
            Assert.Equal(gradients, config.HostCheckpointGpuGradients);
            Assert.Equal(packed, config.PackedAttentionScores);
            Assert.Equal("exact", config.Iq2ForwardPrecision);
            var execution = config.ExecutionOptions(new[] { 128 }, resume: false);
            Assert.Equal(gradients, execution.TrainingHostCheckpointGpuGradients);
            Assert.Equal(packed, execution.TrainingPackedAttentionScores);
            Assert.False(execution.TrainingIQ2Fp16XmxForward);
            Assert.Equal(8, execution.TrainingTransposeOctetRows);
            File.WriteAllText(path, JsonSerializer.Serialize(config, LoraConfiguration.Json));
            var saved = QwenLoraTrainingConfiguration.Load(path);
            Assert.Equal(gradients, saved.HostCheckpointGpuGradients);
            Assert.Equal(packed, saved.PackedAttentionScores);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void CoreDefaultsAndExplicitOffKeepPrecisionAndTransposeUnchanged()
    {
        var defaults = new Qwen35ExecutionOptions();
        Assert.True(defaults.TrainingHostCheckpointGpuGradients);
        Assert.True(defaults.TrainingPackedAttentionScores);
        Assert.False(defaults.TrainingIQ2Fp16XmxForward);
        Assert.Equal(8, defaults.TrainingTransposeOctetRows);
        var disabled = JsonSerializer.Deserialize<Qwen35ExecutionOptions>(
            "{\"TrainingHostCheckpointGpuGradients\":false,\"TrainingPackedAttentionScores\":false}")!;
        Assert.False(disabled.TrainingHostCheckpointGpuGradients);
        Assert.False(disabled.TrainingPackedAttentionScores);
        Assert.False(disabled.TrainingIQ2Fp16XmxForward);
        Assert.Equal(8, disabled.TrainingTransposeOctetRows);
    }

    [Fact]
    public void RowFusedAttentionCanUseBoundedStreamedTiles()
    {
        var config = JsonSerializer.Deserialize<QwenLoraTrainingConfiguration>(
            "{\"fusedAttentionRows\":true,\"streamedAttentionTileRows\":512}", LoraConfiguration.Json)!;
        config.Validate();
        Assert.True(config.FusedAttentionRows);
        Assert.Equal(512, config.StreamedAttentionTileRows);
        Assert.True(config.PackedAttentionScores);
    }
}
