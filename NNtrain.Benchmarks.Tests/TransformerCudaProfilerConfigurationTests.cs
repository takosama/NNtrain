using System.Text.Json;
using Xunit;

namespace NNtrain.Benchmarks;

public sealed class TransformerCudaProfilerConfigurationTests
{
    [Fact]
    public void ReadsAccumulationBlockSizeAndFixedNewtonSchulzDepth()
    {
        using JsonDocument document = JsonDocument.Parse(
            """
            {
              "gradientAccumulationSteps": 4,
              "bfp8_block_size": 32,
              "optimization": {
                "optimizer": {
                  "nekoMuonNewtonSchulzDepthMode": "fixed",
                  "nekoMuonNewtonSchulzDepth": 5
                }
              }
            }
            """);
        JsonElement root = document.RootElement;
        JsonElement optimizer = root.GetProperty("optimization")
            .GetProperty("optimizer");

        TransformerProfileTrainingControls controls =
            TransformerCudaProfiler.ReadTrainingControls(root, optimizer);

        Assert.Equal(4, controls.GradientAccumulationSteps);
        Assert.Equal(32, controls.Bfp8BlockSize);
        Assert.Equal(
            NekoMuonNewtonSchulzDepthMode.Fixed,
            controls.NewtonSchulzDepthMode);
        Assert.Equal(5f, controls.NewtonSchulzDepth);
    }

    [Fact]
    public void MissingTrainingControlsUseBackwardCompatibleDefaults()
    {
        using JsonDocument document = JsonDocument.Parse(
            """
            {
              "optimization": {
                "optimizer": {}
              }
            }
            """);
        JsonElement root = document.RootElement;
        JsonElement optimizer = root.GetProperty("optimization")
            .GetProperty("optimizer");

        TransformerProfileTrainingControls controls =
            TransformerCudaProfiler.ReadTrainingControls(root, optimizer);

        Assert.Equal(1, controls.GradientAccumulationSteps);
        Assert.Equal(
            Bfp8QuantizationDescriptor.DefaultBlockSize,
            controls.Bfp8BlockSize);
        Assert.Equal(
            NekoMuonNewtonSchulzDepthMode.Adaptive,
            controls.NewtonSchulzDepthMode);
        Assert.Equal(0f, controls.NewtonSchulzDepth);
    }
}
