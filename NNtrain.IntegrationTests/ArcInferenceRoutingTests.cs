using NNtrain;
using Xunit;

public sealed class ArcInferenceRoutingTests
{
    [Fact]
    public void GenerationOverridesDoNotChangeTrainingDeviceSelection()
    {
        var training = new WikiTrainingConfiguration
        {
            Device = "arc",
            DeviceIndices = [0],
            ModelArchitecture = "transformer",
            InferenceDeviceIndices = [0, 1],
            ArcInferenceMode = "auto",
        };
        training.Validate();
        ArcInferenceSettings fromTraining = ArcInferenceRouting.Resolve(training);
        Assert.Equal([0, 1], fromTraining.DeviceIndices);
        Assert.True(fromTraining.UsesTwoDevices);

        var generation = new GenerationConfiguration
        {
            TrainingConfigPath = "training.json",
            SafeTensorsPath = "model.safetensors",
            Prompt = "hello",
            InferenceDeviceIndices = [1, 0],
            ArcInferenceMode = "single",
        };
        generation.Validate();
        ArcInferenceSettings overridden = ArcInferenceRouting.Resolve(training, generation);
        Assert.Equal([1, 0], overridden.DeviceIndices);
        Assert.False(overridden.UsesTwoDevices);
        Assert.Equal([0], training.DeviceIndices);
    }

    [Fact]
    public void TensorParallelRequiresTwoDistinctArcIndices()
    {
        var training = new WikiTrainingConfiguration
        {
            Device = "arc",
            DeviceIndices = [0],
            ModelArchitecture = "transformer",
            ArcInferenceMode = "tensorParallel",
        };
        training.Validate();
        Assert.Throws<ArgumentException>(() => ArcInferenceRouting.Resolve(training));
        Assert.Throws<ArgumentException>(() => ArcInferenceRouting.ValidateOptions([0, 0], "auto"));
        Assert.Throws<ArgumentException>(() => ArcInferenceRouting.ValidateOptions([0, 1, 2], "auto"));
        Assert.Throws<ArgumentException>(() => ArcInferenceRouting.ValidateOptions([0, 1], "invalid"));
    }

    [Fact]
    public void AutoSelectsTwoDevicesOnlyAtFivePercentMedianSpeedup()
    {
        Assert.True(ArcInferenceRouting.ShouldUseTensorParallel(100d, 95d));
        Assert.False(ArcInferenceRouting.ShouldUseTensorParallel(100d, 95.01d));
        Assert.False(ArcInferenceRouting.ShouldUseTensorParallel(100d, 100d));
    }
}
