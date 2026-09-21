using NNtrain;
using Xunit;

public sealed class DpoDeviceConfigurationTests
{
    private static LoraTrainingConfiguration Config => new() {
        Objective = "dpo", Device = "cuda", PromptPrefix = "", ResponsePrefix = "" };
    [Fact]
    public void DefaultAndMultiGpuMappingKeepTrainingDeviceIndependent()
    {
        var single = Config with { DeviceIndex = 1 };
        single.Validate();
        Assert.Equal(new[] { 1, 1, 1, 1 }, Enumerable.Range(0, 4).Select(single.GenerationDeviceForWorker));
        var dual = Config with { GenerationDeviceIndices = [0, 1] };
        dual.Validate();
        Assert.Equal(0, dual.DeviceIndex);
        Assert.Equal(new[] { 0, 1, 0, 1 }, Enumerable.Range(0, 4).Select(dual.GenerationDeviceForWorker));
        var dedicated = Config with { GenerationDeviceIndices = [1] };
        dedicated.Validate();
        Assert.Equal(new[] { 1, 1, 1, 1 }, Enumerable.Range(0, 4).Select(dedicated.GenerationDeviceForWorker));
    }
    [Fact]
    public void InvalidDeviceListsAreRejectedBeforeLoadingModel()
    {
        foreach (int[] devices in new int[][] { [], [-1], [0, 0], [0, 1, 2, 3, 4] })
            Assert.Throws<ArgumentException>(() => (Config with { GenerationDeviceIndices = devices }).Validate());
        Assert.Throws<ArgumentException>(() => (Config with { Device = "cpu", GenerationDeviceIndices = [0] }).Validate());
    }
}
