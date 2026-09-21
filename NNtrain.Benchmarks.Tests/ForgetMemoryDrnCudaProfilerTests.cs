using System.Text.Json;
using Xunit;

namespace NNtrain.Benchmarks;

public sealed class ForgetMemoryDrnCudaProfilerTests
{
    [Theory]
    [InlineData(1, 1, false)]
    [InlineData(0, 501, false)]
    [InlineData(0, 1, true)]
    public void RealProbeRejectsUnboundedOrIncompatibleOptionsBeforeIo(int warmup, int steps, bool detail)
        => Assert.Throws<ArgumentException>(() => ForgetMemoryDrnCudaProfiler.Run(
            "missing.json", warmup, steps, detail, "new-result.json", realData: true));

    [Fact]
    public void NeverOverwritesExistingOutputEvenWithMissingConfiguration()
    {
        string path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        try
        {
            File.WriteAllText(path, "preserve me");
            Assert.Throws<IOException>(() => ForgetMemoryDrnCudaProfiler.Run("missing.json", 0, 1, resultPath: path));
            Assert.Equal("preserve me", File.ReadAllText(path));
        }
        finally { File.Delete(path); }
    }

    [Theory]
    [InlineData(-1, 1, "warmupSteps")]
    [InlineData(0, 0, "measuredSteps")]
    [InlineData(0, -1, "measuredSteps")]
    public void RejectsInvalidStepCountsBeforeReadingConfigurationOrUsingCuda(
        int warmupSteps, int measuredSteps, string parameter)
    {
        string missingPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        var exception = Assert.Throws<ArgumentOutOfRangeException>(() =>
            ForgetMemoryDrnCudaProfiler.Run(missingPath, warmupSteps, measuredSteps));
        Assert.Equal(parameter, exception.ParamName);
        Assert.False(File.Exists(missingPath));
    }

    [Theory]
    [InlineData("transformer", "nekomuon")]
    [InlineData("forgetmemorydrn", "adamw")]
    public void RejectsWrongModelOrOptimizerBeforeUsingCuda(
        string architecture, string optimizer)
    {
        string path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json");
        string json = JsonSerializer.Serialize(new
        {
            modelArchitecture = architecture,
            optimization = new { optimizer = new { type = optimizer } },
        });
        try
        {
            File.WriteAllText(path, json);
            var exception = Assert.Throws<ArgumentException>(() =>
                ForgetMemoryDrnCudaProfiler.Run(path, 0, 1));
            Assert.Contains("DRN profiling requires forgetmemorydrn with Muon or NekoMuon", exception.Message);
            Assert.Equal(json, File.ReadAllText(path));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
