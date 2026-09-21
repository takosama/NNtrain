using NNtrain;
using Xunit;

public sealed class LoraResumePolicyTests
{
    [Fact]
    public void MissingAutoCheckpointStartsNewButExplicitResumeRequiresIt()
    {
        string path = Path.Combine(Path.GetTempPath(), $"missing-lora-{Guid.NewGuid():N}.json");
        using var output = new StringWriter();
        var config = new LoraTrainingConfiguration { AutoResume = true };
        Assert.False(LoraResumePolicy.Resolve(config, path, false, output));
        Assert.Contains("starting a new adapter", output.ToString());
        Assert.Throws<FileNotFoundException>(() => LoraResumePolicy.Resolve(config with { Resume = true }, path, false, output));
        Assert.Throws<FileNotFoundException>(() => LoraResumePolicy.Resolve(config, path, true, output));
    }
    [Fact]
    public void ExistingEvenCorruptCheckpointIsSelectedNotOverwritten()
    {
        string path = Path.Combine(Path.GetTempPath(), $"lora-resume-{Guid.NewGuid():N}.json");
        try
        {
            File.WriteAllText(path, "invalid checkpoint");
            using var output = new StringWriter();
            Assert.True(LoraResumePolicy.Resolve(new() { AutoResume = true }, path, false, output));
            Assert.Contains("restoring adapter checkpoint", output.ToString());
            Assert.Throws<IOException>(() => LoraResumePolicy.Resolve(new(), path, false, output));
            Assert.Equal("invalid checkpoint", File.ReadAllText(path));
        }
        finally { File.Delete(path); }
    }
}
