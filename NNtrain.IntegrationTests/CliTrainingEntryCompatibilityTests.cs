using NNtrain;
using Xunit;

public sealed class CliTrainingEntryCompatibilityTests
{
    [Theory]
    [InlineData("--config", "chosen.json")]
    [InlineData("--CONFIG", "chosen.json")]
    [InlineData("--config", "chosen.json", "--resume")]
    [InlineData("--config", "chosen.json", "--AUTO-RESUME")]
    [InlineData("--config", "chosen.json", "--generate", "a prompt")]
    public void SupportedExplicitFormsLoadTheChosenConfigurationOnce(params string[] args)
    {
        AssertChosenConfiguration(args, "chosen.json");
    }

    [Theory]
    [InlineData()]
    [InlineData("--resume")]
    [InlineData("--AUTO-RESUME")]
    public void SupportedDefaultFormsUseTheExistingConfigurationSearch(params string[] args)
    {
        AssertChosenConfiguration(args, Program.FindDefaultConfiguration());
    }

    [Theory]
    [InlineData("--config")]
    [InlineData("--generate", "prompt")]
    [InlineData("--resume", "--config", "chosen.json")]
    [InlineData("--config", "chosen.json", "--resume", "--auto-resume")]
    [InlineData("--config", "chosen.json", "--generate")]
    [InlineData("--config", "chosen.json", "--generate", "prompt", "extra")]
    [InlineData("--generate-config", "generate.json", "--config", "chosen.json")]
    public void UnsupportedFormsPreserveUsageAndNeverLoadConfiguration(params string[] args)
    {
        using var output = new StringWriter();
        using var error = new StringWriter();
        int loadCount = 0;

        int exit = Program.Run(args, output, error, openLossGraph: false,
            loadConfiguration: _ =>
            {
                loadCount++;
                throw new InvalidOperationException("Unsupported arguments must not load configuration.");
            });

        string expected = string.Join(Environment.NewLine,
            "Usage: NNtrain.Cli [--config <training-config.json>] [--resume | --auto-resume | --generate <prompt>] [--generate-config <generate.json>]",
            "       NNtrain.Cli lora --model <DRN-checkpoint.json> [--config traning-lora.json] [--generate <prompt>]",
            "       NNtrain.Cli gemma-gguf --model <gemma3-Q8_0.gguf> --prompt <text> [--max-new-tokens 128] [--context 4096] [--threads N] [--backend cpu|opencl] [--gpu-layers N] [--opencl-all-devices] [--temperature 0] [--top-p 1] [--top-k 1] [--no-stream]",
            string.Empty);
        Assert.Equal(1, exit);
        Assert.Equal(0, loadCount);
        Assert.Equal(string.Empty, output.ToString());
        Assert.Equal(expected, error.ToString());
    }

    [Fact]
    public void UnexpectedConfigurationFailureRemainsVisibleToTheCaller()
    {
        using var output = new StringWriter();
        using var error = new StringWriter();
        var failure = new InvalidOperationException("unexpected failure");

        Exception observed = Assert.Throws<InvalidOperationException>(() =>
            Program.Run(["--config", "chosen.json"], output, error,
                openLossGraph: false,
                loadConfiguration: _ => throw failure));

        Assert.Same(failure, observed);
        Assert.Equal(string.Empty, error.ToString());
    }

    private static void AssertChosenConfiguration(string[] args, string expectedPath)
    {
        using var output = new StringWriter();
        using var error = new StringWriter();
        int loadCount = 0;

        int exit = Program.Run(args, output, error, openLossGraph: false,
            loadConfiguration: path =>
            {
                Assert.Equal(expectedPath, path);
                loadCount++;
                throw new ArgumentException("configuration sentinel");
            });

        Assert.Equal(2, exit);
        Assert.Equal(1, loadCount);
        Assert.Equal(string.Empty, output.ToString());
        Assert.Equal("Error: configuration sentinel" + Environment.NewLine, error.ToString());
    }
}
