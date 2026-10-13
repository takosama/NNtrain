using NNtrain;
using Xunit;

public sealed class GemmaGgufCommandTests
{
    [Theory]
    [InlineData("gemma-gguf", "--help")]
    [InlineData("GEMMA-GGUF", "-h")]
    public void HelpDispatchesWithoutLoadingTrainingConfiguration(string command, string help)
    {
        using var output = new StringWriter();
        using var error = new StringWriter();

        int exit = Program.Run([command, help], output, error,
            openLossGraph: false,
            loadConfiguration: _ => throw new InvalidOperationException("Training configuration must not be loaded."));

        Assert.Equal(0, exit);
        Assert.StartsWith("Usage: NNtrain.Cli gemma-gguf", output.ToString());
        Assert.Equal(string.Empty, error.ToString());
    }

    [Theory]
    [InlineData("--backend cpu requires --gpu-layers 0.", "--backend", "cpu", "--gpu-layers", "1")]
    [InlineData("--backend opencl requires --gpu-layers N where N is positive.", "--backend", "opencl")]
    [InlineData("--opencl-all-devices requires the OpenCL backend.", "--opencl-all-devices")]
    [InlineData("--tensor-split requires --opencl-all-devices and at least two values.", "--tensor-split", "53,47")]
    [InlineData("--tensor-split requires comma-separated positive finite values.", "--tensor-split", "53,NaN")]
    [InlineData("--top-p must be at most 1.", "--top-p", "1.1")]
    [InlineData("--context must be a positive integer.", "--context", "0")]
    [InlineData("--temperature must be a finite nonnegative number.", "--temperature", "NaN")]
    [InlineData("Missing value for --threads.", "--threads")]
    public void InvalidOptionsFailBeforeLoadingAModel(string message, params string[] options)
    {
        using var output = new StringWriter();
        using var error = new StringWriter();
        string missing = MissingModelPath();

        int exit = Program.Run(
            ["gemma-gguf", "--model", missing, "--prompt", "hello", .. options],
            output, error, openLossGraph: false);

        Assert.Equal(1, exit);
        Assert.Equal("Error: " + message + Environment.NewLine, error.ToString());
        Assert.Equal(string.Empty, output.ToString());
    }

    [Theory]
    [InlineData("--max-new-tokens", "０")]
    [InlineData("--context", "４０９６")]
    [InlineData("--temperature", "０.５")]
    [InlineData("--top-p", "１")]
    public void CompatibilityNumbersReachModelLoading(string option, string value)
    {
        using var output = new StringWriter();
        using var error = new StringWriter();

        int exit = Program.Run(
            ["gemma-gguf", "--model", MissingModelPath(), "--prompt", "hello", option, value, "--no-stream"],
            output, error, openLossGraph: false);

        Assert.Equal(2, exit);
        Assert.Equal("Error: Gemma 3 GGUF not found." + Environment.NewLine, error.ToString());
        Assert.Equal(string.Empty, output.ToString());
    }

    private static string MissingModelPath()
        => Path.Combine(Path.GetTempPath(), $"missing-gemma-{Guid.NewGuid():N}.gguf");
}
