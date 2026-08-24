using Xunit;

namespace NNtrain;

public sealed class GenerationConfigurationTests
{
    [Fact]
    public void LoadsRelativePathsAndTemplatorAlias()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "trainingConfigPath": "training.json",
              "safeTensorsPath": "model.safetensors",
              "tokenizerPath": "tokenizer.json",
              "prompt": "hello",
              "sampling": "topK",
              "maxNewTokens": 12,
              "templator": 0.6,
              "topK": 7
            }
            """);

        GenerationConfiguration configuration =
            GenerationConfiguration.Load(path);

        Assert.Equal(Path.Combine(directory.Root, "training.json"), configuration.TrainingConfigPath);
        Assert.Equal(Path.Combine(directory.Root, "model.safetensors"), configuration.SafeTensorsPath);
        Assert.Equal(Path.Combine(directory.Root, "tokenizer.json"), configuration.TokenizerPath);
        Assert.Equal(0.6f, configuration.EffectiveTemperature);
        Assert.False(configuration.IsGreedy);
    }

    [Fact]
    public void GreedyModeIsCaseInsensitive()
    {
        var configuration = new GenerationConfiguration
        {
            TrainingConfigPath = "training.json",
            SafeTensorsPath = "model.safetensors",
            Prompt = "hello",
            Sampling = "GREEDY",
        };

        configuration.Validate();

        Assert.True(configuration.IsGreedy);
    }

    [Fact]
    public void RejectsTemperatureAndTemplatorTogether()
    {
        var configuration = new GenerationConfiguration
        {
            TrainingConfigPath = "training.json",
            SafeTensorsPath = "model.safetensors",
            Prompt = "hello",
            Temperature = 0.8f,
            Templator = 0.8f,
        };

        Assert.Throws<ArgumentException>(configuration.Validate);
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        internal TemporaryDirectory()
        {
            Root = Path.Combine(
                Path.GetTempPath(),
                $"NNtrain.GenerationConfigTests-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Root);
        }

        internal string Root { get; }

        internal string Write(string json)
        {
            string path = Path.Combine(Root, "generate.json");
            File.WriteAllText(path, json);
            return path;
        }

        public void Dispose() => Directory.Delete(Root, recursive: true);
    }
}
