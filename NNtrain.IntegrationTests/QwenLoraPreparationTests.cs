using System.Text.Json;
using NNtrain;
using Xunit;

public sealed class QwenLoraPreparationTests
{
    [Fact]
    public void NormalizationRetainsMarkdownAndMathBackslashesAndRealNewlines()
    {
        const string raw = "{\"messages\":[{\"role\":\"user\",\"content\":\"question\"},{\"role\":\"assistant\",\"content\":\"\\*bold\\*\\n\\[\\beta\\neq\\times\\forall\\operatorname{x}\\]\"}]}";
        var converted = QwenLoraPreparation.ConvertMessages(raw);
        Assert.Equal("\\*bold\\*\n\\[\\beta\\neq\\times\\forall\\operatorname{x}\\]", converted.Examples[0].Response);
        Assert.Equal(2, converted.MessageCount);
        Assert.Single(converted.Repairs);
        Assert.Equal(9, converted.Repairs[0].LiteralBackslashesEscaped);
        using var normalized = JsonDocument.Parse(converted.NormalizedJsonl.TrimEnd());
        Assert.Equal(converted.Examples[0].Response, normalized.RootElement.GetProperty("messages")[1].GetProperty("content").GetString());
        Assert.DoesNotContain('\b', converted.Examples[0].Response);
        Assert.DoesNotContain('\t', converted.Examples[0].Response);
        Assert.DoesNotContain('\f', converted.Examples[0].Response);
    }

    [Fact]
    public void ValidJsonIsNotRepairedOrReinterpreted()
    {
        string valid = JsonSerializer.Serialize(new { messages = new[] {
            new { role = "user", content = " q " }, new { role = "assistant", content = "\n\\beta \\*text\\*\n" } } });
        Assert.Equal(valid, QwenLoraPreparation.NormalizeMalformedLine(valid, out int repairs));
        Assert.Equal(0, repairs);
        Assert.Equal("\n\\beta \\*text\\*\n", QwenLoraPreparation.ConvertMessages(valid).Examples[0].Response);
    }

    [Fact]
    public void AllHistoryRolesAndLongContentAreRetainedInOrderWithoutTrimming()
    {
        string answer = " leading " + new string('x', 20000) + " trailing\n";
        string json = JsonSerializer.Serialize(new { messages = new[] {
            new { role = "system", content = "rules" }, new { role = "user", content = "first" },
            new { role = "assistant", content = "old answer" }, new { role = "user", content = "last" },
            new { role = "assistant", content = answer } } });
        var converted = QwenLoraPreparation.ConvertMessages(json + "\n" + json);
        Assert.Equal(2, converted.Examples.Length);
        Assert.Equal(10, converted.MessageCount);
        Assert.Equal(answer, converted.Examples[0].Response);
        Assert.StartsWith("<|im_start|>system\nrules<|im_end|>\n<|im_start|>user\nfirst", converted.Examples[0].Prompt);
        Assert.Contains("<|im_start|>assistant\nold answer<|im_end|>\n<|im_start|>user\nlast", converted.Examples[0].Prompt);
        var saved = converted.TrainingJsonl.Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(line => JsonSerializer.Deserialize<QwenLoraCommand.Example>(line, LoraConfiguration.Json)!).ToArray();
        Assert.Equal(converted.Examples, saved);
    }

    [Fact]
    public void IsolatedFinalAssistantSpellingTypoIsRecordedWithoutContentLoss()
    {
        const string json = "{\"messages\":[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assisutant\",\"content\":\" exact content \"}]}";
        var converted = QwenLoraPreparation.ConvertMessages(json);
        Assert.Single(converted.RoleRepairs);
        Assert.Equal("assisutant", converted.RoleRepairs[0].Original);
        Assert.Equal(" exact content ", converted.Examples[0].Response);
        using var parsed = JsonDocument.Parse(converted.NormalizedJsonl.TrimEnd());
        Assert.Equal("assistant", parsed.RootElement.GetProperty("messages")[1].GetProperty("role").GetString());
    }

    [Theory]
    [InlineData("{\"messages\":[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\",\"content\":\"\\q\"}]}")]
    [InlineData("{\"messages\":[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"tool\",\"content\":\"a\"}]}")]
    [InlineData("{\"messages\":[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\",\"content\":\"a\",\"extra\":1}]}")]
    [InlineData("{\"messages\":[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\",\"content\":\"a\"}],\"extra\":1}")]
    [InlineData("{\"messages\":[{\"role\":\"assistant\",\"content\":\"q\"},{\"role\":\"user\",\"content\":\"a\"}]}")]
    public void UncertainInputOrExtraFieldsAreRejectedRatherThanDropped(string json)
        => Assert.Throws<InvalidDataException>(() => QwenLoraPreparation.ConvertMessages(json));

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DryRunFlagAndExplicitResumeFailBeforeGpuOnMissingInput(bool resume)
    {
        string directory = Path.Combine(Path.GetTempPath(), "qa142-cpu-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string model = Path.Combine(directory, "model.gguf");
            string config = Path.Combine(directory, "train.json");
            File.WriteAllText(Path.Combine(directory, "data.jsonl"), "{\"prompt\":\"a\",\"response\":\"b\"}\n");
            File.WriteAllText(config, JsonSerializer.Serialize(new QwenLoraTrainingConfiguration {
                DataPath = "data.jsonl", AdapterPath = "adapter.bin" }, LoraConfiguration.Json));
            if (resume) File.WriteAllBytes(model, Array.Empty<byte>());
            using var output = new StringWriter(); using var error = new StringWriter();
            int result = QwenLoraCommand.Run(new[] { "qwen-lora", "--model", model, "--config", config, "--dry-run" }
                .Concat(resume ? new[] { "--resume" } : Array.Empty<string>()).ToArray(), output, error);
            Assert.NotEqual(0, result);
            Assert.Contains(resume ? "--resume requires an existing adapter" : "GGUF model not found", error.ToString());
            Assert.DoesNotContain("Unknown qwen-lora option", error.ToString());
            Assert.False(File.Exists(Path.Combine(directory, "adapter.bin")));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }
}
