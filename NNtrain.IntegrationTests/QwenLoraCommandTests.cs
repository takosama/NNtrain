using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class QwenLoraCommandTests
{
    [Fact]
    public void TrainResumeAndGeneratePreserveBaseAndDeterministicDatasetCursor()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var fixture = new Fixture();
        string configPath = Path.Combine(fixture.DirectoryPath, "train.json");
        string adapterPath = Path.Combine(fixture.DirectoryPath, "adapter.bin");
        string dataPath = Path.Combine(fixture.DirectoryPath, "train.jsonl");
        const string dataset = "{\"prompt\":\"a\",\"response\":\"b\"}\n{\"prompt\":\"b\",\"response\":\"ca\"}\n";
        File.WriteAllText(dataPath, dataset);
        byte[] baseHash = SHA256.HashData(File.ReadAllBytes(fixture.ModelPath));
        var config = new QwenLoraTrainingConfiguration
        {
            DataPath = "train.jsonl", AdapterPath = "adapter.bin", Devices = [0],
            ContextLength = 8, Rank = 2, Alpha = 4, MaxSteps = 1, SaveEverySteps = 1,
            PromptPrefix = "", ResponsePrefix = "", IncludeOutput = true, OpenLossGraph = false
        };
        (int Exit, string Output, string Error) Run(bool resume = false)
        {
            File.WriteAllText(configPath, JsonSerializer.Serialize(config, LoraConfiguration.Json));
            using var output = new StringWriter(); using var error = new StringWriter();
            int exit = Program.Run(["qwen-lora", "--model", fixture.ModelPath, "--config", configPath,
                .. (resume ? new[] { "--resume" } : Array.Empty<string>())], output, error);
            return (exit, output.ToString(), error.ToString());
        }

        var first = Run();
        Assert.True(first.Exit == 0, first.Error);
        Assert.Contains("step=1, example=1/2", first.Output);
        Assert.Contains("supervised-tokens=2", first.Output);
        Assert.Contains("Arc 0 GPU bytes: encoded=", first.Output);
        Assert.Contains("encoded weight byte count unchanged", first.Output);
        Assert.True(File.Exists(Path.ChangeExtension(configPath, ".html")));
        Assert.True(first.Output.IndexOf("loss graph =", StringComparison.Ordinal)
            < first.Output.IndexOf("LoRA step=1", StringComparison.Ordinal));
        Assert.True(File.Exists(adapterPath));
        byte[] afterOne = File.ReadAllBytes(adapterPath);
        config = config with { MaxSteps = 2 };
        var resumed = Run(resume: true);
        Assert.True(resumed.Exit == 0, resumed.Error);
        Assert.Contains("step=2, example=2/2", resumed.Output);
        Assert.Contains("supervised-tokens=3", resumed.Output);
        Assert.NotEqual(Convert.ToHexString(SHA256.HashData(afterOne)),
            Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(adapterPath))));

        byte[] protectedAdapter = File.ReadAllBytes(adapterPath);
        config = config with { MaxSteps = 3 };
        Assert.Equal(2, Run().Exit); // Fresh runs never replace an existing adapter.
        File.WriteAllText(dataPath, dataset + "{\"prompt\":\"c\",\"response\":\"a\"}\n");
        var mismatchedData = Run(resume: true);
        Assert.Equal(2, mismatchedData.Exit);
        Assert.Equal(protectedAdapter, File.ReadAllBytes(adapterPath));
        File.WriteAllText(dataPath, dataset);
        config = config with { LearningRate = config.LearningRate * 2 };
        Assert.Equal(2, Run(resume: true).Exit);
        Assert.Equal(protectedAdapter, File.ReadAllBytes(adapterPath));
        config = config with { LearningRate = config.LearningRate / 2, MaxSteps = 2, AdapterPath = "continuous.bin" };
        var continuous = Run();
        Assert.True(continuous.Exit == 0, continuous.Error);

        float[] Logits(string path)
        {
            using var model = Qwen35QuantizedModel.Load(fixture.ModelPath, [0]);
            model.LoadLora(path);
            Assert.Equal(2L, (long)model.LoraStep);
            Assert.True(model.LoraParameterCount > 0);
            return model.ForwardToken(0);
        }
        Assert.Equal(Logits(Path.Combine(fixture.DirectoryPath, "continuous.bin")), Logits(adapterPath));
        using var generatedOutput = new StringWriter(); using var generatedError = new StringWriter();
        int generated = Program.Run(["qwen-gguf", "--model", fixture.ModelPath, "--adapter", adapterPath,
            "--device", "0", "--prompt", "a", "--max-new-tokens", "2", "--no-stream"], generatedOutput, generatedError);
        Assert.True(generated == 0, generatedError.ToString());
        Assert.Contains("LoRA adapter =", generatedOutput.ToString());
        Assert.Contains("generated:", generatedOutput.ToString());
        // Zero new tokens still verifies the supplied adapter rather than
        // silently taking the tokenizer-only shortcut.
        string corrupt = Path.Combine(fixture.DirectoryPath, "corrupt.bin");
        File.WriteAllBytes(corrupt, [1, 2, 3]);
        using var zeroOutput = new StringWriter(); using var zeroError = new StringWriter();
        Assert.Equal(2, Program.Run(["qwen-gguf", "--model", fixture.ModelPath, "--adapter", corrupt,
            "--device", "0", "--prompt", "a", "--max-new-tokens", "0"], zeroOutput, zeroError));
        Assert.Contains("checkpoint", zeroError.ToString());
        Assert.Equal(baseHash, SHA256.HashData(File.ReadAllBytes(fixture.ModelPath)));
        Assert.Empty(Directory.GetFiles(fixture.DirectoryPath, "*.tmp"));
    }

    [Fact]
    public void ResponseMaskIncludesFirstResponseTokenAndEosWithoutTruncation()
    {
        using var fixture = new Fixture();
        var tokenizer = Qwen2GgufTokenizer.Load(fixture.ModelPath);
        var config = new QwenLoraTrainingConfiguration { PromptPrefix = "", ResponsePrefix = "", ContextLength = 4 };
        QwenLoraCommand.TrainingExample example = QwenLoraCommand.BuildExample(new("a", "bc"), tokenizer, config);
        Assert.Equal([0, 1, 2, 3], example.Tokens);
        Assert.Equal(1, example.ResponseStartIndex);
        Assert.Contains("including EOS", Assert.Throws<InvalidDataException>(() =>
            QwenLoraCommand.BuildExample(new("a", "bc"), tokenizer, config with { ContextLength = 3 })).Message);
        Assert.Throws<InvalidDataException>(() => QwenLoraCommand.BuildExample(new("", "b"), tokenizer, config));
        Assert.Throws<InvalidDataException>(() => QwenLoraCommand.BuildExample(new("a", ""), tokenizer, config));
        Assert.Contains("line 3", Assert.Throws<InvalidDataException>(() => QwenLoraCommand.ReadExamples(
            Encoding.UTF8.GetBytes("{\"prompt\":\"a\",\"response\":\"b\"}\n\n{bad}\n"), tokenizer, config)).Message);
    }

    [Fact]
    public void ResumeIdentityProtectsDataAndNumericalContractButAllowsExtendingSteps()
    {
        byte[] dataset = Encoding.UTF8.GetBytes("{\"prompt\":\"a\",\"response\":\"b\"}\n");
        var config = new QwenLoraTrainingConfiguration();
        string identity = QwenLoraCommand.TrainingIdentity(dataset, 3, config);
        Assert.Equal(identity, QwenLoraCommand.TrainingIdentity(dataset, 3, config with
        { MaxSteps = 100, SaveEverySteps = 5, AdapterPath = "different.bin", DataPath = "copy.jsonl", Devices = [1],
            LossGraphPath = "different.html", ShowLossGraph = false, OpenLossGraph = false, LossGraphEverySteps = 100 }));
        Assert.NotEqual(identity, QwenLoraCommand.TrainingIdentity([.. dataset, 10], 3, config));
        Assert.NotEqual(identity, QwenLoraCommand.TrainingIdentity(dataset, 3, config with { Seed = 9 }));
        Assert.NotEqual(identity, QwenLoraCommand.TrainingIdentity(dataset, 3, config with { IncludeOutput = true }));
        Assert.NotEqual(identity, QwenLoraCommand.TrainingIdentity(dataset, 3, config with { ResponsePrefix = "" }));
        Assert.NotEqual(identity, QwenLoraCommand.TrainingIdentity(dataset, 2, config));
    }

    [Fact]
    public void InvalidPathsAndConfigurationFailBeforeModelAllocation()
    {
        using var fixture = new Fixture();
        string configPath = Path.Combine(fixture.DirectoryPath, "train.json");
        byte[] original = File.ReadAllBytes(fixture.ModelPath);
        var config = new QwenLoraTrainingConfiguration { AdapterPath = fixture.ModelPath };
        File.WriteAllText(configPath, JsonSerializer.Serialize(config, LoraConfiguration.Json));
        using var output = new StringWriter(); using var error = new StringWriter();
        Assert.Equal(2, Program.Run(["qwen-lora", "--model", fixture.ModelPath, "--config", configPath], output, error));
        Assert.Contains("distinct", error.ToString());
        Assert.Equal(original, File.ReadAllBytes(fixture.ModelPath));
        Assert.Throws<ArgumentException>(() => (config with { MaxSteps = 0 }).Validate());
        Assert.Throws<ArgumentException>(() => (config with { Devices = [0, 0] }).Validate());
        Assert.Throws<ArgumentException>(() => (config with { Targets = ["not_a_projection"] }).Validate());
        Assert.Throws<ArgumentException>(() => (config with { LearningRate = float.NaN }).Validate());
        File.WriteAllText(configPath, "{\"unknownOption\":1}");
        Assert.Throws<JsonException>(() => QwenLoraTrainingConfiguration.Load(configPath));
    }

    [Fact]
    public void AdapterIsRejectedForQwen2AndCannotAliasGguf()
    {
        using var fixture = new Fixture();
        using var output = new StringWriter(); using var error = new StringWriter();
        Assert.Equal(2, Program.Run(["qwen-gguf", "--model", fixture.ModelPath, "--adapter", fixture.ModelPath,
            "--prompt", "a", "--max-new-tokens", "0"], output, error));
        Assert.Contains("distinct", error.ToString());
        fixture.WriteModel("qwen2");
        error.GetStringBuilder().Clear();
        Assert.Equal(2, Program.Run(["qwen-gguf", "--model", fixture.ModelPath, "--adapter", "adapter.bin",
            "--prompt", "a", "--max-new-tokens", "0"], output, error));
        Assert.Contains("qwen35", error.ToString());
    }

    private sealed class Fixture : IDisposable
    {
        internal string DirectoryPath { get; } = Path.Combine(Path.GetTempPath(), "NNtrain.QwenLora." + Guid.NewGuid().ToString("N"));
        internal string ModelPath => Path.Combine(DirectoryPath, "base.gguf");
        internal Fixture() { Directory.CreateDirectory(DirectoryPath); WriteModel(); }

        internal void WriteModel(string architecture = "qwen35")
        {
            var metadata = new Dictionary<string, object>
            {
                ["general.architecture"] = architecture, ["tokenizer.ggml.model"] = "gpt2",
                ["tokenizer.ggml.pre"] = "qwen35", ["tokenizer.ggml.tokens"] = new[] { "a", "b", "c", "<|end|>" },
                ["tokenizer.ggml.merges"] = Array.Empty<string>(), ["tokenizer.ggml.token_type"] = new[] { 1, 1, 1, 3 },
                ["tokenizer.ggml.eos_token_id"] = 3u,
                ["qwen35.block_count"] = 2u, ["qwen35.embedding_length"] = 256u,
                ["qwen35.attention.head_count"] = 2u, ["qwen35.attention.head_count_kv"] = 1u,
                ["qwen35.attention.key_length"] = 128u, ["qwen35.attention.value_length"] = 128u,
                ["qwen35.context_length"] = 16u, ["qwen35.feed_forward_length"] = 512u,
                ["qwen35.rope.dimension_count"] = 64u, ["qwen35.ssm.group_count"] = 1u,
                ["qwen35.ssm.time_step_rank"] = 2u, ["qwen35.ssm.state_size"] = 128u,
                ["qwen35.ssm.inner_size"] = 256u, ["qwen35.ssm.conv_kernel"] = 4u,
                ["qwen35.full_attention_interval"] = 2u
            };
            var directory = new List<GgufTensorInfo>();
            void Matrix(string name, ulong input, ulong output) => directory.Add(new(name, [input, output], Qwen2Gguf.Q4KType, 0));
            void Dense(string name, params ulong[] shape) => directory.Add(new(name, shape, Qwen2Gguf.F32Type, 0));
            Matrix("token_embd.weight", 256, 4); Matrix("output.weight", 256, 4); Dense("output_norm.weight", 256);
            for (int layer = 0; layer < 2; ++layer)
            {
                string p = $"blk.{layer}.";
                Dense(p + "attn_norm.weight", 256); Dense(p + "post_attention_norm.weight", 256);
                Matrix(p + "ffn_gate.weight", 256, 512); Matrix(p + "ffn_up.weight", 256, 512);
                Matrix(p + "ffn_down.weight", 512, 256);
                if (layer == 0)
                {
                    Matrix(p + "attn_qkv.weight", 256, 512); Matrix(p + "attn_gate.weight", 256, 256);
                    Matrix(p + "ssm_alpha.weight", 256, 2); Matrix(p + "ssm_beta.weight", 256, 2);
                    Dense(p + "ssm_conv1d.weight", 4, 512); Dense(p + "ssm_a", 2); Dense(p + "ssm_dt.bias", 2);
                    Dense(p + "ssm_norm.weight", 128); Matrix(p + "ssm_out.weight", 256, 256);
                }
                else
                {
                    Matrix(p + "attn_q.weight", 256, 512); Matrix(p + "attn_k.weight", 256, 128);
                    Matrix(p + "attn_v.weight", 256, 128); Dense(p + "attn_q_norm.weight", 128);
                    Dense(p + "attn_k_norm.weight", 128); Matrix(p + "attn_output.weight", 256, 256);
                }
            }
            using var payload = new MemoryStream();
            var tensors = new List<GgufTensorInfo>();
            foreach (GgufTensorInfo tensor in directory)
            {
                while (payload.Position % 32 != 0) payload.WriteByte(0);
                tensors.Add(tensor with { Offset = (ulong)payload.Position });
                int elements = checked((int)tensor.Shape.Aggregate(1UL, (n, x) => n * x));
                if (tensor.Type == Qwen2Gguf.F32Type)
                {
                    using var dense = new BinaryWriter(payload, Encoding.UTF8, leaveOpen: true);
                    for (int i = 0; i < elements; ++i)
                        dense.Write(tensor.Name.EndsWith("ssm_a", StringComparison.Ordinal) ? -0.4f
                            : tensor.Name.EndsWith("ssm_dt.bias", StringComparison.Ordinal) ? 0f
                            : tensor.Name.EndsWith("ssm_conv1d.weight", StringComparison.Ordinal) ? (i % 4 == 3 ? 0.75f : 0.08f) : 1f);
                }
                else
                {
                    var random = new Random(tensor.Name.Aggregate(17, (seed, ch) => unchecked(seed * 31 + ch)));
                    byte[] bytes = new byte[elements / 256 * 144];
                    for (int offset = 0; offset < bytes.Length; offset += 144)
                    {
                        BitConverter.TryWriteBytes(bytes.AsSpan(offset), BitConverter.HalfToUInt16Bits((Half)(1f / 1024)));
                        BitConverter.TryWriteBytes(bytes.AsSpan(offset + 2), BitConverter.HalfToUInt16Bits((Half)(8f / 1024)));
                        bytes.AsSpan(offset + 4, 8).Fill(1); bytes.AsSpan(offset + 12, 4).Fill(0x11);
                        random.NextBytes(bytes.AsSpan(offset + 16, 128));
                    }
                    payload.Write(bytes);
                }
            }
            using var writer = new BinaryWriter(File.Create(ModelPath), Encoding.UTF8);
            writer.Write(0x46554747u); writer.Write(3u); writer.Write((ulong)tensors.Count); writer.Write((ulong)metadata.Count);
            foreach (var (name, value) in metadata)
            {
                WriteString(writer, name);
                switch (value)
                {
                    case string scalarText: writer.Write((uint)GgufValueType.String); WriteString(writer, scalarText); break;
                    case uint scalarNumber: writer.Write((uint)GgufValueType.UInt32); writer.Write(scalarNumber); break;
                    case string[] strings:
                        writer.Write((uint)GgufValueType.Array); writer.Write((uint)GgufValueType.String); writer.Write((ulong)strings.Length);
                        foreach (string text in strings) WriteString(writer, text);
                        break;
                    case int[] integers:
                        writer.Write((uint)GgufValueType.Array); writer.Write((uint)GgufValueType.Int32); writer.Write((ulong)integers.Length);
                        foreach (int number in integers) writer.Write(number);
                        break;
                    default: throw new InvalidOperationException();
                }
            }
            foreach (GgufTensorInfo tensor in tensors)
            {
                WriteString(writer, tensor.Name); writer.Write((uint)tensor.Shape.Count);
                foreach (ulong dimension in tensor.Shape) writer.Write(dimension);
                writer.Write(tensor.Type); writer.Write(tensor.Offset);
            }
            while (writer.BaseStream.Position % 32 != 0) writer.Write((byte)0);
            writer.Write(payload.ToArray());
        }

        private static void WriteString(BinaryWriter writer, string text)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(text); writer.Write((ulong)bytes.Length); writer.Write(bytes);
        }

        public void Dispose() => Directory.Delete(DirectoryPath, recursive: true);
    }
}
