using NNtrain;
using System.Text.Json;
using Xunit;
using Parquet;
using Parquet.Schema;

public sealed class LoraCommandTests
{
    [Theory]
    [InlineData("cpu", "float32", false)]
    [InlineData("cpu", "mix8_32", false)]
    [InlineData("cuda", "mix16_32", false)]
    [InlineData("cuda", "mix8_32", false)]
    [InlineData("cpu", "float32", true)]
    [InlineData("cpu", "mix8_32", true)]
    [InlineData("cuda", "mix16_32", true)]
    [InlineData("cuda", "mix8_32", true)]
    [InlineData("cuda", "mix16_32", true, true)]
    [InlineData("cuda", "mix8_32", true, true)]
    public async Task DrnLoraTrainsResumesAndGeneratesWithoutChangingBase(string device, string precision, bool dpo, bool twoGpu = false)
    {
        Assert.SkipWhen(device == "cuda" && !Tensor.IsCudaAvailable(), "CUDA unavailable.");
        Assert.SkipWhen(twoGpu && !Tensor.IsCudaAvailable(1), "Second CUDA GPU unavailable.");
        string directory = Path.Combine(Path.GetTempPath(), "NNtrain.Lora." + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string modelPath = Path.Combine(directory, "base.json"), configPath = Path.Combine(directory, "train.json");
            var tokenizer = BpeTokenizer.Train(["hello world good answer", "question response"], vocabularySize: 300);
            tokenizer.save(Path.Combine(directory, "tokenizer.json"));
            var model = new ForgetMemoryDRNGpt(tokenizer.VocabularySize, 64, 8, 16, 1, 2, 2, dtype: TensorDType.Float32);
            var checkpoint = new WikiLanguageModelCommand.WikiModelCheckpoint(6, 0, 1,
                tokenizer.VocabularySize, 64, 8, 1, 16, 1, 0, .02f, model.state_dict(),
                ModelArchitecture: "forgetmemorydrn", ForgetMemoryKeyWidth: 2, ForgetMemoryValueWidth: 2,
                ModelDType: TensorDType.Float32, PrecisionMode: TensorPrecisionMode.Float32);
            torch.save(checkpoint, modelPath);
            byte[] original = File.ReadAllBytes(modelPath);
            File.WriteAllText(Path.Combine(directory, "lora.json"), JsonSerializer.Serialize(new LoraConfiguration {
                Rank = 2, Alpha = 4, PrecisionMode = precision }, LoraConfiguration.Json));
            File.WriteAllLines(Path.Combine(directory, "data.jsonl"), Enumerable.Repeat(dpo
                ? "{\"text\":\"hello world good answer question response\"}"
                : "{\"prompt\":\"hello\",\"response\":\"world\"}", dpo ? 9 : 8));
            if (dpo)
            {
                Directory.CreateDirectory(Path.Combine(directory, "fineweb"));
                var field = new DataField<string>("text");
                using var file = File.Create(Path.Combine(directory, "fineweb", "train-00000.parquet"));
                await using var writer = await ParquetWriter.CreateAsync(new ParquetSchema(field), file, cancellationToken: TestContext.Current.CancellationToken);
                using var group = writer.CreateRowGroup();
                await group.WriteAsync(field, Enumerable.Repeat("hello world good answer question response", 9).ToArray());
            }
            var config = new LoraTrainingConfiguration {
                Objective = dpo ? "dpo" : "sft",
                LoraConfig = "lora.json", TokenizerPath = "tokenizer.json", DataPath = dpo ? "fineweb" : "data.jsonl",
                Dataset = dpo ? "fineweb" : "jsonl", GenerationSlots = 4,
                GenerationDeviceIndices = twoGpu ? [0, 1] : null,
                AdapterPath = "adapter.json", Device = device, ContextLength = 32, BatchSize = dpo ? 2 : 1,
                GradientAccumulationSteps = 2, Epochs = 1, MaxSteps = 1, SaveEverySteps = 1,
                AutoResume = true,
                SampleEverySteps = 2,
                PromptPrefix = "", ResponsePrefix = dpo ? "" : " ", MaxNewTokens = 2 };
            int Run(params string[] suffix)
            {
                File.WriteAllText(configPath, JsonSerializer.Serialize(config, LoraConfiguration.Json));
                bool existingAdapter = File.Exists(Path.Combine(directory, config.AdapterPath));
                using var output = new StringWriter(); using var error = new StringWriter();
                int result = Program.Run(["lora", "--model", modelPath, "--config", configPath, ..suffix], output, error);
                Assert.True(result == 0, error.ToString());
                if (dpo && suffix.Length == 0 && config.MaxSteps % 2 == 0)
                {
                    Assert.Contains($"DPO sample at global step {config.MaxSteps} (updated policy)", output.ToString());
                    Assert.Contains("[prefix]", output.ToString());
                    Assert.Contains("[chosen]", output.ToString());
                    Assert.Contains("[model continuation]", output.ToString());
                }
                Assert.DoesNotContain("DPO sample at global step 1 ", output.ToString());
                if (suffix.Length != 0 || !dpo) Assert.DoesNotContain("DPO sample at global step", output.ToString());
                if (suffix.Length == 0 && config.AutoResume && !config.Resume)
                    Assert.Contains(existingAdapter ? "auto-resume = restoring adapter checkpoint" : "auto-resume = no adapter checkpoint", output.ToString());
                if (dpo && suffix.Length == 0)
                {
                    Assert.Contains("dataset = fineweb", output.ToString());
                    Assert.Contains("generation worker=0", output.ToString());
                    Assert.Contains("generation worker=1", output.ToString());
                    foreach (int worker in Enumerable.Range(0, config.GenerationSlots))
                    {
                        string log = output.ToString().Split('\n').First(l => l.Contains($"generation worker={worker},"));
                        Assert.Contains($"device={device}:{config.GenerationDeviceForWorker(worker)},", log);
                    }
                    Assert.Contains($"training device = {device}:0", output.ToString());
                    if (device == "cuda")
                    {
                        string[] streams = output.ToString().Split('\n').Where(l => l.Contains("generation worker="))
                            .Select(l => l[(l.IndexOf("stream=", StringComparison.Ordinal) + 7)..].Trim()).Distinct().ToArray();
                        Assert.True(streams.Length >= config.GenerationSlots);
                        Assert.DoesNotContain("0x0", streams);
                    }
                    int[] peaks = output.ToString().Split('\n').Where(l => l.Contains("peak generation workers="))
                        .Select(l => int.Parse(l[(l.IndexOf("peak generation workers=", StringComparison.Ordinal) + 24)..].Trim())).ToArray();
                    if (!existingAdapter) Assert.Contains(peaks, peak => peak >= 2);
                }
                return result;
            }
            Run();
            if (dpo)
            {
                var first = torch.load<DpoCommand.Checkpoint>(Path.Combine(directory, "adapter.json"));
                string graphPath = Path.Combine(directory, "adapter.loss.html");
                Assert.Contains("<title>step 1:", File.ReadAllText(graphPath));
                string firstMetric = File.ReadLines(TrainingMetricReporter.GetSidecarPath(graphPath)).First();
                Assert.Equal(1, first.Step); Assert.Equal(4, first.Trained);
                Assert.Equal(first.Issued - first.Trained, first.Ready.Length);
                config = config with { Resume = false, MaxSteps = 2 };
                config = config with { BatchSize = 1, GradientAccumulationSteps = 4, CompletedQueueCapacity = 8 };
                if (twoGpu) config = config with { GenerationDeviceIndices = [1] };
                Run();
                var resumed = torch.load<DpoCommand.Checkpoint>(Path.Combine(directory, "adapter.json"));
                Assert.Equal(2, resumed.Step); Assert.Equal(8, resumed.Trained);
                Assert.Contains("<title>step 2:", File.ReadAllText(graphPath));
                Assert.Equal(firstMetric, File.ReadLines(TrainingMetricReporter.GetSidecarPath(graphPath)).First());
                Assert.Contains(first.Adapter.Parameters.Zip(resumed.Adapter.Parameters), p => !p.First.Values.SequenceEqual(p.Second.Values));
                Run("--generate", "hello");
                config = config with { Resume = false, AdapterPath = "uninterrupted.json" };
                Run();
                var continuous = torch.load<DpoCommand.Checkpoint>(Path.Combine(directory, "uninterrupted.json"));
                // Completion order and snapshot age are deliberately asynchronous now.
                // Verify committed cursor/queue integrity, not bitwise-identical training schedules.
                Assert.Equal(continuous.Issued - continuous.Trained, continuous.Ready.Length);
                Assert.Equal(resumed.Issued - resumed.Trained, resumed.Ready.Length);
                Assert.Equal(resumed.Ready.Length, resumed.Ready.Select(p => p.Id).Distinct().Count());
                Assert.All(resumed.Adapter.Parameters, p => Assert.All(p.Values, value => Assert.True(float.IsFinite(value))));
                config = config with { Resume = true, MaxSteps = 3 };
                Run();
                var tail = torch.load<DpoCommand.Checkpoint>(Path.Combine(directory, "uninterrupted.json"));
                Assert.Equal(3, tail.Step); Assert.Equal(9, tail.Trained);
                Assert.Empty(tail.Ready);
                config = config with { Epochs = 2, MaxSteps = 4 };
                Run();
                var nextEpoch = torch.load<DpoCommand.Checkpoint>(Path.Combine(directory, "uninterrupted.json"));
                Assert.Equal(4, nextEpoch.Step); Assert.Equal(2, nextEpoch.Epoch);
                Assert.Equal(4, nextEpoch.Trained);
                Assert.Equal(original, File.ReadAllBytes(modelPath));
                // Auto-resume must not silently replace a valid checkpoint on config mismatch.
                byte[] protectedCheckpoint = File.ReadAllBytes(Path.Combine(directory, config.AdapterPath));
                config = config with { Resume = false, DpoBeta = config.DpoBeta * 2 };
                File.WriteAllText(configPath, JsonSerializer.Serialize(config, LoraConfiguration.Json));
                using var mismatchOutput = new StringWriter(); using var mismatchError = new StringWriter();
                Assert.Equal(2, Program.Run(["lora", "--model", modelPath, "--config", configPath], mismatchOutput, mismatchError));
                Assert.Contains("mismatch", mismatchError.ToString());
                Assert.Equal(protectedCheckpoint, File.ReadAllBytes(Path.Combine(directory, config.AdapterPath)));
                // The user may intentionally remove/move an adapter after a
                // mismatch. Old metrics must survive, without blocking a new run.
                string priorAdapter = Path.Combine(directory, "prior.adapter.json");
                File.Move(Path.Combine(directory, config.AdapterPath), priorAdapter);
                string priorGraph = Path.ChangeExtension(Path.Combine(directory, config.AdapterPath), ".loss.html");
                string priorMetrics = TrainingMetricReporter.GetSidecarPath(priorGraph);
                byte[] oldGraph = File.ReadAllBytes(priorGraph), oldMetrics = File.ReadAllBytes(priorMetrics);
                config = config with { DpoBeta = config.DpoBeta / 2, MaxSteps = 1, Epochs = 1 };
                Run();
                Assert.Equal(1, torch.load<DpoCommand.Checkpoint>(Path.Combine(directory, config.AdapterPath)).Step);
                Assert.Equal(protectedCheckpoint, File.ReadAllBytes(priorAdapter));
                string[] archivedGraphs = Directory.GetFiles(directory, "*.previous-*.html");
                Assert.Equal(oldGraph, File.ReadAllBytes(Assert.Single(archivedGraphs)));
                Assert.Equal(oldMetrics, File.ReadAllBytes(TrainingMetricReporter.GetSidecarPath(archivedGraphs[0])));
                var freshHistory = new NNtrain.Training.Metrics.MetricJournalJsonlRepository(priorMetrics).Load().Journal.Entries;
                Assert.Equal(1, Assert.Single(freshHistory).GlobalStep);
                return;
            }
            var saved = torch.load<LoraCommand.AdapterCheckpoint>(Path.Combine(directory, "adapter.json"));
            Assert.Equal(1, saved.Step); Assert.Equal(2, saved.Documents);
            config = config with { Resume = false, MaxSteps = 2 };
            Run();
            saved = torch.load<LoraCommand.AdapterCheckpoint>(Path.Combine(directory, "adapter.json"));
            Assert.Equal(2, saved.Step); Assert.Equal(4, saved.Documents);
            Run("--generate", "hello");
            Assert.Equal(original, File.ReadAllBytes(modelPath));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public void DpoLegacyCheckpointResumesAndExplicitPrecisionConversionPreservesTrainingState()
    {
        using var session = new NNtrain.Runtime.Execution.ExecutionSession(new NNtrain.Runtime.Execution.ExecutionOptions {
            Device = NNtrain.Runtime.Execution.ExecutionDeviceKind.Cpu });
        using var scope = session.Enter();
        string directory = Path.Combine(Path.GetTempPath(), "NNtrain.LoraPrecision." + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string modelPath = Path.Combine(directory, "base.json");
            string configPath = Path.Combine(directory, "train.json");
            string adapterPath = Path.Combine(directory, "adapter.json");
            var tokenizer = BpeTokenizer.Train(["hello world good answer", "question response"], vocabularySize: 300);
            tokenizer.save(Path.Combine(directory, "tokenizer.json"));
            var model = new ForgetMemoryDRNGpt(tokenizer.VocabularySize, 32, 8, 16, 1, 2, 2, dtype: TensorDType.Float32);
            torch.save(new WikiLanguageModelCommand.WikiModelCheckpoint(6, 0, 1,
                tokenizer.VocabularySize, 32, 8, 1, 16, 1, 0, .02f, model.state_dict(),
                ModelArchitecture: "forgetmemorydrn", ForgetMemoryKeyWidth: 2, ForgetMemoryValueWidth: 2,
                ModelDType: TensorDType.Float32, PrecisionMode: TensorPrecisionMode.Float32), modelPath);
            byte[] originalBase = File.ReadAllBytes(modelPath);
            File.WriteAllText(Path.Combine(directory, "lora.json"), """
                { "rank": 2, "alpha": 4, "precisionMode": "mix16_32" }
                """);
            File.WriteAllLines(Path.Combine(directory, "data.jsonl"), Enumerable.Repeat(
                "{\"text\":\"hello world good answer question response\"}", 6));
            var config = new LoraTrainingConfiguration {
                Objective = "dpo", Device = "cpu", LoraConfig = "lora.json", TokenizerPath = "tokenizer.json",
                DataPath = "data.jsonl", AdapterPath = "adapter.json", ContextLength = 16,
                BatchSize = 1, GradientAccumulationSteps = 1, GenerationSlots = 1, CompletedQueueCapacity = 2,
                MaxSteps = 1, SaveEverySteps = 1, AutoResume = true, SampleEverySteps = 0,
                PromptPrefix = "", ResponsePrefix = "", MaxNewTokens = 2 };
            void Run(bool success = true)
            {
                File.WriteAllText(configPath, JsonSerializer.Serialize(config, LoraConfiguration.Json));
                using var output = new StringWriter();
                using var error = new StringWriter();
                int exitCode = Program.Run(["lora", "--model", modelPath, "--config", configPath], output, error);
                Assert.True(exitCode == (success ? 0 : 2), output + "\n" + error);
                if (!success) Assert.Contains("mismatch", error.ToString());
            }

            Run();
            var first = torch.load<DpoCommand.Checkpoint>(adapterPath);
            Assert.Equal(1, first.Step);
            Assert.Equal(1, first.Trained);
            var legacyContract = System.Text.Json.Nodes.JsonNode.Parse(first.Contract)!.AsObject();
            legacyContract["lora"]!.AsObject().Remove("bfp8BlockSize");
            // Populate a valid, nonzero second moment so later updates prove
            // that resume restored moments, not merely the optimizer step.
            AdamWState firstOptimizer = first.Optimizer.StateJson!.Value.Deserialize<AdamWState>()!;
            AdamWState seededOptimizer = firstOptimizer with {
                ParameterStates = firstOptimizer.ParameterStates.Select(state => state with {
                    SecondMoment = Enumerable.Repeat(.25f, state.SecondMoment.Length).ToArray() }).ToArray() };
            torch.save(first with { Contract = legacyContract.ToJsonString(),
                Optimizer = new OptimizerStateDictionary("AdamW", JsonSerializer.SerializeToElement(seededOptimizer), []) }, adapterPath);

            config = config with { MaxSteps = 2 };
            Run();
            var legacyResumed = torch.load<DpoCommand.Checkpoint>(adapterPath);
            AdamWState previousOptimizer = legacyResumed.Optimizer.StateJson!.Value.Deserialize<AdamWState>()!;
            Assert.Equal(2, legacyResumed.Step);
            Assert.Equal(2, previousOptimizer.Step);
            Assert.Equal(2, legacyResumed.Trained);
            Assert.All(previousOptimizer.ParameterStates, state => Assert.All(state.SecondMoment,
                value => Assert.True(value >= .25f * previousOptimizer.Options.Beta2 - 1e-7f)));

            byte[] beforeConversion = File.ReadAllBytes(adapterPath);
            config = config with { PrecisionMode = "mix8_32", Bfp8BlockSize = 32, MaxSteps = 3 };
            Run(success: false);
            Assert.Equal(beforeConversion, File.ReadAllBytes(adapterPath));
            config = config with { AllowPrecisionConversionOnResume = true };
            Run();
            var converted = torch.load<DpoCommand.Checkpoint>(adapterPath);
            AdamWState convertedOptimizer = converted.Optimizer.StateJson!.Value.Deserialize<AdamWState>()!;
            Assert.Equal(3, converted.Step);
            Assert.Equal(3, convertedOptimizer.Step);
            Assert.Equal(3, converted.Trained);
            Assert.Equal(legacyResumed.Epoch, converted.Epoch);
            Assert.Equal(converted.Issued - converted.Trained, converted.Ready.Length);
            Assert.Equal(converted.Ready.Length, converted.Ready.Select(pair => pair.Id).Distinct().Count());
            Assert.All(converted.Adapter.Parameters, parameter => {
                Assert.Equal(TensorDType.Bfp8, parameter.DType);
                Assert.All(parameter.Values, value => Assert.True(float.IsFinite(value)));
            });
            Assert.Equal(previousOptimizer.ParameterStates.Length, convertedOptimizer.ParameterStates.Length);
            foreach (var (previous, current) in previousOptimizer.ParameterStates.Zip(convertedOptimizer.ParameterStates))
            {
                Assert.Equal(previous.Shape, current.Shape);
                foreach (var (oldMoment, newMoment) in previous.SecondMoment.Zip(current.SecondMoment))
                    Assert.True(newMoment >= oldMoment * convertedOptimizer.Options.Beta2 - 1e-7f);
            }
            using var contract = JsonDocument.Parse(converted.Contract);
            Assert.Equal("mix8_32", contract.RootElement.GetProperty("lora").GetProperty("precisionMode").GetString());
            Assert.Equal(32, contract.RootElement.GetProperty("lora").GetProperty("bfp8BlockSize").GetInt32());
            byte[] protectedConverted = File.ReadAllBytes(adapterPath);
            config = config with { DpoBeta = config.DpoBeta * 2, MaxSteps = 4 };
            Run(success: false);
            Assert.Equal(protectedConverted, File.ReadAllBytes(adapterPath));
            Assert.Equal(originalBase, File.ReadAllBytes(modelPath));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public void ConfigsAndMissingModelOptionFailClearly()
    {
        Assert.Throws<NotSupportedException>(() => new LoraConfiguration { Architecture = "transformer" }.Validate());
        Assert.Throws<NotSupportedException>(() => new LoraTrainingConfiguration { Objective = "unknown" }.Validate());
        using var output = new StringWriter(); using var error = new StringWriter();
        Assert.Equal(2, Program.Run(["lora"], output, error));
        Assert.Contains("--model", error.ToString());
    }

    [Theory]
    [InlineData(0, 100, false)]
    [InlineData(99, 100, false)]
    [InlineData(100, 100, true)]
    [InlineData(101, 100, false)]
    [InlineData(200, 100, true)]
    [InlineData(200, 0, false)]
    public void SamplesUseCommittedGlobalStep(long step, int interval, bool expected)
    {
        Assert.Equal(expected, DpoCommand.ShouldGenerateSample(step, interval));
        Assert.Equal(100, new LoraTrainingConfiguration().SampleEverySteps);
        Assert.Throws<ArgumentException>(() => new LoraTrainingConfiguration { SampleEverySteps = -1 }.Validate());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SampleRestoresModeAndDoesNotChangeParameters(bool cancel)
    {
        using var session = new NNtrain.Runtime.Execution.ExecutionSession(new NNtrain.Runtime.Execution.ExecutionOptions {
            Device = NNtrain.Runtime.Execution.ExecutionDeviceKind.Cpu });
        using var scope = session.Enter();
        var tokenizer = BpeTokenizer.Train(["hello world"], vocabularySize: 270);
        var model = new ForgetMemoryDRNGpt(tokenizer.VocabularySize, 32, 8, 16, 1, 2, 2, dtype: TensorDType.Float32);
        model.AttachLora(2, 4, ["memoryOutput"], 1234);
        var before = model.state_dict();
        var pair = new DpoCommand.Pair(0, [BpeTokenizer.BosTokenId], tokenizer.Encode("hello"), []);
        using var output = new StringWriter();
        using var cancellation = new CancellationTokenSource();
        if (cancel) cancellation.Cancel();
        void Sample() => DpoCommand.WriteSample(model, tokenizer, pair,
            new LoraTrainingConfiguration { MaxNewTokens = 1 }, 100, 1234, output, cancellation.Token);
        if (cancel) Assert.Throws<OperationCanceledException>(Sample); else Sample();
        Assert.True(model.IsTraining);
        Assert.All(before.Parameters.Zip(model.state_dict().Parameters), p => Assert.Equal(p.First.Values, p.Second.Values));
        model.eval();
        if (cancel) Assert.Throws<OperationCanceledException>(Sample); else Sample();
        Assert.False(model.IsTraining);
    }
}
