using System.Text.Json;
using NNtrain;
using NNtrain.Arc;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Metrics;
using Parquet;
using Parquet.Schema;
using Xunit;

[CollectionDefinition("Arc training integration", DisableParallelization = true)]
public sealed class ArcTrainingIntegrationCollection { }

[Collection("Arc training integration")]
public sealed class ArcTrainingIntegrationTests
{
    [Theory]
    [InlineData("cpu", TensorDevice.Cpu)]
    [InlineData("cuda", TensorDevice.Cuda)]
    [InlineData("arc", TensorDevice.Arc)]
    public void ConfigurationKeepsDistinctExecutionDevices(string device, TensorDevice expected)
    {
        WikiTrainingConfiguration config = SmallConfiguration() with { Device = device };
        config.Validate();
        Assert.Equal(expected, config.GetExecutionDevice());
    }

    [Theory]
    [InlineData("float32", "muon")]
    [InlineData("mix16_32", "muon")]
    [InlineData("mix8_32", "muon")]
    [InlineData("float32", "nekomuon")]
    [InlineData("mix16_32", "nekomuon")]
    [InlineData("mix8_32", "nekomuon")]
    [InlineData("float32", "adamw")]
    [InlineData("mix16_32", "adamw")]
    [InlineData("mix8_32", "adamw")]
    public void ArcAcceptsSupportedPrecisionAndOptimizerWithoutTouchingGpu(string precision, string optimizer)
    {
        WikiTrainingConfiguration config = SmallConfiguration() with
        {
            PrecisionMode = precision,
            Optimizer = optimizer,
        };
        config.Validate();
        Assert.Equal(TensorDevice.Arc, config.GetExecutionDevice());
        Assert.Equal(TensorPrecisionModeNames.Parse(precision), config.GetPrecisionMode());
    }

    [Theory]
    [InlineData("bfloat16")]
    [InlineData("bfp8")]
    public void ArcRejectsUnimplementedPureLowPrecisionBeforeOpeningGpu(string precision)
    {
        WikiTrainingConfiguration config = SmallConfiguration() with { PrecisionMode = precision };
        NotSupportedException failure = Assert.Throws<NotSupportedException>(config.Validate);
        Assert.Contains("pure low-precision", failure.Message);
        Assert.Throws<NotSupportedException>(() =>
            ProductionTrainingSessionFactory.CreateExecutionSession(
                config.GetPrecisionMode(), TensorDevice.Arc, [0]));
    }

    [Fact]
    public void ArcRejectsDrnBeforeModelConstruction()
    {
        WikiTrainingConfiguration config = SmallConfiguration() with { ModelArchitecture = "forgetmemorydrn" };
        NotSupportedException failure = Assert.Throws<NotSupportedException>(config.Validate);
        Assert.Contains("transformer", failure.Message);
        Assert.Contains("DRN", failure.Message);
    }

    [Fact]
    public void ArcRejectsMultiDeviceInsteadOfSilentlyUsingCudaDataParallel()
    {
        WikiTrainingConfiguration config = SmallConfiguration() with { DeviceIndices = [0, 1] };
        NotSupportedException failure = Assert.Throws<NotSupportedException>(config.Validate);
        Assert.Contains("single GPU", failure.Message);
        Assert.Throws<NotSupportedException>(() =>
            ProductionTrainingSessionFactory.CreateExecutionSession(
                TensorPrecisionMode.Mix8_32, TensorDevice.Arc, [0, 1]));
    }

    [Fact]
    public void ArcRejectsUnsupportedOptimizerRatherThanComputingOnCpu()
    {
        WikiTrainingConfiguration config = SmallConfiguration() with { Optimizer = "lion" };
        Assert.Contains("Muon, NekoMuon and AdamW",
            Assert.Throws<NotSupportedException>(config.Validate).Message);
    }

    [Fact]
    public void ProductionFactoryOwnsArcLaneAndRestoresAmbientSession()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc OpenCL GPU unavailable; real Arc execution is required.");
        ExecutionSession? previousSession = ExecutionSession.Current;
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int cudaFactoryCalls = 0;
        ArcExecutionLane lane;
        using (ExecutionSession execution = ProductionTrainingSessionFactory.CreateExecutionSession(
            TensorPrecisionMode.Mix8_32, TensorDevice.Arc, [0],
            _ => { cudaFactoryCalls++; throw new InvalidOperationException("Arc must not create a CUDA lane."); }))
        {
            lane = Assert.IsType<ArcExecutionLane>(Assert.Single(execution.Lanes));
            Assert.Equal(ExecutionDeviceKind.Arc, execution.Options.Device);
            Assert.Equal(PrecisionMode.Mix8_32, execution.Options.Precision.Mode);
            Assert.Equal(0, execution.Options.ArcDeviceIndex);
            Assert.Contains("Arc", lane.Device.Name, StringComparison.OrdinalIgnoreCase);
            using (execution.Enter())
            {
                Assert.Same(execution, ExecutionSession.Current);
                Assert.Equal(TensorDevice.Arc, Tensor.ExecutionDevice);
                _ = lane.Allocate(16); // Deliberately let the session own this allocation.
                Assert.True(lane.AllocatedBytes >= 64);
            }
            Assert.Same(previousSession, ExecutionSession.Current);
            Assert.Equal(previousDevice, Tensor.ExecutionDevice);
        }
        Assert.Equal(0, cudaFactoryCalls);
        Assert.Equal(0, lane.AllocatedBytes);
        Assert.Throws<ObjectDisposedException>(() => lane.Allocate(1));
    }

    [Fact]
    public async Task FineWebMuonMixed8TrainsResumesAndGeneratesWithPersistentHtml()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc OpenCL GPU unavailable; training must not silently fall back to CPU.");
        using var directory = new TemporaryDirectory();
        string dataDirectory = Path.Combine(directory.Root, "fineweb");
        Directory.CreateDirectory(dataDirectory);
        string[] documents = Enumerable.Range(0, 8).Select(index =>
            $"Document {index}: the quick brown fox jumps over the lazy dog. " +
            "Small transformer training checks accumulation, optimizer state, checkpoint resume and generation. " +
            "日本語のテスト文書です。学習を再開して履歴が継続することを確認します。").ToArray();
        await WriteShard(Path.Combine(dataDirectory, "train-00000.parquet"), documents);
        BpeTokenizer tokenizer = BpeTokenizer.Train(documents, vocabularySize: 300);
        Assert.Equal(300, tokenizer.VocabularySize);
        tokenizer.save(Path.Combine(directory.Root, "tokenizer.json"));
        string configPath = Path.Combine(directory.Root, "training.arc.json");
        string checkpointPath = Path.Combine(directory.Root, "checkpoint.json");
        string graphPath = Path.ChangeExtension(configPath, ".loss.html");
        string metricsPath = TrainingMetricReporter.GetSidecarPath(graphPath);
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousCudaIndices = Tensor.CudaDeviceIndices.ToArray();
        bool previousSimd = Tensor.SimdEnabled;
        int previousWorkers = Tensor.MaxDegreeOfParallelism;
        try
        {
            void WriteConfiguration(int epochs) => File.WriteAllText(configPath, JsonSerializer.Serialize(new
            {
                task = "gpt_rin_wiki_jp", dataset = "fineweb", dataPath = "fineweb", textColumn = "text",
                tokenizerPath = "tokenizer.json", checkpointPath = "checkpoint.json", autoResume = true,
                checkpointIntervalMinutes = 30, vocabularySize = 300,
                tokenizerTrainingDocuments = 8, tokenizerTrainingBytes = 10000,
                maxTrainingDocuments = 0, maxTrainingTokens = 0, maxDocumentTokens = 4,
                shuffleBufferSize = 0, validationFraction = 0, epochs,
                batchSize = 2, gradientAccumulationSteps = 2, contextLength = 4,
                modelWidth = 8, heads = 2, hiddenSize = 16, layers = 2,
                modelArchitecture = "transformer", device = "arc", deviceIndex = 0, deviceIndices = new[] { 0 },
                precisionMode = "mix8_32", bfp8_block_size = 32, tieWordEmbeddings = true, dropout = 0.1,
                optimizer = "muon", learningRate = 0.001, auxiliaryLearningRate = 0.0003,
                nekoMuonNewtonSchulzDepthMode = "fixed", nekoMuonNewtonSchulzDepth = 5,
                nekoMuonNewtonSchulzInterval = 1, weightDecay = 0.01, warmupPercent = 0,
                seed = 37, logEveryBatches = 1, showLossGraph = true, graphUpdateSteps = 1,
                datasetSampleEverySteps = 1, datasetSamplePoolSize = 1,
                maxNewTokens = 1, temperature = 0.0, topK = 1,
            }));

            string Run(params string[] suffix)
            {
                using var output = new StringWriter();
                using var error = new StringWriter();
                int exit = Program.Run(["--config", configPath, .. suffix], output, error, openLossGraph: false);
                Assert.True(exit == 0, $"Arc CLI failed ({exit}):{Environment.NewLine}{error}{Environment.NewLine}{output}");
                Assert.Equal(string.Empty, error.ToString()); // Sample-generation warnings must not hide unsupported operations.
                Assert.Contains("device = arc", output.ToString());
                Assert.Contains("Intel(R) Arc", output.ToString());
                Assert.DoesNotContain("CUDA data parallel =", output.ToString());
                return output.ToString();
            }

            WriteConfiguration(1);
            string firstOutput = Run();
            Assert.Contains("accumulation 2/2", firstOutput);
            Assert.Contains("optimizer = Muon", firstOutput);
            Assert.Contains("+ AdamW", firstOutput);
            Assert.Contains("[model continuation]", firstOutput);
            Assert.Contains("dataset continuation sample at step 1", firstOutput);
            WikiLanguageModelCommand.WikiModelCheckpoint first =
                torch.load<WikiLanguageModelCommand.WikiModelCheckpoint>(checkpointPath);
            Assert.Equal(1, first.CompletedEpoch);
            Assert.True(first.GlobalStep > 0);
            Assert.Equal(TensorPrecisionMode.Mix8_32, first.PrecisionMode);
            Assert.Equal(TensorDType.Bfp8, first.ModelDType);
            Assert.Equal(new[] { "NekoMuon", "AdamW" }, first.OptimizerStateTypes);
            ModuleState firstState = safetensors.torch.load_file(
                WikiLanguageModelCommand.GetCurrentModelArtifactPath(checkpointPath, first.ArtifactSlot));
            Assert.All(firstState.Parameters, parameter => Assert.All(parameter.Values, value => Assert.True(float.IsFinite(value))));
            string[] firstMetricLines = File.ReadAllLines(metricsPath);
            Assert.NotEmpty(firstMetricLines);
            Assert.Contains("train points", File.ReadAllText(graphPath));
            Assert.False(File.Exists(TrainingRunGuard.GetMarkerPath(checkpointPath)));

            // Reproduce automatic resume without mutating any real run: only this disposable fixture's marker is written.
            File.WriteAllText(TrainingRunGuard.GetMarkerPath(checkpointPath), "{\"interrupted\":true}");
            WriteConfiguration(2);
            string resumedOutput = Run();
            Assert.Contains("auto-resume = interrupted training detected", resumedOutput);
            Assert.Contains("resumed checkpoint = " + checkpointPath + ", next epoch 2", resumedOutput);
            Assert.Contains("resume Muon policy", resumedOutput);
            WikiLanguageModelCommand.WikiModelCheckpoint resumed =
                torch.load<WikiLanguageModelCommand.WikiModelCheckpoint>(checkpointPath);
            Assert.Equal(2, resumed.CompletedEpoch);
            Assert.True(resumed.GlobalStep > first.GlobalStep);
            Assert.Equal(first.PrecisionMode, resumed.PrecisionMode);
            Assert.Equal(first.OptimizerStateTypes, resumed.OptimizerStateTypes);
            ModuleState resumedState = safetensors.torch.load_file(
                WikiLanguageModelCommand.GetCurrentModelArtifactPath(checkpointPath, resumed.ArtifactSlot));
            Assert.All(resumedState.Parameters, parameter => Assert.All(parameter.Values, value => Assert.True(float.IsFinite(value))));
            Assert.Contains(firstState.Parameters.Zip(resumedState.Parameters),
                pair => !pair.First.Values.SequenceEqual(pair.Second.Values));
            string[] resumedMetricLines = File.ReadAllLines(metricsPath);
            Assert.True(resumedMetricLines.Length > firstMetricLines.Length);
            Assert.Equal(firstMetricLines, resumedMetricLines.Take(firstMetricLines.Length).ToArray());
            MetricJournalLoadResult metrics = new MetricJournalJsonlRepository(metricsPath).Load();
            Assert.Contains(metrics.Journal.Entries, entry => entry.GlobalStep <= first.GlobalStep);
            Assert.Contains(metrics.Journal.Entries, entry => entry.GlobalStep > first.GlobalStep);
            Assert.Contains("train points", File.ReadAllText(graphPath));
            Assert.False(File.Exists(TrainingRunGuard.GetMarkerPath(checkpointPath)));

            byte[] checkpointBeforeGeneration = File.ReadAllBytes(checkpointPath);
            byte[] metricsBeforeGeneration = File.ReadAllBytes(metricsPath);
            Assert.Contains("generated text:", Run("--generate", "Small transformer"));
            Assert.Equal(checkpointBeforeGeneration, File.ReadAllBytes(checkpointPath));
            Assert.Equal(metricsBeforeGeneration, File.ReadAllBytes(metricsPath));
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousCudaIndices;
            Tensor.ExecutionDevice = previousDevice;
            Tensor.SimdEnabled = previousSimd;
            Tensor.MaxDegreeOfParallelism = previousWorkers;
        }
    }

    private static WikiTrainingConfiguration SmallConfiguration() => new()
    {
        Device = "arc", DeviceIndices = [0], ModelArchitecture = "transformer", PrecisionMode = "mix8_32",
        VocabularySize = 300, ContextLength = 4, ModelWidth = 8, Heads = 2, HiddenSize = 16, Layers = 2,
        Optimizer = "muon", NekoMuonNewtonSchulzInterval = 1,
    };

    private static async Task WriteShard(string path, string[] documents)
    {
        var text = new DataField<string>("text");
        await using Stream file = File.Create(path);
        await using ParquetWriter writer = await ParquetWriter.CreateAsync(new ParquetSchema(text), file,
            cancellationToken: TestContext.Current.CancellationToken);
        using ParquetRowGroupWriter group = writer.CreateRowGroup();
        await group.WriteAsync(text, documents);
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        internal string Root { get; } = Path.Combine(Path.GetTempPath(), "NNtrain.ArcIntegration-" + Guid.NewGuid().ToString("N"));
        internal TemporaryDirectory() => Directory.CreateDirectory(Root);
        public void Dispose() => Directory.Delete(Root, recursive: true);
    }
}
