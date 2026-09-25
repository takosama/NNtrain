using System.Security.Cryptography;
using System.Text.Json;
using NNtrain;
using NNtrain.Arc;
using NNtrain.Runtime.Execution;
using Xunit;

[Collection("Arc training integration")]
public sealed class ArcGenerationCommandTests
{
    [Theory]
    [InlineData("single", 1)]
    [InlineData("tensorParallel", 2)]
    public void BothGenerationCommandsRunBoundedMix8_16ContinuationAndReleaseTheirLanes(string mode, int deviceCount)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count < deviceCount,
            $"{deviceCount} Intel Arc OpenCL GPU(s) are required.");
        using var directory = new TemporaryDirectory();
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousCudaIndices = Tensor.CudaDeviceIndices.ToArray();
        bool previousSimd = Tensor.SimdEnabled;
        int previousWorkers = Tensor.MaxDegreeOfParallelism;
        ExecutionSession? previousSession = ExecutionSession.Current;
        try
        {
            const string prompt = "abc";
            const int newTokens = 6, context = 8, width = 64, heads = 2, hidden = 64, layers = 1;
            BpeTokenizer tokenizer = BpeTokenizer.Train([prompt], BpeTokenizer.BaseVocabularySize);
            int nextToken = Assert.Single(tokenizer.Encode("a"));
            string tokenizerPath = Path.Combine(directory.Root, "tokenizer.json");
            string checkpointPath = Path.Combine(directory.Root, "checkpoint.json");
            string safeTensorsPath = Path.Combine(directory.Root, "generation-weights.safetensors");
            string trainingPath = Path.Combine(directory.Root, "training.json");
            string generationPath = Path.Combine(directory.Root, "generate.json");
            tokenizer.save(tokenizerPath);

            ModuleState checkpointState;
            // Artifact creation needs no GPU. FP32 SafeTensors are converted
            // to the configured mix8_16 model by the real generation command.
            using (ExecutionSession cpu = ProductionTrainingSessionFactory.CreateExecutionSession(
                TensorPrecisionMode.Float32, TensorDevice.Cpu, [0]))
            using (cpu.Enter())
            {
                var model = new GptRinWikiJp(tokenizer.VocabularySize, context, width, heads,
                    hidden, layers, new Random(31), tieWordEmbeddings: true);
                ModuleState state = model.state_dict();
                ModuleParameterState bias = Assert.Single(state.Parameters, parameter =>
                    parameter.Name == "B" && parameter.Shape.SequenceEqual([tokenizer.VocabularySize]));
                Array.Clear(bias.Values);
                bias.Values[nextToken] = 100f;
                model.load_state_dict(state);
                safetensors.torch.save_file(state, safeTensorsPath);
                model.to(TensorPrecisionMode.Mix8_16, 32);
                checkpointState = model.state_dict();
            }
            var checkpoint = new WikiLanguageModelCommand.WikiModelCheckpoint(
                FormatVersion: 6, Epoch: 1, ValidationLoss: 1f,
                VocabularySize: tokenizer.VocabularySize, ContextLength: context,
                ModelWidth: width, Heads: heads, HiddenSize: hidden, Layers: layers,
                Dropout: 0f, InitializationScale: .02f, Model: checkpointState,
                ModelArchitecture: "transformer", ModelDType: TensorDType.Bfp8,
                TieWordEmbeddings: true, PrecisionMode: TensorPrecisionMode.Mix8_16,
                Bfp8BlockSize: 32);
            torch.save(checkpoint, checkpointPath);
            int[] indices = Enumerable.Range(0, deviceCount).ToArray();
            File.WriteAllText(trainingPath, JsonSerializer.Serialize(new
            {
                task = "gpt_rin_wiki_jp", dataPath = "unused", tokenizerPath = "tokenizer.json",
                checkpointPath = "checkpoint.json", vocabularySize = tokenizer.VocabularySize,
                contextLength = context, modelWidth = width, heads, hiddenSize = hidden, layers,
                modelArchitecture = "transformer", precisionMode = "mix8_16", bfp8_block_size = 32,
                tieWordEmbeddings = true, device = "arc", deviceIndices = new[] { 0 },
                inferenceDeviceIndices = indices, arcInferenceMode = mode,
                dropout = 0f, maxNewTokens = newTokens, temperature = 0f, topK = 1, seed = 31,
                showLossGraph = false,
            }));
            File.WriteAllText(generationPath, JsonSerializer.Serialize(new
            {
                trainingConfigPath = "training.json", safeTensorsPath = "generation-weights.safetensors",
                prompt, sampling = "greedy", maxNewTokens = newTokens,
                inferenceDeviceIndices = indices, arcInferenceMode = mode,
            }));
            string[] artifacts = [tokenizerPath, checkpointPath, safeTensorsPath, trainingPath, generationPath];
            var originalHashes = artifacts.ToDictionary(path => path, HashFile);
            string expectedText = prompt + new string('a', newTokens);
            string expectedRoute = deviceCount == 2
                ? "Arc inference = tensorParallel GPUs [0,1]" : "Arc inference = single GPU [0]";

            void Run(params string[] arguments)
            {
                using var output = new SessionObservingWriter();
                using var error = new StringWriter();
                int exit = Program.Run(arguments, output, error, openLossGraph: false);
                Assert.True(exit == 0, $"Generation failed ({exit}):{Environment.NewLine}"
                    + error + Environment.NewLine + output);
                Assert.Equal(string.Empty, error.ToString());
                Assert.Contains(expectedRoute, output.ToString());
                Assert.Contains("generated text:" + Environment.NewLine + expectedText, output.ToString());
                Assert.NotNull(output.GenerationSession);
                Assert.True(output.GenerationSession.IsDisposed);
                Assert.Same(previousSession, ExecutionSession.Current);
                Assert.Equal(deviceCount, output.Lanes.Length);
                Assert.Equal(indices, output.Lanes.Select(lane => lane.DeviceIndex).ToArray());
                Assert.All(output.Lanes, lane => {
                    Assert.True(lane.KernelLaunchCount > 0);
                    Assert.Equal(0, lane.AllocatedBytes);
                    Assert.Equal(0, lane.CachedBytes);
                });
                foreach (string path in artifacts)
                    Assert.Equal(originalHashes[path], HashFile(path));
            }

            // The six-token suffix crosses the eight-token absolute-position
            // limit, exercising incremental decoding and the window fallback
            // when the production KV-cache default is enabled.
            Run("--config", trainingPath, "--generate", prompt);
            Run("--generate-config", generationPath);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousCudaIndices;
            Tensor.ExecutionDevice = previousDevice;
            Tensor.SimdEnabled = previousSimd;
            Tensor.MaxDegreeOfParallelism = previousWorkers;
        }
    }

    private static string HashFile(string path)
    {
        using FileStream input = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(input));
    }

    private sealed class SessionObservingWriter : StringWriter
    {
        internal ExecutionSession? GenerationSession { get; private set; }
        internal ArcExecutionLane[] Lanes { get; private set; } = [];

        private void ObserveSession()
        {
            if (ExecutionSession.Current is { Options.Device: ExecutionDeviceKind.Arc } session)
            {
                Assert.False(session.IsDisposed);
                GenerationSession = session;
                Lanes = session.Lanes.OfType<ArcExecutionLane>().OrderBy(lane => lane.DeviceIndex).ToArray();
            }
        }

        public override void Write(string? value) { ObserveSession(); base.Write(value); }
        public override void WriteLine(string? value) { ObserveSession(); base.WriteLine(value); }
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        internal string Root { get; } = Path.Combine(Path.GetTempPath(), "NNtrain.ArcGeneration-" + Guid.NewGuid().ToString("N"));
        internal TemporaryDirectory() => Directory.CreateDirectory(Root);
        public void Dispose() => Directory.Delete(Root, recursive: true);
    }
}
