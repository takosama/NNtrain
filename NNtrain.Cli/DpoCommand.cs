using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using NNtrain.Runtime.Execution;
using NNtrain.Cuda.Execution;

namespace NNtrain;

internal static class DpoCommand
{
    internal sealed record Pair(long Id, int[] Prefix, int[] Chosen, int[] Rejected);
    internal sealed record TextExample(string Text);
    private sealed record AdapterSnapshot(int Step, ModuleState State);
    private sealed record GenerationRequest(Pair Pair, Random Random, AdapterSnapshot Snapshot);
    internal sealed record Checkpoint(int Version, string BaseHash, string TokenizerHash, string DataHash,
        string Contract, int Step, int Epoch, long Issued, long Trained, Pair[] Ready,
        ModuleState Adapter, OptimizerStateDictionary Optimizer);

    internal static int Run(string modelPath, string configPath, LoraTrainingConfiguration config,
        string? generate, TextWriter output)
    {
        string Resolve(string p) => Path.GetFullPath(p, Path.GetDirectoryName(configPath)!);
        var lora = LoraConfiguration.Load<LoraConfiguration>(Resolve(config.LoraConfig)).WithTrainingOverrides(config);
        lora.Validate();
        if (config.Device == "cuda")
        {
            int[] requiredDevices = generate is null
                ? Enumerable.Range(0, config.GenerationSlots).Select(config.GenerationDeviceForWorker).Append(config.DeviceIndex).Distinct().ToArray()
                : [config.DeviceIndex];
            foreach (int device in requiredDevices)
                if (!Tensor.IsCudaAvailable(device)) throw new InvalidOperationException($"Configured CUDA device {device} is unavailable; no automatic CPU/single-GPU fallback.");
        }
        string adapterPath = Resolve(config.AdapterPath), dataPath = Resolve(config.DataPath);
        string tokenizerPath = Resolve(config.TokenizerPath);
        string graphPath = config.LossGraphPath is null ? Path.ChangeExtension(adapterPath, ".loss.html") : Resolve(config.LossGraphPath);
        string metricPath = TrainingMetricReporter.GetSidecarPath(graphPath);
        string[] protectedPaths = [modelPath, tokenizerPath, dataPath, configPath, Resolve(config.LoraConfig), adapterPath];
        if (generate is null && (graphPath.Equals(metricPath, StringComparison.OrdinalIgnoreCase) ||
            protectedPaths.Any(p => p.Equals(graphPath, StringComparison.OrdinalIgnoreCase) || p.Equals(metricPath, StringComparison.OrdinalIgnoreCase))))
            throw new ArgumentException("DPO graph/metrics output must not overwrite model, adapter, data or configuration.");
        if (new[] { modelPath, tokenizerPath, dataPath, configPath, Resolve(config.LoraConfig) }
            .Any(p => p.Equals(adapterPath, StringComparison.OrdinalIgnoreCase)))
            throw new ArgumentException("DPO output must not overwrite input files.");
        bool resume = LoraResumePolicy.Resolve(config, adapterPath, generate is not null, output);
        var tokenizer = tokenizers.load_bpe(tokenizerPath);
        string tokenizerHash = Hash(tokenizerPath), dataHash = generate is null ? DpoDataSource.Fingerprint(dataPath, config.Dataset) : "";
        string contract = JsonSerializer.Serialize(new { lora, config.ContextLength, config.BatchSize,
            config.GradientAccumulationSteps, config.LearningRate, config.WeightDecay, config.GradientClip,
            config.DpoBeta, config.GenerationSlots, config.CompletedQueueCapacity, config.CutMinimum,
            config.CutMaximum, config.Temperature, config.TopK, config.Dataset, config.TextColumn,
            generationExecution = "parallel-streams-v1" }, LoraConfiguration.Json);
        var mode = TensorPrecisionModeNames.Parse(lora.PrecisionMode);
        ForgetMemoryDRNGpt model;
        LoraAdapterSet adapters;
        string baseHash;
        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu, 0)))
        {
            (model, baseHash) = WikiLanguageModelCommand.LoadLoraBase(modelPath, mode, lora.Seed, lora.Bfp8BlockSize);
            if (model.VocabularySize != tokenizer.VocabularySize || config.ContextLength > model.ContextLength)
                throw new InvalidDataException("DPO tokenizer vocabulary/context does not match base model.");
            adapters = model.AttachLora(lora.Rank, lora.Alpha, lora.Targets, lora.Seed);
            model.FreezeLoraBaseLinear();
        }
        bool cuda = config.Device == "cuda";
        using var session = new ExecutionSession(new ExecutionOptions {
            Device = cuda ? ExecutionDeviceKind.Cuda : ExecutionDeviceKind.Cpu,
            CudaDevices = new DeviceSet([config.DeviceIndex]), Precision = PrecisionPolicy.Parse(lora.PrecisionMode)
        }, cuda ? [CudaExecutionLaneFactory.Create(config.DeviceIndex)] : []);
        using var scope = session.Enter();
        using var policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Current with {
            DisableCudaGraphs = true, EnableBlockBfp8OptimizerState = false });
        var optimizer = new AdamW(adapters.parameters(), new AdamWOptions {
            LearningRate = config.LearningRate, WeightDecay = config.WeightDecay });
        using var cancellation = new CancellationTokenSource();
        ConsoleCancelEventHandler cancel = (_, e) => { e.Cancel = true; cancellation.Cancel(); };
        Console.CancelKeyPress += cancel;
        try
        {
            int step = 0, epoch = 1;
            long issued = 0, trained = 0;
            Pair[] restored = [];
            if (resume)
            {
                var saved = torch.load<Checkpoint>(adapterPath);
                var mismatches = new List<string>();
                if (saved.Version != 1) mismatches.Add("checkpoint version");
                if (saved.BaseHash != baseHash) mismatches.Add("base model");
                if (saved.TokenizerHash != tokenizerHash) mismatches.Add("tokenizer");
                var contractMismatches = ContractMismatches(saved.Contract, contract, config.AllowPrecisionConversionOnResume);
                if (contractMismatches.Count > 0)
                    mismatches.Add("training config [" + string.Join("; ", contractMismatches) + "]");
                if (generate is null && saved.DataHash != dataHash) mismatches.Add("dataset");
                if (mismatches.Count > 0)
                    throw new InvalidDataException("DPO checkpoint mismatch: " + string.Join(", ", mismatches) + ". Checkpoint was not modified.");
                if (saved.Step < 0 || saved.Epoch < 1 || saved.Issued < 0 || saved.Trained < 0 ||
                    saved.Trained > saved.Issued || saved.Ready.Length != saved.Issued - saved.Trained)
                    throw new InvalidDataException("Invalid DPO committed cursor/queue.");
                adapters.load_state_dict(saved.Adapter);
                if (generate is null) optimizer.load_state_dict(saved.Optimizer);
                if (!CompatibleContract(saved.Contract, contract))
                    output.WriteLine($"resume precision conversion = {lora.PrecisionMode} (block {lora.Bfp8BlockSize}); adapter/optimizer/cursor restored; base/reference quantization changes, so loss need not be identical.");
                step = saved.Step; epoch = saved.Epoch; issued = saved.Issued; trained = saved.Trained;
                restored = saved.Ready;
                if (generate is null) output.WriteLine($"resumed DPO checkpoint = {adapterPath}, epoch={epoch}, global step={step}, issued={issued}, trained={trained}, queued={restored.Length}");
            }
            if (cuda) model.to(TensorDevice.Cuda);
            if (generate is not null)
            {
                model.eval();
                using var noGrad = AutogradContext.NoGrad();
                GenerateCommand.StreamGeneration(model, tokenizer, generate, config.MaxNewTokens,
                    config.Temperature, config.TopK, new Random(lora.Seed), output);
                return 0;
            }
            using var metrics = new DpoMetricReporter(graphPath, config.MaxSteps, config.LossGraphEverySteps, resume, step);
            foreach (string path in metrics.ArchivedPaths)
                output.WriteLine($"previous DPO history preserved = {path}");
            if (metrics.ArchivedPaths.Count > 0)
                output.WriteLine("DPO starts a new adapter/run; old loss history is archived, not resumed.");
            output.WriteLine($"loss graph = {metrics.HtmlPath}, every {config.LossGraphEverySteps} step(s)");
            output.WriteLine($"metrics = {metrics.SidecarPath}");
            if (step >= config.MaxSteps)
            {
                output.WriteLine($"DPO already reached maxSteps={config.MaxSteps} (global step={step}); increase maxSteps to continue.");
                return 0;
            }
            ((IOptimizer)optimizer).prepare();
            output.WriteLine($"DPO DRN: {config.Device}:{config.DeviceIndex}, {lora.PrecisionMode}" +
                (mode == TensorPrecisionMode.Mix8_32 ? $" (block {lora.Bfp8BlockSize}, FP32 master/gradient/AdamW statistics)" : "") +
                $", batch={config.BatchSize}, accumulation={config.GradientAccumulationSteps}, slots={config.GenerationSlots}, queue={config.CompletedQueueCapacity}, beta={config.DpoBeta}");
            output.WriteLine($"dataset = {config.Dataset}, path = {dataPath}, text column = {config.TextColumn}");
            output.WriteLine($"training device = {config.Device}:{config.DeviceIndex}; generation devices = [{string.Join(',', Enumerable.Range(0, config.GenerationSlots).Select(config.GenerationDeviceForWorker).Distinct())}], worker devices = [{string.Join(',', Enumerable.Range(0, config.GenerationSlots).Select(config.GenerationDeviceForWorker))}]");
            output.WriteLine("generation = parallel dedicated workers + independent CUDA streams/model replicas; training = single-device optimizer (no gradient all-reduce), pipelined accumulation; reference = frozen base without adapters; base linear dW/db omitted, embedding/normalization gradients retained");
            AdapterSnapshot snapshot = new(step, adapters.state_dict());
            for (; epoch <= config.Epochs && step < config.MaxSteps; epoch++, issued = 0, trained = 0)
            {
                using var lines = DpoDataSource.Read(dataPath, config.Dataset, config.TextColumn, cancellation.Token).GetEnumerator();
                for (long i = 0; i < issued; i++)
                    if (!lines.MoveNext()) throw new InvalidDataException("DPO resume cursor exceeds data.");
                GenerationRequest? Create()
                {
                    if (!lines.MoveNext()) return null;
                    int[] tokens = tokenizer.Encode(lines.Current, addBos: true).Append(BpeTokenizer.EosTokenId)
                        .Take(config.ContextLength + 1).ToArray();
                    var random = new Random(unchecked(lora.Seed + epoch * 1000003 + (int)issued * 9176));
                    int low = Math.Clamp((int)Math.Ceiling(tokens.Length * config.CutMinimum), 1, tokens.Length - 1);
                    int high = Math.Clamp((int)Math.Floor(tokens.Length * config.CutMaximum), low, tokens.Length - 1);
                    int cut = random.Next(low, high + 1);
                    return new GenerationRequest(new Pair(issued++, tokens[..cut], tokens[cut..], []), random, Volatile.Read(ref snapshot));
                }
                using var queue = new ParallelGenerationQueue<GenerationRequest, Pair>(config.GenerationSlots,
                    config.CompletedQueueCapacity,
                    index => new GenerationWorker(index, modelPath, baseHash, lora,
                        config with { DeviceIndex = config.GenerationDeviceForWorker(index), GenerationDeviceIndices = null }, output),
                    Create, restored, cancellation.Token);
                restored = [];
                queue.Resume();
                int lastSavedStep = -1;
                void Save(bool resume)
                {
                    if (lastSavedStep == step) { if (resume) queue.Resume(); return; }
                    Pair[] pending = queue.PauseAndSnapshot();
                    Directory.CreateDirectory(Path.GetDirectoryName(adapterPath)!);
                    string temporary = adapterPath + "." + Guid.NewGuid().ToString("N") + ".tmp";
                    torch.save(new Checkpoint(1, baseHash, tokenizerHash, dataHash, contract, step, epoch,
                        issued, trained, pending, adapters.state_dict(), optimizer.state_dict()), temporary);
                    File.Move(temporary, adapterPath, overwrite: true);
                    lastSavedStep = step;
                    metrics.Flush();
                    output.WriteLine($"DPO checkpoint step={step}, issued={issued}, trained={trained}, queued={pending.Length}, peak generation workers={queue.PeakActive}");
                    if (resume) queue.Resume();
                }
                while (step < config.MaxSteps)
                {
                    var stepWatch = Stopwatch.StartNew();
                    var watch = new Stopwatch();
                    double generationMs = 0;
                    int pairs = 0;
                    int plannedPairs = checked(config.BatchSize * config.GradientAccumulationSteps);
                    double value = 0;
                    Pair? samplePair = null;
                    model.ZeroGrad();
                    for (int a = 0; a < config.GradientAccumulationSteps; a++)
                    {
                        var wait = Stopwatch.StartNew();
                        Pair[] batch = queue.Take(config.BatchSize);
                        generationMs += wait.Elapsed.TotalMilliseconds;
                        if (batch.Length == 0) break;
                        samplePair ??= batch[0];
                        pairs += batch.Length;
                        watch.Start();
                        cancellation.Token.ThrowIfCancellationRequested();
                        var packed = Pack(batch);
                        float[] reference;
                        adapters.SetEnabled(false);
                        model.eval();
                        try
                        {
                            using var noGrad = AutogradContext.NoGrad();
                            using var inference = CudaInferenceScope.Begin();
                            reference = Scores(model, packed).Select(t => t.item()).ToArray();
                        }
                        finally { adapters.SetEnabled(true); model.train(); }
                        Tensor[] ce = Scores(model, packed);
                        Tensor loss = Tensor.DpoLoss(ce, packed.Counts, reference, config.DpoBeta);
                        float weight = (float)batch.Length / plannedPairs;
                        value += loss.item() * weight;
                        loss.BackwardAndRelease([weight]);
                        watch.Stop();
                    }
                    if (pairs == 0) break;
                    watch.Start();
                    // The dataset tail may contain fewer pairs than planned. Restore a pair mean before clipping.
                    float correction = (float)plannedPairs / pairs;
                    foreach (var parameter in adapters.parameters()) parameter.T.ScaleDpoGradient(correction);
                    value *= correction;
                    float norm = nn.utils.clip_grad_norm_(adapters.parameters(), config.GradientClip);
                    if (!double.IsFinite(value) || !float.IsFinite(norm)) throw new InvalidOperationException("Non-finite DPO loss/gradient; no update committed.");
                    optimizer.step();
                    step++; trained += pairs;
                    metrics.Append(step, epoch, value);
                    Volatile.Write(ref snapshot, new AdapterSnapshot(step, adapters.state_dict()));
                    output.WriteLine($"DPO epoch={epoch}, step={step}, pairs={pairs}, loss={value:F6}, grad norm={norm:G6}, generation={generationMs:F1} ms, train={watch.Elapsed.TotalMilliseconds:F1} ms, step={stepWatch.Elapsed.TotalMilliseconds:F1} ms, queued={queue.Count}, active={queue.ActiveCount}");
                    output.Flush();
                    bool sampleDue = ShouldGenerateSample(step, config.SampleEverySteps);
                    if (step % config.SaveEverySteps == 0) Save(resume: !sampleDue && step < config.MaxSteps);
                    if (sampleDue)
                    {
                        // Drain active inference before reusing the training model; keep queued pairs intact.
                        queue.PauseAndSnapshot();
                        WriteSample(model, tokenizer, samplePair!, config, step, lora.Seed, output, cancellation.Token);
                        if (step < config.MaxSteps) queue.Resume();
                    }
                }
                Save(resume: false);
                if (step >= config.MaxSteps) break;
                if (issued == 0) throw new InvalidDataException("DPO dataset has no examples.");
            }
            return 0;
        }
        catch (OperationCanceledException) when (cancellation.IsCancellationRequested)
        {
            output.WriteLine("DPO cancelled; resume from the last completed checkpoint (uncommitted work is replayed).");
            return 130;
        }
        finally { Console.CancelKeyPress -= cancel; optimizer.DisposeCudaResources(); }
    }

    internal static bool CompatibleContract(string saved, string current, bool allowPrecisionConversion = false)
        => ContractMismatches(saved, current, allowPrecisionConversion).Count == 0;

    internal static IReadOnlyList<string> ContractMismatches(string saved, string current, bool allowPrecisionConversion = false)
    {
        // Placement, buffering and microbatch partitioning may change; the effective update batch must not.
        var left = JsonNode.Parse(saved)!.AsObject();
        var right = JsonNode.Parse(current)!.AsObject();
        var differences = new List<string>();
        long Effective(JsonObject obj) => checked(
            (long)obj["batchSize"]!.GetValue<int>() * obj["gradientAccumulationSteps"]!.GetValue<int>());
        long savedBatch = Effective(left), currentBatch = Effective(right);
        if (savedBatch != currentBatch)
        {
            string Batch(JsonObject obj) => $"batchSize={obj["batchSize"]}, gradientAccumulationSteps={obj["gradientAccumulationSteps"]}";
            string difference = $"effective batch: saved={savedBatch} ({Batch(left)}), current={currentBatch} ({Batch(right)})";
            int currentMicrobatch = right["batchSize"]!.GetValue<int>();
            if (currentMicrobatch > 0 && savedBatch > 0 && savedBatch % currentMicrobatch == 0 && savedBatch / currentMicrobatch <= int.MaxValue)
                difference += $". To preserve {savedBatch} with batchSize={currentMicrobatch}, set gradientAccumulationSteps={savedBatch / currentMicrobatch}";
            differences.Add(difference);
        }
        foreach (string key in new[] { "batchSize", "gradientAccumulationSteps", "generationSlots", "completedQueueCapacity" })
        { left.Remove(key); right.Remove(key); }
        static bool Supported(string? mode) => mode is "float32" or "mix16_32" or "mix8_32";
        static bool HasSupportedPrecision(JsonObject contract) => contract["lora"] is not JsonObject lora ||
            Supported(lora["precisionMode"]?.GetValue<string>());
        void NormalizePrecision(JsonObject contract)
        {
            if (contract["lora"] is not JsonObject lora) return;
            string? mode = lora["precisionMode"]?.GetValue<string>();
            // Old non-BFP8 checkpoints predate this field; block size has no meaning for them.
            if (mode != "mix8_32" || allowPrecisionConversion) lora.Remove("bfp8BlockSize");
            else if (!lora.ContainsKey("bfp8BlockSize")) lora["bfp8BlockSize"] = 32;
            if (allowPrecisionConversion) lora.Remove("precisionMode");
        }
        bool savedPrecisionSupported = HasSupportedPrecision(left), currentPrecisionSupported = HasSupportedPrecision(right);
        if (!savedPrecisionSupported || !currentPrecisionSupported)
            differences.Add($"unsupported lora.precisionMode: saved={left["lora"]?["precisionMode"]?.ToJsonString() ?? "<missing>"}, current={right["lora"]?["precisionMode"]?.ToJsonString() ?? "<missing>"}");
        else
        { NormalizePrecision(left); NormalizePrecision(right); }
        void Compare(JsonObject a, JsonObject b, string prefix)
        {
            foreach (string key in a.Select(p => p.Key).Union(b.Select(p => p.Key)).Order(StringComparer.Ordinal))
            {
                string path = prefix + key;
                bool hasSaved = a.TryGetPropertyValue(key, out JsonNode? x), hasCurrent = b.TryGetPropertyValue(key, out JsonNode? y);
                if (hasSaved && hasCurrent && x is JsonObject xObject && y is JsonObject yObject)
                { Compare(xObject, yObject, path + "."); continue; }
                if (hasSaved && hasCurrent && JsonNode.DeepEquals(x, y)) continue;
                string difference = $"{path}: saved={(hasSaved ? x?.ToJsonString() ?? "null" : "<missing>")}, current={(hasCurrent ? y?.ToJsonString() ?? "null" : "<missing>")}";
                if (!allowPrecisionConversion && savedPrecisionSupported && currentPrecisionSupported &&
                    path is "lora.precisionMode" or "lora.bfp8BlockSize")
                    difference += ". Precision conversion requires allowPrecisionConversionOnResume=true";
                differences.Add(difference);
            }
        }
        Compare(left, right, "");
        return differences;
    }

    internal static bool ShouldGenerateSample(long step, int everySteps) =>
        step > 0 && everySteps > 0 && step % everySteps == 0;

    internal static void WriteSample(ForgetMemoryDRNGpt model, BpeTokenizer tokenizer, Pair pair,
        LoraTrainingConfiguration config, int step, int seed, TextWriter output, CancellationToken cancellationToken)
    {
        bool wasTraining = model.IsTraining;
        model.eval();
        try
        {
            output.WriteLine($"DPO sample at global step {step} (updated policy)");
            output.WriteLine("[prefix]");
            output.WriteLine(tokenizer.Decode(pair.Prefix));
            output.WriteLine("[chosen]");
            output.WriteLine(tokenizer.Decode(pair.Chosen));
            output.WriteLine("[model continuation]");
            output.Flush();
            var decoder = tokenizer.CreateIncrementalDecoder();
            void WriteToken(int token)
            {
                if (token == BpeTokenizer.EosTokenId) return;
                output.Write(decoder.Append(token));
                output.Flush();
            }
            // Independent RNG and recurrent state: do not consume the dataset/worker RNG or retain a graph.
            using var job = new GenerationJob(model, pair, config,
                new Random(unchecked(seed ^ step * 9176)), Math.Min(config.MaxNewTokens, pair.Chosen.Length), WriteToken);
            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (job.Advance(out _)) break;
            }
            output.WriteLine(decoder.Flush());
            output.Flush();
        }
        finally { if (wasTraining) model.train(); }
    }

    internal sealed record Packed(int[] Input, int[][] Labels, int[] Counts, int Length);
    internal static Packed Pack(Pair[] pairs)
    {
        int length = pairs.Max(p => p.Prefix.Length + Math.Max(p.Chosen.Length, p.Rejected.Length) - 1);
        int[] input = new int[checked(2 * pairs.Length * length)];
        int[][] labels = new int[2 * pairs.Length][];
        int[] counts = new int[labels.Length];
        for (int i = 0; i < labels.Length; i++)
        {
            Pair p = pairs[i / 2];
            int[] completion = i % 2 == 0 ? p.Chosen : p.Rejected;
            if (p.Prefix.Length == 0 || completion.Length == 0) throw new InvalidDataException("Empty DPO prefix/completion.");
            int[] all = p.Prefix.Concat(completion).ToArray();
            labels[i] = Enumerable.Repeat(-1, length).ToArray();
            counts[i] = completion.Length;
            for (int t = 0; t + 1 < all.Length; t++)
            {
                input[i * length + t] = all[t];
                if (t + 1 >= p.Prefix.Length) labels[i][t] = all[t + 1];
            }
        }
        return new Packed(input, labels, counts, length);
    }
    internal static Tensor[] Scores(ForgetMemoryDRNGpt model, Packed batch)
    {
        return model.ForwardCompletionScores(batch.Input, batch.Labels, batch.Length);
    }
    private static string Hash(string path) { using var file = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(file)); }

    private sealed class GenerationWorker : ParallelGenerationQueue<GenerationRequest, Pair>.IWorker
    {
        private readonly ForgetMemoryDRNGpt generator;
        private readonly LoraAdapterSet adapters;
        private readonly LoraTrainingConfiguration config;
        private ExecutionSession? session;
        private IDisposable? scope, policy;
        private int snapshotStep = -1;
        internal GenerationWorker(int index, string modelPath, string expectedHash, LoraConfiguration lora,
            LoraTrainingConfiguration config, TextWriter output)
        {
            this.config = config;
            using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu, 0)))
            {
                (generator, string hash) = WikiLanguageModelCommand.LoadLoraBase(modelPath,
                    TensorPrecisionModeNames.Parse(lora.PrecisionMode), lora.Seed, lora.Bfp8BlockSize);
                if (hash != expectedHash) throw new IOException("Base checkpoint changed while starting generation workers.");
                adapters = generator.AttachLora(lora.Rank, lora.Alpha, lora.Targets, lora.Seed);
                generator.eval();
            }
            try
            {
                bool cuda = config.Device == "cuda";
                var lane = cuda ? CudaExecutionLaneFactory.Create(config.DeviceIndex) : null;
                session = new ExecutionSession(new ExecutionOptions {
                    Device = cuda ? ExecutionDeviceKind.Cuda : ExecutionDeviceKind.Cpu,
                    CudaDevices = new DeviceSet([config.DeviceIndex]), Precision = PrecisionPolicy.Parse(lora.PrecisionMode)
                }, lane is null ? [] : [lane]);
                scope = session.Enter();
                policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Current with { DisableCudaGraphs = true });
                lane?.ActivateComputeStream();
                if (cuda) generator.to(TensorDevice.Cuda);
                lock (output) output.WriteLine($"generation worker={index}, thread={Environment.CurrentManagedThreadId}, device={config.Device}:{config.DeviceIndex}, stream=0x{lane?.ComputeStreamHandle ?? 0:X}");
            }
            catch { Dispose(); throw; }
        }
        public Pair Generate(GenerationRequest request, CancellationToken token)
        {
            token.ThrowIfCancellationRequested();
            if (snapshotStep != request.Snapshot.Step)
            {
                // Only this worker mutates its private replica, between complete generations.
                adapters.load_state_dict(request.Snapshot.State);
                snapshotStep = request.Snapshot.Step;
            }
            using var job = new GenerationJob(generator, request.Pair, config, request.Random);
            while (true)
            {
                token.ThrowIfCancellationRequested();
                if (job.Advance(out Pair result)) return result;
            }
        }
        public void Dispose()
        {
            try { policy?.Dispose(); }
            finally { try { scope?.Dispose(); } finally { session?.Dispose(); } }
        }
    }

    private sealed class GenerationJob : CompletedGenerationQueue<Pair>.IJob
    {
        private readonly ForgetMemoryDRNGpt model;
        private readonly ForgetMemoryV2RecurrentState state;
        private readonly Pair pair;
        private readonly LoraTrainingConfiguration config;
        private readonly Random random;
        private readonly List<int> rejected = [];
        private readonly int maximumTokens;
        private readonly Action<int>? onToken;
        private int prefilled;
        internal GenerationJob(ForgetMemoryDRNGpt model, Pair pair, LoraTrainingConfiguration config, Random random,
            int? maximumTokens = null, Action<int>? onToken = null)
        {
            this.model = model; this.pair = pair; this.config = config; this.random = random;
            this.maximumTokens = maximumTokens ?? pair.Chosen.Length; this.onToken = onToken;
            state = model.CreateRecurrentState();
        }
        public bool Advance(out Pair result)
        {
            result = null!;
            using var noGrad = AutogradContext.NoGrad();
            using var inference = CudaInferenceScope.Begin();
            int[] input;
            if (prefilled < pair.Prefix.Length)
            {
                int count = Math.Min(ForgetMemoryV2Gpt.GenerationPrefillChunkTokens, pair.Prefix.Length - prefilled);
                input = pair.Prefix[prefilled..(prefilled + count)];
                prefilled += count;
            }
            else input = [rejected[^1]];
            Tensor logits = model.AdvanceToLastLogits(input, state);
            if (prefilled < pair.Prefix.Length) return false;
            int token = LanguageModel.SampleLogits(logits, 0, model.VocabularySize, config.Temperature, config.TopK, random);
            rejected.Add(token);
            onToken?.Invoke(token);
            if (token != BpeTokenizer.EosTokenId && rejected.Count < maximumTokens) return false;
            result = pair with { Rejected = rejected.ToArray() };
            return true;
        }
        public void Dispose() => state.Dispose();
    }
}
