using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using NNtrain.Runtime.Execution;
using NNtrain.Cuda.Execution;

namespace NNtrain;

internal static class LoraCommand
{
    internal sealed record AdapterCheckpoint(int Version, string BaseFingerprint, string TokenizerHash,
        string DataHash, LoraConfiguration Lora, string TrainingContract, int Step, int Epoch, long Documents,
        ModuleState Adapter, OptimizerStateDictionary Optimizer);
    internal sealed record Example(string Prompt, string Response);

    internal static int Run(string[] args, TextWriter output, TextWriter error)
    {
        try
        {
            string? modelPath = null, generate = null;
            string configPath = "traning-lora.json";
            var seen = new HashSet<string>();
            for (int i = 1; i < args.Length; i += 2)
            {
                if (i + 1 >= args.Length || !seen.Add(args[i])) throw new ArgumentException("Missing or duplicate LoRA option.");
                switch (args[i])
                {
                    case "--model": modelPath = Path.GetFullPath(args[i + 1]); break;
                    case "--config": configPath = args[i + 1]; break;
                    case "--generate": generate = args[i + 1]; break;
                    default: throw new ArgumentException($"Unknown LoRA option: {args[i]}");
                }
            }
            if (modelPath is null) throw new ArgumentException("Usage: lora --model <DRN-checkpoint.json> [--config traning-lora.json] [--generate <prompt>]");
            configPath = Path.GetFullPath(configPath);
            var config = LoraConfiguration.Load<LoraTrainingConfiguration>(configPath);
            config.Validate();
            if (config.Objective == "dpo") return DpoCommand.Run(modelPath, configPath, config, generate, output);
            string Resolve(string path) => Path.GetFullPath(path, Path.GetDirectoryName(configPath)!);
            var lora = LoraConfiguration.Load<LoraConfiguration>(Resolve(config.LoraConfig)).WithTrainingOverrides(config);
            lora.Validate();
            string tokenizerPath = Resolve(config.TokenizerPath), adapterPath = Resolve(config.AdapterPath), dataPath = Resolve(config.DataPath);
            if (Path.GetFullPath(adapterPath).Equals(modelPath, StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("Adapter output must not overwrite the base checkpoint.");
            bool resume = LoraResumePolicy.Resolve(config, adapterPath, generate is not null, output);
            if (generate is null && !File.Exists(dataPath))
                throw new FileNotFoundException("LoRA SFT data missing. Supply JSONL records with prompt and response fields.", dataPath);
            if (resume)
                if (!File.Exists(adapterPath)) throw new FileNotFoundException("LoRA adapter checkpoint not found.", adapterPath);
            var tokenizer = tokenizers.load_bpe(tokenizerPath);
            string tokenizerHash = Hash(tokenizerPath);
            string dataHash = generate is null ? Hash(dataPath) : "";
            TensorPrecisionMode mode = TensorPrecisionModeNames.Parse(lora.PrecisionMode);
            ForgetMemoryDRNGpt model;
            string fingerprint;
            LoraAdapterSet adapters;
            using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu, 0)))
            {
                (model, fingerprint) = WikiLanguageModelCommand.LoadLoraBase(modelPath, mode, lora.Seed, lora.Bfp8BlockSize);
                if (tokenizer.VocabularySize != model.VocabularySize) throw new InvalidDataException("Tokenizer vocabulary does not match base model; no tokenizer retraining is allowed.");
                if (config.ContextLength > model.ContextLength) throw new ArgumentException("LoRA context exceeds base context.");
                adapters = model.AttachLora(lora.Rank, lora.Alpha, lora.Targets, lora.Seed);
            }
            bool cuda = config.Device == "cuda";
            using var session = new ExecutionSession(new ExecutionOptions {
                Device = cuda ? ExecutionDeviceKind.Cuda : ExecutionDeviceKind.Cpu,
                CudaDevices = new DeviceSet([config.DeviceIndex]),
                Precision = PrecisionPolicy.Parse(lora.PrecisionMode),
            }, cuda ? [CudaExecutionLaneFactory.Create(config.DeviceIndex)] : []);
            using var scope = session.Enter();
            using var policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Current with {
                DisableCudaGraphs = true, EnableBlockBfp8OptimizerState = false });
            using var optimizer = new LoraOptimizerLifetime(adapters, config);
            int step = 0, epoch = 1;
            long documents = 0;
            string contract = JsonSerializer.Serialize(new { config.ContextLength, config.BatchSize,
                config.GradientAccumulationSteps, config.PromptPrefix, config.ResponsePrefix,
                config.LearningRate, config.WeightDecay, config.GradientClip }, LoraConfiguration.Json);
            if (resume)
            {
                var saved = torch.load<AdapterCheckpoint>(adapterPath);
                if (saved.Version != 1 || saved.BaseFingerprint != fingerprint || saved.TokenizerHash != tokenizerHash ||
                    JsonSerializer.Serialize(saved.Lora, LoraConfiguration.Json) != JsonSerializer.Serialize(lora, LoraConfiguration.Json) ||
                    saved.TrainingContract != contract || (generate is null && saved.DataHash != dataHash))
                    throw new InvalidDataException("LoRA adapter base/tokenizer/config/data identity mismatch.");
                adapters.load_state_dict(saved.Adapter);
                if (generate is null) optimizer.Value.load_state_dict(saved.Optimizer);
                step = saved.Step; epoch = saved.Epoch; documents = saved.Documents;
            }
            if (cuda) model.to(TensorDevice.Cuda);
            output.WriteLine($"LoRA DRN: rank={lora.Rank}, alpha={lora.Alpha}, trainable={adapters.parameters().Sum(p => (long)p.T.Numel):N0}, precision={lora.PrecisionMode}, device={config.Device}:{config.DeviceIndex}");
            output.WriteLine("base weights excluded from optimizer; adapter-only checkpoint; base gradients are still computed in this first implementation");
            if (generate is not null)
            {
                model.eval();
                using var noGrad = AutogradContext.NoGrad();
                GenerateCommand.StreamGeneration(model, tokenizer, config.PromptPrefix + generate + config.ResponsePrefix,
                    config.MaxNewTokens, config.Temperature, config.TopK, new Random(lora.Seed), output);
                output.WriteLine();
                return 0;
            }
            ((IOptimizer)optimizer.Value).prepare();
            model.train();
            void Save()
            {
                string directory = Path.GetDirectoryName(adapterPath)!;
                Directory.CreateDirectory(directory);
                string temporary = adapterPath + "." + Guid.NewGuid().ToString("N") + ".tmp";
                torch.save(new AdapterCheckpoint(1, fingerprint, tokenizerHash, dataHash, lora, contract,
                    step, epoch, documents, adapters.state_dict(), optimizer.Value.state_dict()), temporary);
                File.Move(temporary, adapterPath, overwrite: true);
                output.WriteLine($"LoRA checkpoint = {adapterPath}, step {step}");
            }
            for (; epoch <= config.Epochs && step < config.MaxSteps; epoch++, documents = 0)
            {
                var pending = new List<Example>();
                long read = 0;
                int capacity = checked(config.BatchSize * config.GradientAccumulationSteps);
                foreach (string line in File.ReadLines(dataPath))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    if (read++ < documents) continue;
                    var sample = JsonSerializer.Deserialize<Example>(line, LoraConfiguration.Json)
                        ?? throw new InvalidDataException("Empty LoRA example.");
                    if (sample.Prompt is null || string.IsNullOrWhiteSpace(sample.Response)) throw new InvalidDataException("LoRA examples require prompt and non-empty response.");
                    pending.Add(sample);
                    if (pending.Count == capacity)
                    {
                        Update(pending); pending.Clear();
                        if (step >= config.MaxSteps) break;
                    }
                }
                if (read < documents) throw new InvalidDataException("LoRA resume cursor exceeds dataset.");
                if (pending.Count > 0 && step < config.MaxSteps) Update(pending);
                if (read == 0) throw new InvalidDataException("LoRA dataset has no examples.");
                Save();
                if (step >= config.MaxSteps) break;

                void Update(List<Example> examples)
                {
                    var watch = Stopwatch.StartNew();
                    model.ZeroGrad();
                    var batches = examples.Chunk(config.BatchSize).Select(part => BuildBatch(part, tokenizer, config)).ToArray();
                    long valid = batches.Sum(b => (long)b.Target.Count(t => t >= 0));
                    double lossValue = 0;
                    foreach (var batch in batches)
                    {
                        float weight = (float)batch.Target.Count(t => t >= 0) / valid;
                        Tensor loss = model.ForwardLoss(batch.Input, batch.Target, batch.Batch, config.ContextLength);
                        lossValue += loss.item() * weight;
                        loss.BackwardAndRelease([weight]);
                    }
                    float norm = nn.utils.clip_grad_norm_(adapters.parameters(), config.GradientClip);
                    if (!double.IsFinite(lossValue) || !float.IsFinite(norm)) throw new InvalidOperationException("Non-finite LoRA loss/gradient; update not committed.");
                    optimizer.Value.step();
                    step++; documents += examples.Count;
                    output.WriteLine($"LoRA epoch {epoch}, step {step}, documents {documents}, loss={lossValue:F6}, grad norm={norm:G6}, elapsed={watch.Elapsed.TotalMilliseconds:F1} ms");
                    output.Flush();
                    if (step % config.SaveEverySteps == 0) Save();
                }
            }
            return 0;
        }
        catch (Exception exception) when (exception is not OutOfMemoryException and not StackOverflowException)
        { error.WriteLine($"Error: {exception.Message}"); error.WriteLine(exception.StackTrace); return 2; }
    }

    internal static (int[] Input, int[] Target, int Batch) BuildBatch(Example[] examples, BpeTokenizer tokenizer, LoraTrainingConfiguration config)
    {
        int length = config.ContextLength;
        var input = new int[checked(examples.Length * length)];
        var target = Enumerable.Repeat(-1, input.Length).ToArray();
        for (int b = 0; b < examples.Length; b++)
        {
            int[] prompt = tokenizer.Encode(config.PromptPrefix + examples[b].Prompt + config.ResponsePrefix, addBos: true);
            if (prompt.Length > length) throw new InvalidDataException("LoRA prompt exceeds context; shorten it explicitly.");
            int[] all = prompt.Concat(tokenizer.Encode(examples[b].Response)).Append(BpeTokenizer.EosTokenId).ToArray();
            for (int t = 0; t < length && t + 1 < all.Length; t++)
            {
                input[b * length + t] = all[t];
                if (t + 1 >= prompt.Length) target[b * length + t] = all[t + 1];
            }
        }
        return (input, target, examples.Length);
    }
    private static string Hash(string path) { using var file = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(file)); }
    private sealed class LoraOptimizerLifetime : IDisposable
    {
        internal AdamW Value { get; }
        internal LoraOptimizerLifetime(LoraAdapterSet adapters, LoraTrainingConfiguration config)
            => Value = new AdamW(adapters.parameters(), new AdamWOptions { LearningRate = config.LearningRate, WeightDecay = config.WeightDecay });
        public void Dispose() => Value.DisposeCudaResources();
    }
}
