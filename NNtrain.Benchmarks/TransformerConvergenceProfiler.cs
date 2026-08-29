using System.Diagnostics;
using System.Text.Json;

namespace NNtrain.Benchmarks;

/// <summary>
/// Short, deterministic convergence probe using the production checkpoint,
/// optimizer moments, tokenizer, and real Wikipedia tokens. Each invocation
/// runs one candidate in a fresh process so every A/B arm starts from the same
/// weights and optimizer state without retaining another model on the GPUs.
/// </summary>
internal static class TransformerConvergenceProfiler
{
    private const int CorpusShuffleSeedSalt = 0x31B7;
    private const double StableUntilProgress = 0.8d;
    private const int ProbeBatchSize = 4;

    internal static void Run(
        string configurationPath,
        int steps,
        float matrixLearningRate,
        float auxiliaryLearningRate,
        string schedule,
        bool forceFullNewtonSchulz)
    {
        if (steps <= 0)
            throw new ArgumentOutOfRangeException(nameof(steps));
        if (!float.IsFinite(matrixLearningRate) || matrixLearningRate <= 0f)
            throw new ArgumentOutOfRangeException(nameof(matrixLearningRate));
        if (!float.IsFinite(auxiliaryLearningRate)
            || auxiliaryLearningRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(auxiliaryLearningRate));
        }
        bool stableThenCosine = schedule switch
        {
            "pure-cosine" => false,
            "stable-then-cosine" => true,
            _ => throw new ArgumentException(
                "Schedule must be pure-cosine or stable-then-cosine.",
                nameof(schedule)),
        };

        string path = Path.GetFullPath(configurationPath);
        string directory = Path.GetDirectoryName(path)!;
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;
        JsonElement optimizerConfiguration = root.GetProperty("optimization")
            .GetProperty("optimizer");
        TensorPrecisionMode precisionMode = PrecisionModeConfiguration.Read(root);
        TensorDType dtype = precisionMode.ToStorageDType();
        int bfp8BlockSize = root.TryGetProperty(
            "bfp8_block_size", out JsonElement configuredBlockSize)
            ? configuredBlockSize.GetInt32()
            : Bfp8QuantizationDescriptor.DefaultBlockSize;
        int batch = root.GetProperty("batchSize").GetInt32();
        int sequence = root.GetProperty("contextLength").GetInt32();
        int seed = root.GetProperty("seed").GetInt32();
        int[] devices = root.GetProperty("deviceIndices")
            .EnumerateArray()
            .Select(element => element.GetInt32())
            .ToArray();
        string dataPath = Path.GetFullPath(
            root.GetProperty("dataPath").GetString()!, directory);
        string tokenizerPath = Path.GetFullPath(
            root.GetProperty("tokenizerPath").GetString()!, directory);
        string textColumn = root.TryGetProperty("textColumn", out JsonElement text)
            ? text.GetString() ?? "text"
            : "text";
        int maxDocumentTokens = root.TryGetProperty(
            "maxDocumentTokens", out JsonElement maxDocument)
            ? maxDocument.GetInt32()
            : 0;
        string checkpointPath = ResolveCheckpointPath(root, directory, path);
        using JsonDocument checkpoint = JsonDocument.Parse(
            File.ReadAllText(checkpointPath));
        JsonElement checkpointRoot = checkpoint.RootElement;
        int artifactSlot = checkpointRoot.GetProperty("ArtifactSlot").GetInt32();
        long globalStep = checkpointRoot.GetProperty("GlobalStep").GetInt64();
        double initialProgress = checkpointRoot.GetProperty("Scheduler")
            .GetProperty("LastProgress").GetDouble();
        string artifactStem = Path.Combine(
            Path.GetDirectoryName(checkpointPath)!,
            Path.GetFileNameWithoutExtension(checkpointPath));
        string modelArtifact =
            $"{artifactStem}.current.{artifactSlot}.safetensors";
        string nekoArtifact =
            $"{artifactStem}.optimizer.{artifactSlot}.0.bin";
        string adamArtifact =
            $"{artifactStem}.optimizer.{artifactSlot}.1.bin";

        BpeTokenizer tokenizer = BpeTokenizer.Load(tokenizerPath);
        ProbeBatch[] batches = LoadRealBatches(
            dataPath,
            textColumn,
            tokenizer,
            maxDocumentTokens,
            batch,
            sequence,
            steps + 1,
            CombineSeed(seed, 1, CorpusShuffleSeedSalt));
        ulong dataHash = HashBatches(batches);

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = devices;
            Tensor.SimdEnabled = true;
            CudaDataParallel.ConfigureAdaptiveSharding(
                ReadAdaptiveSharding(root));

            bool bfp8Mode = precisionMode is TensorPrecisionMode.Bfp8
                or TensorPrecisionMode.Mix8_32;
            var model = new GptRinWikiJp(
                tokenizer.VocabularySize,
                sequence,
                root.GetProperty("modelWidth").GetInt32(),
                root.GetProperty("heads").GetInt32(),
                root.GetProperty("hiddenSize").GetInt32(),
                root.GetProperty("layers").GetInt32(),
                new Random(seed),
                root.GetProperty("initializationScale").GetSingle(),
                root.GetProperty("dropout").GetSingle(),
                bfp8Mode ? TensorDType.Float32 : dtype,
                root.GetProperty("tieWordEmbeddings").GetBoolean());
            if (bfp8Mode)
                model.to(precisionMode, bfp8BlockSize);
            else
                model.SetPrecisionMode(precisionMode);
            ModuleState? state = safetensors.torch.load_file(modelArtifact);
            model.load_state_dict(state);
            state = null;

            var nekoMuon = new NekoMuon(
                model.HiddenWeightParameters,
                new NekoMuonOptions
                {
                    LearningRate = matrixLearningRate,
                    WeightDecay = optimizerConfiguration
                        .GetProperty("weightDecay").GetSingle(),
                    MaxNewtonSchulzSteps = 5,
                    NewtonSchulzInterval = optimizerConfiguration
                        .GetProperty("nekoMuonNewtonSchulzInterval").GetInt32(),
                });
            var adamW = new AdamW(
                model.AuxiliaryParameters,
                new AdamWOptions
                {
                    LearningRate = auxiliaryLearningRate,
                    Beta1 = 0.9f,
                    Beta2 = 0.95f,
                    Epsilon = 1e-8f,
                    WeightDecay = optimizerConfiguration
                        .GetProperty("weightDecay").GetSingle(),
                    UseBFloat16FirstMoment =
                        precisionMode == TensorPrecisionMode.BFloat16,
                    UseBFloat16SecondMoment =
                        precisionMode == TensorPrecisionMode.BFloat16,
                });
            using (var stream = File.OpenRead(nekoArtifact))
                OptimizerStateStream.LoadStateBinary(nekoMuon, stream);
            using (var stream = File.OpenRead(adamArtifact))
                OptimizerStateStream.LoadStateBinary(adamW, stream);
            nekoMuon.ForceFullNewtonSchulz = forceFullNewtonSchulz;
            var optimizer = new CompositeOptimizer(nekoMuon, adamW);

            Parameter hiddenProbe = model.HiddenWeightParameters[0];
            Parameter auxiliaryProbe = model.AuxiliaryParameters[0];
            float[] hiddenBefore = hiddenProbe.T.CaptureData(preferMaster: true);
            float[] auxiliaryBefore = auxiliaryProbe.T.CaptureData(
                preferMaster: true);
            model.to(TensorDevice.Cuda);
            model.train();
            GC.Collect(
                GC.MaxGeneration,
                GCCollectionMode.Aggressive,
                blocking: true,
                compacting: true);

            ProbeBatch probe = SliceFirstSequences(
                batches[^1],
                sequence,
                ProbeBatchSize);
            float initialProbeLoss = Evaluate(
                model,
                probe,
                ProbeBatchSize,
                sequence);
            Parameter[] parameters = model.Parameters().ToArray();
            var losses = new float[steps];
            var gradientNorms = new float[steps];
            var elapsed = new double[steps];
            var nsConfidences = new float[steps];
            var nsDepths = new float[steps];
            NekoMuonDiagnostics initialNekoDiagnostics =
                nekoMuon.GetDiagnostics();
            long estimatedTotalSteps = initialProgress > 0d
                ? Math.Max(globalStep + steps,
                    (long)Math.Round(globalStep / initialProgress))
                : Math.Max(globalStep + steps, 1L);
            for (int step = 0; step < steps; step++)
            {
                double progress = Math.Clamp(
                    initialProgress + (step + 1d) / estimatedTotalSteps,
                    0d,
                    1d);
                float factor = ScheduleFactor(progress, stableThenCosine);
                nekoMuon.SetLearningRate(matrixLearningRate * factor);
                adamW.SetLearningRate(auxiliaryLearningRate * factor);
                var timer = Stopwatch.StartNew();
                optimizer.zero_grad();
                losses[step] = CudaDataParallel.ForwardBackward(
                    model,
                    batches[step].Input,
                    batches[step].Target,
                    batch,
                    sequence);
                gradientNorms[step] = nn.utils.clip_grad_norm_(
                    parameters,
                    max_norm: 1f);
                optimizer.step();
                timer.Stop();
                elapsed[step] = timer.Elapsed.TotalMilliseconds;
                NekoMuonDiagnostics diagnostics = nekoMuon.GetDiagnostics();
                nsConfidences[step] = diagnostics.MeanConfidence;
                nsDepths[step] = diagnostics.MeanNewtonSchulzDepth;
                Console.WriteLine(
                    $"step {globalStep + step + 1:N0}: loss " +
                    $"{losses[step]:F6}, grad norm " +
                    $"{gradientNorms[step]:F4}" +
                    $"{(gradientNorms[step] > 1f ? " clipped" : string.Empty)}, " +
                    $"lr {nekoMuon.LearningRate:G}/{adamW.LearningRate:G}, " +
                    $"NS depth {nsDepths[step]:F3}, " +
                    $"{elapsed[step]:F2} ms, shards " +
                    string.Join('/',
                        CudaDataParallel.GetLastShardBatchSizes(model)));
            }
            float finalProbeLoss = Evaluate(
                model,
                probe,
                ProbeBatchSize,
                sequence);
            float[] hiddenAfter = hiddenProbe.T.CaptureData(preferMaster: true);
            float[] auxiliaryAfter = auxiliaryProbe.T.CaptureData(
                preferMaster: true);
            (double hiddenRms, double hiddenRelative) = UpdateRms(
                hiddenBefore,
                hiddenAfter);
            (double auxiliaryRms, double auxiliaryRelative) = UpdateRms(
                auxiliaryBefore,
                auxiliaryAfter);
            int lossWindow = Math.Min(10, losses.Length);
            double firstLossWindow = losses
                .Take(lossWindow)
                .Average();
            double lastLossWindow = losses
                .Skip(losses.Length - lossWindow)
                .Average();
            double[] orderedElapsed = elapsed.Order().ToArray();
            double medianElapsed = orderedElapsed.Length % 2 == 0
                ? (orderedElapsed[orderedElapsed.Length / 2 - 1]
                    + orderedElapsed[orderedElapsed.Length / 2]) / 2d
                : orderedElapsed[orderedElapsed.Length / 2];

            Console.WriteLine(
                $"convergence A/B: schedule={schedule}, NS=" +
                $"{(forceFullNewtonSchulz ? "full-5" : "adaptive")}, base LR=" +
                $"{matrixLearningRate:G}/{auxiliaryLearningRate:G}, " +
                $"checkpoint step={globalStep:N0}, progress=" +
                $"{initialProgress:P3}");
            Console.WriteLine(
                $"real-data hash=0x{dataHash:X16}, batches={steps}, " +
                $"batch={batch}, sequence={sequence}, tokens/step=" +
                $"{batch * sequence:N0}");
            Console.WriteLine(
                $"probe loss {initialProbeLoss:F6} -> {finalProbeLoss:F6} " +
                $"({finalProbeLoss - initialProbeLoss:+0.000000;-0.000000;0.000000}); " +
                $"train loss {losses[0]:F6} -> {losses[^1]:F6}");
            Console.WriteLine(
                $"train loss mean first/last {lossWindow}: " +
                $"{firstLossWindow:F6} -> {lastLossWindow:F6} " +
                $"({lastLossWindow - firstLossWindow:+0.000000;-0.000000;0.000000})");
            Console.WriteLine(
                $"clip frequency {gradientNorms.Count(value => value > 1f)}/" +
                $"{steps}, mean pre-clip norm {gradientNorms.Average():F4}; " +
                $"NS confidence {initialNekoDiagnostics.MeanConfidence:F3}" +
                $" -> mean {nsConfidences.Average():F3}, depth " +
                $"{initialNekoDiagnostics.MeanNewtonSchulzDepth:F3}" +
                $" -> mean {nsDepths.Average():F3}; step mean/p50 " +
                $"{elapsed.Average():F2}/{medianElapsed:F2} ms");
            Console.WriteLine(
                $"update RMS hidden={hiddenRms:E4} " +
                $"({hiddenRelative:P4} of weight RMS), auxiliary=" +
                $"{auxiliaryRms:E4} ({auxiliaryRelative:P4})");
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }

    private static ProbeBatch[] LoadRealBatches(
        string dataPath,
        string textColumn,
        BpeTokenizer tokenizer,
        int maxDocumentTokens,
        int batch,
        int sequence,
        int count,
        int shuffleSeed)
    {
        int targetsPerBatch = checked(batch * sequence);
        var tokenBuffer = new List<int>(checked(targetsPerBatch * count + 1024));
        var batches = new List<ProbeBatch>(count);
        IAsyncEnumerator<string> documents = WikiParquetCorpus.ReadTextsAsync(
            dataPath,
            textColumn,
            shuffleSeed: shuffleSeed).GetAsyncEnumerator();
        try
        {
            while (batches.Count < count
                && documents.MoveNextAsync().AsTask().GetAwaiter().GetResult())
            {
                string document = documents.Current;
                tokenBuffer.Add(BpeTokenizer.BosTokenId);
                int[] encoded = tokenizer.Encode(document);
                bool truncated = maxDocumentTokens > 0
                    && encoded.Length > maxDocumentTokens;
                int take = truncated ? maxDocumentTokens : encoded.Length;
                tokenBuffer.AddRange(encoded.AsSpan(0, take).ToArray());
                if (!truncated)
                    tokenBuffer.Add(BpeTokenizer.EosTokenId);
                while (batches.Count < count
                    && tokenBuffer.Count - 1 >= targetsPerBatch)
                {
                    var input = new int[targetsPerBatch];
                    var target = new int[targetsPerBatch];
                    tokenBuffer.CopyTo(0, input, 0, targetsPerBatch);
                    tokenBuffer.CopyTo(1, target, 0, targetsPerBatch);
                    tokenBuffer.RemoveRange(0, targetsPerBatch);
                    batches.Add(new ProbeBatch(
                        input,
                        target,
                        targetsPerBatch));
                }
            }
        }
        finally
        {
            documents.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
        if (batches.Count != count)
        {
            throw new InvalidDataException(
                $"Corpus produced only {batches.Count} of {count} batches.");
        }
        return batches.ToArray();
    }

    private static float Evaluate(
        LanguageModel model,
        ProbeBatch batch,
        int batchSize,
        int sequence)
    {
        bool wasTraining = model.IsTraining;
        model.eval();
        try
        {
            using (AutogradContext.NoGrad())
            using (CudaInferenceScope scope = CudaInferenceScope.Begin(
                resetPool: true,
                clearPoolOnDispose: true))
            {
                Tensor logits = model.Forward(
                    batch.Input,
                    batchSize,
                    sequence);
                return logits.CrossEntropyWithLogits(batch.Target).item();
            }
        }
        finally
        {
            if (wasTraining)
                model.train();
        }
    }

    private static ProbeBatch SliceFirstSequences(
        ProbeBatch batch,
        int sequence,
        int batchSize)
        => new(
            batch.Input.AsSpan(0, checked(sequence * batchSize)).ToArray(),
            batch.Target.AsSpan(0, checked(sequence * batchSize)).ToArray(),
            checked(sequence * batchSize));

    private static float ScheduleFactor(
        double progress,
        bool stableThenCosine)
    {
        if (!stableThenCosine)
            return WarmupCosineProgressLRScheduler.CalculateFactor(progress, 0f);
        if (progress <= StableUntilProgress)
            return 1f;
        double decayProgress = (progress - StableUntilProgress)
            / (1d - StableUntilProgress);
        return MathF.Max(
            1e-6f,
            0.5f * (1f + MathF.Cos(MathF.PI * (float)decayProgress)));
    }

    private static CudaAdaptiveShardingOptions ReadAdaptiveSharding(
        JsonElement root)
        => new()
        {
            Enabled = root.TryGetProperty(
                "adaptiveCudaSharding", out JsonElement enabled)
                && enabled.GetBoolean(),
            EmaAlpha = root.TryGetProperty(
                "cudaShardEmaAlpha", out JsonElement alpha)
                ? alpha.GetDouble()
                : 0.2d,
            MinimumRelativeShardSize = root.TryGetProperty(
                "cudaMinimumRelativeShardSize", out JsonElement minimum)
                ? minimum.GetDouble()
                : 0.5d,
            MaximumBatchAdjustmentPerStep = root.TryGetProperty(
                "cudaMaximumBatchAdjustmentPerStep", out JsonElement maximum)
                ? maximum.GetInt32()
                : 1,
        };

    private static string ResolveCheckpointPath(
        JsonElement root,
        string directory,
        string configurationPath)
    {
        JsonElement checkpoint = root.GetProperty("checkpoint");
        string checkpointDirectory = Path.GetFullPath(
            checkpoint.GetProperty("directory").GetString()!,
            directory);
        string fileName = checkpoint.TryGetProperty(
            "fileName", out JsonElement configured)
            ? configured.GetString()
                ?? Path.GetFileNameWithoutExtension(configurationPath) + ".json"
            : Path.GetFileNameWithoutExtension(configurationPath) + ".json";
        return Path.Combine(checkpointDirectory, fileName);
    }

    private static int CombineSeed(int first, int second, int third)
    {
        uint hash = 2_166_136_261u;
        hash = unchecked((hash ^ (uint)first) * 16_777_619u);
        hash = unchecked((hash ^ (uint)second) * 16_777_619u);
        hash = unchecked((hash ^ (uint)third) * 16_777_619u);
        return unchecked((int)hash);
    }

    private static ulong HashBatches(IEnumerable<ProbeBatch> batches)
    {
        ulong hash = 14_695_981_039_346_656_037ul;
        foreach (ProbeBatch batch in batches)
        {
            foreach (int value in batch.Input.Concat(batch.Target))
            {
                hash ^= unchecked((uint)value);
                hash *= 1_099_511_628_211ul;
            }
        }
        return hash;
    }

    private static (double Rms, double Relative) UpdateRms(
        float[] before,
        float[] after)
    {
        if (before.Length != after.Length)
            throw new ArgumentException("Probe lengths must match.");
        double deltaSquared = 0d;
        double weightSquared = 0d;
        for (int index = 0; index < before.Length; index++)
        {
            double delta = after[index] - before[index];
            deltaSquared += delta * delta;
            weightSquared += (double)before[index] * before[index];
        }
        double rms = Math.Sqrt(deltaSquared / before.Length);
        double weightRms = Math.Sqrt(weightSquared / before.Length);
        return (rms, weightRms == 0d ? 0d : rms / weightRms);
    }

    private readonly record struct ProbeBatch(
        int[] Input,
        int[] Target,
        int ValidTargetCount);
}
