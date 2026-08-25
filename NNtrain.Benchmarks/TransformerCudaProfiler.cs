using System.Diagnostics;
using System.Text.Json;

namespace NNtrain.Benchmarks;

internal static class TransformerCudaProfiler
{
    internal static void RunDetailedFromConfiguration(
        string configurationPath,
        int warmupSteps,
        int measuredSteps,
        string? precisionModeOverride = null)
    {
        string path = Path.GetFullPath(configurationPath);
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;
        JsonElement optimizer = root.GetProperty("optimization")
            .GetProperty("optimizer");
        TensorPrecisionMode precisionMode = precisionModeOverride is null
            ? PrecisionModeConfiguration.Read(root)
            : TensorPrecisionModeNames.Parse(precisionModeOverride);
        TensorDType dtype = precisionMode.ToStorageDType();
        RunDetailedCore(
            warmupSteps,
            measuredSteps,
            root.GetProperty("batchSize").GetInt32(),
            root.GetProperty("contextLength").GetInt32(),
            root.GetProperty("deviceIndices").EnumerateArray()
                .Select(element => element.GetInt32()).ToArray(),
            root.GetProperty("vocabularySize").GetInt32(),
            root.GetProperty("modelWidth").GetInt32(),
            root.GetProperty("heads").GetInt32(),
            root.GetProperty("hiddenSize").GetInt32(),
            root.GetProperty("layers").GetInt32(),
            root.GetProperty("seed").GetInt32(),
            root.GetProperty("dropout").GetSingle(),
            root.GetProperty("initializationScale").GetSingle(),
            root.GetProperty("tieWordEmbeddings").GetBoolean(),
            dtype,
            precisionMode,
            optimizer.GetProperty("learningRate").GetSingle(),
            optimizer.GetProperty("auxiliaryLearningRate").GetSingle(),
            optimizer.GetProperty("weightDecay").GetSingle(),
            optimizer.GetProperty("nekoMuonNewtonSchulzInterval").GetInt32(),
            precisionMode == TensorPrecisionMode.BFloat16,
            precisionMode == TensorPrecisionMode.BFloat16,
            path);
    }

    internal static void RunFromConfiguration(
        string configurationPath,
        int warmupSteps,
        int measuredSteps,
        int generationEverySteps = 0,
        int generatedTokens = 0,
        string? precisionModeOverride = null)
    {
        string path = Path.GetFullPath(configurationPath);
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;
        string architecture = root.GetProperty("modelArchitecture").GetString()
            ?? string.Empty;
        if (!string.Equals(architecture, "transformer", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Expected transformer architecture, got '{architecture}'.");
        }
        TensorPrecisionMode precisionMode = precisionModeOverride is null
            ? PrecisionModeConfiguration.Read(root)
            : TensorPrecisionModeNames.Parse(precisionModeOverride);
        TensorDType dtype = precisionMode.ToStorageDType();
        int[] devices = root.GetProperty("deviceIndices")
            .EnumerateArray()
            .Select(element => element.GetInt32())
            .ToArray();
        JsonElement optimizer = root.GetProperty("optimization")
            .GetProperty("optimizer");
        string optimizerType = optimizer.GetProperty("type").GetString()
            ?? string.Empty;
        if (!string.Equals(optimizerType, "nekomuon", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Expected NekoMuon optimizer, got '{optimizerType}'.");
        }
        RunCore(
            warmupSteps,
            measuredSteps,
            root.GetProperty("batchSize").GetInt32(),
            root.GetProperty("contextLength").GetInt32(),
            devices,
            root.GetProperty("vocabularySize").GetInt32(),
            root.GetProperty("modelWidth").GetInt32(),
            root.GetProperty("heads").GetInt32(),
            root.GetProperty("hiddenSize").GetInt32(),
            root.GetProperty("layers").GetInt32(),
            root.GetProperty("seed").GetInt32(),
            root.GetProperty("dropout").GetSingle(),
            root.GetProperty("initializationScale").GetSingle(),
            root.GetProperty("tieWordEmbeddings").GetBoolean(),
            dtype,
            precisionMode,
            optimizer.GetProperty("learningRate").GetSingle(),
            optimizer.GetProperty("auxiliaryLearningRate").GetSingle(),
            optimizer.GetProperty("weightDecay").GetSingle(),
            optimizer.GetProperty("nekoMuonNewtonSchulzInterval").GetInt32(),
            path,
            generationEverySteps: generationEverySteps,
            generatedTokens: generatedTokens);
    }

    internal static void Run(
        int warmupSteps,
        int measuredSteps,
        int batch = 8,
        int sequence = 128,
        int requestedDeviceCount = 2,
        bool useNekoMuon = false,
        string? precisionMode = null)
    {
        if (Tensor.CudaDeviceCount == 0)
            throw new InvalidOperationException("CUDA is unavailable.");
        int[] devices = Enumerable.Range(
            0, Math.Min(requestedDeviceCount, Tensor.CudaDeviceCount)).ToArray();
        TensorPrecisionMode resolvedPrecisionMode = precisionMode is null
            ? TensorPrecisionMode.BFloat16
            : TensorPrecisionModeNames.Parse(precisionMode);
        TensorDType dtype = resolvedPrecisionMode.ToStorageDType();
        RunCore(
            warmupSteps, measuredSteps, batch, sequence, devices,
            vocabulary: 4096, width: 128, heads: 4, hidden: 256, layers: 2,
            seed: 1234, dropout: 0.1f, initializationScale: 0.02f,
            tieWordEmbeddings: false, dtype,
            resolvedPrecisionMode,
            learningRate: 3e-4f, auxiliaryLearningRate: 3e-4f,
            weightDecay: 0.01f, newtonSchulzInterval: 1,
            configurationDescription: "built-in CUDA profile",
            useNekoMuon);
    }

    private static void RunCore(
        int warmupSteps,
        int measuredSteps,
        int batch,
        int sequence,
        int[] devices,
        int vocabulary,
        int width,
        int heads,
        int hidden,
        int layers,
        int seed,
        float dropout,
        float initializationScale,
        bool tieWordEmbeddings,
        TensorDType dtype,
        TensorPrecisionMode precisionMode,
        float learningRate,
        float auxiliaryLearningRate,
        float weightDecay,
        int newtonSchulzInterval,
        string configurationDescription,
        bool useNekoMuon = true,
        int generationEverySteps = 0,
        int generatedTokens = 0)
    {
        Tensor.ExecutionDevice = TensorDevice.Cuda;
        Tensor.CudaDeviceIndices = devices;
        Tensor.SimdEnabled = true;
        var model = new GptRinWikiJp(
            vocabulary, sequence, width, heads, hidden, layers,
            new Random(seed), initializationScale, dropout,
            dtype, tieWordEmbeddings);
        model.SetPrecisionMode(precisionMode);
        IOptimizer optimizer = useNekoMuon
            ? new CompositeOptimizer(
                new NekoMuon(
                    model.HiddenWeightParameters,
                    new NekoMuonOptions
                    {
                        LearningRate = learningRate,
                        WeightDecay = weightDecay,
                        MaxNewtonSchulzSteps = 5,
                        NewtonSchulzInterval = newtonSchulzInterval,
                    }),
                new AdamW(
                    model.AuxiliaryParameters,
                    new AdamWOptions
                    {
                        LearningRate = auxiliaryLearningRate,
                        WeightDecay = weightDecay,
                        UseBFloat16FirstMoment =
                            precisionMode == TensorPrecisionMode.BFloat16,
                        UseBFloat16SecondMoment =
                            precisionMode == TensorPrecisionMode.BFloat16,
                    }))
            : new AdamW(
                model.Parameters(),
                new AdamWOptions
                {
                    LearningRate = learningRate,
                    WeightDecay = weightDecay,
                    UseBFloat16FirstMoment =
                        precisionMode == TensorPrecisionMode.BFloat16,
                    UseBFloat16SecondMoment =
                        precisionMode == TensorPrecisionMode.BFloat16,
                });
        var random = new Random(seed ^ 0x5A17);
        int[] input = Enumerable.Range(0, batch * sequence)
            .Select(_ => random.Next(vocabulary)).ToArray();
        int[] target = Enumerable.Range(0, batch * sequence)
            .Select(_ => random.Next(vocabulary)).ToArray();

        (double Total, double ForwardBackward, double Optimizer, float Loss,
            long Allocated, int Gen0, int Gen1, int Gen2) Step()
        {
            optimizer.zero_grad();
            long allocated = GC.GetTotalAllocatedBytes(precise: false);
            int gen0 = GC.CollectionCount(0);
            int gen1 = GC.CollectionCount(1);
            int gen2 = GC.CollectionCount(2);
            var timer = Stopwatch.StartNew();
            float loss = CudaDataParallel.ForwardBackward(
                model, input, target, batch, sequence);
            double forwardBackward = timer.Elapsed.TotalMilliseconds;
            optimizer.step();
            timer.Stop();
            return (
                timer.Elapsed.TotalMilliseconds,
                forwardBackward,
                timer.Elapsed.TotalMilliseconds - forwardBackward,
                loss,
                GC.GetTotalAllocatedBytes(precise: false) - allocated,
                GC.CollectionCount(0) - gen0,
                GC.CollectionCount(1) - gen1,
                GC.CollectionCount(2) - gen2);
        }

        for (int step = 0; step < warmupSteps; step++)
            Step();
        var samples = new (double Total, double ForwardBackward,
            double Optimizer, float Loss, long Allocated, int Gen0, int Gen1,
            int Gen2)[measuredSteps];
        for (int step = 0; step < measuredSteps; step++)
        {
            samples[step] = Step();
            if (!float.IsFinite(samples[step].Loss))
            {
                throw new InvalidOperationException(
                    $"Non-finite loss at measured step {step + 1}: " +
                    samples[step].Loss);
            }
            Console.WriteLine(
                $"step {step + 1} = {samples[step].Total:F2} ms " +
                $"(fwd+bwd {samples[step].ForwardBackward:F2}, " +
                $"optimizer {samples[step].Optimizer:F2}, " +
                $"loss {samples[step].Loss:F6}, " +
                $"alloc {samples[step].Allocated / 1024d:N0} KiB, " +
                $"GC {samples[step].Gen0}/{samples[step].Gen1}/" +
                $"{samples[step].Gen2})");
            if (generationEverySteps > 0
                && generatedTokens > 0
                && (step + 1) % generationEverySteps == 0)
            {
                var generationTimer = Stopwatch.StartNew();
                int[] generated = model.GenerateTokenIds(
                    Enumerable.Repeat(1, sequence),
                    generatedTokens,
                    temperature: 0f,
                    topK: 1,
                    stopTokenId: null,
                    random: new Random(seed + step));
                generationTimer.Stop();
                Console.WriteLine(
                    $"generated {generated.Length - sequence} tokens after " +
                    $"step {step + 1} in " +
                    $"{generationTimer.Elapsed.TotalMilliseconds:F2} ms");
            }
        }
        double mean = samples.Average(sample => sample.Total);
        var orderedSamples = samples
            .OrderBy(sample => sample.Total)
            .ToArray();
        double median = orderedSamples[orderedSamples.Length / 2].Total;
        double tokensPerSecond = batch * sequence / (mean / 1000d);
        double[] clean = samples
            .Where(sample => sample.Gen0 == 0)
            .Select(sample => sample.Total)
            .OrderBy(value => value)
            .ToArray();
        double cleanMedian = clean.Length == 0
            ? double.NaN
            : clean[clean.Length / 2];
        double cleanTokensPerSecond = batch * sequence / (cleanMedian / 1000d);
        double cleanP25 = clean.Length == 0
            ? double.NaN
            : clean[clean.Length / 4];
        int trendWindow = Math.Min(50, Math.Max(1, measuredSteps / 10));
        double firstWindowMean = samples
            .Take(trendWindow).Average(sample => sample.Total);
        double lastWindowMean = samples
            .TakeLast(trendWindow).Average(sample => sample.Total);
        double xMean = (measuredSteps - 1) / 2d;
        double covariance = 0d;
        double xVariance = 0d;
        for (int index = 0; index < measuredSteps; index++)
        {
            double centered = index - xMean;
            covariance += centered * (samples[index].Total - mean);
            xVariance += centered * centered;
        }
        double trendPerStep = xVariance == 0d ? 0d : covariance / xVariance;
        Console.WriteLine(
            $"configuration = {configurationDescription}");
        Console.WriteLine(
            $"transformer CUDA ({devices.Length} GPU, " +
            $"{TensorPrecisionModeNames.Format(precisionMode)}): " +
            $"mean {mean:F2} ms, " +
            $"median {median:F2} ms, {tokensPerSecond:N0} tokens/s");
        Console.WriteLine(
            $"GC-free median {cleanMedian:F2} ms, " +
            $"{cleanTokensPerSecond:N0} tokens/s ({clean.Length} samples)");
        Console.WriteLine(
            $"GC-free p25 {cleanP25:F2} ms; shape batch={batch}, " +
            $"sequence={sequence}, width={width}, heads={heads}, " +
            $"hidden={hidden}, layers={layers}, vocabulary={vocabulary}, optimizer=" +
            $"{(useNekoMuon ? "NekoMuon+AdamW" : "AdamW")}");
        Console.WriteLine(
            $"time trend: first {trendWindow} mean {firstWindowMean:F2} ms, " +
            $"last {trendWindow} mean {lastWindowMean:F2} ms, " +
            $"linear slope {trendPerStep:F4} ms/step, " +
            $"max {samples.Max(sample => sample.Total):F2} ms; " +
            $"loss {samples[0].Loss:F6} -> {samples[^1].Loss:F6}");
    }

    private static void RunDetailedCore(
        int warmupSteps,
        int measuredSteps,
        int batch,
        int sequence,
        int[] devices,
        int vocabulary,
        int width,
        int heads,
        int hidden,
        int layers,
        int seed,
        float dropout,
        float initializationScale,
        bool tieWordEmbeddings,
        TensorDType dtype,
        TensorPrecisionMode precisionMode,
        float learningRate,
        float auxiliaryLearningRate,
        float weightDecay,
        int newtonSchulzInterval,
        bool adamFirstMomentBFloat16,
        bool adamSecondMomentBFloat16,
        string configurationDescription)
    {
        if (measuredSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(measuredSteps));
        Tensor.ExecutionDevice = TensorDevice.Cuda;
        Tensor.CudaDeviceIndices = devices;
        Tensor.SimdEnabled = true;
        var model = new GptRinWikiJp(
            vocabulary, sequence, width, heads, hidden, layers,
            new Random(seed), initializationScale, dropout,
            dtype, tieWordEmbeddings);
        model.SetPrecisionMode(precisionMode);
        var nekoMuon = new NekoMuon(
            model.HiddenWeightParameters,
            new NekoMuonOptions
            {
                LearningRate = learningRate,
                WeightDecay = weightDecay,
                MaxNewtonSchulzSteps = 5,
                NewtonSchulzInterval = newtonSchulzInterval,
            });
        var adamW = new AdamW(
            model.AuxiliaryParameters,
            new AdamWOptions
            {
                LearningRate = auxiliaryLearningRate,
                WeightDecay = weightDecay,
                UseBFloat16FirstMoment = adamFirstMomentBFloat16,
                UseBFloat16SecondMoment = adamSecondMomentBFloat16,
            });
        var optimizer = new CompositeOptimizer(nekoMuon, adamW);
        var random = new Random(seed ^ 0x5A17);
        int[] input = Enumerable.Range(0, batch * sequence)
            .Select(_ => random.Next(vocabulary)).ToArray();
        int[] target = Enumerable.Range(0, batch * sequence)
            .Select(_ => random.Next(vocabulary)).ToArray();

        void SynchronizeAll()
        {
            foreach (int device in devices)
                ForgetMemoryV2Cuda.GetAccelerator(device).Synchronize();
        }

        Console.WriteLine($"detailed configuration = {configurationDescription}");
        Console.WriteLine(
            $"shape batch={batch}, sequence={sequence}, width={width}, " +
            $"heads={heads}, hidden={hidden}, layers={layers}, " +
            $"vocabulary={vocabulary}; precision=" +
            $"{TensorPrecisionModeNames.Format(precisionMode)}; " +
            $"GPUs=[{string.Join(',', devices)}]");
        Console.WriteLine(
            $"optimizer=NekoMuon(interval={newtonSchulzInterval})+AdamW " +
            $"(moments={(adamFirstMomentBFloat16 ? "bf16" : "fp32")}/" +
            $"{(adamSecondMomentBFloat16 ? "bf16" : "fp32")})");

        for (int step = 0; step < warmupSteps; step++)
        {
            optimizer.zero_grad();
            _ = CudaDataParallel.ForwardBackward(
                model, input, target, batch, sequence);
            optimizer.step();
        }
        SynchronizeAll();

        var results = new List<DetailedStep>();
        var operationTotals = new Dictionary<string, double>();
        for (int step = 0; step < measuredSteps; step++)
        {
            var totalTimer = Stopwatch.StartNew();
            var phaseTimer = Stopwatch.StartNew();
            optimizer.zero_grad();
            SynchronizeAll();
            double zeroGrad = phaseTimer.Elapsed.TotalMilliseconds;

            CudaDataParallelProfile dataParallel;
            IReadOnlyList<CudaOperationProfileSample> operations;
            double nekoMuonMilliseconds;
            double adamWMilliseconds;
            using (CudaOperationProfiler.Begin())
            {
                dataParallel = CudaDataParallel.ForwardBackwardProfiled(
                    model, input, target, batch, sequence);
                phaseTimer.Restart();
                nekoMuon.step();
                SynchronizeAll();
                double measuredNekoMuon = phaseTimer.Elapsed.TotalMilliseconds;

                phaseTimer.Restart();
                adamW.step();
                SynchronizeAll();
                double measuredAdamW = phaseTimer.Elapsed.TotalMilliseconds;
                operations = CudaOperationProfiler.Snapshot();
                nekoMuonMilliseconds = measuredNekoMuon;
                adamWMilliseconds = measuredAdamW;
            }
            totalTimer.Stop();

            int optimizerStep = warmupSteps + step + 1;
            bool newtonSchulz = optimizerStep % newtonSchulzInterval == 0;
            var result = new DetailedStep(
                optimizerStep,
                newtonSchulz,
                zeroGrad,
                dataParallel,
                nekoMuonMilliseconds,
                adamWMilliseconds,
                totalTimer.Elapsed.TotalMilliseconds,
                operations);
            results.Add(result);
            PrintDetailedStep(result);

            foreach (IGrouping<string, CudaOperationProfileSample> group in
                operations.GroupBy(sample => sample.Operation))
            {
                double critical = group.Max(sample => sample.TotalMilliseconds);
                operationTotals[group.Key] = operationTotals.GetValueOrDefault(group.Key)
                    + critical;
            }
        }

        Console.WriteLine("=== mean wall-clock phases ===");
        Console.WriteLine(
            $"zero_grad {results.Average(value => value.ZeroGrad):F2} ms; " +
            $"forward {results.Average(value => value.DataParallel.Shards.Max(shard => shard.ForwardMilliseconds)):F2} ms; " +
            $"loss {results.Average(value => value.DataParallel.Shards.Max(shard => shard.LossMilliseconds)):F2} ms; " +
            $"backward {results.Average(value => value.DataParallel.Shards.Max(shard => shard.BackwardMilliseconds)):F2} ms; " +
            $"all-reduce {results.Average(value => value.DataParallel.AllReduceMilliseconds):F2} ms; " +
            $"NekoMuon {results.Average(value => value.NekoMuon):F2} ms; " +
            $"AdamW {results.Average(value => value.AdamW):F2} ms; " +
            $"total {results.Average(value => value.Total):F2} ms");
        Console.WriteLine(
            $"attention backend = " +
            $"{(CudaFlashAttention.TensorCoreBackendActive
                ? "native CUDA BF16 Tensor Core flash"
                : CudaFlashAttention.NativeBackendActive
                    ? "native CUDA scalar flash"
                : "native CUDA backend unavailable")}");
        Console.WriteLine(
            $"linear backend = {(CudaBlasLt.BackendActive
                ? "cuBLASLt BF16 Tensor Core fused epilogue"
                : "cuBLAS GEMM + separate epilogue")}");
        Console.WriteLine("=== mean CUDA operation critical time (max of GPUs) ===");
        foreach ((string operation, double total) in operationTotals
            .OrderByDescending(pair => pair.Value))
        {
            Console.WriteLine($"{operation,-42} {total / measuredSteps,9:F2} ms");
        }
    }

    private static void PrintDetailedStep(DetailedStep result)
    {
        double forward = result.DataParallel.Shards.Max(
            shard => shard.ForwardMilliseconds);
        double loss = result.DataParallel.Shards.Max(
            shard => shard.LossMilliseconds);
        double backward = result.DataParallel.Shards.Max(
            shard => shard.BackwardMilliseconds);
        Console.WriteLine(
            $"=== optimizer step {result.Step} " +
            $"(Newton-Schulz={(result.NewtonSchulz ? "yes" : "no")}) ===");
        Console.WriteLine(
            $"wall: total {result.Total:F2} ms | zero_grad {result.ZeroGrad:F2} | " +
            $"gradient-prepare {result.DataParallel.GradientPreparationMilliseconds:F2} | " +
            $"forward {forward:F2} | loss {loss:F2} | backward {backward:F2} | " +
            $"all-reduce {result.DataParallel.AllReduceMilliseconds:F2} | " +
            $"NekoMuon {result.NekoMuon:F2} | AdamW {result.AdamW:F2}");
        foreach (CudaShardProfile shard in result.DataParallel.Shards)
        {
            Console.WriteLine(
                $"GPU {shard.Device}: batch={shard.BatchSize}, " +
                $"host-shard {shard.DataPreparationMilliseconds:F2} ms, " +
                $"forward {shard.ForwardMilliseconds:F2}, " +
                $"loss {shard.LossMilliseconds:F2}, " +
                $"backward {shard.BackwardMilliseconds:F2}");
        }
        foreach (IGrouping<string, CudaOperationProfileSample> group in
            result.Operations.GroupBy(sample => sample.Operation)
                .OrderByDescending(group => group.Max(
                    sample => sample.TotalMilliseconds)))
        {
            string perGpu = string.Join(", ", group.OrderBy(sample => sample.Device)
                .Select(sample =>
                    $"GPU{sample.Device} {sample.TotalMilliseconds:F2}ms/{sample.Count}x"));
            Console.WriteLine($"  {group.Key,-40} {perGpu}");
        }
    }

    private sealed record DetailedStep(
        int Step,
        bool NewtonSchulz,
        double ZeroGrad,
        CudaDataParallelProfile DataParallel,
        double NekoMuon,
        double AdamW,
        double Total,
        IReadOnlyList<CudaOperationProfileSample> Operations);
}
