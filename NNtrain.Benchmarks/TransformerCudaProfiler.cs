using System.Diagnostics;
using System.Text.Json;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;

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
        string optimizerType = optimizer.GetProperty("type").GetString()
            ?? string.Empty;
        bool ordinaryMuon = IsOrdinaryMuon(optimizerType);
        if (!ordinaryMuon && !IsNekoMuon(optimizerType))
        {
            throw new InvalidDataException(
                $"Expected Muon or NekoMuon optimizer, got " +
                $"'{optimizerType}'.");
        }
        TensorPrecisionMode precisionMode = precisionModeOverride is null
            ? PrecisionModeConfiguration.Read(root)
            : TensorPrecisionModeNames.Parse(precisionModeOverride);
        TensorDType dtype = precisionMode.ToStorageDType();
        TransformerProfileTrainingControls controls =
            ReadTrainingControls(root, optimizer);
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
            ordinaryMuon
                ? 1
                : ReadNewtonSchulzInterval(optimizer),
            precisionMode == TensorPrecisionMode.BFloat16,
            precisionMode == TensorPrecisionMode.BFloat16,
            path,
            controls.Bfp8BlockSize,
            controls.NewtonSchulzDepthMode,
            controls.NewtonSchulzDepth,
            ordinaryMuon);
    }

    internal static void RunFromConfiguration(
        string configurationPath,
        int warmupSteps,
        int measuredSteps,
        int generationEverySteps = 0,
        int generatedTokens = 0,
        string? precisionModeOverride = null,
        float? learningRateOverride = null,
        float? auxiliaryLearningRateOverride = null)
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
        bool ordinaryMuon = IsOrdinaryMuon(optimizerType);
        if (!ordinaryMuon && !IsNekoMuon(optimizerType))
        {
            throw new InvalidDataException(
                $"Expected Muon or NekoMuon optimizer, got " +
                $"'{optimizerType}'.");
        }
        TransformerProfileTrainingControls controls =
            ReadTrainingControls(root, optimizer);
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
            learningRateOverride
                ?? optimizer.GetProperty("learningRate").GetSingle(),
            auxiliaryLearningRateOverride
                ?? optimizer.GetProperty("auxiliaryLearningRate").GetSingle(),
            optimizer.GetProperty("weightDecay").GetSingle(),
            ordinaryMuon
                ? 1
                : ReadNewtonSchulzInterval(optimizer),
            path,
            generationEverySteps: generationEverySteps,
            generatedTokens: generatedTokens,
            gradientAccumulationSteps: controls.GradientAccumulationSteps,
            bfp8BlockSize: controls.Bfp8BlockSize,
            newtonSchulzDepthMode: controls.NewtonSchulzDepthMode,
            newtonSchulzDepth: controls.NewtonSchulzDepth,
            ordinaryMuon: ordinaryMuon);
    }

    internal static TransformerProfileTrainingControls ReadTrainingControls(
        JsonElement root,
        JsonElement optimizer)
    {
        int gradientAccumulationSteps = root.TryGetProperty(
            "gradientAccumulationSteps",
            out JsonElement accumulationElement)
                ? accumulationElement.GetInt32()
                : 1;
        if (gradientAccumulationSteps <= 0)
        {
            throw new InvalidDataException(
                "gradientAccumulationSteps must be positive.");
        }

        int bfp8BlockSize;
        if (root.TryGetProperty(
                "bfp8_block_size",
                out JsonElement blockSizeElement)
            || root.TryGetProperty(
                "bfp8BlockSize",
                out blockSizeElement))
        {
            bfp8BlockSize = blockSizeElement.GetInt32();
        }
        else
        {
            bfp8BlockSize = Bfp8QuantizationDescriptor.DefaultBlockSize;
        }
        if (bfp8BlockSize <= 0)
            throw new InvalidDataException("BFP8 block size must be positive.");

        string optimizerType = optimizer.TryGetProperty(
            "type",
            out JsonElement optimizerTypeElement)
                ? optimizerTypeElement.GetString() ?? string.Empty
                : string.Empty;
        bool ordinaryMuon = IsOrdinaryMuon(optimizerType);
        NekoMuonNewtonSchulzDepthMode depthMode = ordinaryMuon
            ? NekoMuonNewtonSchulzDepthMode.Fixed
            : NekoMuonNewtonSchulzDepthMode.Adaptive;
        if (optimizer.TryGetProperty(
                "nekoMuonNewtonSchulzDepthMode",
                out JsonElement depthModeElement))
        {
            string configuredMode = depthModeElement.GetString()
                ?? string.Empty;
            if (!Enum.TryParse(
                    configuredMode,
                    ignoreCase: true,
                    out depthMode)
                || !Enum.IsDefined(depthMode))
            {
                throw new InvalidDataException(
                    $"Unsupported NekoMuon Newton-Schulz depth mode " +
                    $"'{configuredMode}'. Expected adaptive, minimum, or " +
                    "fixed.");
            }
        }

        bool hasDepth = optimizer.TryGetProperty(
            "nekoMuonNewtonSchulzDepth",
            out JsonElement depthElement);
        float depth = hasDepth
            ? depthElement.GetSingle()
            : ordinaryMuon ? 5f : 0f;
        if (ordinaryMuon
            && (depthMode != NekoMuonNewtonSchulzDepthMode.Fixed
                || depth != 5f))
        {
            throw new InvalidDataException(
                "Muon requires fixed Newton-Schulz depth 5 on every " +
                "optimizer step.");
        }
        if (depthMode == NekoMuonNewtonSchulzDepthMode.Adaptive)
        {
            if (hasDepth)
            {
                throw new InvalidDataException(
                    "Adaptive NekoMuon Newton-Schulz depth must not specify " +
                    "nekoMuonNewtonSchulzDepth.");
            }
        }
        else
        {
            if (!hasDepth && !ordinaryMuon)
            {
                throw new InvalidDataException(
                    $"NekoMuon Newton-Schulz depth mode '{depthMode}' " +
                    "requires nekoMuonNewtonSchulzDepth.");
            }
            int maximumDepth = new NekoMuonOptions().MaxNewtonSchulzSteps;
            if (!float.IsFinite(depth) || depth < 0f || depth > maximumDepth)
            {
                throw new InvalidDataException(
                    $"NekoMuon Newton-Schulz depth must be finite and in " +
                    $"[0, {maximumDepth}].");
            }
        }

        return new TransformerProfileTrainingControls(
            gradientAccumulationSteps,
            bfp8BlockSize,
            depthMode,
            depth);
    }

    private static bool IsOrdinaryMuon(string optimizerType)
        => string.Equals(
            optimizerType,
            "muon",
            StringComparison.OrdinalIgnoreCase);

    private static bool IsNekoMuon(string optimizerType)
        => string.Equals(
            optimizerType,
            "nekomuon",
            StringComparison.OrdinalIgnoreCase);

    private static int ReadNewtonSchulzInterval(JsonElement optimizer)
        => optimizer.TryGetProperty(
            "nekoMuonNewtonSchulzInterval",
            out JsonElement interval)
                ? interval.GetInt32()
                : 5;

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
        int generatedTokens = 0,
        int gradientAccumulationSteps = 1,
        int bfp8BlockSize = Bfp8QuantizationDescriptor.DefaultBlockSize,
        NekoMuonNewtonSchulzDepthMode newtonSchulzDepthMode =
            NekoMuonNewtonSchulzDepthMode.Adaptive,
        float newtonSchulzDepth = 0f,
        bool ordinaryMuon = false)
    {
        if (gradientAccumulationSteps <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(gradientAccumulationSteps));
        }
        if (bfp8BlockSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(bfp8BlockSize));
        Tensor.ExecutionDevice = TensorDevice.Cuda;
        Tensor.CudaDeviceIndices = devices;
        Tensor.SimdEnabled = true;
        using var execution = new ExecutionSession(new ExecutionOptions
        {
            Device = ExecutionDeviceKind.Cuda,
            CudaDevices = new DeviceSet(devices),
            Precision = PrecisionPolicy.Parse(
                TensorPrecisionModeNames.Format(precisionMode)),
        });
        foreach (int device in devices)
            execution.AttachLane(CudaExecutionLaneFactory.Create(device));
        using IDisposable executionScope = execution.Enter();
        var trainingRandom = new CheckpointableRandom(seed);
        var model = new GptRinWikiJp(
            vocabulary, sequence, width, heads, hidden, layers,
            trainingRandom, initializationScale, dropout,
            dtype, tieWordEmbeddings);
        trainingRandom.BeginRuntime();
        model.AttachTrainingRandom(trainingRandom);
        // Precision is a physical storage contract, not metadata. In
        // particular mix8_32 must convert constructor-created tensor-wide
        // BFP8 storage to block-scaled storage before CUDA execution.
        model.to(precisionMode, bfp8BlockSize);
        IOptimizer optimizer = useNekoMuon
            ? new CompositeOptimizer(
                CreateMatrixOptimizer(
                    model.HiddenWeightParameters,
                    learningRate,
                    weightDecay,
                    newtonSchulzInterval,
                    newtonSchulzDepthMode,
                    newtonSchulzDepth,
                    ordinaryMuon),
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
        var microBatches = new CudaLanguageModelMicroBatch[
            gradientAccumulationSteps];
        for (int index = 0; index < microBatches.Length; index++)
        {
            int[] input = Enumerable.Range(0, checked(batch * sequence))
                .Select(_ => random.Next(vocabulary)).ToArray();
            int[] target = Enumerable.Range(0, checked(batch * sequence))
                .Select(_ => random.Next(vocabulary)).ToArray();
            microBatches[index] = new CudaLanguageModelMicroBatch(
                input,
                target,
                batch,
                sequence);
        }
        Parameter[] benchmarkParameters = model.Parameters().ToArray();
        using var dataParallelEngine =
            new CudaDataParallelEngine(model, devices);
        dataParallelEngine.PrepareForTraining(batch);
        optimizer.prepare();
        long globalStep = 0;

        (double Total, double ZeroGrad, double ForwardBackward,
            double GradientClip, double Optimizer, float Loss,
            float GradientNorm, long Allocated,
            int Gen0, int Gen1, int Gen2,
            NativeCudaAllocationTelemetry NativeAllocations) Step()
        {
            NativeCudaAllocationTelemetry nativeBefore =
                NativeCudaRuntime.AllocationTelemetry;
            long allocated = GC.GetTotalAllocatedBytes(precise: false);
            int gen0 = GC.CollectionCount(0);
            int gen1 = GC.CollectionCount(1);
            int gen2 = GC.CollectionCount(2);
            var timer = Stopwatch.StartNew();
            optimizer.zero_grad();
            double zeroGrad = timer.Elapsed.TotalMilliseconds;
            float loss;
            if (microBatches.Length == 1)
            {
                CudaLanguageModelMicroBatch microBatch = microBatches[0];
                loss = dataParallelEngine.ForwardBackward(
                    microBatch.Input,
                    microBatch.Target,
                    microBatch.BatchSize,
                    microBatch.SequenceLength,
                    Tensor.DefaultCrossEntropyIgnoreIndex,
                    globalStep);
            }
            else
            {
                loss = dataParallelEngine.ForwardBackwardAccumulated(
                    microBatches,
                    Tensor.DefaultCrossEntropyIgnoreIndex,
                    globalStep);
            }
            globalStep = checked(globalStep + 1);
            double afterForwardBackward = timer.Elapsed.TotalMilliseconds;
            float gradientNorm = nn.utils.clip_grad_norm_(
                benchmarkParameters,
                max_norm: 1f);
            double afterGradientClip = timer.Elapsed.TotalMilliseconds;
            optimizer.step();
            timer.Stop();
            return (
                timer.Elapsed.TotalMilliseconds,
                zeroGrad,
                afterForwardBackward - zeroGrad,
                afterGradientClip - afterForwardBackward,
                timer.Elapsed.TotalMilliseconds - afterGradientClip,
                loss,
                gradientNorm,
                GC.GetTotalAllocatedBytes(precise: false) - allocated,
                GC.CollectionCount(0) - gen0,
                GC.CollectionCount(1) - gen1,
                GC.CollectionCount(2) - gen2,
                NativeCudaRuntime.AllocationTelemetry - nativeBefore);
        }

        for (int step = 0; step < warmupSteps; step++)
            Step();
        CudaTrainingGraphTelemetry graphBeforeMeasurement =
            dataParallelEngine.TrainingGraphTelemetry;
        NativeCudaTransferTelemetry transfersBeforeMeasurement =
            NativeCudaRuntime.TransferTelemetry;
        NativeCudaTransferTelemetry gradientTransfersBeforeMeasurement =
            NativeCudaRuntime.GradientCollectiveTransferTelemetry;
        var samples = new (double Total, double ZeroGrad,
            double ForwardBackward, double GradientClip, double Optimizer,
            float Loss, float GradientNorm, long Allocated,
            int Gen0, int Gen1, int Gen2,
            NativeCudaAllocationTelemetry NativeAllocations)
            [measuredSteps];
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
                $"(zero {samples[step].ZeroGrad:F2}, " +
                $"fwd+bwd {samples[step].ForwardBackward:F2}, " +
                $"clip {samples[step].GradientClip:F2}, " +
                $"optimizer {samples[step].Optimizer:F2}, " +
                $"loss {samples[step].Loss:F6}, " +
                $"grad norm {samples[step].GradientNorm:F4}" +
                $"{(samples[step].GradientNorm > 1f ? " (clipped), " : ", ")}" +
                $"alloc {samples[step].Allocated / 1024d:N0} KiB, " +
                $"CUDA malloc/free " +
                $"{samples[step].NativeAllocations.AllocationCount}/" +
                $"{samples[step].NativeAllocations.FreeCount} " +
                $"({samples[step].NativeAllocations.AllocationBytes
                    / 1048576d:F1}/" +
                $"{samples[step].NativeAllocations.FreeBytes
                    / 1048576d:F1} MiB), " +
                $"GC {samples[step].Gen0}/{samples[step].Gen1}/" +
                $"{samples[step].Gen2}, gpu shard " +
                $"{string.Join('/', dataParallelEngine.LastShardBatchSizes)})");
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
        int p95Index = Math.Clamp(
            (int)Math.Ceiling(orderedSamples.Length * 0.95d) - 1,
            0,
            orderedSamples.Length - 1);
        double p95 = orderedSamples[p95Index].Total;
        long tokensPerUpdate = checked(
            (long)batch * sequence * gradientAccumulationSteps);
        double tokensPerSecond = tokensPerUpdate / (mean / 1000d);
        double[] clean = samples
            .Where(sample => sample.Gen0 == 0)
            .Select(sample => sample.Total)
            .OrderBy(value => value)
            .ToArray();
        double cleanMedian = clean.Length == 0
            ? double.NaN
            : clean[clean.Length / 2];
        double cleanTokensPerSecond = tokensPerUpdate
            / (cleanMedian / 1000d);
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
            $"learning rates = " +
            $"{(ordinaryMuon ? "Muon" : "NekoMuon")} " +
            $"{learningRate:G}, " +
            $"AdamW {auxiliaryLearningRate:G}; gradient clipping " +
            $"{samples.Count(sample => sample.GradientNorm > 1f)}/" +
            $"{samples.Length} measured steps, mean pre-clip norm " +
            $"{samples.Average(sample => sample.GradientNorm):F4}");
        Console.WriteLine(
            $"transformer CUDA ({devices.Length} GPU, " +
            $"{TensorPrecisionModeNames.Format(precisionMode)}): " +
            $"mean {mean:F2} ms, " +
            $"p50 {median:F2} ms, p95 {p95:F2} ms, " +
            $"{tokensPerSecond:N0} tokens/s");
        Console.WriteLine(
            $"GC-free median {cleanMedian:F2} ms, " +
            $"{cleanTokensPerSecond:N0} tokens/s ({clean.Length} samples)");
        Console.WriteLine(
            $"GC-free p25 {cleanP25:F2} ms; shape microbatch={batch}, " +
            $"accumulation={gradientAccumulationSteps}, effective-batch=" +
            $"{checked((long)batch * gradientAccumulationSteps)}, " +
            $"sequence={sequence}, width={width}, heads={heads}, " +
            $"hidden={hidden}, layers={layers}, vocabulary={vocabulary}, optimizer=" +
            $"{(useNekoMuon
                ? ordinaryMuon ? "Muon+AdamW" : "NekoMuon+AdamW"
                : "AdamW")}");
        Console.WriteLine(
            $"time trend: first {trendWindow} mean {firstWindowMean:F2} ms, " +
            $"last {trendWindow} mean {lastWindowMean:F2} ms, " +
            $"linear slope {trendPerStep:F4} ms/step, " +
            $"max {samples.Max(sample => sample.Total):F2} ms; " +
            $"loss {samples[0].Loss:F6} -> {samples[^1].Loss:F6}");
        Console.WriteLine(
            $"final adaptive GPU shard = [" +
            $"{string.Join(',', dataParallelEngine.LastShardBatchSizes)}]");
        if (devices.Length > 1)
        {
            var peerRoutes = new List<string>();
            for (int source = 0; source < devices.Length; source++)
            {
                for (int destination = 0;
                    destination < devices.Length;
                    destination++)
                {
                    if (source == destination)
                        continue;
                    peerRoutes.Add(
                        $"{devices[source]}->{devices[destination]}=" +
                        (NativeCudaRuntime.CanAccessPeer(
                            devices[source], devices[destination])
                            ? "yes"
                            : "no"));
                }
            }
            Console.WriteLine(
                $"CUDA peer access: {string.Join(", ", peerRoutes)}");
        }
        foreach (int device in devices)
        {
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(device);
            long totalBytes = accelerator.MemorySize;
            long freeBytes = accelerator.GetFreeMemory();
            Console.WriteLine(
                $"GPU {device} VRAM: used=" +
                $"{(totalBytes - freeBytes) / 1048576d:F1} MiB, " +
                $"free={freeBytes / 1048576d:F1} MiB, " +
                $"total={totalBytes / 1048576d:F1} MiB");
        }
        NativeCudaTransferTelemetry measuredTransfers =
            NativeCudaRuntime.TransferTelemetry
                - transfersBeforeMeasurement;
        NativeCudaTransferTelemetry measuredGradientTransfers =
            NativeCudaRuntime.GradientCollectiveTransferTelemetry
                - gradientTransfersBeforeMeasurement;
        NativeCudaTransferTelemetry measuredNonCollectiveTransfers =
            measuredTransfers - measuredGradientTransfers;
        Console.WriteLine(
            $"physical transfers ({measuredSteps} measured steps): " +
            $"total H2D={measuredTransfers.HostToDeviceCopyCount}/" +
            $"{measuredTransfers.HostToDeviceBytes:N0} B, " +
            $"D2H={measuredTransfers.DeviceToHostCopyCount}/" +
            $"{measuredTransfers.DeviceToHostBytes:N0} B; " +
            $"gradient collective H2D=" +
            $"{measuredGradientTransfers.HostToDeviceCopyCount}/" +
            $"{measuredGradientTransfers.HostToDeviceBytes:N0} B, " +
            $"D2H={measuredGradientTransfers.DeviceToHostCopyCount}/" +
            $"{measuredGradientTransfers.DeviceToHostBytes:N0} B; " +
            $"batch/scalar H2D=" +
            $"{measuredNonCollectiveTransfers.HostToDeviceCopyCount}/" +
            $"{measuredNonCollectiveTransfers.HostToDeviceBytes:N0} B, " +
            $"D2H={measuredNonCollectiveTransfers.DeviceToHostCopyCount}/" +
            $"{measuredNonCollectiveTransfers.DeviceToHostBytes:N0} B");
        CudaTrainingGraphTelemetry graphAfterMeasurement =
            dataParallelEngine.TrainingGraphTelemetry;
        long measuredGraphCaptures = graphAfterMeasurement.CaptureCount
            - graphBeforeMeasurement.CaptureCount;
        long measuredGraphReplays = graphAfterMeasurement.ReplayCount
            - graphBeforeMeasurement.ReplayCount;
        long measuredGraphFallbacks = graphAfterMeasurement.FallbackCount
            - graphBeforeMeasurement.FallbackCount;
        long measuredReadyEvents =
            graphAfterMeasurement.CapturedReadyEventRecordCount
                - graphBeforeMeasurement.CapturedReadyEventRecordCount;
        double measuredReadyEventMilliseconds = Math.Max(
            0d,
            graphAfterMeasurement.CapturedReadyEventRecordMilliseconds
                - graphBeforeMeasurement.CapturedReadyEventRecordMilliseconds);
        bool fullyCompiledReplay =
            graphBeforeMeasurement.CachedCompiledPlanCount > 0
            && measuredGraphCaptures == 0
            && measuredGraphFallbacks == 0
            && measuredGraphReplays == checked(
                (long)measuredSteps * gradientAccumulationSteps);
        Console.WriteLine(
            $"CUDA Graph: capture={graphAfterMeasurement.CaptureCount} " +
            $"(measured +{measuredGraphCaptures}), " +
            $"replay={graphAfterMeasurement.ReplayCount} " +
            $"(measured +{measuredGraphReplays}), " +
            $"fallback={graphAfterMeasurement.FallbackCount} " +
            $"(measured +{measuredGraphFallbacks}), " +
            $"compiled={graphAfterMeasurement.CachedCompiledPlanCount}, " +
            $"pinned={graphAfterMeasurement.GraphPinnedBytes:N0} B, " +
            $"post-replay ready-events={measuredReadyEvents}/" +
            $"{measuredReadyEventMilliseconds:F3} ms, measured-path=" +
            (fullyCompiledReplay
                ? "compiled replay"
                : "not fully compiled replay"));
        if (dataParallelEngine.LastGraphFailure is { } graphFailure)
        {
            Console.WriteLine(
                $"CUDA Graph last failure: {graphFailure.GetType().Name}: " +
                graphFailure.Message);
        }
        if (dataParallelEngine.LastGradientOverlapTelemetry is
            { } overlap)
        {
            static string Time(double? value)
                => value.HasValue ? $"{value.Value:F2}" : "n/a";
            Console.WriteLine(
                $"gradient overlap (last step): external-event=" +
                $"{overlap.UsedExternalCapturedReadyEvents}, " +
                $"host-work-before-Complete=" +
                $"{overlap.HostWorkCompletedBeforeComplete}/" +
                $"{overlap.ScheduledHostWorkCount}, " +
                $"bucket-ready={Time(overlap.FirstBucketReadyMilliseconds)}.." +
                $"{Time(overlap.LastBucketReadyMilliseconds)} ms, " +
                $"host-start={Time(overlap.FirstHostWorkStartedMilliseconds)} ms, " +
                $"host-first-finished=" +
                $"{Time(overlap.FirstHostWorkCompletedMilliseconds)} ms, " +
                $"host-finished=" +
                $"{Time(overlap.LastHostWorkCompletedMilliseconds)} ms, " +
                $"Complete-enter={overlap.CompleteEnteredMilliseconds:F2} ms, " +
                $"Complete-finished={overlap.CompleteFinishedMilliseconds:F2} ms, " +
                $"host-wait={overlap.CompleteHostWaitMilliseconds:F2} ms, " +
                $"bucket-order=[{string.Join(',', overlap.BucketPublicationOrder)}]");
        }
    }

    private static NekoMuon CreateMatrixOptimizer(
        IEnumerable<Parameter> parameters,
        float learningRate,
        float weightDecay,
        int newtonSchulzInterval,
        NekoMuonNewtonSchulzDepthMode newtonSchulzDepthMode,
        float newtonSchulzDepth,
        bool ordinaryMuon)
    {
        var optimizer = new NekoMuon(
            parameters,
            new NekoMuonOptions
            {
                LearningRate = learningRate,
                WeightDecay = weightDecay,
                MaxNewtonSchulzSteps = 5,
                NewtonSchulzInterval = newtonSchulzInterval,
                NewtonSchulzDepthMode = newtonSchulzDepthMode,
                NewtonSchulzDepth = newtonSchulzDepth,
            });
        if (ordinaryMuon)
            optimizer.SetOrdinaryMuonPolicy();
        return optimizer;
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
        string configurationDescription,
        int bfp8BlockSize,
        NekoMuonNewtonSchulzDepthMode newtonSchulzDepthMode,
        float newtonSchulzDepth,
        bool ordinaryMuon)
    {
        if (measuredSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(measuredSteps));
        Tensor.ExecutionDevice = TensorDevice.Cuda;
        Tensor.CudaDeviceIndices = devices;
        Tensor.SimdEnabled = true;
        using var execution = new ExecutionSession(new ExecutionOptions
        {
            Device = ExecutionDeviceKind.Cuda,
            CudaDevices = new DeviceSet(devices),
            Precision = PrecisionPolicy.Parse(
                TensorPrecisionModeNames.Format(precisionMode)),
        });
        foreach (int device in devices)
            execution.AttachLane(CudaExecutionLaneFactory.Create(device));
        using IDisposable executionScope = execution.Enter();
        var trainingRandom = new CheckpointableRandom(seed);
        var model = new GptRinWikiJp(
            vocabulary, sequence, width, heads, hidden, layers,
            trainingRandom, initializationScale, dropout,
            dtype, tieWordEmbeddings);
        trainingRandom.BeginRuntime();
        model.AttachTrainingRandom(trainingRandom);
        model.to(precisionMode, bfp8BlockSize);
        NekoMuon nekoMuon = CreateMatrixOptimizer(
            model.HiddenWeightParameters,
            learningRate,
            weightDecay,
            newtonSchulzInterval,
            newtonSchulzDepthMode,
            newtonSchulzDepth,
            ordinaryMuon);
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
        bool skipOptimizer = string.Equals(
            Environment.GetEnvironmentVariable(
                "NNTRAIN_PROFILE_SKIP_OPTIMIZER"),
            "1",
            StringComparison.Ordinal);
        var random = new Random(seed ^ 0x5A17);
        int[] input = Enumerable.Range(0, batch * sequence)
            .Select(_ => random.Next(vocabulary)).ToArray();
        int[] target = Enumerable.Range(0, batch * sequence)
            .Select(_ => random.Next(vocabulary)).ToArray();
        Parameter[] benchmarkParameters = model.Parameters().ToArray();
        optimizer.prepare();

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
            $"optimizer={(ordinaryMuon ? "Muon" : "NekoMuon")}" +
            $"(interval={newtonSchulzInterval})+AdamW " +
            $"(moments={(adamFirstMomentBFloat16 ? "bf16" : "fp32")}/" +
            $"{(adamSecondMomentBFloat16 ? "bf16" : "fp32")})" +
            (skipOptimizer ? " [optimizer step skipped]" : ""));
        Console.WriteLine(
            "training plan=eager synchronized diagnostic using the " +
            "production ForwardLoss path; CUDA Graph is disabled for " +
            "operation attribution; the decomposed Forward->BFP8 logits" +
            "->CrossEntropy path is diagnostic-only and is not timed");

        // Normal warmup intentionally exercises the compiled CUDA Graph path.
        // Dispose that graph before the eager diagnostic pass: a production
        // shape can pin several GiB, and retaining it while allocating a
        // second eager activation graph makes the profiler itself OOM even
        // though normal training fits.
        using (var warmupDataParallelEngine =
            new CudaDataParallelEngine(model, devices))
        {
            warmupDataParallelEngine.PrepareForTraining(batch);
            for (int step = 0; step < warmupSteps; step++)
            {
                optimizer.zero_grad();
                _ = warmupDataParallelEngine.ForwardBackward(
                    input, target, batch, sequence);
                _ = nn.utils.clip_grad_norm_(
                    benchmarkParameters,
                    max_norm: 1f);
                if (!skipOptimizer)
                    optimizer.step();
            }
            SynchronizeAll();
        }
        using var dataParallelEngine =
            new CudaDataParallelEngine(model, devices);
        dataParallelEngine.PrepareForTraining(batch);
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
            double measuredGradientClip;
            double nekoMuonMilliseconds;
            double adamWMilliseconds;
            using (CudaOperationProfiler.Begin())
            {
                dataParallel = dataParallelEngine.ForwardBackwardProfiled(
                    input, target, batch, sequence);
                phaseTimer.Restart();
                _ = nn.utils.clip_grad_norm_(benchmarkParameters, max_norm: 1f);
                SynchronizeAll();
                measuredGradientClip =
                    phaseTimer.Elapsed.TotalMilliseconds;

                double measuredNekoMuon = 0d;
                double measuredAdamW = 0d;
                if (!skipOptimizer)
                {
                    phaseTimer.Restart();
                    nekoMuon.step();
                    SynchronizeAll();
                    measuredNekoMuon = phaseTimer.Elapsed.TotalMilliseconds;

                    phaseTimer.Restart();
                    adamW.step();
                    SynchronizeAll();
                    measuredAdamW = phaseTimer.Elapsed.TotalMilliseconds;
                }
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
                measuredGradientClip,
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
            $"forward+loss {results.Average(value => value.DataParallel.Shards.Max(shard => shard.ForwardMilliseconds)):F2} ms; " +
            $"loss scalar readback {results.Average(value => value.DataParallel.Shards.Max(shard => shard.LossMilliseconds)):F2} ms; " +
            $"backward {results.Average(value => value.DataParallel.Shards.Max(shard => shard.BackwardMilliseconds)):F2} ms; " +
            $"all-reduce {results.Average(value => value.DataParallel.AllReduceMilliseconds):F2} ms; " +
            $"gradient-clip {results.Average(value => value.GradientClip):F2} ms; " +
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
                ? $"cuBLASLt BF16 Tensor Core fused epilogue; backward=" +
                    $"{(CudaBlasLt.BackwardBackendActive ? "cuBLASLt" : "cuBLAS")}"
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
            $"forward+loss {forward:F2} | loss-readback {loss:F2} | " +
            $"backward {backward:F2} | " +
            $"all-reduce {result.DataParallel.AllReduceMilliseconds:F2} | " +
            $"gradient-clip {result.GradientClip:F2} | " +
            $"NekoMuon {result.NekoMuon:F2} | AdamW {result.AdamW:F2}");
        foreach (CudaShardProfile shard in result.DataParallel.Shards)
        {
            Console.WriteLine(
                $"GPU {shard.Device}: batch={shard.BatchSize}, " +
                $"host-shard {shard.DataPreparationMilliseconds:F2} ms, " +
                $"forward+loss {shard.ForwardMilliseconds:F2}, " +
                $"loss-readback {shard.LossMilliseconds:F2}, " +
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
        double GradientClip,
        double NekoMuon,
        double AdamW,
        double Total,
        IReadOnlyList<CudaOperationProfileSample> Operations);
}

internal readonly record struct TransformerProfileTrainingControls(
    int GradientAccumulationSteps,
    int Bfp8BlockSize,
    NekoMuonNewtonSchulzDepthMode NewtonSchulzDepthMode,
    float NewtonSchulzDepth);
