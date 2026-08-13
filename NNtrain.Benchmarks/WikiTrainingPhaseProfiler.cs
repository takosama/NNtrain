using System.Diagnostics;
using System.Text.Json;
using NNtrain;

namespace NNtrain.Benchmarks;

internal static class WikiTrainingPhaseProfiler
{
    internal static void Run(
        string configurationPath,
        TensorDType? dtypeOverride = null,
        bool? nativeFloat16Override = null,
        int? warmupStepsOverride = null,
        int? measuredStepsOverride = null)
    {
        string path = Path.GetFullPath(configurationPath);
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;
        int vocabulary = ReadInt(root, "vocabularySize");
        int batch = ReadInt(root, "batchSize");
        int sequence = ReadInt(root, "contextLength");
        int width = ReadInt(root, "modelWidth");
        int hidden = ReadInt(root, "hiddenSize");
        int layers = ReadInt(root, "layers");
        int keyWidth = ReadInt(root, "forgetMemoryKeyWidth");
        int valueWidth = ReadInt(root, "forgetMemoryValueWidth");
        float retentionMinimum = ReadSingle(
            root,
            "forgetMemoryRetentionMinimum");
        float retentionMaximum = ReadSingle(
            root,
            "forgetMemoryRetentionMaximum");
        float dropout = ReadSingle(root, "dropout");
        float initializationScale = ReadSingle(root, "initializationScale");
        float learningRate = ReadSingle(root, "learningRate");
        float auxiliaryLearningRate = ReadSingle(
            root,
            "auxiliaryLearningRate");
        int newtonSchulzInterval = ReadInt(
            root,
            "nekoMuonNewtonSchulzInterval");
        float weightDecay = ReadSingle(root, "weightDecay");
        int seed = ReadInt(root, "seed");
        TensorDType dtype = dtypeOverride ?? ReadDType(root);

        Tensor.SimdEnabled = root.GetProperty("useSimd").GetBoolean();
        if (nativeFloat16Override.HasValue)
            Tensor.Float16NativeEnabled = nativeFloat16Override.Value;
        Tensor.MaxDegreeOfParallelism = ReadInt(
            root,
            "maxDegreeOfParallelism");
        var model = new FrogetMemoryV2Gpt(
            vocabulary,
            sequence,
            width,
            hidden,
            layers,
            keyWidth,
            valueWidth,
            retentionMinimum,
            retentionMaximum,
            new Random(seed),
            initializationScale,
            dropout,
            dtype);
        var neko = new NekoMuon(
            model.HiddenWeightParameters,
            new NekoMuonOptions
            {
                LearningRate = learningRate,
                NewtonSchulzInterval = newtonSchulzInterval,
                WeightDecay = weightDecay,
            });
        var adam = new AdamW(
            model.AuxiliaryParameters,
            new AdamWOptions
            {
                LearningRate = auxiliaryLearningRate,
                Beta1 = 0.9f,
                Beta2 = 0.95f,
                WeightDecay = weightDecay,
            });
        var random = new Random(seed ^ 0x5A17);
        int[] tokens = Enumerable.Range(0, checked(batch * sequence))
            .Select(_ => random.Next(vocabulary))
            .ToArray();
        int[] targets = Enumerable.Range(0, checked(batch * sequence))
            .Select(_ => random.Next(vocabulary))
            .ToArray();

        Console.WriteLine($"configuration = {path}");
        Console.WriteLine(
            $"model = batch {batch}, sequence {sequence}, width {width}, " +
            $"hidden {hidden}, layers {layers}, vocabulary {vocabulary}, " +
            $"key/value {keyWidth}/{valueWidth}, dtype {dtype}, " +
            $"native-f16c {Tensor.IsFloat16NativeAccelerated}");
        Console.WriteLine(
            $"parameters = {model.Parameters().Sum(parameter => (long)parameter.T.Numel):N0}, " +
            $"Neko matrices = {model.HiddenWeightParameters.Count}");

        int warmupSteps = warmupStepsOverride
            ?? Math.Max(2, newtonSchulzInterval);
        int measuredSteps = measuredStepsOverride
            ?? Math.Max(10, checked(newtonSchulzInterval * 2));
        if (warmupSteps < 0 || measuredSteps <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(measuredStepsOverride),
                "Warmup must be non-negative and measured steps positive.");
        }
        for (int step = 0; step < warmupSteps; step++)
            MeasureStep(
                model,
                neko,
                adam,
                tokens,
                targets,
                batch,
                sequence);

        var results = new PhaseTimes[measuredSteps];
        for (int step = 0; step < measuredSteps; step++)
        {
            results[step] = MeasureStep(
                model,
                neko,
                adam,
                tokens,
                targets,
                batch,
                sequence);
            Console.WriteLine(
                $"step {step + 1}: total {results[step].TotalMilliseconds:F2} ms " +
                $"(fwd {results[step].Forward:F2}, " +
                $"bwd {results[step].Backward:F2}, " +
                $"neko {results[step].Neko:F2}, " +
                $"GC {results[step].Gen0Collections}/" +
                $"{results[step].Gen1Collections}/" +
                $"{results[step].Gen2Collections}, " +
                $"alloc {results[step].AllocatedBytes / 1_048_576d:F1} MiB)");
        }

        double zero = results.Average(result => result.ZeroGrad);
        double forward = results.Average(result => result.Forward);
        double loss = results.Average(result => result.Loss);
        double backward = results.Average(result => result.Backward);
        double nekoTime = results.Average(result => result.Neko);
        double adamTime = results.Average(result => result.Adam);
        double total = zero + forward + loss + backward + nekoTime + adamTime;
        PrintPhase("ZeroGrad", zero, total);
        PrintPhase("Forward", forward, total);
        PrintPhase("Loss", loss, total);
        PrintPhase("Backward", backward, total);
        PrintPhase("NekoMuon", nekoTime, total);
        PrintPhase("AdamW", adamTime, total);
        Console.WriteLine($"mean phase sum = {total:F2} ms");

        // Keep the wall-clock result free of per-parameter Stopwatch and
        // Interlocked overhead. One separate step collects the detailed,
        // summed worker-CPU breakdown below.
        int completedSteps = neko.CaptureState().Step;
        int stepsUntilRefresh = newtonSchulzInterval
            - completedSteps % newtonSchulzInterval;
        for (int step = 1; step < stepsUntilRefresh; step++)
        {
            MeasureStep(
                model,
                neko,
                adam,
                tokens,
                targets,
                batch,
                sequence);
        }
        neko.ProfilingEnabled = true;
        MeasureStep(
            model,
            neko,
            adam,
            tokens,
            targets,
            batch,
            sequence);
        NekoMuonStepProfile nekoProfile = neko.LastStepProfile;
        double nekoCpu = nekoProfile.TotalCpuMilliseconds;
        Console.WriteLine("NekoMuon summed worker CPU profile:");
        PrintPhase(
            "Moments",
            nekoProfile.UpdateMomentsMilliseconds,
            nekoCpu);
        PrintPhase(
            "Confidence",
            nekoProfile.ConfidenceMilliseconds,
            nekoCpu);
        PrintPhase(
            "Initialize",
            nekoProfile.InitializeMilliseconds,
            nekoCpu);
        PrintPhase(
            "Newton",
            nekoProfile.NewtonSchulzMilliseconds,
            nekoCpu);
        PrintPhase(
            "Transpose",
            nekoProfile.TransposeMilliseconds,
            nekoCpu);
        PrintPhase(
            "Apply",
            nekoProfile.ApplyUpdateMilliseconds,
            nekoCpu);
        Console.WriteLine("Newton-Schulz summed worker CPU detail:");
        double newtonDetail = nekoProfile.FirstGramMilliseconds
            + nekoProfile.GramSquaredMilliseconds
            + nekoProfile.PolynomialMilliseconds;
        PrintPhase("Gram(X)", nekoProfile.FirstGramMilliseconds, newtonDetail);
        PrintPhase(
            "Gram^2",
            nekoProfile.GramSquaredMilliseconds,
            newtonDetail);
        PrintPhase(
            "Polynomial",
            nekoProfile.PolynomialMilliseconds,
            newtonDetail);
    }

    private static PhaseTimes MeasureStep(
        FrogetMemoryV2Gpt model,
        NekoMuon neko,
        AdamW adam,
        int[] tokens,
        int[] targets,
        int batch,
        int sequence)
    {
        long allocatedBefore = GC.GetTotalAllocatedBytes(precise: false);
        int gen0Before = GC.CollectionCount(0);
        int gen1Before = GC.CollectionCount(1);
        int gen2Before = GC.CollectionCount(2);
        long start = Stopwatch.GetTimestamp();
        model.ZeroGrad();
        long afterZero = Stopwatch.GetTimestamp();
        Tensor logits = model.Forward(tokens, batch, sequence);
        long afterForward = Stopwatch.GetTimestamp();
        Tensor loss = logits.CrossEntropyWithLogits(targets);
        long afterLoss = Stopwatch.GetTimestamp();
        loss.Backward();
        long afterBackward = Stopwatch.GetTimestamp();
        neko.Step();
        long afterNeko = Stopwatch.GetTimestamp();
        adam.Step();
        long afterAdam = Stopwatch.GetTimestamp();
        GC.KeepAlive(loss);
        return new PhaseTimes(
            ToMilliseconds(start, afterZero),
            ToMilliseconds(afterZero, afterForward),
            ToMilliseconds(afterForward, afterLoss),
            ToMilliseconds(afterLoss, afterBackward),
            ToMilliseconds(afterBackward, afterNeko),
            ToMilliseconds(afterNeko, afterAdam),
            GC.GetTotalAllocatedBytes(precise: false) - allocatedBefore,
            GC.CollectionCount(0) - gen0Before,
            GC.CollectionCount(1) - gen1Before,
            GC.CollectionCount(2) - gen2Before);
    }

    private static double ToMilliseconds(long start, long end)
        => (end - start) * 1000d / Stopwatch.Frequency;

    private static void PrintPhase(string name, double value, double total)
        => Console.WriteLine(
            $"{name,-10} {value,10:F2} ms {value / total,9:P1}");

    private static int ReadInt(JsonElement root, string name)
        => root.GetProperty(name).GetInt32();

    private static float ReadSingle(JsonElement root, string name)
        => root.GetProperty(name).GetSingle();

    private static TensorDType ReadDType(JsonElement root)
    {
        if (!root.TryGetProperty("modelDType", out JsonElement element)
            || element.ValueKind == JsonValueKind.Null)
        {
            return TensorDType.Float16;
        }

        return element.GetString()?.ToLowerInvariant() switch
        {
            "float16" or "half" => TensorDType.Float16,
            "float32" => TensorDType.Float32,
            string value => throw new InvalidDataException(
                $"Unsupported modelDType '{value}'."),
            _ => throw new InvalidDataException(
                "modelDType must be a string."),
        };
    }

    private readonly record struct PhaseTimes(
        double ZeroGrad,
        double Forward,
        double Loss,
        double Backward,
        double Neko,
        double Adam,
        long AllocatedBytes,
        int Gen0Collections,
        int Gen1Collections,
        int Gen2Collections)
    {
        internal double TotalMilliseconds
            => ZeroGrad + Forward + Loss + Backward + Neko + Adam;
    }
}
