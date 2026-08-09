using System.Diagnostics;
using System.Text.Json;

namespace NNtrain;

class Program
{
    static int Main(string[] args)
        => Run(args, Console.Out, Console.Error, openLossGraph: true);

    internal static int Run(
        string[] args,
        TextWriter output,
        TextWriter error,
        bool openLossGraph = false)
    {
        ArgumentNullException.ThrowIfNull(args);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(error);

        string configurationPath;
        if (args.Length == 0)
        {
            configurationPath = FindDefaultConfiguration();
        }
        else if (args.Length == 2
            && string.Equals(
                args[0],
                "--config",
                StringComparison.OrdinalIgnoreCase))
        {
            configurationPath = args[1];
        }
        else
        {
            error.WriteLine(
                "Usage: NNtrain.Cli [--config <training-config.json>]");
            return 1;
        }

        try
        {
            TrainingConfiguration config =
                TrainingConfiguration.Load(configurationPath);
            Tensor.SimdEnabled = config.UseSimd;
            IImageClassificationDataset trainData = CreateDataset(
                config.TrainingData,
                "Training");
            IImageClassificationDataset evalData = CreateDataset(
                config.EvaluationData,
                "Evaluation");

            ValidateDatasetCompatibility(trainData, evalData);
            config.Model.ValidateForModelWidth(trainData.Columns);
            var model = new TransformerClassifier(
                seqLen: trainData.Rows,
                dModel: trainData.Columns,
                numHeads: config.Model.Heads,
                dHidden: config.Model.HiddenSize,
                numLayers: config.Model.Layers,
                numClasses: trainData.ClassCount,
                rng: new Random(config.Model.Seed),
                initScale: config.Model.InitializationScale,
                dropout: config.Model.Dropout);
            IOptimizer optimizer = CreateOptimizer(
                model,
                config);
            LossGraph? lossGraph = null;
            if (config.ShowLossGraph)
            {
                string graphPath = Path.ChangeExtension(
                    Path.GetFullPath(configurationPath),
                    ".loss.html");
                lossGraph = new LossGraph(graphPath, config.Epochs);
                lossGraph.Write();
                output.WriteLine($"loss graph = {lossGraph.Path}");
                if (openLossGraph)
                    lossGraph.TryOpen(error);
            }
            var shuffleRandom = new Random(config.Seed);
            var augmentationRandom = new Random(
                config.Seed ^ 0x51F15EED);
            int[] trainingOrder = Enumerable.Range(0, trainData.Count).ToArray();
            output.WriteLine(
                $"workers = {Environment.ProcessorCount}");
            output.WriteLine(
                $"simd = {GetSimdStatus()}");
            output.WriteLine(
                $"optimizer = {config.Optimizer.ToLowerInvariant()}");
            output.WriteLine(
                $"learning rate = {config.LearningRate:F6}");
            if (config.IsOptimizer(
                TrainingConfiguration.NekoMuonOptimizer))
            {
                output.WriteLine(
                    "auxiliary optimizer = adamw");
                output.WriteLine(
                    "auxiliary learning rate = " +
                    $"{config.AuxiliaryLearningRate:F6}");
            }
            output.WriteLine(
                $"label smoothing = {config.LabelSmoothing:F3}");
            output.WriteLine(
                $"weight decay = {config.WeightDecay:F6}");
            if (config.IsOptimizer(
                TrainingConfiguration.GainShareAdamWOptimizer))
            {
                output.WriteLine(
                    $"gainshare = block depth " +
                    $"{config.GainShareBlockDepth}, " +
                    $"betas ({config.GainShareBeta1:F3}, " +
                    $"{config.GainShareBeta2:F3}), " +
                    $"eps {config.GainShareEpsilon:E1}, " +
                    $"rho {config.GainShareRho:F3}, " +
                    $"gamma {config.GainShareGamma:F3}, " +
                    $"scale [{config.GainShareMinScale:F3}, " +
                    $"{config.GainShareMaxScale:F3}]");
            }
            output.WriteLine(
                $"dropout = {config.Model.Dropout:F3}");
            output.WriteLine(
                $"learning-rate schedule = warmup {config.WarmupEpochs} " +
                $"epoch(s), cosine to " +
                $"{config.MinimumLearningRateRatio:P1}");
            output.WriteLine(
                config.EarlyStoppingPatience > 0
                    ? $"early stopping = patience " +
                        $"{config.EarlyStoppingPatience} epoch(s)"
                    : "early stopping = disabled");
            if (trainData is Cifar100)
            {
                var cifar100 = (Cifar100)trainData;
                Cifar100Options augmentation = cifar100.Options;
                output.WriteLine(
                    $"tokenization = {augmentation.PatchSize}x" +
                    $"{augmentation.PatchSize} patches, " +
                    $"{trainData.Rows} tokens x " +
                    $"{trainData.Columns} features");
                output.WriteLine(
                    $"normalization = " +
                    (augmentation.Normalize ? "cifar100" : "disabled"));
                output.WriteLine(
                    GetCifar100AugmentationDescription(augmentation));
            }

            ModuleState? bestModelState = null;
            float bestEvaluationLoss = float.PositiveInfinity;
            float earlyStoppingReferenceLoss = float.PositiveInfinity;
            int bestEpoch = 0;
            int epochsWithoutImprovement = 0;

            for (int epoch = 1; epoch <= config.Epochs; epoch++)
            {
                LearningRates learningRates = SetScheduledLearningRates(
                    optimizer,
                    config,
                    epoch);
                model.Train();
                float trainLoss = 0f;
                int trainCorrect = 0;
                var trainTimer = Stopwatch.StartNew();
                Shuffle(trainingOrder, shuffleRandom);
                int batchCount =
                    (trainData.Count + config.BatchSize - 1)
                    / config.BatchSize;

                for (int batch = 0; batch < batchCount; batch++)
                {
                    optimizer.ZeroGrad();
                    int batchStart = batch * config.BatchSize;
                    int samplesInBatch = Math.Min(
                        config.BatchSize,
                        trainData.Count - batchStart);
                    BatchSamples samples = ReadBatch(
                        trainData,
                        trainingOrder,
                        batchStart,
                        samplesInBatch,
                        augmentationRandom);
                    Tensor logits = model.ForwardBatch(samples.Input);
                    Tensor loss = logits.CrossEntropyWithLogits(
                        samples.Answers,
                        config.LabelSmoothing);
                    float batchLoss = loss.Data[0];

                    loss.Backward();
                    trainLoss += batchLoss * samplesInBatch;
                    trainCorrect += CountCorrect(
                        logits.Data,
                        samples.Answers,
                        trainData.ClassCount);

                    optimizer.Step();

                    output.WriteLine(
                        $"epoch {epoch}, " +
                        $"batch {batch + 1}/{batchCount}, " +
                        $"loss = {batchLoss:F6}");
                }

                trainTimer.Stop();

                float evalLoss = 0f;
                int evalCorrectCount = 0;
                int completedEvaluationSamples = 0;
                int lastEvalPercent = -1;
                var evalTimer = Stopwatch.StartNew();
                int evaluationBatchCount =
                    (evalData.Count + config.BatchSize - 1)
                    / config.BatchSize;

                model.Eval();
                using (AutogradContext.NoGrad())
                {
                    for (int batch = 0;
                        batch < evaluationBatchCount;
                        batch++)
                    {
                        int batchStart = batch * config.BatchSize;
                        int samplesInBatch = Math.Min(
                            config.BatchSize,
                            evalData.Count - batchStart);
                        BatchSamples samples = ReadBatch(
                            evalData,
                            order: null,
                            batchStart,
                            samplesInBatch);
                        Tensor logits = model.ForwardBatch(samples.Input);
                        Tensor loss = logits.CrossEntropyWithLogits(
                            samples.Answers,
                            labelSmoothing: 0f);
                        evalLoss += loss.Data[0] * samplesInBatch;
                        evalCorrectCount += CountCorrect(
                            logits.Data,
                            samples.Answers,
                            evalData.ClassCount);

                        completedEvaluationSamples += samplesInBatch;
                        int evalPercent =
                            completedEvaluationSamples
                            * 100
                            / evalData.Count;
                        if (evalPercent > lastEvalPercent)
                        {
                            output.Write(
                                $"\repoch {epoch}, " +
                                $"eval {evalPercent,3}%");
                            lastEvalPercent = evalPercent;
                        }
                    }
                }

                evalTimer.Stop();
                float averageTrainingLoss = trainLoss / trainData.Count;
                float averageEvaluationLoss = evalLoss / evalData.Count;
                if (bestModelState is null
                    || averageEvaluationLoss < bestEvaluationLoss)
                {
                    bestEvaluationLoss = averageEvaluationLoss;
                    bestEpoch = epoch;
                    bestModelState = model.CaptureState();
                }

                bool meaningfulImprovement =
                    averageEvaluationLoss
                        < earlyStoppingReferenceLoss
                            - config.EarlyStoppingMinimumDelta;
                if (meaningfulImprovement)
                {
                    earlyStoppingReferenceLoss = averageEvaluationLoss;
                    epochsWithoutImprovement = 0;
                }
                else
                {
                    epochsWithoutImprovement++;
                }

                lossGraph?.AddEpoch(
                    epoch,
                    averageTrainingLoss,
                    averageEvaluationLoss);
                lossGraph?.Write();
                output.WriteLine();
                output.WriteLine(
                    $"epoch {epoch}, " +
                    $"train loss = {averageTrainingLoss:F6}, " +
                    $"train acc = {100f * trainCorrect / trainData.Count:F2}%, " +
                    $"eval loss = {averageEvaluationLoss:F6}, " +
                    $"eval acc = {100f * evalCorrectCount / evalData.Count:F2}%, " +
                    $"lr = {learningRates.Primary:F8}, " +
                    $"train time = {trainTimer.Elapsed.TotalSeconds:F2} sec, " +
                    $"eval time = {evalTimer.Elapsed.TotalSeconds:F2} sec");

                if (config.EarlyStoppingPatience > 0
                    && epochsWithoutImprovement
                        >= config.EarlyStoppingPatience)
                {
                    output.WriteLine(
                        $"early stopping at epoch {epoch}: eval loss did " +
                        $"not improve for " +
                        $"{config.EarlyStoppingPatience} epoch(s).");
                    break;
                }
            }

            if (bestModelState is not null)
            {
                model.RestoreState(bestModelState);
                string checkpointPath = Path.ChangeExtension(
                    Path.GetFullPath(configurationPath),
                    ".best-model.json");
                SaveModelCheckpoint(
                    checkpointPath,
                    bestEpoch,
                    bestEvaluationLoss,
                    bestModelState);
                output.WriteLine(
                    $"best model = epoch {bestEpoch}, " +
                    $"eval loss {bestEvaluationLoss:F6}");
                output.WriteLine($"checkpoint = {checkpointPath}");
            }

            return 0;
        }
        catch (Exception exception) when (
            exception is IOException
            or System.Text.Json.JsonException
            or ArgumentException)
        {
            error.WriteLine($"Error: {exception.Message}");
            return 2;
        }
    }

    private static void Shuffle(int[] values, Random random)
    {
        for (int index = values.Length - 1; index > 0; index--)
        {
            int swapIndex = random.Next(index + 1);
            (values[index], values[swapIndex]) =
                (values[swapIndex], values[index]);
        }
    }

    private static string GetSimdStatus()
    {
        if (!Tensor.SimdEnabled)
            return "disabled (scalar)";

        return Tensor.IsSimdHardwareAccelerated
            ? "enabled (Vector256)"
            : "enabled, hardware unavailable (scalar)";
    }

    internal static IOptimizer CreateOptimizer(
        TransformerClassifier model,
        TrainingConfiguration configuration)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(configuration);

        if (configuration.IsOptimizer(
            TrainingConfiguration.GainShareAdamWOptimizer))
        {
            return new GainShareAdamW(
                model.MakeGainShareParameterGroups(
                    configuration.GainShareBlockDepth),
                new GainShareAdamWOptions
                {
                    LearningRate = configuration.LearningRate,
                    Beta1 = configuration.GainShareBeta1,
                    Beta2 = configuration.GainShareBeta2,
                    Epsilon = configuration.GainShareEpsilon,
                    Rho = configuration.GainShareRho,
                    Gamma = configuration.GainShareGamma,
                    MinScale = configuration.GainShareMinScale,
                    MaxScale = configuration.GainShareMaxScale,
                    WeightDecay = configuration.WeightDecay,
                });
        }

        if (configuration.IsOptimizer(
            TrainingConfiguration.NekoMuonOptimizer))
        {
            var nekoMuon = new NekoMuon(
                model.HiddenWeightParameters,
                new NekoMuonOptions
                {
                    LearningRate = configuration.LearningRate,
                    WeightDecay = configuration.WeightDecay,
                });
            var auxiliaryAdamW = new AdamW(
                model.AuxiliaryParameters,
                new AdamWOptions
                {
                    LearningRate = configuration.AuxiliaryLearningRate,
                    Beta1 = 0.9f,
                    Beta2 = 0.95f,
                    Epsilon = 1e-8f,
                    WeightDecay = configuration.WeightDecay,
                });
            return new CompositeOptimizer(nekoMuon, auxiliaryAdamW);
        }

        if (configuration.IsOptimizer(TrainingConfiguration.LionOptimizer))
        {
            return new Lion(
                model.Parameters(),
                new LionOptions
                {
                    LearningRate = configuration.LearningRate,
                    WeightDecay = configuration.WeightDecay,
                });
        }

        return new AdamW(
            model.Parameters(),
            new AdamWOptions
            {
                LearningRate = configuration.LearningRate,
                WeightDecay = configuration.WeightDecay,
            });
    }

    internal static float CalculateLearningRateFactor(
        int epoch,
        int totalEpochs,
        int warmupEpochs,
        float minimumRatio)
    {
        if (epoch <= 0 || epoch > totalEpochs)
            throw new ArgumentOutOfRangeException(nameof(epoch));
        if (totalEpochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(totalEpochs));
        if (warmupEpochs < 0 || warmupEpochs >= totalEpochs)
            throw new ArgumentOutOfRangeException(nameof(warmupEpochs));
        if (!float.IsFinite(minimumRatio)
            || minimumRatio <= 0f
            || minimumRatio > 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(minimumRatio));
        }

        if (warmupEpochs > 0 && epoch <= warmupEpochs)
            return (float)epoch / warmupEpochs;

        int decayEpochs = totalEpochs - warmupEpochs;
        float progress = (float)(epoch - warmupEpochs) / decayEpochs;
        float cosine = 0.5f * (1f + MathF.Cos(MathF.PI * progress));
        return minimumRatio + (1f - minimumRatio) * cosine;
    }

    internal static LearningRates SetScheduledLearningRates(
        IOptimizer optimizer,
        TrainingConfiguration configuration,
        int epoch)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(configuration);

        float factor = CalculateLearningRateFactor(
            epoch,
            configuration.Epochs,
            configuration.WarmupEpochs,
            configuration.MinimumLearningRateRatio);
        float primary = configuration.LearningRate * factor;

        if (optimizer is CompositeOptimizer composite)
        {
            if (composite.Optimizers.Count != 2
                || composite.Optimizers[0]
                    is not ILearningRateAdjustable primaryOptimizer
                || composite.Optimizers[1]
                    is not ILearningRateAdjustable auxiliaryOptimizer)
            {
                throw new InvalidOperationException(
                    "The configured composite optimizer does not expose " +
                    "the expected learning-rate groups.");
            }

            float auxiliary = configuration.AuxiliaryLearningRate * factor;
            primaryOptimizer.SetLearningRate(primary);
            auxiliaryOptimizer.SetLearningRate(auxiliary);
            return new LearningRates(primary, auxiliary);
        }

        if (optimizer is not ILearningRateAdjustable adjustable)
        {
            throw new InvalidOperationException(
                $"Optimizer '{optimizer.GetType().Name}' does not support " +
                "learning-rate scheduling.");
        }

        adjustable.SetLearningRate(primary);
        return new LearningRates(primary, null);
    }

    private static void SaveModelCheckpoint(
        string path,
        int epoch,
        float evaluationLoss,
        ModuleState modelState)
    {
        var checkpoint = new ModelCheckpoint(
            ModelCheckpoint.CurrentFormatVersion,
            epoch,
            evaluationLoss,
            modelState);
        string temporaryPath = path + ".tmp";
        string json = JsonSerializer.Serialize(checkpoint);
        File.WriteAllText(temporaryPath, json);
        File.Move(temporaryPath, path, overwrite: true);
    }

    private static string FindDefaultConfiguration()
    {
        const string fileName = "training.example.json";

        foreach (string startPath in new[]
        {
            Environment.CurrentDirectory,
            AppContext.BaseDirectory,
        })
        {
            DirectoryInfo? directory = new DirectoryInfo(startPath);
            while (directory is not null)
            {
                string candidate = Path.Combine(directory.FullName, fileName);
                if (File.Exists(candidate))
                    return candidate;

                directory = directory.Parent;
            }
        }

        return Path.Combine(Environment.CurrentDirectory, fileName);
    }

    private static IImageClassificationDataset CreateDataset(
        DatasetConfiguration configuration,
        string role)
    {
        if (configuration.IsType(DatasetConfiguration.MnistType))
        {
            EnsureDataFile(configuration.ImagePath, $"{role} image");
            EnsureDataFile(configuration.LabelPath, $"{role} label");
            return new Mnist(
                configuration.ImagePath,
                configuration.LabelPath);
        }

        if (configuration.IsType(DatasetConfiguration.Cifar100Type))
        {
            EnsureDataFile(configuration.DataPath, $"{role} CIFAR-100");
            return new Cifar100(
                configuration.DataPath,
                new Cifar100Options
                {
                    PatchSize = configuration.PatchSize,
                    Normalize = configuration.Normalize,
                    RandomCropPadding =
                        configuration.Augmentation.RandomCropPadding,
                    HorizontalFlip =
                        configuration.Augmentation.HorizontalFlip,
                    VerticalFlip =
                        configuration.Augmentation.VerticalFlip,
                });
        }

        throw new ArgumentException(
            $"Unsupported {role.ToLowerInvariant()} dataset type " +
            $"'{configuration.Type}'.",
            nameof(configuration));
    }

    private static void ValidateDatasetCompatibility(
        IImageClassificationDataset training,
        IImageClassificationDataset evaluation)
    {
        if (training.Rows != evaluation.Rows
            || training.Columns != evaluation.Columns)
        {
            throw new ArgumentException(
                $"Training input shape '{training.Rows}x" +
                $"{training.Columns}' does not match evaluation input " +
                $"shape '{evaluation.Rows}x{evaluation.Columns}'.");
        }

        if (training.ClassCount != evaluation.ClassCount)
        {
            throw new ArgumentException(
                $"Training class count '{training.ClassCount}' does not " +
                $"match evaluation class count " +
                $"'{evaluation.ClassCount}'.");
        }
    }

    private static string GetCifar100AugmentationDescription(
        Cifar100Options options)
    {
        var operations = new List<string>();
        if (options.RandomCropPadding > 0)
        {
            operations.Add(
                $"random crop (padding {options.RandomCropPadding})");
        }
        if (options.HorizontalFlip)
            operations.Add("horizontal flip");
        if (options.VerticalFlip)
            operations.Add("vertical flip");

        return operations.Count == 0
            ? "augmentation = disabled"
            : $"augmentation = {string.Join(", ", operations)}";
    }

    private static BatchSamples ReadBatch(
        IImageClassificationDataset dataset,
        int[]? order,
        int start,
        int count,
        Random? trainingRandom = null)
    {
        var inputValues = new float[count * dataset.ImageSize];
        var answers = new int[count];

        for (int offset = 0; offset < count; offset++)
        {
            int index = order is null ? start + offset : order[start + offset];
            Span<float> destination = inputValues.AsSpan(
                offset * dataset.ImageSize,
                dataset.ImageSize);
            int answer = trainingRandom is null
                ? dataset.ReadSample(index, destination)
                : dataset.ReadTrainingSample(
                    index,
                    destination,
                    trainingRandom);
            answers[offset] = answer;
        }

        return new BatchSamples(
            Tensor.FromOwnedData(
                inputValues,
                [count, dataset.Rows, dataset.Columns],
                "classifierInput"),
            answers);
    }

    private static int CountCorrect(
        IReadOnlyList<float> logits,
        IReadOnlyList<int> answers,
        int classCount)
    {
        int correct = 0;
        for (int sample = 0; sample < answers.Count; sample++)
        {
            if (ArgMax(logits, sample * classCount, classCount)
                == answers[sample])
            {
                correct++;
            }
        }

        return correct;
    }

    private static int ArgMax(
        IReadOnlyList<float> values,
        int offset,
        int count)
    {
        int result = 0;
        float best = values[offset];
        for (int index = 1; index < count; index++)
        {
            if (values[offset + index] > best)
            {
                best = values[offset + index];
                result = index;
            }
        }

        return result;
    }

    private static void EnsureDataFile(string path, string role)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                $"{role} data file was not found at '{path}'. " +
                "Check the corresponding path in the training " +
                "configuration.",
                path);
        }
    }

    private readonly record struct BatchSamples(
        Tensor Input,
        int[] Answers);

    internal readonly record struct LearningRates(
        float Primary,
        float? Auxiliary);

    private sealed record ModelCheckpoint(
        int FormatVersion,
        int Epoch,
        float EvaluationLoss,
        ModuleState Model)
    {
        internal const int CurrentFormatVersion = 1;
    }
}
