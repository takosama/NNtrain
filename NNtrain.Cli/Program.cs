using System.Diagnostics;

namespace NNtrain;

class Program
{
    static int Main(string[] args)
        => Run(args, Console.Out, Console.Error);

    internal static int Run(
        string[] args,
        TextWriter output,
        TextWriter error)
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

            config.Model.ValidateForModelWidth(trainData.Columns);
            var model = new TransformerClassifier(
                seqLen: trainData.Rows,
                dModel: trainData.Columns,
                numHeads: config.Model.Heads,
                dHidden: config.Model.HiddenSize,
                numLayers: config.Model.Layers,
                numClasses: trainData.ClassCount,
                rng: new Random(config.Model.Seed),
                initScale: config.Model.InitializationScale);
            var optimizer = new AdamW(
                model.Parameters(),
                new AdamWOptions
                {
                    LearningRate = config.LearningRate,
                });
            var random = new Random(config.Seed);
            int[] trainingOrder = Enumerable.Range(0, trainData.Count).ToArray();
            output.WriteLine(
                $"workers = {Environment.ProcessorCount}");
            output.WriteLine(
                $"simd = {GetSimdStatus()}");
            output.WriteLine(
                $"label smoothing = {config.LabelSmoothing:F3}");

            for (int epoch = 1; epoch <= config.Epochs; epoch++)
            {
                float trainLoss = 0f;
                int trainCorrect = 0;
                var trainTimer = Stopwatch.StartNew();
                Shuffle(trainingOrder, random);
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
                        samplesInBatch);
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
                            config.LabelSmoothing);
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
                output.WriteLine();
                output.WriteLine(
                    $"epoch {epoch}, " +
                    $"train loss = {trainLoss / trainData.Count:F6}, " +
                    $"train acc = {100f * trainCorrect / trainData.Count:F2}%, " +
                    $"eval loss = {evalLoss / evalData.Count:F6}, " +
                    $"eval acc = {100f * evalCorrectCount / evalData.Count:F2}%, " +
                    $"train time = {trainTimer.Elapsed.TotalSeconds:F2} sec, " +
                    $"eval time = {evalTimer.Elapsed.TotalSeconds:F2} sec");
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
            return new Cifar100(configuration.DataPath);
        }

        throw new ArgumentException(
            $"Unsupported {role.ToLowerInvariant()} dataset type " +
            $"'{configuration.Type}'.",
            nameof(configuration));
    }

    private static BatchSamples ReadBatch(
        IImageClassificationDataset dataset,
        int[]? order,
        int start,
        int count)
    {
        var inputValues = new float[count * dataset.ImageSize];
        var answers = new int[count];

        for (int offset = 0; offset < count; offset++)
        {
            int index = order is null ? start + offset : order[start + offset];
            int answer = dataset.ReadSample(
                index,
                inputValues.AsSpan(
                    offset * dataset.ImageSize,
                    dataset.ImageSize));
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
}
