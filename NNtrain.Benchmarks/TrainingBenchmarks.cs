using BenchmarkDotNet.Attributes;

namespace NNtrain.Benchmarks;

/// <summary>
/// Measures one deterministic training epoch plus evaluation. The workload
/// intentionally uses an in-memory MNIST-shaped dataset so storage performance
/// does not obscure Tensor, autograd, module, optimizer, and Trainer costs.
/// </summary>
public class TrainingBenchmarks
{
    private const int ClassCount = 10;
    private const int ModelSeed = 3;
    private const int TrainingSeed = 7;
    private const float LearningRate = 0.001f;
    private const float InitializationScale = 0.01f;

    private Trainer _trainer = null!;

    [Params(28)]
    public int InputRows { get; set; }

    [Params(28)]
    public int InputColumns { get; set; }

    [Params(1)]
    public int AttentionHeads { get; set; }

    [Params(4)]
    public int HiddenSize { get; set; }

    [Params(1)]
    public int Layers { get; set; }

    [Params(192)]
    public int TrainingSteps { get; set; }

    [Params(32)]
    public int EvaluationSamples { get; set; }

    [IterationSetup]
    public void CreateTrainingRun()
    {
        var trainingDataset = new DeterministicImageDataset(
            count: 32,
            InputRows,
            InputColumns,
            ClassCount);
        var evaluationDataset = new DeterministicImageDataset(
            EvaluationSamples,
            InputRows,
            InputColumns,
            ClassCount);
        var model = new TransformerClassifier(
            seqLen: InputRows,
            dModel: InputColumns,
            numHeads: AttentionHeads,
            dHidden: HiddenSize,
            numLayers: Layers,
            numClasses: ClassCount,
            rng: new Random(ModelSeed),
            initScale: InitializationScale);
        var optimizer = new AdamW(
            model.Parameters(),
            new AdamWOptions
            {
                LearningRate = LearningRate,
            });

        _trainer = new Trainer(
            model,
            trainingDataset,
            evaluationDataset,
            optimizer,
            new TrainerOptions
            {
                Epochs = 1,
                StepsPerEpoch = TrainingSteps,
                RandomSeed = TrainingSeed,
            });
    }

    [Benchmark(Description = "Train and evaluate one epoch")]
    [BenchmarkCategory("Training")]
    public TrainingEpochResult TrainAndEvaluateOneEpoch()
    {
        return _trainer.Run()[0];
    }

    private sealed class DeterministicImageDataset
        : IImageClassificationDataset
    {
        internal DeterministicImageDataset(
            int count,
            int rows,
            int columns,
            int classCount)
        {
            Count = count;
            Rows = rows;
            Columns = columns;
            ClassCount = classCount;
        }

        public int Count { get; }
        public int Rows { get; }
        public int Columns { get; }
        public int ImageSize => Rows * Columns;
        public int ClassCount { get; }

        public int ReadSample(int index, Span<float> destination)
        {
            if ((uint)index >= (uint)Count)
                throw new ArgumentOutOfRangeException(nameof(index));

            if (destination.Length < ImageSize)
                throw new ArgumentException(
                    "The destination is shorter than one image.",
                    nameof(destination));

            for (int pixel = 0; pixel < ImageSize; pixel++)
            {
                int value = (index * 17 + pixel * 31) & byte.MaxValue;
                destination[pixel] = value / 255f;
            }

            return index % ClassCount;
        }
    }
}
