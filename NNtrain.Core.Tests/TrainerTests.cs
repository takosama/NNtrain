using NNtrain;
using Xunit;

public sealed class TrainerTests
{
    [Fact]
    public void RunTrainsAndEvaluatesEveryEpochThroughItsContracts()
    {
        var model = new FakeModel();
        var trainingDataset = new FakeDataset();
        var evaluationDataset = new FakeDataset();
        var optimizer = new CountingOptimizer();
        var trainer = new Trainer(
            model,
            trainingDataset,
            evaluationDataset,
            optimizer,
            new TrainerOptions
            {
                Epochs = 2,
                StepsPerEpoch = 3,
                RandomSeed = 17,
            });
        var reported = new List<TrainingEpochResult>();
        var reportedBatches = new List<TrainingBatchResult>();

        IReadOnlyList<TrainingEpochResult> results =
            trainer.Run(reported.Add, reportedBatches.Add);

        Assert.Equal(6, trainingDataset.ReadCount);
        Assert.Equal(2, evaluationDataset.ReadCount);
        Assert.Equal(8, model.ForwardCount);
        Assert.Equal(6, optimizer.ZeroGradCount);
        Assert.Equal(6, optimizer.StepCount);
        Assert.Equal(
            [true, true, true, false, true, true, true, false],
            model.RecordingStates);
        Assert.Equal(
            [
                "ZeroGrad", "Step",
                "ZeroGrad", "Step",
                "ZeroGrad", "Step",
                "ZeroGrad", "Step",
                "ZeroGrad", "Step",
                "ZeroGrad", "Step",
            ],
            optimizer.Operations);
        Assert.Equal(results, reported);
        Assert.Collection(
            reportedBatches,
            result => AssertBatch(result, 1, 1),
            result => AssertBatch(result, 1, 2),
            result => AssertBatch(result, 1, 3),
            result => AssertBatch(result, 2, 1),
            result => AssertBatch(result, 2, 2),
            result => AssertBatch(result, 2, 3));
        Assert.Collection(
            results,
            result => AssertEpoch(result, expectedEpoch: 1),
            result => AssertEpoch(result, expectedEpoch: 2));
    }

    private static void AssertBatch(
        TrainingBatchResult result,
        int expectedEpoch,
        int expectedBatch)
    {
        Assert.Equal(expectedEpoch, result.Epoch);
        Assert.Equal(expectedBatch, result.Batch);
        Assert.Equal(3, result.TotalBatches);
        Assert.InRange(result.Loss, 0.3132f, 0.3133f);
        Assert.True(result.IsCorrect);
    }

    [Theory]
    [InlineData(0, 1, "Epochs")]
    [InlineData(1, 0, "StepsPerEpoch")]
    public void RejectsNonPositiveTrainingCounts(
        int epochs,
        int stepsPerEpoch,
        string parameterName)
    {
        var options = new TrainerOptions
        {
            Epochs = epochs,
            StepsPerEpoch = stepsPerEpoch,
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            () => new Trainer(
                new FakeModel(),
                new FakeDataset(),
                new FakeDataset(),
                new CountingOptimizer(),
                options));

        Assert.Equal(parameterName, exception.ParamName);
    }

    [Fact]
    public void RejectsDatasetAndModelShapeMismatch()
    {
        var exception = Assert.Throws<ArgumentException>(
            () => new Trainer(
                new FakeModel(inputRows: 2),
                new FakeDataset(),
                new FakeDataset(),
                new CountingOptimizer()));

        Assert.Contains("does not match model input shape", exception.Message);
    }

    [Fact]
    public void RejectsDatasetAndModelClassCountMismatch()
    {
        var exception = Assert.Throws<ArgumentException>(
            () => new Trainer(
                new FakeModel(classCount: 3),
                new FakeDataset(),
                new FakeDataset(),
                new CountingOptimizer()));

        Assert.Contains("does not match model class count", exception.Message);
    }

    private static void AssertEpoch(
        TrainingEpochResult result,
        int expectedEpoch)
    {
        Assert.Equal(expectedEpoch, result.Epoch);
        Assert.Equal(3, result.TrainingSteps);
        Assert.Equal(1, result.EvaluationSamples);
        Assert.InRange(result.Training.Loss, 0.3132f, 0.3133f);
        Assert.Equal(1f, result.Training.Accuracy);
        Assert.True(result.Training.Elapsed >= TimeSpan.Zero);
        Assert.InRange(result.Evaluation.Loss, 0.3132f, 0.3133f);
        Assert.Equal(1f, result.Evaluation.Accuracy);
        Assert.True(result.Evaluation.Elapsed >= TimeSpan.Zero);
    }

    private sealed class FakeDataset : IImageClassificationDataset
    {
        public int Count => 1;
        public int Rows => 1;
        public int Columns => 1;
        public int ImageSize => 1;
        public int ClassCount => 2;
        public int ReadCount { get; private set; }

        public int ReadSample(int index, Span<float> destination)
        {
            Assert.Equal(0, index);
            destination[0] = 0.5f;
            ReadCount++;
            return 0;
        }
    }

    private sealed class FakeModel : IClassificationModel
    {
        internal FakeModel(int inputRows = 1, int classCount = 2)
        {
            InputRows = inputRows;
            ClassCount = classCount;
        }

        public int InputRows { get; }
        public int InputColumns => 1;
        public int ClassCount { get; }
        public int ForwardCount { get; private set; }
        public List<bool> RecordingStates { get; } = [];

        public Tensor Forward(Tensor input)
        {
            ForwardCount++;
            RecordingStates.Add(AutogradContext.IsRecordingEnabled);
            return Tensor.From1D([1f, 0f], "fakeLogits");
        }
    }

    private sealed class CountingOptimizer : IOptimizer
    {
        public int ZeroGradCount { get; private set; }
        public int StepCount { get; private set; }
        public List<string> Operations { get; } = [];

        public void ZeroGrad()
        {
            ZeroGradCount++;
            Operations.Add("ZeroGrad");
        }

        public void Step()
        {
            StepCount++;
            Operations.Add("Step");
        }
    }
}
