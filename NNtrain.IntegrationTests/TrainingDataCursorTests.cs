using NNtrain;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Execution;
using Xunit;

public sealed class TrainingDataCursorTests
{
    [Fact]
    public void CursorOperationsAcquireBeforeTaskPhasesWithoutDelegates()
    {
        using var execution = new ExecutionSession(new ExecutionOptions());
        using var session = new TrainingSession(execution);
        var executor = new TrainingStepExecutor(session);
        var cursor = new SequenceCursor([17, 23]);
        var adapter = new RecordingAdapter();
        var operations = new CursorTrainingStepOperations<int>(
            cursor,
            adapter);

        executor.Execute(operations);
        executor.Execute(operations);

        Assert.Equal(2, cursor.Position);
        Assert.Equal([17, 23], adapter.Accepted);
        Assert.Equal(
            [
                "accept:17", "zero", "forward", "backward", "reduce",
                "clip", "schedule", "optimizer", "metrics",
                "accept:23", "zero", "forward", "backward", "reduce",
                "clip", "schedule", "optimizer", "metrics",
            ],
            adapter.Phases);
        Assert.Equal(1, adapter.PrepareCalls);
    }

    [Fact]
    public void FixedWikiCursorResumesAtCompletedBatchInExistingOrder()
    {
        int[] tokens = [10, 11, 12, 13, 14, 15, 16];
        int[] order = [1, 0, 2];
        var cursor = new WikiLanguageModelCommand.FixedWikiTrainingDataCursor(
            tokens,
            order,
            batchSize: 2,
            sequenceLength: 2);

        cursor.StartEpoch(completedBatches: 1);
        WikiLanguageModelCommand.WikiTrainingBatch resumed =
            cursor.AcquireNext();

        Assert.Equal(2, cursor.Position);
        Assert.Equal(1, resumed.BatchIndex);
        Assert.Equal(1, resumed.BatchSize);
        Assert.Equal([14, 15], resumed.Values.Input);
        Assert.Equal([15, 16], resumed.Values.Target);
        Assert.Equal(2, resumed.Values.ValidTargetCount);
        Assert.Throws<InvalidOperationException>(
            () => cursor.AcquireNext());
    }

    [Fact]
    public void FixedWikiUpdateCursorAccumulatesAndKeepsTailShapeSeparate()
    {
        int[] tokens = Enumerable.Range(0, 11).ToArray();
        int[] order = [0, 1, 2, 3, 4];
        var microBatches =
            new WikiLanguageModelCommand.FixedWikiTrainingDataCursor(
                tokens,
                order,
                batchSize: 2,
                sequenceLength: 2);
        var updates =
            new WikiLanguageModelCommand.FixedWikiTrainingUpdateCursor(
                microBatches,
                accumulationSteps: 2);
        updates.StartEpoch(completedBatches: 0);

        WikiLanguageModelCommand.WikiTrainingUpdate first =
            updates.AcquireNext();
        WikiLanguageModelCommand.WikiTrainingUpdate tail =
            updates.AcquireNext();

        Assert.Equal(2, first.Count);
        Assert.All(
            first.MicroBatches,
            batch => Assert.Equal(2, batch.BatchSize));
        Assert.Equal(1, tail.Count);
        Assert.Equal(1, tail.Last.BatchSize);
        Assert.Equal(3, updates.Position);
    }

    [Fact]
    public void StreamingWikiCursorKeepsRestoredBufferAndBatchPosition()
    {
        var buffer = new List<int> { 3, 5, 7, 11, 13 };
        var cursor =
            new WikiLanguageModelCommand.StreamingWikiTrainingDataCursor(
                buffer);
        cursor.StartEpoch(completedBatches: 7);
        cursor.ConfigureNext(
            batchSize: 1,
            sequenceLength: 2,
            documentsProcessed: 41);

        WikiLanguageModelCommand.WikiTrainingBatch resumed =
            cursor.AcquireNext();

        Assert.Equal(8, cursor.Position);
        Assert.Equal(7, resumed.BatchIndex);
        Assert.Equal(41, resumed.DocumentsProcessed);
        Assert.Equal([3, 5], resumed.Values.Input);
        Assert.Equal([5, 7], resumed.Values.Target);
        Assert.Equal([7, 11, 13], buffer);
        Assert.Throws<InvalidOperationException>(
            () => cursor.AcquireNext());
    }

    [Fact]
    public void ClassificationCursorReplaysSkippedMicrobatchesInOrder()
    {
        DataBatch[] source = Enumerable.Range(0, 5)
            .Select(index => new DataBatch(
                torch.tensor([(float)index], [1, 1]),
                [index]))
            .ToArray();
        using IEnumerator<DataBatch> batches =
            ((IEnumerable<DataBatch>)source).GetEnumerator();
        var cursor = new Program.ClassificationTrainingDataCursor();
        cursor.StartEpoch(batches, microBatchesToSkip: 2);
        cursor.ConfigureNext(
            update: 1,
            updateTotal: 3,
            firstMicroBatch: 2,
            microBatchCount: 2,
            microBatchTotal: 5,
            sampleCount: 2);

        Program.ClassificationUpdateBatch resumed = cursor.AcquireNext();
        DataBatch first = resumed.AcquireMicroBatch(0);
        DataBatch second = resumed.AcquireMicroBatch(1);

        Assert.Equal(4, cursor.Position);
        Assert.Equal(1, resumed.Update);
        Assert.Equal(2, resumed.FirstMicroBatch);
        Assert.Equal(2, first.Target[0]);
        Assert.Equal(3, second.Target[0]);
        Assert.Throws<InvalidOperationException>(
            () => resumed.AcquireMicroBatch(1));

        cursor.ConfigureNext(
            update: 2,
            updateTotal: 3,
            firstMicroBatch: 4,
            microBatchCount: 1,
            microBatchTotal: 5,
            sampleCount: 1);
        Program.ClassificationUpdateBatch final = cursor.AcquireNext();

        Assert.Equal(4, final.AcquireMicroBatch(0).Target[0]);
        Assert.Equal(5, cursor.Position);
    }

    private sealed class SequenceCursor(int[] values)
        : ITrainingDataCursor<int>
    {
        private int _position;

        public long Position => _position;

        public int AcquireNext()
        {
            if (_position == values.Length)
                throw new InvalidOperationException("end");
            return values[_position++];
        }
    }

    private sealed class RecordingAdapter : ITrainingTaskAdapter<int>
    {
        internal List<int> Accepted { get; } = [];
        internal List<string> Phases { get; } = [];
        internal int PrepareCalls { get; private set; }

        public TrainingGradientExecutionMode GradientExecutionMode
            => TrainingGradientExecutionMode.Separate;

        public void Prepare() => PrepareCalls++;

        public void AcceptBatch(int batch)
        {
            Accepted.Add(batch);
            Phases.Add($"accept:{batch}");
        }

        public void ClearGradients() => Phases.Add("zero");

        public void Forward() => Phases.Add("forward");

        public void Backward() => Phases.Add("backward");

        public void ReduceGradients() => Phases.Add("reduce");

        public void ForwardBackwardReduced()
            => throw new InvalidOperationException("not fused");

        public void ClipGradients() => Phases.Add("clip");

        public void ApplySchedule() => Phases.Add("schedule");

        public void CommitOptimizer() => Phases.Add("optimizer");

        public void CommitMetrics() => Phases.Add("metrics");
    }
}
