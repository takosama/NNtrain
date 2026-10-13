using NNtrain.Training.Execution;
using NNtrain.Training.Optimization;

namespace NNtrain;

/// <summary>Retains the existing internal entrypoints while commands own their implementations.</summary>
internal static partial class Program
{
    internal readonly record struct ClassificationUpdateBatch
    {
        private readonly ClassificationTrainingDataCursor _cursor;
        private readonly int _generation;

        internal ClassificationUpdateBatch(
            ClassificationTrainingDataCursor cursor,
            int generation,
            int update,
            int updateTotal,
            int firstMicroBatch,
            int microBatchCount,
            int microBatchTotal,
            int sampleCount)
        {
            _cursor = cursor;
            _generation = generation;
            Update = update;
            UpdateTotal = updateTotal;
            FirstMicroBatch = firstMicroBatch;
            MicroBatchCount = microBatchCount;
            MicroBatchTotal = microBatchTotal;
            SampleCount = sampleCount;
        }

        internal int Update { get; }

        internal int UpdateTotal { get; }

        internal int FirstMicroBatch { get; }

        internal int MicroBatchCount { get; }

        internal int MicroBatchTotal { get; }

        internal int SampleCount { get; }

        internal DataBatch AcquireMicroBatch(int accumulationIndex)
            => _cursor.AcquireMicroBatch(
                _generation,
                accumulationIndex);
    }

    internal sealed class ClassificationTrainingDataCursor
        : ITrainingDataCursor<ClassificationUpdateBatch>
    {
        private readonly global::NNtrain.ClassificationTrainingDataCursor _cursor = new();

        public long Position => _cursor.Position;

        internal void StartEpoch(
            IEnumerator<DataBatch> batches,
            int microBatchesToSkip)
            => _cursor.StartEpoch(batches, microBatchesToSkip);

        internal void ConfigureNext(
            int update,
            int updateTotal,
            int firstMicroBatch,
            int microBatchCount,
            int microBatchTotal,
            int sampleCount)
            => _cursor.ConfigureNext(
                update,
                updateTotal,
                firstMicroBatch,
                microBatchCount,
                microBatchTotal,
                sampleCount);

        public ClassificationUpdateBatch AcquireNext()
        {
            global::NNtrain.ClassificationUpdateBatch batch = _cursor.AcquireNext();
            return new ClassificationUpdateBatch(
                this,
                batch.Generation,
                batch.Update,
                batch.UpdateTotal,
                batch.FirstMicroBatch,
                batch.MicroBatchCount,
                batch.MicroBatchTotal,
                batch.SampleCount);
        }

        internal DataBatch AcquireMicroBatch(
            int generation,
            int accumulationIndex)
            => _cursor.AcquireMicroBatch(generation, accumulationIndex);
    }

    internal static bool ResolveAutomaticResume(
        bool explicitResume,
        bool autoResume,
        TrainingRunGuard run,
        string checkpointPath,
        TextWriter output)
        => TrainingResumePolicy.Resolve(
            explicitResume,
            autoResume,
            run,
            checkpointPath,
            output);

    internal static IOptimizer CreateOptimizer(
        TransformerClassifier model,
        TrainingConfiguration configuration)
        => CreateOptimizerBundle(model, configuration).RootOptimizer;

    internal static OptimizerBundle CreateOptimizerBundle(
        TransformerClassifier model,
        TrainingConfiguration configuration)
        => ClassificationOptimizerFactory.CreateBundle(model, configuration);

    internal static void ApplyClassificationMuonPolicyAfterResume(
        TrainingConfiguration configuration,
        IOptimizer optimizer,
        TextWriter output)
        => ClassificationOptimizerFactory.ApplyMuonPolicyAfterResume(
            configuration,
            optimizer,
            output);

    internal static string GetBestModelCheckpointPath(
        string trainingCheckpointPath)
        => ClassificationModelCheckpoint.GetBestModelCheckpointPath(trainingCheckpointPath);

    internal static string FindDefaultConfiguration()
        => TrainingCommandArguments.FindDefaultConfiguration();

    internal static int CountCorrectForMetrics(
        Tensor logits,
        int[] answers,
        int classCount)
        => ClassificationMetrics.CountCorrectForMetrics(logits, answers, classCount);

    internal readonly record struct LearningRates(
        float Primary,
        float? Auxiliary);
}
