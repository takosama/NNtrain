using NNtrain.Training.Execution;

namespace NNtrain;

internal static partial class Program
{
    /// <summary>
    /// Treats one gradient-accumulation update as one transaction. AcquireBatch
    /// acquires only the first existing DataBatch; the fused phase consumes it
    /// and then advances the iterator one microbatch at a time, preserving the
    /// RNG/augmentation order without buffering an entire update.
    /// </summary>
    private sealed class ClassificationTrainingStepOperations
        : ITrainingStepOperations
    {
        private readonly TransformerClassifier _model;
        private readonly IOptimizer _optimizer;
        private readonly ILRScheduler _scheduler;
        private readonly TextWriter _output;
        private readonly float _labelSmoothing;
        private readonly int _classCount;
        private IEnumerator<DataBatch>? _trainingBatches;
        private int _epoch;
        private int _update;
        private int _updateTotal;
        private int _firstMicroBatch;
        private int _microBatchesInUpdate;
        private int _microBatchTotal;
        private int _samplesInUpdate;
        private double _pendingLoss;
        private int _pendingCorrect;
        private int _pendingSamples;
        private bool _schedulePending;
        private DataBatch? _firstBatch;

        internal ClassificationTrainingStepOperations(
            TransformerClassifier model,
            IOptimizer optimizer,
            ILRScheduler scheduler,
            TextWriter output,
            float labelSmoothing,
            int classCount,
            long globalStep)
        {
            _model = model;
            _optimizer = optimizer;
            _scheduler = scheduler;
            _output = output;
            _labelSmoothing = labelSmoothing;
            _classCount = classCount;
            GlobalStep = globalStep;
        }

        public TrainingGradientExecutionMode GradientExecutionMode
            => TrainingGradientExecutionMode.FusedForwardBackwardReduced;

        internal long GlobalStep { get; private set; }

        internal double TrainingLossSum { get; private set; }

        internal int TrainingCorrect { get; private set; }

        internal int TrainingSamples { get; private set; }

        internal IReadOnlyList<float> LearningRates { get; private set; }
            = [];

        internal void StartEpoch(
            int epoch,
            IEnumerator<DataBatch> trainingBatches,
            double trainingLossSum,
            int trainingCorrect,
            int trainingSamples,
            bool advanceSchedule)
        {
            _epoch = epoch;
            _trainingBatches = trainingBatches;
            TrainingLossSum = trainingLossSum;
            TrainingCorrect = trainingCorrect;
            TrainingSamples = trainingSamples;
            _schedulePending = advanceSchedule;
            LearningRates = _scheduler.get_last_lr();
        }

        internal void ConfigureUpdate(
            int update,
            int updateTotal,
            int firstMicroBatch,
            int microBatchesInUpdate,
            int microBatchTotal,
            int samplesInUpdate)
        {
            _update = update;
            _updateTotal = updateTotal;
            _firstMicroBatch = firstMicroBatch;
            _microBatchesInUpdate = microBatchesInUpdate;
            _microBatchTotal = microBatchTotal;
            _samplesInUpdate = samplesInUpdate;
        }

        public void AcquireBatch()
        {
            if (_trainingBatches is null)
            {
                throw new InvalidOperationException(
                    "The classification epoch has no training batch cursor.");
            }
            if (_microBatchesInUpdate <= 0 || _samplesInUpdate <= 0)
            {
                throw new InvalidOperationException(
                    "The classification update has no samples.");
            }
            _pendingLoss = 0d;
            _pendingCorrect = 0;
            _pendingSamples = 0;
            if (!_trainingBatches.MoveNext())
            {
                throw new InvalidOperationException(
                    "DataLoader ended before the expected training " +
                    "microbatch count.");
            }
            _firstBatch = _trainingBatches.Current;
        }

        public void ClearGradients() => _optimizer.zero_grad();

        public void Forward()
            => throw new InvalidOperationException(
                "Classification accumulation uses the fused phase.");

        public void Backward()
            => throw new InvalidOperationException(
                "Classification accumulation uses the fused phase.");

        public void ReduceGradients()
            => throw new InvalidOperationException(
                "Classification accumulation uses the fused phase.");

        public void ForwardBackwardReduced()
        {
            IEnumerator<DataBatch> trainingBatches = _trainingBatches!;
            for (int accumulation = 0;
                accumulation < _microBatchesInUpdate;
                accumulation++)
            {
                int microBatch = _firstMicroBatch + accumulation;
                DataBatch samples;
                if (accumulation == 0)
                {
                    samples = _firstBatch
                        ?? throw new InvalidOperationException(
                            "AcquireBatch did not retain the first microbatch.");
                }
                else if (trainingBatches.MoveNext())
                {
                    samples = trainingBatches.Current;
                }
                else
                {
                    throw new InvalidOperationException(
                        "DataLoader ended before the expected training " +
                        "microbatch count.");
                }
                int samplesInMicroBatch = samples.target.Length;
                Tensor logits = _model.forward(samples.input);
                Tensor loss = nn.functional.cross_entropy(
                    logits,
                    samples.target,
                    label_smoothing: _labelSmoothing);
                float microBatchLoss = loss.item();
                float gradientWeight =
                    (float)samplesInMicroBatch / _samplesInUpdate;

                loss.backward([gradientWeight]);
                _pendingLoss += microBatchLoss * samplesInMicroBatch;
                _pendingSamples += samplesInMicroBatch;
                _pendingCorrect += CountCorrect(
                    logits.Data,
                    samples.target,
                    _classCount);

                _output.WriteLine(
                    $"epoch {_epoch}, " +
                    $"microbatch {microBatch + 1}/" +
                    $"{_microBatchTotal}, accumulation " +
                    $"{accumulation + 1}/{_microBatchesInUpdate}, " +
                    $"update {_update + 1}/{_updateTotal}, " +
                    $"loss = {microBatchLoss:F6}");
            }
            _firstBatch = null;
        }

        public void ClipGradients()
        {
            // Classification retains its existing unclipped optimizer math.
        }

        public void ApplySchedule()
        {
            if (_schedulePending)
            {
                LearningRates = _scheduler.step();
                _schedulePending = false;
            }
            else
            {
                LearningRates = _scheduler.get_last_lr();
            }
        }

        public void CommitOptimizer() => _optimizer.step();

        public void CommitMetrics()
        {
            TrainingLossSum += _pendingLoss;
            TrainingCorrect = checked(TrainingCorrect + _pendingCorrect);
            TrainingSamples = checked(TrainingSamples + _pendingSamples);
            GlobalStep = checked(GlobalStep + 1);
        }
    }
}
