using NNtrain.Training.Execution;
using NNtrain.Training.Metrics;

namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    private abstract class WikiTrainingStepOperations
        : ITrainingStepOperations
    {
        private readonly LanguageModel _model;
        private readonly IOptimizer _optimizer;
        private readonly Parameter[] _trainingParameters;
        private readonly CudaDataParallelEngine? _dataParallelEngine;
        private LanguageBatch _batch;
        private Tensor? _loss;

        protected WikiTrainingStepOperations(
            LanguageModel model,
            IOptimizer optimizer,
            Parameter[] trainingParameters,
            CudaDataParallelEngine? dataParallelEngine,
            long globalStep)
        {
            _model = model;
            _optimizer = optimizer;
            _trainingParameters = trainingParameters;
            _dataParallelEngine = dataParallelEngine;
            GlobalStep = globalStep;
        }

        public TrainingGradientExecutionMode GradientExecutionMode
            => _dataParallelEngine is not null && BatchSize > 1
                ? TrainingGradientExecutionMode
                    .FusedForwardBackwardReduced
                : TrainingGradientExecutionMode.Separate;

        internal int BatchSize { get; set; }

        internal int SequenceLength { get; set; }

        internal long GlobalStep { get; set; }

        internal float LossValue { get; private set; }

        internal float GradientNorm { get; private set; }

        internal IReadOnlyList<float> LearningRates { get; set; }
            = [];

        internal IReadOnlyList<int> LastShardBatchSizes
            => _dataParallelEngine?.LastShardBatchSizes ?? [];

        protected LanguageBatch Batch => _batch;

        public void AcquireBatch()
        {
            _batch = CreateBatch();
            if (_batch.Input.Length != checked(BatchSize * SequenceLength)
                || _batch.Target.Length != _batch.Input.Length)
            {
                throw new InvalidOperationException(
                    "The acquired language batch does not match its configured shape.");
            }
            _loss = null;
            LossValue = 0f;
            GradientNorm = 0f;
            LearningRates = [];
        }

        public void ClearGradients() => _optimizer.zero_grad();

        public void Forward()
        {
            Tensor logits = _model.forward(
                _batch.Input,
                BatchSize,
                SequenceLength);
            _loss = nn.functional.cross_entropy(logits, _batch.Target);
            LossValue = _loss.item();
        }

        public void Backward()
        {
            Tensor loss = _loss
                ?? throw new InvalidOperationException(
                    "Backward requires a completed forward phase.");
            if (Tensor.ExecutionDevice == TensorDevice.Cuda)
                loss.BackwardAndRelease();
            else
                loss.backward();
        }

        public void ReduceGradients()
        {
            // A local gradient is already authoritative. Data-parallel
            // reduction is completed inside ForwardBackwardReduced instead.
        }

        public void ForwardBackwardReduced()
        {
            CudaDataParallelEngine engine = _dataParallelEngine
                ?? throw new InvalidOperationException(
                    "A fused data-parallel step requires a session-owned engine.");
            LossValue = engine.ForwardBackward(
                _batch.Input,
                _batch.Target,
                BatchSize,
                SequenceLength);
        }

        public void ClipGradients()
            => GradientNorm = nn.utils.clip_grad_norm_(
                _trainingParameters,
                max_norm: 1f);

        public abstract void ApplySchedule();

        public void CommitOptimizer() => _optimizer.step();

        public abstract void CommitMetrics();

        protected abstract LanguageBatch CreateBatch();
    }

    private sealed class FixedWikiTrainingStepOperations
        : WikiTrainingStepOperations
    {
        private readonly WarmupCosineProgressLRScheduler _scheduler;
        private readonly TrainingMetricReporter _metricReporter;
        private readonly int _graphUpdateSteps;
        private readonly long _totalTrainingSteps;
        private int[] _tokens = [];
        private int[] _order = [];
        private int _batchStart;
        private int _batchIndex;
        private int _batchTotal;
        private int _epoch;

        internal FixedWikiTrainingStepOperations(
            LanguageModel model,
            IOptimizer optimizer,
            Parameter[] trainingParameters,
            CudaDataParallelEngine? dataParallelEngine,
            WarmupCosineProgressLRScheduler scheduler,
            TrainingMetricReporter metricReporter,
            int graphUpdateSteps,
            long totalTrainingSteps,
            long globalStep)
            : base(
                model,
                optimizer,
                trainingParameters,
                dataParallelEngine,
                globalStep)
        {
            _scheduler = scheduler;
            _metricReporter = metricReporter;
            _graphUpdateSteps = graphUpdateSteps;
            _totalTrainingSteps = totalTrainingSteps;
        }

        internal double TotalLoss { get; private set; }

        internal long CompletedTargets { get; private set; }

        internal float GraphWindowLoss { get; private set; }

        internal int GraphWindowTargets { get; private set; }

        internal int CompletedBatches { get; private set; }

        internal bool EpochEnd => CompletedBatches == _batchTotal;

        internal void StartEpoch(
            int epoch,
            int batchTotal,
            int completedBatches,
            double totalLoss,
            long completedTargets)
        {
            _epoch = epoch;
            _batchTotal = batchTotal;
            TotalLoss = totalLoss;
            CompletedTargets = completedTargets;
            CompletedBatches = completedBatches;
            GraphWindowLoss = 0f;
            GraphWindowTargets = 0;
        }

        internal void ConfigureBatch(
            int[] tokens,
            int[] order,
            int batchStart,
            int batchSize,
            int sequenceLength,
            int batchIndex)
        {
            _tokens = tokens;
            _order = order;
            _batchStart = batchStart;
            BatchSize = batchSize;
            SequenceLength = sequenceLength;
            _batchIndex = batchIndex;
        }

        public override void ApplySchedule()
            => LearningRates = _scheduler.step(
                (GlobalStep + 1d) / _totalTrainingSteps);

        public override void CommitMetrics()
        {
            long nextGlobalStep = checked(GlobalStep + 1);
            int validTargets = Batch.ValidTargetCount;
            float contribution = LossValue * validTargets;
            double nextTotalLoss = TotalLoss + contribution;
            long nextCompletedTargets = CompletedTargets + validTargets;
            float nextGraphWindowLoss = GraphWindowLoss + contribution;
            int nextGraphWindowTargets = checked(
                GraphWindowTargets + validTargets);
            int nextCompletedBatches = _batchIndex + 1;
            bool epochEnd = nextCompletedBatches == _batchTotal;
            bool flushGraph = nextGlobalStep % _graphUpdateSteps == 0
                && !epochEnd;

            if (flushGraph)
            {
                double epochPosition = _epoch - 1d
                    + (double)nextCompletedBatches / _batchTotal;
                _metricReporter.AppendCommittedLoss(
                    nextGlobalStep,
                    epochPosition,
                    MetricKinds.TrainLoss,
                    nextGraphWindowLoss / nextGraphWindowTargets);
            }

            TotalLoss = nextTotalLoss;
            CompletedTargets = nextCompletedTargets;
            CompletedBatches = nextCompletedBatches;
            GlobalStep = nextGlobalStep;
            GraphWindowLoss = flushGraph ? 0f : nextGraphWindowLoss;
            GraphWindowTargets = flushGraph
                ? 0
                : nextGraphWindowTargets;
        }

        protected override LanguageBatch CreateBatch()
            => WikiLanguageModelCommand.CreateBatch(
                _tokens,
                _order,
                _batchStart,
                BatchSize,
                SequenceLength);
    }

    private sealed class StreamingWikiTrainingStepOperations
        : WikiTrainingStepOperations
    {
        private readonly WarmupCosineProgressLRScheduler _scheduler;
        private readonly TrainingMetricReporter _metricReporter;
        private readonly List<int> _buffer;
        private readonly long _documentsPerEpoch;
        private readonly int _totalEpochs;
        private readonly int _graphUpdateSteps;
        private int _epoch;
        private long _documentsProcessed;

        internal StreamingWikiTrainingStepOperations(
            LanguageModel model,
            IOptimizer optimizer,
            Parameter[] trainingParameters,
            CudaDataParallelEngine? dataParallelEngine,
            WarmupCosineProgressLRScheduler scheduler,
            TrainingMetricReporter metricReporter,
            List<int> buffer,
            long documentsPerEpoch,
            int totalEpochs,
            int graphUpdateSteps,
            long globalStep,
            double graphWindowLoss,
            long graphWindowTargets)
            : base(
                model,
                optimizer,
                trainingParameters,
                dataParallelEngine,
                globalStep)
        {
            _scheduler = scheduler;
            _metricReporter = metricReporter;
            _buffer = buffer;
            _documentsPerEpoch = documentsPerEpoch;
            _totalEpochs = totalEpochs;
            _graphUpdateSteps = graphUpdateSteps;
            GraphWindowLoss = graphWindowLoss;
            GraphWindowTargets = graphWindowTargets;
        }

        internal double TotalLoss { get; private set; }

        internal long CompletedTargets { get; private set; }

        internal double GraphWindowLoss { get; private set; }

        internal long GraphWindowTargets { get; private set; }

        internal int CompletedBatches { get; private set; }

        internal double OverallProgress { get; private set; }

        internal void StartEpoch(
            int epoch,
            double totalLoss,
            long completedTargets,
            int completedBatches)
        {
            _epoch = epoch;
            TotalLoss = totalLoss;
            CompletedTargets = completedTargets;
            CompletedBatches = completedBatches;
        }

        internal void ConfigureBatch(
            int batchSize,
            int sequenceLength,
            long documentsProcessed)
        {
            BatchSize = batchSize;
            SequenceLength = sequenceLength;
            _documentsProcessed = documentsProcessed;
        }

        public override void ApplySchedule()
        {
            double documentProgress = _documentsPerEpoch == 0
                ? 0d
                : Math.Min(
                    1d,
                    (double)_documentsProcessed / _documentsPerEpoch);
            OverallProgress =
                (_epoch - 1d + documentProgress) / _totalEpochs;
            LearningRates = _scheduler.step(OverallProgress);
        }

        public override void CommitMetrics()
        {
            long nextGlobalStep = checked(GlobalStep + 1);
            long targets = Batch.ValidTargetCount;
            double contribution = LossValue * targets;
            double nextTotalLoss = TotalLoss + contribution;
            long nextCompletedTargets = checked(CompletedTargets + targets);
            double nextGraphWindowLoss = GraphWindowLoss + contribution;
            long nextGraphWindowTargets = checked(
                GraphWindowTargets + targets);
            bool flushGraph = nextGlobalStep % _graphUpdateSteps == 0;

            if (flushGraph)
            {
                double progress = _documentsPerEpoch == 0
                    ? 0d
                    : Math.Min(
                        1d,
                        (double)_documentsProcessed / _documentsPerEpoch);
                _metricReporter.AppendCommittedLoss(
                    nextGlobalStep,
                    _epoch - 1d + progress,
                    MetricKinds.TrainLoss,
                    nextGraphWindowLoss / nextGraphWindowTargets);
            }

            TotalLoss = nextTotalLoss;
            CompletedTargets = nextCompletedTargets;
            CompletedBatches++;
            GlobalStep = nextGlobalStep;
            GraphWindowLoss = flushGraph ? 0d : nextGraphWindowLoss;
            GraphWindowTargets = flushGraph
                ? 0
                : nextGraphWindowTargets;
        }

        protected override LanguageBatch CreateBatch()
            => WikiLanguageModelCommand.CreateStreamingBatch(
                _buffer,
                BatchSize,
                SequenceLength);
    }
}
