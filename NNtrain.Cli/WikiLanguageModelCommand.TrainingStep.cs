using NNtrain.Training.Execution;
using NNtrain.Training.Metrics;

namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    private abstract class WikiTrainingStepOperations
        : ITrainingTaskAdapter<WikiTrainingBatch>
    {
        private readonly LanguageModel _model;
        private readonly IOptimizer _optimizer;
        private readonly Parameter[] _trainingParameters;
        private readonly CudaDataParallelEngine? _dataParallelEngine;
        private LanguageBatch _batch;
        private Tensor? _loss;
        private NativeCudaScalarReadback? _lossReadback;

        protected WikiTrainingStepOperations(
            LanguageModel model,
            IOptimizer optimizer,
            Parameter[] trainingParameters,
            CudaDataParallelEngine? dataParallelEngine,
            int preparedBatchSize,
            int preparedSequenceLength,
            long globalStep)
        {
            _model = model;
            _optimizer = optimizer;
            _trainingParameters = trainingParameters;
            _dataParallelEngine = dataParallelEngine;
            if (preparedBatchSize <= 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(preparedBatchSize));
            }
            if (preparedSequenceLength <= 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(preparedSequenceLength));
            }
            BatchSize = preparedBatchSize;
            SequenceLength = preparedSequenceLength;
            GlobalStep = globalStep;
        }

        public TrainingGradientExecutionMode GradientExecutionMode
            => _dataParallelEngine is not null && BatchSize > 1
                ? TrainingGradientExecutionMode
                    .FusedForwardBackwardReduced
                : TrainingGradientExecutionMode.Separate;

        internal int BatchSize { get; private set; }

        internal int SequenceLength { get; private set; }

        internal int BatchIndex { get; private set; }

        internal long DocumentsProcessed { get; private set; }

        internal long GlobalStep { get; set; }

        internal float LossValue { get; private set; }

        internal float GradientNorm { get; private set; }

        internal IReadOnlyList<float> LearningRates { get; set; }
            = [];

        internal IReadOnlyList<int> LastShardBatchSizes
            => _dataParallelEngine?.LastShardBatchSizes ?? [];

        protected LanguageBatch Batch => _batch;

        public void Prepare()
        {
            _dataParallelEngine?.PrepareForTraining(BatchSize);
            _optimizer.prepare();
        }

        public void AcceptBatch(WikiTrainingBatch batch)
        {
            _batch = batch.Values;
            BatchIndex = batch.BatchIndex;
            BatchSize = batch.BatchSize;
            SequenceLength = batch.SequenceLength;
            DocumentsProcessed = batch.DocumentsProcessed;
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
            _loss = _model.forward_loss(
                _batch.Input,
                _batch.Target,
                BatchSize,
                SequenceLength);
            if (Tensor.ExecutionDevice == TensorDevice.Cuda)
            {
                int deviceIndex = Tensor.CudaDeviceIndex;
                NativeCudaDevice accelerator =
                    ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
                NativeCudaScalarReadback readback =
                    NativeCudaScalarReadback.Rent(deviceIndex);
                readback.Begin(
                    _loss.EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                    accelerator.DefaultStream);
                _lossReadback = readback;
            }
            else
            {
                LossValue = _loss.item();
            }
        }

        public void Backward()
        {
            Tensor loss = _loss
                ?? throw new InvalidOperationException(
                    "Backward requires a completed forward phase.");
            Exception? backwardFailure = null;
            try
            {
                if (Tensor.ExecutionDevice == TensorDevice.Cuda)
                    loss.BackwardAndRelease();
                else
                    loss.backward();
            }
            catch (Exception exception)
            {
                backwardFailure = exception;
            }

            Exception? readbackFailure = null;
            try
            {
                if (_lossReadback is { } readback)
                    LossValue = readback.CompleteAndReturn();
            }
            catch (Exception exception)
            {
                readbackFailure = exception;
            }
            finally
            {
                _lossReadback = null;
            }
            if (backwardFailure is not null && readbackFailure is not null)
            {
                throw new AggregateException(
                    "CUDA backward and its asynchronous loss readback failed.",
                    backwardFailure,
                    readbackFailure);
            }
            if (backwardFailure is not null)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo
                    .Capture(backwardFailure).Throw();
            }
            if (readbackFailure is not null)
            {
                System.Runtime.ExceptionServices.ExceptionDispatchInfo
                    .Capture(readbackFailure).Throw();
            }
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
                SequenceLength,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                GlobalStep);
        }

        public void ClipGradients()
            => GradientNorm = nn.utils.clip_grad_norm_(
                _trainingParameters,
                max_norm: 1f);

        public abstract void ApplySchedule();

        public void CommitOptimizer() => _optimizer.step();

        public abstract void CommitMetrics();

    }

    private sealed class FixedWikiTrainingStepOperations
        : WikiTrainingStepOperations
    {
        private readonly WarmupCosineProgressLRScheduler _scheduler;
        private readonly TrainingMetricReporter _metricReporter;
        private readonly int _graphUpdateSteps;
        private readonly long _totalTrainingSteps;
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
            int preparedBatchSize,
            int preparedSequenceLength,
            long globalStep)
            : base(
                model,
                optimizer,
                trainingParameters,
                dataParallelEngine,
                preparedBatchSize,
                preparedSequenceLength,
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
            int nextCompletedBatches = BatchIndex + 1;
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
    }

    private sealed class StreamingWikiTrainingStepOperations
        : WikiTrainingStepOperations
    {
        private readonly WarmupCosineProgressLRScheduler _scheduler;
        private readonly TrainingMetricReporter _metricReporter;
        private readonly long _documentsPerEpoch;
        private readonly int _totalEpochs;
        private readonly int _graphUpdateSteps;
        private int _epoch;

        internal StreamingWikiTrainingStepOperations(
            LanguageModel model,
            IOptimizer optimizer,
            Parameter[] trainingParameters,
            CudaDataParallelEngine? dataParallelEngine,
            WarmupCosineProgressLRScheduler scheduler,
            TrainingMetricReporter metricReporter,
            long documentsPerEpoch,
            int totalEpochs,
            int graphUpdateSteps,
            int preparedBatchSize,
            int preparedSequenceLength,
            long globalStep,
            double graphWindowLoss,
            long graphWindowTargets)
            : base(
                model,
                optimizer,
                trainingParameters,
                dataParallelEngine,
                preparedBatchSize,
                preparedSequenceLength,
                globalStep)
        {
            _scheduler = scheduler;
            _metricReporter = metricReporter;
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

        public override void ApplySchedule()
        {
            double documentProgress = _documentsPerEpoch == 0
                ? 0d
                : Math.Min(
                    1d,
                    (double)DocumentsProcessed / _documentsPerEpoch);
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
                        (double)DocumentsProcessed / _documentsPerEpoch);
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
    }
}
