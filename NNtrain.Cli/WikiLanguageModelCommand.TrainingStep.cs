using NNtrain.Training.Execution;
using NNtrain.Training.Metrics;

namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    private abstract class WikiTrainingStepOperations
        : ITrainingTaskAdapter<WikiTrainingUpdate>
    {
        private readonly LanguageModel _model;
        private readonly IOptimizer _optimizer;
        private readonly Parameter[] _trainingParameters;
        private readonly CudaDataParallelEngine? _dataParallelEngine;
        private IReadOnlyList<WikiTrainingBatch> _microBatches = [];
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
            => TrainingGradientExecutionMode.FusedForwardBackwardReduced;

        internal int BatchSize { get; private set; }

        internal int SequenceLength { get; private set; }

        internal int BatchIndex { get; private set; }

        internal long DocumentsProcessed { get; private set; }

        internal long GlobalStep { get; set; }

        internal float LossValue { get; private set; }

        internal int MicroBatchCount => _microBatches.Count;

        internal int ValidTargetCount { get; private set; }

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

        public void AcceptBatch(WikiTrainingUpdate update)
        {
            if (update.Count == 0)
            {
                throw new InvalidOperationException(
                    "The acquired Wiki update has no microbatches.");
            }
            _microBatches = update.MicroBatches;
            WikiTrainingBatch last = update.Last;
            _batch = last.Values;
            BatchIndex = last.BatchIndex;
            BatchSize = last.BatchSize;
            SequenceLength = last.SequenceLength;
            DocumentsProcessed = last.DocumentsProcessed;
            int validTargets = 0;
            foreach (WikiTrainingBatch microBatch in _microBatches)
            {
                LanguageBatch values = microBatch.Values;
                if (values.Input.Length != checked(
                        microBatch.BatchSize * microBatch.SequenceLength)
                    || values.Target.Length != values.Input.Length)
                {
                    throw new InvalidOperationException(
                        "An acquired language microbatch does not match " +
                        "its configured shape.");
                }
                validTargets = checked(
                    validTargets + values.ValidTargetCount);
            }
            if (validTargets == 0)
            {
                throw new InvalidOperationException(
                    "The acquired Wiki update has no valid targets.");
            }
            ValidTargetCount = validTargets;
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
            if (_dataParallelEngine is { } engine)
            {
                var cudaBatches = new CudaLanguageModelMicroBatch[
                    _microBatches.Count];
                for (int index = 0; index < _microBatches.Count; index++)
                {
                    WikiTrainingBatch microBatch = _microBatches[index];
                    cudaBatches[index] = new CudaLanguageModelMicroBatch(
                        microBatch.Values.Input,
                        microBatch.Values.Target,
                        microBatch.BatchSize,
                        microBatch.SequenceLength);
                }
                LossValue = engine.ForwardBackwardAccumulated(
                    cudaBatches,
                    Tensor.DefaultCrossEntropyIgnoreIndex,
                    GlobalStep);
                return;
            }

            double weightedLoss = 0d;
            foreach (WikiTrainingBatch microBatch in _microBatches)
            {
                LanguageBatch values = microBatch.Values;
                Tensor loss = _model.forward_loss(
                    values.Input,
                    values.Target,
                    microBatch.BatchSize,
                    microBatch.SequenceLength);
                float gradientWeight =
                    (float)values.ValidTargetCount / ValidTargetCount;
                float microBatchLoss = BackwardAndReadLoss(
                    loss,
                    gradientWeight);
                weightedLoss += microBatchLoss * values.ValidTargetCount;
            }
            LossValue = (float)(weightedLoss / ValidTargetCount);
        }

        private static float BackwardAndReadLoss(
            Tensor loss,
            float gradientWeight)
        {
            if (Tensor.ExecutionDevice != TensorDevice.Cuda)
            {
                float cpuValue = loss.item();
                loss.backward([gradientWeight]);
                return cpuValue;
            }

            int deviceIndex = Tensor.CudaDeviceIndex;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            NativeCudaScalarReadback readback =
                NativeCudaScalarReadback.Rent(deviceIndex);
            readback.Begin(
                loss.EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                accelerator.DefaultStream);
            Exception? backwardFailure = null;
            try
            {
                loss.BackwardAndRelease([gradientWeight]);
            }
            catch (Exception exception)
            {
                backwardFailure = exception;
            }
            float value = 0f;
            Exception? readbackFailure = null;
            try
            {
                value = readback.CompleteAndReturn();
            }
            catch (Exception exception)
            {
                readbackFailure = exception;
            }
            if (backwardFailure is not null && readbackFailure is not null)
            {
                throw new AggregateException(
                    "CUDA accumulated backward and loss readback failed.",
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
            return value;
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
            int validTargets = ValidTargetCount;
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
            long targets = ValidTargetCount;
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
            CompletedBatches = checked(
                CompletedBatches + MicroBatchCount);
            GlobalStep = nextGlobalStep;
            GraphWindowLoss = flushGraph ? 0d : nextGraphWindowLoss;
            GraphWindowTargets = flushGraph
                ? 0
                : nextGraphWindowTargets;
        }
    }
}
