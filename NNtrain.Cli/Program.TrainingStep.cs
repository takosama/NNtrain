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
        : ITrainingTaskAdapter<ClassificationUpdateBatch>
    {
        private readonly TransformerClassifier _model;
        private readonly IOptimizer _optimizer;
        private readonly ILRScheduler _scheduler;
        private readonly TextWriter _output;
        private readonly float _labelSmoothing;
        private readonly int _classCount;
        private int _epoch;
        private double _pendingLoss;
        private int _pendingCorrect;
        private int _pendingSamples;
        private bool _schedulePending;
        private ClassificationUpdateBatch _batch;
        private bool _hasBatch;

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
            double trainingLossSum,
            int trainingCorrect,
            int trainingSamples,
            bool advanceSchedule)
        {
            _epoch = epoch;
            TrainingLossSum = trainingLossSum;
            TrainingCorrect = trainingCorrect;
            TrainingSamples = trainingSamples;
            _schedulePending = advanceSchedule;
            LearningRates = _scheduler.get_last_lr();
        }

        public void Prepare() => _optimizer.prepare();

        public void AcceptBatch(ClassificationUpdateBatch batch)
        {
            if (batch.MicroBatchCount <= 0 || batch.SampleCount <= 0)
            {
                throw new InvalidOperationException(
                    "The classification update has no samples.");
            }
            _pendingLoss = 0d;
            _pendingCorrect = 0;
            _pendingSamples = 0;
            _batch = batch;
            _hasBatch = true;
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
            if (!_hasBatch)
            {
                throw new InvalidOperationException(
                    "Classification forward requires an acquired update.");
            }
            ClassificationUpdateBatch batch = _batch;
            for (int accumulation = 0;
                accumulation < batch.MicroBatchCount;
                accumulation++)
            {
                int microBatch = batch.FirstMicroBatch + accumulation;
                DataBatch samples = batch.AcquireMicroBatch(accumulation);
                int samplesInMicroBatch = samples.target.Length;
                if (Tensor.ExecutionDevice == TensorDevice.Cuda)
                {
                    samples.input.PrepareCudaBatchInput(
                        Tensor.CudaDeviceIndex);
                }
                Tensor logits = _model.forward(samples.input);
                Tensor loss = nn.functional.cross_entropy(
                    logits,
                    samples.target,
                    label_smoothing: _labelSmoothing);
                using CudaClassificationCorrectCountReadback?
                    correctCountReadback = BeginCountCorrect(
                        logits,
                        samples.target,
                        _classCount);
                float gradientWeight =
                    (float)samplesInMicroBatch / batch.SampleCount;
                float microBatchLoss = BackwardAndReadLoss(
                    loss,
                    gradientWeight);
                _pendingLoss += microBatchLoss * samplesInMicroBatch;
                _pendingSamples += samplesInMicroBatch;
                _pendingCorrect += CompleteCountCorrect(
                    logits,
                    samples.target,
                    _classCount,
                    correctCountReadback);

                _output.WriteLine(
                    $"epoch {_epoch}, " +
                    $"microbatch {microBatch + 1}/" +
                    $"{batch.MicroBatchTotal}, accumulation " +
                    $"{accumulation + 1}/{batch.MicroBatchCount}, " +
                    $"update {batch.Update + 1}/{batch.UpdateTotal}, " +
                    $"loss = {microBatchLoss:F6}");
            }
            _hasBatch = false;
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

            float lossValue = 0f;
            Exception? readbackFailure = null;
            try
            {
                lossValue = readback.CompleteAndReturn();
            }
            catch (Exception exception)
            {
                readbackFailure = exception;
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
            return lossValue;
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
