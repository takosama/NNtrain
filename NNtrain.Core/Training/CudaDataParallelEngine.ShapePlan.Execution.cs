using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain;

public sealed partial class CudaDataParallelEngine
{
    private sealed partial class CudaTrainingShapePlan
    {
        internal void PrepareEager(
            CudaBfp8GradientAllReducePlan? bfp8Plan,
            CudaBFloat16GradientAllReducePlan? bfloat16Plan,
            ICudaGradientReductionPlan? reductionPlan,
            long reductionStepId,
            int totalValid,
            int ignoreIndex,
            bool useFixedInputs = false,
            bool beginReductionDeviceStep = true,
            NativeCudaBuffer<float>[]? accumulatedLossBuffers = null,
            bool completeAccumulatedLoss = false)
        {
            _bfp8Plan = bfp8Plan;
            _bfloat16Plan = bfloat16Plan;
            _reductionPlan = reductionPlan;
            _reductionStepId = reductionStepId;
            _totalValid = totalValid;
            _ignoreIndex = ignoreIndex;
            _useFixedInputs = useFixedInputs;
            _readEagerLoss = !useFixedInputs;
            _beginReductionDeviceStep = beginReductionDeviceStep;
            _accumulatedLossBuffers = accumulatedLossBuffers;
            _completeAccumulatedLoss = completeAccumulatedLoss;
            _phase = WorkPhase.Eager;
        }

        internal void PrepareSynchronization()
            => _phase = WorkPhase.Synchronize;

        public void Execute(
            int replicaIndex,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _disposed) != 0,
                this);
            if (_phase == WorkPhase.Synchronize)
            {
                int deviceIndex = Devices[replicaIndex];
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
                    .Synchronize(
                        $"data-parallel backward device {deviceIndex}");
                ShardElapsed[replicaIndex] =
                    System.Diagnostics.Stopwatch.GetElapsedTime(
                        ShardStarted[replicaIndex]).TotalMilliseconds;
                return;
            }
            if (_phase == WorkPhase.DisposeGraph)
            {
                DisposeDeviceGraph(replicaIndex);
                return;
            }
            if (_phase == WorkPhase.InitializeGraphInputs)
            {
                InitializeDeviceGraphInputs(replicaIndex);
                return;
            }
            if (_phase == WorkPhase.ReservationPrewarm)
            {
                ExecuteReservationPrewarm(replicaIndex);
                return;
            }
            if (_phase == WorkPhase.Capture)
            {
                CaptureDeviceGraph(replicaIndex);
                return;
            }
            if (_phase == WorkPhase.Replay)
            {
                ReplayDeviceGraph(replicaIndex);
                return;
            }
            if (_phase != WorkPhase.Eager)
            {
                throw new InvalidOperationException(
                    "CUDA training shape work was not prepared.");
            }

            ExecuteEager(replicaIndex);
        }

        private void ExecuteEager(int replicaIndex)
        {
            ShardStarted[replicaIndex] =
                System.Diagnostics.Stopwatch.GetTimestamp();
            int device = Devices[replicaIndex];
            int shardBatch = ShardBatches[replicaIndex];
            int[] shardInput = ShardInputs[replicaIndex];
            int[] shardTarget = ShardTargets[replicaIndex];
            int shardValid = ShardValidTargets[replicaIndex];

            if (_beginReductionDeviceStep)
            {
                _bfp8Plan?.BeginDeviceStep(_reductionStepId, device);
                _bfloat16Plan?.BeginDeviceStep(_reductionStepId, device);
                _accumulatedLossBuffers?[replicaIndex].MemSetToZero();
            }
            if (shardValid == 0 && _bfloat16Plan is not null)
            {
                // BeginDeviceStep already cleared this worker's gradient
                // storage. Publish its zero contribution without attempting
                // cross entropy on an all-ignore shard.
                foreach (Parameter parameter in _owner._parameters)
                    _reductionPlan?.NotifyGradientReady(
                        parameter.T, device, _reductionStepId);
                WeightedLosses[replicaIndex] = 0d;
                if (_accumulatedLossBuffers is not null && _completeAccumulatedLoss)
                {
                    NativeCudaScalarReadback readback = NativeCudaScalarReadback.Rent(device);
                    readback.Begin(_accumulatedLossBuffers[replicaIndex].NativePtr,
                        ForgetMemoryV2Cuda.GetAccelerator(device).DefaultStream);
                    WeightedLosses[replicaIndex] = readback.CompleteAndReturn();
                }
                ShardElapsed[replicaIndex] = System.Diagnostics.Stopwatch.GetElapsedTime(
                    ShardStarted[replicaIndex]).TotalMilliseconds;
                return;
            }
            using IDisposable? reductionScope = _reductionPlan is null
                ? null
                : CudaGradientReductionContext.Push(
                    _reductionPlan, device, _reductionStepId);
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(device);
            CudaGraphBatchInputs.CudaGraphBatchInputCaptureScope? inputScope = null;
            IDisposable? bfp8GraphPublicationScope = null;
            if (_useFixedInputs)
            {
                CudaGraphBatchInputs inputs = _graphInputs[replicaIndex]
                    ?? throw new InvalidOperationException(
                        "CUDA Graph inputs were not initialized.");
                inputScope = inputs.PushCaptureScope();
                bfp8GraphPublicationScope = _graphBfp8Publications[replicaIndex]
                    ?.PushRecording();
            }
            try
            {
                Tensor loss = _owner._model.ForwardLoss(
                    shardInput,
                    shardTarget,
                    shardBatch,
                    SequenceLength,
                    _ignoreIndex);
                if (_owner._dispatchPolicy.SynchronizeDataParallelPhases)
                {
                    accelerator.Synchronize(
                        $"data-parallel forward device {device}");
                }
                float weight = (float)shardValid / _totalValid;
                NativeCudaScalarReadback? readback = null;
                bool accumulatedLoss =
                    _accumulatedLossBuffers is not null;
                if (accumulatedLoss)
                {
                    NativeCudaBuffer<float> lossBuffer =
                        loss.EnsureCudaFloat32Buffer(device);
                    CudaTensorNative.Scale(
                        device,
                        lossBuffer.NativePtr,
                        1,
                        shardValid);
                    CudaTensorNative.Accumulate(
                        device,
                        lossBuffer.NativePtr,
                        _accumulatedLossBuffers![replicaIndex].NativePtr,
                        1);
                    if (_completeAccumulatedLoss)
                    {
                        readback = NativeCudaScalarReadback.Rent(device);
                        readback.Begin(
                            _accumulatedLossBuffers[replicaIndex].NativePtr,
                            accelerator.DefaultStream);
                    }
                }
                else if (_readEagerLoss)
                {
                    readback = NativeCudaScalarReadback.Rent(device);
                    readback.Begin(
                        loss.EnsureCudaFloat32Buffer(device).NativePtr,
                        accelerator.DefaultStream);
                }
                if (_owner._dispatchPolicy.SynchronizeDataParallelPhases)
                {
                    accelerator.Synchronize(
                        $"data-parallel loss device {device}");
                }
                loss.BackwardAndRelease([weight]);
                if (_owner._dispatchPolicy.SynchronizeDataParallelPhases)
                {
                    accelerator.Synchronize(
                        $"data-parallel backward device {device}");
                }
                WeightedLosses[replicaIndex] = readback is null
                    ? 0d
                    : readback.CompleteAndReturn()
                        * (accumulatedLoss ? 1 : shardValid);
            }
            finally
            {
                bfp8GraphPublicationScope?.Dispose();
                inputScope?.Dispose();
            }
            ShardElapsed[replicaIndex] =
                System.Diagnostics.Stopwatch.GetElapsedTime(
                    ShardStarted[replicaIndex]).TotalMilliseconds;
        }

        internal bool Matches(
            int batchSize,
            int sequenceLength,
            IReadOnlyList<int> devices,
            IReadOnlyList<int> shardBatches)
        {
            if (BatchSize != batchSize
                || SequenceLength != sequenceLength
                || Devices.Length != devices.Count
                || ShardBatches.Length != shardBatches.Count)
            {
                return false;
            }
            for (int index = 0; index < Devices.Length; index++)
            {
                if (Devices[index] != devices[index]
                    || ShardBatches[index] != shardBatches[index])
                {
                    return false;
                }
            }
            return true;
        }

        internal void PrepareBatch(
            ReadOnlySpan<int> input,
            ReadOnlySpan<int> target,
            int ignoreIndex)
        {
            Array.Clear(WeightedLosses);
            Array.Clear(ShardElapsed);
            Array.Clear(ShardStarted);
            for (int shard = 0; shard < Devices.Length; shard++)
            {
                int elementStart = checked(
                    ShardStarts[shard] * SequenceLength);
                int elementCount = ShardInputs[shard].Length;
                input.Slice(elementStart, elementCount)
                    .CopyTo(ShardInputs[shard]);
                target.Slice(elementStart, elementCount)
                    .CopyTo(ShardTargets[shard]);
                int valid = 0;
                foreach (int value in ShardTargets[shard])
                {
                    if (value != ignoreIndex)
                        valid++;
                }
                ShardValidTargets[shard] = valid;
            }
        }

    }
}
