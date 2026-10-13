using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain;

public sealed partial class CudaDataParallelEngine
{
    private sealed partial class CudaTrainingShapePlan
    {
        internal void PrepareGraphInputs(int ignoreIndex)
        {
            _ignoreIndex = ignoreIndex;
            _phase = WorkPhase.InitializeGraphInputs;
        }

        internal void PrepareReservationPrewarm(
            CudaBfp8GradientAllReducePlan? bfp8Plan,
            CudaBFloat16GradientAllReducePlan? bfloat16Plan,
            ICudaGradientReductionPlan? reductionPlan,
            long reductionStepId,
            long globalStep,
            ulong rootSeed)
        {
            _bfp8Plan = bfp8Plan;
            _bfloat16Plan = bfloat16Plan;
            _reductionPlan = reductionPlan;
            _reductionStepId = reductionStepId;
            _globalStep = globalStep;
            _rootSeed = rootSeed;
            _useFixedInputs = true;
            _phase = WorkPhase.ReservationPrewarm;
        }

        internal void PrepareCapture(
            CudaBfp8GradientAllReducePlan? bfp8Plan,
            CudaBFloat16GradientAllReducePlan? bfloat16Plan,
            ICudaGradientReductionPlan? reductionPlan,
            long reductionStepId,
            int totalValid,
            int ignoreIndex,
            long globalStep,
            ulong rootSeed)
        {
            _bfp8Plan = bfp8Plan;
            _bfloat16Plan = bfloat16Plan;
            _reductionPlan = reductionPlan;
            _reductionStepId = reductionStepId;
            _totalValid = totalValid;
            _ignoreIndex = ignoreIndex;
            _globalStep = globalStep;
            _rootSeed = rootSeed;
            _phase = WorkPhase.Capture;
        }

        internal void PrepareReplay(
            CudaBfp8GradientAllReducePlan? bfp8Plan,
            CudaBFloat16GradientAllReducePlan? bfloat16Plan,
            long reductionStepId,
            long globalStep,
            bool reuseCaptureInputs,
            bool beginReductionDeviceStep = true,
            bool publishGradientsAfterReplay = true,
            int rngSequence = 0,
            NativeCudaBuffer<float>[]? accumulatedLossBuffers = null,
            bool completeAccumulatedLoss = false)
        {
            if (!IsCompiled)
                throw new InvalidOperationException(
                    "The CUDA training graph is not compiled.");
            _bfp8Plan = bfp8Plan;
            _bfloat16Plan = bfloat16Plan;
            _reductionPlan = null;
            _reductionStepId = reductionStepId;
            _globalStep = globalStep;
            _reuseCaptureInputs = reuseCaptureInputs;
            _beginReductionDeviceStep = beginReductionDeviceStep;
            _publishGradientsAfterReplay = publishGradientsAfterReplay;
            _rngSequence = rngSequence;
            _accumulatedLossBuffers = accumulatedLossBuffers;
            _completeAccumulatedLoss = completeAccumulatedLoss;
            _phase = WorkPhase.Replay;
        }

        internal void PrepareGraphDisposal()
            => _phase = WorkPhase.DisposeGraph;

        internal void CommitCapture()
        {
            if (_compiledGraphs.Any(static graph => graph is null))
            {
                throw new InvalidOperationException(
                    "CUDA Graph capture did not publish every device graph.");
            }
            _compiledTotalValid = _totalValid;
        }

        internal void RecordWarmup()
        {
            if (_warmupCount < 2)
                _warmupCount++;
        }

        internal void DisableGraph(Exception reason)
        {
            ArgumentNullException.ThrowIfNull(reason);
            _graphDisabled = true;
        }

        private void InitializeDeviceGraphInputs(int replicaIndex)
        {
            int device = Devices[replicaIndex];
            if (!TensorExecutionContext.TryGetCudaStreamLane(
                    device,
                    out IStreamExecutionLane streamLane)
                || streamLane is not CudaExecutionLane lane)
            {
                throw new NotSupportedException(
                    $"CUDA Graph inputs require a session lane for device {device}.");
            }
            if (_graphInputs[replicaIndex] is not null)
                return;
            CudaGraphBatchInputs inputs = CudaGraphBatchInputs.Create(
                lane,
                ShardInputs[replicaIndex].Length,
                _owner._model.VocabularySize,
                _ignoreIndex);
            CudaGraphRngState? rng = null;
            NativeCudaBuffer<float>? lossSlot = null;
            CudaBfp8GraphGradientPublication? bfp8Publication = null;
            try
            {
                rng = CudaGraphRngState.Create(lane);
                NativeCudaDevice accelerator =
                    ForgetMemoryV2Cuda.GetAccelerator(device);
                lossSlot = accelerator.Allocate1D<float>(
                    1,
                    CudaMemoryKind.Persistent);
                if (_owner._model.PrecisionMode == TensorPrecisionMode.Bfp8
                    && Devices.Length == 1)
                {
                    bfp8Publication = new CudaBfp8GraphGradientPublication(
                        device,
                        _owner._parameters);
                }
                inputs.Update(
                    ShardInputs[replicaIndex],
                    ShardTargets[replicaIndex],
                    lane.ComputeStreamHandle);
                _graphInputs[replicaIndex] = inputs;
                _graphRngStates[replicaIndex] = rng;
                _graphLossSlots[replicaIndex] = lossSlot;
                _graphBfp8Publications[replicaIndex] = bfp8Publication;
            }
            catch (Exception initializationFailure)
            {
                var failures = new List<Exception> { initializationFailure };
                TryDispose(bfp8Publication, failures);
                TryDispose(lossSlot, failures);
                TryDispose(rng, failures);
                TryDispose(inputs, failures);
                if (failures.Count == 1)
                {
                    System.Runtime.ExceptionServices.ExceptionDispatchInfo
                        .Capture(initializationFailure)
                        .Throw();
                }
                throw new AggregateException(
                    "CUDA Graph input-state initialization and cleanup failed.",
                    failures);
            }
        }

        private void ExecuteReservationPrewarm(int replicaIndex)
        {
            int device = Devices[replicaIndex];
            if (!TensorExecutionContext.TryGetCudaStreamLane(
                    device,
                    out IStreamExecutionLane streamLane)
                || streamLane is not CudaExecutionLane lane)
            {
                throw new NotSupportedException(
                    $"CUDA Graph reservation requires a session lane for device {device}.");
            }
            if (_captureMemoryScopes[replicaIndex] is not null)
            {
                throw new InvalidOperationException(
                    "CUDA Graph memory reservation prewarm was already started.");
            }
            CudaGraphBatchInputs inputs = _graphInputs[replicaIndex]
                ?? throw new InvalidOperationException(
                    "CUDA Graph inputs were not initialized.");
            CudaGraphRngState rng = _graphRngStates[replicaIndex]
                ?? throw new InvalidOperationException(
                    "CUDA Graph RNG was not initialized.");
            CudaGraphCaptureScope captureMemory =
                lane.Memory.BeginGraphCaptureReservation();
            _capturePinnedBaselines[replicaIndex] =
                lane.Memory.GraphPinnedBytes;
            _captureMemoryScopes[replicaIndex] = captureMemory;
            rng.SetCounter(DeriveCounter(_rootSeed, _globalStep, device));
            _bfp8Plan?.BeginDeviceStep(_reductionStepId, device);
            _bfloat16Plan?.BeginDeviceStep(_reductionStepId, device);
            using IDisposable? reductionScope = _reductionPlan is null
                ? null
                : CudaGradientReductionContext.Push(
                    _reductionPlan,
                    device,
                    _reductionStepId);
            using IDisposable? recording =
                BeginCapturedBackwardRecording(
                    device,
                    CudaCapturedBackwardRecordingMode.ReservationPrewarm);
            using CudaGraphBatchInputs.CudaGraphBatchInputCaptureScope
                inputScope = inputs.PushCaptureScope();
            using CudaGraphDropoutCaptureScope dropoutScope =
                CudaGraphDropoutCaptureScope.Begin(
                    rng,
                    DeriveOperationSeed(_rootSeed, device));
            using IDisposable? bfp8PublicationScope =
                _graphBfp8Publications[replicaIndex]?.PushRecording();
            Tensor loss = _owner._model.ForwardLoss(
                ShardInputs[replicaIndex],
                ShardTargets[replicaIndex],
                ShardBatches[replicaIndex],
                SequenceLength,
                _ignoreIndex);
            float weight = (float)ShardValidTargets[replicaIndex]
                / _totalValid;
            loss.BackwardAndRelease([weight]);
            lane.SynchronizeComputeStream();
        }

        private void CaptureDeviceGraph(int replicaIndex)
        {
            int device = Devices[replicaIndex];
            if (!TensorExecutionContext.TryGetCudaStreamLane(
                    device,
                    out IStreamExecutionLane streamLane)
                || streamLane is not CudaExecutionLane lane)
            {
                throw new NotSupportedException(
                    $"CUDA Graph capture requires a session lane for device {device}.");
            }

            _bfp8Plan?.BeginDeviceStep(_reductionStepId, device);
            _bfloat16Plan?.BeginDeviceStep(_reductionStepId, device);

            CudaGraphBatchInputs inputs = _graphInputs[replicaIndex]
                ?? throw new InvalidOperationException(
                    "CUDA Graph inputs were not initialized before prewarm.");
            CudaGraphRngState rng = _graphRngStates[replicaIndex]
                ?? throw new InvalidOperationException(
                    "CUDA Graph RNG was not initialized before capture.");
            NativeCudaBuffer<float> lossSlot = _graphLossSlots[replicaIndex]
                ?? throw new InvalidOperationException(
                    "CUDA Graph loss slot was not initialized before capture.");
            CudaGraphExecutable? graph = null;
            IDisposable? memoryReservation = null;
            CudaGraphCaptureScope captureMemory =
                _captureMemoryScopes[replicaIndex]
                ?? throw new InvalidOperationException(
                    "CUDA Graph capture memory was not reservation-prewarmed.");
            long pinnedBefore = _capturePinnedBaselines[replicaIndex];
            try
            {
                NativeCudaDevice accelerator =
                    ForgetMemoryV2Cuda.GetAccelerator(device);
                rng.SetCounter(DeriveCounter(
                    _rootSeed,
                    _globalStep,
                    device));
                lane.SynchronizeComputeStream();

                using IDisposable? reductionScope = _reductionPlan is null
                    ? null
                    : CudaGradientReductionContext.Push(
                        _reductionPlan,
                        device,
                        _reductionStepId);
                graph = CudaGraphExecutable.Capture(lane, () =>
                {
                    using IDisposable? recording =
                        BeginCapturedBackwardRecording(
                            device,
                            CudaCapturedBackwardRecordingMode.StreamCapture);
                    CudaGraphBatchInputs.CudaGraphBatchInputCaptureScope
                        inputScope = inputs.PushCaptureScope();
                    try
                    {
                        using CudaGraphBfp8ParameterRefreshScope
                            parameterRefreshScope =
                                CudaGraphBfp8ParameterRefreshScope.Begin(
                                    device);
                        using CudaGraphDropoutCaptureScope dropoutScope =
                            CudaGraphDropoutCaptureScope.Begin(
                                rng,
                                DeriveOperationSeed(_rootSeed, device));
                        using IDisposable? bfp8PublicationScope =
                            _graphBfp8Publications[replicaIndex]
                                ?.PushRecording();
                        Tensor loss = _owner._model.ForwardLoss(
                            ShardInputs[replicaIndex],
                            ShardTargets[replicaIndex],
                            ShardBatches[replicaIndex],
                            SequenceLength,
                            _ignoreIndex);
                        nint lossPointer = loss
                            .EnsureCudaFloat32Buffer(device)
                            .NativePtr;
                        NativeCudaRuntime.Check(
                            NativeCudaRuntime.CopyDeviceToDeviceAsyncNative(
                                device,
                                lossSlot.NativePtr,
                                device,
                                lossPointer,
                                sizeof(float),
                                lane.ComputeStreamHandle),
                            "capture CUDA loss scalar snapshot");
                        float weight = (float)ShardValidTargets[replicaIndex]
                            / _totalValid;
                        loss.BackwardAndRelease([weight]);
                    }
                    catch (Exception recordFailure)
                    {
                        try
                        {
                            inputScope.Dispose();
                        }
                        catch (Exception scopeFailure)
                        {
                            throw new AggregateException(
                                "CUDA Graph record and input-scope cleanup failed.",
                                recordFailure,
                                scopeFailure);
                        }
                        throw;
                    }
                    inputScope.Dispose();
                });
                memoryReservation = captureMemory.Commit();
                captureMemory.Dispose();
                _captureMemoryScopes[replicaIndex] = null;
                long pinnedBytes = Math.Max(
                    0,
                    lane.Memory.GraphPinnedBytes - pinnedBefore);
                int[]? bfloat16BucketOrder = _bfloat16Plan
                    ?.SnapshotCapturedBucketOrderForGraph(device);
                _compiledGraphs[replicaIndex] = new CompiledDeviceGraph(
                    lane,
                    graph,
                    memoryReservation,
                    lossSlot,
                    rng,
                    inputs,
                    _graphBfp8Publications[replicaIndex],
                    bfloat16BucketOrder,
                    pinnedBytes);
                _graphInputs[replicaIndex] = null;
                _graphRngStates[replicaIndex] = null;
                _graphLossSlots[replicaIndex] = null;
                _graphBfp8Publications[replicaIndex] = null;
                graph = null;
                memoryReservation = null;
            }
            catch (Exception captureFailure)
            {
                var failures = new List<Exception> { captureFailure };
                TryDispose(graph, failures);
                TryDispose(memoryReservation, failures);
                if (failures.Count == 1)
                {
                    System.Runtime.ExceptionServices.ExceptionDispatchInfo
                        .Capture(captureFailure)
                        .Throw();
                }
                throw new AggregateException(
                    "CUDA Graph capture cleanup failed.",
                    failures);
            }
        }

        private IDisposable? BeginCapturedBackwardRecording(
            int device,
            CudaCapturedBackwardRecordingMode recordingMode)
        {
            if (_bfp8Plan is not null)
            {
                return _bfp8Plan.BeginCapturedBackwardRecording(
                    _reductionStepId,
                    device);
            }
            return _bfloat16Plan?.BeginCapturedBackwardRecording(
                _reductionStepId,
                device,
                recordingMode);
        }

        private void ReplayDeviceGraph(int replicaIndex)
        {
            CompiledDeviceGraph compiled = _compiledGraphs[replicaIndex]
                ?? throw new InvalidOperationException(
                    "The CUDA device graph is not compiled.");
            int device = Devices[replicaIndex];
            ShardStarted[replicaIndex] =
                System.Diagnostics.Stopwatch.GetTimestamp();
            if (_beginReductionDeviceStep)
            {
                _bfp8Plan?.BeginDeviceStep(_reductionStepId, device);
                _bfloat16Plan?.BeginDeviceStep(_reductionStepId, device);
                _accumulatedLossBuffers?[replicaIndex].MemSetToZero();
            }
            if (!_reuseCaptureInputs)
            {
                compiled.Inputs.Update(
                    ShardInputs[replicaIndex],
                    ShardTargets[replicaIndex],
                    compiled.Lane.ComputeStreamHandle);
            }
            compiled.Rng.SetCounter(DeriveCounter(
                _rootSeed,
                _globalStep,
                device,
                _rngSequence));
            compiled.Graph.Launch();
            NativeCudaScalarReadback? readback = null;
            bool accumulatedLoss = _accumulatedLossBuffers is not null;
            if (accumulatedLoss)
            {
                CudaTensorNative.Scale(
                    device,
                    compiled.LossSlot.NativePtr,
                    1,
                    ShardValidTargets[replicaIndex]);
                CudaTensorNative.Accumulate(
                    device,
                    compiled.LossSlot.NativePtr,
                    _accumulatedLossBuffers![replicaIndex].NativePtr,
                    1);
                if (_completeAccumulatedLoss)
                {
                    readback = NativeCudaScalarReadback.Rent(device);
                    readback.Begin(
                        _accumulatedLossBuffers[replicaIndex].NativePtr,
                        compiled.Lane.ComputeStreamHandle);
                }
            }
            else
            {
                readback = NativeCudaScalarReadback.Rent(device);
                readback.Begin(
                    compiled.LossSlot.NativePtr,
                    compiled.Lane.ComputeStreamHandle);
            }
            if (_publishGradientsAfterReplay && _bfp8Plan is not null)
            {
                _bfp8Plan.PublishCapturedDeviceGradientsAfterReplay(
                    _reductionStepId,
                    device);
            }
            else if (_publishGradientsAfterReplay
                && _bfloat16Plan is not null)
            {
                _bfloat16Plan.PublishCapturedDeviceGradientsForReplay(
                    _reductionStepId,
                    device,
                    compiled.BFloat16BucketOrder);
            }
            else if (_publishGradientsAfterReplay
                && Devices.Length == 1)
            {
                // Replay writes captured buffers without running managed
                // backward publication. There is no reducer to publish the
                // new local generation on a single GPU.
                NumericFormat gradientFormat = ResolvePrecisionPolicy(
                    _owner._model.PrecisionMode).Gradient;
                foreach (Parameter parameter in _owner._parameters)
                {
                    if (gradientFormat == NumericFormat.BFloat16)
                        parameter.T.MarkCudaBFloat16GradientMutated(device);
                    else if (gradientFormat == NumericFormat.Float32)
                        parameter.T.MarkCudaGradientMutated(device);
                }
            }
            float lossValue = readback?.CompleteAndReturn() ?? 0f;
            if (_publishGradientsAfterReplay)
                compiled.Bfp8Publication?.PublishAfterReplay();
            WeightedLosses[replicaIndex] =
                lossValue * (accumulatedLoss
                    ? 1
                    : ShardValidTargets[replicaIndex]);
            ShardElapsed[replicaIndex] =
                System.Diagnostics.Stopwatch.GetElapsedTime(
                    ShardStarted[replicaIndex]).TotalMilliseconds;
        }

    }
}
