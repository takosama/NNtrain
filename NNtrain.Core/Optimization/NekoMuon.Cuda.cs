using System.Diagnostics;
using NNtrain.Cuda.Interop;

namespace NNtrain;

public sealed partial class NekoMuon
{
    private void StepCuda(
        NekoMuonOptions options,
        float fastCorrection,
        float slowCorrection)
    {
        if (_parameters.Count == 0)
            return;

        int[] devices = Tensor.CudaDeviceIndices.ToArray();
        if (devices.Length == 0)
        {
            throw new InvalidOperationException(
                "CUDA NekoMuon requires at least one device.");
        }
        _cudaStateAuthorityDevice = devices[0];
        bool pureBFloat16 = UsesPureBFloat16OptimizerState();
        if (TensorExecutionContext.ActivePrecisionPolicy?.OptimizerState
                == NNtrain.Runtime.Execution.NumericFormat.BFloat16
            && !pureBFloat16)
        {
            throw new InvalidOperationException(
                "The pure BFloat16 NekoMuon contract requires every " +
                "parameter to use physical BFloat16 storage.");
        }
        if (pureBFloat16)
        {
            StepCudaBFloat16(
                options, fastCorrection, slowCorrection, devices);
            return;
        }
        TransitionFromCudaBFloat16State(devices[0]);
        bool pureBfp8 = _parameters.All(parameter =>
            UsesPureBfp8OptimizerState(parameter.T));
        bool anyMix8 = _parameters.Any(parameter =>
            UsesMix8Parameter(parameter.T));
        bool mix8 = _parameters.All(parameter =>
            UsesMix8Parameter(parameter.T));
        if (anyMix8 && !mix8)
        {
            throw new InvalidOperationException(
                "The mix8_32 NekoMuon contract cannot mix block-scaled " +
                "BFP8 parameters with another storage format.");
        }
        if (anyMix8)
            ValidateMix8OptimizerContract();
        if (TensorExecutionContext.ActivePrecisionPolicy?.OptimizerState
                == NNtrain.Runtime.Execution.NumericFormat.Bfp8
            && !pureBfp8)
        {
            throw new InvalidOperationException(
                "The pure BFP8 NekoMuon contract requires every parameter " +
                "to use tensor-wide BFP8 storage.");
        }
        if (pureBfp8)
        {
            StepCudaBfp8(options, fastCorrection, slowCorrection, devices);
            return;
        }
        if (TensorExecutionContext.ActivePrecisionPolicy?.Mode
                == NNtrain.Runtime.Execution.PrecisionMode.Mix8_32
            && !mix8)
        {
            throw new InvalidOperationException(
                "The mix8_32 NekoMuon contract requires every parameter " +
                "to use block-scaled BFP8 storage.");
        }
        // Ordinary Muon recursively reuses m_t and feeds its Nesterov
        // direction to NS5 on every step. Requantizing those values to BFP8
        // can erase small momentum components and change the orthogonalized
        // update enough to stall convergence. Keep its recurrent state in
        // FP32 even when the optional low-memory NekoMuon state is enabled;
        // mix8_32 parameters and activations remain block-BFP8 and the
        // parameter update still accumulates in the resident FP32 master.
        if (mix8
            && _cudaDispatchPolicy.EnableBlockBfp8OptimizerState
            && !options.Nesterov)
        {
            StepCudaBfp8(
                options,
                fastCorrection,
                slowCorrection,
                devices,
                mixedBlockState: true);
            return;
        }
        bool runNewtonSchulz =
            _state.Step % options.NewtonSchulzInterval == 0;
        bool deviceOnlyFixedFive = runNewtonSchulz
            && options.MaxNewtonSchulzSteps == 5
            && (ForceFullNewtonSchulz
                || options.NewtonSchulzDepthMode
                    == NekoMuonNewtonSchulzDepthMode.Fixed
                && options.NewtonSchulzDepth == 5f);
        bool deviceOnlyAdaptive = mix8
            && !ForceFullNewtonSchulz
            && options.NewtonSchulzDepthMode
                != NekoMuonNewtonSchulzDepthMode.Fixed;
        if (_cudaBfp8States.Any(state => state is not null))
        {
            int primaryDevice = devices[0];
            for (int index = 0; index < _cudaBfp8States.Length; index++)
            {
                CudaOptimizerKernels.NekoMuonBfp8ResidentState? state =
                    _cudaBfp8States[index];
                if (state is null)
                    continue;
                state.SynchronizeHost(primaryDevice);
                NekoMuonParameterState parameterState =
                    _state.ParameterStates[index];
                _state.ParameterStates[index] = parameterState with
                {
                    Confidence = state.SynchronizeConfidence(primaryDevice),
                };
                state.Dispose();
                _cudaBfp8States[index] = null;
            }
            foreach (CudaOptimizerKernels.NekoMuonBfp8StatsBatch batch
                in _cudaBfp8StatsBatches.Values)
            {
                batch.Dispose();
            }
            _cudaBfp8StatsBatches.Clear();
        }
        int maximumLength = _parameters.Max(parameter => parameter.T.Numel);
        int maximumGramLength = 0;
        foreach (Parameter parameter in _parameters)
        {
            GetMatrixShape(parameter, out int shapeRows, out int shapeColumns);
            int rows = Math.Min(shapeRows, shapeColumns);
            maximumGramLength = Math.Max(
                maximumGramLength,
                checked(rows * rows));
        }

        CudaOptimizerKernels.NekoMuonDeviceScratch GetCudaScratch(
            int deviceIndex)
        {
            lock (_cudaScratch)
            {
                if (!_cudaScratch.TryGetValue(
                    deviceIndex,
                    out CudaOptimizerKernels.NekoMuonDeviceScratch? scratch))
                {
                    scratch = new CudaOptimizerKernels.NekoMuonDeviceScratch(
                        deviceIndex,
                        maximumLength,
                        maximumGramLength,
                        _cudaBatchCapacity,
                        !_cudaDispatchPolicy.DisableTensorCoreNekoMuon);
                    _cudaScratch.Add(deviceIndex, scratch);
                }
                return scratch;
            }
        }

        // Materialize one reusable work area per device before moment kernels
        // are queued.  Each device has a single ordered stream, so subsequent
        // parameter updates can safely reuse it without a host-side wait.
        var deviceScratch = new CudaOptimizerKernels.NekoMuonDeviceScratch[
            devices.Length];
        for (int deviceSlot = 0; deviceSlot < devices.Length; deviceSlot++)
            deviceScratch[deviceSlot] = GetCudaScratch(devices[deviceSlot]);
        NativeCudaBuffer<int>[]? deviceFiniteStatus = null;
        if (mix8 || deviceOnlyFixedFive)
        {
            deviceFiniteStatus = new NativeCudaBuffer<int>[devices.Length];
            for (int deviceSlot = 0;
                deviceSlot < devices.Length;
                deviceSlot++)
            {
                deviceFiniteStatus[deviceSlot] =
                    GetOrCreateBfp8FiniteStatus(devices[deviceSlot]);
                deviceFiniteStatus[deviceSlot].MemSetToZero();
            }
        }
        NativeCudaBuffer<CudaMix8DiagnosticAccumulator>?
            mix8Diagnostics = null;
        if (mix8)
        {
            _cudaMix8Diagnostics ??=
                new CudaMix8QuantizationDiagnostics();
            _cudaMix8DiagnosticsDevice = devices[0];
            mix8Diagnostics = _cudaMix8Diagnostics.Reset(devices[0]);
        }

        CudaOptimizerKernels.NekoMuonResidentState GetCudaState(
            int parameterIndex)
        {
            NekoMuonParameterState parameterState =
                _state.ParameterStates[parameterIndex];
            return _cudaStates[parameterIndex] ??=
                new CudaOptimizerKernels.NekoMuonResidentState(
                    parameterState.FastMoment,
                    parameterState.SlowMoment,
                    parameterState.Confidence);
        }

        // Queue every moment/statistics update before waiting. Previously the
        // tiny D2H statistics copy serialized the stream once per parameter,
        // preventing kernels for later parameters from filling the GPU.
        void PrepareMomentsAndStatistics()
        {
            for (int index = 0; index < _parameters.Count; index++)
            {
                CudaOptimizerKernels.NekoMuonResidentState cudaState =
                    GetCudaState(index);
                for (int deviceSlot = 0;
                    deviceSlot < devices.Length;
                    deviceSlot++)
                {
                    int deviceIndex = devices[deviceSlot];
                    if (mix8)
                    {
                        CudaOptimizerKernels.NekoMuonPrepareMix8StatsResident(
                            _parameters[index].T,
                            deviceIndex,
                            cudaState,
                            deviceFiniteStatus![deviceSlot],
                            options.BetaFast,
                            options.BetaSlow,
                            fastCorrection,
                            slowCorrection,
                            options.Epsilon,
                            options.Rho,
                            options.Nesterov);
                    }
                    else if (deviceOnlyFixedFive)
                    {
                        CudaOptimizerKernels
                            .NekoMuonPrepareFixedFiveStatsResident(
                                _parameters[index].T,
                                deviceIndex,
                                cudaState,
                                deviceFiniteStatus![deviceSlot],
                                options.BetaFast,
                                options.BetaSlow,
                                fastCorrection,
                                slowCorrection,
                                options.Epsilon,
                                options.Rho,
                                options.Nesterov);
                    }
                    else
                    {
                        CudaOptimizerKernels.NekoMuonPrepareStatsResident(
                            _parameters[index].T,
                            deviceIndex,
                            cudaState,
                            options.BetaFast,
                            options.BetaSlow,
                            fastCorrection,
                            slowCorrection,
                            options.Nesterov);
                    }
                }
            }
        }
        if (CudaOperationProfiler.IsEnabled)
        {
            CudaOperationProfiler.MeasureDevices(
                "optimizer.nekomuon.moments_stats",
                devices,
                PrepareMomentsAndStatistics);
        }
        else
        {
            PrepareMomentsAndStatistics();
        }

        void ReadStatistics()
        {
            Parallel.For(0, devices.Length, deviceSlot =>
            {
                int deviceIndex = devices[deviceSlot];
                CudaOptimizerKernels.NekoMuonStatsBatch batch;
                lock (_cudaStatsBatches)
                {
                    if (!_cudaStatsBatches.TryGetValue(
                        deviceIndex,
                        out batch!))
                    {
                        var states = new CudaOptimizerKernels
                            .NekoMuonResidentState[_parameters.Count];
                        for (int index = 0; index < states.Length; ++index)
                            states[index] = GetCudaState(index);
                        batch = new CudaOptimizerKernels.NekoMuonStatsBatch(
                            deviceIndex, states);
                        _cudaStatsBatches[deviceIndex] = batch;
                    }
                }
                batch.GatherAndRead();
            });
        }
        if (!deviceOnlyFixedFive && !deviceOnlyAdaptive
            && CudaOperationProfiler.IsEnabled)
        {
            CudaOperationProfiler.MeasureDevices(
                "optimizer.nekomuon.stats_d2h",
                devices,
                ReadStatistics);
        }
        else if (!deviceOnlyFixedFive && !deviceOnlyAdaptive)
        {
            ReadStatistics();
        }

        if (mix8)
        {
            for (int deviceSlot = 0;
                deviceSlot < devices.Length;
                deviceSlot++)
            {
                int deviceIndex = devices[deviceSlot];
                if (_cudaConfidenceBatches.ContainsKey(deviceIndex))
                    continue;
                var states = new CudaOptimizerKernels
                    .NekoMuonResidentState[_parameters.Count];
                for (int index = 0; index < states.Length; index++)
                    states[index] = GetCudaState(index);
                _cudaConfidenceBatches.Add(
                    deviceIndex,
                    new CudaOptimizerKernels.NekoMuonConfidenceBatch(
                        deviceIndex, states));
            }
        }

        // Give every CUDA device its own host dispatch loop. This avoids the
        // parameter-major GPU0/GPU1 alternation and keeps both default streams
        // populated during Newton-Schulz and weight publication.
        float[,]? confidences = deviceOnlyFixedFive || deviceOnlyAdaptive
            ? null
            : new float[devices.Length, _parameters.Count];
        void FinishUpdates()
        {
            Parallel.For(0, devices.Length, deviceSlot =>
            {
                int deviceIndex = devices[deviceSlot];
                NativeCudaBuffer<CudaMix8DiagnosticAccumulator>?
                    deviceMix8Diagnostics = deviceIndex
                        == _cudaMix8DiagnosticsDevice
                        ? mix8Diagnostics
                        : null;
                CudaOptimizerKernels.NekoMuonDeviceScratch scratch =
                    deviceScratch[deviceSlot];
                if (deviceOnlyFixedFive)
                {
                    var fixedBatchItems = new CudaOptimizerKernels
                        .NekoMuonBatchItem[_parameters.Count];
                    for (int parameterIndex = 0;
                        parameterIndex < _parameters.Count;
                        parameterIndex++)
                    {
                        Parameter parameter = _parameters[parameterIndex];
                        GetMatrixShape(
                            parameter,
                            out int originalRows,
                            out int originalColumns);
                        bool applyWeightDecay =
                            parameter.WeightDecay == WeightDecayPolicy.Apply
                            || (options.Decay1D && parameter.T.Rank == 1);
                        fixedBatchItems[parameterIndex] =
                            new CudaOptimizerKernels.NekoMuonBatchItem(
                                parameter.T,
                                GetCudaState(parameterIndex),
                                originalRows,
                                originalColumns,
                                PreviousConfidence: 0f,
                                ApplyWeightDecay: applyWeightDecay);
                    }
                    CudaOptimizerKernels
                        .NekoMuonFinishFixedFiveGroupedDeviceResident(
                            deviceIndex,
                            fixedBatchItems,
                            scratch,
                            deviceFiniteStatus![deviceSlot],
                            fastCorrection,
                            options.Epsilon,
                            NewtonSchulzA,
                            NewtonSchulzB,
                            NewtonSchulzC,
                            options.LearningRate,
                            options.WeightDecay,
                            publishMix8: mix8,
                            nesterov: options.Nesterov,
                            mix8Diagnostics: deviceMix8Diagnostics);
                    return;
                }
                if (deviceOnlyAdaptive)
                {
                    var adaptiveBatchItems = new CudaOptimizerKernels
                        .NekoMuonBatchItem[_parameters.Count];
                    for (int parameterIndex = 0;
                        parameterIndex < _parameters.Count;
                        parameterIndex++)
                    {
                        Parameter parameter = _parameters[parameterIndex];
                        GetMatrixShape(
                            parameter,
                            out int originalRows,
                            out int originalColumns);
                        bool applyWeightDecay =
                            parameter.WeightDecay == WeightDecayPolicy.Apply
                            || (options.Decay1D && parameter.T.Rank == 1);
                        adaptiveBatchItems[parameterIndex] =
                            new CudaOptimizerKernels.NekoMuonBatchItem(
                                parameter.T,
                                GetCudaState(parameterIndex),
                                originalRows,
                                originalColumns,
                                PreviousConfidence: 0f,
                                ApplyWeightDecay: applyWeightDecay);
                    }
                    CudaOptimizerKernels
                        .NekoMuonFinishAdaptiveGroupedDeviceResident(
                            deviceIndex,
                            adaptiveBatchItems,
                            scratch,
                            deviceFiniteStatus![deviceSlot],
                            fastCorrection,
                            options.Epsilon,
                            options.MaxNewtonSchulzSteps,
                            options.NewtonSchulzDepthMode,
                            options.NewtonSchulzDepth,
                            runNewtonSchulz,
                            NewtonSchulzA,
                            NewtonSchulzB,
                            NewtonSchulzC,
                            options.LearningRate,
                            options.WeightDecay,
                            options.Nesterov,
                            deviceMix8Diagnostics);
                    foreach (Parameter parameter in _parameters)
                    {
                        CudaOptimizerKernels.PublishMix8Master(
                            parameter.T,
                            deviceIndex,
                            deviceFiniteStatus[deviceSlot],
                            deviceMix8Diagnostics);
                    }
                    return;
                }
                var batchItems = new CudaOptimizerKernels
                    .NekoMuonBatchItem[_parameters.Count];
                for (int parameterIndex = 0;
                    parameterIndex < _parameters.Count;
                    parameterIndex++)
                {
                    Parameter parameter = _parameters[parameterIndex];
                    NekoMuonParameterState parameterState =
                        _state.ParameterStates[parameterIndex];
                    GetMatrixShape(
                        parameter,
                        out int originalRows,
                        out int originalColumns);
                    bool applyWeightDecay =
                        parameter.WeightDecay == WeightDecayPolicy.Apply
                        || (options.Decay1D && parameter.T.Rank == 1);
                    batchItems[parameterIndex] = new CudaOptimizerKernels
                        .NekoMuonBatchItem(
                            parameter.T,
                            GetCudaState(parameterIndex),
                            originalRows,
                            originalColumns,
                            parameterState.Confidence,
                            applyWeightDecay);
                }
                float[] deviceConfidences = CudaOptimizerKernels
                    .NekoMuonFinishStepGrouped(
                        deviceIndex,
                        batchItems,
                        scratch,
                        fastCorrection,
                        slowCorrection,
                        options.Epsilon,
                        options.Rho,
                        options.MaxNewtonSchulzSteps,
                        options.NewtonSchulzDepthMode,
                        options.NewtonSchulzDepth,
                        runNewtonSchulz,
                        NewtonSchulzA,
                        NewtonSchulzB,
                        NewtonSchulzC,
                        options.LearningRate,
                        options.WeightDecay,
                        ForceFullNewtonSchulz,
                        mix8 ? deviceFiniteStatus![deviceSlot] : null,
                        deviceMix8Diagnostics);
                for (int parameterIndex = 0;
                    parameterIndex < deviceConfidences.Length;
                    parameterIndex++)
                {
                    confidences![deviceSlot, parameterIndex] =
                        deviceConfidences[parameterIndex];
                }
                if (mix8)
                {
                    foreach (Parameter parameter in _parameters)
                    {
                        CudaOptimizerKernels.PublishMix8Master(
                            parameter.T,
                            deviceIndex,
                            deviceFiniteStatus![deviceSlot],
                            deviceMix8Diagnostics);
                    }
                }
            });
        }
        if (CudaOperationProfiler.IsEnabled)
        {
            CudaOperationProfiler.MeasureDevices(
                "optimizer.nekomuon.initialize_ns_apply",
                devices,
                FinishUpdates);
        }
        else
        {
            FinishUpdates();
        }
        int primarySlot = Array.IndexOf(devices, Tensor.CudaDeviceIndex);
        if (primarySlot < 0)
            primarySlot = 0;
        int finalPrimarySlot = primarySlot;
        CudaOptimizerFiniteStatusReadback[]? mix8Readbacks = mix8
            ? devices.Select(GetOrCreateBfp8FiniteReadback).ToArray()
            : null;
        CudaOptimizerStepBatch.CompleteAfterSynchronization(
            devices,
            "NekoMuon update",
            queueReadback: mix8
                ? () =>
                {
                    for (int deviceSlot = 0;
                        deviceSlot < devices.Length;
                        deviceSlot++)
                    {
                            mix8Readbacks![deviceSlot].Begin(
                            deviceFiniteStatus![deviceSlot]);
                    }
                }
                : null,
            finalize: () =>
            {
                if (mix8)
                {
                    ThrowIfMix8PublicationNonFinite(
                        mix8Readbacks!,
                        devices,
                        _state.Step);
                }

                for (int parameterIndex = 0;
                    parameterIndex < _parameters.Count;
                    parameterIndex++)
                {
                    NekoMuonParameterState parameterState =
                        _state.ParameterStates[parameterIndex];
                    if (!deviceOnlyFixedFive && !deviceOnlyAdaptive)
                    {
                        _state.ParameterStates[parameterIndex] =
                            parameterState with
                            {
                                Confidence = confidences![
                                    finalPrimarySlot,
                                    parameterIndex],
                            };
                    }
                    _parameters[parameterIndex].T
                        .MarkCudaDataReplicasSynchronized(devices);
                }
            });
    }

}
