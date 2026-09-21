namespace NNtrain;

public sealed partial class NekoMuon
{
    private bool UsesPureBFloat16OptimizerState()
        => TensorExecutionContext.ActivePrecisionPolicy?.OptimizerState
                == NNtrain.Runtime.Execution.NumericFormat.BFloat16
            && _parameters.All(parameter =>
                parameter.T.DType == TensorDType.BFloat16);

    private void PrepareCudaBFloat16Residency(int[] devices)
    {
        TransitionFromCudaBfp8State(devices[0]);
        TransitionFromCudaFloatState(devices[0]);
        (int maximumLength, int maximumGramLength) =
            GetCudaScratchCapacities();
        var states = new CudaOptimizerKernels
            .NekoMuonBFloat16ResidentState[_parameters.Count];
        foreach (int deviceIndex in devices)
        {
            _ = GetOrCreatePreparedCudaScratch(
                deviceIndex, maximumLength, maximumGramLength);
            _ = GetOrCreateBfp8FiniteStatus(deviceIndex);
        }
        for (int index = 0; index < _parameters.Count; index++)
        {
            NekoMuonParameterState parameterState =
                _state.ParameterStates[index];
            CudaOptimizerKernels.NekoMuonBFloat16ResidentState state =
                _cudaBFloat16States[index] ??= new CudaOptimizerKernels
                    .NekoMuonBFloat16ResidentState(
                        parameterState.FastMoment,
                        parameterState.SlowMoment,
                        parameterState.Confidence);
            states[index] = state;
            foreach (int deviceIndex in devices)
            {
                _ = _parameters[index].T
                    .EnsureCudaBFloat16Buffer(deviceIndex);
                _ = state.GetOrCreate(deviceIndex);
                if (_state.Options.NewtonSchulzDepthMode
                    != NekoMuonNewtonSchulzDepthMode.Fixed && !ForceFullNewtonSchulz)
                {
                    _ = GetOrCreatePreparedCudaScratch(deviceIndex,
                        maximumLength, maximumGramLength)
                        .GetAdaptiveConfidencePointers(
                            [state.GetOrCreate(deviceIndex).Confidence.NativePtr]);
                }
            }
        }
        if (states.Length > 0)
        {
            foreach (int deviceIndex in devices)
            {
                if (!_cudaConfidenceBatches.ContainsKey(deviceIndex))
                    _cudaConfidenceBatches.Add(deviceIndex,
                        new CudaOptimizerKernels.NekoMuonConfidenceBatch(deviceIndex,
                            states.Select(state => state.GetOrCreate(deviceIndex)
                                .Confidence.NativePtr).ToArray()));
            }
        }
    }

    private void TransitionFromCudaBFloat16State(int primaryDevice)
    {
        if (!_cudaBFloat16States.Any(state => state is not null))
            return;
        for (int index = 0; index < _cudaBFloat16States.Length; index++)
        {
            CudaOptimizerKernels.NekoMuonBFloat16ResidentState? state =
                _cudaBFloat16States[index];
            if (state is null)
                continue;
            state.SynchronizeHost(primaryDevice);
            NekoMuonParameterState parameterState =
                _state.ParameterStates[index];
            if (state.IsDeviceConfidenceAuthoritative)
            {
                _state.ParameterStates[index] = parameterState with
                {
                    Confidence = state.SynchronizeConfidence(primaryDevice),
                };
            }
            state.Dispose();
            _cudaBFloat16States[index] = null;
        }
        foreach (CudaOptimizerKernels.NekoMuonBFloat16StatsBatch batch
            in _cudaBFloat16StatsBatches.Values)
        {
            batch.Dispose();
        }
        _cudaBFloat16StatsBatches.Clear();
        foreach (var batch in _cudaConfidenceBatches.Values)
            batch.Dispose();
        _cudaConfidenceBatches.Clear();
    }

    private void StepCudaBFloat16(
        NekoMuonOptions options,
        float fastCorrection,
        float slowCorrection,
        int[] devices)
    {
        if (devices.Length == 0)
        {
            throw new InvalidOperationException(
                "Pure BFloat16 NekoMuon requires at least one CUDA device.");
        }
        PrepareCudaBFloat16Residency(devices);
        bool runNewtonSchulz =
            _state.Step % options.NewtonSchulzInterval == 0;
        bool deviceOnlyFixedFive = runNewtonSchulz
            && options.MaxNewtonSchulzSteps == 5
            && (ForceFullNewtonSchulz
                || options.NewtonSchulzDepthMode
                    == NekoMuonNewtonSchulzDepthMode.Fixed
                && options.NewtonSchulzDepth == 5f);
        (int maximumLength, int maximumGramLength) =
            GetCudaScratchCapacities();
        var scratch = new CudaOptimizerKernels
            .NekoMuonDeviceScratch[devices.Length];
        var statuses = new NativeCudaBuffer<int>[devices.Length];
        for (int deviceSlot = 0; deviceSlot < devices.Length; deviceSlot++)
        {
            int deviceIndex = devices[deviceSlot];
            scratch[deviceSlot] = GetOrCreatePreparedCudaScratch(
                deviceIndex, maximumLength, maximumGramLength);
            statuses[deviceSlot] = GetOrCreateBfp8FiniteStatus(deviceIndex);
            statuses[deviceSlot].MemSetToZero();
        }

        Parallel.For(0, devices.Length, deviceSlot =>
        {
            int deviceIndex = devices[deviceSlot];
            for (int parameterIndex = 0;
                parameterIndex < _parameters.Count;
                parameterIndex++)
            {
                CudaOptimizerKernels.NekoMuonPrepareBFloat16StatsResident(
                    _parameters[parameterIndex].T,
                    deviceIndex,
                    _cudaBFloat16States[parameterIndex]!,
                    statuses[deviceSlot],
                    options.BetaFast,
                    options.BetaSlow,
                    fastCorrection,
                    slowCorrection,
                    options.Epsilon,
                    options.Rho,
                    deviceControl: true,
                    options.Nesterov);
            }
        });

        Parallel.For(0, devices.Length, deviceSlot =>
        {
            int deviceIndex = devices[deviceSlot];
            if (deviceOnlyFixedFive)
            {
                var batchItems = new CudaOptimizerKernels
                    .NekoMuonBFloat16BatchItem[_parameters.Count];
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
                    batchItems[parameterIndex] = new CudaOptimizerKernels
                        .NekoMuonBFloat16BatchItem(
                            parameter.T,
                            _cudaBFloat16States[parameterIndex]!,
                            originalRows,
                            originalColumns,
                            applyWeightDecay);
                }
                CudaOptimizerKernels
                    .NekoMuonFinishFixedFiveBFloat16GroupedDeviceResident(
                        deviceIndex,
                        batchItems,
                        scratch[deviceSlot],
                        statuses[deviceSlot],
                        fastCorrection,
                        options.Epsilon,
                        NewtonSchulzA,
                        NewtonSchulzB,
                        NewtonSchulzC,
                        options.LearningRate,
                        options.WeightDecay,
                        options.Nesterov);
                return;
            }
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
                _ =
                    CudaOptimizerKernels
                        .NekoMuonFinishBFloat16StepResident(
                            parameter.T,
                            deviceIndex,
                            _cudaBFloat16States[parameterIndex]!,
                            scratch[deviceSlot],
                            statuses[deviceSlot],
                            originalRows,
                            originalColumns,
                            fastCorrection,
                            options.Epsilon,
                            _state.ParameterStates[parameterIndex].Confidence,
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
                            applyWeightDecay,
                            deviceOnlyFixedFive,
                            ForceFullNewtonSchulz,
                            options.Nesterov);
            }
        });

        CudaOptimizerStepBatch.CompleteAfterSynchronization(
            devices,
            "pure BFloat16 NekoMuon update",
            queueReadback: null,
            finalize: () =>
            {
                for (int parameterIndex = 0;
                    parameterIndex < _parameters.Count;
                    parameterIndex++)
                {
                    _parameters[parameterIndex].T
                        .MarkCudaDataReplicasSynchronized(devices);
                }
            });
    }
}
