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
        bool deviceOnlyFixedFive =
            UsesDeviceOnlyFixedFiveOnEveryStep(_state.Options);
        (int maximumLength, int maximumGramLength) =
            GetCudaScratchCapacities();
        var states = new CudaOptimizerKernels
            .NekoMuonBFloat16ResidentState[_parameters.Count];
        foreach (int deviceIndex in devices)
        {
            _ = GetOrCreatePreparedCudaScratch(
                deviceIndex, maximumLength, maximumGramLength);
            if (deviceOnlyFixedFive)
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
            }
        }
        if (deviceOnlyFixedFive)
            return;
        foreach (int deviceIndex in devices)
        {
            if (!_cudaBFloat16StatsBatches.ContainsKey(deviceIndex))
            {
                _cudaBFloat16StatsBatches.Add(
                    deviceIndex,
                    new CudaOptimizerKernels.NekoMuonBFloat16StatsBatch(
                        deviceIndex, states));
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
                    deviceOnlyFixedFive,
                    options.Nesterov);
            }
        });

        if (!deviceOnlyFixedFive)
        {
            Parallel.For(0, devices.Length, deviceSlot =>
            {
                int deviceIndex = devices[deviceSlot];
                CudaOptimizerKernels.NekoMuonBFloat16StatsBatch batch;
                lock (_cudaBFloat16StatsBatches)
                {
                    batch = _cudaBFloat16StatsBatches[deviceIndex];
                }
                batch.GatherAndRead();
            });
        }

        float[,]? confidences = deviceOnlyFixedFive
            ? null
            : new float[devices.Length, _parameters.Count];
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
                confidences![deviceSlot, parameterIndex] =
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
                            ForceFullNewtonSchulz);
            }
        });

        int primarySlot = Array.IndexOf(devices, Tensor.CudaDeviceIndex);
        if (primarySlot < 0)
            primarySlot = 0;
        int finalPrimarySlot = primarySlot;
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
                    if (!deviceOnlyFixedFive)
                    {
                        NekoMuonParameterState parameterState =
                            _state.ParameterStates[parameterIndex];
                        _state.ParameterStates[parameterIndex] =
                            parameterState with
                            {
                                Confidence = confidences![
                                    finalPrimarySlot, parameterIndex],
                            };
                    }
                    _parameters[parameterIndex].T
                        .MarkCudaDataReplicasSynchronized(devices);
                }
            });
    }
}
