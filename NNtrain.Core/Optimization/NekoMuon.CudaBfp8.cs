using System.Diagnostics;
using NNtrain.Cuda.Interop;

namespace NNtrain;

public sealed partial class NekoMuon
{
    private void StepCudaBfp8(
        NekoMuonOptions options,
        float fastCorrection,
        float slowCorrection,
        int[] devices,
        bool mixedBlockState = false)
    {
        if (devices.Length == 0)
        {
            throw new InvalidOperationException(
                "Pure BFP8 NekoMuon requires at least one CUDA device.");
        }

        int primaryDevice = devices[0];
        bool runNewtonSchulz =
            _state.Step % options.NewtonSchulzInterval == 0;
        bool deviceOnlyFixedFive = runNewtonSchulz
            && options.MaxNewtonSchulzSteps == 5
            && (ForceFullNewtonSchulz
                || options.NewtonSchulzDepthMode
                    == NekoMuonNewtonSchulzDepthMode.Fixed
                && options.NewtonSchulzDepth == 5f);
        if (_cudaStates.Any(state => state is not null))
        {
            for (int index = 0; index < _cudaStates.Length; index++)
            {
                CudaOptimizerKernels.NekoMuonResidentState? state =
                    _cudaStates[index];
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
                _cudaStates[index] = null;
            }
            foreach (CudaOptimizerKernels.NekoMuonStatsBatch batch
                in _cudaStatsBatches.Values)
            {
                batch.Dispose();
            }
            _cudaStatsBatches.Clear();
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

        var scratch = new CudaOptimizerKernels
            .NekoMuonDeviceScratch[devices.Length];
        var statuses = new NativeCudaBuffer<int>[devices.Length];
        for (int deviceSlot = 0; deviceSlot < devices.Length; deviceSlot++)
        {
            int deviceIndex = devices[deviceSlot];
            lock (_cudaScratch)
            {
                if (!_cudaScratch.TryGetValue(
                        deviceIndex,
                        out CudaOptimizerKernels.NekoMuonDeviceScratch?
                            deviceScratch))
                {
                    deviceScratch = new CudaOptimizerKernels
                        .NekoMuonDeviceScratch(
                            deviceIndex,
                            maximumLength,
                            maximumGramLength,
                            _cudaBatchCapacity,
                            !_cudaDispatchPolicy.DisableTensorCoreNekoMuon);
                    _cudaScratch.Add(deviceIndex, deviceScratch);
                }
                scratch[deviceSlot] = deviceScratch;
            }
            statuses[deviceSlot] = GetOrCreateBfp8FiniteStatus(deviceIndex);
            statuses[deviceSlot].MemSetToZero();
        }
        NativeCudaBuffer<CudaMix8DiagnosticAccumulator>?
            mix8Diagnostics = null;
        if (mixedBlockState)
        {
            _cudaMix8Diagnostics ??=
                new CudaMix8QuantizationDiagnostics();
            _cudaMix8DiagnosticsDevice = devices[0];
            mix8Diagnostics = _cudaMix8Diagnostics.Reset(devices[0]);
        }

        var states = new CudaOptimizerKernels
            .NekoMuonBfp8ResidentState[_parameters.Count];
        for (int parameterIndex = 0;
            parameterIndex < _parameters.Count;
            parameterIndex++)
        {
            NekoMuonParameterState parameterState =
                _state.ParameterStates[parameterIndex];
            CudaOptimizerKernels.NekoMuonBfp8ResidentState state =
                _cudaBfp8States[parameterIndex] ??=
                    new CudaOptimizerKernels.NekoMuonBfp8ResidentState(
                        parameterState.FastMoment,
                        parameterState.SlowMoment,
                        parameterState.Confidence,
                        mixedBlockState
                            ? Bfp8QuantizationDescriptor.Mix8_32
                            : Bfp8QuantizationDescriptor.TensorWide);
            states[parameterIndex] = state;
            foreach (int deviceIndex in devices)
                state.GetOrCreate(deviceIndex);
        }

        Parallel.For(0, devices.Length, deviceSlot =>
        {
            int deviceIndex = devices[deviceSlot];
            for (int parameterIndex = 0;
                parameterIndex < _parameters.Count;
                parameterIndex++)
            {
                CudaOptimizerKernels.NekoMuonPrepareBfp8StatsResident(
                    _parameters[parameterIndex].T,
                    deviceIndex,
                    states[parameterIndex],
                    scratch[deviceSlot],
                    statuses[deviceSlot],
                    options.BetaFast,
                    options.BetaSlow,
                    fastCorrection,
                    slowCorrection,
                    options.Epsilon,
                    options.Rho,
                    mixedBlockState,
                    options.Nesterov);
            }
        });

        if (!deviceOnlyFixedFive)
        {
            Parallel.For(0, devices.Length, deviceSlot =>
            {
                int deviceIndex = devices[deviceSlot];
                CudaOptimizerKernels.NekoMuonBfp8StatsBatch batch;
                lock (_cudaBfp8StatsBatches)
                {
                    if (!_cudaBfp8StatsBatches.TryGetValue(
                            deviceIndex,
                            out batch!))
                    {
                        batch = new CudaOptimizerKernels
                            .NekoMuonBfp8StatsBatch(
                                deviceIndex,
                                states);
                        _cudaBfp8StatsBatches.Add(deviceIndex, batch);
                    }
                }
                batch.GatherAndRead();
            });
        }

        bool useGroupedPureBfp8 = deviceOnlyFixedFive
            && !mixedBlockState;
        float[,]? confidences = useGroupedPureBfp8
            ? null
            : new float[devices.Length, _parameters.Count];
        Parallel.For(0, devices.Length, deviceSlot =>
        {
            int deviceIndex = devices[deviceSlot];
            NativeCudaBuffer<CudaMix8DiagnosticAccumulator>?
                deviceMix8Diagnostics = deviceIndex
                    == _cudaMix8DiagnosticsDevice
                    ? mix8Diagnostics
                    : null;
            if (useGroupedPureBfp8)
            {
                var batchItems = new CudaOptimizerKernels
                    .NekoMuonBfp8BatchItem[_parameters.Count];
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
                        .NekoMuonBfp8BatchItem(
                            parameter.T,
                            states[parameterIndex],
                            originalRows,
                            originalColumns,
                            applyWeightDecay);
                }
                CudaOptimizerKernels
                    .NekoMuonFinishFixedFiveBfp8GroupedDeviceResident(
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
                NekoMuonParameterState parameterState =
                    _state.ParameterStates[parameterIndex];
                GetMatrixShape(
                    parameter,
                    out int originalRows,
                    out int originalColumns);
                bool applyWeightDecay =
                    parameter.WeightDecay == WeightDecayPolicy.Apply
                    || (options.Decay1D && parameter.T.Rank == 1);
                confidences![deviceSlot, parameterIndex] =
                    CudaOptimizerKernels.NekoMuonFinishBfp8StepResident(
                        parameter.T,
                        deviceIndex,
                        states[parameterIndex],
                        scratch[deviceSlot],
                        statuses[deviceSlot],
                        originalRows,
                        originalColumns,
                        fastCorrection,
                        options.Epsilon,
                        parameterState.Confidence,
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
                        mixedBlockState,
                        ForceFullNewtonSchulz,
                        options.Nesterov,
                        deviceMix8Diagnostics);
            }
        });

        int primarySlot = Array.IndexOf(devices, Tensor.CudaDeviceIndex);
        if (primarySlot < 0)
            primarySlot = 0;
        int finalPrimarySlot = primarySlot;
        CudaOptimizerFiniteStatusReadback[] readbacks = devices
            .Select(GetOrCreateBfp8FiniteReadback)
            .ToArray();
        CudaOptimizerStepBatch.CompleteAfterSynchronization(
            devices,
            mixedBlockState
                ? "block-BFP8-state mix8_32 NekoMuon update"
                : "pure BFP8 NekoMuon update",
            queueReadback: () =>
            {
                for (int deviceSlot = 0;
                    deviceSlot < devices.Length;
                    deviceSlot++)
                {
                    readbacks[deviceSlot].Begin(statuses[deviceSlot]);
                }
            },
            finalize: () =>
            {
                int nonFiniteDevice = -1;
                for (int deviceSlot = 0;
                    deviceSlot < devices.Length;
                    deviceSlot++)
                {
                    int finite =
                        readbacks[deviceSlot].ReadAfterSynchronization();
                    if (finite != 0 && nonFiniteDevice < 0)
                        nonFiniteDevice = devices[deviceSlot];
                }
                if (nonFiniteDevice >= 0)
                {
                    throw new InvalidOperationException(
                        $"Non-finite CUDA value detected while publishing " +
                        $"pure BFP8 NekoMuon state on device " +
                        $"{nonFiniteDevice} at optimizer step " +
                        $"{_state.Step}.");
                }

                for (int parameterIndex = 0;
                    parameterIndex < _parameters.Count;
                    parameterIndex++)
                {
                    NekoMuonParameterState parameterState =
                        _state.ParameterStates[parameterIndex];
                    if (!deviceOnlyFixedFive)
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
                        .MarkCudaBfp8DataReplicasSynchronized(devices);
                }
            });
    }

}
