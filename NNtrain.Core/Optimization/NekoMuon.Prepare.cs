namespace NNtrain;

public sealed partial class NekoMuon
{
    private readonly object _cudaPreparationSync = new();

    /// <summary>
    /// Prewarms all CUDA optimizer state, control scalars, statistics pointer
    /// tables, and Newton-Schulz workspace before transfer guarding begins.
    /// </summary>
    public void prepare()
    {
        if (Tensor.ExecutionDevice != TensorDevice.Cuda)
            return;

        int[] devices = Tensor.CudaDeviceIndices.ToArray();
        if (devices.Length == 0)
        {
            throw new InvalidOperationException(
                "CUDA NekoMuon preparation requires at least one device.");
        }

        lock (_cudaPreparationSync)
        {
            PrepareCudaResidency(devices);
            _cudaStateAuthorityDevice = devices[0];
        }
    }

    private void PrepareCudaResidency(int[] devices)
    {
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
            PrepareCudaBFloat16Residency(devices);
            return;
        }
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
            PrepareCudaBfp8Residency(devices);
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

        PrepareCudaFloatStateResidency(devices, mix8);
    }

    private void PrepareCudaFloatStateResidency(
        int[] devices,
        bool mix8)
    {
        TransitionFromCudaBFloat16State(devices[0]);
        TransitionFromCudaBfp8State(devices[0]);
        bool deviceOnlyFixedFive =
            UsesDeviceOnlyFixedFiveOnEveryStep(_state.Options);
        (int maximumLength, int maximumGramLength) =
            GetCudaScratchCapacities();
        foreach (int deviceIndex in devices)
        {
            _ = GetOrCreatePreparedCudaScratch(
                deviceIndex,
                maximumLength,
                maximumGramLength);
            if (mix8 || deviceOnlyFixedFive)
            {
                _ = GetOrCreateBfp8FiniteStatus(deviceIndex);
                if (mix8)
                    _ = GetOrCreateBfp8FiniteReadback(deviceIndex);
            }
        }

        var states = new CudaOptimizerKernels.NekoMuonResidentState[
            _parameters.Count];
        for (int parameterIndex = 0;
            parameterIndex < _parameters.Count;
            parameterIndex++)
        {
            Parameter parameter = _parameters[parameterIndex];
            NekoMuonParameterState parameterState =
                _state.ParameterStates[parameterIndex];
            CudaOptimizerKernels.NekoMuonResidentState state =
                _cudaStates[parameterIndex] ??=
                    new CudaOptimizerKernels.NekoMuonResidentState(
                        parameterState.FastMoment,
                        parameterState.SlowMoment,
                        parameterState.Confidence);
            states[parameterIndex] = state;
            foreach (int deviceIndex in devices)
            {
                if (mix8)
                    _ = parameter.T.EnsureCudaBfp8Buffer(deviceIndex);
                _ = parameter.T.EnsureCudaMasterFloat32Buffer(deviceIndex);
                _ = state.GetOrCreate(deviceIndex);
            }
        }

        if (!deviceOnlyFixedFive)
        {
            foreach (int deviceIndex in devices)
            {
                if (_cudaStatsBatches.ContainsKey(deviceIndex))
                    continue;
                _cudaStatsBatches.Add(
                    deviceIndex,
                    new CudaOptimizerKernels.NekoMuonStatsBatch(
                        deviceIndex,
                        states));
            }
        }
    }

    private void PrepareCudaBfp8Residency(int[] devices)
    {
        TransitionFromCudaBFloat16State(devices[0]);
        TransitionFromCudaFloatState(devices[0]);
        (int maximumLength, int maximumGramLength) =
            GetCudaScratchCapacities();
        var states = new CudaOptimizerKernels.NekoMuonBfp8ResidentState[
            _parameters.Count];
        for (int parameterIndex = 0;
            parameterIndex < _parameters.Count;
            parameterIndex++)
        {
            Parameter parameter = _parameters[parameterIndex];
            NekoMuonParameterState parameterState =
                _state.ParameterStates[parameterIndex];
            CudaOptimizerKernels.NekoMuonBfp8ResidentState state =
                _cudaBfp8States[parameterIndex] ??=
                    new CudaOptimizerKernels.NekoMuonBfp8ResidentState(
                        parameterState.FastMoment,
                        parameterState.SlowMoment,
                        parameterState.Confidence);
            states[parameterIndex] = state;
            foreach (int deviceIndex in devices)
            {
                _ = parameter.T.EnsureCudaBfp8Buffer(deviceIndex);
                _ = state.GetOrCreate(deviceIndex);
                _ = GetOrCreateBfp8FiniteStatus(deviceIndex);
                _ = GetOrCreateBfp8FiniteReadback(deviceIndex);
                CudaOptimizerKernels.NekoMuonDeviceScratch scratch =
                    GetOrCreatePreparedCudaScratch(
                        deviceIndex,
                        maximumLength,
                        maximumGramLength);
                _ = scratch.GetBfp8Buffers(parameter.T.Numel);
            }
        }

        if (!UsesDeviceOnlyFixedFiveOnEveryStep(_state.Options))
        {
            foreach (int deviceIndex in devices)
            {
                if (_cudaBfp8StatsBatches.ContainsKey(deviceIndex))
                    continue;
                _cudaBfp8StatsBatches.Add(
                    deviceIndex,
                    new CudaOptimizerKernels.NekoMuonBfp8StatsBatch(
                        deviceIndex,
                        states));
            }
        }
    }

    private void TransitionFromCudaBfp8State(int primaryDevice)
    {
        if (!_cudaBfp8States.Any(state => state is not null))
            return;
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

    private void TransitionFromCudaFloatState(int primaryDevice)
    {
        if (!_cudaStates.Any(state => state is not null))
            return;
        for (int index = 0; index < _cudaStates.Length; index++)
        {
            CudaOptimizerKernels.NekoMuonResidentState? state =
                _cudaStates[index];
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
            _cudaStates[index] = null;
        }
        foreach (CudaOptimizerKernels.NekoMuonStatsBatch batch
            in _cudaStatsBatches.Values)
        {
            batch.Dispose();
        }
        _cudaStatsBatches.Clear();
    }

    private (int MaximumLength, int MaximumGramLength)
        GetCudaScratchCapacities()
    {
        int maximumLength = _parameters.Max(parameter => parameter.T.Numel);
        int maximumGramLength = 0;
        foreach (Parameter parameter in _parameters)
        {
            GetMatrixShape(
                parameter,
                out int shapeRows,
                out int shapeColumns);
            int rows = Math.Min(shapeRows, shapeColumns);
            maximumGramLength = Math.Max(
                maximumGramLength,
                checked(rows * rows));
        }
        return (maximumLength, maximumGramLength);
    }

    private CudaOptimizerKernels.NekoMuonDeviceScratch
        GetOrCreatePreparedCudaScratch(
            int deviceIndex,
            int maximumLength,
            int maximumGramLength)
    {
        lock (_cudaScratch)
        {
            if (_cudaScratch.TryGetValue(
                    deviceIndex,
                    out CudaOptimizerKernels.NekoMuonDeviceScratch? scratch))
            {
                return scratch;
            }
            scratch = new CudaOptimizerKernels.NekoMuonDeviceScratch(
                deviceIndex,
                maximumLength,
                maximumGramLength,
                _cudaBatchCapacity,
                !_cudaDispatchPolicy.DisableTensorCoreNekoMuon);
            _cudaScratch.Add(deviceIndex, scratch);
            return scratch;
        }
    }

    private bool UsesDeviceOnlyFixedFiveOnEveryStep(
        NekoMuonOptions options)
    {
        return options.NewtonSchulzInterval == 1
            && options.MaxNewtonSchulzSteps == 5
            && (ForceFullNewtonSchulz
                || options.NewtonSchulzDepthMode
                    == NekoMuonNewtonSchulzDepthMode.Fixed
                && options.NewtonSchulzDepth == 5f);
    }
}
