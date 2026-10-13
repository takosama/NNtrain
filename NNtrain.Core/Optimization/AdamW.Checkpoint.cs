namespace NNtrain;

public partial class AdamW
{
    public AdamWState CaptureState()
    {
        AdamWState state = CaptureStateForStreaming();
        return _usesPackedBFloat16Checkpoint
            || _options.UseBFloat16FirstMoment
            || _options.UseBFloat16SecondMoment
            ? state
            : CloneState(state);
    }

    internal AdamWState CaptureStateForStreaming()
    {
        SynchronizeStateForStreaming();
        if (_usesPackedBFloat16Checkpoint)
        {
            var packedStates = new AdamWParameterState[_parameters.Count];
            for (int index = 0; index < packedStates.Length; index++)
            {
                Parameter parameter = _parameters[index];
                AdamWParameterRuntime runtime = _parameterRuntime[index];
                packedStates[index] = new AdamWParameterState(index,
                    parameter.Name, parameter.T.Shape.ToArray(),
                    DecodePackedBFloat16(runtime.FirstMomentPacked!),
                    DecodePackedBFloat16(runtime.SecondMomentPacked!));
            }
            return new AdamWState(AdamWState.CurrentFormatVersion,
                _step, _options with { }, packedStates);
        }
        if (_options.UseBFloat16FirstMoment
            || _options.UseBFloat16SecondMoment)
        {
            var states = new AdamWParameterState[_parameters.Count];
            for (int index = 0; index < states.Length; index++)
            {
                Parameter parameter = _parameters[index];
                states[index] = new AdamWParameterState(
                    index,
                    parameter.Name,
                    parameter.T.Shape.ToArray(),
                    _parameterRuntime[index].FirstMomentBFloat16 is not null
                        ? DecodeBFloat16(
                            _parameterRuntime[index]
                                .FirstMomentBFloat16!)
                        : _parameterRuntime[index].FirstMoment.ToArray(),
                    _parameterRuntime[index].SecondMomentBFloat16 is not null
                        ? DecodeBFloat16(
                            _parameterRuntime[index]
                                .SecondMomentBFloat16!)
                        : _parameterRuntime[index].SecondMoment.ToArray());
            }
            return new AdamWState(
                AdamWState.CurrentFormatVersion,
                _step,
                _options with { },
                states);
        }

        return new AdamWState(
            AdamWState.CurrentFormatVersion,
            _step,
            _options,
            _parameterStates);
    }

    /// <summary>
    /// Makes the managed moment arrays authoritative before a streaming
    /// serializer reads them.  Unlike <see cref="CaptureStateForStreaming"/>,
    /// this does not decode every BF16 moment into a second full-size FP32
    /// object graph.
    /// </summary>
    internal void SynchronizeStateForStreaming()
    {
        _arcState.Synchronize();
        if (_cudaStateAuthorityDevice is not int primaryDevice)
            return;
        foreach (AdamWParameterRuntime runtime in _parameterRuntime)
        {
            runtime.CudaState?.SynchronizeHost(primaryDevice);
            runtime.CudaBFloat16State?.SynchronizeHost(primaryDevice);
            runtime.CudaMixedState?.SynchronizeHost(primaryDevice);
            runtime.CudaBfp8State?.SynchronizeHost(primaryDevice);
        }
    }

    internal int StreamingStep => _step;

    internal AdamWOptions StreamingOptions => _options;

    internal int StreamingParameterCount => _parameterStates.Length;

    internal (ushort[] First, ushort[] Second) GetStreamingPackedMoments(int index)
    {
        AdamWParameterRuntime runtime = _parameterRuntime[index];
        return (runtime.FirstMomentPacked
                ?? throw new InvalidOperationException("No packed first moment."),
            runtime.SecondMomentPacked
                ?? throw new InvalidOperationException("No packed second moment."));
    }

    private static ushort[] PackBFloat16(float[] values)
    {
        var packed = new ushort[values.Length];
        for (int index = 0; index < values.Length; index++)
            packed[index] = TensorStorageCodec.EncodeBFloat16(values[index]);
        return packed;
    }

    private static float[] DecodePackedBFloat16(ushort[] values)
    {
        var decoded = new float[values.Length];
        for (int index = 0; index < values.Length; index++)
            decoded[index] = TensorStorageCodec.DecodeBFloat16(values[index]);
        return decoded;
    }

    internal AdamWStreamingParameterState GetStreamingParameterState(int index)
    {
        AdamWParameterState state = _parameterStates[index];
        AdamWParameterRuntime runtime = _parameterRuntime[index];
        bool bfp8State = runtime.CudaBfp8State is not null;
        return new AdamWStreamingParameterState(
            state.Index,
            state.Name,
            state.Shape,
            runtime.FirstMoment,
            bfp8State ? null : runtime.FirstMomentBFloat16,
            runtime.SecondMoment,
            bfp8State ? null : runtime.SecondMomentBFloat16);
    }

    public void RestoreState(AdamWState state)
        => RestoreState(state, takeOwnership: false);

    private void RestoreState(
        AdamWState state,
        bool takeOwnership)
    {
        ArgumentNullException.ThrowIfNull(state);
        ValidateState(state);
        AdamWState restored = takeOwnership ? state : CloneState(state);
        _step = restored.Step;
        _options = TensorExecutionContext.ActivePrecisionPolicy?.Mode
                == NNtrain.Runtime.Execution.PrecisionMode.Mix8_16
            ? restored.Options with
            {
                UseBFloat16FirstMoment = false,
                UseBFloat16SecondMoment = false,
            }
            : TensorExecutionContext.ActivePrecisionPolicy?.OptimizerState
                    == NNtrain.Runtime.Execution.NumericFormat.BFloat16
                ? restored.Options with
                {
                    UseBFloat16FirstMoment = true,
                    UseBFloat16SecondMoment = true,
                }
                : restored.Options;
        _parameterStates = restored.ParameterStates;
        DisposeCudaResources();
        if (_usesPackedBFloat16Checkpoint)
        {
            for (int index = 0; index < _parameterRuntime.Length; index++)
            {
                AdamWParameterRuntime runtime = _parameterRuntime[index];
                runtime.FirstMomentPacked = PackBFloat16(_parameterStates[index].FirstMoment);
                runtime.SecondMomentPacked = PackBFloat16(_parameterStates[index].SecondMoment);
                runtime.FirstMoment = runtime.SecondMoment = [];
                runtime.FirstMomentBFloat16 = runtime.SecondMomentBFloat16 = null;
                _parameterStates[index] = _parameterStates[index] with
                {
                    FirstMoment = [],
                    SecondMoment = [],
                };
            }
            RefreshWeightDecayFlags();
            return;
        }
        for (int index = 0; index < _parameterRuntime.Length; index++)
        {
            bool bfp8State = UsesPureBfp8OptimizerState(
                _parameterRuntime[index].Parameter.T);
            bool mix8State = UsesMix8Parameter(
                _parameterRuntime[index].Parameter.T);
            if (_options.UseBFloat16FirstMoment
                && !bfp8State
                && !mix8State)
            {
                _parameterRuntime[index].FirstMoment = [];
                _parameterRuntime[index].FirstMomentBFloat16 =
                    EncodeBFloat16(_parameterStates[index].FirstMoment);
                _parameterStates[index] = _parameterStates[index] with
                {
                    FirstMoment = [],
                };
            }
            else
            {
                _parameterRuntime[index].FirstMoment =
                    _parameterStates[index].FirstMoment;
                _parameterRuntime[index].FirstMomentBFloat16 = null;
            }
            if (_options.UseBFloat16SecondMoment
                && !bfp8State
                && !mix8State)
            {
                _parameterRuntime[index].SecondMoment = [];
                _parameterRuntime[index].SecondMomentBFloat16 =
                    EncodeBFloat16(_parameterStates[index].SecondMoment);
                _parameterStates[index] = _parameterStates[index] with
                {
                    SecondMoment = [],
                };
            }
            else
            {
                _parameterRuntime[index].SecondMoment =
                    _parameterStates[index].SecondMoment;
                _parameterRuntime[index].SecondMomentBFloat16 = null;
            }
        }
        RefreshWeightDecayFlags();
    }

    internal void RestoreStateOwned(AdamWState state)
        => RestoreState(state, takeOwnership: true);

    public OptimizerStateDictionary state_dict()
        => OptimizerStateDictionary.Create("AdamW", CaptureState());

    public void load_state_dict(OptimizerStateDictionary state)
    {
        ArgumentNullException.ThrowIfNull(state);
        // The deserializer already created arrays owned by this restore. Do
        // not duplicate all first/second moments before assigning them.
        RestoreState(
            state.Read<AdamWState>("AdamW"),
            takeOwnership: true);
    }
}
