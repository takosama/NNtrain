using System.Diagnostics;
using NNtrain.Cuda.Interop;

namespace NNtrain;

public sealed partial class NekoMuon
{
    public NekoMuonState CaptureState()
    {
        NekoMuonState state = CaptureStateForStreaming();
        return _usesPackedBFloat16Checkpoint ? state : CloneState(state);
    }

    internal NekoMuonState CaptureStateForStreaming()
    {
        _arcState.Synchronize();
        if (_usesPackedBFloat16Checkpoint)
        {
            var states = new NekoMuonParameterState[_state.ParameterStates.Length];
            for (int index = 0; index < states.Length; index++)
            {
                NekoMuonParameterState parameter = _state.ParameterStates[index];
                states[index] = parameter with
                {
                    Shape = parameter.Shape.ToArray(),
                    FastMoment = DecodeArcPackedMoment(_arcPackedFastMoments[index]),
                    SlowMoment = DecodeArcPackedMoment(_arcPackedSlowMoments[index]),
                };
            }
            return _state with { ParameterStates = states, Options = _state.Options with { } };
        }
        if (_cudaStateAuthorityDevice is int primaryDevice)
        {
            for (int index = 0; index < _cudaStates.Length; index++)
            {
                CudaOptimizerKernels.NekoMuonResidentState? state =
                    _cudaStates[index];
                state?.SynchronizeHost(primaryDevice);
                // Generic Float32/BF16 CUDA computes confidence from the
                // gathered host statistics and stores it in _state below the
                // step. Its device Confidence slot is only authoritative for
                // mix8_32's device-control path. Do not overwrite a current
                // host value with the generic slot's construction-time zero.
                if (state is not null
                    && state.IsDeviceConfidenceAuthoritative)
                {
                    NekoMuonParameterState parameterState =
                        _state.ParameterStates[index];
                    _state.ParameterStates[index] = parameterState with
                    {
                        Confidence = state.SynchronizeConfidence(primaryDevice),
                    };
                }
            }
            for (int index = 0; index < _cudaBFloat16States.Length; index++)
            {
                CudaOptimizerKernels.NekoMuonBFloat16ResidentState? state =
                    _cudaBFloat16States[index];
                state?.SynchronizeHost(primaryDevice);
                if (state is not null
                    && state.IsDeviceConfidenceAuthoritative)
                {
                    NekoMuonParameterState parameterState =
                        _state.ParameterStates[index];
                    _state.ParameterStates[index] = parameterState with
                    {
                        Confidence = state.SynchronizeConfidence(primaryDevice),
                    };
                }
            }
            for (int index = 0; index < _cudaBfp8States.Length; index++)
            {
                CudaOptimizerKernels.NekoMuonBfp8ResidentState? state =
                    _cudaBfp8States[index];
                state?.SynchronizeHost(primaryDevice);
                if (state is not null)
                {
                    NekoMuonParameterState parameterState =
                        _state.ParameterStates[index];
                    _state.ParameterStates[index] = parameterState with
                    {
                        Confidence = state.SynchronizeConfidence(primaryDevice),
                    };
                }
            }
        }
        return _state;
    }

    internal NekoMuonState GetPackedStreamingState()
    {
        if (!_usesPackedBFloat16Checkpoint)
            throw new InvalidOperationException("This NekoMuon has no packed BF16 state.");
        _arcState.Synchronize();
        return _state;
    }

    internal (ushort[] Fast, ushort[] Slow) GetStreamingPackedMoments(int index)
        => (_arcPackedFastMoments[index], _arcPackedSlowMoments[index]);

    private static ushort[] EncodeArcPackedMoment(float[] values)
    {
        var packed = new ushort[values.Length];
        for (int i = 0; i < values.Length; i++)
            packed[i] = TensorStorageCodec.EncodeBFloat16(values[i]);
        return packed;
    }

    private static float[] DecodeArcPackedMoment(ushort[] values)
    {
        var decoded = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
            decoded[i] = TensorStorageCodec.DecodeBFloat16(values[i]);
        return decoded;
    }

    public void RestoreState(NekoMuonState state)
        => RestoreState(state, takeOwnership: false);

    private void RestoreState(
        NekoMuonState state,
        bool takeOwnership)
    {
        ArgumentNullException.ThrowIfNull(state);
        ValidateState(state);
        NekoMuonState upgraded = UpgradeState(state);
        DisposeCudaResources();
        if (_usesPackedBFloat16Checkpoint)
        {
            var packedStates = new NekoMuonParameterState[upgraded.ParameterStates.Length];
            for (int index = 0; index < packedStates.Length; index++)
            {
                NekoMuonParameterState parameter = upgraded.ParameterStates[index];
                _arcPackedFastMoments[index] = EncodeArcPackedMoment(parameter.FastMoment);
                _arcPackedSlowMoments[index] = EncodeArcPackedMoment(parameter.SlowMoment);
                packedStates[index] = parameter with
                {
                    Shape = parameter.Shape.ToArray(),
                    FastMoment = [],
                    SlowMoment = [],
                };
            }
            _state = upgraded with
            {
                Options = upgraded.Options with { },
                ParameterStates = packedStates,
            };
            return;
        }
        _state = takeOwnership ? upgraded : CloneState(upgraded);
    }

    internal void RestoreStateOwned(NekoMuonState state)
        => RestoreState(state, takeOwnership: true);

    public OptimizerStateDictionary state_dict()
        => OptimizerStateDictionary.Create("NekoMuon", CaptureState());

    public void load_state_dict(OptimizerStateDictionary state)
    {
        ArgumentNullException.ThrowIfNull(state);
        // Read<T> creates a private object graph from serialized JSON. Taking
        // ownership avoids immediately cloning every moment array during a
        // checkpoint restore, which otherwise doubles the resume peak.
        RestoreState(
            state.Read<NekoMuonState>("NekoMuon"),
            takeOwnership: true);
    }

}
