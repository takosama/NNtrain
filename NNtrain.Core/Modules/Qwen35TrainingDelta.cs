using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>Zero-state sequence DeltaNet with exact recurrent and convolution BPTT.</summary>
internal sealed class Qwen35TrainingDelta : IDisposable
{
    private readonly ArcExecutionLane _lane;
    private readonly ArcBuffer _gate, _alpha, _beta, _convWeights, _dt, _a, _norm;
    private readonly ArcBuffer _pre, _mixed, _raw;
    private readonly int _sequence, _keys, _heads, _width, _kernel, _channels, _values, _stateSize;
    private readonly float _eps;
    private bool _disposed;
    internal ArcBuffer Output { get; }

    internal Qwen35TrainingDelta(ArcExecutionLane lane, ArcBuffer qkv, ArcBuffer gate,
        ArcBuffer alpha, ArcBuffer beta, ArcBuffer convWeights, ArcBuffer dt, ArcBuffer a,
        ArcBuffer norm, Qwen35GgufDescriptor d, int sequence)
    {
        ArgumentNullException.ThrowIfNull(lane);
        ArgumentNullException.ThrowIfNull(d);
        if (sequence <= 0 || d.LinearKeyHeads <= 0 || d.LinearValueHeads <= 0
            || d.LinearValueHeads % d.LinearKeyHeads != 0 || d.LinearHeadWidth <= 0
            || d.ConvKernel <= 0 || !float.IsFinite(d.RmsEpsilon) || d.RmsEpsilon <= 0f)
            throw new ArgumentException("Invalid Gated DeltaNet training dimensions.");
        _lane = lane; _sequence = sequence; _keys = d.LinearKeyHeads;
        _heads = d.LinearValueHeads; _width = d.LinearHeadWidth; _kernel = d.ConvKernel;
        _eps = d.RmsEpsilon;
        _values = checked(_heads * _width);
        _channels = checked((2 * _keys + _heads) * _width);
        _stateSize = checked(_values * _width);
        // ArcBuffer uses Int32 element counts; fail before allocating any saved activations.
        _ = checked((sequence + 1) * _stateSize);
        Check(qkv, checked(sequence * _channels), nameof(qkv));
        Check(gate, checked(sequence * _values), nameof(gate));
        Check(alpha, checked(sequence * _heads), nameof(alpha));
        Check(beta, checked(sequence * _heads), nameof(beta));
        Check(convWeights, checked(_channels * _kernel), nameof(convWeights));
        Check(dt, _heads, nameof(dt)); Check(a, _heads, nameof(a)); Check(norm, _width, nameof(norm));
        _gate = gate; _alpha = alpha; _beta = beta; _convWeights = convWeights;
        _dt = dt; _a = a; _norm = norm;
        var owned = new List<ArcBuffer>();
        ArcBuffer Allocate(int count) { var buffer = lane.Allocate(count); owned.Add(buffer); return buffer; }
        try
        {
            _pre = Allocate(checked(sequence * _channels));
            _mixed = Allocate(checked(sequence * _channels));
            _raw = Allocate(checked(sequence * _values));
            Output = Allocate(checked(sequence * _values));
            using ArcBuffer state = lane.Allocate(_stateSize);
            lane.Run("q35t_delta_conv", sequence * _channels, 0,
                qkv, convWeights, _pre, _mixed, sequence, _channels, _kernel);
            lane.Run("q35t_delta_qk", sequence * _keys, 0,
                _mixed, sequence, _keys, _heads, _width, _eps);
            Recurrent(state, false);
            lane.Run("q35t_delta_gate", sequence * _heads, 0,
                _raw, gate, norm, Output, sequence, _heads, _width, _eps);
        }
        catch
        {
            foreach (ArcBuffer buffer in owned) buffer.Dispose();
            throw;
        }
    }

    internal void Backward(ArcBuffer dy, ArcBuffer dqkv, ArcBuffer dgate, ArcBuffer dalpha, ArcBuffer dbeta)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Check(dy, checked(_sequence * _values), nameof(dy));
        Check(dqkv, checked(_sequence * _channels), nameof(dqkv));
        Check(dgate, checked(_sequence * _values), nameof(dgate));
        Check(dalpha, checked(_sequence * _heads), nameof(dalpha));
        Check(dbeta, checked(_sequence * _heads), nameof(dbeta));
        // The large recurrent tape exists only while this one layer is backpropagated.
        using ArcBuffer states = _lane.Allocate(checked((_sequence + 1) * _stateSize));
        using ArcBuffer adj = _lane.Allocate(_stateSize);
        using ArcBuffer scratch = _lane.Allocate(checked(_values * 4));
        using ArcBuffer dmixed = _lane.Allocate(checked(_sequence * _channels));
        using ArcBuffer draw = _lane.Allocate(checked(_sequence * _values));
        Recurrent(states, true);
        _lane.Run("q35a_zero", _stateSize, 0, adj, _stateSize);
        _lane.Run("q35t_delta_gate_backward", _sequence * _heads, 0,
            _raw, _gate, _norm, dy, draw, dgate, _sequence, _heads, _width, _eps);
        for (int t = _sequence - 1; t >= 0; --t)
        {
            _lane.Run("q35t_delta_prepare", _values, 0,
                _mixed, _alpha, _beta, _dt, _a, states, draw, adj, scratch, dmixed,
                t, _keys, _heads, _width);
            bool cooperative = _lane.Options.Qwen35CooperativeDelta && _width >= 32;
            _lane.Run(cooperative ? "q35t_delta_qk_backward_step_cooperative" : "q35t_delta_qk_backward_step",
                (long)_keys * _width * (cooperative ? 32 : 1), cooperative ? 32 : 0,
                _alpha, _dt, _a, states, draw, adj, scratch, dmixed, t, _keys, _heads, _width);
            _lane.Run("q35t_delta_finish", _values, 0,
                _mixed, _alpha, _beta, _dt, _a, scratch, adj, dalpha, dbeta,
                t, _keys, _heads, _width);
        }
        _lane.Run("q35t_delta_qk_backward", _sequence * _keys * 2, 0,
            _pre, dmixed, _sequence, _keys, _heads, _width, _eps);
        _lane.Run("q35t_delta_conv_backward", _sequence * _channels, 0,
            _pre, _convWeights, dmixed, dqkv, _sequence, _channels, _kernel);
    }

    private void Recurrent(ArcBuffer state, bool tape)
        => _lane.Run("q35t_delta_recur", _values, 0,
            _mixed, _alpha, _beta, _dt, _a, state, _raw,
            _sequence, _keys, _heads, _width, tape ? 1 : 0);

    private static void Check(ArcBuffer buffer, int elements, string name)
    {
        ArgumentNullException.ThrowIfNull(buffer, name);
        if (!buffer.IsAlive || buffer.ByteLength < checked((long)elements * sizeof(float)))
            throw new ArgumentException("DeltaNet training buffer is disposed or smaller than its declared shape.", name);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Output.Dispose(); _raw.Dispose(); _mixed.Dispose(); _pre.Dispose();
    }
}
