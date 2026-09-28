using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>Zero-state sequence DeltaNet with exact recurrent and convolution BPTT.</summary>
internal sealed class Qwen35TrainingDelta : IDisposable
{
    // Only one short slice of the recurrent states is resident during BPTT.
    // The adjoint itself is carried across slices, so this does not truncate it.
    private const int RecurrentChunkSize = 64;
    private readonly ArcExecutionLane _lane;
    private readonly ArcBuffer _gate, _alpha, _beta, _convWeights, _dt, _a, _norm;
    private readonly ArcBuffer _pre, _mixed, _raw;
    private readonly ArcBuffer? _checkpoints;
    private readonly int _sequence, _keys, _heads, _width, _kernel, _channels, _values, _stateSize;
    private readonly float _eps;
    private bool _disposed;
    internal ArcBuffer Output { get; }
    // Test oracle: the original per-token launches remain available for parity checks.
    internal bool UsePersistentBackward { get; set; } = true;

    internal Qwen35TrainingDelta(ArcExecutionLane lane, ArcBuffer qkv, ArcBuffer gate,
        ArcBuffer alpha, ArcBuffer beta, ArcBuffer convWeights, ArcBuffer dt, ArcBuffer a,
        ArcBuffer norm, Qwen35GgufDescriptor d, int sequence,
        bool captureCheckpoints = true)
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
        // ArcBuffer uses Int32 element counts. A full-sequence tape would be
        // several GiB at the 27B model's head dimensions; validate the bounded
        // tape and its sparse boundary checkpoints instead.
        _ = checked((Math.Min(sequence, RecurrentChunkSize) + 1) * _stateSize);
        _ = checked((1 + (sequence - 1) / RecurrentChunkSize) * _stateSize);
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
            // Long sequences retain only chunk boundaries for this one layer.
            // Its caller disposes the tape before advancing to the next layer.
            _checkpoints = captureCheckpoints && sequence > RecurrentChunkSize
                ? Allocate(checked((1 + (sequence - 1) / RecurrentChunkSize) * _stateSize))
                : null;
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
        int chunks = 1 + (_sequence - 1) / RecurrentChunkSize;
        // The checkpoint-building forward pass can omit capture because its
        // tape is disposed without backward. If a caller nevertheless invokes
        // Backward on such a tape, replay boundaries for correctness.
        using ArcBuffer? transientCheckpoints = _checkpoints is null
            ? _lane.Allocate(checked(chunks * _stateSize)) : null;
        ArcBuffer checkpoints = _checkpoints ?? transientCheckpoints!;
        using ArcBuffer states = _lane.Allocate(checked((Math.Min(_sequence, RecurrentChunkSize) + 1) * _stateSize));
        using ArcBuffer adj = _lane.Allocate(_stateSize);
        bool persistent = UsePersistentBackward && _lane.Options.Qwen35CooperativeDelta && _width >= 32;
        int chunkCapacity = Math.Min(_sequence, RecurrentChunkSize);
        using ArcBuffer scratch = _lane.Allocate(checked((persistent ? chunkCapacity : 1) * _values * 4));
        using ArcBuffer? adjSnapshots = persistent ? _lane.Allocate(checked(chunkCapacity * _stateSize)) : null;
        using ArcBuffer dmixed = _lane.Allocate(checked(_sequence * _channels));
        using ArcBuffer draw = _lane.Allocate(checked(_sequence * _values));
        // Replay each chunk from its exact forward boundary. A single-chunk
        // sequence has no interior boundary and only needs its zero state.
        if (_checkpoints is null)
        {
            if (chunks == 1)
                _lane.Run("q35a_zero", _stateSize, 0, checkpoints, _stateSize);
            else
            {
                using ArcBuffer checkpointState = _lane.Allocate(_stateSize);
                _lane.Run("q35t_delta_checkpoints", _values, 0,
                    _mixed, _alpha, _beta, _dt, _a, checkpointState, checkpoints,
                    _sequence, RecurrentChunkSize, _keys, _heads, _width);
            }
        }
        _lane.Run("q35a_zero", _stateSize, 0, adj, _stateSize);
        _lane.Run("q35t_delta_gate_backward", _sequence * _heads, 0,
            _raw, _gate, _norm, Output, dy, draw, dgate,
            _sequence, _heads, _width, _eps);
        for (int chunk = chunks - 1; chunk >= 0; --chunk)
        {
            int first = chunk * RecurrentChunkSize;
            int count = Math.Min(RecurrentChunkSize, _sequence - first);
            _lane.Run("q35t_delta_recur_chunk", _values, 0,
                _mixed, _alpha, _beta, _dt, _a, checkpoints, states,
                first, count, chunk, _keys, _heads, _width);
            if (persistent)
            {
                // One work item carries each adjoint column through all tokens
                // in this chunk. The saved pre-finish adjoints let the following
                // parallel launch reproduce the old q/k and alpha/beta gradients.
                _lane.Run("q35t_delta_reverse_chunk", _values, 0,
                    _mixed, _alpha, _beta, _dt, _a, states, draw, adj,
                    adjSnapshots!, scratch, dmixed, first, count, _keys, _heads, _width);
                long groups = (long)count * (_keys * _width + _heads);
                _lane.Run("q35t_delta_chunk_grad", groups * 32, 32,
                    _alpha, _beta, _dt, _a, states, draw, adjSnapshots!, scratch,
                    dmixed, dalpha, dbeta, first, count, _keys, _heads, _width);
                continue;
            }
            for (int t = first + count - 1; t >= first; --t)
            {
                _lane.Run("q35t_delta_prepare", _values, 0,
                    _mixed, _alpha, _beta, _dt, _a, states, draw, adj, scratch, dmixed,
                    t, first, _keys, _heads, _width);
                bool cooperative = _lane.Options.Qwen35CooperativeDelta && _width >= 32;
                _lane.Run(cooperative ? "q35t_delta_qk_backward_step_cooperative" : "q35t_delta_qk_backward_step",
                    (long)_keys * _width * (cooperative ? 32 : 1), cooperative ? 32 : 0,
                    _alpha, _dt, _a, states, draw, adj, scratch, dmixed, t, first, _keys, _heads, _width);
                _lane.Run("q35t_delta_finish", _values, 0,
                    _mixed, _alpha, _beta, _dt, _a, scratch, adj, dalpha, dbeta,
                    t, _keys, _heads, _width);
            }
        }
        _lane.Run("q35t_delta_qk_backward", _sequence * _keys * 2, 0,
            _pre, dmixed, _sequence, _keys, _heads, _width, _eps);
        _lane.Run("q35t_delta_conv_backward", _sequence * _channels, 0,
            _pre, _convWeights, dmixed, dqkv, _sequence, _channels, _kernel);
    }

    private void Recurrent(ArcBuffer state, bool tape)
        => _lane.Run("q35t_delta_recur", _values, 0,
            _mixed, _alpha, _beta, _dt, _a, state, _checkpoints ?? state, _raw,
            _sequence, RecurrentChunkSize, _keys, _heads, _width,
            tape ? 1 : 0, _checkpoints is null ? 0 : 1);

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
        _checkpoints?.Dispose(); Output.Dispose(); _raw.Dispose(); _mixed.Dispose(); _pre.Dispose();
    }
}
