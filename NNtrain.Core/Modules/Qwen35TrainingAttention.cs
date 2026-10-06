using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>
/// Full-sequence causal GQA with frozen Q/K normalization weights. Inputs remain
/// caller-owned and must stay unchanged until backward completes. This object owns
/// its output and saved activations; backward adds to the caller's gradient buffers.
/// </summary>
internal sealed class Qwen35TrainingAttention : IDisposable
{
    private const int GroupSize = 128;
    private readonly ArcExecutionLane _lane;
    private readonly ArcBuffer _qAndGate, _key, _value, _qNorm, _kNorm;
    private readonly Qwen35GgufDescriptor _descriptor;
    private readonly int _sequence, _queryElements, _kvElements, _scoreElements;
    private readonly long _denseScoreElements;
    private readonly bool _packedScores;
    private readonly bool _fusedRows;
    private readonly bool _fusedOutput;
    private readonly int _streamedTileRows;
    private readonly List<ArcBuffer> _owned = [];
    private readonly ArcBuffer _query = null!, _normalizedKey = null!, _probabilities = null!, _context = null!;
    private bool _disposed;

    internal ArcBuffer Output { get; } = null!;

    internal Qwen35TrainingAttention(ArcExecutionLane lane, ArcBuffer qAndGate,
        ArcBuffer key, ArcBuffer value, ArcBuffer qNorm, ArcBuffer kNorm,
        Qwen35GgufDescriptor d, int sequence, bool packedScores = false,
        int streamedTileRows = 0, bool fusedRows = false,
        bool fusedOutput = false)
    {
        ArgumentNullException.ThrowIfNull(lane);
        ArgumentNullException.ThrowIfNull(d);
        if (sequence <= 0 || sequence > d.ContextLength)
            throw new ArgumentOutOfRangeException(nameof(sequence));
        if (d.HeadCount <= 0 || d.KvHeadCount <= 0 || d.HeadCount % d.KvHeadCount != 0
            || d.HeadWidth <= 0 || d.RopeDimensionCount < 0
            || d.RopeDimensionCount > d.HeadWidth || (d.RopeDimensionCount & 1) != 0)
            throw new ArgumentException("Invalid Qwen3.5 attention dimensions.", nameof(d));
        if (!float.IsFinite(d.RmsEpsilon) || d.RmsEpsilon < 0
            || !float.IsFinite(d.RopeTheta) || d.RopeTheta <= 0)
            throw new ArgumentException("Invalid Qwen3.5 normalization epsilon or RoPE theta.", nameof(d));
        if (streamedTileRows is not (0 or 64 or 128 or 256 or 512 or 1024))
            throw new ArgumentOutOfRangeException(nameof(streamedTileRows));
        if (fusedRows && !packedScores)
            throw new ArgumentException("Fused attention rows require packed scores.", nameof(fusedRows));
        if (fusedOutput && !fusedRows)
            throw new ArgumentException("Fused attention output requires fused rows.", nameof(fusedOutput));
        _queryElements = checked(sequence * d.HeadCount * d.HeadWidth);
        _kvElements = checked(sequence * d.KvHeadCount * d.HeadWidth);
        _denseScoreElements = checked((long)sequence * d.HeadCount * sequence);
        _streamedTileRows = Math.Min(streamedTileRows, sequence);
        _scoreElements = _streamedTileRows > 0
            ? checked(_streamedTileRows * d.HeadCount * sequence)
            : packedScores
                ? checked((int)((long)sequence * (sequence + 1) / 2 * d.HeadCount))
                : checked((int)_denseScoreElements);
        _packedScores = packedScores;
        _fusedRows = fusedRows;
        _fusedOutput = fusedOutput;
        ValidateBuffer(qAndGate, checked(2 * _queryElements), nameof(qAndGate));
        ValidateBuffer(key, _kvElements, nameof(key));
        ValidateBuffer(value, _kvElements, nameof(value));
        ValidateBuffer(qNorm, d.HeadWidth, nameof(qNorm));
        ValidateBuffer(kNorm, d.HeadWidth, nameof(kNorm));
        (_lane, _qAndGate, _key, _value, _qNorm, _kNorm, _descriptor, _sequence)
            = (lane, qAndGate, key, value, qNorm, kNorm, d, sequence);
        try
        {
            _query = Own(_queryElements);
            _normalizedKey = Own(_kvElements);
            _probabilities = Own(_scoreElements);
            _context = Own(_queryElements);
            Output = Own(_queryElements);
            lane.Run("q35a_rms_norm", (long)sequence * d.HeadCount * GroupSize, GroupSize,
                qAndGate, qNorm, _query, d.HeadWidth, checked(2 * d.HeadWidth), d.RmsEpsilon);
            lane.Run("q35a_rms_norm", (long)sequence * d.KvHeadCount * GroupSize, GroupSize,
                key, kNorm, _normalizedKey, d.HeadWidth, d.HeadWidth, d.RmsEpsilon);
            Rotate(_query, d.HeadCount, inverse: false);
            Rotate(_normalizedKey, d.KvHeadCount, inverse: false);
            if (_streamedTileRows > 0)
                ForwardStreamed();
            else if (_fusedRows)
            {
                if (_fusedOutput)
                    lane.Run("q35t_attn_score_softmax_output_rowfused",
                        (long)sequence * d.HeadCount * GroupSize, GroupSize,
                        _query, _normalizedKey, value, qAndGate,
                        _probabilities, _context, Output,
                        sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth);
                else
                {
                    lane.Run("q35t_attn_score_softmax_rowfused",
                        (long)sequence * d.HeadCount * GroupSize, GroupSize,
                        _query, _normalizedKey, _probabilities,
                        sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth);
                    lane.Run("q35t_attn_output_packed", _queryElements, GroupSize,
                        _probabilities, value, qAndGate, _context, Output,
                        sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth);
                }
            }
            else
            {
                lane.Run(AttentionKernel("q35t_attn_scores"), _denseScoreElements, GroupSize,
                    _query, _normalizedKey, _probabilities, sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth);
                lane.Run(AttentionKernel("q35t_attn_softmax"), (long)sequence * d.HeadCount * GroupSize, GroupSize,
                    _probabilities, sequence, d.HeadCount);
                lane.Run(AttentionKernel("q35t_attn_output"), _queryElements, GroupSize,
                    _probabilities, value, qAndGate, _context, Output,
                    sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth);
            }
        }
        catch { Dispose(); throw; }
    }

    internal void Backward(ArcBuffer dy, ArcBuffer dqAndGate, ArcBuffer dk, ArcBuffer dv)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ValidateBuffer(dy, _queryElements, nameof(dy));
        ValidateBuffer(dqAndGate, checked(2 * _queryElements), nameof(dqAndGate));
        ValidateBuffer(dk, _kvElements, nameof(dk));
        ValidateBuffer(dv, _kvElements, nameof(dv));
        Qwen35GgufDescriptor d = _descriptor;
        using ArcBuffer dcontext = _lane.Allocate(_queryElements);
        using ArcBuffer dscores = _lane.Allocate(_scoreElements);
        using ArcBuffer dquery = _lane.Allocate(_queryElements);
        using ArcBuffer dkey = _lane.Allocate(_kvElements);
        _lane.Run("q35t_attn_gate_backward", _queryElements, GroupSize,
            dy, _context, _qAndGate, dcontext, dqAndGate, _queryElements, d.HeadWidth);
        if (_streamedTileRows > 0)
            BackwardStreamed(dcontext, dscores, dquery, dkey, dv);
        else if (_fusedRows)
        {
            _lane.Run("q35t_attn_probability_softmax_backward_rowfused",
                (long)_sequence * d.HeadCount * GroupSize, GroupSize,
                dcontext, _value, _probabilities, dscores,
                _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth);
            _lane.Run("q35t_attn_query_backward_packed", _queryElements, GroupSize,
                dscores, _normalizedKey, dquery, _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth);
            _lane.Run("q35t_attn_key_value_backward_packed", _kvElements, GroupSize,
                dscores, _probabilities, _query, dcontext, dkey, dv,
                _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth);
        }
        else
        {
            _lane.Run(AttentionKernel("q35t_attn_probability_backward"), _denseScoreElements, GroupSize,
                dcontext, _value, dscores, _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth);
            _lane.Run(AttentionKernel("q35t_attn_softmax_backward"), (long)_sequence * d.HeadCount * GroupSize, GroupSize,
                _probabilities, dscores, _sequence, d.HeadCount, d.HeadWidth);
            _lane.Run(AttentionKernel("q35t_attn_query_backward"), _queryElements, GroupSize,
                dscores, _normalizedKey, dquery, _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth);
            _lane.Run(AttentionKernel("q35t_attn_key_value_backward"), _kvElements, GroupSize,
                dscores, _probabilities, _query, dcontext, dkey, dv,
                _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth);
        }
        Rotate(dquery, d.HeadCount, inverse: true);
        Rotate(dkey, d.KvHeadCount, inverse: true);
        _lane.Run("q35t_attn_norm_backward", (long)_sequence * d.HeadCount * GroupSize, GroupSize,
            _qAndGate, _qNorm, dquery, dqAndGate, d.HeadWidth, checked(2 * d.HeadWidth), d.RmsEpsilon);
        _lane.Run("q35t_attn_norm_backward", (long)_sequence * d.KvHeadCount * GroupSize, GroupSize,
            _key, _kNorm, dkey, dk, d.HeadWidth, d.HeadWidth, d.RmsEpsilon);
    }

    private void ForwardStreamed()
    {
        Qwen35GgufDescriptor d = _descriptor;
        for (int first = 0; first < _sequence; first += _streamedTileRows)
        {
            int count = Math.Min(_streamedTileRows, _sequence - first);
            if (_fusedRows)
            {
                long rowWork = (long)count * d.HeadCount * GroupSize;
                if (_fusedOutput)
                    _lane.Run("q35t_attn_score_softmax_output_streamed_rowfused", rowWork, GroupSize,
                        _query, _normalizedKey, _value, _qAndGate,
                        _probabilities, _context, Output,
                        _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth, first, count);
                else
                {
                    _lane.Run("q35t_attn_score_softmax_streamed_rowfused", rowWork, GroupSize,
                        _query, _normalizedKey, _probabilities,
                        _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth, first, count);
                    _lane.Run("q35t_attn_output_streamed", (long)count * d.HeadCount * d.HeadWidth, GroupSize,
                        _probabilities, _value, _qAndGate, _context, Output,
                        _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth, first, count);
                }
                continue;
            }
            _lane.Run("q35t_attn_scores_streamed", (long)count * d.HeadCount * (first + count), GroupSize,
                _query, _normalizedKey, _probabilities,
                _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth, first, count);
            _lane.Run("q35t_attn_softmax_streamed", (long)count * d.HeadCount * GroupSize, GroupSize,
                _probabilities, _sequence, d.HeadCount, first, count);
            _lane.Run("q35t_attn_output_streamed", (long)count * d.HeadCount * d.HeadWidth, GroupSize,
                _probabilities, _value, _qAndGate, _context, Output,
                _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth, first, count);
        }
    }

    private void BackwardStreamed(ArcBuffer dcontext, ArcBuffer dscores,
        ArcBuffer dquery, ArcBuffer dkey, ArcBuffer dv)
    {
        Qwen35GgufDescriptor d = _descriptor;
        for (int first = 0; first < _sequence; first += _streamedTileRows)
        {
            int count = Math.Min(_streamedTileRows, _sequence - first);
            if (_fusedRows)
            {
                _lane.Run("q35t_attn_backward_streamed_rowfused",
                    (long)count * d.HeadCount * GroupSize, GroupSize,
                    _query, _normalizedKey, _value, dcontext,
                    _probabilities, dscores, dquery,
                    _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth, first, count);
                _lane.Run("q35t_attn_key_value_backward_streamed", _kvElements, GroupSize,
                    dscores, _probabilities, _query, dcontext, dkey, dv,
                    _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth, first, count, first == 0 ? 0 : 1);
                continue;
            }
            long scoreWork = (long)count * d.HeadCount * (first + count);
            _lane.Run("q35t_attn_scores_streamed", scoreWork, GroupSize,
                _query, _normalizedKey, _probabilities,
                _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth, first, count);
            _lane.Run("q35t_attn_softmax_streamed", (long)count * d.HeadCount * GroupSize, GroupSize,
                _probabilities, _sequence, d.HeadCount, first, count);
            _lane.Run("q35t_attn_probability_backward_streamed", scoreWork, GroupSize,
                dcontext, _value, dscores,
                _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth, first, count);
            _lane.Run("q35t_attn_softmax_backward_streamed", (long)count * d.HeadCount * GroupSize, GroupSize,
                _probabilities, dscores, _sequence, d.HeadCount, d.HeadWidth, first, count);
            _lane.Run("q35t_attn_query_backward_streamed", (long)count * d.HeadCount * d.HeadWidth, GroupSize,
                dscores, _normalizedKey, dquery,
                _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth, first, count);
            _lane.Run("q35t_attn_key_value_backward_streamed", _kvElements, GroupSize,
                dscores, _probabilities, _query, dcontext, dkey, dv,
                _sequence, d.HeadCount, d.KvHeadCount, d.HeadWidth, first, count, first == 0 ? 0 : 1);
        }
    }

    private string AttentionKernel(string name) => _packedScores ? name + "_packed" : name;

    private void Rotate(ArcBuffer buffer, int heads, bool inverse)
    {
        Qwen35GgufDescriptor d = _descriptor;
        if (d.RopeDimensionCount == 0) return;
        _lane.Run("q35t_attn_rope", (long)_sequence * heads * (d.RopeDimensionCount / 2), GroupSize,
            buffer, _sequence, heads, d.HeadWidth, d.RopeDimensionCount, d.RopeTheta, inverse ? 1 : 0);
    }

    private ArcBuffer Own(int elements)
    {
        ArcBuffer buffer = _lane.Allocate(elements);
        _owned.Add(buffer);
        return buffer;
    }

    private static void ValidateBuffer(ArcBuffer buffer, int elements, string name)
    {
        ArgumentNullException.ThrowIfNull(buffer, name);
        if (!buffer.IsAlive || buffer.ByteLength < (long)elements * sizeof(float))
            throw new ArgumentException("The Arc buffer is disposed or too small for its declared dimensions.", name);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (ArcBuffer buffer in _owned) buffer.Dispose();
        _owned.Clear();
    }
}
