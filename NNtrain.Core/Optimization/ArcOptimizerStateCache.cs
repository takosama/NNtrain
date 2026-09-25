using NNtrain.Arc;
using NNtrain.Runtime.Execution;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>Session-bounded optimizer state; host arrays are refreshed only at explicit handoffs.</summary>
internal sealed class ArcOptimizerStateCache : IDisposable
{
    private readonly Dictionary<float[], CachedState> _values = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<ushort[], ArcBuffer> _packedValues = new(ReferenceEqualityComparer.Instance);
    private ArcExecutionLane? _lane;
    private IDisposable? _registration;

    internal ArcBuffer Get(float[] values) => GetCore(values, packedBFloat16: false);

    /// <summary>Returns a physically packed BF16 moment buffer (two bytes per value).</summary>
    internal ArcBuffer GetBFloat16(float[] values) => GetCore(values, packedBFloat16: true);

    internal ArcBuffer GetBFloat16(ushort[] values)
    {
        var lane = Tensor.ArcLane;
        if (_lane is not null && !ReferenceEquals(_lane, lane)) Dispose();
        if (_lane is null)
        {
            _lane = lane;
            _registration = ExecutionSession.Current!.RegisterBeforeDispose(this, static owner => ((ArcOptimizerStateCache)owner).Dispose());
        }
        if (!_packedValues.TryGetValue(values, out ArcBuffer? buffer))
            _packedValues.Add(values, buffer = lane.UploadRaw(values));
        return buffer;
    }

    private ArcBuffer GetCore(float[] values, bool packedBFloat16)
    {
        var lane = Tensor.ArcLane;
        if (_lane is not null && !ReferenceEquals(_lane, lane)) Dispose();
        if (_lane is null)
        {
            _lane = lane;
            _registration = ExecutionSession.Current!.RegisterBeforeDispose(this, static owner => ((ArcOptimizerStateCache)owner).Dispose());
        }
        if (!_values.TryGetValue(values, out CachedState? state))
        {
            ArcBuffer buffer = packedBFloat16 ? lane.UploadRaw(Pack(values)) : lane.Upload(values);
            try
            {
                if (!packedBFloat16 && TensorExecutionContext.ActivePrecisionPolicy?.Mode == PrecisionMode.Mix8_16)
                    lane.Run("round_bf16_values", values.Length, 0, buffer, values.Length);
                state = new CachedState(buffer, packedBFloat16);
                _values.Add(values, state);
            }
            catch { buffer.Dispose(); throw; }
        }
        else if (state.PackedBFloat16 != packedBFloat16)
        {
            throw new InvalidOperationException("Arc optimizer moment buffer format changed within one execution session.");
        }
        return state.Buffer;
    }

    internal void Synchronize()
    {
        if (_lane is null) return;
        foreach (var (host, state) in _values)
        {
            if (state.PackedBFloat16)
            {
                var packed = new ushort[host.Length];
                _lane.ReadRaw(state.Buffer, packed);
                for (int i = 0; i < host.Length; i++)
                    host[i] = BitConverter.UInt32BitsToSingle((uint)packed[i] << 16);
            }
            else _lane.Read(state.Buffer, host);
        }
        foreach (var (host, device) in _packedValues) _lane.ReadRaw(device, host);
    }

    private static ushort[] Pack(float[] values)
    {
        var packed = new ushort[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            float rounded = TensorStorageCodec.RoundToBFloat16(values[i]);
            packed[i] = (ushort)(BitConverter.SingleToUInt32Bits(rounded) >> 16);
        }
        return packed;
    }

    public void Dispose()
    {
        try { Synchronize(); }
        finally
        {
            foreach (CachedState state in _values.Values) state.Buffer.Dispose();
            foreach (ArcBuffer buffer in _packedValues.Values) buffer.Dispose();
            _values.Clear(); _packedValues.Clear(); _lane = null;
            _registration?.Dispose(); _registration = null;
        }
    }

    private sealed record CachedState(ArcBuffer Buffer, bool PackedBFloat16);
}
