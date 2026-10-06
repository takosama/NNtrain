using NNtrain.Arc;
using System.Runtime.InteropServices;

namespace NNtrain.Audio;

/// <summary>ASR-owned Arc lane. CPU frontend/nonlinear/cache operations; resident FP16 dense projections on Arc.</summary>
internal sealed class AsrArcLinear : IDisposable
{
    private readonly ArcExecutionLane _lane;
    private readonly IReadOnlyDictionary<string, AsrHalfCheckpoint.Weight> _weights;
    private readonly Dictionary<string, ArcExecutionLane.ArcBuffer> _resident = [];
    private readonly ArcExecutionLane.ArcBuffer _zero;
    public long PeakDeviceBufferBytes => _lane.PeakAllocatedBytes;
    public long ResidentDeviceBytes => _resident.Values.Sum(x => x.ByteLength);
    public string DeviceName => _lane.Device.Name;

    internal AsrArcLinear(IReadOnlyDictionary<string, AsrHalfCheckpoint.Weight> weights, int device, CancellationToken ct)
    {
        _weights = weights;
        _lane = new(device, new ArcExecutionOptions { AsrKernelsOnly = true, Qwen35InferenceKernelsOnly = true,
            BufferPoolBytes = 4L * 1024 * 1024, DeferredReleaseBytes = 0,
            PhysicalBufferBudgetBytes = 2L * 1024 * 1024 * 1024 });
        try
        {
            _zero = _lane.AllocateBytes(4); _lane.WriteRaw(_zero, new short[2]);
            // Populate before recording so the first utterance has no weight-upload latency.
            foreach (var (name, weight) in weights)
                if (name.EndsWith(".weight", StringComparison.Ordinal) && name != "decoder.embedding.weight"
                    && (weight.Shape.Length == 2 || weight.Shape.Length > 2 && weight.Shape.Skip(2).All(x => x == 1)))
                {
                    ct.ThrowIfCancellationRequested(); Upload(name);
                    string bias = name[..^7] + ".bias";
                    if (weights.ContainsKey(bias)) Upload(bias);
                }
            foreach (string name in weights.Keys.Where(x => x.StartsWith("decoder.lstm.", StringComparison.Ordinal)))
            { ct.ThrowIfCancellationRequested(); Upload(name); }
        }
        catch { _lane.Dispose(); throw; }
    }

    private ArcExecutionLane.ArcBuffer Upload(string name)
    {
        if (_resident.TryGetValue(name, out var buffer)) return buffer;
        Half[] values = _weights[name].Values;
        buffer = _lane.AllocateBytes(checked(values.Length * 2));
        try { _lane.WriteRaw(buffer, MemoryMarshal.Cast<Half, ushort>(values.AsSpan()).ToArray()); _resident.Add(name, buffer); return buffer; }
        catch { buffer.Dispose(); throw; }
    }

    internal float[][] Linear(float[][] rows, string name)
        => Project(rows, name + ".weight", name + ".bias");

    internal float[] LinearLstm(float[] row, string weight, string bias) => Project([row], weight, bias)[0];

    private float[][] Project(float[][] rows, string weightName, string biasName)
    {
        if (rows.Length == 0) return [];
        var descriptor = _weights[weightName];
        int inputs = descriptor.Shape[1], outputs = descriptor.Shape[0];
        if (rows.Any(row => row.Length != inputs)) throw new InvalidDataException("Arc ASR linear shape mismatch.");
        float[] flat = new float[checked(rows.Length * inputs)];
        for (int i = 0; i < rows.Length; i++) rows[i].CopyTo(flat, i * inputs);
        using var input = _lane.Upload(flat);
        using var output = _lane.Allocate(checked(rows.Length * outputs));
        bool hasBias = _weights.ContainsKey(biasName);
        _lane.Run("asr_linear", checked((long)rows.Length * outputs * 64), 64,
            input, Upload(weightName), hasBias ? Upload(biasName) : _zero, output, inputs, outputs, hasBias ? 1 : 0);
        var result = new float[rows.Length * outputs];
        _lane.ReadRaw(output, result);
        return Enumerable.Range(0, rows.Length).Select(i => result.AsSpan(i * outputs, outputs).ToArray()).ToArray();
    }

    public void Dispose() => _lane.Dispose();
}
