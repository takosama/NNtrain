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
    internal Action<string, double>? TimingObserver { get; set; }
    public long PeakDeviceBufferBytes => _lane.PeakAllocatedBytes;
    public long ResidentDeviceBytes => _resident.Values.Sum(x => x.ByteLength);
    public string DeviceName => _lane.Device.Name;

    internal AsrArcLinear(IReadOnlyDictionary<string, AsrHalfCheckpoint.Weight> weights, int device, CancellationToken ct,
        Action<string, double>? loadingObserver = null)
    {
        ct.ThrowIfCancellationRequested();
        long started = System.Diagnostics.Stopwatch.GetTimestamp();
        _weights = weights;
        _lane = new(device, new ArcExecutionOptions { AsrKernelsOnly = true, Qwen35InferenceKernelsOnly = true,
            BufferPoolBytes = 4L * 1024 * 1024, DeferredReleaseBytes = 0,
            PhysicalBufferBudgetBytes = 2L * 1024 * 1024 * 1024 });
        try
        {
            _zero = _lane.AllocateBytes(4); _lane.WriteRaw(_zero, new short[2]);
            loadingObserver?.Invoke("gpu.laneInitialization", System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds);
            started = System.Diagnostics.Stopwatch.GetTimestamp();
            double transferBefore = _lane.TransferMilliseconds;
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
            foreach (string name in weights.Keys.Where(x => x.EndsWith(".pos_bias_u", StringComparison.Ordinal) || x.EndsWith(".pos_bias_v", StringComparison.Ordinal)))
            { ct.ThrowIfCancellationRequested(); Upload(name); }
            double total = System.Diagnostics.Stopwatch.GetElapsedTime(started).TotalMilliseconds;
            double transfer = _lane.TransferMilliseconds - transferBefore;
            loadingObserver?.Invoke("gpu.transfer", transfer);
            loadingObserver?.Invoke("gpu.prepareAndAllocate", Math.Max(0, total - transfer));
            ct.ThrowIfCancellationRequested();
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

    private static float[] Flatten(float[][] rows)
    {
        if (rows.Length == 0) return [];
        int width = rows[0].Length;
        if (rows.Any(row => row.Length != width)) throw new InvalidDataException("ASR row shape mismatch.");
        var flat = new float[checked(rows.Length * width)];
        for (int i = 0; i < rows.Length; i++) rows[i].CopyTo(flat, i * width);
        return flat;
    }

    private void RunProjection(ArcExecutionLane.ArcBuffer input, ArcExecutionLane.ArcBuffer output, int rows, string name)
    {
        var descriptor = _weights[name + ".weight"];
        if (descriptor.Shape.Length < 2 || descriptor.Shape.Skip(2).Any(dimension => dimension != 1))
            throw new InvalidDataException("ASR projection requires a matrix.");
        int inputs = descriptor.Shape[1], outputs = descriptor.Shape[0];
        string bias = name + ".bias"; bool hasBias = _weights.ContainsKey(bias);
        if (input.ByteLength < checked((long)rows * inputs * 4) || output.ByteLength < checked((long)rows * outputs * 4)
            || hasBias && _weights[bias].Values.Length != outputs)
            throw new InvalidDataException("ASR projection buffer shape mismatch.");
        _lane.Run("asr_linear", checked((long)rows * outputs * 64), 64,
            input, Upload(name + ".weight"), hasBias ? Upload(bias) : _zero, output, inputs, outputs, hasBias ? 1 : 0);
    }

    internal float[][] FeedForward(float[][] rows, string name, CancellationToken ct)
    {
        if (rows.Length == 0) return [];
        var first = _weights[name + ".linear1.weight"]; var second = _weights[name + ".linear2.weight"];
        int hidden = rows[0].Length;
        if (first.Shape.Length < 2 || second.Shape.Length < 2 || first.Shape[1] != hidden
            || second.Shape[0] != hidden || second.Shape[1] != first.Shape[0])
            throw new InvalidDataException("ASR feed-forward shape mismatch.");
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        using var input = _lane.Upload(Flatten(rows));
        using var intermediate = _lane.Allocate(checked(rows.Length * first.Shape[0]));
        using var output = _lane.Allocate(checked(rows.Length * hidden));
        ct.ThrowIfCancellationRequested(); RunProjection(input, intermediate, rows.Length, name + ".linear1");
        ct.ThrowIfCancellationRequested();
        _lane.Run("asr_silu", checked((long)rows.Length * first.Shape[0]), 64, intermediate, checked(rows.Length * first.Shape[0]));
        ct.ThrowIfCancellationRequested(); RunProjection(intermediate, output, rows.Length, name + ".linear2");
        var flat = new float[checked(rows.Length * hidden)]; _lane.ReadRaw(output, flat);
        TimingObserver?.Invoke("arc.feedForward", System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        ct.ThrowIfCancellationRequested();
        return Enumerable.Range(0, rows.Length).Select(i => flat.AsSpan(i * hidden, hidden).ToArray()).ToArray();
    }

    internal float[][] RelativeAttention(float[][] rows, float[][] positions, string name, int heads, CancellationToken ct)
    {
        int length = rows.Length;
        if (length == 0) return [];
        int hidden = rows[0].Length;
        if (heads < 1 || hidden % heads != 0 || positions.Length != 2 * length - 1 || positions.Any(row => row.Length != hidden))
            throw new InvalidDataException("ASR relative attention shape mismatch.");
        foreach (string projection in new[] { "linear_q", "linear_k", "linear_v", "linear_pos", "linear_out" })
        {
            var matrix = _weights[name + "." + projection + ".weight"];
            if (matrix.Shape.Length < 2 || matrix.Shape[0] != hidden || matrix.Shape[1] != hidden
                || matrix.Shape.Skip(2).Any(dimension => dimension != 1)
                || _weights.TryGetValue(name + "." + projection + ".bias", out var bias) && bias.Values.Length != hidden)
                throw new InvalidDataException("ASR attention projection shape mismatch.");
        }
        if (_weights[name + ".pos_bias_u"].Values.Length != hidden || _weights[name + ".pos_bias_v"].Values.Length != hidden)
            throw new InvalidDataException("ASR relative bias shape mismatch.");
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        using var input = _lane.Upload(Flatten(rows));
        using var positionInput = _lane.Upload(Flatten(positions));
        using var query = _lane.Allocate(checked(length * hidden));
        using var key = _lane.Allocate(checked(length * hidden));
        using var value = _lane.Allocate(checked(length * hidden));
        using var position = _lane.Allocate(checked(positions.Length * hidden));
        using var scores = _lane.Allocate(checked(length * length * heads));
        using var context = _lane.Allocate(checked(length * hidden));
        using var output = _lane.Allocate(checked(length * hidden));
        ct.ThrowIfCancellationRequested(); RunProjection(input, query, length, name + ".linear_q");
        ct.ThrowIfCancellationRequested(); RunProjection(input, key, length, name + ".linear_k");
        ct.ThrowIfCancellationRequested(); RunProjection(input, value, length, name + ".linear_v");
        ct.ThrowIfCancellationRequested(); RunProjection(positionInput, position, positions.Length, name + ".linear_pos");
        ct.ThrowIfCancellationRequested();
        _lane.Run("asr_relative_scores", checked((long)length * length * heads * 64), 64,
            query, key, position, Upload(name + ".pos_bias_u"), Upload(name + ".pos_bias_v"), scores, length, hidden, heads);
        _lane.Run("asr_attention_softmax", checked((long)length * heads * 64), 64, scores, length);
        ct.ThrowIfCancellationRequested();
        _lane.Run("asr_attention_context", checked((long)length * heads * 64), 64, scores, value, context, length, hidden, heads);
        RunProjection(context, output, length, name + ".linear_out");
        var flat = new float[checked(length * hidden)]; _lane.ReadRaw(output, flat);
        TimingObserver?.Invoke("arc.relativeAttention", System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        ct.ThrowIfCancellationRequested();
        return Enumerable.Range(0, length).Select(i => flat.AsSpan(i * hidden, hidden).ToArray()).ToArray();
    }

    private float[][] Project(float[][] rows, string weightName, string biasName)
    {
        if (rows.Length == 0) return [];
        var descriptor = _weights[weightName];
        int inputs = descriptor.Shape[1], outputs = descriptor.Shape[0];
        if (rows.Any(row => row.Length != inputs)) throw new InvalidDataException("Arc ASR linear shape mismatch.");
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        float[] flat = new float[checked(rows.Length * inputs)];
        for (int i = 0; i < rows.Length; i++) rows[i].CopyTo(flat, i * inputs);
        TimingObserver?.Invoke("arc.pack", System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        start = System.Diagnostics.Stopwatch.GetTimestamp();
        using var input = _lane.Upload(flat);
        using var output = _lane.Allocate(checked(rows.Length * outputs));
        TimingObserver?.Invoke("arc.uploadAllocate", System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        bool hasBias = _weights.ContainsKey(biasName);
        start = System.Diagnostics.Stopwatch.GetTimestamp();
        _lane.Run("asr_linear", checked((long)rows.Length * outputs * 64), 64,
            input, Upload(weightName), hasBias ? Upload(biasName) : _zero, output, inputs, outputs, hasBias ? 1 : 0);
        TimingObserver?.Invoke("arc.enqueue", System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        start = System.Diagnostics.Stopwatch.GetTimestamp();
        var result = new float[rows.Length * outputs];
        _lane.ReadRaw(output, result);
        TimingObserver?.Invoke("arc.waitReadback", System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        return Enumerable.Range(0, rows.Length).Select(i => result.AsSpan(i * outputs, outputs).ToArray()).ToArray();
    }

    public void Dispose() => _lane.Dispose();
}
