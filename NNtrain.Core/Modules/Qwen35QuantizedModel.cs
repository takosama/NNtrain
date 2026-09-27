using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>
/// Text-only Qwen3.5 inference. K-quant matrices remain encoded on fixed Arc
/// devices. Small normalization, attention and recurrent state operations use
/// Float32 host arrays. A model instance owns one sequence and is not thread safe.
/// </summary>
public sealed class Qwen35QuantizedModel : IDisposable
{
    private readonly List<ArcExecutionLane> _lanes = [];
    private readonly Dictionary<string, Matrix> _matrices = new(StringComparer.Ordinal);
    private readonly Dictionary<string, float[]> _dense = new(StringComparer.Ordinal);
    private readonly LayerState[] _states;
    private int _position;
    private bool _disposed;

    private Qwen35QuantizedModel(Qwen35GgufDescriptor descriptor)
    {
        Descriptor = descriptor;
        _states = new LayerState[descriptor.LayerCount];
        for (int i = 0; i < _states.Length; i++)
            _states[i] = new LayerState(descriptor, descriptor.IsRecurrent(i));
    }

    public Qwen35GgufDescriptor Descriptor { get; }
    public IReadOnlyList<long> ResidentWeightBytes => _lanes.Select(lane =>
        _matrices.Values.Where(matrix => ReferenceEquals(matrix.Lane, lane))
            .Sum(matrix => (long)matrix.StorageBytes)).ToArray();
    public IReadOnlyList<long> UploadedBytes => _lanes.Select(lane => lane.H2DBytes).ToArray();

    public static Qwen35QuantizedModel Load(
        string path, IReadOnlyList<int>? devices = null, Action<string>? progress = null)
    {
        Qwen35GgufDescriptor d = Qwen35Gguf.Inspect(path);
        ArcDeviceInfo[] available = ArcDevices.Enumerate().ToArray();
        if (available.Length == 0)
            throw new NotSupportedException("Qwen3.5 quantized inference requires Intel Arc.");
        int[] selected = devices?.ToArray() ?? SelectDevices(d, available);
        if (selected.Length == 0 || selected.Length > d.LayerCount
            || selected.Distinct().Count() != selected.Length
            || selected.Any(index => index < 0 || index >= available.Length))
            throw new ArgumentException("Specify distinct, available Arc device indices.", nameof(devices));
        long[] planned = PlanWeightBytes(d, selected.Length);
        for (int slot = 0; slot < selected.Length; slot++)
        {
            ArcDeviceInfo device = available[selected[slot]];
            if (planned[slot] + 256L * 1024 * 1024 > (long)(device.GlobalMemoryBytes * 0.9))
                throw new NotSupportedException(
                    $"Arc {device.Index} needs {planned[slot] / 1073741824.0:F2} GiB of quantized weights " +
                    "plus workspace. Select more GPUs with --devices 0,1.");
            foreach (GgufTensorInfo tensor in d.Tensors.Where(IsQuantized))
                if (DeviceSlot(tensor.Name, d.LayerCount, selected.Length) == slot
                    && (ulong)EncodedBytes(tensor) > device.MaximumAllocationBytes)
                    throw new NotSupportedException($"Tensor '{tensor.Name}' exceeds Arc {device.Index}'s allocation limit.");
        }

        var model = new Qwen35QuantizedModel(d);
        try
        {
            foreach (int index in selected)
                model._lanes.Add(new ArcExecutionLane(index, new ArcExecutionOptions
                {
                    BufferPoolBytes = 64L * 1024 * 1024,
                    DeferredReleaseBytes = 0
                }));
            using var gguf = new GgufReader(path);
            int loaded = 0;
            foreach (GgufTensorInfo tensor in d.Tensors)
            {
                if (IsQuantized(tensor))
                {
                    var lane = model._lanes[DeviceSlot(tensor.Name, d.LayerCount, selected.Length)];
                    // UploadRaw completes the transfer before returning; no dense
                    // expansion or model-sized retained host payload is needed.
                    model._matrices.Add(tensor.Name, new Matrix(lane, tensor,
                        gguf.ReadTensorBytes(tensor, EncodedBytes(tensor))));
                }
                else model._dense.Add(tensor.Name, Qwen2Gguf.ReadTensor(gguf, tensor));
                if (++loaded % 100 == 0) progress?.Invoke($"Loading tensors: {loaded}/{d.Tensors.Count}");
            }
            for (int slot = 0; slot < selected.Length; slot++)
                progress?.Invoke($"Arc {selected[slot]}: quantized weights resident = {model.ResidentWeightBytes[slot]} bytes " +
                    $"({model.ResidentWeightBytes[slot] / 1073741824.0:F2} GiB), " +
                    $"live device allocation = {model._lanes[slot].AllocatedBytes} bytes");
            return model;
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    internal static long[] PlanWeightBytes(Qwen35GgufDescriptor d, int deviceCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(deviceCount);
        var bytes = new long[deviceCount];
        foreach (GgufTensorInfo tensor in d.Tensors.Where(IsQuantized))
            bytes[DeviceSlot(tensor.Name, d.LayerCount, deviceCount)] += EncodedBytes(tensor);
        return bytes;
    }

    private static int[] SelectDevices(Qwen35GgufDescriptor d, ArcDeviceInfo[] available)
    {
        for (int count = 1; count <= Math.Min(available.Length, d.LayerCount); count++)
        {
            long[] bytes = PlanWeightBytes(d, count);
            if (bytes.Select((value, slot) => value + 256L * 1024 * 1024
                <= (long)(available[slot].GlobalMemoryBytes * 0.9)).All(fits => fits))
                return Enumerable.Range(0, count).ToArray();
        }
        throw new NotSupportedException("The quantized Qwen3.5 weights exceed the available Arc VRAM budget.");
    }

    private static int DeviceSlot(string name, int layers, int devices)
    {
        if (name.StartsWith("blk.", StringComparison.Ordinal))
        {
            int end = name.IndexOf('.', 4);
            int layer = int.Parse(name.AsSpan(4, end - 4));
            return Math.Min(devices - 1, layer * devices / layers);
        }
        return name == "token_embd.weight" ? 0 : devices - 1;
    }

    private static bool IsQuantized(GgufTensorInfo tensor)
        => tensor.Type is Qwen2Gguf.Q4KType or Qwen2Gguf.Q6KType;

    private static int EncodedBytes(GgufTensorInfo tensor)
        => checked((int)(tensor.Shape[0] / 256 * tensor.Shape[1]
            * (tensor.Type == Qwen2Gguf.Q4KType ? 144UL : 210UL)));

    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _position = 0;
        foreach (LayerState state in _states) state.Reset();
    }

    /// <summary>Advances recurrent/KV state by one token; returns final logits when requested.</summary>
    public float[] ForwardToken(int tokenId, bool returnLogits = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Qwen35GgufDescriptor d = Descriptor;
        if ((uint)tokenId >= (uint)d.VocabularySize) throw new ArgumentOutOfRangeException(nameof(tokenId));
        if (_position >= d.ContextLength) throw new InvalidOperationException("Qwen3.5 context length exceeded.");
        float[] hidden = _matrices["token_embd.weight"].Embedding(tokenId);
        for (int layer = 0; layer < d.LayerCount; layer++)
        {
            string p = $"blk.{layer}.";
            float[] normalized = Qwen35Math.RmsNorm(hidden, _dense[p + "attn_norm.weight"], d.RmsEpsilon);
            float[] attention;
            if (d.IsRecurrent(layer))
            {
                LayerState state = _states[layer];
                float[] delta = Qwen35Math.DeltaStep(
                    Project(p + "attn_qkv.weight", normalized),
                    Project(p + "attn_gate.weight", normalized),
                    Project(p + "ssm_alpha.weight", normalized),
                    Project(p + "ssm_beta.weight", normalized),
                    _dense[p + "ssm_conv1d.weight"], _dense[p + "ssm_dt.bias"],
                    _dense[p + "ssm_a"], _dense[p + "ssm_norm.weight"],
                    state.Convolution, state.Recurrent, d.LinearKeyHeads, d.LinearValueHeads,
                    d.LinearHeadWidth, d.ConvKernel, d.RmsEpsilon);
                attention = Project(p + "ssm_out.weight", delta);
            }
            else attention = FullAttention(normalized, p, _states[layer]);
            for (int i = 0; i < hidden.Length; i++) hidden[i] += attention[i];
            normalized = Qwen35Math.RmsNorm(hidden, _dense[p + "post_attention_norm.weight"], d.RmsEpsilon);
            float[] gate = Project(p + "ffn_gate.weight", normalized);
            float[] up = Project(p + "ffn_up.weight", normalized);
            for (int i = 0; i < gate.Length; i++) gate[i] = Qwen35Math.Silu(gate[i]) * up[i];
            float[] down = Project(p + "ffn_down.weight", gate);
            for (int i = 0; i < hidden.Length; i++) hidden[i] += down[i];
        }
        _position++;
        if (!returnLogits) return [];
        hidden = Qwen35Math.RmsNorm(hidden, _dense["output_norm.weight"], d.RmsEpsilon);
        float[] logits = Project(_matrices.ContainsKey("output.weight") ? "output.weight" : "token_embd.weight", hidden);
        if (logits.Any(value => !float.IsFinite(value)))
            throw new ArithmeticException("Qwen3.5 produced non-finite logits.");
        return logits;
    }

    private float[] Project(string name, float[] input) => _matrices[name].Forward(input);

    private float[] FullAttention(float[] input, string p, LayerState state)
    {
        Qwen35GgufDescriptor d = Descriptor;
        float[] qAndGate = Project(p + "attn_q.weight", input);
        float[] key = Project(p + "attn_k.weight", input);
        float[] value = Project(p + "attn_v.weight", input);
        var query = new float[d.HeadCount * d.HeadWidth];
        var gate = new float[query.Length];
        for (int h = 0; h < d.HeadCount; h++)
        {
            Array.Copy(qAndGate, h * 2 * d.HeadWidth, query, h * d.HeadWidth, d.HeadWidth);
            Array.Copy(qAndGate, (h * 2 + 1) * d.HeadWidth, gate, h * d.HeadWidth, d.HeadWidth);
        }
        query = Qwen35Math.RmsNorm(query, _dense[p + "attn_q_norm.weight"], d.RmsEpsilon, d.HeadWidth);
        key = Qwen35Math.RmsNorm(key, _dense[p + "attn_k_norm.weight"], d.RmsEpsilon, d.HeadWidth);
        Qwen35Math.Rope(query, d.HeadCount, d.HeadWidth, d.RopeDimensionCount, _position, d.RopeTheta);
        Qwen35Math.Rope(key, d.KvHeadCount, d.HeadWidth, d.RopeDimensionCount, _position, d.RopeTheta);
        state.Keys.Add(key);
        state.Values.Add(value);
        var output = new float[query.Length];
        var scores = new float[state.Keys.Count];
        float scale = 1f / MathF.Sqrt(d.HeadWidth);
        for (int h = 0; h < d.HeadCount; h++)
        {
            int qStart = h * d.HeadWidth;
            int kvStart = h / (d.HeadCount / d.KvHeadCount) * d.HeadWidth;
            float maximum = float.NegativeInfinity;
            for (int t = 0; t < scores.Length; t++)
            {
                float dot = 0;
                for (int j = 0; j < d.HeadWidth; j++) dot += query[qStart + j] * state.Keys[t][kvStart + j];
                scores[t] = dot * scale;
                maximum = MathF.Max(maximum, scores[t]);
            }
            float sum = 0;
            for (int t = 0; t < scores.Length; t++) sum += scores[t] = MathF.Exp(scores[t] - maximum);
            for (int j = 0; j < d.HeadWidth; j++)
            {
                float result = 0;
                for (int t = 0; t < scores.Length; t++) result += scores[t] / sum * state.Values[t][kvStart + j];
                output[qStart + j] = result * Qwen35Math.Sigmoid(gate[qStart + j]);
            }
        }
        return Project(p + "attn_output.weight", output);
    }

    /// <summary>
    /// Starts a new sequence and samples greedily. The final emitted token has
    /// not yet been passed through ForwardToken when this method returns.
    /// </summary>
    public int[] GenerateTokenIds(IReadOnlyList<int> prompt, int maxNewTokens, int? eosTokenId = null)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentOutOfRangeException.ThrowIfNegative(maxNewTokens);
        if (prompt.Count == 0 || prompt.Count > Descriptor.ContextLength)
            throw new ArgumentException("Prompt must contain 1..context-length tokens.", nameof(prompt));
        Reset();
        var result = prompt.ToList();
        if (maxNewTokens == 0 || result.Count == Descriptor.ContextLength) return result.ToArray();
        float[] logits = [];
        for (int i = 0; i < prompt.Count; i++) logits = ForwardToken(prompt[i], i == prompt.Count - 1);
        for (int generated = 0; generated < maxNewTokens && result.Count < Descriptor.ContextLength; generated++)
        {
            int next = 0;
            for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[next]) next = i;
            result.Add(next);
            if (next == eosTokenId || generated + 1 == maxNewTokens || result.Count == Descriptor.ContextLength) break;
            logits = ForwardToken(next);
        }
        return result.ToArray();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (Matrix matrix in _matrices.Values) matrix.Dispose();
        foreach (ArcExecutionLane lane in _lanes) lane.Dispose();
        _matrices.Clear();
        _dense.Clear();
        foreach (LayerState state in _states) state.Reset();
    }

    private sealed class LayerState
    {
        internal readonly float[] Convolution, Recurrent;
        internal readonly List<float[]> Keys = [], Values = [];
        internal LayerState(Qwen35GgufDescriptor d, bool recurrent)
        {
            Convolution = recurrent ? new float[checked((2 * d.LinearKeyHeads + d.LinearValueHeads)
                * d.LinearHeadWidth * (d.ConvKernel - 1))] : [];
            Recurrent = recurrent ? new float[checked(d.LinearValueHeads * d.LinearHeadWidth * d.LinearHeadWidth)] : [];
        }
        internal void Reset() { Array.Clear(Convolution); Array.Clear(Recurrent); Keys.Clear(); Values.Clear(); }
    }

    private sealed class Matrix : IDisposable
    {
        internal readonly ArcExecutionLane Lane;
        internal readonly int StorageBytes;
        private readonly ArcBuffer _encoded;
        private readonly uint _type;
        private readonly int _inputWidth, _outputWidth;
        private readonly float[] _zeroBias;
        internal Matrix(ArcExecutionLane lane, GgufTensorInfo info, byte[] payload)
        {
            Lane = lane; StorageBytes = payload.Length; _type = info.Type;
            _inputWidth = checked((int)info.Shape[0]); _outputWidth = checked((int)info.Shape[1]);
            _zeroBias = new float[_outputWidth];
            _encoded = lane.UploadRaw(payload);
        }
        internal float[] Forward(float[] input)
        {
            if (input.Length != _inputWidth) throw new ArgumentException("Qwen3.5 projection width mismatch.");
            var output = new float[_outputWidth];
            Lane.Run(_type == Qwen2Gguf.Q4KType ? "qwen_linear_q4_k" : "qwen_linear_q6_k",
                _outputWidth, 0, In(input), _encoded, In(_zeroBias), Out(output), 1, _inputWidth, _outputWidth);
            return output;
        }
        internal float[] Embedding(int tokenId)
        {
            var output = new float[_inputWidth];
            Lane.Run(_type == Qwen2Gguf.Q4KType ? "qwen_embedding_q4_k" : "qwen_embedding_q6_k",
                _inputWidth, 0, _encoded, In(new[] { tokenId }), Out(output), _inputWidth);
            return output;
        }
        public void Dispose() => _encoded.Dispose();
    }
}
