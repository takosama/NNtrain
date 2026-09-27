using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>
/// Text-only Qwen3.5 inference. Weights, activations, attention caches and
/// recurrent state stay on their owning Arc device. Host staging is used only
/// between different OpenCL contexts. Instances own one sequence and are not thread safe.
/// </summary>
public sealed class Qwen35QuantizedModel : IDisposable
{
    private const long WorkspaceReserveBytes = 64L * 1024 * 1024;
    private readonly List<ArcExecutionLane> _lanes = [];
    private readonly Dictionary<string, Matrix> _matrices = new(StringComparer.Ordinal);
    private readonly Dictionary<string, ArcBuffer> _dense = new(StringComparer.Ordinal);
    private readonly Dictionary<ArcExecutionLane, ArcBuffer> _zeroBias = [];
    private readonly Dictionary<ArcExecutionLane, long> _auxiliaryBytes = [];
    private readonly List<LayerState> _states = [];
    private int _position;
    private bool _disposed, _faulted;
    private readonly Qwen35ExecutionOptions _options;

    private Qwen35QuantizedModel(Qwen35GgufDescriptor descriptor, Qwen35ExecutionOptions options)
        => (Descriptor, _options) = (descriptor, options);

    public Qwen35GgufDescriptor Descriptor { get; }
    public IReadOnlyList<long> ResidentWeightBytes => _lanes.Select(lane =>
        _matrices.Values.Where(matrix => ReferenceEquals(matrix.Lane, lane))
            .Sum(matrix => (long)matrix.StorageBytes)).ToArray();
    public IReadOnlyList<long> ResidentAuxiliaryWeightBytes => _lanes.Select(lane => _auxiliaryBytes.GetValueOrDefault(lane)).ToArray();
    public IReadOnlyList<long> ResidentStateBytes => _lanes.Select(lane =>
        _states.Where(state => ReferenceEquals(state.Lane, lane)).Sum(state => state.StorageBytes)).ToArray();
    public IReadOnlyList<long> UploadedBytes => _lanes.Select(lane => lane.H2DBytes).ToArray();
    public IReadOnlyList<long> DownloadedBytes => _lanes.Select(lane => lane.D2HBytes).ToArray();
    public IReadOnlyList<long> LiveDeviceBytes => _lanes.Select(lane => lane.AllocatedBytes).ToArray();
    public IReadOnlyList<long> PeakDeviceBytes => _lanes.Select(lane => lane.PeakAllocatedBytes).ToArray();
    public IReadOnlyDictionary<string, double> KernelMilliseconds => _lanes
        .SelectMany(lane => lane.KernelTimings).GroupBy(pair => pair.Key)
        .ToDictionary(group => group.Key, group => group.Sum(pair => pair.Value));

    public static Qwen35QuantizedModel Load(
        string path, IReadOnlyList<int>? devices = null, Action<string>? progress = null,
        Qwen35ExecutionOptions? options = null)
    {
        options ??= new Qwen35ExecutionOptions();
        if (!Enum.IsDefined(options.QuantizedKernel) || options.QueuedKernelLimit is < 16 or > 4096)
            throw new ArgumentException("Invalid Qwen3.5 execution options.", nameof(options));
        Qwen35GgufDescriptor d = Qwen35Gguf.Inspect(path);
        ArcDeviceInfo[] available = ArcDevices.Enumerate().ToArray();
        if (available.Length == 0)
            throw new NotSupportedException("Qwen3.5 quantized inference requires Intel Arc.");
        int[] selected = devices?.ToArray() ?? SelectDevices(d, available);
        if (selected.Length == 0 || selected.Length > d.LayerCount
            || selected.Distinct().Count() != selected.Length
            || selected.Any(index => index < 0 || index >= available.Length))
            throw new ArgumentException("Specify distinct, available Arc device indices.", nameof(devices));
        long[] planned = PlanPersistentBytes(d, selected.Length);
        for (int slot = 0; slot < selected.Length; slot++)
        {
            ArcDeviceInfo device = available[selected[slot]];
            if (planned[slot] + WorkspaceReserveBytes > DeviceBudget(device))
                throw new NotSupportedException(
                    $"Arc {device.Index} needs {planned[slot] / 1073741824.0:F2} GiB for weights and initial GPU state " +
                    "plus workspace. Select more GPUs with --devices 0,1.");
            foreach (GgufTensorInfo tensor in d.Tensors)
                if (DeviceSlot(tensor.Name, d.LayerCount, selected.Length) == slot
                    && (ulong)(IsQuantized(tensor) ? EncodedBytes(tensor) : DenseBytes(tensor)) > device.MaximumAllocationBytes)
                    throw new NotSupportedException($"Tensor '{tensor.Name}' exceeds Arc {device.Index}'s allocation limit.");
        }

        var model = new Qwen35QuantizedModel(d, options);
        try
        {
            foreach (int index in selected)
            {
                progress?.Invoke($"Preparing Arc {index}: {available[index].Name}");
                model._lanes.Add(new ArcExecutionLane(index, new ArcExecutionOptions
                {
                    Qwen35InferenceKernelsOnly = true,
                    BufferPoolBytes = WorkspaceReserveBytes,
                    DeferredReleaseBytes = 0,
                    QueuedKernelLimit = options.QueuedKernelLimit,
                    PhysicalBufferBudgetBytes = DeviceBudget(available[index])
                }));
            }
            using var gguf = new GgufReader(path);
            int loaded = 0;
            foreach (GgufTensorInfo tensor in d.Tensors)
            {
                var lane = model._lanes[DeviceSlot(tensor.Name, d.LayerCount, selected.Length)];
                if (IsQuantized(tensor))
                {
                    // UploadRaw is blocking; no model-sized host payload is retained.
                    model._matrices.Add(tensor.Name, new Matrix(lane, tensor,
                        gguf.ReadTensorBytes(tensor, EncodedBytes(tensor)), options.QuantizedKernel));
                }
                else
                {
                    model._dense.Add(tensor.Name, lane.Upload(Qwen2Gguf.ReadTensor(gguf, tensor)));
                    model._auxiliaryBytes[lane] = model._auxiliaryBytes.GetValueOrDefault(lane) + DenseBytes(tensor);
                }
                if (++loaded % 100 == 0) progress?.Invoke($"Loading tensors: {loaded}/{d.Tensors.Count}");
            }
            foreach (ArcExecutionLane lane in model._lanes)
            {
                int width = model._matrices.Values.Where(matrix => ReferenceEquals(matrix.Lane, lane))
                    .Max(matrix => matrix.OutputWidth);
                ArcBuffer zeroBias = lane.Allocate(width);
                model._zeroBias.Add(lane, zeroBias);
                lane.Run("q35a_zero", width, 0, zeroBias, width);
            }
            for (int layer = 0; layer < d.LayerCount; layer++)
            {
                var lane = model._lanes[DeviceSlot($"blk.{layer}.", d.LayerCount, selected.Length)];
                var state = new LayerState(lane, d, d.IsRecurrent(layer));
                model._states.Add(state);
                state.Initialize();
            }
            for (int slot = 0; slot < selected.Length; slot++)
            {
                model._lanes[slot].Synchronize();
                progress?.Invoke($"Arc {selected[slot]}: quantized weights = {model.ResidentWeightBytes[slot]} bytes " +
                    $"({model.ResidentWeightBytes[slot] / 1073741824.0:F2} GiB), " +
                    $"GPU state = {model.ResidentStateBytes[slot]} bytes, " +
                    $"live device allocation = {model._lanes[slot].AllocatedBytes} bytes");
            }
            return model;
        }
        catch { model.Dispose(); throw; }
    }

    internal static long[] PlanWeightBytes(Qwen35GgufDescriptor d, int deviceCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(deviceCount);
        var bytes = new long[deviceCount];
        foreach (GgufTensorInfo tensor in d.Tensors.Where(IsQuantized))
            bytes[DeviceSlot(tensor.Name, d.LayerCount, deviceCount)] += EncodedBytes(tensor);
        return bytes;
    }

    private static long[] PlanPersistentBytes(Qwen35GgufDescriptor d, int devices)
    {
        long[] bytes = PlanWeightBytes(d, devices);
        var maxOutput = new int[devices];
        foreach (GgufTensorInfo tensor in d.Tensors)
        {
            int slot = DeviceSlot(tensor.Name, d.LayerCount, devices);
            if (IsQuantized(tensor)) maxOutput[slot] = Math.Max(maxOutput[slot], checked((int)tensor.Shape[1]));
            else bytes[slot] = checked(bytes[slot] + DenseBytes(tensor));
        }
        for (int i = 0; i < devices; i++) bytes[i] += 4L * maxOutput[i];
        for (int layer = 0; layer < d.LayerCount; layer++)
        {
            int slot = DeviceSlot($"blk.{layer}.", d.LayerCount, devices);
            long stateBytes = d.IsRecurrent(layer)
                ? checked(4L * d.LinearValueHeads * d.LinearHeadWidth * d.LinearHeadWidth
                    + Math.Max(4L, 4L * (2 * d.LinearKeyHeads + d.LinearValueHeads) * d.LinearHeadWidth * (d.ConvKernel - 1)))
                : checked(8L * d.KvHeadCount * d.HeadWidth * Math.Min(16, d.ContextLength));
            bytes[slot] = checked(bytes[slot] + stateBytes);
        }
        return bytes;
    }

    private static int[] SelectDevices(Qwen35GgufDescriptor d, ArcDeviceInfo[] available)
    {
        for (int count = 1; count <= Math.Min(available.Length, d.LayerCount); count++)
        {
            long[] bytes = PlanPersistentBytes(d, count);
            if (bytes.Select((value, slot) => value + WorkspaceReserveBytes <= DeviceBudget(available[slot])).All(fits => fits))
                return Enumerable.Range(0, count).ToArray();
        }
        throw new NotSupportedException("The Qwen3.5 weights and initial GPU state exceed the available Arc VRAM budget.");
    }

    private static long DeviceBudget(ArcDeviceInfo device) => checked((long)(device.GlobalMemoryBytes / 10 * 9));

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
        => Qwen35Gguf.IsSupportedQuantization(tensor.Type);
    private static int EncodedBytes(GgufTensorInfo tensor)
        => checked((int)(tensor.Shape[0] / 256 * tensor.Shape[1]
            * (ulong)Qwen35Gguf.QuantizedBlockBytes(tensor.Type)));
    private static long DenseBytes(GgufTensorInfo tensor)
        => checked(tensor.Shape.Aggregate(1L, (count, dimension) => checked(count * (long)dimension)) * 4);

    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _faulted = true;
        foreach (LayerState state in _states) state.Reset();
        _position = 0;
        _faulted = false;
    }

    /// <summary>Advances GPU state by one token. Only requested final logits are downloaded.</summary>
    public float[] ForwardToken(int tokenId, bool returnLogits = true)
    {
        using ArcBuffer? logits = ForwardTokenDevice(tokenId, returnLogits);
        if (logits is null) return [];
        try
        {
            var values = new float[Descriptor.VocabularySize];
            OutputMatrix.Lane.Read(logits, values);
            if (values.Any(value => !float.IsFinite(value)))
                throw new ArithmeticException("Qwen3.5 produced non-finite logits.");
            return values;
        }
        catch { _faulted = true; throw; }
    }

    private Matrix OutputMatrix => _matrices.TryGetValue("output.weight", out Matrix? head)
        ? head : _matrices["token_embd.weight"];

    private ArcBuffer? ForwardTokenDevice(int tokenId, bool returnLogits)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_faulted) throw new InvalidOperationException("Qwen3.5 GPU state is invalid after a failed step; call Reset before continuing.");
        Qwen35GgufDescriptor d = Descriptor;
        if ((uint)tokenId >= (uint)d.VocabularySize) throw new ArgumentOutOfRangeException(nameof(tokenId));
        if (_position >= d.ContextLength) throw new InvalidOperationException("Qwen3.5 context length exceeded.");
        // Grow caches before mutating any layer's sequence state. Replacement
        // uses D2D copies and keeps old buffers alive until the copy succeeds.
        foreach (LayerState state in _states) state.EnsureCapacity(_position + 1);
        ArcExecutionLane lane = _matrices["token_embd.weight"].Lane;
        ArcBuffer? hidden = null;
        try
        {
            hidden = _matrices["token_embd.weight"].Embedding(tokenId);
            for (int layer = 0; layer < d.LayerCount; layer++)
            {
                LayerState state = _states[layer];
                MoveToLane(ref hidden, ref lane, state.Lane, d.EmbeddingLength);
                string p = $"blk.{layer}.";
                using ArcBuffer normalized = Qwen35Gpu.RmsNorm(lane, hidden!, _dense[p + "attn_norm.weight"], 1, d.EmbeddingLength, d.RmsEpsilon);
                using ArcBuffer attention = d.IsRecurrent(layer)
                    ? RecurrentAttention(normalized, p, state)
                    : FullAttention(normalized, p, state);
                Qwen35Gpu.AddInPlace(lane, hidden!, attention, d.EmbeddingLength);
                using ArcBuffer postNorm = Qwen35Gpu.RmsNorm(lane, hidden!, _dense[p + "post_attention_norm.weight"], 1, d.EmbeddingLength, d.RmsEpsilon);
                using ArcBuffer gate = Project(p + "ffn_gate.weight", postNorm);
                using ArcBuffer up = Project(p + "ffn_up.weight", postNorm);
                using ArcBuffer activated = Qwen35Gpu.SiluMultiply(lane, gate, up, d.FeedForwardLength);
                using ArcBuffer down = Project(p + "ffn_down.weight", activated);
                Qwen35Gpu.AddInPlace(lane, hidden!, down, d.EmbeddingLength);
            }
            if (!returnLogits) { _position++; return null; }
            ArcBuffer finalNorm = Qwen35Gpu.RmsNorm(lane, hidden!, _dense["output_norm.weight"], 1, d.EmbeddingLength, d.RmsEpsilon);
            hidden!.Dispose(); hidden = finalNorm;
            MoveToLane(ref hidden, ref lane, OutputMatrix.Lane, d.EmbeddingLength);
            ArcBuffer logits = OutputMatrix.Forward(hidden!, _zeroBias[lane]);
            _position++;
            return logits;
        }
        catch { _faulted = true; throw; }
        finally { hidden?.Dispose(); }
    }

    private static void MoveToLane(ref ArcBuffer? buffer, ref ArcExecutionLane source,
        ArcExecutionLane destination, int width)
    {
        if (ReferenceEquals(source, destination)) return;
        // Separate OpenCL contexts cannot share buffers. This is transport,
        // not host arithmetic; all layer computation remains on its GPU.
        var staging = new float[width];
        source.Read(buffer!, staging);
        ArcBuffer moved = destination.Upload(staging);
        buffer!.Dispose(); buffer = moved; source = destination;
    }

    private ArcBuffer Project(string name, ArcBuffer input)
    {
        Matrix matrix = _matrices[name];
        return matrix.Forward(input, _zeroBias[matrix.Lane]);
    }

    private ArcBuffer RecurrentAttention(ArcBuffer input, string p, LayerState state)
    {
        Qwen35GgufDescriptor d = Descriptor;
        using ArcBuffer qkv = Project(p + "attn_qkv.weight", input);
        using ArcBuffer gate = Project(p + "attn_gate.weight", input);
        using ArcBuffer alpha = Project(p + "ssm_alpha.weight", input);
        using ArcBuffer beta = Project(p + "ssm_beta.weight", input);
        using ArcBuffer delta = _options.FusedDelta ? Qwen35Gpu.DeltaStepFused(state.Lane, qkv, gate, alpha, beta,
            _dense[p + "ssm_conv1d.weight"], _dense[p + "ssm_dt.bias"], _dense[p + "ssm_a"],
            _dense[p + "ssm_norm.weight"], state.Convolution!, state.Recurrent!,
            d.LinearKeyHeads, d.LinearValueHeads, d.LinearHeadWidth, d.ConvKernel, d.RmsEpsilon)
            : Qwen35Gpu.DeltaStep(state.Lane, qkv, gate, alpha, beta,
            _dense[p + "ssm_conv1d.weight"], _dense[p + "ssm_dt.bias"], _dense[p + "ssm_a"],
            _dense[p + "ssm_norm.weight"], state.Convolution!, state.Recurrent!,
            d.LinearKeyHeads, d.LinearValueHeads, d.LinearHeadWidth, d.ConvKernel, d.RmsEpsilon);
        return Project(p + "ssm_out.weight", delta);
    }

    private ArcBuffer FullAttention(ArcBuffer input, string p, LayerState state)
    {
        Qwen35GgufDescriptor d = Descriptor;
        using ArcBuffer qAndGate = Project(p + "attn_q.weight", input);
        using ArcBuffer key = Project(p + "attn_k.weight", input);
        using ArcBuffer value = Project(p + "attn_v.weight", input);
        using ArcBuffer attention = Qwen35Gpu.AttentionStep(state.Lane, qAndGate, key, value,
            _dense[p + "attn_q_norm.weight"], _dense[p + "attn_k_norm.weight"], state.Keys!, state.Values!,
            _position, d.HeadCount, d.KvHeadCount, d.HeadWidth, d.RopeDimensionCount, d.RopeTheta, d.RmsEpsilon);
        return Project(p + "attn_output.weight", attention);
    }

    /// <summary>
    /// Starts a new sequence and samples greedily on the GPU. Only the selected
    /// token/status is downloaded. The final emitted token has not yet been forwarded.
    /// </summary>
    public int[] GenerateTokenIds(IReadOnlyList<int> prompt, int maxNewTokens,
        int? eosTokenId = null, Action<int>? onToken = null)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentOutOfRangeException.ThrowIfNegative(maxNewTokens);
        if (prompt.Count == 0 || prompt.Count > Descriptor.ContextLength)
            throw new ArgumentException("Prompt must contain 1..context-length tokens.", nameof(prompt));
        if (prompt.Any(token => (uint)token >= (uint)Descriptor.VocabularySize))
            throw new ArgumentOutOfRangeException(nameof(prompt));
        Reset();
        var result = prompt.ToList();
        if (maxNewTokens == 0 || result.Count == Descriptor.ContextLength) return result.ToArray();
        ArcBuffer? logits = null;
        try
        {
            for (int i = 0; i < prompt.Count; i++) logits = ForwardTokenDevice(prompt[i], i == prompt.Count - 1);
            for (int generated = 0; generated < maxNewTokens && result.Count < Descriptor.ContextLength; generated++)
            {
                int next = Qwen35Gpu.ArgMax(OutputMatrix.Lane, logits!, Descriptor.VocabularySize);
                logits!.Dispose(); logits = null;
                result.Add(next);
                onToken?.Invoke(next);
                if (next == eosTokenId || generated + 1 == maxNewTokens || result.Count == Descriptor.ContextLength) break;
                logits = ForwardTokenDevice(next, true);
            }
            return result.ToArray();
        }
        catch { _faulted = true; throw; }
        finally { logits?.Dispose(); }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (LayerState state in _states) state.Dispose();
        foreach (Matrix matrix in _matrices.Values) matrix.Dispose();
        foreach (ArcBuffer buffer in _dense.Values) buffer.Dispose();
        foreach (ArcBuffer buffer in _zeroBias.Values) buffer.Dispose();
        foreach (ArcExecutionLane lane in _lanes) lane.Dispose();
        _states.Clear(); _matrices.Clear(); _dense.Clear(); _zeroBias.Clear();
    }

    private sealed class LayerState(ArcExecutionLane lane, Qwen35GgufDescriptor d, bool recurrent) : IDisposable
    {
        internal ArcExecutionLane Lane { get; } = lane;
        internal ArcBuffer? Convolution, Recurrent, Keys, Values;
        private int _capacity;
        internal long StorageBytes => (Convolution?.ByteLength ?? 0) + (Recurrent?.ByteLength ?? 0)
            + (Keys?.ByteLength ?? 0) + (Values?.ByteLength ?? 0);
        internal void Initialize()
        {
            if (recurrent)
            {
                Convolution = Lane.Allocate(checked((2 * d.LinearKeyHeads + d.LinearValueHeads) * d.LinearHeadWidth * (d.ConvKernel - 1)));
                Recurrent = Lane.Allocate(checked(d.LinearValueHeads * d.LinearHeadWidth * d.LinearHeadWidth));
                Reset();
            }
            else EnsureCapacity(Math.Min(16, d.ContextLength));
        }
        internal void Reset()
        {
            if (Convolution is not null) Zero(Convolution);
            if (Recurrent is not null) Zero(Recurrent);
            // Cached rows beyond the current position are never read. Resetting
            // the model position is enough; capacity remains reusable on device.
        }
        private void Zero(ArcBuffer buffer)
        {
            int length = checked((int)(buffer.ByteLength / 4));
            Lane.Run("q35a_zero", length, 0, buffer, length);
        }
        internal void EnsureCapacity(int required)
        {
            if (recurrent || required <= _capacity) return;
            int capacity = (int)Math.Min(d.ContextLength, Math.Max(required, Math.Max(16L, 2L * _capacity)));
            long bytes = checked((long)capacity * d.KvHeadCount * d.HeadWidth * 4);
            if (bytes > int.MaxValue || (ulong)bytes > Lane.Device.MaximumAllocationBytes)
                throw new NotSupportedException("Qwen3.5 KV cache exceeds the device allocation limit.");
            if (Lane.AllocatedBytes + 2 * bytes + WorkspaceReserveBytes > DeviceBudget(Lane.Device))
                throw new NotSupportedException("Qwen3.5 KV cache growth exceeds the GPU memory budget; use a shorter context or more devices.");
            ArcBuffer? keys = null, values = null;
            try
            {
                keys = Lane.AllocateBytes((int)bytes); values = Lane.AllocateBytes((int)bytes);
                if (Keys is not null)
                {
                    int oldBytes = checked((int)Keys.ByteLength);
                    Lane.CopyBytes(Keys, keys, 0, 0, oldBytes);
                    Lane.CopyBytes(Values!, values, 0, 0, oldBytes);
                    Lane.Synchronize();
                }
                Keys?.Dispose(); Values?.Dispose();
                Keys = keys; Values = values; keys = values = null; _capacity = capacity;
            }
            finally { keys?.Dispose(); values?.Dispose(); }
        }
        public void Dispose() { Convolution?.Dispose(); Recurrent?.Dispose(); Keys?.Dispose(); Values?.Dispose(); }
    }

    private sealed class Matrix : IDisposable
    {
        internal readonly ArcExecutionLane Lane;
        internal readonly int StorageBytes, OutputWidth;
        private readonly ArcBuffer _encoded;
        private readonly uint _type;
        private readonly string _quantization;
        private readonly int _inputWidth;
        private readonly Qwen35QuantizedKernel _kernel;
        internal Matrix(ArcExecutionLane lane, GgufTensorInfo info, byte[] payload, Qwen35QuantizedKernel kernel)
        {
            Lane = lane; StorageBytes = payload.Length; _type = info.Type;
            _quantization = _type switch
            {
                Qwen2Gguf.Q4KType => "q4_k", Qwen35Gguf.Q5KType => "q5_k",
                Qwen2Gguf.Q6KType => "q6_k", Qwen35Gguf.IQ2SType => "iq2_s",
                Qwen35Gguf.IQ3SType => "iq3_s",
                _ => throw new NotSupportedException($"Unsupported matrix storage type {_type}.")
            };
            _inputWidth = checked((int)info.Shape[0]); OutputWidth = checked((int)info.Shape[1]);
            bool subgroup = lane.Options.XmxMatrices && lane.Device.SupportsXmx
                && lane.Device.MinimumSubgroupSize == 16
                && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups");
            _kernel = kernel == Qwen35QuantizedKernel.Auto
                ? (subgroup ? Qwen35QuantizedKernel.Subgroup : Qwen35QuantizedKernel.Cooperative) : kernel;
            if (_kernel == Qwen35QuantizedKernel.Subgroup && !subgroup)
                throw new NotSupportedException("The Qwen3.5 subgroup kernel requires Intel SG16 support.");
            _encoded = lane.UploadRaw(payload);
        }
        internal ArcBuffer Forward(ArcBuffer input, ArcBuffer zeroBias)
        {
            if (input.ByteLength < 4L * _inputWidth) throw new ArgumentException("Qwen3.5 projection width mismatch.");
            ArcBuffer output = Lane.Allocate(OutputWidth);
            try
            {
                if (_kernel == Qwen35QuantizedKernel.Subgroup)
                    Lane.Run($"q35l_{_quantization}_sg16",
                        ((long)OutputWidth + 1) / 2 * 32, 32, input, _encoded, zeroBias, output, 1, _inputWidth, OutputWidth);
                else if (_kernel == Qwen35QuantizedKernel.Cooperative)
                    Lane.Run2D($"q35l_{_quantization}_coop64",
                        (long)OutputWidth * 64, 1, 64, 1, input, _encoded, zeroBias, output, 1, _inputWidth, OutputWidth);
                else
                    Lane.Run(_type is Qwen2Gguf.Q4KType or Qwen2Gguf.Q6KType
                        ? $"qwen_linear_{_quantization}" : $"q35l_{_quantization}_reference",
                        OutputWidth, 0, input, _encoded, zeroBias, output, 1, _inputWidth, OutputWidth);
                return output;
            }
            catch { output.Dispose(); throw; }
        }
        internal ArcBuffer Embedding(int tokenId)
        {
            ArcBuffer output = Lane.Allocate(_inputWidth);
            try
            {
                using ArcBuffer id = Lane.UploadRaw(new[] { tokenId });
                Lane.Run(_type is Qwen2Gguf.Q4KType or Qwen2Gguf.Q6KType
                    ? $"qwen_embedding_{_quantization}" : $"q35l_{_quantization}_embedding",
                    _inputWidth, 0, _encoded, id, output, _inputWidth);
                return output;
            }
            catch { output.Dispose(); throw; }
        }
        public void Dispose() => _encoded.Dispose();
    }
}
