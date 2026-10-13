using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>
/// Text-only Qwen3.5 inference. Weights, activations, attention caches and
/// recurrent state stay on their owning Arc device. Host staging is used only
/// between different OpenCL contexts. Instances own one sequence and are not thread safe.
/// </summary>
public sealed partial class Qwen35QuantizedModel : IDisposable
{
    private const long WorkspaceReserveBytes = 64L * 1024 * 1024;
    private readonly List<ArcExecutionLane> _lanes = [];
    private readonly Dictionary<string, Matrix> _matrices = new(StringComparer.Ordinal);
    private readonly Dictionary<string, ArcBuffer> _dense = new(StringComparer.Ordinal);
    private readonly Dictionary<ArcExecutionLane, ArcBuffer> _zeroBias = [];
    private readonly Dictionary<ArcExecutionLane, long> _auxiliaryBytes = [];
    private readonly List<LayerState> _states = [];
    private readonly List<LayerBindings> _layerBindings = [];
    private Matrix _embedding = null!;
    private ArcBuffer _outputNorm = null!;
    private int _position;
    private int[]? _cachedPromptTokens;
    private int _lastReusedPromptTokens;
    private bool _disposed, _faulted;
    private readonly Qwen35ExecutionOptions _options;
    private Qwen35PrismMetadata? _prism;
    private readonly Dictionary<(ArcExecutionLane Lane, int Width), ArcBuffer> _prismSigns = [];

    private Qwen35QuantizedModel(Qwen35GgufDescriptor descriptor, Qwen35ExecutionOptions options)
        => (Descriptor, _options) = (descriptor, options);

    public Qwen35GgufDescriptor Descriptor { get; }
    public IReadOnlyList<long> ResidentWeightBytes => _lanes.Select(lane =>
        _matrices.Values.Where(matrix => ReferenceEquals(matrix.Lane, lane))
            .Sum(matrix => (long)matrix.StorageBytes)
            + (_splitOutputHead is { } split && ReferenceEquals(split.Peer, lane) ? split.WeightBytes : 0)).ToArray();
    public IReadOnlyList<long> ResidentAuxiliaryWeightBytes => _lanes.Select(lane => _auxiliaryBytes.GetValueOrDefault(lane)).ToArray();
    public IReadOnlyList<long> ResidentStateBytes => _lanes.Select(lane =>
        _states.Where(state => ReferenceEquals(state.Lane, lane)).Sum(state => state.StorageBytes)).ToArray();
    public IReadOnlyList<long> UploadedBytes => _lanes.Select(lane => lane.H2DBytes).ToArray();
    public IReadOnlyList<long> DownloadedBytes => _lanes.Select(lane => lane.D2HBytes).ToArray();
    public IReadOnlyList<long> LiveDeviceBytes => _lanes.Select(lane => lane.AllocatedBytes).ToArray();
    /// <summary>Native memory still owned by each lane, including idle and retired buffers.</summary>
    public IReadOnlyList<long> TotalNativeDeviceBytes => _lanes.Select(lane =>
        checked(lane.AllocatedBytes + lane.CachedBytes + lane.RetiredBytes)).ToArray();
    public IReadOnlyList<long> PeakDeviceBytes => _lanes.Select(lane => lane.PeakAllocatedBytes).ToArray();
    /// <summary>
    /// Account for another allocator on a model device. Call between model
    /// operations, under the same owner serialization used for generation.
    /// The slot is the index in the model's selected device list.
    /// </summary>
    public void SetExternalDeviceMemoryReservation(int slot, long bytes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)slot >= (uint)_lanes.Count) throw new ArgumentOutOfRangeException(nameof(slot));
        ArcExecutionLane lane = _lanes[slot];
        ReleaseSplitOutputHeadForReservation(lane, bytes);
        lane.SetExternalMemoryReservation(bytes);
    }
    /// <summary>Prompt tokens skipped by the most recent prefix-reuse generation.</summary>
    public int LastReusedPromptTokens => _lastReusedPromptTokens;
    public IReadOnlyDictionary<string, double> KernelMilliseconds => _lanes
        .SelectMany(lane => lane.KernelTimings).GroupBy(pair => pair.Key)
        .ToDictionary(group => group.Key, group => group.Sum(pair => pair.Value));

    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _faulted = true;
        _cachedPromptTokens = null;
        _cachedMixedPromptTokens = null;
        _lastReusedPromptTokens = 0;
        foreach (LayerState state in _states) state.Reset();
        _position = 0;
        _faulted = false;
    }

    /// <summary>Advances GPU state by one token. Only requested final logits are downloaded.</summary>
    public float[] ForwardToken(int tokenId, bool returnLogits = true)
    {
        InvalidatePromptCheckpoint();
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
        => ForwardTokenDevice(tokenId, returnLogits, null, null);

    private ArcBuffer? ForwardTokenDevice(int tokenId, bool returnLogits,
        float[]? overrideEmbedding, Qwen35Position? ropePosition)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_faulted) throw new InvalidOperationException("Qwen3.5 GPU state is invalid after a failed step; call Reset before continuing.");
        Qwen35GgufDescriptor d = Descriptor;
        if (overrideEmbedding is null && (uint)tokenId >= (uint)d.VocabularySize)
            throw new ArgumentOutOfRangeException(nameof(tokenId));
        if (overrideEmbedding is not null && (overrideEmbedding.Length != d.EmbeddingLength
            || overrideEmbedding.Any(value => !float.IsFinite(value))))
            throw new ArgumentException("Image embedding width or values are invalid.", nameof(overrideEmbedding));
        if (_position >= d.ContextLength) throw new InvalidOperationException("Qwen3.5 context length exceeded.");
        // Grow caches before mutating any layer's sequence state. Replacement
        // uses D2D copies and keeps old buffers alive until the copy succeeds.
        foreach (LayerState state in _states) state.EnsureCapacity(_position + 1);
        ArcExecutionLane lane = _embedding.Lane;
        ArcBuffer? hidden = null;
        bool fusedResidualRms = _options.InferenceFusedResidualRms && !_options.LoraTraining;
        try
        {
            hidden = overrideEmbedding is null ? _embedding.Embedding(tokenId) : lane.Upload(overrideEmbedding);
            if (overrideEmbedding is null && _prism?.InverseWeights.Contains("token_embd.weight") == true)
            {
                ArcBuffer restored = PrismTransform("token_embd.weight", _embedding, hidden, inverse: true);
                hidden.Dispose(); hidden = restored;
            }
            for (int layer = 0; layer < d.LayerCount; layer++)
            {
                LayerBindings bindings = _layerBindings[layer];
                LayerState state = bindings.State;
                MoveToLane(ref hidden, ref lane, state.Lane, d.EmbeddingLength);
                using ArcBuffer normalized = Qwen35Gpu.RmsNorm(lane, hidden!, bindings.AttentionNorm, 1, d.EmbeddingLength, d.RmsEpsilon);
                using ArcBuffer attention = bindings.Recurrent
                    ? RecurrentAttention(normalized, bindings)
                    : FullAttention(normalized, bindings, ropePosition);
                using ArcBuffer postNorm = AddResidualAndRmsNorm(lane, hidden!, attention,
                    bindings.PostAttentionNorm, d.EmbeddingLength, d.RmsEpsilon, fusedResidualRms);
                using ArcBuffer gate = Project(bindings.FfnGate, postNorm);
                using ArcBuffer up = Project(bindings.FfnUp, postNorm);
                using ArcBuffer activated = Qwen35Gpu.SiluMultiply(lane, gate, up, d.FeedForwardLength);
                using ArcBuffer down = Project(bindings.FfnDown, activated);
                if (fusedResidualRms && returnLogits && layer == d.LayerCount - 1)
                {
                    ArcBuffer finalNorm = AddResidualAndRmsNorm(lane, hidden!, down,
                        _outputNorm, d.EmbeddingLength, d.RmsEpsilon, fused: true);
                    hidden!.Dispose(); hidden = finalNorm;
                }
                else Qwen35Gpu.AddInPlace(lane, hidden!, down, d.EmbeddingLength);
            }
            if (!returnLogits) { _position++; return null; }
            if (!fusedResidualRms)
            {
                ArcBuffer finalNorm = Qwen35Gpu.RmsNorm(lane, hidden!, _outputNorm, 1, d.EmbeddingLength, d.RmsEpsilon);
                hidden!.Dispose(); hidden = finalNorm;
            }
            MoveToLane(ref hidden, ref lane, OutputMatrix.Lane, d.EmbeddingLength);
            ArcBuffer logits = TrySplitOutputHead(hidden!) ?? ProjectMatrix("output.weight", OutputMatrix, hidden!);
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

    private static ArcBuffer AddResidualAndRmsNorm(ArcExecutionLane lane, ArcBuffer hidden,
        ArcBuffer residual, ArcBuffer weight, int width, float epsilon, bool fused)
    {
        if (!fused)
        {
            Qwen35Gpu.AddInPlace(lane, hidden, residual, width);
            return Qwen35Gpu.RmsNorm(lane, hidden, weight, 1, width, epsilon);
        }
        ArcBuffer normalized = lane.Allocate(width);
        try
        {
            bool fast = lane.Options.Qwen35FastRmsNorm && lane.Options.XmxMatrices
                && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16
                && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups");
            lane.Run(fast ? "q35a_add_rms_norm_sg16_exact" : "q35a_add_rms_norm",
                128, 128, hidden, residual, weight, normalized, width, width, epsilon);
            return normalized;
        }
        catch { normalized.Dispose(); throw; }
    }

    private ArcBuffer Project(string name, ArcBuffer input)
        => ProjectMatrix(name, _matrices[name], input);

    private ArcBuffer Project(Matrix matrix, ArcBuffer input)
        => ProjectMatrix(matrix.Name, matrix, input);

    private ArcBuffer ProjectMatrix(string name, Matrix matrix, ArcBuffer input)
    {
        if (_loraFaulted) throw new InvalidOperationException("LoRA optimizer state is invalid; reload the last saved checkpoint in a new model.");
        if (_prism?.ForwardWeights.Contains(name) == true)
        {
            using ArcBuffer transformed = PrismTransform(name, matrix, input, inverse: false);
            return matrix.Forward(transformed, _zeroBias[matrix.Lane]);
        }
        if (_options.InferenceFusedLora && !_options.LoraTraining && matrix.SupportsFusedLora
            && _lora.TryGetValue(name, out var adapter))
        {
            using ArcBuffer z = adapter.ProjectA(input, 1);
            return matrix.ForwardFusedLora(input, _zeroBias[matrix.Lane], z, adapter);
        }
        ArcBuffer output = matrix.Forward(input, _zeroBias[matrix.Lane]);
        try { ApplyLora(name, input, output, 1); return output; }
        catch { output.Dispose(); throw; }
    }

    private ArcBuffer PrismTransform(string name, Matrix matrix, ArcBuffer input, bool inverse)
    {
        ArcBuffer output = matrix.Lane.Allocate(matrix._inputWidth);
        try
        {
            bool grouped = !inverse && _prism!.GroupedValueHeads && name.EndsWith(".ssm_out.weight", StringComparison.Ordinal);
            matrix.Lane.Run("q35l_prism_hadamard", (long)matrix._inputWidth / 1024 * 256, 256,
                input, _prismSigns[(matrix.Lane, matrix._inputWidth)], output, matrix._inputWidth, 1, inverse ? 1 : 0,
                grouped ? Descriptor.LinearKeyHeads : 0, grouped ? Descriptor.LinearValueHeads : 0, Descriptor.LinearHeadWidth);
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    private ArcBuffer RecurrentAttention(ArcBuffer input, LayerBindings bindings)
    {
        Qwen35GgufDescriptor d = Descriptor;
        LayerState state = bindings.State;
        using ArcBuffer qkv = Project(bindings.Qkv!, input);
        using ArcBuffer gate = Project(bindings.RecurrentGate!, input);
        using ArcBuffer alpha = Project(bindings.Alpha!, input);
        using ArcBuffer beta = Project(bindings.Beta!, input);
        using ArcBuffer delta = _options.FusedDelta ? Qwen35Gpu.DeltaStepFused(state.Lane, qkv, gate, alpha, beta,
            bindings.Convolution!, bindings.DtBias!, bindings.A!,
            bindings.RecurrentNorm!, state.Convolution!, state.Recurrent!,
            d.LinearKeyHeads, d.LinearValueHeads, d.LinearHeadWidth, d.ConvKernel, d.RmsEpsilon,
            cacheState: _options.InferenceCachedDeltaDecode && !_options.LoraTraining)
            : Qwen35Gpu.DeltaStep(state.Lane, qkv, gate, alpha, beta,
            bindings.Convolution!, bindings.DtBias!, bindings.A!,
            bindings.RecurrentNorm!, state.Convolution!, state.Recurrent!,
            d.LinearKeyHeads, d.LinearValueHeads, d.LinearHeadWidth, d.ConvKernel, d.RmsEpsilon);
        return Project(bindings.AttentionOutput, delta);
    }

    private ArcBuffer FullAttention(ArcBuffer input, LayerBindings bindings,
        Qwen35Position? ropePosition = null)
    {
        Qwen35GgufDescriptor d = Descriptor;
        LayerState state = bindings.State;
        using ArcBuffer qAndGate = Project(bindings.Query!, input);
        using ArcBuffer key = Project(bindings.Key!, input);
        using ArcBuffer value = Project(bindings.Value!, input);
        using ArcBuffer attention = Qwen35Gpu.AttentionStep(state.Lane, qAndGate, key, value,
            bindings.QueryNorm!, bindings.KeyNorm!, state.Keys!, state.Values!,
            _position, d.HeadCount, d.KvHeadCount, d.HeadWidth, d.RopeDimensionCount, d.RopeTheta, d.RmsEpsilon,
            ropePosition, d.RopeDimensionSections);
        return Project(bindings.AttentionOutput, attention);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var adapter in _lora.Values) adapter.Dispose();
        _lora.Clear();
        _splitOutputHead?.Dispose(); _splitOutputHead = null;
        foreach (LayerState state in _states) state.Dispose();
        foreach (Matrix matrix in _matrices.Values) matrix.Dispose();
        foreach (ArcBuffer buffer in _dense.Values) buffer.Dispose();
        foreach (ArcBuffer buffer in _zeroBias.Values) buffer.Dispose();
        foreach (ArcBuffer buffer in _prismSigns.Values) buffer.Dispose();
        foreach (ArcExecutionLane lane in _lanes) lane.Dispose();
        _modelSource?.Dispose();
        _states.Clear(); _matrices.Clear(); _dense.Clear(); _zeroBias.Clear(); _prismSigns.Clear();
        _layerBindings.Clear(); _embedding = null!; _outputNorm = null!;
    }


}
