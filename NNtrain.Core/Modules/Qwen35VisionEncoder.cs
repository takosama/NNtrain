using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>Validated GGUF Qwen3-VL merger and vision tower dimensions.</summary>
public sealed record Qwen35VisionDescriptor(
    int ProjectionDimension,
    int PatchSize,
    int MergeSize,
    int EmbeddingLength,
    int ImageSize,
    int LayerCount,
    int HeadCount,
    int HeadWidth,
    int FeedForwardLength,
    int GridPositionSide,
    float LayerNormEpsilon);

/// <summary>
/// In-process Qwen3-VL static-image encoder backed by Intel Arc OpenCL.
/// The input contains one RGB patch for each spatial location in 2x2 merge
/// order; the two temporal convolution weights both see that static patch.
/// One instance owns one GPU lane and serializes calls to Encode.
/// </summary>
public sealed class Qwen35VisionEncoder : IDisposable
{
    private readonly ArcExecutionLane _lane;
    private readonly bool _useOptimizedKernels;
    private readonly bool _useXmxLinear;
    private readonly bool _useXmxAttention;
    private readonly bool _useFlashAttention;
    private readonly Dictionary<string, ArcBuffer> _weights = new(StringComparer.Ordinal);
    private readonly SemaphoreSlim _operation = new(1, 1);
    private bool _disposed;

    private Qwen35VisionEncoder(Qwen35VisionDescriptor descriptor, ArcExecutionLane lane,
        bool useOptimizedKernels, bool useXmxLinear, bool useXmxAttention, bool useFlashAttention)
    {
        Descriptor = descriptor;
        _lane = lane;
        _useOptimizedKernels = useOptimizedKernels;
        _useXmxLinear = useXmxLinear && lane.Device.SupportsXmx
            && lane.Device.MinimumSubgroupSize == 16 && lane.Device.Extensions.Split(' ').Contains("cl_khr_fp16");
        _useXmxAttention = useXmxAttention && lane.Device.SupportsXmx
            && lane.Device.MinimumSubgroupSize == 16
            && lane.Device.Extensions.Split(' ').Contains("cl_khr_fp16");
        _useFlashAttention = useFlashAttention && lane.Device.SupportsXmx
            && lane.Device.MinimumSubgroupSize == 16 && descriptor.HeadWidth <= 80
            && lane.Device.Extensions.Split(' ').Contains("cl_khr_fp16");
    }

    public Qwen35VisionDescriptor Descriptor { get; }
    public long ResidentWeightBytes => _weights.Values.Sum(weight => weight.ByteLength);
    public IReadOnlyDictionary<string, double> KernelTimings => new Dictionary<string, double>(_lane.KernelTimings);
    public bool ProgramBinaryCacheHit => _lane.ProgramBinaryCacheHit;
    public bool UsesXmxLinear => _useOptimizedKernels && _useXmxLinear;
    public bool UsesXmxAttention => _useOptimizedKernels && _useXmxAttention;
    public bool UsesFlashAttention => _useOptimizedKernels && _useFlashAttention;

    /// <summary>Validates the mmproj directory without reading tensor payloads or creating a GPU lane.</summary>
    public static Qwen35VisionDescriptor Inspect(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using var reader = new GgufReader(path);
        return Inspect(reader);
    }

    /// <summary>
    /// Loads a Qwen3-VL vision GGUF directly into one Arc GPU. The default
    /// tiled path retains FP32 accumulation order. Optional XMX comparison
    /// uses a two-component F16 expansion and changes rounding.
    /// </summary>
    public static Qwen35VisionEncoder Load(string path, int device = 0, bool collectKernelTimings = false,
        bool useOptimizedKernels = true, bool useXmxLinear = false, bool useXmxAttention = false,
        bool useFlashAttention = false)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using var reader = new GgufReader(path);
        Qwen35VisionDescriptor descriptor = Inspect(reader);
        var lane = new ArcExecutionLane(device, new ArcExecutionOptions
        {
            Qwen35InferenceKernelsOnly = true,
            Qwen35VisionKernelsOnly = useOptimizedKernels,
            Qwen35VisionFlashAttention = useOptimizedKernels && useFlashAttention,
            CollectKernelTimings = collectKernelTimings,
            CacheProgramBinary = useOptimizedKernels,
            CacheKernelArguments = useOptimizedKernels,
            BufferPoolBytes = (useOptimizedKernels ? 256L : 64L) * 1024 * 1024,
            DeferredReleaseBytes = useOptimizedKernels ? 128L * 1024 * 1024 : 0
        });
        var encoder = new Qwen35VisionEncoder(descriptor, lane, useOptimizedKernels, useXmxLinear,
            useXmxAttention, useFlashAttention);
        try
        {
            foreach (GgufTensorInfo tensor in reader.Tensors)
            {
                ArcBuffer buffer;
                if (tensor.Type == Qwen2Gguf.F16Type && tensor.Shape.Count > 1
                    && tensor.Name != "v.position_embd.weight")
                {
                    int elements = ElementCount(tensor);
                    byte[] payload = reader.ReadTensorBytes(tensor, checked(elements * 2));
                    buffer = lane.AllocateBytes(payload.Length);
                    try { lane.WriteRaw(buffer, payload); }
                    catch { buffer.Dispose(); throw; }
                    if (encoder.UsesXmxLinear && tensor.Shape.Count == 2)
                    {
                        int inputWidth = checked((int)tensor.Shape[0]);
                        int outputWidth = checked((int)tensor.Shape[1]);
                        long packedPairs = Round16(inputWidth) * Round16(outputWidth) / 2;
                        ArcBuffer? packed = null;
                        try
                        {
                            packed = lane.AllocateBytes(checked((int)(packedPairs * sizeof(uint))));
                            lane.Run("q35v_linear_pack_b_f16", packedPairs, 0,
                                buffer, packed, inputWidth, outputWidth);
                            buffer.Dispose();
                            buffer = packed;
                            packed = null;
                        }
                        catch { packed?.Dispose(); buffer.Dispose(); throw; }
                    }
                }
                else
                {
                    // Norms, biases and learned positions are stored as F32 on
                    // device, including any GGUF F16 vector tensors.
                    buffer = lane.Upload(Qwen2Gguf.ReadTensor(reader, tensor));
                }
                encoder._weights.Add(tensor.Name, buffer);
            }
            lane.Synchronize();
            return encoder;
        }
        catch
        {
            encoder.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Encodes one preprocessed static image. Returned dimensions are the
    /// spatial grid after 2x2 merge; Values is row-major [token, feature].
    /// </summary>
    public Qwen35VisionEmbedding Encode(Qwen35VisionInput input,
        CancellationToken cancellationToken = default, Action<string>? progress = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        cancellationToken.ThrowIfCancellationRequested();
        _operation.Wait(cancellationToken);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            Qwen35VisionDescriptor d = Descriptor;
            int h = input.GridHeight, w = input.GridWidth;
            if (h <= 0 || w <= 0 || h % d.MergeSize != 0 || w % d.MergeSize != 0
                || (long)h * w > (long)(d.ImageSize / d.PatchSize) * (d.ImageSize / d.PatchSize))
                throw new ArgumentOutOfRangeException(nameof(input),
                    "Vision patch grid exceeds the GGUF image area or spatial merge.");
            int rows = checked(h * w);
            int patchWidth = checked(3 * d.PatchSize * d.PatchSize);
            if (input.Patches.Length != checked(rows * patchWidth)
                || input.Patches.Any(value => !float.IsFinite(value)))
                throw new ArgumentException("Vision patches have invalid length or non-finite values.", nameof(input));
            int width = d.EmbeddingLength;
            long scoreElementsLong = checked((long)rows * d.HeadCount * rows);
            if (scoreElementsLong > int.MaxValue
                || scoreElementsLong * sizeof(float) > (long)_lane.Device.MaximumAllocationBytes)
                throw new NotSupportedException("Vision attention score matrix exceeds this Arc GPU's allocation limit.");

            progress?.Invoke($"Vision patch embedding: {h}x{w} patches");
            using ArcBuffer patches = _lane.Upload(input.Patches);
            using ArcBuffer scores = _lane.Allocate(UsesFlashAttention ? 1 : (int)scoreElementsLong);
            ArcBuffer? hidden = _lane.Allocate(checked(rows * width));
            try
            {
                _lane.Run("q35v_patch_embed", (long)rows * width, 128,
                    patches, Weight("v.patch_embd.weight"), Weight("v.patch_embd.weight.1"),
                    Weight("v.patch_embd.bias"), hidden, rows, patchWidth, width);
                _lane.Run("q35v_add_position", (long)rows * width, 128,
                    hidden, Weight("v.position_embd.weight"), rows, width,
                    h, w, d.MergeSize, d.GridPositionSide);
                for (int layer = 0; layer < d.LayerCount; layer++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    string p = $"v.blk.{layer}.";
                    using (ArcBuffer norm1 = LayerNorm(hidden,
                        p + "ln1.weight", p + "ln1.bias", rows, width))
                    using (ArcBuffer qkv = Linear(norm1, p + "attn_qkv.weight",
                        p + "attn_qkv.bias", rows, width, 3 * width))
                    {
                        _lane.Run("q35v_rope",
                            (long)rows * d.HeadCount * (d.HeadWidth / 2), 128,
                            qkv, rows, width, d.HeadCount, d.HeadWidth,
                            w, d.MergeSize, 10_000f);
                        using ArcBuffer context = Attention(qkv, scores, rows, d.HeadCount, d.HeadWidth);
                        using ArcBuffer attention = Linear(context,
                            p + "attn_out.weight", p + "attn_out.bias",
                            rows, width, width);
                        _lane.Run("q35v_add_in_place", (long)rows * width, 128,
                            hidden, attention, rows * width);
                    }
                    using (ArcBuffer norm2 = LayerNorm(hidden,
                        p + "ln2.weight", p + "ln2.bias", rows, width))
                    using (ArcBuffer up = Linear(norm2,
                        p + "ffn_up.weight", p + "ffn_up.bias",
                        rows, width, d.FeedForwardLength, gelu: true))
                    using (ArcBuffer down = Linear(up,
                        p + "ffn_down.weight", p + "ffn_down.bias",
                        rows, d.FeedForwardLength, width))
                    {
                        _lane.Run("q35v_add_in_place", (long)rows * width, 128,
                            hidden, down, rows * width);
                    }
                    progress?.Invoke($"Vision block {layer + 1}/{d.LayerCount}");
                }
                cancellationToken.ThrowIfCancellationRequested();
                using ArcBuffer postNorm = LayerNorm(hidden,
                    "v.post_ln.weight", "v.post_ln.bias", rows, width);
                int mergedRows = rows / checked(d.MergeSize * d.MergeSize);
                int mergeWidth = checked(width * d.MergeSize * d.MergeSize);
                using ArcBuffer merged = Linear(postNorm, "mm.0.weight", "mm.0.bias",
                    mergedRows, mergeWidth, mergeWidth, gelu: true);
                using ArcBuffer projected = Linear(merged, "mm.2.weight", "mm.2.bias",
                    mergedRows, mergeWidth, d.ProjectionDimension);
                var values = new float[checked(mergedRows * d.ProjectionDimension)];
                _lane.Read(projected, values);
                if (values.Any(value => !float.IsFinite(value)))
                    throw new ArithmeticException("Vision encoder produced non-finite embeddings.");
                return new Qwen35VisionEmbedding(values, d.ProjectionDimension,
                    h / d.MergeSize, w / d.MergeSize);
            }
            finally { hidden?.Dispose(); }
        }
        finally { _operation.Release(); }
    }

    private ArcBuffer Weight(string name) => _weights[name];

    private ArcBuffer Attention(ArcBuffer qkv, ArcBuffer scores, int rows, int heads, int headWidth)
    {
        if (UsesFlashAttention)
            return AttentionXmxFlash(_lane, qkv, rows, heads, headWidth);
        if (UsesXmxAttention)
            return AttentionXmx(_lane, qkv, scores, rows, heads, headWidth);
        ArcBuffer context = _lane.Allocate(checked(rows * heads * headWidth));
        try
        {
            if (_useOptimizedKernels)
                _lane.Run2D("q35v_attention_scores_tiled", Round16(rows),
                    heads * Round16(rows), 16, 16, qkv, scores, rows, heads, headWidth);
            else
                _lane.Run("q35v_attention_scores", (long)rows * heads * rows, 128,
                    qkv, scores, rows, heads, headWidth);
            _lane.Run("q35v_attention_softmax", (long)rows * heads * 256, 256, scores, rows);
            if (_useOptimizedKernels)
                _lane.Run2D("q35v_attention_context_tiled", Round16(headWidth),
                    heads * Round16(rows), 16, 16, qkv, scores, context, rows, heads, headWidth);
            else
                _lane.Run("q35v_attention_context", (long)rows * heads * headWidth, 128,
                    qkv, scores, context, rows, heads, headWidth);
            return context;
        }
        catch { context.Dispose(); throw; }
    }

    /// <summary>
    /// Full FP32 softmax with high+residual F16 XMX dot products. Every cross
    /// product is included; accumulation order differs from exact FP32 FMA.
    /// Q/K/V and probability packing are transient and do not change weights.
    /// </summary>
    internal static ArcBuffer AttentionXmx(ArcExecutionLane lane, ArcBuffer qkv, ArcBuffer scores,
        int rows, int heads, int headWidth, bool directProbabilities = true,
        bool specializeHead = true)
    {
        ArgumentNullException.ThrowIfNull(lane);
        if (rows < 1 || heads < 1 || headWidth < 1)
            throw new ArgumentOutOfRangeException(nameof(rows));
        int rowBlocks = checked((rows + 7) / 8), rowColumns = checked((rows + 15) / 16);
        int featureBlocks = checked((headWidth + 15) / 16);
        long queryElements = checked((long)heads * rowBlocks * featureBlocks * 128);
        long keyPairs = checked((long)heads * rowColumns * featureBlocks * 128);
        long valuePairs = checked((long)heads * featureBlocks * rowColumns * 128);
        long probabilityElements = checked((long)heads * rowBlocks * rowColumns * 128);
        ArcBuffer output = lane.Allocate(checked(rows * heads * headWidth));
        try
        {
            using (ArcBuffer qHigh = lane.AllocateBytes(checked((int)(queryElements * sizeof(ushort)))))
            using (ArcBuffer qLow = lane.AllocateBytes(checked((int)(queryElements * sizeof(ushort)))))
            using (ArcBuffer kHigh = lane.AllocateBytes(checked((int)(keyPairs * sizeof(uint)))))
            using (ArcBuffer kLow = lane.AllocateBytes(checked((int)(keyPairs * sizeof(uint)))))
            {
                lane.Run("q35v_attention_pack_q", queryElements, 0,
                    qkv, qHigh, qLow, rows, heads, headWidth);
                lane.Run("q35v_attention_pack_k", keyPairs, 0,
                    qkv, kHigh, kLow, rows, heads, headWidth);
                lane.Run2D("q35v_attention_scores_xmx", (rows + 63L) / 64 * 16,
                    heads * ((rows + 127L) / 128) * 16, 16, 16,
                    qHigh, qLow, kHigh, kLow, scores, rows, heads, headWidth);
            }
            lane.Run("q35v_attention_softmax", (long)rows * heads * 256, 256, scores, rows);
            if (directProbabilities)
            {
                using ArcBuffer directVHigh = lane.AllocateBytes(checked((int)(valuePairs * sizeof(uint))));
                using ArcBuffer directVLow = lane.AllocateBytes(checked((int)(valuePairs * sizeof(uint))));
                lane.Run("q35v_attention_pack_v", valuePairs, 0,
                    qkv, directVHigh, directVLow, rows, heads, headWidth);
                bool head80 = specializeHead && headWidth <= 80;
                lane.Run2D(head80 ? "q35v_attention_context_xmx_direct80"
                    : "q35v_attention_context_xmx_direct", head80 ? 16 : (headWidth + 63L) / 64 * 16,
                    heads * ((rows + 127L) / 128) * 16, 16, 16,
                    scores, directVHigh, directVLow, output, rows, heads, headWidth);
                return output;
            }
            using ArcBuffer pHigh = lane.AllocateBytes(checked((int)(probabilityElements * sizeof(ushort))));
            using ArcBuffer pLow = lane.AllocateBytes(checked((int)(probabilityElements * sizeof(ushort))));
            using ArcBuffer vHigh = lane.AllocateBytes(checked((int)(valuePairs * sizeof(uint))));
            using ArcBuffer vLow = lane.AllocateBytes(checked((int)(valuePairs * sizeof(uint))));
            lane.Run("q35v_attention_pack_p", probabilityElements, 0, scores, pHigh, pLow, rows, heads);
            lane.Run("q35v_attention_pack_v", valuePairs, 0, qkv, vHigh, vLow, rows, heads, headWidth);
            lane.Run2D("q35v_attention_context_xmx", (headWidth + 63L) / 64 * 16,
                heads * ((rows + 127L) / 128) * 16, 16, 16,
                pHigh, pLow, vHigh, vLow, output, rows, heads, headWidth);
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    /// <summary>Full attention with online FP32 softmax and no quadratic workspace.</summary>
    internal static ArcBuffer AttentionXmxFlash(ArcExecutionLane lane, ArcBuffer qkv,
        int rows, int heads, int headWidth)
    {
        ArgumentNullException.ThrowIfNull(lane);
        if (rows < 1 || heads < 1 || headWidth is < 1 or > 80)
            throw new ArgumentOutOfRangeException(nameof(headWidth));
        int rowBlocks = checked((rows + 7) / 8), rowColumns = checked((rows + 15) / 16);
        int featureBlocks = checked((headWidth + 15) / 16);
        long queryElements = checked((long)heads * rowBlocks * featureBlocks * 128);
        long keyPairs = checked((long)heads * rowColumns * featureBlocks * 128);
        long valuePairs = checked((long)heads * featureBlocks * rowColumns * 128);
        ArcBuffer output = lane.Allocate(checked(rows * heads * headWidth));
        try
        {
            using ArcBuffer qHigh = lane.AllocateBytes(checked((int)(queryElements * sizeof(ushort))));
            using ArcBuffer qLow = lane.AllocateBytes(checked((int)(queryElements * sizeof(ushort))));
            using ArcBuffer kHigh = lane.AllocateBytes(checked((int)(keyPairs * sizeof(uint))));
            using ArcBuffer kLow = lane.AllocateBytes(checked((int)(keyPairs * sizeof(uint))));
            using ArcBuffer vHigh = lane.AllocateBytes(checked((int)(valuePairs * sizeof(uint))));
            using ArcBuffer vLow = lane.AllocateBytes(checked((int)(valuePairs * sizeof(uint))));
            lane.Run("q35v_attention_pack_q", queryElements, 0, qkv, qHigh, qLow, rows, heads, headWidth);
            lane.Run("q35v_attention_pack_k", keyPairs, 0, qkv, kHigh, kLow, rows, heads, headWidth);
            lane.Run("q35v_attention_pack_v", valuePairs, 0, qkv, vHigh, vLow, rows, heads, headWidth);
            lane.Run2D("q35v_attention_flash_xmx", 16,
                heads * ((rows + 127L) / 128) * 16, 16, 16,
                qHigh, qLow, kHigh, kLow, vHigh, vLow, output, rows, heads, headWidth);
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    private ArcBuffer LayerNorm(ArcBuffer input, string scale, string bias, int rows, int width)
    {
        ArcBuffer result = _lane.Allocate(checked(rows * width));
        try
        {
            _lane.Run("q35v_layer_norm", (long)rows * 256, 256,
                input, Weight(scale), Weight(bias), result, width, Descriptor.LayerNormEpsilon);
            return result;
        }
        catch { result.Dispose(); throw; }
    }

    private ArcBuffer Linear(ArcBuffer input, string weight, string bias,
        int rows, int inputWidth, int outputWidth, bool gelu = false)
    {
        ArcBuffer result = _lane.Allocate(checked(rows * outputWidth));
        try
        {
            if (UsesXmxLinear)
            {
                long packedElements = (rows + 7L) / 8 * 8 * Round16(inputWidth);
                using ArcBuffer high = _lane.AllocateBytes(checked((int)(packedElements * sizeof(ushort))));
                using ArcBuffer low = _lane.AllocateBytes(checked((int)(packedElements * sizeof(ushort))));
                _lane.Run("q35v_linear_pack_a_f16", packedElements, 0,
                    input, high, low, rows, inputWidth);
                _lane.Run2D("q35v_linear_f16_xmx_packed", (outputWidth + 63L) / 64 * 16,
                    (rows + 127L) / 128 * 16, 16, 16,
                    high, low, Weight(weight), Weight(bias), result,
                    rows, inputWidth, outputWidth, gelu ? 1 : 0);
                return result;
            }
            int rowTile = _useOptimizedKernels ? 64 : 16;
            long globalX = _useOptimizedKernels ? (outputWidth + 63L) / 64 * 16 : Round16(outputWidth);
            long globalY = (rows + (long)rowTile - 1) / rowTile * 16;
            string kernel = _useOptimizedKernels ? "q35v_linear_f16_fast" : "q35v_linear_f16";
            _lane.Run2D(kernel, globalX, globalY, 16, 16,
                input, Weight(weight), Weight(bias), result,
                rows, inputWidth, outputWidth, gelu ? 1 : 0);
            return result;
        }
        catch { result.Dispose(); throw; }
    }

    private static long Round16(int value) => (value + 15L) / 16 * 16;

    public void Dispose()
    {
        _operation.Wait();
        try
        {
            if (_disposed) return;
            _disposed = true;
            foreach (ArcBuffer buffer in _weights.Values) buffer.Dispose();
            _weights.Clear();
            _lane.Dispose();
        }
        finally { _operation.Release(); }
    }

    private static Qwen35VisionDescriptor Inspect(GgufReader reader)
    {
        if (String("general.architecture") != "clip"
            || String("clip.projector_type") != "qwen3vl_merger")
            throw new NotSupportedException("Expected a Qwen3-VL clip/qwen3vl_merger GGUF.");
        if (reader.Metadata.TryGetValue("clip.has_vision_encoder", out object? hasVision)
            && hasVision is not true)
            throw new NotSupportedException("GGUF has no vision encoder.");
        if (reader.Metadata.TryGetValue("clip.use_gelu", out object? gelu) && gelu is not true)
            throw new NotSupportedException("Only GELU Qwen3-VL vision towers are supported.");
        if (reader.Metadata.TryGetValue("clip.vision.is_deepstack_layers", out object? deepstack)
            && deepstack is object[] flags && flags.Any(flag => flag is not false))
            throw new NotSupportedException("Qwen3-VL deepstack vision layers are not supported.");
        int projection = PositiveInt("clip.vision.projection_dim");
        int patch = PositiveInt("clip.vision.patch_size");
        int merge = PositiveInt("clip.vision.spatial_merge_size");
        int width = PositiveInt("clip.vision.embedding_length");
        int imageSize = PositiveInt("clip.vision.image_size");
        int layers = PositiveInt("clip.vision.block_count");
        int heads = PositiveInt("clip.vision.attention.head_count");
        int feedForward = PositiveInt("clip.vision.feed_forward_length");
        float epsilon = PositiveFloat("clip.vision.attention.layer_norm_epsilon");
        if (merge != 2 || patch != 16 || width % heads != 0 || width / heads % 4 != 0)
            throw new NotSupportedException("Unsupported Qwen3-VL patch, merge or head geometry.");
        var tensors = reader.Tensors.ToDictionary(t => t.Name, StringComparer.Ordinal);
        var expected = new HashSet<string>(StringComparer.Ordinal);
        int mergeWidth = checked(width * merge * merge);
        Matrix("v.patch_embd.weight", [patch, patch, 3, width]);
        Matrix("v.patch_embd.weight.1", [patch, patch, 3, width]);
        Vector("v.patch_embd.bias", width);
        GgufTensorInfo positions = Vector("v.position_embd.weight", width,
            checked(imageSize / patch * (imageSize / patch)));
        int side = checked((int)Math.Sqrt(positions.Shape[1]));
        if (side * side != (int)positions.Shape[1] || imageSize / patch != side)
            throw new InvalidDataException("Vision learned position table must be a square patch grid.");
        Vector("v.post_ln.weight", width);
        Vector("v.post_ln.bias", width);
        Matrix("mm.0.weight", [mergeWidth, mergeWidth]);
        Vector("mm.0.bias", mergeWidth);
        Matrix("mm.2.weight", [mergeWidth, projection]);
        Vector("mm.2.bias", projection);
        for (int layer = 0; layer < layers; layer++)
        {
            string p = $"v.blk.{layer}.";
            Matrix(p + "attn_qkv.weight", [width, 3 * width]);
            Vector(p + "attn_qkv.bias", 3 * width);
            Matrix(p + "attn_out.weight", [width, width]);
            Vector(p + "attn_out.bias", width);
            Matrix(p + "ffn_up.weight", [width, feedForward]);
            Vector(p + "ffn_up.bias", feedForward);
            Matrix(p + "ffn_down.weight", [feedForward, width]);
            Vector(p + "ffn_down.bias", width);
            Vector(p + "ln1.weight", width);
            Vector(p + "ln1.bias", width);
            Vector(p + "ln2.weight", width);
            Vector(p + "ln2.bias", width);
        }
        if (expected.Count != tensors.Count)
            throw new NotSupportedException("GGUF contains unsupported Qwen3-VL vision tensors.");
        return new Qwen35VisionDescriptor(projection, patch, merge, width,
            imageSize, layers, heads, width / heads, feedForward, side, epsilon);

        string String(string key)
            => reader.Metadata.TryGetValue(key, out object? value) && value is string text
                ? text : throw new InvalidDataException($"Missing GGUF string metadata '{key}'.");

        int PositiveInt(string key)
        {
            if (!reader.Metadata.TryGetValue(key, out object? value))
                throw new InvalidDataException($"Missing GGUF metadata '{key}'.");
            int parsed = value switch
            {
                byte v => v, ushort v => v, uint v when v <= int.MaxValue => (int)v,
                int v => v, long v when v is > 0 and <= int.MaxValue => (int)v,
                ulong v when v is > 0 and <= int.MaxValue => (int)v,
                _ => throw new InvalidDataException($"Invalid GGUF integer metadata '{key}'.")
            };
            return parsed > 0 ? parsed
                : throw new InvalidDataException($"GGUF metadata '{key}' must be positive.");
        }

        float PositiveFloat(string key)
        {
            if (!reader.Metadata.TryGetValue(key, out object? value))
                throw new InvalidDataException($"Missing GGUF metadata '{key}'.");
            float parsed = value switch { float f => f, double d => (float)d,
                _ => throw new InvalidDataException($"Invalid GGUF float metadata '{key}'.") };
            return parsed > 0 && float.IsFinite(parsed) ? parsed
                : throw new InvalidDataException($"GGUF metadata '{key}' must be positive.");
        }

        GgufTensorInfo Require(string name, uint[] allowedTypes, params int[] shape)
        {
            expected.Add(name);
            if (!tensors.TryGetValue(name, out GgufTensorInfo? tensor))
                throw new InvalidDataException($"Missing Qwen3-VL tensor '{name}'.");
            if (!tensor.Shape.SequenceEqual(shape.Select(dimension => (ulong)dimension))
                || !allowedTypes.Contains(tensor.Type))
                throw new NotSupportedException($"Unsupported Qwen3-VL tensor '{name}' shape or type.");
            return tensor;
        }
        GgufTensorInfo Matrix(string name, int[] shape)
            => Require(name, [Qwen2Gguf.F16Type], shape);
        GgufTensorInfo Vector(string name, params int[] shape)
            => Require(name, [Qwen2Gguf.F32Type, Qwen2Gguf.F16Type], shape);
    }

    private static int ElementCount(GgufTensorInfo tensor)
    {
        long result = 1;
        foreach (ulong dimension in tensor.Shape)
            result = checked(result * (long)dimension);
        return checked((int)result);
    }
}
