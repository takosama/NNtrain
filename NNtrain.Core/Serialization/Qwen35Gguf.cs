namespace NNtrain;

/// <summary>Validates the dense Qwen3.5 text architecture without reading tensor payloads.</summary>
public static class Qwen35Gguf
{
    public const uint Q5KType = 13, IQ3SType = 21, IQ2SType = 22;
    public const uint PQ20Type = 142, PTQ10Type = 143;

    internal static bool IsSupportedQuantization(uint type)
        => type is Qwen2Gguf.Q4KType or Q5KType or Qwen2Gguf.Q6KType or IQ2SType or IQ3SType or PQ20Type or PTQ10Type;

    internal static bool IsSupportedMatrixStorage(uint type)
        => IsSupportedQuantization(type) || type == Qwen2Gguf.BF16Type;

    internal static int QuantizedBlockElements(uint type) => type switch
    {
        PQ20Type or PTQ10Type => 128,
        Qwen2Gguf.BF16Type => 1,
        _ when IsSupportedQuantization(type) => 256,
        _ => throw new NotSupportedException($"Unsupported Qwen3.5 matrix type {type}.")
    };

    internal static int QuantizedBlockBytes(uint type) => type switch
    {
        Qwen2Gguf.Q4KType => GgufQ4K.BlockBytes,
        Q5KType => 176,
        Qwen2Gguf.Q6KType => GgufQ6K.BlockBytes,
        IQ2SType => 82,
        IQ3SType => 110,
        PQ20Type => 34,
        PTQ10Type => 28,
        Qwen2Gguf.BF16Type => 2,
        _ => throw new NotSupportedException($"Unsupported Qwen3.5 quantization type {type}.")
    };

    public static Qwen35GgufDescriptor Inspect(string path)
    {
        using var gguf = new GgufReader(path);
        return Inspect(gguf);
    }

    internal static Qwen35GgufDescriptor Inspect(GgufReader gguf)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        if (!gguf.Metadata.TryGetValue("general.architecture", out object? architecture)
            || architecture is not string name || name != "qwen35")
            throw new InvalidDataException($"Expected qwen35 GGUF, got '{architecture}'.");
        if (!gguf.Metadata.TryGetValue("tokenizer.ggml.tokens", out object? tokens)
            || tokens is not object[] vocabulary || vocabulary.Length == 0
            || vocabulary.Any(token => token is not string))
            throw new InvalidDataException("GGUF tokenizer token table is missing or invalid.");

        int layers = RequiredInt(gguf, "block_count");
        int width = RequiredInt(gguf, "embedding_length");
        int heads = RequiredInt(gguf, "attention.head_count");
        int kvHeads = RequiredInt(gguf, "attention.head_count_kv");
        int headWidth = RequiredInt(gguf, "attention.key_length");
        int valueWidth = RequiredInt(gguf, "attention.value_length");
        int context = RequiredInt(gguf, "context_length");
        int feedForward = RequiredInt(gguf, "feed_forward_length");
        int ropeDimensions = RequiredInt(gguf, "rope.dimension_count");
        int linearKeyHeads = RequiredInt(gguf, "ssm.group_count");
        int linearValueHeads = RequiredInt(gguf, "ssm.time_step_rank");
        int linearHeadWidth = RequiredInt(gguf, "ssm.state_size");
        int innerSize = RequiredInt(gguf, "ssm.inner_size");
        int convKernel = RequiredInt(gguf, "ssm.conv_kernel");
        int interval = RequiredInt(gguf, "full_attention_interval");
        float epsilon = PositiveFloat(gguf, "attention.layer_norm_rms_epsilon", 1e-6f);
        float theta = PositiveFloat(gguf, "rope.freq_base", 10_000_000f);

        if (heads % kvHeads != 0 || headWidth != valueWidth)
            throw new InvalidDataException("Qwen3.5 attention requires matching key/value lengths and query heads divisible by KV heads.");
        if (ropeDimensions > headWidth || (ropeDimensions & 1) != 0)
            throw new InvalidDataException("Qwen3.5 RoPE dimension count must be even and no larger than the head width.");
        if (linearValueHeads % linearKeyHeads != 0
            || (long)linearValueHeads * linearHeadWidth != innerSize)
            throw new InvalidDataException("Qwen3.5 ssm.inner_size must equal time_step_rank * state_size, and value heads must be divisible by key heads.");
        foreach (string key in new[] { "expert_count", "expert_used_count" })
            if (gguf.Metadata.TryGetValue("qwen35." + key, out object? expertCount)
                && NonnegativeInt(expertCount, "qwen35." + key) != 0)
                throw new NotSupportedException("Qwen3.5 MoE models are not supported by the dense text loader.");
        if (gguf.Metadata.TryGetValue("qwen35.rope.scaling.type", out object? scaling)
            && scaling is not "none")
            throw new NotSupportedException("Qwen3.5 scaled RoPE is not implemented by the text loader.");
        if (gguf.Metadata.TryGetValue("qwen35.rope.dimension_sections", out object? sections))
        {
            if (sections is not object[] values || values.Length == 0
                || values.Sum(value => (long)NonnegativeInt(value, "qwen35.rope.dimension_sections")) * 2 != ropeDimensions)
                throw new InvalidDataException("Qwen3.5 RoPE sections must sum to half the rotary dimension count.");
        }

        var descriptor = new Qwen35GgufDescriptor(
            vocabulary.Length, layers, width, heads, kvHeads, headWidth,
            context, feedForward, epsilon, theta, ropeDimensions,
            linearKeyHeads, linearValueHeads, linearHeadWidth, convKernel,
            interval, gguf.Tensors.ToArray());
        ValidateRecurrentPattern(gguf, descriptor, "qwen35.recurrent_layers");
        ValidateRecurrentPattern(gguf, descriptor, "qwen35.attention.recurrent_layers");
        ValidateDirectory(gguf, descriptor);
        _ = Qwen35PrismMetadata.Read(gguf);
        return descriptor;
    }

    private static void ValidateDirectory(GgufReader gguf, Qwen35GgufDescriptor d)
    {
        var tensors = new Dictionary<string, GgufTensorInfo>(StringComparer.Ordinal);
        foreach (GgufTensorInfo tensor in gguf.Tensors)
            if (!tensors.TryAdd(tensor.Name, tensor))
                throw new InvalidDataException($"Duplicate GGUF tensor '{tensor.Name}'.");
        var expected = new HashSet<string>(StringComparer.Ordinal);
        int queryWidth = DimensionProduct(d.HeadCount, d.HeadWidth);
        int kvWidth = DimensionProduct(d.KvHeadCount, d.HeadWidth);
        int linearWidth = DimensionProduct(d.LinearValueHeads, d.LinearHeadWidth);
        int qkvWidth = DimensionProduct(2L * d.LinearKeyHeads + d.LinearValueHeads, d.LinearHeadWidth);
        Matrix("token_embd.weight", d.EmbeddingLength, d.VocabularySize);
        Dense("output_norm.weight", d.EmbeddingLength);
        if (tensors.ContainsKey("output.weight"))
            Matrix("output.weight", d.EmbeddingLength, d.VocabularySize);
        for (int layer = 0; layer < d.LayerCount; ++layer)
        {
            string p = $"blk.{layer}.";
            Dense(p + "attn_norm.weight", d.EmbeddingLength);
            Dense(p + "post_attention_norm.weight", d.EmbeddingLength);
            Matrix(p + "ffn_gate.weight", d.EmbeddingLength, d.FeedForwardLength);
            Matrix(p + "ffn_up.weight", d.EmbeddingLength, d.FeedForwardLength);
            Matrix(p + "ffn_down.weight", d.FeedForwardLength, d.EmbeddingLength);
            if (d.IsRecurrent(layer))
            {
                Matrix(p + "attn_qkv.weight", d.EmbeddingLength, qkvWidth);
                Matrix(p + "attn_gate.weight", d.EmbeddingLength, linearWidth);
                Matrix(p + "ssm_alpha.weight", d.EmbeddingLength, d.LinearValueHeads);
                Matrix(p + "ssm_beta.weight", d.EmbeddingLength, d.LinearValueHeads);
                Dense(p + "ssm_conv1d.weight", d.ConvKernel, qkvWidth);
                Dense(p + "ssm_a", d.LinearValueHeads);
                Dense(p + "ssm_dt.bias", d.LinearValueHeads);
                Dense(p + "ssm_norm.weight", d.LinearHeadWidth);
                Matrix(p + "ssm_out.weight", linearWidth, d.EmbeddingLength);
            }
            else
            {
                Matrix(p + "attn_q.weight", d.EmbeddingLength, DimensionProduct(2, queryWidth));
                Matrix(p + "attn_k.weight", d.EmbeddingLength, kvWidth);
                Matrix(p + "attn_v.weight", d.EmbeddingLength, kvWidth);
                Dense(p + "attn_q_norm.weight", d.HeadWidth);
                Dense(p + "attn_k_norm.weight", d.HeadWidth);
                Matrix(p + "attn_output.weight", queryWidth, d.EmbeddingLength);
            }
        }
        foreach (string name in tensors.Keys)
            if (!expected.Contains(name))
                throw new NotSupportedException($"Unexpected tensor '{name}'; only the dense Qwen3.5 text tensor directory is supported.");

        GgufTensorInfo Shape(string name, int[] shape)
        {
            expected.Add(name);
            if (!tensors.TryGetValue(name, out GgufTensorInfo? tensor))
                throw new InvalidDataException($"Required GGUF tensor '{name}' is missing.");
            if (tensor.Shape.Count != shape.Length
                || shape.Where((size, index) => tensor.Shape[index] != (ulong)size).Any())
                throw new InvalidDataException($"GGUF tensor '{name}' shape [{string.Join(",", tensor.Shape)}] does not match expected [{string.Join(",", shape)}].");
            return tensor;
        }

        void Matrix(string name, int inputWidth, int outputWidth)
        {
            GgufTensorInfo tensor = Shape(name, [inputWidth, outputWidth]);
            if (!IsSupportedMatrixStorage(tensor.Type))
                throw new NotSupportedException($"Qwen3.5 matrix '{name}' requires Q4_K/Q5_K/Q6_K/IQ2_S/IQ3_S/PQ2_0/PTQ1_0/BF16 storage; got type {tensor.Type}.");
            int blockElements = QuantizedBlockElements(tensor.Type);
            if (inputWidth % blockElements != 0)
                throw new InvalidDataException($"Qwen3.5 matrix '{name}' requires a row width divisible by {blockElements}.");
            long bytes = (long)outputWidth * (inputWidth / blockElements)
                * QuantizedBlockBytes(tensor.Type);
            if (bytes > int.MaxValue)
                throw new NotSupportedException($"Qwen3.5 matrix '{name}' exceeds the managed payload limit.");
        }

        void Dense(string name, params int[] shape)
        {
            GgufTensorInfo tensor = Shape(name, shape);
            if (tensor.Type is not (Qwen2Gguf.F32Type or Qwen2Gguf.F16Type or Qwen2Gguf.BF16Type))
                throw new NotSupportedException($"Qwen3.5 tensor '{name}' requires F32/F16/BF16 storage; got type {tensor.Type}.");
            long elements = shape.Aggregate(1L, (count, size) => checked(count * size));
            if (elements > int.MaxValue / (tensor.Type == Qwen2Gguf.F32Type ? 4 : 2))
                throw new NotSupportedException($"Qwen3.5 tensor '{name}' exceeds the managed payload limit.");
        }
    }

    private static void ValidateRecurrentPattern(GgufReader gguf, Qwen35GgufDescriptor d, string key)
    {
        if (!gguf.Metadata.TryGetValue(key, out object? value)) return;
        if (value is not object[] pattern || pattern.Length != d.LayerCount)
            throw new InvalidDataException($"GGUF metadata '{key}' must contain one recurrent flag per layer.");
        for (int layer = 0; layer < pattern.Length; ++layer)
        {
            int flag = pattern[layer] is bool recurrent ? (recurrent ? 1 : 0)
                : NonnegativeInt(pattern[layer], key);
            if (flag > 1)
                throw new InvalidDataException($"GGUF metadata '{key}' must contain only boolean or 0/1 flags.");
            if ((flag == 1) != d.IsRecurrent(layer))
                throw new NotSupportedException($"GGUF metadata '{key}' conflicts with full_attention_interval at layer {layer}.");
        }
    }

    private static int RequiredInt(GgufReader gguf, string suffix)
    {
        string key = "qwen35." + suffix;
        if (!gguf.Metadata.TryGetValue(key, out object? value))
            throw new InvalidDataException($"Missing GGUF metadata '{key}'.");
        int result = NonnegativeInt(value, key);
        return result > 0 ? result
            : throw new InvalidDataException($"GGUF metadata '{key}' must be positive.");
    }

    private static int NonnegativeInt(object value, string key)
    {
        long number = value switch
        {
            byte v => v, sbyte v => v, ushort v => v, short v => v,
            uint v => v, int v => v, long v => v,
            ulong v when v <= int.MaxValue => (long)v,
            _ => throw new InvalidDataException($"GGUF metadata '{key}' must be an integer in the supported range.")
        };
        if (number is < 0 or > int.MaxValue)
            throw new InvalidDataException($"GGUF metadata '{key}' is outside the supported range.");
        return (int)number;
    }

    private static float PositiveFloat(GgufReader gguf, string suffix, float fallback)
    {
        string key = "qwen35." + suffix;
        if (!gguf.Metadata.TryGetValue(key, out object? value)) return fallback;
        float number = value switch
        {
            float v => v, double v => (float)v,
            _ => throw new InvalidDataException($"GGUF metadata '{key}' must be floating point.")
        };
        if (!(number > 0f) || !float.IsFinite(number))
            throw new InvalidDataException($"GGUF metadata '{key}' must be finite and positive.");
        return number;
    }

    private static int DimensionProduct(long left, int right)
    {
        long product = checked(left * right);
        return product <= int.MaxValue ? (int)product
            : throw new InvalidDataException("Qwen3.5 tensor dimensions exceed the supported range.");
    }
}

public sealed record Qwen35GgufDescriptor(
    int VocabularySize,
    int LayerCount,
    int EmbeddingLength,
    int HeadCount,
    int KvHeadCount,
    int HeadWidth,
    int ContextLength,
    int FeedForwardLength,
    float RmsEpsilon,
    float RopeTheta,
    int RopeDimensionCount,
    int LinearKeyHeads,
    int LinearValueHeads,
    int LinearHeadWidth,
    int ConvKernel,
    int FullAttentionInterval,
    IReadOnlyList<GgufTensorInfo> Tensors)
{
    public bool IsRecurrent(int layer) => (layer + 1) % FullAttentionInterval != 0;
}
