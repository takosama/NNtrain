namespace NNtrain;

/// <summary>Helpers for inspecting a Qwen2-family GGUF before model construction.</summary>
public static class Qwen2Gguf
{
    // llama.cpp ggml_type value for Q4_K.
    public const uint Q4KType = 12;

    public static Qwen2GgufDescriptor Inspect(string path)
    {
        using var gguf = new GgufReader(path);
        string architecture = RequiredString(gguf, "general.architecture");
        if (!string.Equals(architecture, "qwen2", StringComparison.Ordinal))
            throw new InvalidDataException($"Expected qwen2 GGUF, got '{architecture}'.");

        int layers = RequiredInt(gguf, "qwen2.block_count");
        int embedding = RequiredInt(gguf, "qwen2.embedding_length");
        int heads = RequiredInt(gguf, "qwen2.attention.head_count");
        int kvHeads = RequiredInt(gguf, "qwen2.attention.head_count_kv");
        int context = RequiredInt(gguf, "qwen2.context_length");
        int feedForward = RequiredInt(gguf, "qwen2.feed_forward_length");

        return new Qwen2GgufDescriptor(
            layers, embedding, heads, kvHeads, context, feedForward,
            gguf.Tensors.ToArray());
    }

    public static float[] ReadQ4KTensor(string path, string tensorName)
    {
        using var gguf = new GgufReader(path);
        GgufTensorInfo tensor = gguf.GetTensor(tensorName);
        if (tensor.Type != Q4KType)
            throw new NotSupportedException($"Tensor '{tensorName}' has GGML type {tensor.Type}, not Q4_K.");

        long elements64 = 1;
        foreach (ulong dimension in tensor.Shape)
            elements64 = checked(elements64 * (long)dimension);
        if (elements64 > int.MaxValue)
            throw new NotSupportedException("A single tensor exceeds the current managed decoder limit.");
        int elements = (int)elements64;
        int blocks = checked((elements + GgufQ4K.BlockElements - 1) / GgufQ4K.BlockElements);
        int bytes = checked(blocks * GgufQ4K.BlockBytes);
        byte[] payload = new byte[bytes];
        Stream stream = gguf.OpenTensorData(tensor);
        stream.ReadExactly(payload);
        return GgufQ4K.Dequantize(payload, elements);
    }

    private static string RequiredString(GgufReader gguf, string key)
        => gguf.Metadata.TryGetValue(key, out object? value) && value is string text
            ? text : throw new InvalidDataException($"Missing GGUF metadata '{key}'.");

    private static int RequiredInt(GgufReader gguf, string key)
    {
        if (!gguf.Metadata.TryGetValue(key, out object? value))
            throw new InvalidDataException($"Missing GGUF metadata '{key}'.");
        return checked((int)Convert.ToUInt64(value));
    }
}

public sealed record Qwen2GgufDescriptor(
    int LayerCount,
    int EmbeddingLength,
    int HeadCount,
    int KvHeadCount,
    int ContextLength,
    int FeedForwardLength,
    IReadOnlyList<GgufTensorInfo> Tensors);
