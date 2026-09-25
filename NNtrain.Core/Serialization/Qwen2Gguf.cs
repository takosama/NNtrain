using System.Buffers.Binary;

namespace NNtrain;

/// <summary>Qwen2 GGUF inspection, dequantization and model loading.</summary>
public static class Qwen2Gguf
{
    public const uint F32Type = 0;
    public const uint F16Type = 1;
    public const uint Q4KType = 12;
    public const uint Q6KType = 14;
    public const uint BF16Type = 30;

    public static Qwen2GgufDescriptor Inspect(string path)
    {
        using var gguf = new GgufReader(path);
        string architecture = RequiredString(gguf, "general.architecture");
        if (!string.Equals(architecture, "qwen2", StringComparison.Ordinal))
            throw new InvalidDataException($"Expected qwen2 GGUF, got '{architecture}'.");

        int vocabulary = gguf.Metadata.TryGetValue("tokenizer.ggml.tokens", out object? tokens)
            && tokens is object[] tokenArray
                ? tokenArray.Length
                : throw new InvalidDataException("GGUF tokenizer token table is missing.");
        int layers = RequiredInt(gguf, "qwen2.block_count");
        int embedding = RequiredInt(gguf, "qwen2.embedding_length");
        int heads = RequiredInt(gguf, "qwen2.attention.head_count");
        int kvHeads = RequiredInt(gguf, "qwen2.attention.head_count_kv");
        int context = RequiredInt(gguf, "qwen2.context_length");
        int feedForward = RequiredInt(gguf, "qwen2.feed_forward_length");
        float rmsEpsilon = OptionalFloat(
            gguf, "qwen2.attention.layer_norm_rms_epsilon", 1e-6f);
        float ropeTheta = OptionalFloat(
            gguf, "qwen2.rope.freq_base", 1_000_000f);

        return new Qwen2GgufDescriptor(
            vocabulary, layers, embedding, heads, kvHeads, context,
            feedForward, rmsEpsilon, ropeTheta, gguf.Tensors.ToArray());
    }

    public static Qwen2QuantizedForCausalLM LoadQuantizedModel(
        string path,
        TensorDType activationDType = TensorDType.Float32)
    {
        Qwen2GgufDescriptor d = Inspect(path);
        using var gguf = new GgufReader(path);

        Parameter embedding = DenseParameter(
            gguf, "token_embd.weight", "token_embd.weight", activationDType);
        var blocks = new QwenQuantizedBlock[d.LayerCount];
        for (int i = 0; i < blocks.Length; ++i)
        {
            string p = $"blk.{i}.";
            var inputNorm = new QwenRmsNorm(d.EmbeddingLength, d.RmsEpsilon, activationDType);
            Copy(gguf, inputNorm.Weight, p + "attn_norm.weight");
            var postNorm = new QwenRmsNorm(d.EmbeddingLength, d.RmsEpsilon, activationDType);
            Copy(gguf, postNorm.Weight, p + "ffn_norm.weight");

            int headWidth = d.EmbeddingLength / d.HeadCount;
            int kvWidth = checked(d.KvHeadCount * headWidth);
            var attention = new QwenQuantizedAttention(
                QuantizedLinear(gguf, p + "attn_q.weight", p + "attn_q.bias",
                    d.EmbeddingLength, d.EmbeddingLength, activationDType),
                QuantizedLinear(gguf, p + "attn_k.weight", p + "attn_k.bias",
                    d.EmbeddingLength, kvWidth, activationDType),
                QuantizedLinear(gguf, p + "attn_v.weight", p + "attn_v.bias",
                    d.EmbeddingLength, kvWidth, activationDType),
                QuantizedLinear(gguf, p + "attn_output.weight", null,
                    d.EmbeddingLength, d.EmbeddingLength, activationDType),
                d.HeadCount, d.KvHeadCount, d.RopeTheta, activationDType);
            var mlp = new QwenQuantizedMlp(
                QuantizedLinear(gguf, p + "ffn_gate.weight", null,
                    d.EmbeddingLength, d.FeedForwardLength, activationDType),
                QuantizedLinear(gguf, p + "ffn_up.weight", null,
                    d.EmbeddingLength, d.FeedForwardLength, activationDType),
                QuantizedLinear(gguf, p + "ffn_down.weight", null,
                    d.FeedForwardLength, d.EmbeddingLength, activationDType),
                activationDType);
            blocks[i] = new QwenQuantizedBlock(
                inputNorm, attention, postNorm, mlp, activationDType);
        }

        var finalNorm = new QwenRmsNorm(d.EmbeddingLength, d.RmsEpsilon, activationDType);
        Copy(gguf, finalNorm.Weight, "output_norm.weight");

        QwenQuantizedLinear head;
        if (gguf.Tensors.Any(t => t.Name == "output.weight"))
        {
            head = QuantizedLinear(
                gguf, "output.weight", null,
                d.EmbeddingLength, d.VocabularySize, activationDType);
        }
        else
        {
            throw new NotSupportedException(
                "Tied token-embedding output is dense in this first resident path. " +
                "Use a GGUF containing output.weight until the quantized tied-head path is added.");
        }

        return new Qwen2QuantizedForCausalLM(
            d.VocabularySize, d.ContextLength, d.EmbeddingLength,
            d.HeadCount, d.KvHeadCount, embedding, blocks, finalNorm,
            head, activationDType);
    }

    private static QwenQuantizedLinear QuantizedLinear(
        GgufReader gguf, string weightName, string? biasName,
        int inputWidth, int outputWidth, TensorDType dtype)
    {
        GgufTensorInfo weight = gguf.GetTensor(weightName);
        if (weight.Type is not (Q4KType or Q6KType))
            throw new NotSupportedException(
                $"Resident inference requires Q4_K/Q6_K matrix '{weightName}', got type {weight.Type}.");
        int blockBytes = weight.Type == Q4KType ? GgufQ4K.BlockBytes : GgufQ6K.BlockBytes;
        int bytes = checked(outputWidth * (inputWidth / 256) * blockBytes);
        byte[] payload = gguf.ReadTensorBytes(weight, bytes);
        float[]? bias = null;
        if (biasName is not null)
        {
            GgufTensorInfo? biasTensor = gguf.Tensors.FirstOrDefault(t => t.Name == biasName);
            if (biasTensor is not null) bias = ReadTensor(gguf, biasTensor);
        }
        return new QwenQuantizedLinear(
            payload, weight.Type, inputWidth, outputWidth, bias, dtype);
    }

    private static Parameter DenseParameter(
        GgufReader gguf, string tensorName, string parameterName, TensorDType dtype)
    {
        GgufTensorInfo tensor = gguf.GetTensor(tensorName);
        int[] shape = tensor.Shape.Reverse().Select(v => checked((int)v)).ToArray();
        return new Parameter(
            ReadTensor(gguf, tensor), shape, parameterName,
            WeightDecayPolicy.Exclude, dtype);
    }

    public static Qwen2ForCausalLM LoadModel(
        string path,
        TensorDType dtype = TensorDType.Float32)
    {
        Qwen2GgufDescriptor d = Inspect(path);
        var model = new Qwen2ForCausalLM(
            d.VocabularySize,
            d.ContextLength,
            d.EmbeddingLength,
            d.HeadCount,
            d.KvHeadCount,
            d.FeedForwardLength,
            d.LayerCount,
            d.RmsEpsilon,
            d.RopeTheta,
            dtype: dtype);
        LoadWeights(path, model);
        return model;
    }

    public static void LoadWeights(string path, Qwen2ForCausalLM model)
    {
        ArgumentNullException.ThrowIfNull(model);
        using var gguf = new GgufReader(path);

        Copy(gguf, model.TokenEmbedding, "token_embd.weight");
        for (int i = 0; i < model.Blocks.Count; ++i)
        {
            QwenBlock block = model.Blocks[i];
            string p = $"blk.{i}.";
            Copy(gguf, block.InputNorm.Weight, p + "attn_norm.weight");
            Copy(gguf, block.Attention.QProj.W, p + "attn_q.weight");
            CopyIfPresent(gguf, block.Attention.QProj.B, p + "attn_q.bias");
            Copy(gguf, block.Attention.KProj.W, p + "attn_k.weight");
            CopyIfPresent(gguf, block.Attention.KProj.B, p + "attn_k.bias");
            Copy(gguf, block.Attention.VProj.W, p + "attn_v.weight");
            CopyIfPresent(gguf, block.Attention.VProj.B, p + "attn_v.bias");
            Copy(gguf, block.Attention.OProj.W, p + "attn_output.weight");
            Zero(block.Attention.OProj.B);

            Copy(gguf, block.PostAttentionNorm.Weight, p + "ffn_norm.weight");
            Copy(gguf, block.Mlp.GateProj.W, p + "ffn_gate.weight");
            Copy(gguf, block.Mlp.UpProj.W, p + "ffn_up.weight");
            Copy(gguf, block.Mlp.DownProj.W, p + "ffn_down.weight");
            Zero(block.Mlp.GateProj.B);
            Zero(block.Mlp.UpProj.B);
            Zero(block.Mlp.DownProj.B);
        }

        Copy(gguf, model.FinalNorm.Weight, "output_norm.weight");
        if (gguf.Tensors.Any(t => t.Name == "output.weight"))
            Copy(gguf, model.LmHead.W, "output.weight");
        else
            CopyValues(model.TokenEmbedding, model.LmHead.W);
        Zero(model.LmHead.B);
    }

    public static float[] ReadTensor(string path, string tensorName)
    {
        using var gguf = new GgufReader(path);
        return ReadTensor(gguf, gguf.GetTensor(tensorName));
    }

    internal static float[] ReadTensor(GgufReader gguf, GgufTensorInfo tensor)
    {
        int elements = ElementCount(tensor);
        int bytes = tensor.Type switch
        {
            F32Type => checked(elements * 4),
            F16Type or BF16Type => checked(elements * 2),
            Q4KType when elements % GgufQ4K.BlockElements == 0
                => checked(elements / GgufQ4K.BlockElements * GgufQ4K.BlockBytes),
            Q6KType when elements % GgufQ6K.BlockElements == 0
                => checked(elements / GgufQ6K.BlockElements * GgufQ6K.BlockBytes),
            Q4KType or Q6KType => throw new InvalidDataException(
                $"Tensor '{tensor.Name}' has an incomplete K-quant block."),
            _ => throw new NotSupportedException(
                $"GGML tensor type {tensor.Type} is not implemented for '{tensor.Name}'.")
        };

        byte[] payload = new byte[bytes];
        gguf.OpenTensorData(tensor).ReadExactly(payload);
        if (tensor.Type == Q4KType) return GgufQ4K.Dequantize(payload, elements);
        if (tensor.Type == Q6KType) return GgufQ6K.Dequantize(payload, elements);

        var result = new float[elements];
        if (tensor.Type == F32Type)
        {
            for (int i = 0; i < elements; ++i)
            {
                int bits = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(i * 4, 4));
                result[i] = BitConverter.Int32BitsToSingle(bits);
            }
        }
        else if (tensor.Type == F16Type)
        {
            for (int i = 0; i < elements; ++i)
            {
                ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(i * 2, 2));
                result[i] = (float)BitConverter.UInt16BitsToHalf(bits);
            }
        }
        else
        {
            for (int i = 0; i < elements; ++i)
            {
                ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(i * 2, 2));
                result[i] = BitConverter.Int32BitsToSingle(bits << 16);
            }
        }
        return result;
    }

    private static void Copy(GgufReader gguf, Parameter target, string tensorName)
    {
        GgufTensorInfo tensor = gguf.GetTensor(tensorName);
        ValidateShape(target, tensor);
        float[] values = ReadTensor(gguf, tensor);
        using Tensor.DataMutation mutation = target.BeginUpdate();
        values.AsSpan().CopyTo(mutation.Values);
    }

    private static void CopyIfPresent(
        GgufReader gguf, Parameter target, string tensorName)
    {
        GgufTensorInfo? tensor = gguf.Tensors.FirstOrDefault(t => t.Name == tensorName);
        if (tensor is null) { Zero(target); return; }
        ValidateShape(target, tensor);
        float[] values = ReadTensor(gguf, tensor);
        using Tensor.DataMutation mutation = target.BeginUpdate();
        values.AsSpan().CopyTo(mutation.Values);
    }

    private static void Zero(Parameter target)
    {
        using Tensor.DataMutation mutation = target.BeginUpdate();
        mutation.Values.Clear();
    }

    private static void CopyValues(Parameter source, Parameter target)
    {
        if (!source.T.Shape.SequenceEqual(target.T.Shape))
            throw new InvalidDataException("Tied output and token embedding shapes do not match.");
        float[] values = source.T.CaptureData(preferMaster: true);
        using Tensor.DataMutation mutation = target.BeginUpdate();
        values.AsSpan().CopyTo(mutation.Values);
    }

    private static void ValidateShape(Parameter target, GgufTensorInfo tensor)
    {
        int[] expected = target.T.Shape.Reverse().ToArray();
        ulong[] actual = tensor.Shape.ToArray();
        if (expected.Length != actual.Length
            || expected.Where((dimension, i) => (ulong)dimension != actual[i]).Any())
        {
            throw new InvalidDataException(
                $"GGUF tensor '{tensor.Name}' shape [{string.Join(",", actual)}] " +
                $"does not match NNtrain parameter '{target.Name}' shape " +
                $"[{string.Join(",", target.T.Shape)}].");
        }
    }

    private static int ElementCount(GgufTensorInfo tensor)
    {
        long count = 1;
        foreach (ulong dimension in tensor.Shape)
            count = checked(count * (long)dimension);
        if (count > int.MaxValue)
            throw new NotSupportedException(
                $"Tensor '{tensor.Name}' exceeds the managed decoder limit.");
        return (int)count;
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

    private static float OptionalFloat(GgufReader gguf, string key, float fallback)
        => gguf.Metadata.TryGetValue(key, out object? value)
            ? Convert.ToSingle(value)
            : fallback;
}

public sealed record Qwen2GgufDescriptor(
    int VocabularySize,
    int LayerCount,
    int EmbeddingLength,
    int HeadCount,
    int KvHeadCount,
    int ContextLength,
    int FeedForwardLength,
    float RmsEpsilon,
    float RopeTheta,
    IReadOnlyList<GgufTensorInfo> Tensors);
