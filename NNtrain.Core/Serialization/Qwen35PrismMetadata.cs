using System.Collections.Frozen;
using System.Globalization;

namespace NNtrain;

/// <summary>
/// Validates the explicit Prism transform used by the supported Bonsai GGUFs.
/// Forward weights need the activation transform; inverse weights need the
/// inverse transform after embedding lookup. A missing declaration is an error.
/// </summary>
internal sealed class Qwen35PrismMetadata
{
    internal const int BlockSize = 1024;
    private const string Prefix = "prism.hadamard.";
    private static readonly FrozenSet<string> KnownKeys = new[]
    {
        "version", "block_size", "transform", "axis", "sign_mode", "weight_names",
        "inverse_weight_names", "sign_widths", "sign_values", "gdn_v_grouped", "tied_output"
    }.Select(suffix => Prefix + suffix).ToFrozenSet(StringComparer.Ordinal);
    private static readonly FrozenSet<string> ForwardKinds = new[]
    {
        "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "attn_output",
        "ffn_gate", "ffn_up", "ffn_down", "ssm_out"
    }.ToFrozenSet(StringComparer.Ordinal);

    internal IReadOnlySet<string> ForwardWeights { get; }
    internal IReadOnlySet<string> InverseWeights { get; }
    internal IReadOnlyDictionary<int, float[]> Signs { get; }
    internal bool GroupedValueHeads { get; }

    private Qwen35PrismMetadata(HashSet<string> forward, HashSet<string> inverse,
        Dictionary<int, float[]> signs, bool groupedValueHeads)
    {
        ForwardWeights = forward.ToFrozenSet(StringComparer.Ordinal);
        InverseWeights = inverse.ToFrozenSet(StringComparer.Ordinal);
        Signs = signs.ToFrozenDictionary();
        GroupedValueHeads = groupedValueHeads;
    }

    internal static Qwen35PrismMetadata? Read(GgufReader gguf)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        string[] prismKeys = gguf.Metadata.Keys.Where(key => key.StartsWith(Prefix, StringComparison.Ordinal)).ToArray();
        bool hasPrismStorage = gguf.Tensors.Any(tensor => tensor.Type is 142 or 143);
        if (prismKeys.Length == 0)
        {
            if (hasPrismStorage)
                throw new InvalidDataException("PQ2_0/PTQ1_0 matrices require prism.hadamard metadata.");
            return null;
        }
        foreach (string key in prismKeys)
            if (!KnownKeys.Contains(key))
                throw new NotSupportedException($"Unsupported Prism metadata key '{key}'.");
        if (!gguf.Metadata.TryGetValue("general.architecture", out object? architecture) || architecture is not "qwen35")
            throw new NotSupportedException("Prism transforms are supported only for dense qwen35 models.");

        int version = Integer(Required("version"), Prefix + "version");
        if (version != 1)
            throw new NotSupportedException($"Unsupported prism.hadamard.version {version}; only version 1 is supported.");
        int blockSize = Integer(Required("block_size"), Prefix + "block_size");
        if (blockSize != BlockSize)
            throw new NotSupportedException($"Unsupported prism.hadamard.block_size {blockSize}; expected {BlockSize}.");
        RequireString("transform", "normalized-sylvester-walsh-hadamard");
        RequireString("axis", "input-last-dimension");
        RequireString("sign_mode", "explicit");
        if (OptionalBoolean("tied_output"))
            throw new NotSupportedException("prism.hadamard.tied_output is not supported by version 1.");
        bool grouped = OptionalBoolean("gdn_v_grouped");

        var tensors = new Dictionary<string, GgufTensorInfo>(StringComparer.Ordinal);
        foreach (GgufTensorInfo tensor in gguf.Tensors)
            if (!tensors.TryAdd(tensor.Name, tensor))
                throw new InvalidDataException($"Duplicate GGUF tensor '{tensor.Name}'.");

        HashSet<string> forward = Names("weight_names", required: true);
        HashSet<string> inverse = Names("inverse_weight_names", required: false);
        if (forward.Count == 0)
            throw new InvalidDataException("prism.hadamard.weight_names must not be empty.");
        foreach (string name in forward)
            if (!IsForwardWeight(name))
                throw new NotSupportedException($"Prism forward weight '{name}' is not on a supported transformed matmul path.");
        foreach (string name in inverse)
        {
            if (name != "token_embd.weight")
                throw new NotSupportedException($"Prism inverse weight '{name}' is not a supported embedding lookup.");
            if (forward.Contains(name))
                throw new InvalidDataException($"Prism weight '{name}' is declared for both forward and inverse transforms.");
        }
        if (inverse.Contains("token_embd.weight") && !tensors.ContainsKey("output.weight"))
            throw new NotSupportedException("A Prism inverse embedding requires a separate output.weight in version 1.");

        var widthsUsed = new HashSet<int>();
        foreach (string name in forward.Concat(inverse))
        {
            if (!tensors.TryGetValue(name, out GgufTensorInfo? tensor))
                throw new InvalidDataException($"Prism weight '{name}' does not exist in the GGUF tensor directory.");
            if (tensor.Shape.Count != 2 || tensor.Shape[0] is 0 or > int.MaxValue
                || tensor.Shape[1] is 0 or > int.MaxValue)
                throw new InvalidDataException($"Prism weight '{name}' must have a positive two-dimensional matrix shape.");
            int width = (int)tensor.Shape[0];
            if (width % BlockSize != 0)
                throw new InvalidDataException($"Prism weight '{name}' input width {width} must be divisible by {BlockSize}.");
            widthsUsed.Add(width);
        }
        // These two storage formats are accepted only together with explicit
        // transform declarations. Otherwise loading could produce plausible
        // values in the wrong basis without reporting a decoding error.
        foreach (GgufTensorInfo tensor in gguf.Tensors)
            if (tensor.Type is 142 or 143 && !forward.Contains(tensor.Name) && !inverse.Contains(tensor.Name))
                throw new InvalidDataException($"Prism matrix '{tensor.Name}' is missing its transform declaration.");

        object[] widths = Array("sign_widths", required: true);
        object[] values = Array("sign_values", required: true);
        if (widths.Length == 0)
            throw new InvalidDataException("prism.hadamard.sign_widths must not be empty in explicit mode.");
        int[] parsedWidths = widths.Select(value => Integer(value, Prefix + "sign_widths")).ToArray();
        var uniqueWidths = new HashSet<int>();
        long total = 0;
        foreach (int width in parsedWidths)
        {
            if (width <= 0 || width % BlockSize != 0 || !uniqueWidths.Add(width))
                throw new InvalidDataException($"Invalid or duplicate prism.hadamard sign width {width}.");
            total += width;
        }
        if (total != values.Length)
            throw new InvalidDataException("prism.hadamard.sign_values length does not match the sum of sign_widths.");
        foreach (int width in widthsUsed)
            if (!uniqueWidths.Contains(width))
                throw new InvalidDataException($"Prism sign vector for transformed input width {width} is missing.");
        // Fail closed on metadata outside the supported transform inventory.
        // The two supported Bonsai files use every declared sign vector.
        foreach (int width in uniqueWidths)
            if (!widthsUsed.Contains(width))
                throw new NotSupportedException($"Prism sign width {width} is unused by the declared transformed matrices.");

        var signs = new Dictionary<int, float[]>();
        int offset = 0;
        foreach (int width in parsedWidths)
        {
            float[] vector = new float[width];
            for (int i = 0; i < width; i++)
            {
                int sign = Integer(values[offset++], Prefix + "sign_values");
                if (sign is not (-1 or 1))
                    throw new InvalidDataException("prism.hadamard.sign_values must contain only -1 or +1.");
                vector[i] = sign;
            }
            signs.Add(width, vector);
        }
        return new(forward, inverse, signs, grouped);

        object Required(string suffix)
            => gguf.Metadata.TryGetValue(Prefix + suffix, out object? value) ? value
                : throw new InvalidDataException($"Missing GGUF metadata '{Prefix + suffix}'.");

        void RequireString(string suffix, string expected)
        {
            object value = Required(suffix);
            if (value is not string text)
                throw new InvalidDataException($"GGUF metadata '{Prefix + suffix}' must be a string.");
            if (text != expected)
                throw new NotSupportedException($"Unsupported '{Prefix + suffix}' value '{text}'; expected '{expected}'.");
        }

        bool OptionalBoolean(string suffix)
        {
            if (!gguf.Metadata.TryGetValue(Prefix + suffix, out object? value)) return false;
            return value is bool flag ? flag
                : throw new InvalidDataException($"GGUF metadata '{Prefix + suffix}' must be a boolean.");
        }

        object[] Array(string suffix, bool required)
        {
            if (!gguf.Metadata.TryGetValue(Prefix + suffix, out object? value))
                return required ? throw new InvalidDataException($"Missing GGUF metadata '{Prefix + suffix}'.") : [];
            return value as object[] ?? throw new InvalidDataException($"GGUF metadata '{Prefix + suffix}' must be an array.");
        }

        HashSet<string> Names(string suffix, bool required)
        {
            var result = new HashSet<string>(StringComparer.Ordinal);
            foreach (object value in Array(suffix, required))
            {
                if (value is not string name || string.IsNullOrWhiteSpace(name))
                    throw new InvalidDataException($"GGUF metadata '{Prefix + suffix}' must contain tensor names.");
                if (!result.Add(name))
                    throw new InvalidDataException($"Duplicate Prism tensor name '{name}' in '{Prefix + suffix}'.");
            }
            return result;
        }
    }

    private static bool IsForwardWeight(string name)
    {
        if (name == "output.weight") return true;
        string[] parts = name.Split('.');
        return parts.Length == 4 && parts[0] == "blk" && parts[3] == "weight"
            && int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out int layer)
            && layer >= 0 && ForwardKinds.Contains(parts[2]);
    }

    private static int Integer(object value, string key)
    {
        long number = value switch
        {
            byte v => v, sbyte v => v, ushort v => v, short v => v,
            uint v => v, int v => v, long v => v,
            ulong v when v <= int.MaxValue => (long)v,
            _ => throw new InvalidDataException($"GGUF metadata '{key}' must contain supported integers.")
        };
        return number is >= int.MinValue and <= int.MaxValue ? (int)number
            : throw new InvalidDataException($"GGUF metadata '{key}' integer is outside the supported range.");
    }
}
