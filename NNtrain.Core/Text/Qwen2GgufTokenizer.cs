using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace NNtrain;

/// <summary>
/// Qwen2 byte-level BPE reconstructed directly from GGUF tokenizer metadata.
/// Token ids are exactly the GGUF vocabulary indices.
/// </summary>
public sealed class Qwen2GgufTokenizer
{
    // Equivalent to Qwen2's Unicode-property split, expressed without a
    // dependency on a third-party regex engine.
    private static readonly Regex SplitRegex = new(
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^
p{L}p{N}]?p{L}+|p{N}| ?[^sp{L}p{N}]+[
]*|s*[
]+|s+(?!S)|s+",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    private readonly string[] _tokens;
    private readonly Dictionary<string, int> _vocabulary;
    private readonly Dictionary<(string Left, string Right), int> _mergeRanks;
    private readonly Dictionary<char, byte> _byteDecoder;
    private readonly HashSet<string> _specialTokens;

    private Qwen2GgufTokenizer(
        string[] tokens,
        string[] merges,
        int? bosTokenId,
        int? eosTokenId,
        int? padTokenId,
        int? unknownTokenId,
        IReadOnlyList<int>? tokenTypes)
    {
        _tokens = tokens;
        _vocabulary = tokens
            .Select((token, id) => (token, id))
            .ToDictionary(x => x.token, x => x.id, StringComparer.Ordinal);
        _mergeRanks = new Dictionary<(string, string), int>(merges.Length);
        for (int rank = 0; rank < merges.Length; ++rank)
        {
            int separator = merges[rank].IndexOf(' ');
            if (separator <= 0 || separator == merges[rank].Length - 1)
                throw new InvalidDataException($"Invalid GGUF BPE merge '{merges[rank]}'.");
            _mergeRanks.TryAdd(
                (merges[rank][..separator], merges[rank][(separator + 1)..]),
                rank);
        }
        _byteDecoder = BuildByteEncoder()
            .ToDictionary(pair => pair.Value, pair => pair.Key);
        _specialTokens = new HashSet<string>(StringComparer.Ordinal);
        if (tokenTypes is not null)
        {
            for (int i = 0; i < Math.Min(tokenTypes.Count, tokens.Length); ++i)
                if (tokenTypes[i] == 3) _specialTokens.Add(tokens[i]);
        }

        BosTokenId = bosTokenId;
        EosTokenId = eosTokenId;
        PadTokenId = padTokenId;
        UnknownTokenId = unknownTokenId;
    }

    public int VocabularySize => _tokens.Length;
    public int? BosTokenId { get; }
    public int? EosTokenId { get; }
    public int? PadTokenId { get; }
    public int? UnknownTokenId { get; }

    public static Qwen2GgufTokenizer Load(string path)
    {
        using var gguf = new GgufReader(path);
        string model = StringValue(gguf, "tokenizer.ggml.model");
        if (!string.Equals(model, "gpt2", StringComparison.Ordinal))
            throw new NotSupportedException($"Qwen GGUF tokenizer model '{model}' is not GPT-2 BPE.");

        string[] tokens = StringArray(gguf, "tokenizer.ggml.tokens");
        string[] merges = StringArray(gguf, "tokenizer.ggml.merges");
        int[]? tokenTypes = OptionalIntArray(gguf, "tokenizer.ggml.token_type");
        return new Qwen2GgufTokenizer(
            tokens, merges,
            OptionalInt(gguf, "tokenizer.ggml.bos_token_id"),
            OptionalInt(gguf, "tokenizer.ggml.eos_token_id"),
            OptionalInt(gguf, "tokenizer.ggml.padding_token_id"),
            OptionalInt(gguf, "tokenizer.ggml.unknown_token_id"),
            tokenTypes);
    }

    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var ids = new List<int>();
        foreach (string piece in SplitSpecial(text))
        {
            if (_specialTokens.Contains(piece) && _vocabulary.TryGetValue(piece, out int special))
            {
                ids.Add(special);
                continue;
            }
            foreach (Match match in SplitRegex.Matches(piece))
                EncodePiece(match.Value, ids);
        }
        return ids.ToArray();
    }

    public string Decode(IEnumerable<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        var text = new StringBuilder();
        using var bytes = new MemoryStream();

        void FlushBytes()
        {
            if (bytes.Length == 0) return;
            text.Append(Encoding.UTF8.GetString(bytes.ToArray()));
            bytes.SetLength(0);
        }

        foreach (int id in tokenIds)
        {
            if ((uint)id >= (uint)_tokens.Length)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), id, "Qwen token id is outside the vocabulary.");
            string token = _tokens[id];
            if (_specialTokens.Contains(token))
            {
                FlushBytes();
                text.Append(token);
                continue;
            }
            foreach (char ch in token)
            {
                if (!_byteDecoder.TryGetValue(ch, out byte value))
                    throw new InvalidDataException($"Token {id} contains byte-alphabet character U+{(int)ch:X4} not present in GPT-2 mapping.");
                bytes.WriteByte(value);
            }
        }
        FlushBytes();
        return text.ToString();
    }

    private void EncodePiece(string piece, List<int> destination)
    {
        if (piece.Length == 0) return;
        Dictionary<byte, char> byteEncoder = BuildByteEncoder();
        string encoded = string.Concat(Encoding.UTF8.GetBytes(piece).Select(b => byteEncoder[b]));
        var symbols = encoded.Select(ch => ch.ToString()).ToList();

        while (symbols.Count > 1)
        {
            int bestRank = int.MaxValue, bestIndex = -1;
            for (int i = 0; i + 1 < symbols.Count; ++i)
            {
                if (_mergeRanks.TryGetValue((symbols[i], symbols[i + 1]), out int rank)
                    && rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }
            if (bestIndex < 0) break;
            string left = symbols[bestIndex], right = symbols[bestIndex + 1];
            for (int i = 0; i + 1 < symbols.Count;)
            {
                if (symbols[i] == left && symbols[i + 1] == right)
                {
                    symbols[i] += symbols[i + 1];
                    symbols.RemoveAt(i + 1);
                }
                else ++i;
            }
        }

        foreach (string symbol in symbols)
        {
            if (!_vocabulary.TryGetValue(symbol, out int id))
                throw new InvalidDataException($"Qwen BPE symbol '{symbol}' is missing from the GGUF vocabulary.");
            destination.Add(id);
        }
    }

    private IEnumerable<string> SplitSpecial(string text)
    {
        if (_specialTokens.Count == 0) { yield return text; yield break; }
        int cursor = 0;
        while (cursor < text.Length)
        {
            string? found = null;
            int first = int.MaxValue;
            foreach (string token in _specialTokens)
            {
                int index = text.IndexOf(token, cursor, StringComparison.Ordinal);
                if (index >= 0 && index < first) { first = index; found = token; }
            }
            if (found is null) { yield return text[cursor..]; yield break; }
            if (first > cursor) yield return text[cursor..first];
            yield return found;
            cursor = first + found.Length;
        }
    }

    private static Dictionary<byte, char> BuildByteEncoder()
    {
        var bytes = new List<int>();
        for (int i = 33; i <= 126; ++i) bytes.Add(i);
        for (int i = 161; i <= 172; ++i) bytes.Add(i);
        for (int i = 174; i <= 255; ++i) bytes.Add(i);
        var chars = new List<int>(bytes);
        int extra = 0;
        for (int b = 0; b < 256; ++b)
        {
            if (bytes.Contains(b)) continue;
            bytes.Add(b);
            chars.Add(256 + extra++);
        }
        var result = new Dictionary<byte, char>(256);
        for (int i = 0; i < bytes.Count; ++i)
            result[(byte)bytes[i]] = (char)chars[i];
        return result;
    }

    private static string StringValue(GgufReader gguf, string key)
        => gguf.Metadata.TryGetValue(key, out object? value) && value is string text
            ? text : throw new InvalidDataException($"Missing GGUF tokenizer metadata '{key}'.");

    private static string[] StringArray(GgufReader gguf, string key)
        => gguf.Metadata.TryGetValue(key, out object? value) && value is object[] values
            ? values.Select(v => v as string
                ?? throw new InvalidDataException($"GGUF '{key}' contains a non-string value.")).ToArray()
            : throw new InvalidDataException($"Missing GGUF tokenizer array '{key}'.");

    private static int? OptionalInt(GgufReader gguf, string key)
        => gguf.Metadata.TryGetValue(key, out object? value)
            ? checked((int)Convert.ToUInt64(value, CultureInfo.InvariantCulture)) : null;

    private static int[]? OptionalIntArray(GgufReader gguf, string key)
        => gguf.Metadata.TryGetValue(key, out object? value) && value is object[] values
            ? values.Select(v => checked((int)Convert.ToInt64(v, CultureInfo.InvariantCulture))).ToArray()
            : null;
}
