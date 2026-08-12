using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NNtrain;

/// <summary>
/// A reversible byte-level byte-pair-encoding tokenizer.
/// </summary>
public sealed class BpeTokenizer
{
    public const string PadToken = "<pad>";
    public const string BosToken = "<bos>";
    public const string EosToken = "<eos>";
    public const string UnknownToken = "<unk>";

    public const int PadTokenId = 0;
    public const int BosTokenId = 1;
    public const int EosTokenId = 2;
    public const int UnknownTokenId = 3;
    public const int ByteTokenOffset = 4;
    public const int BaseVocabularySize = ByteTokenOffset + 256;

    private const int CurrentFormatVersion = 1;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
    };

    private readonly MergeRule[] _merges;
    private readonly Dictionary<TokenPair, MergeRank> _mergeRanks;
    private readonly byte[][] _tokenBytes;

    private BpeTokenizer(IEnumerable<MergeRule> merges)
    {
        _merges = merges.ToArray();
        _mergeRanks = new Dictionary<TokenPair, MergeRank>(_merges.Length);
        _tokenBytes = BuildTokenBytes(_merges);

        for (int rank = 0; rank < _merges.Length; rank++)
        {
            MergeRule merge = _merges[rank];
            var pair = new TokenPair(merge.Left, merge.Right);
            if (!_mergeRanks.TryAdd(pair, new MergeRank(rank, merge.Id)))
            {
                throw new InvalidDataException(
                    $"BPE merge pair ({merge.Left}, {merge.Right}) is " +
                    "defined more than once.");
            }
        }
    }

    public int VocabularySize => BaseVocabularySize + _merges.Length;

    public int vocab_size => VocabularySize;

    public static BpeTokenizer Train(
        IEnumerable<string> documents,
        int vocabularySize,
        int maxTrainingBytes = int.MaxValue)
    {
        ArgumentNullException.ThrowIfNull(documents);
        if (vocabularySize < BaseVocabularySize)
        {
            throw new ArgumentOutOfRangeException(
                nameof(vocabularySize),
                vocabularySize,
                $"Vocabulary size must be at least {BaseVocabularySize}.");
        }
        if (maxTrainingBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(maxTrainingBytes),
                maxTrainingBytes,
                "Maximum training bytes must be positive.");
        }

        var symbols = new List<int>();
        var previous = new List<int>();
        var next = new List<int>();
        int retainedBytes = 0;

        foreach (string document in documents)
        {
            if (document is null)
                throw new ArgumentException(
                    "Tokenizer documents cannot contain null.",
                    nameof(documents));
            if (retainedBytes >= maxTrainingBytes)
                break;

            byte[] bytes = Encoding.UTF8.GetBytes(document);
            int take = Math.Min(bytes.Length, maxTrainingBytes - retainedBytes);
            int firstNode = symbols.Count;
            for (int index = 0; index < take; index++)
            {
                int node = symbols.Count;
                symbols.Add(ByteTokenOffset + bytes[index]);
                previous.Add(index == 0 ? -1 : node - 1);
                next.Add(index + 1 == take ? -1 : node + 1);
            }

            if (take == 0 && symbols.Count == firstNode)
                continue;
            retainedBytes += take;
        }

        if (retainedBytes == 0)
        {
            throw new ArgumentException(
                "At least one non-empty tokenizer document is required.",
                nameof(documents));
        }

        int[] symbolArray = symbols.ToArray();
        int[] previousArray = previous.ToArray();
        int[] nextArray = next.ToArray();
        bool[] active = Enumerable.Repeat(true, symbolArray.Length).ToArray();
        var occurrences = new Dictionary<TokenPair, HashSet<int>>();

        for (int node = 0; node < symbolArray.Length; node++)
        {
            int right = nextArray[node];
            if (right >= 0)
            {
                AddOccurrence(
                    new TokenPair(symbolArray[node], symbolArray[right]),
                    node,
                    occurrences,
                    dirty: null);
            }
        }

        var queue = new PriorityQueue<TokenPair, (int, int, int)>();
        foreach ((TokenPair pair, HashSet<int> starts) in occurrences)
            EnqueuePair(queue, pair, starts.Count);

        var merges = new List<MergeRule>(
            vocabularySize - BaseVocabularySize);
        while (BaseVocabularySize + merges.Count < vocabularySize)
        {
            if (!TryTakeMostFrequentPair(queue, occurrences, out TokenPair pair))
                break;

            int resultId = BaseVocabularySize + merges.Count;
            merges.Add(new MergeRule(pair.Left, pair.Right, resultId));
            int[] candidates = occurrences[pair].ToArray();
            var dirty = new HashSet<TokenPair>();

            foreach (int leftNode in candidates)
            {
                int rightNode = nextArray[leftNode];
                if (!active[leftNode]
                    || rightNode < 0
                    || !active[rightNode]
                    || symbolArray[leftNode] != pair.Left
                    || symbolArray[rightNode] != pair.Right)
                {
                    RemoveOccurrence(pair, leftNode, occurrences, dirty);
                    continue;
                }

                int outerLeft = previousArray[leftNode];
                int outerRight = nextArray[rightNode];
                if (outerLeft >= 0)
                {
                    RemoveOccurrence(
                        new TokenPair(
                            symbolArray[outerLeft],
                            symbolArray[leftNode]),
                        outerLeft,
                        occurrences,
                        dirty);
                }
                RemoveOccurrence(pair, leftNode, occurrences, dirty);
                if (outerRight >= 0)
                {
                    RemoveOccurrence(
                        new TokenPair(
                            symbolArray[rightNode],
                            symbolArray[outerRight]),
                        rightNode,
                        occurrences,
                        dirty);
                }

                symbolArray[leftNode] = resultId;
                active[rightNode] = false;
                nextArray[leftNode] = outerRight;
                if (outerRight >= 0)
                    previousArray[outerRight] = leftNode;

                if (outerLeft >= 0)
                {
                    AddOccurrence(
                        new TokenPair(
                            symbolArray[outerLeft],
                            symbolArray[leftNode]),
                        outerLeft,
                        occurrences,
                        dirty);
                }
                if (outerRight >= 0)
                {
                    AddOccurrence(
                        new TokenPair(
                            symbolArray[leftNode],
                            symbolArray[outerRight]),
                        leftNode,
                        occurrences,
                        dirty);
                }
            }

            foreach (TokenPair changedPair in dirty)
            {
                if (occurrences.TryGetValue(changedPair, out HashSet<int>? starts))
                    EnqueuePair(queue, changedPair, starts.Count);
            }
        }

        return new BpeTokenizer(merges);
    }

    public int[] Encode(
        string text,
        bool addBos = false,
        bool addEos = false)
    {
        ArgumentNullException.ThrowIfNull(text);
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        if (bytes.Length == 0)
        {
            var emptyResult = new List<int>(2);
            if (addBos)
                emptyResult.Add(BosTokenId);
            if (addEos)
                emptyResult.Add(EosTokenId);
            return emptyResult.ToArray();
        }

        int length = bytes.Length;
        var symbols = new int[length];
        var previous = new int[length];
        var next = new int[length];
        var active = new bool[length];
        for (int index = 0; index < length; index++)
        {
            symbols[index] = ByteTokenOffset + bytes[index];
            previous[index] = index - 1;
            next[index] = index + 1 == length ? -1 : index + 1;
            active[index] = true;
        }

        var queue = new PriorityQueue<EncodeCandidate, int>();
        for (int index = 0; index + 1 < length; index++)
            EnqueueEncodeCandidate(index, symbols, next, queue);

        while (queue.TryDequeue(out EncodeCandidate candidate, out int rank))
        {
            int left = candidate.LeftNode;
            int right = next[left];
            if (!active[left]
                || right < 0
                || !active[right]
                || symbols[left] != candidate.LeftToken
                || symbols[right] != candidate.RightToken
                || !_mergeRanks.TryGetValue(
                    new TokenPair(symbols[left], symbols[right]),
                    out MergeRank currentMerge)
                || currentMerge.Rank != rank)
            {
                continue;
            }

            symbols[left] = currentMerge.ResultId;
            active[right] = false;
            int outerRight = next[right];
            next[left] = outerRight;
            if (outerRight >= 0)
                previous[outerRight] = left;

            int outerLeft = previous[left];
            if (outerLeft >= 0)
                EnqueueEncodeCandidate(outerLeft, symbols, next, queue);
            EnqueueEncodeCandidate(left, symbols, next, queue);
        }

        var result = new List<int>(length + 2);
        if (addBos)
            result.Add(BosTokenId);
        for (int node = 0; node >= 0; node = next[node])
            result.Add(symbols[node]);
        if (addEos)
            result.Add(EosTokenId);
        return result.ToArray();
    }

    public int[] encode(
        string text,
        bool add_bos = false,
        bool add_eos = false)
        => Encode(text, add_bos, add_eos);

    public string Decode(IEnumerable<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        using var bytes = new MemoryStream();
        foreach (int tokenId in tokenIds)
        {
            if (tokenId is PadTokenId or BosTokenId or EosTokenId)
                continue;
            if (tokenId == UnknownTokenId)
            {
                bytes.Write(Encoding.UTF8.GetBytes("\uFFFD"));
                continue;
            }
            if ((uint)tokenId >= (uint)_tokenBytes.Length)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(tokenIds),
                    tokenId,
                    $"Token id must be between 0 and {VocabularySize - 1}.");
            }

            bytes.Write(_tokenBytes[tokenId]);
        }

        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    public string decode(IEnumerable<int> token_ids) => Decode(token_ids);

    public void Save(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        var file = new TokenizerFile(CurrentFormatVersion, _merges);
        string temporaryPath = fullPath + ".tmp";
        File.WriteAllText(
            temporaryPath,
            JsonSerializer.Serialize(file, JsonOptions));
        File.Move(temporaryPath, fullPath, overwrite: true);
    }

    public void save(string path) => Save(path);

    public static BpeTokenizer Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        TokenizerFile file = JsonSerializer.Deserialize<TokenizerFile>(
            File.ReadAllText(Path.GetFullPath(path)),
            JsonOptions)
            ?? throw new InvalidDataException("BPE tokenizer JSON cannot be null.");
        if (file.FormatVersion != CurrentFormatVersion)
        {
            throw new InvalidDataException(
                $"Unsupported BPE tokenizer format version " +
                $"'{file.FormatVersion}'. Expected '{CurrentFormatVersion}'.");
        }
        if (file.Merges is null)
            throw new InvalidDataException("BPE tokenizer merges cannot be null.");

        ValidateMerges(file.Merges);
        return new BpeTokenizer(file.Merges);
    }

    private void EnqueueEncodeCandidate(
        int leftNode,
        IReadOnlyList<int> symbols,
        IReadOnlyList<int> next,
        PriorityQueue<EncodeCandidate, int> queue)
    {
        int rightNode = next[leftNode];
        if (rightNode < 0)
            return;

        int leftToken = symbols[leftNode];
        int rightToken = symbols[rightNode];
        if (_mergeRanks.TryGetValue(
            new TokenPair(leftToken, rightToken),
            out MergeRank merge))
        {
            queue.Enqueue(
                new EncodeCandidate(leftNode, leftToken, rightToken),
                merge.Rank);
        }
    }

    private static byte[][] BuildTokenBytes(IReadOnlyList<MergeRule> merges)
    {
        var result = new byte[BaseVocabularySize + merges.Count][];
        result[PadTokenId] = [];
        result[BosTokenId] = [];
        result[EosTokenId] = [];
        result[UnknownTokenId] = Encoding.UTF8.GetBytes("\uFFFD");
        for (int value = 0; value < 256; value++)
            result[ByteTokenOffset + value] = [(byte)value];

        for (int index = 0; index < merges.Count; index++)
        {
            MergeRule merge = merges[index];
            byte[] left = result[merge.Left];
            byte[] right = result[merge.Right];
            byte[] combined = new byte[left.Length + right.Length];
            left.CopyTo(combined, 0);
            right.CopyTo(combined, left.Length);
            result[merge.Id] = combined;
        }

        return result;
    }

    private static void ValidateMerges(IReadOnlyList<MergeRule> merges)
    {
        for (int index = 0; index < merges.Count; index++)
        {
            MergeRule merge = merges[index]
                ?? throw new InvalidDataException(
                    $"BPE merge at index {index} cannot be null.");
            int expectedId = BaseVocabularySize + index;
            if (merge.Id != expectedId
                || merge.Left < ByteTokenOffset
                || merge.Left >= expectedId
                || merge.Right < ByteTokenOffset
                || merge.Right >= expectedId)
            {
                throw new InvalidDataException(
                    $"BPE merge at index {index} is invalid.");
            }
        }
    }

    private static bool TryTakeMostFrequentPair(
        PriorityQueue<TokenPair, (int, int, int)> queue,
        IReadOnlyDictionary<TokenPair, HashSet<int>> occurrences,
        out TokenPair result)
    {
        while (queue.TryDequeue(out TokenPair pair, out (int, int, int) priority))
        {
            if (occurrences.TryGetValue(pair, out HashSet<int>? starts)
                && starts.Count >= 2
                && -priority.Item1 == starts.Count)
            {
                result = pair;
                return true;
            }
        }

        result = default;
        return false;
    }

    private static void EnqueuePair(
        PriorityQueue<TokenPair, (int, int, int)> queue,
        TokenPair pair,
        int count)
    {
        if (count >= 2)
            queue.Enqueue(pair, (-count, pair.Left, pair.Right));
    }

    private static void AddOccurrence(
        TokenPair pair,
        int start,
        Dictionary<TokenPair, HashSet<int>> occurrences,
        HashSet<TokenPair>? dirty)
    {
        if (!occurrences.TryGetValue(pair, out HashSet<int>? starts))
        {
            starts = [];
            occurrences.Add(pair, starts);
        }
        if (starts.Add(start))
            dirty?.Add(pair);
    }

    private static void RemoveOccurrence(
        TokenPair pair,
        int start,
        Dictionary<TokenPair, HashSet<int>> occurrences,
        HashSet<TokenPair> dirty)
    {
        if (occurrences.TryGetValue(pair, out HashSet<int>? starts)
            && starts.Remove(start))
        {
            dirty.Add(pair);
        }
    }

    private sealed record TokenizerFile(
        int FormatVersion,
        MergeRule[] Merges);

    private sealed record MergeRule(int Left, int Right, int Id);

    private readonly record struct TokenPair(int Left, int Right);

    private readonly record struct MergeRank(int Rank, int ResultId);

    private readonly record struct EncodeCandidate(
        int LeftNode,
        int LeftToken,
        int RightToken);
}
