using NNtrain.Arc;
using System.Runtime.InteropServices;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>Temporal, row and column coordinates for one Qwen3.5 prompt position.</summary>
public readonly record struct Qwen35Position(int Temporal, int Height, int Width)
{
    public static Qwen35Position Scalar(int position) => new(position, position, position);
}

/// <summary>
/// One text or projected image position. For an image, Embedding replaces the
/// token embedding and TokenId may be -1. The embedding must already have the
/// text model's hidden width.
/// </summary>
public sealed record Qwen35PromptToken(int TokenId, float[]? Embedding, Qwen35Position Position);

public sealed partial class Qwen35QuantizedModel
{
    private Qwen35PromptToken[]? _cachedMixedPromptTokens;

    /// <summary>
    /// Explicitly prepares an exact mixed prefix without generating an answer.
    /// Image rows are copied for immutable, bit-exact cache identity. A later
    /// question may reuse this state only if every prefix entry still matches.
    /// </summary>
    public (int ReusedTokens, bool Cached) PrimePromptWithEmbeddings(
        IReadOnlyList<Qwen35PromptToken> prompt, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ValidateMixedPrompt(prompt);
        if (prompt.Count >= Descriptor.ContextLength)
            throw new ArgumentException("Prefix must leave room for a following question.", nameof(prompt));
        cancellationToken.ThrowIfCancellationRequested();
        Qwen35PromptToken[] owned = prompt.Select(entry => new Qwen35PromptToken(entry.TokenId,
            entry.Embedding?.ToArray(), entry.Position)).ToArray();
        if (CanReuseMixedPromptPrefix(owned, allowEqual: true))
        {
            int length = _cachedMixedPromptTokens!.Length;
            if (length == owned.Length)
            {
                _lastReusedPromptTokens = length;
                return (length, true);
            }
        }
        int reused = CanReuseMixedPromptPrefix(owned) ? _cachedMixedPromptTokens!.Length : 0;
        if (reused == 0) Reset();
        _lastReusedPromptTokens = reused;
        try
        {
            foreach (LayerState state in _states) state.EnsureCapacity(owned.Length);
            int i = reused;
            for (; CanPrefillChunk(owned.Length - i);
                i += Math.Min(_options.InferencePrefillChunkTokens, owned.Length - i))
            {
                cancellationToken.ThrowIfCancellationRequested();
                ForwardPromptChunkDevice(owned, i, Math.Min(_options.InferencePrefillChunkTokens, owned.Length - i),
                    cancellationToken);
            }
            for (; i < owned.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Qwen35PromptToken entry = owned[i];
                _ = ForwardTokenDevice(entry.TokenId, false, entry.Embedding, entry.Position);
            }
            cancellationToken.ThrowIfCancellationRequested();
            if (!TryCapturePromptCheckpoint()) return (reused, false);
            _cachedMixedPromptTokens = owned;
            return (reused, true);
        }
        catch { _faulted = true; InvalidatePromptCheckpoint(); throw; }
    }

    internal static bool MixedPromptPrefixMatches(IReadOnlyList<Qwen35PromptToken> cached,
        IReadOnlyList<Qwen35PromptToken> prompt, bool allowEqual = false)
    {
        if (cached.Count == 0 || cached.Count > prompt.Count || !allowEqual && cached.Count == prompt.Count) return false;
        for (int i = 0; i < cached.Count; i++)
        {
            Qwen35PromptToken left = cached[i], right = prompt[i];
            if (left.TokenId != right.TokenId || left.Position != right.Position
                || (left.Embedding is null) != (right.Embedding is null)) return false;
            if (left.Embedding is { } embedding && !MemoryMarshal.AsBytes(embedding.AsSpan())
                .SequenceEqual(MemoryMarshal.AsBytes(right.Embedding!.AsSpan()))) return false;
        }
        return true;
    }

    private bool CanReuseMixedPromptPrefix(IReadOnlyList<Qwen35PromptToken> prompt, bool allowEqual = false)
        => !_faulted && _cachedMixedPromptTokens is { } cached && _position == cached.Length
            && _states.All(state => state.HasPromptState) && MixedPromptPrefixMatches(cached, prompt, allowEqual);

    private void ValidateMixedPrompt(IReadOnlyList<Qwen35PromptToken> prompt)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        if (prompt.Count == 0 || prompt.Count > Descriptor.ContextLength)
            throw new ArgumentException("Prompt must contain 1..context-length entries.", nameof(prompt));
        bool hasSpatialPositions = false;
        foreach (Qwen35PromptToken entry in prompt)
        {
            if (entry is null || entry.TokenId < -1 || entry.TokenId >= Descriptor.VocabularySize
                || entry.Embedding is null && entry.TokenId < 0
                || entry.Embedding is { } embedding && (embedding.Length != Descriptor.EmbeddingLength
                    || embedding.Any(value => !float.IsFinite(value)))
                || entry.Position.Temporal < 0 || entry.Position.Height < 0 || entry.Position.Width < 0)
                throw new ArgumentException("Prompt contains an invalid token, image embedding or position.", nameof(prompt));
            hasSpatialPositions |= entry.Position.Temporal != entry.Position.Height || entry.Position.Temporal != entry.Position.Width;
        }
        if (hasSpatialPositions && Descriptor.RopeDimensionCount > 0 && Descriptor.RopeDimensionSections.Count < 3)
            throw new NotSupportedException("Qwen3.5 multimodal RoPE sections are missing from the GGUF.");
    }

    /// <summary>
    /// Generates from a mixed text/image prompt. The returned token IDs include
    /// the prompt entries (image entries may be -1) followed by generated IDs.
    /// Only an explicitly primed mixed prefix may be reused. Token IDs, MRoPE
    /// positions and all image embedding bits must match; other prompts reset.
    /// </summary>
    public int[] GenerateTokenIdsWithEmbeddings(IReadOnlyList<Qwen35PromptToken> prompt,
        int maxNewTokens, int nextPosition, CancellationToken cancellationToken,
        int? eosTokenId = null, Action<int>? onToken = null,
        float temperature = 0f, float topP = 1f, int topK = 1)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ValidateMixedPrompt(prompt);
        ArgumentOutOfRangeException.ThrowIfNegative(maxNewTokens);
        ArgumentOutOfRangeException.ThrowIfNegative(nextPosition);
        ValidateSampling(temperature, topP, topK);
        int generatedCapacity = Math.Min(maxNewTokens, Descriptor.ContextLength - prompt.Count);
        if (nextPosition > int.MaxValue - generatedCapacity)
            throw new ArgumentOutOfRangeException(nameof(nextPosition));

        cancellationToken.ThrowIfCancellationRequested();

        int reused = CanReuseMixedPromptPrefix(prompt) ? _cachedMixedPromptTokens!.Length : 0;
        if (reused == 0) Reset();
        _lastReusedPromptTokens = reused;
        var result = prompt.Select(entry => entry.TokenId).ToList();
        if (generatedCapacity == 0)
        {
            InvalidatePromptCheckpoint();
            return result.ToArray();
        }

        ArcBuffer? logits = null;
        bool greedy = temperature == 0f || topK == 1;
        float[]? hostLogits = greedy ? null : new float[Descriptor.VocabularySize];
        try
        {
            foreach (LayerState state in _states) state.EnsureCapacity(prompt.Count);
            int i = reused;
            for (; CanPrefillChunk(prompt.Count - 1 - i);
                i += Math.Min(_options.InferencePrefillChunkTokens, prompt.Count - 1 - i))
            {
                cancellationToken.ThrowIfCancellationRequested();
                ForwardPromptChunkDevice(prompt, i,
                    Math.Min(_options.InferencePrefillChunkTokens, prompt.Count - 1 - i), cancellationToken);
            }
            for (; i < prompt.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Qwen35PromptToken entry = prompt[i];
                logits = ForwardTokenDevice(entry.TokenId, i == prompt.Count - 1,
                    entry.Embedding, entry.Position);
            }
            for (int generated = 0; generated < generatedCapacity; generated++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int next;
                if (greedy)
                    next = Qwen35Gpu.ArgMax(OutputMatrix.Lane, logits!, Descriptor.VocabularySize);
                else
                {
                    OutputMatrix.Lane.Read(logits!, hostLogits!);
                    next = SampleLogits(hostLogits, temperature, topP, topK, Random.Shared);
                }
                logits!.Dispose(); logits = null;
                result.Add(next);
                onToken?.Invoke(next);
                cancellationToken.ThrowIfCancellationRequested();
                if (next == eosTokenId || generated + 1 == generatedCapacity)
                    break;
                logits = ForwardTokenDevice(next, true, null,
                    Qwen35Position.Scalar(nextPosition + generated));
            }
            if (reused > 0)
            {
                // Keep the attachment checkpoint. The unknown question and
                // generated answer are never silently published as a prefix.
                foreach (LayerState state in _states) state.RestorePromptState();
                foreach (ArcExecutionLane lane in _lanes) lane.Synchronize();
                _position = reused;
            }
            return result.ToArray();
        }
        catch
        {
            _faulted = true;
            InvalidatePromptCheckpoint();
            throw;
        }
        finally { logits?.Dispose(); }
    }
}
