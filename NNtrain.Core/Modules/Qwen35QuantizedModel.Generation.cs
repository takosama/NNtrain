using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

public sealed partial class Qwen35QuantizedModel
{
    /// <summary>
    /// Starts a new sequence. Greedy selection stays on the GPU; stochastic
    /// sampling downloads one logits row for top-k and top-p filtering.
    /// The final emitted token has not yet been forwarded.
    /// </summary>
    public int[] GenerateTokenIds(IReadOnlyList<int> prompt, int maxNewTokens,
        int? eosTokenId = null, Action<int>? onToken = null,
        float temperature = 0f, float topP = 1f, int topK = 1, Random? random = null)
        => GenerateTokenIdsCore(prompt, maxNewTokens, eosTokenId, onToken, reusePromptPrefix: false,
            temperature, topP, topK, random, CancellationToken.None);

    public int[] GenerateTokenIds(IReadOnlyList<int> prompt, int maxNewTokens,
        CancellationToken cancellationToken, int? eosTokenId = null, Action<int>? onToken = null,
        float temperature = 0f, float topP = 1f, int topK = 1, Random? random = null)
        => GenerateTokenIdsCore(prompt, maxNewTokens, eosTokenId, onToken, reusePromptPrefix: false,
            temperature, topP, topK, random, cancellationToken);

    /// <summary>
    /// Reuses a previous prompt's GPU state only when its token IDs are an exact
    /// prefix of this prompt. Full-attention K/V rows are retained in place;
    /// recurrent state is restored from a GPU-side copy after each response.
    /// A changed or shorter prompt falls back to a full prefill.
    /// </summary>
    public int[] GenerateTokenIdsWithPrefixReuse(IReadOnlyList<int> prompt, int maxNewTokens,
        int? eosTokenId = null, Action<int>? onToken = null,
        float temperature = 0f, float topP = 1f, int topK = 1, Random? random = null)
        => GenerateTokenIdsCore(prompt, maxNewTokens, eosTokenId, onToken, reusePromptPrefix: true,
            temperature, topP, topK, random, CancellationToken.None);

    public int[] GenerateTokenIdsWithPrefixReuse(IReadOnlyList<int> prompt, int maxNewTokens,
        CancellationToken cancellationToken, int? eosTokenId = null, Action<int>? onToken = null,
        float temperature = 0f, float topP = 1f, int topK = 1, Random? random = null)
        => GenerateTokenIdsCore(prompt, maxNewTokens, eosTokenId, onToken, reusePromptPrefix: true,
            temperature, topP, topK, random, cancellationToken);

    /// <summary>
    /// Advances the resident state to an exact conversation prefix without
    /// sampling. This lets the next user turn reuse a sanitized assistant
    /// answer while leaving earlier thinking text out of the context.
    /// </summary>
    public (int ReusedTokens, bool Cached) PrimePromptPrefix(IReadOnlyList<int> prompt,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _cachedMixedPromptTokens = null;
        ArgumentNullException.ThrowIfNull(prompt);
        if (prompt.Count == 0 || prompt.Count >= Descriptor.ContextLength)
            throw new ArgumentException("Prefix must leave room for a following turn.", nameof(prompt));
        if (prompt.Any(token => (uint)token >= (uint)Descriptor.VocabularySize))
            throw new ArgumentOutOfRangeException(nameof(prompt));
        cancellationToken.ThrowIfCancellationRequested();
        if (_cachedPromptTokens is { } cached && _position == cached.Length
            && _states.All(state => state.HasPromptState) && cached.SequenceEqual(prompt))
        {
            _lastReusedPromptTokens = cached.Length;
            return (cached.Length, true);
        }
        int reused = CanReusePromptPrefix(prompt) ? _cachedPromptTokens!.Length : 0;
        if (reused == 0) Reset();
        _lastReusedPromptTokens = reused;
        try
        {
            foreach (LayerState state in _states) state.EnsureCapacity(prompt.Count);
            int i = reused;
            for (; CanPrefillChunk(prompt.Count - i); i += Math.Min(_options.InferencePrefillChunkTokens, prompt.Count - i))
            {
                cancellationToken.ThrowIfCancellationRequested();
                ForwardPromptChunkDevice(prompt, i, Math.Min(_options.InferencePrefillChunkTokens, prompt.Count - i),
                    cancellationToken);
            }
            for (; i < prompt.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                _ = ForwardTokenDevice(prompt[i], returnLogits: false);
            }
            cancellationToken.ThrowIfCancellationRequested();
            if (!TryCapturePromptCheckpoint()) return (reused, false);
            _cachedPromptTokens = prompt.ToArray();
            return (reused, true);
        }
        catch { _faulted = true; InvalidatePromptCheckpoint(); throw; }
    }

    private int[] GenerateTokenIdsCore(IReadOnlyList<int> prompt, int maxNewTokens,
        int? eosTokenId, Action<int>? onToken, bool reusePromptPrefix,
        float temperature, float topP, int topK, Random? random,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentOutOfRangeException.ThrowIfNegative(maxNewTokens);
        ValidateSampling(temperature, topP, topK);
        if (prompt.Count == 0 || prompt.Count > Descriptor.ContextLength)
            throw new ArgumentException("Prompt must contain 1..context-length tokens.", nameof(prompt));
        if (prompt.Any(token => (uint)token >= (uint)Descriptor.VocabularySize))
            throw new ArgumentOutOfRangeException(nameof(prompt));
        cancellationToken.ThrowIfCancellationRequested();
        int reused = reusePromptPrefix && CanReusePromptPrefix(prompt)
            ? _cachedPromptTokens!.Length : 0;
        if (reused == 0) Reset();
        _lastReusedPromptTokens = reused;
        var result = prompt.ToList();
        if (maxNewTokens == 0 || result.Count == Descriptor.ContextLength)
        {
            // No final logits are computed in this path, so it cannot seed a
            // later continuation even when an earlier prefix was reusable.
            InvalidatePromptCheckpoint();
            return result.ToArray();
        }
        ArcBuffer? logits = null;
        bool checkpointCaptured = false;
        bool greedy = temperature == 0f || topK == 1;
        float[]? hostLogits = greedy ? null : new float[Descriptor.VocabularySize];
        random ??= Random.Shared;
        try
        {
            foreach (LayerState state in _states) state.EnsureCapacity(prompt.Count);
            int i = reused;
            for (; CanPrefillChunk(prompt.Count - 1 - i); i += Math.Min(_options.InferencePrefillChunkTokens, prompt.Count - 1 - i))
            {
                cancellationToken.ThrowIfCancellationRequested();
                ForwardPromptChunkDevice(prompt, i, Math.Min(_options.InferencePrefillChunkTokens, prompt.Count - 1 - i),
                    cancellationToken);
            }
            for (; i < prompt.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                logits = ForwardTokenDevice(prompt[i], i == prompt.Count - 1);
            }
            cancellationToken.ThrowIfCancellationRequested();
            if (reusePromptPrefix)
                checkpointCaptured = TryCapturePromptCheckpoint();
            for (int generated = 0; generated < maxNewTokens && result.Count < Descriptor.ContextLength; generated++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int next;
                if (greedy)
                    next = Qwen35Gpu.ArgMax(OutputMatrix.Lane, logits!, Descriptor.VocabularySize);
                else
                {
                    OutputMatrix.Lane.Read(logits!, hostLogits!);
                    next = SampleLogits(hostLogits, temperature, topP, topK, random);
                }
                logits!.Dispose(); logits = null;
                result.Add(next);
                onToken?.Invoke(next);
                cancellationToken.ThrowIfCancellationRequested();
                if (next == eosTokenId || generated + 1 == maxNewTokens || result.Count == Descriptor.ContextLength) break;
                logits = ForwardTokenDevice(next, true);
            }
            if (checkpointCaptured)
            {
                foreach (LayerState state in _states) state.RestorePromptState();
                foreach (ArcExecutionLane lane in _lanes) lane.Synchronize();
                _position = prompt.Count;
                _cachedPromptTokens = prompt.ToArray();
            }
            return result.ToArray();
        }
        catch { _faulted = true; InvalidatePromptCheckpoint(); throw; }
        finally { logits?.Dispose(); }
    }

    private static void ValidateSampling(float temperature, float topP, int topK)
        => Qwen35LogitSampler.ValidateSampling(temperature, topP, topK);

    /// <summary>
    /// Selects top-k before nucleus filtering, as required by Qwen sampling.
    /// The returned ID is reproducible when the caller supplies a seeded Random.
    /// </summary>
    internal static int SampleLogits(ReadOnlySpan<float> logits,
        float temperature, float topP, int topK, Random random)
        => Qwen35LogitSampler.SampleLogits(logits, temperature, topP, topK, random);

    private bool CanReusePromptPrefix(IReadOnlyList<int> prompt)
    {
        int[]? cached = _cachedPromptTokens;
        if (_faulted || cached is null || cached.Length >= prompt.Count || _position != cached.Length
            || _states.Any(state => !state.HasPromptState)) return false;
        for (int i = 0; i < cached.Length; i++)
            if (cached[i] != prompt[i]) return false;
        return true;
    }

    private bool TryCapturePromptCheckpoint()
    {
        // The previous snapshot is no longer a valid prefix once the prompt
        // advances, but its GPU buffers can hold the new recurrent state.
        _cachedPromptTokens = null;
        _cachedMixedPromptTokens = null;
        foreach (ArcExecutionLane lane in _lanes)
        {
            long bytes = _states.Where(state => ReferenceEquals(state.Lane, lane))
                .Sum(state => state.AdditionalPromptSnapshotBytes);
            if (lane.AllocatedBytes + bytes + WorkspaceReserveBytes > lane.EffectivePhysicalBufferBudgetBytes)
            {
                InvalidatePromptCheckpoint();
                return false; // Generate normally when a snapshot will not fit.
            }
        }
        try
        {
            foreach (LayerState state in _states) state.CapturePromptState();
            foreach (ArcExecutionLane lane in _lanes) lane.Synchronize();
            return true;
        }
        catch
        {
            InvalidatePromptCheckpoint();
            throw;
        }
    }

    private void InvalidatePromptCheckpoint()
    {
        _cachedPromptTokens = null;
        _cachedMixedPromptTokens = null;
        foreach (LayerState state in _states) state.ClearPromptState();
    }
}
