using NNtrain.Runtime.Execution;

namespace NNtrain;

public sealed partial class GptRinWikiJp
{
    private (int Generated, bool Stopped) GenerateArcCached(
        List<int> result, int maximum, float temperature, int topK,
        int? stopTokenId, Random random, Action<int>? onToken)
    {
        var owned = new List<ArcAttentionKvCache>();
        try
        {
            int capacity = (int)Math.Min(ContextLength, (long)result.Count + maximum - 1);
            ArcAttentionKvCache[] CreateCaches(int device, bool sharded)
            {
                using var scope = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, device));
                return _blocks.Select(block =>
                {
                    var cache = new ArcAttentionKvCache(
                        ModelWidth / (sharded ? 2 : 1),
                        block.Attn.NumHeads / (sharded ? 2 : 1), capacity);
                    owned.Add(cache);
                    return cache;
                }).ToArray();
            }

            ExecutionSession session = ExecutionSession.Current!;
            int primary = Tensor.ArcLane.DeviceIndex;
            ArcAttentionKvCache[] first = CreateCaches(primary, ArcTensorParallelEnabled);
            ArcAttentionKvCache[]? second = ArcTensorParallelEnabled
                ? CreateCaches(session.Options.ArcDevices![1], true) : null;
            int generated = 0;
            bool stopped = false;
            int position = 0;
            int[] input = result.ToArray();
            while (generated < maximum && !stopped && position < ContextLength)
            {
                using IDisposable? frame = Tensor.BeginArcInferenceFrame();
                Tensor hidden = ArcTensorParallelEnabled
                    ? ForwardHiddenArcTensorParallel(input, input.Length, first, second, position)
                    : ForwardHiddenArcCached(input, position, first);
                Tensor logits = _languageModelHead.ForwardBatch(hidden.SelectLastSequenceToken());
                int next = SampleLogits(logits, 0, VocabularySize, temperature, topK, random);
                result.Add(next);
                generated++;
                stopped = stopTokenId.HasValue && next == stopTokenId.Value;
                onToken?.Invoke(next);
                position = result.Count - 1;
                input = [next];
            }
            return (generated, stopped);
        }
        finally
        {
            foreach (ArcAttentionKvCache cache in owned) cache.Dispose();
        }
    }

    private Tensor ArcInferenceEmbedding(int[] tokenIds, int position)
    {
        if (Tensor.ArcResident && !AutogradContext.IsRecordingEnabled)
            return _embeddingDropout.Forward(_tokenEmbedding.T.ArcInferenceEmbeddingWithPositions(
                _positionEmbedding.T, tokenIds, position));
        if (position == 0)
            return _embeddingDropout.Forward(_tokenEmbedding.T.EmbeddingLookupWithPositions(
                _positionEmbedding.T, tokenIds, 1, tokenIds.Length));
        Tensor token = _tokenEmbedding.T.EmbeddingLookup(tokenIds, 1).Reshape(1, 1, ModelWidth);
        Tensor positional = _positionEmbedding.T.EmbeddingLookup([position], 1).Reshape(1, 1, ModelWidth);
        return _embeddingDropout.Forward(token + positional);
    }

    private static Tensor ArcCachedAttention(Tensor qkv, int heads, int sequence,
        ArcAttentionKvCache? cache, int position)
    {
        if (cache is null) return qkv.FusedMultiHeadAttention(heads, causal: true);
        if (position != 0) return qkv.ArcIncrementalCausalAttention(cache, position);
        qkv.ArcPrefillAttentionKvCache(cache, sequence);
        return qkv.FusedMultiHeadAttention(heads, causal: true);
    }

    private Tensor ForwardHiddenArcCached(int[] tokens, int position, ArcAttentionKvCache[] caches)
    {
        Tensor hidden = ArcInferenceEmbedding(tokens, position);
        for (int layer = 0; layer < _blocks.Length; layer++)
        {
            TransformerBlock block = _blocks[layer];
            Tensor qkv = block.Attn.Qkv.ForwardBatch(hidden);
            Tensor attended = ArcCachedAttention(qkv, block.Attn.NumHeads, tokens.Length, caches[layer], position);
            Tensor projected = block.Attn.Wo.ForwardBatch(attended);
            Tensor normed = block.Ln1.ForwardResidualDropout(hidden, projected, block.AttnDropout);
            hidden = block.Ln2.ForwardResidualDropout(normed, block.Ffn.Forward(normed), block.FfnDropout);
        }
        return _finalNorm.Forward(hidden);
    }
}
