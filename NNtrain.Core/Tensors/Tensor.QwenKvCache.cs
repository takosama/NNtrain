using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    internal Tensor ArcQwenCachedAttention(Tensor key, Tensor value,
        ArcQwenKvCache cache, int position, float ropeTheta)
    {
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(value);
        ArgumentNullException.ThrowIfNull(cache);
        if (!ArcResident || AutogradContext.IsRecordingEnabled)
            throw new InvalidOperationException("Qwen K/V caching requires resident Arc no-grad inference.");
        if (!float.IsFinite(ropeTheta) || ropeTheta <= 0f)
            throw new ArgumentOutOfRangeException(nameof(ropeTheta));
        if (Rank != 3 || _shape[0] != 1 || _shape[2] != cache.QueryHeads * cache.HeadWidth
            || key.Rank != 3 || key._shape[0] != 1 || key._shape[1] != _shape[1]
            || key._shape[2] != cache.KvHeads * cache.HeadWidth
            || !key._shape.AsSpan().SequenceEqual(value._shape))
            throw new ArgumentException("Qwen cached GQA requires matching batch-one query/key/value shapes.");
        int tokens = _shape[1];
        if (position != 0 && tokens != 1)
            throw new ArgumentException("Qwen cached decode accepts one token after prefill.");
        var lane = ArcLane;
        cache.Validate(lane, position, tokens);
        using var k = key.ArcUploadValues(matrixOperand: true);
        using var v = value.ArcUploadValues(matrixOperand: true);
        lane.Run("qwen_kv_append", checked((long)tokens * cache.KvHeads * cache.HeadWidth / 2), 0,
            k, v, cache.Key, cache.Value, tokens, position, cache.KvHeads,
            cache.HeadWidth, ropeTheta);

        if (position == 0)
        {
            // Preserve the reference prefill arithmetic; the cache holds its
            // projected K/V for all later single-token passes.
            Tensor prefill = QwenGroupedQueryAttention(key, value,
                cache.QueryHeads, cache.KvHeads, ropeTheta, causal: true);
            cache.MarkAppended(tokens);
            return prefill;
        }

        using var q = ArcUploadValues(matrixOperand: true);
        using var output = lane.Allocate(Numel);
        int length = position + 1;
        if (length <= 4096)
            lane.Run("qwen_kv_decode", cache.QueryHeads * 64L, 64,
                q, cache.Key, cache.Value, output, position, cache.QueryHeads,
                cache.KvHeads, cache.HeadWidth, ropeTheta,
                new LocalMemory(checked((length + cache.HeadWidth) * sizeof(float))));
        else
        {
            // Long decode rows use device scratch, avoiding a context-length
            // dependent local-memory limit. The scratch dies with this token.
            using var scores = lane.Allocate(checked(cache.QueryHeads * length));
            using var rotatedQuery = lane.Allocate(Numel);
            lane.Run("qwen_kv_rotate_query", Numel / 2, 0, q, rotatedQuery,
                position, cache.QueryHeads, cache.HeadWidth, ropeTheta);
            lane.Run("qwen_kv_scores", checked((long)cache.QueryHeads * length), 0,
                rotatedQuery, cache.Key, scores, length, cache.QueryHeads,
                cache.KvHeads, cache.HeadWidth);
            lane.Run("qwen_kv_output", cache.QueryHeads * 64L, 64,
                scores, cache.Value, output, length, cache.QueryHeads,
                cache.KvHeads, cache.HeadWidth);
        }
        Tensor result = ArcDeviceResult(output, _shape, [this, key, value]);
        cache.MarkAppended(tokens);
        return result;
    }
}
