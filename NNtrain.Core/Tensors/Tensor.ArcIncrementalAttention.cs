using NNtrain.Arc;
using NNtrain.Runtime.Execution;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Copies the projected K/V of a full-prompt prefill into a persistent
    /// cache. The caller separately computes that prompt's ordinary attention
    /// output, so this operation never changes the first generated token.
    /// </summary>
    internal void ArcPrefillAttentionKvCache(ArcAttentionKvCache cache, int sequence)
    {
        ArgumentNullException.ThrowIfNull(cache);
        if (!ArcResident || AutogradContext.IsRecordingEnabled)
            throw new InvalidOperationException("Arc K/V prefill requires resident no-grad inference.");
        ArcExecutionLane lane = ArcLane;
        cache.Validate(lane);
        if (sequence < 1 || sequence > cache.Capacity || cache.Length != 0)
            throw new ArgumentOutOfRangeException(nameof(sequence),
                "Arc K/V prefill must run once with a sequence within capacity.");
        if (Rank != 3 || _shape[0] != 1 || _shape[1] != sequence
            || _shape[2] != checked(3 * cache.Width))
            throw new ArgumentException("Arc K/V prefill requires QKV [1, sequence, 3*width].", nameof(cache));

        using var qkv = ArcUploadValues(matrixOperand: true);
        lane.Run("attention_kv_prefill_fp32", checked((long)sequence * cache.Width), 256,
            qkv, cache.Key, cache.Value,
            sequence, cache.Capacity, cache.Width, cache.Heads);
        cache.MarkPrefilled(sequence);
    }

    /// <summary>
    /// Appends one projected K/V pair and computes one causal attention output.
    /// The cache and this QKV token must belong to the current Arc lane.
    /// </summary>
    internal Tensor ArcIncrementalCausalAttention(ArcAttentionKvCache cache, int position)
    {
        ArgumentNullException.ThrowIfNull(cache);
        if (!ArcResident || AutogradContext.IsRecordingEnabled)
            throw new InvalidOperationException("Arc incremental attention requires resident no-grad inference.");
        ArcExecutionLane lane = ArcLane;
        cache.Validate(lane);
        if (position != cache.Length || position < 0 || position >= cache.Capacity)
            throw new ArgumentOutOfRangeException(nameof(position),
                "Arc incremental tokens must append in position order within capacity.");
        if (Rank != 3 || _shape[0] != 1 || _shape[1] != 1
            || _shape[2] != checked(3 * cache.Width))
            throw new ArgumentException("Arc incremental attention requires QKV [1, 1, 3*width].", nameof(cache));

        using var qkv = ArcUploadValues(matrixOperand: true);
        using var output = lane.Allocate(cache.Width);
        int sequence = position + 1, headWidth = cache.Width / cache.Heads;
        PrecisionPolicy? policy = TensorExecutionContext.ActivePrecisionPolicy;
        bool sg16Xmx = lane.Options.XmxMatrices && lane.Device.SupportsXmx
            && lane.Device.MinimumSubgroupSize == 16;
        bool batchedFullAttention = lane.Options.BatchedAttention
            && (long)sequence * sequence * sizeof(float) <= 32L * 1024 * 1024
            && !CanUseArcFlashAttention(sequence, cache.Width, cache.Heads);
        bool speedPolicy = policy is { Mode: PrecisionMode.Mix8_16,
            AllowNonWeightReassociation: true };
        bool fastXmxProducts = lane.Options.Mix8_16XmxAttentionProducts
            && speedPolicy && policy is not null
            && (policy.AllowedNonWeightComputeFormats & NumericFormatSet.BFloat16) != 0
            && lane.Options.DirectXmxAttentionProducts
            && batchedFullAttention && sg16Xmx && DType != TensorDType.Float32
            && sequence == 2048 && headWidth == 32;
        bool cachedProbabilities = lane.Options.CachedAttentionProbabilities
            && !lane.Options.SubgroupAttentionReduction && sg16Xmx
            && sequence == 2048 && lane.Options.CachedAttentionProbabilities2048;
        bool nativeExp = lane.Options.Mix8_16NativeExpAttention
            && speedPolicy && batchedFullAttention && !cachedProbabilities
            && !lane.Options.SubgroupAttentionReduction && sg16Xmx
            && sequence == 2048 && headWidth == 32;
        lane.Run("attention_kv_decode_causal_fp32", cache.Heads * 64L, 64,
            qkv, cache.Key, cache.Value, output,
            position, cache.Capacity, cache.Width, cache.Heads,
            fastXmxProducts ? 1 : 0, nativeExp ? 1 : 0,
            new LocalMemory(checked((position + 1) * sizeof(float))));
        Tensor result = ArcDeviceResult(output, [1, 1, cache.Width], [this]);
        cache.MarkAppended(position);
        return result;
    }
}
