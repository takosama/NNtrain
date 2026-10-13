namespace NNtrain.Gui;

/// <summary>Conservative checked memory estimates shared by vision validation.</summary>
internal static class InferenceMemoryBudget
{
    internal static long VisionWorkspaceBytes(Qwen35VisionDescriptor d, bool xmxAttention = false)
    {
        // The learned position table is interpolated to the actual image grid;
        // ImageSize does not bound a 768-pixel-cap input's attention workspace.
        long rows = 768L * 768 / ((long)d.PatchSize * d.PatchSize);
        long mergedRows = rows / ((long)d.MergeSize * d.MergeSize);
        long attention = checked(rows * 7 * d.EmbeddingLength);
        long feedForward = checked(rows * (3L * d.EmbeddingLength + d.FeedForwardLength));
        long merger = checked(rows * 3 * d.EmbeddingLength + mergedRows * d.ProjectionDimension);
        long scratch = checked((rows * rows * d.HeadCount
            + rows * 3 * d.PatchSize * d.PatchSize + Math.Max(attention, Math.Max(feedForward, merger))) * sizeof(float));
        if (xmxAttention)
            scratch = checked(scratch + rows * 3 * d.EmbeddingLength * sizeof(float));
        // The vision lane retains a 256 MiB pool plus 128 MiB retired buffers.
        // The extra 64 MiB covers program/driver allocations and alignment.
        return checked(scratch + 448L * 1024 * 1024);
    }

    internal static long LanguageWorkspaceBytes(Qwen35GgufDescriptor d, int chunkTokens,
        bool xmx, bool batchAttention)
    {
        long rows = Math.Max(1, chunkTokens);
        long query = checked((long)d.HeadCount * d.HeadWidth);
        long kv = checked((long)d.KvHeadCount * d.HeadWidth);
        long values = checked((long)d.LinearValueHeads * d.LinearHeadWidth);
        long qkv = checked((2L * d.LinearKeyHeads + d.LinearValueHeads) * d.LinearHeadWidth);
        long feedForward = checked(5L * d.EmbeddingLength + 3L * d.FeedForwardLength);
        long recurrent = checked(3L * d.EmbeddingLength + 2 * qkv + 2 * values + 2L * d.LinearValueHeads);
        long attention = checked(3L * d.EmbeddingLength + 4 * query + 3 * kv);
        long scratch = checked(rows * Math.Max(feedForward, Math.Max(recurrent, attention)) * sizeof(float));
        if (xmx) scratch = checked(scratch + rows * Math.Max(d.FeedForwardLength, d.EmbeddingLength) * sizeof(float));
        if (batchAttention && rows > 1) scratch = checked(scratch + 128L * 1024 * 1024);
        return Math.Max(scratch, 64L * 1024 * 1024);
    }

}
