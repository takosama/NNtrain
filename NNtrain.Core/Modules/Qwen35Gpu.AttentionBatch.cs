using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

internal static partial class Qwen35Gpu
{
    /// <summary>
    /// Chunked causal attention. Q/K normalization, rotary positions and cache
    /// writes span the chunk; each row still attends exactly its own prefix.
    /// </summary>
    internal static ArcBuffer AttentionRows(ArcExecutionLane lane, ArcBuffer qAndGate,
        ArcBuffer key, ArcBuffer value, ArcBuffer qNorm, ArcBuffer kNorm,
        ArcBuffer keysCache, ArcBuffer valuesCache, int startPosition, int rows,
        int heads, int kvHeads, int headWidth, int ropeDimensions, float theta,
        float eps, IReadOnlyList<Qwen35Position> ropePositions,
        IReadOnlyList<int>? ropeSections = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(startPosition);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        if (heads <= 0 || kvHeads <= 0 || heads % kvHeads != 0 || headWidth <= 0)
            throw new ArgumentException("Invalid grouped-query attention dimensions.");
        if (ropeDimensions < 0 || ropeDimensions > headWidth || (ropeDimensions & 1) != 0)
            throw new ArgumentOutOfRangeException(nameof(ropeDimensions));
        if (!float.IsFinite(theta) || theta <= 0) throw new ArgumentOutOfRangeException(nameof(theta));
        ValidateAttentionEpsilon(eps);
        ArgumentNullException.ThrowIfNull(ropePositions);
        if (ropePositions.Count != rows)
            throw new ArgumentException("Each attention row needs a rotary position.", nameof(ropePositions));
        bool spatial = false;
        var coordinates = new int[checked(rows * 3)];
        for (int row = 0; row < rows; row++)
        {
            Qwen35Position position = ropePositions[row];
            if (position.Temporal < 0 || position.Height < 0 || position.Width < 0)
                throw new ArgumentOutOfRangeException(nameof(ropePositions));
            coordinates[row * 3] = position.Temporal;
            coordinates[row * 3 + 1] = position.Height;
            coordinates[row * 3 + 2] = position.Width;
            spatial |= position.Temporal != position.Height || position.Temporal != position.Width;
        }
        if (spatial && ropeDimensions > 0 && (ropeSections is null || ropeSections.Count < 3
            || ropeSections[1] < 0 || ropeSections[2] < 0
            || ropeSections[1] > ropeDimensions / 2 || ropeSections[2] > ropeDimensions / 2))
            throw new NotSupportedException("Qwen3.5 multimodal RoPE sections are missing or invalid.");
        int querySize = checked(heads * headWidth), kvSize = checked(kvHeads * headWidth);
        int sequence = checked(startPosition + rows), cacheSize = checked(sequence * kvSize);
        ValidateAttentionBuffer(qAndGate, checked(rows * 2 * querySize), nameof(qAndGate));
        ValidateAttentionBuffer(key, checked(rows * kvSize), nameof(key));
        ValidateAttentionBuffer(value, checked(rows * kvSize), nameof(value));
        ValidateAttentionBuffer(qNorm, headWidth, nameof(qNorm));
        ValidateAttentionBuffer(kNorm, headWidth, nameof(kNorm));
        ValidateAttentionBuffer(keysCache, cacheSize, nameof(keysCache));
        ValidateAttentionBuffer(valuesCache, cacheSize, nameof(valuesCache));
        int scoreCapacity = AttentionRowsScoreCapacity(sequence);
        int scoreElements = checked(rows * heads * scoreCapacity);
        if ((ulong)scoreElements * sizeof(float) > lane.Device.MaximumAllocationBytes)
            throw new ArgumentOutOfRangeException(nameof(rows), "Attention chunk scratch exceeds the device allocation limit.");

        // One position upload covers every row in the attention chunk.
        using ArcBuffer positionBuffer = lane.UploadRaw(coordinates);
        using ArcBuffer query = NormalizeAttentionRows(lane, qAndGate, qNorm,
            checked(rows * heads), headWidth, checked(2 * headWidth), eps);
        using ArcBuffer normalizedKey = RmsNorm(lane, key, kNorm,
            checked(rows * kvHeads), headWidth, eps);
        if (ropeDimensions > 0)
        {
            int heightSection = ropeSections is { Count: >= 3 } ? ropeSections[1] : 0;
            int widthSection = ropeSections is { Count: >= 3 } ? ropeSections[2] : 0;
            lane.Run("q35a_mrope_rows", (long)rows * heads * (ropeDimensions / 2), AttentionReductionSize,
                query, positionBuffer, rows, heads, headWidth, ropeDimensions,
                heightSection, widthSection, theta);
            lane.Run("q35a_mrope_rows", (long)rows * kvHeads * (ropeDimensions / 2), AttentionReductionSize,
                normalizedKey, positionBuffer, rows, kvHeads, headWidth, ropeDimensions,
                heightSection, widthSection, theta);
        }
        int cacheOffset = checked(startPosition * kvSize * sizeof(float));
        int chunkBytes = checked(rows * kvSize * sizeof(float));
        lane.CopyBytes(normalizedKey, keysCache, 0, cacheOffset, chunkBytes);
        lane.CopyBytes(value, valuesCache, 0, cacheOffset, chunkBytes);
        using ArcBuffer scores = lane.Allocate(scoreElements);
        lane.Run("q35a_scores_rows", (long)rows * heads * sequence, AttentionReductionSize,
            query, keysCache, scores, rows, startPosition, heads, kvHeads,
            headWidth, sequence, scoreCapacity);
        lane.Run("q35a_softmax_rows", (long)rows * heads * AttentionReductionSize, AttentionReductionSize,
            scores, heads, startPosition, scoreCapacity);
        ArcBuffer output = lane.Allocate(checked(rows * querySize));
        try
        {
            lane.Run("q35a_attend_rows", (long)rows * querySize, AttentionReductionSize,
                scores, valuesCache, qAndGate, output, rows, startPosition,
                heads, kvHeads, headWidth, scoreCapacity);
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    internal static bool CanAttentionRows(ArcExecutionLane lane, int startPosition, int rows, int heads)
    {
        long elements = (long)rows * heads * AttentionRowsScoreCapacity(checked(startPosition + rows));
        return elements <= int.MaxValue && (ulong)elements * sizeof(float)
            <= Math.Min(lane.Device.MaximumAllocationBytes, 128UL * 1024 * 1024);
    }

    private static int AttentionRowsScoreCapacity(int sequence)
    {
        int capacity = 16;
        while (capacity < sequence && capacity <= int.MaxValue / 2) capacity *= 2;
        return Math.Max(capacity, sequence);
    }
}
