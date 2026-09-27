using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>Resident Float32 operations for Qwen3.5 inference. Callers own returned buffers.</summary>
internal static partial class Qwen35Gpu
{
    private const int AttentionReductionSize = 128;

    internal static ArcBuffer RmsNorm(ArcExecutionLane lane, ArcBuffer input, ArcBuffer weight,
        int rows, int width, float eps)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ValidateAttentionEpsilon(eps);
        ValidateAttentionBuffer(input, checked(rows * width), nameof(input));
        ValidateAttentionBuffer(weight, width, nameof(weight));
        return NormalizeAttentionRows(lane, input, weight, rows, width, width, eps);
    }

    private static ArcBuffer NormalizeAttentionRows(ArcExecutionLane lane, ArcBuffer input,
        ArcBuffer weight, int rows, int width, int inputStride, float eps)
    {
        ArcBuffer output = lane.Allocate(checked(rows * width));
        try
        {
            lane.Run("q35a_rms_norm", (long)rows * AttentionReductionSize, AttentionReductionSize,
                input, weight, output, width, inputStride, eps);
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    internal static ArcBuffer SiluMultiply(ArcExecutionLane lane, ArcBuffer gate, ArcBuffer up, int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        ValidateAttentionBuffer(gate, length, nameof(gate));
        ValidateAttentionBuffer(up, length, nameof(up));
        ArcBuffer output = lane.Allocate(length);
        try
        {
            lane.Run("q35a_silu_multiply", length, AttentionReductionSize, gate, up, output, length);
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    internal static void AddInPlace(ArcExecutionLane lane, ArcBuffer destination, ArcBuffer source, int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        ValidateAttentionBuffer(destination, length, nameof(destination));
        ValidateAttentionBuffer(source, length, nameof(source));
        lane.Run("q35a_add_in_place", length, AttentionReductionSize, destination, source, length);
    }

    /// <summary>
    /// Normalizes and rotates this token's Q/K, appends K/V into the caller's caches,
    /// then applies causal grouped-query attention and the per-head sigmoid gate.
    /// Cache layout is [time, KV head, component]. Scratch is O(heads * sequence).
    /// </summary>
    internal static ArcBuffer AttentionStep(ArcExecutionLane lane, ArcBuffer qAndGate,
        ArcBuffer key, ArcBuffer value, ArcBuffer qNorm, ArcBuffer kNorm,
        ArcBuffer keysCache, ArcBuffer valuesCache, int position, int heads, int kvHeads,
        int headWidth, int ropeDimensions, float theta, float eps)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        if (heads <= 0 || kvHeads <= 0 || heads % kvHeads != 0 || headWidth <= 0)
            throw new ArgumentException("Invalid grouped-query attention dimensions.");
        if (ropeDimensions < 0 || ropeDimensions > headWidth || (ropeDimensions & 1) != 0)
            throw new ArgumentOutOfRangeException(nameof(ropeDimensions));
        if (!float.IsFinite(theta) || theta <= 0) throw new ArgumentOutOfRangeException(nameof(theta));
        ValidateAttentionEpsilon(eps);
        int querySize = checked(heads * headWidth), kvSize = checked(kvHeads * headWidth);
        int sequence = checked(position + 1), cacheSize = checked(sequence * kvSize);
        ValidateAttentionBuffer(qAndGate, checked(2 * querySize), nameof(qAndGate));
        ValidateAttentionBuffer(key, kvSize, nameof(key));
        ValidateAttentionBuffer(value, kvSize, nameof(value));
        ValidateAttentionBuffer(qNorm, headWidth, nameof(qNorm));
        ValidateAttentionBuffer(kNorm, headWidth, nameof(kNorm));
        ValidateAttentionBuffer(keysCache, cacheSize, nameof(keysCache));
        ValidateAttentionBuffer(valuesCache, cacheSize, nameof(valuesCache));

        using ArcBuffer query = NormalizeAttentionRows(lane, qAndGate, qNorm, heads,
            headWidth, checked(2 * headWidth), eps);
        using ArcBuffer normalizedKey = RmsNorm(lane, key, kNorm, kvHeads, headWidth, eps);
        if (ropeDimensions > 0)
        {
            lane.Run("q35a_rope", (long)heads * (ropeDimensions / 2), AttentionReductionSize,
                query, heads, headWidth, ropeDimensions, position, theta);
            lane.Run("q35a_rope", (long)kvHeads * (ropeDimensions / 2), AttentionReductionSize,
                normalizedKey, kvHeads, headWidth, ropeDimensions, position, theta);
        }
        int cacheOffset = checked(position * kvSize * sizeof(float)), kvBytes = checked(kvSize * sizeof(float));
        lane.CopyBytes(normalizedKey, keysCache, 0, cacheOffset, kvBytes);
        lane.CopyBytes(value, valuesCache, 0, cacheOffset, kvBytes);
        using ArcBuffer scores = lane.Allocate(checked(heads * sequence));
        lane.Run("q35a_scores", (long)heads * sequence, AttentionReductionSize,
            query, keysCache, scores, heads, kvHeads, headWidth, sequence);
        lane.Run("q35a_softmax", (long)heads * AttentionReductionSize, AttentionReductionSize, scores, sequence);
        ArcBuffer output = lane.Allocate(querySize);
        try
        {
            lane.Run("q35a_attend", querySize, AttentionReductionSize,
                scores, valuesCache, qAndGate, output, heads, kvHeads, headWidth, sequence);
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    /// <summary>Downloads only the selected token ID and a non-finite status flag.</summary>
    internal static int ArgMax(ArcExecutionLane lane, ArcBuffer logits, int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        ValidateAttentionBuffer(logits, count, nameof(logits));
        using ArcBuffer result = lane.Allocate(2);
        const int partitionSize = 4096;
        if (lane.Options.Qwen35ParallelArgmax && count >= partitionSize)
        {
            int partitions = checked((int)(((long)count + partitionSize - 1) / partitionSize));
            // Each partition stores the maximum's float bits, its token ID and
            // a non-finite flag. Only the final two integers leave the device.
            using ArcBuffer partials = lane.Allocate(checked(partitions * 3));
            lane.Run("q35a_argmax_partition", (long)partitions * AttentionReductionSize,
                AttentionReductionSize, logits, partials, count, partitionSize);
            lane.Run("q35a_argmax_reduce", AttentionReductionSize, AttentionReductionSize,
                partials, result, partitions);
        }
        else
            lane.Run("q35a_argmax", AttentionReductionSize, AttentionReductionSize, logits, result, count);
        int[] status = new int[2];
        lane.ReadRaw(result, status);
        if (status[1] != 0) throw new ArithmeticException("Qwen3.5 produced non-finite logits.");
        return status[0];
    }

    private static void ValidateAttentionBuffer(ArcBuffer buffer, int elements, string name)
    {
        ArgumentNullException.ThrowIfNull(buffer, name);
        if (!buffer.IsAlive || buffer.ByteLength < (long)elements * sizeof(float))
            throw new ArgumentException("The Arc buffer is disposed or too small for its declared dimensions.", name);
    }

    private static void ValidateAttentionEpsilon(float eps)
    {
        if (!float.IsFinite(eps) || eps < 0f) throw new ArgumentOutOfRangeException(nameof(eps));
    }
}
