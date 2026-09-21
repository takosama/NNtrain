namespace NNtrain;

/// <summary>
/// Shape-only admission estimate for saved transformer activations. This is
/// not a free-VRAM query or an allocation guarantee: other applications and
/// allocator caches still require a device-wide memory budget.
/// </summary>
internal readonly record struct ArcTransformerMemoryPlan(
    int CheckpointPrefixLayers,
    bool CheckpointFfn,
    long UncheckpointedActivationBytes,
    long EstimatedActivationBytes,
    long ActivationBudgetBytes,
    long EstimatedPersistentBytes,
    long WorkingGuardBytes)
{
    internal bool FitsEstimatedBudget => EstimatedActivationBytes <= ActivationBudgetBytes;

    internal static ArcTransformerMemoryPlan Create(
        int batch, int sequence, int width, int heads, int hidden, int layers,
        TensorDType dtype, int bfp8BlockSize, long parameterElements,
        long parameterStorageBytes, ulong globalMemoryBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batch);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sequence);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(heads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hidden);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layers);
        ArgumentOutOfRangeException.ThrowIfNegative(bfp8BlockSize);
        ArgumentOutOfRangeException.ThrowIfNegative(parameterElements);
        ArgumentOutOfRangeException.ThrowIfNegative(parameterStorageBytes);
        if (width % heads != 0) throw new ArgumentException("Head count must divide the model width.");
        if (dtype is not (TensorDType.Float32 or TensorDType.BFloat16 or TensorDType.Bfp8))
            throw new NotSupportedException($"Arc checkpoint planning does not support {dtype}.");

        checked
        {
            long rows = (long)batch * sequence;
            long Packed(long elements) => dtype switch
            {
                TensorDType.Float32 => elements * 4,
                TensorDType.BFloat16 => elements * 2,
                _ => elements + (bfp8BlockSize == 0 ? 1 : (elements + bfp8BlockSize - 1) / bfp8BlockSize) * 4,
            };
            long activation = Packed(rows * width), expanded = Packed(rows * hidden);
            // QKV(3D), attention(D), projection(D), LN1(D), FFN(H),
            // FC2(D), LN2(D), plus FP32 attention/LN saved statistics.
            long perLayer = activation * 8 + expanded + rows * heads * 8 + rows * 16;
            long outsideBlocks = activation * 3 + rows * 16; // embeddings/final norm, stats and token/target ids
            long uncheckpointed = perLayer * layers + outsideBlocks;
            // Master, gradient, and two moments are FP32 even with packed
            // weights. This is stable across warmup and accumulation steps.
            long persistent = parameterStorageBytes + parameterElements * 16;
            // Two wide FP32 work buffers and two width-sized gradients, plus
            // bounded attention/GEMM workspace and driver/allocator headroom.
            long working = 768L * 1024 * 1024 + rows * (Math.Max(3L * width, hidden) + width) * 8;
            long global = (long)Math.Min(globalMemoryBytes, (ulong)long.MaxValue);
            long available = Math.Max(0, Fraction(global, 85, 100) - persistent - working);
            long budget = Math.Min(Fraction(global, 3, 5), available);
            if (uncheckpointed <= budget)
                return new(0, false, uncheckpointed, uncheckpointed, budget, persistent, working);

            // FFN recomputation removes the largest individual activation at
            // much lower compute cost than replaying attention. Add only the
            // minimum prefix of full blocks needed after that first saving.
            long ffnOnly = uncheckpointed - expanded * layers;
            long additionalSavingPerFullBlock = perLayer - expanded - activation;
            long shortage = Math.Max(0, ffnOnly - budget);
            int prefix = (int)Math.Min(layers,
                (shortage + additionalSavingPerFullBlock - 1) / additionalSavingPerFullBlock);
            long planned = ffnOnly - additionalSavingPerFullBlock * prefix;
            return new(prefix, true, uncheckpointed, planned, budget, persistent, working);
        }
    }

    private static long Fraction(long value, int numerator, int denominator)
        => value / denominator * numerator + value % denominator * numerator / denominator;
}
