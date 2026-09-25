using NNtrain.Runtime.Execution;

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
        TensorDType dtype, PrecisionMode precisionMode, int bfp8BlockSize, long parameterElements,
        long parameterStorageBytes, long largestParameterElements,
        long largestBackwardProducerLeafElements,
        long largestMuonScratchBytes, ulong globalMemoryBytes,
        bool retainAttentionOutputBFloat16 = false,
        bool publishBFloat16Activations = false,
        int fusedBfp8LinearOutputWidthPerLayer = 0,
        bool fusedBfp8FfnIntermediate = false)
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
        ArgumentOutOfRangeException.ThrowIfNegative(largestParameterElements);
        ArgumentOutOfRangeException.ThrowIfNegative(largestBackwardProducerLeafElements);
        ArgumentOutOfRangeException.ThrowIfNegative(largestMuonScratchBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(fusedBfp8LinearOutputWidthPerLayer);
        if (publishBFloat16Activations && precisionMode != PrecisionMode.Mix8_16)
            throw new ArgumentException("BF16 activation selection requires mix8_16 precision.",
                nameof(publishBFloat16Activations));
        if (fusedBfp8LinearOutputWidthPerLayer > 5L * width + hidden)
            throw new ArgumentOutOfRangeException(nameof(fusedBfp8LinearOutputWidthPerLayer));
        if (largestParameterElements > parameterElements)
            throw new ArgumentException("The largest parameter cannot exceed the total parameter count.",
                nameof(largestParameterElements));
        if (largestBackwardProducerLeafElements < largestParameterElements
            || largestBackwardProducerLeafElements > parameterElements)
            throw new ArgumentException("The backward producer leaf count must cover the largest parameter and stay within the model.",
                nameof(largestBackwardProducerLeafElements));
        if (width % heads != 0) throw new ArgumentException("Head count must divide the model width.");
        if (dtype is not (TensorDType.Float32 or TensorDType.BFloat16 or TensorDType.Bfp8))
            throw new NotSupportedException($"Arc checkpoint planning does not support {dtype}.");
        if (precisionMode == PrecisionMode.Mix8_16 && dtype != TensorDType.Bfp8)
            throw new ArgumentException("mix8_16 requires BFP8 parameter storage.", nameof(dtype));

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
            long retainedFfn = expanded, blockBoundary = activation;
            if (publishBFloat16Activations)
            {
                // The direct fused GEMM epilogue still publishes BFP8. Count
                // its eligible output widths separately; all remaining Arc
                // operation results use physical two-byte BF16 publication.
                long totalWidth = 8L * width + hidden;
                long fusedElements = rows * fusedBfp8LinearOutputWidthPerLayer;
                perLayer = (rows * totalWidth - fusedElements) * 2
                    + (fusedElements == 0 ? 0 : Packed(fusedElements))
                    + rows * heads * 8 + rows * 16;
                retainedFfn = fusedBfp8FfnIntermediate ? expanded : rows * hidden * 2;
                blockBoundary = rows * width * 2;
            }
            if (retainAttentionOutputBFloat16 && precisionMode == PrecisionMode.Mix8_16
                && sequence == 2048 && width / heads == 32)
                perLayer += rows * width * 2;
            long outsideBlocks = (publishBFloat16Activations ? rows * width * 2 : activation) * 3
                + rows * 16; // embeddings/final norm, stats and token/target ids
            long uncheckpointed = perLayer * layers + outsideBlocks;
            // mix8_16 retains BF16 master/gradient/two moments (8 B/element).
            // Backward packs leaves after each producer, which can write
            // two leaf gradients together. Muon keeps four FP32 direction
            // buffers and three Gram matrices for its current parameter.
            // Reserve the larger peak, not one FP32 gradient for every leaf.
            // Other Arc modes retain four FP32 buffers (16 B/element).
            long persistent = precisionMode == PrecisionMode.Mix8_16
                ? parameterStorageBytes + parameterElements * 8
                    + Math.Max(largestBackwardProducerLeafElements * 4, largestMuonScratchBytes)
                : parameterStorageBytes + parameterElements * 16;
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
            long ffnOnly = uncheckpointed - retainedFfn * layers;
            long additionalSavingPerFullBlock = perLayer - retainedFfn - blockBoundary;
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
