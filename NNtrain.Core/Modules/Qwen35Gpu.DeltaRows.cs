using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

internal static partial class Qwen35Gpu
{
    /// <summary>
    /// Advances a width-128 DeltaNet prompt chunk with the same convolution,
    /// normalization and token-ordered state updates as DeltaStepFused.
    /// The recurrent dispatches are bounded to keep cancellation and driver
    /// scheduling responsive even when projection chunks contain many rows.
    /// </summary>
    internal static ArcBuffer DeltaRowsFused(
        ArcExecutionLane lane, ArcBuffer qkv, ArcBuffer gate, ArcBuffer alpha, ArcBuffer beta,
        ArcBuffer convWeights, ArcBuffer dt, ArcBuffer a, ArcBuffer norm,
        ArcBuffer convState, ArcBuffer recurrentState,
        int keyHeads, int valueHeads, int headWidth, int convKernel, float eps, int rows,
        bool cacheState = true, CancellationToken cancellationToken = default,
        bool? unrollState = null, bool? subgroupRms = null)
    {
        ArgumentNullException.ThrowIfNull(lane);
        cancellationToken.ThrowIfCancellationRequested();
        if (headWidth != 128 || rows < 1 || keyHeads <= 0 || valueHeads <= 0
            || valueHeads % keyHeads != 0 || convKernel <= 0)
            throw new ArgumentException("Invalid batched Gated DeltaNet dimensions.");
        if (!float.IsFinite(eps) || eps <= 0f) throw new ArgumentOutOfRangeException(nameof(eps));
        if (subgroupRms == true && (!cacheState || !lane.Options.Qwen35DeltaSubgroupRms
            || lane.Device.MinimumSubgroupSize != 16
            || !lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups")))
            throw new ArgumentException("The subgroup RMS candidate requires cached SG16 Intel state.", nameof(subgroupRms));
        int valueSize = checked(valueHeads * headWidth);
        int channels = checked((2 * keyHeads + valueHeads) * headWidth);
        CheckDeltaBuffer(qkv, checked(rows * channels), nameof(qkv));
        CheckDeltaBuffer(gate, checked(rows * valueSize), nameof(gate));
        CheckDeltaBuffer(alpha, checked(rows * valueHeads), nameof(alpha));
        CheckDeltaBuffer(beta, checked(rows * valueHeads), nameof(beta));
        CheckDeltaBuffer(convWeights, checked(channels * convKernel), nameof(convWeights));
        CheckDeltaBuffer(dt, valueHeads, nameof(dt));
        CheckDeltaBuffer(a, valueHeads, nameof(a));
        CheckDeltaBuffer(norm, headWidth, nameof(norm));
        CheckDeltaBuffer(convState, checked(channels * (convKernel - 1)), nameof(convState));
        CheckDeltaBuffer(recurrentState, checked(valueSize * headWidth), nameof(recurrentState));
        using ArcBuffer mixed = lane.Allocate(checked(rows * channels));
        ArcBuffer output = lane.Allocate(checked(rows * valueSize));
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            lane.Run("q35d_convolution_rows", channels, 0,
                qkv, convWeights, convState, mixed, channels, convKernel, rows);
            if (lane.Options.Qwen35ParallelDeltaNorm)
                lane.Run("q35d_normalize_qk_rows_coop128", (long)rows * keyHeads * 128, 128,
                    mixed, keyHeads, channels, eps);
            else
                lane.Run("q35d_normalize_qk_rows", (long)rows * keyHeads, 0,
                    mixed, keyHeads, channels, headWidth, eps);
            const int recurrentRowsPerDispatch = 128;
            // SG16 Arc validation retains every output and final state bitwise.
            // Explicit false together with subgroupRms=false keeps the dynamic
            // column cache for same-binary A/B; cacheState=false uses global state.
            bool useUnrolledState = cacheState
                && (unrollState ?? lane.Device.MinimumSubgroupSize == 16);
            bool useSubgroupRms = cacheState && (subgroupRms
                ?? (lane.Options.Qwen35DeltaSubgroupRms && lane.Device.MinimumSubgroupSize == 16
                    && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups")));
            for (int start = 0; start < rows; start += recurrentRowsPerDispatch)
            {
                cancellationToken.ThrowIfCancellationRequested();
                lane.Run(useSubgroupRms ? "q35d_recurrent_gated_rmsnorm_rows128_subgroup"
                    : useUnrolledState ? "q35d_recurrent_gated_rmsnorm_rows128_unrolled"
                    : cacheState ? "q35d_recurrent_gated_rmsnorm_rows128_cached"
                    : "q35d_recurrent_gated_rmsnorm_rows128", valueSize, 128,
                    mixed, alpha, beta, dt, a, recurrentState, norm, gate, output,
                    keyHeads, valueHeads, channels, start,
                    Math.Min(recurrentRowsPerDispatch, rows - start), eps);
            }
            cancellationToken.ThrowIfCancellationRequested();
            return output;
        }
        catch { output.Dispose(); throw; }
    }
}
