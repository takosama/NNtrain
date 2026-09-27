using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

internal static partial class Qwen35Gpu
{
    /// <summary>
    /// Combines recurrent update and gated RMSNorm for width-128 heads.
    /// Other widths use the generic implementation. All inputs remain caller-owned.
    /// </summary>
    internal static ArcBuffer DeltaStepFused(
        ArcExecutionLane lane, ArcBuffer qkv, ArcBuffer gate, ArcBuffer alpha, ArcBuffer beta,
        ArcBuffer convWeights, ArcBuffer dt, ArcBuffer a, ArcBuffer norm,
        ArcBuffer convState, ArcBuffer recurrentState,
        int keyHeads, int valueHeads, int headWidth, int convKernel, float eps)
    {
        if (headWidth != 128)
            return DeltaStep(lane, qkv, gate, alpha, beta, convWeights, dt, a, norm,
                convState, recurrentState, keyHeads, valueHeads, headWidth, convKernel, eps);
        ArgumentNullException.ThrowIfNull(lane);
        if (keyHeads <= 0 || valueHeads <= 0 || valueHeads % keyHeads != 0 || convKernel <= 0)
            throw new ArgumentException("Invalid Gated DeltaNet dimensions.");
        if (!float.IsFinite(eps) || eps <= 0f) throw new ArgumentOutOfRangeException(nameof(eps));
        int keySize = checked(keyHeads * headWidth);
        int valueSize = checked(valueHeads * headWidth);
        int channels = checked(2 * keySize + valueSize);
        CheckDeltaBuffer(qkv, channels, nameof(qkv));
        CheckDeltaBuffer(gate, valueSize, nameof(gate));
        CheckDeltaBuffer(alpha, valueHeads, nameof(alpha));
        CheckDeltaBuffer(beta, valueHeads, nameof(beta));
        CheckDeltaBuffer(convWeights, checked(channels * convKernel), nameof(convWeights));
        CheckDeltaBuffer(dt, valueHeads, nameof(dt));
        CheckDeltaBuffer(a, valueHeads, nameof(a));
        CheckDeltaBuffer(norm, headWidth, nameof(norm));
        CheckDeltaBuffer(convState, checked(channels * (convKernel - 1)), nameof(convState));
        CheckDeltaBuffer(recurrentState, checked(valueSize * headWidth), nameof(recurrentState));

        using ArcBuffer mixed = lane.Allocate(channels);
        ArcBuffer output = lane.Allocate(valueSize);
        try
        {
            lane.Run("q35d_convolution", channels, 0,
                qkv, convWeights, convState, mixed, channels, convKernel);
            if (lane.Options.Qwen35ParallelDeltaNorm)
                lane.Run("q35d_normalize_qk_coop128", (long)keyHeads * 128, 128, mixed, keyHeads, eps);
            else
                lane.Run("q35d_normalize_qk", keyHeads, 0, mixed, keyHeads, headWidth, eps);
            lane.Run("q35d_recurrent_gated_rmsnorm_fused128", valueSize, 128,
                mixed, alpha, beta, dt, a, recurrentState, norm, gate, output, keyHeads, eps);
            return output;
        }
        catch { output.Dispose(); throw; }
    }
}
