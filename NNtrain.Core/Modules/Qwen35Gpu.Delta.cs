using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

internal static partial class Qwen35Gpu
{
    /// <summary>
    /// Runs one Gated DeltaNet token entirely in device memory. Convolution state is
    /// [channel, oldest..newest]; recurrent state is [value head, key, value].
    /// Inputs and persistent states remain caller-owned; the returned buffer is owned.
    /// </summary>
    internal static ArcBuffer DeltaStep(
        ArcExecutionLane lane, ArcBuffer qkv, ArcBuffer gate, ArcBuffer alpha, ArcBuffer beta,
        ArcBuffer convWeights, ArcBuffer dt, ArcBuffer a, ArcBuffer norm,
        ArcBuffer convState, ArcBuffer recurrentState,
        int keyHeads, int valueHeads, int headWidth, int convKernel, float eps)
    {
        ArgumentNullException.ThrowIfNull(lane);
        if (keyHeads <= 0 || valueHeads <= 0 || valueHeads % keyHeads != 0
            || headWidth <= 0 || convKernel <= 0)
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
        using ArcBuffer unnormalized = lane.Allocate(valueSize);
        ArcBuffer output = lane.Allocate(valueSize);
        try
        {
            lane.Run("q35d_convolution", channels, 0,
                qkv, convWeights, convState, mixed, channels, convKernel);
            lane.Run("q35d_normalize_qk", keyHeads, 0, mixed, keyHeads, headWidth, eps);
            lane.Run("q35d_recurrent", valueSize, 0,
                mixed, alpha, beta, dt, a, recurrentState, unnormalized,
                keyHeads, valueHeads, headWidth);
            lane.Run("q35d_gated_rmsnorm", valueHeads, 0,
                unnormalized, norm, gate, output, valueHeads, headWidth, eps);
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }

    private static void CheckDeltaBuffer(ArcBuffer buffer, int elements, string name)
    {
        ArgumentNullException.ThrowIfNull(buffer, name);
        if (!buffer.IsAlive || buffer.ByteLength < checked((long)elements * sizeof(float)))
            throw new ArgumentException("Gated DeltaNet buffer is disposed or smaller than its declared shape.", name);
    }
}
