using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    private bool CanUseArcFlashAttention(int sequence, int width, int heads) =>
        ArcLane.Options.FlashAttention && ArcLane.Options.XmxMatrices && ArcLane.Device.SupportsXmx
        && ArcLane.Device.MinimumSubgroupSize == 16 && DType != TensorDType.Float32
        && sequence >= 32 && width / heads is > 0 and <= 64;

    internal Tensor ArcFlashAttention(int batch, int sequence, int width, int heads, bool causal, TensorDType? resultDType = null)
    {
        var lane = ArcLane;
        int d = width / heads, depth = d <= 16 ? 16 : d <= 32 ? 32 : 64;
        int padded = checked((sequence + 31) / 32 * 32), totalHeads = checked(batch * heads);
        int length = checked(batch * sequence * width), panel = checked(totalHeads * padded * depth);
        int xmx = lane.Options.FlashAttentionXmxProducts ? 1 : 0;
        int asyncCopy = xmx != 0 && lane.Options.FlashAttentionAsyncCopy ? 1 : 0;
        string suffix = $"_d{depth}_x{xmx}_a{asyncCopy}";
        ArcBuffer Pack()
        {
            using var values = ArcUploadValues(true);
            var packed = lane.AllocateBytes(checked(panel * 3 * 2));
            try
            {
                lane.Run("attention_flash_pack", panel * 3L, 256, values, packed,
                    sequence, width, heads, padded, depth, checked(panel * 3));
                return packed;
            }
            catch { packed.Dispose(); throw; }
        }
        void Forward(ArcBuffer packed, ArcBuffer output, ArcBuffer stats) =>
            lane.Run2D("attention_flash_f" + suffix, padded / 32L * 64, totalHeads, 64, 1,
                packed, output, stats, sequence, width, heads, padded, d, causal ? 1 : 0);
        Tensor result;
        using (var packed = Pack())
        using (var output = lane.Allocate(length))
        using (var stats = lane.Allocate(checked(totalHeads * sequence * 2)))
        {
            Forward(packed, output, stats);
            result = ArcDeviceResult(output, Rank == 3 ? [batch, sequence, width] : [sequence, width], [this], resultDType);
        }
        // Recompute raw output rather than retaining an FP32 activation for
        // every layer, or incorrectly using the quantized published output.
        result.Node.BackwardAction = () => {
            using var packed = Pack();
            using var stats = lane.Allocate(checked(totalHeads * sequence * 2));
            using var delta = lane.Allocate(checked(totalHeads * sequence));
            using var dyPacked = lane.AllocateBytes(checked(panel * 2 * 2));
            ArcBuffer dy = result.ArcGradient();
            using (var raw = lane.Allocate(length))
            {
                Forward(packed, raw, stats);
                lane.Run("attention_flash_delta", totalHeads * (long)sequence, 256,
                    dy, raw, delta, sequence, width, heads, checked(totalHeads * sequence));
            }
            lane.Run("attention_flash_pack_dy", panel * 2L, 256, dy, dyPacked,
                sequence, width, heads, padded, depth, checked(panel * 2));
            ArcBuffer dx = ArcGradient();
            for (int keyOwner = 0; keyOwner <= 1; keyOwner++)
                lane.Run2D((keyOwner == 0 ? "attention_flash_q" : "attention_flash_kv") + suffix, padded / 32L * 64, totalHeads, 64, 1,
                    packed, dyPacked, stats, delta, dx, sequence, width, heads, padded, d, causal ? 1 : 0);
        };
        return result;
    }
}
