using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

internal static class ArcDirectXmxMath
{
    internal static bool TryGemm(ArcExecutionLane lane, ArcBuffer a, ArcBuffer b, ArcBuffer output,
        int m, int n, int k, bool ta, bool tb, bool accumulate, ArcBuffer? bias, bool relu,
        ArcBuffer? gate, int gateOperand)
    {
        if (!lane.Options.DirectXmxMatrices || lane.Device.MinimumSubgroupSize != 16
            || m < 128 || n < 64 || k < 64) return false;
        long aCount = ((m + 7L) / 8) * ((k + 15L) / 16) * 128;
        long bCount = ((n + 15L) / 16) * ((k + 15L) / 16) * 128;
        // Bound device-only BF16 panels; oversized/unmeasured shapes retain SLM GEMM.
        if (aCount * 2 + bCount * 4 > 96L * 1024 * 1024) return false;
        using var packedA = lane.AllocateBytes(checked((int)aCount * 2));
        using var packedB = lane.Allocate(checked((int)bCount));
        if (ta)
            lane.Run2D("xmx_pack_tune_a_transpose", ((m + 31L) / 32) * 256, (k + 31L) / 32, 256, 1,
                a, gate ?? a, packedA, m, k, 1, gateOperand == 1 ? 1 : 0);
        else lane.Run("xmx_storage_pack_a_linear_vec4_f32", aCount / 4, 0, a, a, gate ?? a, packedA,
            m, k, 0, 1, gateOperand == 1 ? 1 : 0, 0, 0);
        if (tb)
            lane.Run2D("xmx_pack_tune_b_transpose", ((n + 31L) / 32) * 256, (k + 31L) / 32, 256, 1,
                b, gate ?? b, packedB, n, k, 1, gateOperand == 2 ? 1 : 0);
        else lane.Run("xmx_next_pack_b", bCount, 0, b, gate ?? b, packedB, n, k, 0, gateOperand == 2 ? 1 : 0);
        bool longReduction = k >= 4096 && !ta;
        string kernel = longReduction ? "gemm_xmx_direct_block_8x32_wg16" : "gemm_xmx_direct_block_16x32_wg16";
        int tileRows = longReduction ? 128 : 256;
        long gx = ((n + 31L) / 32) * 16, gy = ((m + tileRows - 1L) / tileRows) * 16;
        int slices = (k + 2047) / 2048;
        if (lane.Options.ParallelWeightGradients && ta && !tb && accumulate && bias is null && !relu
            && k >= 4096 && m >= 64 && n >= 64 && (long)m * n * slices * 4 <= 64 * 1024 * 1024)
        {
            using var partials = lane.Allocate(checked(m * n * slices));
            lane.Run3D(kernel, gx, gy, slices, 16, 16, 1, packedA, packedB, partials, a, gate ?? a,
                m, n, k, ta ? 1 : 0, tb ? 1 : 0, 3, 0, 0, 0, gateOperand, 0, 2048);
            lane.Run("gemm_split_finish", checked(m * n), 0, partials, output, checked(m * n), slices);
            return true;
        }
        for (int start = 0; start < k; start += 2048)
            lane.Run2D(kernel, gx, gy, 16, 16, packedA, packedB, output, bias ?? a, gate ?? a,
                m, n, k, ta ? 1 : 0, tb ? 1 : 0, 3, accumulate || start != 0 ? 1 : 0,
                bias is not null && start == 0 ? 1 : 0, relu && start + 2048 >= k ? 1 : 0,
                gateOperand, start, Math.Min(2048, k - start));
        return true;
    }
}
