using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

internal static class ArcMuonMath
{
    internal static (string Kernel, int Rows, int Columns) SelectXmxPlan(
        ArcXmxGemmMode mode, int subgroupSize, int m, int n, bool ta, bool tb)
    {
        // Measurements are SG16-specific. Other devices retain their old path.
        if (subgroupSize != 16 || mode == ArcXmxGemmMode.Legacy)
            return ("gemm_xmx", 128, 64);
        if (mode == ArcXmxGemmMode.Auto)
        {
            if (m < 128 || n < 64) return ("gemm_xmx", 128, 64);
            // Wide tiles halve repeated global loads on large output grids.
            // Small/long-K grids (loss dX, NS) need more independent workgroups.
            mode = m >= 256 && n >= 128 && (long)m * n >= 512 * 1024
                ? ArcXmxGemmMode.Wide : ArcXmxGemmMode.Narrow;
        }
        bool wide = mode == ArcXmxGemmMode.Wide;
        string name = (wide, ta, tb) switch {
            (true, false, false) => "gemm_xmx_wide_nn",
            (true, false, true) => "gemm_xmx_wide_nt",
            (true, true, false) => "gemm_xmx_wide_tn",
            (true, true, true) => "gemm_xmx_wide_tt",
            (false, false, false) => "gemm_xmx_narrow_nn",
            (false, false, true) => "gemm_xmx_narrow_nt",
            (false, true, false) => "gemm_xmx_narrow_tn",
            _ => "gemm_xmx_narrow_tt",
        };
        return (name, wide ? 256 : 128, wide ? 128 : 64);
    }

    internal static void Gemm(ArcExecutionLane lane, ArcBuffer a, ArcBuffer b, ArcBuffer output,
        int m, int n, int k, bool ta = false, bool tb = false, int bf16 = 0,
        bool accumulate = false, ArcBuffer? bias = null, bool relu = false, ArcBuffer? gate = null, int gateOperand = 0)
    {
        // Small dW output grids otherwise serialize long reductions in several
        // dispatches. Parallel slices use bounded device scratch and no atomics.
        int slices = (k + 2047) / 2048;
        bool xmx = bf16 == 3 && lane.Options.XmxMatrices && lane.Device.SupportsXmx;
        if (xmx && ArcDirectXmxMath.TryGemm(lane, a, b, output, m, n, k, ta, tb, accumulate, bias, relu, gate, gateOperand))
            return;
        var (xmxKernel, xmxRows, xmxColumns) = SelectXmxPlan(lane.Options.XmxGemmMode,
            lane.Device.MinimumSubgroupSize, m, n, ta, tb);
        if (lane.Options.ParallelWeightGradients && ta && !tb && (bf16 == 0 || xmx) && accumulate
            && bias is null && !relu && k >= 4096 && m >= 64 && n >= 64
            && (long)m * n * slices * 4 <= 64 * 1024 * 1024)
        {
            using var partials = lane.Allocate(checked(m * n * slices));
            int sg = xmx ? lane.Device.MinimumSubgroupSize : 16;
            int localRows = xmx ? xmxRows / 8 : 16;
            lane.Run3D(xmx ? xmxKernel : "gemm_split",
                ((n + (xmx ? xmxColumns : 64) - 1L) / (xmx ? xmxColumns : 64)) * sg,
                ((m + (xmx ? xmxRows : 64) - 1L) / (xmx ? xmxRows : 64)) * localRows, slices, sg, localRows, 1,
                a, b, partials, a, gate ?? a, m, n, k, 1, 0, bf16, 0, 0, 0, gateOperand, 0, 2048);
            lane.Run("gemm_split_finish", checked(m * n), 0, partials, output, checked(m * n), slices);
            return;
        }
        bool wide = lane.Options.WideMatrices && m >= 64 && n >= 64;
        int tile = wide ? 64 : 32;
        int subgroup = lane.Device.MinimumSubgroupSize;
        // Ordered split-K avoids long single dispatches and keeps partial sums on-device.
        for (int start = 0; start < k; start += 2048)
            lane.Run2D(xmx ? xmxKernel : wide ? (lane.Options.DeepMatrices && m >= 1024 && (!ta || k <= 2048) ? "gemm_deep" : "gemm_wide") : "gemm_tiled",
                xmx ? ((n + xmxColumns - 1L) / xmxColumns) * subgroup : ((n + tile - 1L) / tile) * 16,
                xmx ? ((m + xmxRows - 1L) / xmxRows) * (xmxRows / 8) : ((m + tile - 1L) / tile) * 16, xmx ? subgroup : 16, xmx ? xmxRows / 8 : 16,
                a, b, output, bias ?? a, gate ?? a, m, n, k, ta ? 1 : 0, tb ? 1 : 0, bf16,
                accumulate || start != 0 ? 1 : 0, bias is not null && start == 0 ? 1 : 0,
                relu && start + 2048 >= k ? 1 : 0, gateOperand, start, Math.Min(2048, k - start));
    }

    internal static void Orthogonalize(float[] x, int rows, int columns, int whole, float fraction,
        bool bf16, float a, float b, float c)
    {
        var lane = Tensor.ArcLane;
        using var first = lane.Upload(x);
        using var second = lane.Allocate(x.Length);
        using var gram = lane.Allocate(checked(rows * rows));
        using var square = lane.Allocate(checked(rows * rows));
        using var coefficient = lane.Allocate(checked(rows * rows));
        ArcBuffer current = first, next = second;
        void Iterate()
        {
            Gemm(lane, current, current, gram, rows, rows, columns, tb: true, bf16: bf16 ? 3 : 0);
            Gemm(lane, gram, gram, square, rows, rows, rows, tb: true, bf16: bf16 ? 3 : 0);
            lane.Run("ns_coefficient", rows * rows, 0, gram, square, coefficient, rows, a, b, c);
            Gemm(lane, coefficient, current, next, rows, columns, rows, bf16: bf16 ? 3 : 0);
        }
        for (int iteration = 0; iteration < whole; iteration++) { Iterate(); (current, next) = (next, current); }
        if (fraction > 0)
        {
            Iterate();
            lane.Run("axpby", x.Length, 0, current, next, current, x.Length, 1f - fraction, fraction);
        }
        lane.Read(current, x);
    }

    internal static float Confidence(ArcExecutionLane lane, ArcBuffer fast, ArcBuffer slow, int length, float epsilon)
    {
        int groups = (length + 255) / 256;
        using var partials = lane.Allocate(groups * 4);
        float[] statistics = new float[4];
        lane.Run("confidence_blocks", groups * 256L, 256, fast, slow, partials, length, new LocalMemory(4096));
        lane.Run("confidence_finish", 4, 0, partials, Out(statistics), groups);
        double alignment = Math.Max(0, statistics[0] / (Math.Sqrt(statistics[1]) * Math.Sqrt(statistics[2]) + epsilon));
        double persistence = statistics[2] / ((double)statistics[2] + statistics[3] + epsilon);
        return (float)Math.Clamp(alignment * persistence, 0, 1);
    }

    internal static float SumSquares(ArcExecutionLane lane, ArcBuffer input, int length)
    {
        var leases = new List<ArcBuffer>();
        try
        {
            ArcBuffer current = input;
            bool square = true;
            while (true)
            {
                int groups = (length + 255) / 256;
                var next = lane.Allocate(groups); leases.Add(next);
                lane.Run("reduce_sum", groups * 256L, 256, current, next, length, square ? 1 : 0, new LocalMemory(1024));
                if (groups == 1) { float[] scalar = new float[1]; lane.Read(next, scalar); return scalar[0]; }
                current = next; length = groups; square = false;
            }
        }
        finally { foreach (var lease in leases) lease.Dispose(); }
    }

    internal static float Confidence(float[] fast, float[] slow, float epsilon)
    {
        var lane = Tensor.ArcLane;
        int groups = (fast.Length + 255) / 256;
        using var partials = lane.Allocate(groups * 4);
        float[] statistics = new float[4];
        lane.Run("confidence_blocks", groups * 256L, 256, In(fast), In(slow), partials, fast.Length, new LocalMemory(4096));
        lane.Run("confidence_finish", 4, 0, partials, Out(statistics), groups);
        double alignment = Math.Max(0, statistics[0] / (Math.Sqrt(statistics[1]) * Math.Sqrt(statistics[2]) + epsilon));
        double persistence = statistics[2] / ((double)statistics[2] + statistics[3] + epsilon);
        return (float)Math.Clamp(alignment * persistence, 0, 1);
    }
}
