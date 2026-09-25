using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

internal sealed partial class ArcXmxStorageOperand
{
    internal const long StreamedPanelBudget = 96L * 1024 * 1024;
    internal readonly record struct StreamedPlan(bool RowTiled, int TileLength, bool ParallelSlices, long PeakPanelBytes);

    /// <summary>
    /// Keep the full-panel predicate unchanged: the loss head manages and reuses
    /// those panels itself. Only Linear opts into this wider streaming predicate.
    /// </summary>
    internal static bool CanRunAny(ArcExecutionLane lane, int m, int n, int k, bool ta = false, bool tb = false,
        bool accumulate = false, bool hasBias = false, bool relu = false)
        => CanRun(lane, m, n, k) || StreamedBackendAvailable(lane)
            && PlanStreamed(m, n, k, ta, tb, accumulate, hasBias, relu, lane.Options.ParallelWeightGradients) is not null;

    private static bool StreamedBackendAvailable(ArcExecutionLane lane)
        => lane.Options.StreamedXmxMatrices && lane.Options.DirectXmxMatrices && lane.Options.XmxMatrices
            && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16;

    // Pure shape planning is also tested without a GPU. Small test budgets force
    // the same boundaries as a production B64 matrix without allocating B64 data.
    internal static StreamedPlan? PlanStreamed(int m, int n, int k, bool ta, bool tb, bool accumulate,
        bool hasBias, bool relu, bool parallelWeightGradients, long panelBudget = StreamedPanelBudget,
        long partialBudget = 64L * 1024 * 1024)
    {
        if (m < 128 || n < 64 || k < 64 || panelBudget <= 0 || panelBudget > StreamedPanelBudget
            || (long)m * k > int.MaxValue || (long)n * k > int.MaxValue || (long)m * n > int.MaxValue) return null;
        long paddedM = ((m + 7L) / 8) * 8, paddedN = ((n + 15L) / 16) * 16;
        long paddedK = ((k + 15L) / 16) * 16;
        if (!ta)
        {
            long rightBytes = paddedN * paddedK * 2;
            // A contiguous row slice has no extra decode, copy or output buffer.
            long availableRows = (panelBudget - rightBytes) / (paddedK * 2);
            int rows = (int)Math.Min(16384L, Math.Max(0, availableRows / 256 * 256));
            if (rows < 256) return null;
            rows = Math.Min(rows, m);
            long leftBytes = ((rows + 7L) / 8) * 8 * paddedK * 2;
            return new(true, rows, false, rightBytes + leftBytes);
        }
        // The only transposed streaming case is dW = dY^T * X. Both source
        // slices are contiguous in token order, so their BFP8 scale origins stay
        // intact. Other transpose/epilogue combinations keep the existing path.
        if (tb || !accumulate || hasBias || relu) return null;
        long availableK = panelBudget / (2 * (paddedM + paddedN));
        int reduction = (int)Math.Min(16384L, availableK / 2048 * 2048);
        if (reduction < 2048) return null;
        reduction = Math.Min(reduction, k);
        int slices = (int)((k + 2047L) / 2048);
        bool parallel = parallelWeightGradients && k >= 4096
            && (long)m * n * slices * 4 <= partialBudget;
        long bytes = ((reduction + 15L) / 16) * 16 * 2 * (paddedM + paddedN);
        return new(false, reduction, parallel, bytes);
    }

    internal static bool TryGemmStreamed(ArcExecutionLane lane, ArcXmxStorageOperand a, ArcXmxStorageOperand b,
        ArcBuffer output, int m, int n, int k, bool ta = false, bool tb = false, bool accumulate = false,
        ArcBuffer? bias = null, bool relu = false, ArcBuffer? gate = null, int gateOperand = 0, int gateOffset = 0,
        long panelBudget = StreamedPanelBudget)
    {
        if (!ReferenceEquals(lane, a.Lane) || !ReferenceEquals(lane, b.Lane))
            throw new ArgumentException("Matrix operands must belong to the executing Arc lane.");
        if (a.Numel != (long)m * k || b.Numel != (long)n * k) throw new ArgumentException("Matrix operands do not match GEMM dimensions.");
        if (gateOperand is < 0 or > 2 || (gateOperand != 0 && gate is null)) throw new ArgumentException("Invalid matrix gate.");
        ArgumentOutOfRangeException.ThrowIfNegative(gateOffset);
        if (!StreamedBackendAvailable(lane)) return false;
        StreamedPlan? selected = PlanStreamed(m, n, k, ta, tb, accumulate, bias is not null, relu,
            lane.Options.ParallelWeightGradients, panelBudget, lane.Options.StreamedWeightGradientWorkspaceMiB * 1048576L);
        if (selected is not { } plan) return false;
        if (plan.RowTiled)
        {
            using var packedB = b.PackB(n, k, tb, gateOperand == 2 ? gate : null, gateOffset);
            for (int first = 0; first < m; first += plan.TileLength)
            {
                int count = Math.Min(plan.TileLength, m - first);
                using var chunk = a.Slice(checked(first * k), checked(count * k));
                using var packedA = chunk.PackA(count, k, false, gateOperand == 1 ? gate : null,
                    gateOperand == 1 ? checked(gateOffset + first * k) : 0);
                RunStreamedPanels(lane, packedA, packedB, output, count, n, k, false, tb,
                    accumulate, bias, relu, checked(first * n), parallel: false);
            }
            return true;
        }
        int globalSlices = (int)((k + 2047L) / 2048);
        using var partials = plan.ParallelSlices ? lane.Allocate(checked(m * n * globalSlices)) : null;
        for (int first = 0; first < k; first += plan.TileLength)
        {
            int count = Math.Min(plan.TileLength, k - first);
            using var left = a.Slice(checked(first * m), checked(count * m));
            using var right = b.Slice(checked(first * n), checked(count * n));
            using var packedA = left.PackA(m, count, true, gateOperand == 1 ? gate : null,
                gateOperand == 1 ? checked(gateOffset + first * m) : 0);
            using var packedB = right.PackB(n, count, false, gateOperand == 2 ? gate : null,
                gateOperand == 2 ? checked(gateOffset + first * n) : 0);
            // Derive split policy from the original full K, never a chunk's K.
            // Otherwise smaller chunks can accidentally enable split reduction
            // and change accumulation order on matrices exceeding the selected
            // full-K scratch budget. No chunk gets an independent split policy.
            RunStreamedPanels(lane, packedA, packedB, partials ?? output, m, n, count, true, false,
                !plan.ParallelSlices, null, false, plan.ParallelSlices ? checked((first / 2048) * m * n) : 0,
                plan.ParallelSlices);
        }
        if (partials is not null)
            lane.Run("gemm_split_finish", checked(m * n), 0, partials, output, checked(m * n), globalSlices);
        return true;
    }

    private static void RunStreamedPanels(ArcExecutionLane lane, ArcBuffer packedA, ArcBuffer packedB,
        ArcBuffer output, int m, int n, int k, bool ta, bool tb, bool accumulate, ArcBuffer? bias,
        bool relu, int outputOffset, bool parallel)
    {
        bool longReduction = k >= 4096 && !ta;
        string kernel = longReduction ? "gemm_xmx_streamed_block_8x32_wg16" : "gemm_xmx_streamed_block_16x32_wg16";
        int tileRows = longReduction ? 128 : 256;
        int tileColumns = 32;
        if (lane.Options.ExpandedStreamedXmxTiles && m >= 512 && n >= 512 && k >= 512)
        {
            if (ta && !tb && (accumulate || parallel) && bias is null && !relu)
            {
                kernel = "gemm_xmx_streamed_block_32x32_wg16";
                tileRows = 512;
            }
            else if (!ta && !longReduction && accumulate && bias is null && !relu && m >= 4096)
            {
                kernel = "gemm_xmx_streamed_block_16x64_wg16";
                tileColumns = 64;
            }
        }
        long gx = ((n + tileColumns - 1L) / tileColumns) * 16, gy = ((m + tileRows - 1L) / tileRows) * 16;
        if (parallel)
        {
            lane.Run3D(kernel, gx, gy, (k + 2047L) / 2048, 16, 16, 1, packedA, packedB, output, packedA,
                m, n, k, 0, 0, 0, 0, 2048, outputOffset);
            return;
        }
        for (int start = 0; start < k; start += 2048)
            lane.Run2D(kernel, gx, gy, 16, 16, packedA, packedB, output, bias ?? packedA,
                m, n, k, accumulate || start != 0 ? 1 : 0, bias is not null && start == 0 ? 1 : 0,
                relu && start + 2048 >= k ? 1 : 0, start, Math.Min(2048, k - start), outputOffset);
    }
}
