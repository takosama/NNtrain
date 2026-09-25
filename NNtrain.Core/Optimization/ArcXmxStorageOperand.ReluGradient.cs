using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

internal sealed partial class ArcXmxStorageOperand
{
    /// <summary>
    /// Pack one gated gradient into both full XMX A-panel orientations and
    /// produce the usual 256-row bias partials. The caller owns both panels.
    /// </summary>
    internal static (ArcBuffer Normal, ArcBuffer Transposed) PackReluDualGradientA(
        ArcExecutionLane lane, ArcBuffer dy, ArcXmxStorageOperand gate,
        ArcBuffer biasPartials, int rows, int cols)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
        if (!ReferenceEquals(lane, gate.Lane)
            || gate.DType is not (TensorDType.BFloat16 or TensorDType.Bfp8)
            || gate.Offset != 0 || gate.Numel != (long)rows * cols)
            throw new ArgumentException("Invalid resident ReLU gate operand.", nameof(gate));

        int normalElements = checked((int)(((rows + 7L) / 8) * ((cols + 15L) / 16) * 128));
        int transposedElements = checked((int)(((cols + 7L) / 8) * ((rows + 15L) / 16) * 128));
        ArcBuffer normal = lane.AllocateBytes(checked(normalElements * sizeof(ushort)));
        try
        {
            ArcBuffer transposed = lane.AllocateBytes(checked(transposedElements * sizeof(ushort)));
            try
            {
                lane.Run2D("linear_relu_grad_pack_a_bias_dual",
                    ((cols + 31L) / 32) * 32, ((rows + 255L) / 256) * 8,
                    32, 8, dy, gate.Value, gate.Scales ?? gate.Value,
                    normal, transposed, biasPartials, rows, cols, gate.BlockSize,
                    gate.DType == TensorDType.Bfp8 ? 1 : 0);
                return (normal, transposed);
            }
            catch { transposed.Dispose(); throw; }
        }
        catch { normal.Dispose(); throw; }
    }

    // ReLU's published gate is native BF16/BFP8. The usual path first writes
    // an entire row-major BF16 gradient, then packs that image separately for
    // dX and dW. These packs instead reproduce the same gate and BF16 round
    // at the panel boundary. No other GEMM operand changes its storage policy.
    internal static void GemmReluGradient(ArcExecutionLane lane, ArcBuffer dy,
        ArcXmxStorageOperand gate, ArcXmxStorageOperand right, ArcBuffer output,
        int m, int n, int k, bool transpose, bool accumulate,
        long panelBudget = StreamedPanelBudget, ArcBuffer? biasPartials = null)
    {
        if (!ReferenceEquals(lane, gate.Lane) || !ReferenceEquals(lane, right.Lane)
            || gate.DType is not (TensorDType.BFloat16 or TensorDType.Bfp8)
            || gate.Offset != 0 || gate.Numel != (long)m * k || right.Numel != (long)n * k)
            throw new ArgumentException("Invalid resident ReLU gradient operands.");
        if (!CanRunAny(lane, m, n, k, ta: transpose, accumulate: accumulate))
            throw new ArgumentException("The ReLU gradient GEMM shape is unsupported.");
        if (panelBudget is <= 0 or > StreamedPanelBudget)
            throw new ArgumentOutOfRangeException(nameof(panelBudget));
        if (transpose && biasPartials is not null)
            throw new ArgumentException("Only non-transposed ReLU packing can produce bias partials.");

        long fullBytes = ((m + 7L) / 8) * ((k + 15L) / 16) * 128 * 2
            + ((n + 15L) / 16) * ((k + 15L) / 16) * 128 * 4;
        if (CanRun(lane, m, n, k) && fullBytes <= panelBudget)
        {
            using var packedA = PackReluGradientA(lane, dy, gate, m, k, transpose, 0,
                biasPartials);
            using var packedB = right.PackB(n, k, false);
            GemmPanels(lane, packedA, packedB, output, m, n, k,
                ta: transpose, accumulate: accumulate);
            return;
        }

        StreamedPlan plan = PlanStreamed(m, n, k, transpose, false, accumulate, false, false,
            lane.Options.ParallelWeightGradients, panelBudget,
            lane.Options.StreamedWeightGradientWorkspaceMiB * 1048576L)
            ?? throw new InvalidOperationException("ReLU gradient streamed GEMM plan disappeared.");
        if (plan.RowTiled)
        {
            using var packedB = right.PackB(n, k, false);
            for (int first = 0; first < m; first += plan.TileLength)
            {
                int count = Math.Min(plan.TileLength, m - first);
                using var packedA = PackReluGradientA(lane, dy, gate, count, k, false,
                    checked(first * k), biasPartials, first / 256);
                RunStreamedPanels(lane, packedA, packedB, output, count, n, k, false, false,
                    accumulate, null, false, checked(first * n), parallel: false);
            }
            return;
        }

        int globalSlices = (int)((k + 2047L) / 2048);
        using var partials = plan.ParallelSlices ? lane.Allocate(checked(m * n * globalSlices)) : null;
        for (int first = 0; first < k; first += plan.TileLength)
        {
            int count = Math.Min(plan.TileLength, k - first);
            using var packedA = PackReluGradientA(lane, dy, gate, m, count, true,
                checked(first * m));
            using var rightSlice = right.Slice(checked(first * n), checked(count * n));
            using var packedB = rightSlice.PackB(n, count, false);
            RunStreamedPanels(lane, packedA, packedB, partials ?? output, m, n, count,
                true, false, !plan.ParallelSlices, null, false,
                plan.ParallelSlices ? checked((first / 2048) * m * n) : 0,
                plan.ParallelSlices);
        }
        if (partials is not null)
            lane.Run("gemm_split_finish", checked(m * n), 0, partials, output,
                checked(m * n), globalSlices);
    }

    private static ArcBuffer PackReluGradientA(ArcExecutionLane lane, ArcBuffer dy,
        ArcXmxStorageOperand gate, int outer, int reduction, bool transpose, int sourceOffset,
        ArcBuffer? biasPartials = null, int biasGroupOffset = 0)
    {
        int panels = checked((int)(((outer + 7L) / 8) * ((reduction + 15L) / 16) * 128));
        ArcBuffer result = lane.AllocateBytes(checked(panels * 2));
        try
        {
            if (biasPartials is not null)
            {
                if (transpose || sourceOffset % (256L * reduction) != 0)
                    throw new ArgumentException("Bias partials require a 256-row-aligned source tile.");
                lane.Run2D("linear_relu_grad_pack_a_bias",
                    ((reduction + 31L) / 32) * 32, ((outer + 255L) / 256) * 8,
                    32, 8, dy, gate.Value, gate.Scales ?? gate.Value, result,
                    biasPartials, outer, reduction, sourceOffset, gate.BlockSize,
                    gate.DType == TensorDType.Bfp8 ? 1 : 0, biasGroupOffset);
                return result;
            }
            object[] args = [dy, gate.Value, gate.Scales ?? gate.Value, result,
                outer, reduction, sourceOffset, gate.BlockSize,
                gate.DType == TensorDType.Bfp8 ? 1 : 0];
            if (transpose)
                lane.Run2D("linear_relu_grad_pack_a_transpose",
                    ((outer + 31L) / 32) * 256, (reduction + 31L) / 32,
                    256, 1, args);
            else
                lane.Run("linear_relu_grad_pack_a_vec4", panels / 4, 0, args);
            return result;
        }
        catch { result.Dispose(); throw; }
    }
}
