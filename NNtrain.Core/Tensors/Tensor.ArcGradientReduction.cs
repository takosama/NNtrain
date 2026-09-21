using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    private static bool ArcUsesMixedMatrixOperands => ArcLane.Options.MixedBackwardMatrixOperands
        && TensorExecutionContext.ActivePrecisionPolicy?.Mode is
            NNtrain.Runtime.Execution.PrecisionMode.Mix16_32 or NNtrain.Runtime.Execution.PrecisionMode.Mix8_32;

    private static ArcBuffer ArcEncodeMatrixGradient(ArcBuffer source, ArcBuffer? gate, int length, bool mixed)
    {
        if (!mixed) return source.Borrow();
        var encoded = ArcLane.Allocate(length);
        try { ArcLane.Run("matrix_gradient_bf16", length, 0, source, gate ?? source, encoded, length, gate is null ? 0 : 1); return encoded; }
        catch { encoded.Dispose(); throw; }
    }

    private static void ArcBiasGradient(ArcBuffer dy, ArcBuffer? gate, ArcBuffer db, int rows, int width, bool roundBf16 = false)
    {
        var lane = ArcLane;
        if (!lane.Options.ParallelReductions || rows < 512)
        {
            for (int start = 0; start < rows; start += 2048)
                lane.Run(roundBf16 ? "linear_db_bf16_chunk" : "linear_db_chunk", width, 0, dy, gate ?? dy, db, rows, width, gate is null ? 0 : 1, start, Math.Min(2048, rows - start));
            return;
        }
        int groups = (rows + 255) / 256;
        using var partials = lane.Allocate(checked(groups * width));
        lane.Run2D("gradient_rows", ((width + 31L) / 32) * 32, groups * 8L, 32, 8,
            dy, gate ?? dy, dy, partials, rows, width, gate is null ? 0 : 1, roundBf16 ? 2 : 0);
        lane.Run("gradient_rows_finish", width, 0, partials, db, db, groups, width, 0);
    }

    private static void ArcNormGradient(ArcBuffer x, ArcBuffer dy, ArcBuffer stats, ArcBuffer dg, ArcBuffer db, int rows, int width)
    {
        var lane = ArcLane;
        if (!lane.Options.ParallelReductions || rows < 512)
        {
            lane.Run("norm_dw", width, 0, x, dy, stats, dg, db, rows, width);
            return;
        }
        int groups = (rows + 255) / 256;
        using var partials = lane.Allocate(checked(groups * width * 2));
        lane.Run2D("gradient_rows", ((width + 31L) / 32) * 32, groups * 8L, 32, 8,
            dy, x, stats, partials, rows, width, 0, 1);
        lane.Run("gradient_rows_finish", width, 0, partials, db, dg, groups, width, 1);
    }
}
