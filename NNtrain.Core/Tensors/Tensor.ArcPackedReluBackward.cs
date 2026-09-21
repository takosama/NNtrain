using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    private bool TryArcPackedReluBackward(Tensor weight, Tensor result, ArcBuffer dy, ArcBuffer dx, ArcBuffer dw, ArcBuffer db)
    {
        var lane = ArcLane;
        int k = _shape[^1], m = Numel / k, n = weight._shape[0];
        if (!lane.Options.PackedReluBackward || !lane.Options.PackedMatrixStorage
            || result.DType is not (TensorDType.Bfp8 or TensorDType.BFloat16)
            || !ArcXmxStorageOperand.CanRunAny(lane, m, k, n, accumulate: true)
            || !ArcXmxStorageOperand.CanRunAny(lane, n, k, m, ta: true, accumulate: true)) return false;
        using var gate = result.ArcMatrixOperand();
        using var encoded = lane.AllocateBytes(checked(m * n * 2));
        lane.Run("linear_relu_grad_packed", checked(m * n), 0, dy, gate.Value, gate.Scales ?? gate.Value,
            encoded, checked(m * n), gate.BlockSize, gate.DType == TensorDType.Bfp8 ? 1 : 0);
        using var gradient = new ArcXmxStorageOperand(lane, encoded, null, TensorDType.BFloat16, checked(m * n));
        using var input = ArcMatrixOperand(); using var weights = weight.ArcMatrixOperand(cacheWeightPanels: true);
        ArcXmxStorageOperand.TryGemm(lane, gradient, weights, dx, m, k, n, accumulate: true);
        ArcXmxStorageOperand.TryGemm(lane, gradient, input, dw, n, k, m, ta: true, accumulate: true);
        if (!lane.Options.ParallelReductions || m < 512)
        {
            for (int start = 0; start < m; start += 2048)
                lane.Run("linear_db_packed_bf16_chunk", n, 0, encoded, db, m, n, start, Math.Min(2048, m - start));
        }
        else
        {
            int groups = (m + 255) / 256;
            using var partials = lane.Allocate(checked(groups * n));
            lane.Run2D("gradient_rows_packed_bf16", ((n + 31L) / 32) * 32, groups * 8L, 32, 8, encoded, partials, m, n);
            lane.Run("gradient_rows_finish", n, 0, partials, db, db, groups, n, 0);
        }
        return true;
    }
}
