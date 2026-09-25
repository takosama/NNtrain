using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    // This path uses the same storage decode, dropout mask, FP32 serial row
    // reductions and parameter-gradient reduction as the existing packed path.
    // Recomputing the residual sum inside each consumer removes the full FP32
    // pre-normalization buffer from forward and backward.
    private Tensor ArcFusedPackedResidualNorm(Tensor gamma, Tensor beta, float epsilon,
        Tensor branch, uint seed, uint threshold, float scale, int width, int rows)
    {
        var lane = ArcLane;
        int Format(Tensor tensor) => tensor.DType switch {
            TensorDType.Float32 => 0,
            TensorDType.BFloat16 => 1,
            TensorDType.Bfp8 => 2,
            _ => throw new NotSupportedException("Unsupported Arc normalization storage.")
        };
        int xt = Format(this), bt = Format(branch);
        int xb = Bfp8Quantization?.GetEffectiveBlockSize(Numel) ?? 1;
        int bb = branch.Bfp8Quantization?.GetEffectiveBlockSize(Numel) ?? 1;
        long work = ((rows + 3L) / 4) * 64;
        EnsureArcPacked();
        branch.EnsureArcPacked();
        ArcReplica input = _arcReplica!, residual = branch._arcReplica!;
        using var g = gamma.ArcUploadValues();
        using var b = beta.ArcUploadValues();
        using var output = lane.Allocate(Numel);
        var stats = lane.Allocate(checked(rows * 2));
        try
        {
            lane.Run("norm_packed_residual_row_sg16_w64", work, 64,
                input.Value!, input.Scales ?? input.Value!, residual.Value!, residual.Scales ?? residual.Value!,
                g, b, output, stats, rows, width, epsilon, xt, bt, xb, bb, seed, threshold, scale);
            Tensor result = ArcDeviceResult(output, _shape, [this, branch, gamma, beta]);
            if (result.Node.IsDetached) { stats.Dispose(); return result; }
            result.Node.RegisterResource(stats);
            result.Node.BackwardAction = () =>
            {
                EnsureArcPacked();
                branch.EnsureArcPacked();
                ArcReplica x = _arcReplica!, r = branch._arcReplica!;
                using var gv = gamma.ArcUploadValues();
                ArcBuffer dy = result.ArcGradient();
                lane.Run("norm_packed_residual_dx_row_sg16_w64", work, 64,
                    x.Value!, x.Scales ?? x.Value!, r.Value!, r.Scales ?? r.Value!,
                    gv, dy, stats, ArcGradient(), branch.ArcGradient(),
                    rows, width, xt, bt, xb, bb, seed, threshold, scale);
                int groups = (rows + 255) / 256;
                using var partials = lane.Allocate(checked(groups * width * 2));
                lane.Run2D("norm_packed_residual_parameter_parts",
                    ((width + 31L) / 32) * 32, groups * 8L, 32, 8,
                    dy, x.Value!, x.Scales ?? x.Value!, r.Value!, r.Scales ?? r.Value!,
                    stats, partials, rows, width, xt, bt, xb, bb, seed, threshold, scale);
                lane.Run("gradient_rows_finish", width, 0, partials,
                    beta.ArcGradient(), gamma.ArcGradient(), groups, width, 1);
            };
            return result;
        }
        catch { stats.Dispose(); throw; }
    }
}
