using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    private Tensor ArcBlockResidualNorm(Tensor gamma, Tensor beta, float epsilon, Tensor? branch,
        uint seed, uint threshold, float scale, int width, int rows)
    {
        var lane = ArcLane;
        using var input = ArcUploadValues();
        using var residual = branch?.ArcUploadValues();
        using var g = gamma.ArcUploadValues(); using var b = beta.ArcUploadValues();
        using var output = lane.Allocate(Numel);
        var stats = lane.Allocate(checked(rows * 2));
        try
        {
            lane.Run("norm_residual_serial", rows, 0, input, residual ?? input, g, b,
                output, stats, rows, width, epsilon, seed, threshold, scale, branch is null ? 0 : 1);
            Tensor result = ArcDeviceResult(output, _shape, branch is null ? [this, gamma, beta] : [this, branch, gamma, beta]);
            if (result.Node.IsDetached) { stats.Dispose(); return result; }
            result.Node.RegisterResource(stats);
            result.Node.BackwardAction = () => {
                using var x = ArcUploadValues(); using var r = branch?.ArcUploadValues();
                using var gv = gamma.ArcUploadValues();
                ArcBuffer dy = result.ArcGradient(), dx = ArcGradient();
                lane.Run("norm_residual_dx_serial", rows, 0, x, r ?? x, gv, dy, stats,
                    dx, branch?.ArcGradient() ?? dx, rows, width, seed, threshold, scale, branch is null ? 0 : 1);
                int groups = (rows + 255) / 256;
                using var parts = lane.Allocate(checked(groups * width * 2));
                lane.Run2D("norm_residual_parameter_parts", ((width + 31L) / 32) * 32, groups * 8L, 32, 8,
                    dy, x, r ?? x, stats, parts, rows, width, seed, threshold, scale, branch is null ? 0 : 1);
                lane.Run("gradient_rows_finish", width, 0, parts, beta.ArcGradient(), gamma.ArcGradient(), groups, width, 1);
            };
            return result;
        }
        catch { stats.Dispose(); throw; }
    }
}
