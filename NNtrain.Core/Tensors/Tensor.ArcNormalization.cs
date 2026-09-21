using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    private Tensor ArcNormDevice(Tensor gamma, Tensor beta, float eps, Tensor? branch, float probability, Random? random)
    {
        int width = _shape[^1], rows = Numel / width;
        if (gamma.Numel != width || beta.Numel != width) throw new ArgumentException("LayerNorm parameter dimensions do not match.");
        if (branch is not null && !_shape.AsSpan().SequenceEqual(branch._shape)) throw ShapeMismatch(this, branch, "Arc residual");
        uint seed = probability == 0 ? 0 : NextDropoutSeed(random ?? Random.Shared);
        uint threshold = (uint)(probability * (uint.MaxValue + 1d));
        float scale = 1f / (1f - probability);
        var lane = ArcLane;
        ArcBuffer Input()
        {
            ArcBuffer value = ArcUploadValues();
            if (branch is null) return value;
            try
            {
                using var branchValue = branch.ArcUploadValues();
                // Reads residual[i] before replacing that same element; no cross-thread alias.
                lane.Run("dropout", Numel, 0, branchValue, value, value, Numel, seed, threshold, scale, 1);
                return value;
            }
            catch { value.Dispose(); throw; }
        }
        float[] output = new float[Numel], stats = new float[2 * rows];
        using (var input = Input())
        using (var g = gamma.ArcUploadValues())
        using (var b = beta.ArcUploadValues())
            lane.Run("norm", rows, 0, input, g, b, Out(output), Out(stats), rows, width, eps);
        Tensor result = ArcResult(output, _shape, branch is null ? [this, gamma, beta] : [this, branch, gamma, beta]);
        result.Node.BackwardAction = () => {
            // Rebuild the unquantized residual sum on the device. Do not retain an
            // FP32 activation per LayerNorm for the entire forward graph.
            using var input = Input();
            using var g = gamma.ArcUploadValues();
            using var dy = lane.Upload(result._grad);
            using var statistics = lane.Upload(stats);
            using var dx = lane.Allocate(Numel);
            lane.Run("norm_dx_set", rows, 0, input, g, dy, statistics, dx, rows, width);
            lane.Run("norm_dw", width, 0, input, dy, statistics, InOut(gamma._grad), InOut(beta._grad), rows, width);
            lane.Run("copy_scale", Numel, 0, dx, InOut(_grad), Numel, 1f, 1);
            if (branch is not null)
                lane.Run("dropout_back", Numel, 0, dx, InOut(branch._grad), Numel, seed, threshold, scale);
        };
        return result;
    }
}
