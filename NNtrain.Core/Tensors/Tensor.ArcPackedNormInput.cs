using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    // Decode directly into the unquantized residual sum. No full-sized decoded
    // operand is retained, and storage rounding still precedes dropout/addition.
    private ArcBuffer ArcPackedResidualInput(Tensor branch, uint seed, uint threshold, float scale)
    {
        EnsureArcPacked(); branch.EnsureArcPacked();
        ArcReplica x = _arcReplica!, b = branch._arcReplica!;
        var output = ArcLane.Allocate(Numel);
        try
        {
            int Format(Tensor t) => t.DType switch {
                TensorDType.Float32 => 0, TensorDType.BFloat16 => 1, TensorDType.Bfp8 => 2,
                _ => throw new NotSupportedException("Unsupported Arc normalization storage.") };
            ArcLane.Run("norm_packed_residual_input", Numel, 0,
                x.Value!, x.Scales ?? x.Value!, b.Value!, b.Scales ?? b.Value!, output,
                Numel, Format(this), Format(branch),
                Bfp8Quantization?.GetEffectiveBlockSize(Numel) ?? 1,
                branch.Bfp8Quantization?.GetEffectiveBlockSize(Numel) ?? 1,
                seed, threshold, scale);
            return output;
        }
        catch { output.Dispose(); throw; }
    }
}
