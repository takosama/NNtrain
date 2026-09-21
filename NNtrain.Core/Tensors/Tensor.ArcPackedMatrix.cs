using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    /// <summary>Borrow current payload/scales without decoding or reading back.</summary>
    internal ArcXmxStorageOperand ArcMatrixOperand()
    {
        if (!ArcResident) throw new InvalidOperationException("Packed matrix views require resident Arc execution.");
        EnsureArcPacked();
        ArcReplica state = _arcReplica!;
        return new(state.Lane, state.Value!, state.Scales, DType, Numel,
            DType == TensorDType.Bfp8 ? Bfp8Quantization!.GetEffectiveBlockSize(Numel) : 1);
    }

    // Call before ArcUploadValues(true) in resident Linear. Unsupported shapes
    // and mixed Float32/low-precision forward operands keep the existing path.
    private bool TryArcPackedLinear(Tensor weight, ArcBuffer output, ArcBuffer bias, bool relu)
    {
        int k = _shape[^1], m = Numel / k, n = weight._shape[0];
        if (DType == TensorDType.Float32 || weight.DType == TensorDType.Float32
            || !ArcXmxStorageOperand.CanRun(ArcLane, m, n, k)) return false;
        using var a = ArcMatrixOperand();
        using var b = weight.ArcMatrixOperand();
        return ArcXmxStorageOperand.TryGemm(ArcLane, a, b, output, m, n, k, tb: true, bias: bias, relu: relu);
    }

    private bool TryArcPackedLinearBackward(Tensor weight, ArcBuffer dy, ArcBuffer dx, ArcBuffer dw, ArcBuffer? gate)
    {
        int k = _shape[^1], m = Numel / k, n = weight._shape[0];
        var lane = ArcLane;
        if (!lane.Options.PackedMatrixStorage || !ArcXmxStorageOperand.CanRun(lane, m, k, n)
            || !ArcXmxStorageOperand.CanRun(lane, n, k, m)) return false;
        using var gradient = new ArcXmxStorageOperand(lane, dy, null, TensorDType.Float32, checked(m * n));
        using var a = ArcMatrixOperand();
        using var b = weight.ArcMatrixOperand();
        ArcXmxStorageOperand.TryGemm(lane, gradient, b, dx, m, k, n, accumulate: true, gate: gate, gateOperand: gate is null ? 0 : 1);
        ArcXmxStorageOperand.TryGemm(lane, gradient, a, dw, n, k, m, ta: true, accumulate: true, gate: gate, gateOperand: gate is null ? 0 : 1);
        return true;
    }
}
