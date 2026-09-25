using static NNtrain.Arc.ArcExecutionLane;
using NNtrain.Runtime.Execution;

namespace NNtrain;

partial class Tensor
{
    /// <summary>Borrow current payload/scales without decoding or reading back.</summary>
    internal ArcXmxStorageOperand ArcMatrixOperand(bool cacheWeightPanels = false)
    {
        if (!ArcResident) throw new InvalidOperationException("Packed matrix views require resident Arc execution.");
        EnsureArcPacked();
        ArcReplica state = _arcReplica!;
        if (cacheWeightPanels && state.Lane.Options.MatrixPanelCacheMiB > 0)
            state.MatrixPanels ??= new(state.Lane);
        return new(state.Lane, state.Value!, state.Scales, DType, Numel,
            DType == TensorDType.Bfp8 ? Bfp8Quantization!.GetEffectiveBlockSize(Numel) : 1,
            panelCache: cacheWeightPanels ? state.MatrixPanels : null);
    }

    // Call before ArcUploadValues(true) in resident Linear. Unsupported shapes
    // and mixed Float32/low-precision forward operands keep the existing path.
    private bool TryArcPackedLinear(Tensor weight, ArcBuffer output, ArcBuffer bias, bool relu)
    {
        int k = _shape[^1], m = Numel / k, n = weight._shape[0];
        if (DType == TensorDType.Float32 || weight.DType == TensorDType.Float32
            || !ArcXmxStorageOperand.CanRunAny(ArcLane, m, n, k, tb: true, hasBias: true, relu: relu)) return false;
        using var a = ArcMatrixOperand();
        using var b = weight.ArcMatrixOperand(cacheWeightPanels: true);
        return ArcXmxStorageOperand.TryGemm(ArcLane, a, b, output, m, n, k, tb: true, bias: bias, relu: relu);
    }

    private bool TryArcPackedLinearBackward(Tensor weight, ArcBuffer dy, ArcBuffer dx, ArcBuffer dw,
        ArcBuffer? gate, bool relu)
    {
        int k = _shape[^1], m = Numel / k, n = weight._shape[0];
        var lane = ArcLane;
        if (!lane.Options.PackedMatrixStorage || !ArcXmxStorageOperand.CanRunAny(lane, m, k, n, accumulate: true)
            || !ArcXmxStorageOperand.CanRunAny(lane, n, k, m, ta: true, accumulate: true)) return false;
        using var gradient = new ArcXmxStorageOperand(lane, dy, null, TensorDType.Float32, checked(m * n));
        using var a = ArcMatrixOperand();
        using var b = weight.ArcMatrixOperand(cacheWeightPanels: true);
        if (!relu && gate is null && lane.Options.Mix8_16LinearDualGradientPack
            && TensorExecutionContext.ActivePrecisionPolicy?.Mode == PrecisionMode.Mix8_16
            && ArcXmxStorageOperand.CanRun(lane, m, k, n)
            && ArcXmxStorageOperand.CanRun(lane, n, k, m))
        {
            var dual = ArcXmxStorageOperand.PackDualGradientA(lane, dy, m, n);
            using var normalGradient = dual.Normal;
            using var transposedGradient = dual.Transposed;
            using (var packedWeight = b.PackB(k, n, false))
                ArcXmxStorageOperand.GemmPanels(lane, normalGradient, packedWeight,
                    dx, m, k, n, accumulate: true);
            using (var packedInput = a.PackB(k, m, false))
                ArcXmxStorageOperand.GemmPanels(lane, transposedGradient, packedInput,
                    dw, n, k, m, ta: true, accumulate: true);
            return true;
        }
        ArcXmxStorageOperand.TryGemm(lane, gradient, b, dx, m, k, n, accumulate: true, gate: gate, gateOperand: gate is null ? 0 : 1);
        ArcXmxStorageOperand.TryGemm(lane, gradient, a, dw, n, k, m, ta: true, accumulate: true, gate: gate, gateOperand: gate is null ? 0 : 1);
        return true;
    }
}
