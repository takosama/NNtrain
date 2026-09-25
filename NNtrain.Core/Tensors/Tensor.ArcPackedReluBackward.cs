using NNtrain.Arc;
using NNtrain.Runtime.Execution;
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
        if (lane.Options.FusedPackedReluBackward)
        {
            using var fusedInput = ArcMatrixOperand();
            using var fusedWeights = weight.ArcMatrixOperand(cacheWeightPanels: true);
            bool packBias = lane.Options.FusedReluPackBias
                && lane.Options.ParallelReductions && m >= 512;
            int groups = packBias ? (m + 255) / 256 : 0;
            using var biasPartials = packBias ? lane.Allocate(checked(groups * n)) : null;
            if (packBias && lane.Options.Mix8_16ReluDualGradientPack
                && TensorExecutionContext.ActivePrecisionPolicy?.Mode == PrecisionMode.Mix8_16
                && ArcXmxStorageOperand.CanRun(lane, m, k, n)
                && ArcXmxStorageOperand.CanRun(lane, n, k, m))
            {
                var panels = ArcXmxStorageOperand.PackReluDualGradientA(lane, dy, gate,
                    biasPartials!, m, n);
                using var normalGradient = panels.Normal;
                using var transposedGradient = panels.Transposed;
                using (var packedWeights = fusedWeights.PackB(k, n, false))
                    ArcXmxStorageOperand.GemmPanels(lane, normalGradient, packedWeights,
                        dx, m, k, n, accumulate: true);
                using (var packedInput = fusedInput.PackB(k, m, false))
                    ArcXmxStorageOperand.GemmPanels(lane, transposedGradient, packedInput,
                        dw, n, k, m, ta: true, accumulate: true);
            }
            else
            {
                ArcXmxStorageOperand.GemmReluGradient(lane, dy, gate, fusedWeights, dx,
                    m, k, n, transpose: false, accumulate: true,
                    biasPartials: biasPartials);
                ArcXmxStorageOperand.GemmReluGradient(lane, dy, gate, fusedInput, dw,
                    n, k, m, transpose: true, accumulate: true);
            }
            if (biasPartials is not null)
                lane.Run("gradient_rows_finish", n, 0, biasPartials, db, db, groups, n, 0);
            else
                ArcReluBiasGradient(lane, dy, gate, db, m, n);
            return true;
        }
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

    private static void ArcReluBiasGradient(ArcExecutionLane lane, ArcBuffer dy,
        ArcXmxStorageOperand gate, ArcBuffer db, int rows, int width)
    {
        ArcBuffer scales = gate.Scales ?? gate.Value;
        int block = gate.BlockSize;
        int bfp8 = gate.DType == TensorDType.Bfp8 ? 1 : 0;
        if (!lane.Options.ParallelReductions || rows < 512)
        {
            for (int start = 0; start < rows; start += 2048)
                lane.Run("linear_relu_bias_chunk_gate", width, 0, dy, gate.Value,
                    scales, db, rows, width, block, bfp8, start,
                    Math.Min(2048, rows - start));
            return;
        }
        int groups = (rows + 255) / 256;
        using var partials = lane.Allocate(checked(groups * width));
        lane.Run2D("linear_relu_bias_rows_gate", ((width + 31L) / 32) * 32,
            groups * 8L, 32, 8, dy, gate.Value, scales, partials,
            rows, width, block, bfp8);
        lane.Run("gradient_rows_finish", width, 0, partials, db, db, groups, width, 0);
    }
}
