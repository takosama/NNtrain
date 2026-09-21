using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    private Tensor? TryArcFusedBfp8Linear(Tensor weight, Tensor bias, ArcBuffer biases,
        int[] shape, bool relu, TensorDType? dtype)
    {
        var lane = ArcLane;
        int k = _shape[^1], m = Numel / k, n = weight._shape[0];
        if (!lane.Options.FusedBfp8Linear || !lane.Options.PackedMatrixStorage
            || DType == TensorDType.Float32 || weight.DType == TensorDType.Float32
            || m < 4096 || n % 32 != 0 || k > 2048
            || !ArcXmxStorageOperand.CanRunAny(lane, m, n, k, tb: true, hasBias: true, relu: relu)) return null;
        Tensor[] parents = [this, weight, bias];
        if ((dtype ?? TensorDTypeContract.Promote(parents)) != TensorDType.Bfp8
            || !parents.Any(p => p.DType == TensorDType.Bfp8)) return null;
        var descriptor = SelectBfp8ResultDescriptor(parents);
        int length = checked(m * n);
        if (descriptor.GetEffectiveBlockSize(length) != 32) return null;
        int rows = ArcXmxStorageOperand.CanRun(lane, m, n, k) ? m
            : ArcXmxStorageOperand.PlanStreamed(m, n, k, false, true, false, true, relu,
                lane.Options.ParallelWeightGradients)!.Value.TileLength;
        Tensor result = FromStorageResult(TensorStorage.CreateDeviceBfp8Placeholder(length, descriptor), shape, parents);
        try
        {
            ArcReplica state = result.ArcOwner();
            state.Value = lane.AllocateBytes(length);
            state.Scales = lane.Allocate(length / 32);
            using var input = ArcMatrixOperand();
            using var weights = weight.ArcMatrixOperand(cacheWeightPanels: true);
            using var packedB = weights.PackB(n, k, true);
            int columns = lane.Options.ExpandedXmxTiles && m >= 4096 && n >= 512 && n % 64 == 0 ? 64 : 32;
            for (int first = 0; first < m; first += rows)
            {
                int count = Math.Min(rows, m - first);
                using var chunk = input.Slice(checked(first * k), checked(count * k));
                using var packedA = chunk.PackA(count, k, false);
                lane.Run2D($"gemm_xmx_bfp8_epilogue_16x{columns}", n / (long)columns * 16, ((count + 255L) / 256) * 16, 16, 16,
                    packedA, packedB, state.Value, state.Scales, lane.NumericStatus, biases,
                    count, n, k, relu ? 1 : 0, checked(first * n));
            }
            state.DataDirty = true;
            result._device = TensorDevice.Arc;
            result._arcDeviceIndex = lane.DeviceIndex;
            ArcInferenceFrame.Current?.Add(result);
            ArcCheckpointFrame.Current?.Add(result);
            return result;
        }
        catch { result.ReleaseArcReplica(preserve: false); throw; }
    }
}
