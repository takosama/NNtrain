using NNtrain.Runtime.Execution;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    // Experimental forward-only path. Gradients retain the established BF16
    // matrix operand route; only BFP8 x BFP8 forward uses signed INT8 DPAS.
    private Tensor? TryArcInt8Bfp8Linear(Tensor weight, Tensor bias, ArcBuffer biases,
        int[] shape, bool relu, TensorDType? dtype)
    {
        var lane = ArcLane;
        var policy = TensorExecutionContext.ActivePrecisionPolicy;
        int k = _shape[^1], m = Numel / k, n = weight._shape[0];
        if (!lane.Options.Mix8_16Int8Linear
            || policy?.Mode != PrecisionMode.Mix8_16
            || (policy.GemmExecutionFormats & GemmExecutionFormat.Int8) == 0
            || !lane.Options.XmxMatrices || !lane.Device.SupportsXmx
            || lane.Device.MinimumSubgroupSize != 16
            || DType != TensorDType.Bfp8 || weight.DType != TensorDType.Bfp8
            || k is < 32 or > 2048 || k % 32 != 0
            || m < 128 || n < 32 || n % 32 != 0
            || (long)n * k > 128L * 1024 * 1024)
            return null;

        Tensor[] parents = [this, weight, bias];
        if ((dtype ?? TensorDTypeContract.Promote(parents)) != TensorDType.Bfp8)
            return null;
        var descriptor = SelectBfp8ResultDescriptor(parents);
        int length = checked(m * n);
        if (descriptor.GetEffectiveBlockSize(length) != 32)
            return null;

        using var input = ArcMatrixOperand();
        using var weights = weight.ArcMatrixOperand();
        if (input.BlockSize != 32 || weights.BlockSize != 32)
            return null;

        Tensor result = FromStorageResult(
            TensorStorage.CreateDeviceBfp8Placeholder(length, descriptor), shape, parents);
        try
        {
            ArcReplica state = result.ArcOwner();
            state.Value = lane.AllocateBytes(length);
            state.Scales = lane.Allocate(length / 32);
            using var packedB = lane.AllocateBytes(checked(n * k));
            lane.Run2D("xmx_i8_bfp8_pack_b", n / 32L * 256, k / 32L, 256, 1,
                weights.Value, packedB, n, k);

            // Each chunk owns its temporary INT8 A panel. The row offset is
            // also applied to scale indexing and the final BFP8 publication.
            const int maximumChunkRows = 8192;
            for (int first = 0; first < m; first += maximumChunkRows)
            {
                int count = Math.Min(maximumChunkRows, m - first);
                int panelElements = checked(((count + 7) / 8) * (k / 32) * 128);
                using var packedA = lane.AllocateBytes(checked(panelElements * sizeof(ushort)));
                lane.Run("xmx_i8_bfp8_pack_a", panelElements, 0,
                    input.Value, packedA, count, k, first);
                lane.Run2D("xmx_i8_bfp8_linear_16x32", n / 32L * 16,
                    (count + 255L) / 256 * 16, 16, 16,
                    packedA, packedB, input.Scales!, weights.Scales!,
                    state.Value, state.Scales, lane.NumericStatus, biases,
                    count, n, k, relu ? 1 : 0, first);
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
