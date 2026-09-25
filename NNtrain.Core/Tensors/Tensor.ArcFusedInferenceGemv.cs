using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    // Return null for prefill, training, and disabled A/B paths. The caller may
    // then use the older decoded-input GEMV or its normal matrix operation.
    private Tensor? TryArcFusedInferenceGemv(
        Tensor weight, Tensor? bias, int[] outputShape,
        bool relu, TensorDType? outputDType)
    {
        var lane = ArcLane;
        if (!lane.Options.InferenceGemv || !lane.Options.InferenceFusedGemv
            || !ArcResident || AutogradContext.IsRecordingEnabled
            || !lane.Options.XmxMatrices || !lane.Device.SupportsXmx
            || lane.Device.MinimumSubgroupSize != 16 || weight.Rank != 2
            || !weight.ArcResidentOnCurrentLane()
            || (bias is not null && !bias.ArcResidentOnCurrentLane())
            || Numel != _shape[^1] || weight._shape[1] != _shape[^1]
            || weight._shape[0] <= 0
            || (bias is not null && (bias.Rank != 1 || bias.Numel != weight._shape[0])))
            return null;

        static int Format(TensorDType dtype) => dtype switch
        {
            TensorDType.Float32 => 0,
            TensorDType.BFloat16 => 1,
            TensorDType.Bfp8 => 2,
            _ => -1,
        };
        if (Format(DType) < 0 || Format(weight.DType) < 0
            || (bias is not null && Format(bias.DType) < 0)) return null;

        Tensor[] parents = bias is null ? [this, weight] : [this, weight, bias];
        bool directBf16 = outputDType == TensorDType.BFloat16
            || outputDType is null && ArcMayPublishBFloat16Activation(parents);
        int n = weight._shape[0], k = _shape[^1];
        using var input = ArcMatrixOperand();
        using var weights = weight.ArcMatrixOperand();
        using var biases = bias?.ArcMatrixOperand();
        ArcBuffer output = directBf16
            ? lane.AllocateBytes(checked(n * sizeof(ushort))) : lane.Allocate(n);
        bool outputOwnedByResult = false;
        try
        {
            lane.Run2D("arc_inference_gemv_fused_packed", 16,
                ((n + 7L) / 8) * 8, 16, 8,
                input.Value, input.Scales ?? input.Value,
                weights.Value, weights.Scales ?? weights.Value,
                biases?.Value ?? weights.Value, biases?.Scales ?? biases?.Value ?? weights.Value,
                output, n, k,
                Format(input.DType), input.BlockSize,
                Format(weights.DType), weights.BlockSize,
                biases is null ? 0 : Format(biases.DType), biases?.BlockSize ?? 1,
                biases is null ? 0 : 1, relu ? 1 : 0, directBf16 ? 1 : 0);
            if (!directBf16)
                return ArcDeviceResult(output, outputShape, parents, outputDType);
            Tensor result = ArcDeviceBFloat16Result(output, outputShape, parents);
            outputOwnedByResult = true;
            return result;
        }
        finally { if (!outputOwnedByResult) output.Dispose(); }
    }
}
