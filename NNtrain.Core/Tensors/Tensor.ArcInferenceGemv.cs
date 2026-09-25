using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    // The caller supplies an already-decoded bias, preserving its ordinary
    // (non-matrix-operand) storage semantics. A null bias produces an FP32
    // partial for the tensor-parallel sum on the other Arc lane.
    private Tensor? TryArcInferenceGemv(
        Tensor weight, Tensor? bias, ArcBuffer? decodedBias,
        int[] outputShape, bool relu, TensorDType? outputDType)
    {
        var lane = ArcLane;
        if (!lane.Options.InferenceGemv || !ArcResident || AutogradContext.IsRecordingEnabled
            || !lane.Options.XmxMatrices || !lane.Device.SupportsXmx
            || lane.Device.MinimumSubgroupSize != 16 || weight.Rank != 2
            || !weight.ArcResidentOnCurrentLane() || Numel != _shape[^1]
            || weight._shape[1] != _shape[^1] || weight._shape[0] <= 0
            || (bias is null) != (decodedBias is null)) return null;

        int k = _shape[^1], n = weight._shape[0];
        int format = weight.DType switch
        {
            TensorDType.Float32 => 0,
            TensorDType.BFloat16 => 1,
            TensorDType.Bfp8 => 2,
            _ => -1,
        };
        if (format < 0) return null;

        using var input = ArcUploadValues(matrixOperand: true);
        using var packedWeight = weight.ArcMatrixOperand();
        using var output = lane.Allocate(n);
        lane.Run2D("arc_inference_gemv_packed", 16, ((n + 7L) / 8) * 8,
            16, 8, input, packedWeight.Value, packedWeight.Scales ?? packedWeight.Value,
            decodedBias ?? packedWeight.Value, output, n, k, format,
            packedWeight.BlockSize, bias is null ? 0 : 1, relu ? 1 : 0);
        return ArcDeviceResult(output, outputShape,
            bias is null ? [this, weight] : [this, weight, bias], outputDType);
    }

    private bool ArcResidentOnCurrentLane()
        => _arcReplica is null || _arcReplica.Lane == ArcLane;
}
