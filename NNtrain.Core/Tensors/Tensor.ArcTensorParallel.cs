using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Copies the physical packed value through host memory into another Arc
    /// lane. OpenCL contexts are independent, so a same-lane buffer copy is
    /// not valid here. Inference owns the detached copy until its frame ends.
    /// </summary>
    internal Tensor ArcInferenceCopyToDevice(int deviceIndex)
    {
        if (!ArcResident || AutogradContext.IsRecordingEnabled)
            throw new InvalidOperationException("Arc device copies require resident no-grad inference.");
        EnsureHostDataCurrent();
        Tensor copy = FromStorageResult(_data.Clone(), _shape, []);
        copy._device = TensorDevice.Arc;
        copy._arcDeviceIndex = deviceIndex;
        ArcInferenceFrame.Current?.Add(copy);
        return copy;
    }

    internal void ReleaseArcInferenceReplica()
    {
        ReleaseArcReplica(preserve: true);
    }

    /// <summary>
    /// A column-sharded projection. The result remains FP32 until both Arc
    /// lanes have contributed; the caller then applies one bias and one
    /// storage rounding boundary.
    /// </summary>
    internal Tensor ArcInferenceLinearPartial(Tensor weight)
    {
        if (!ArcResident || AutogradContext.IsRecordingEnabled)
            throw new InvalidOperationException("Arc partial projections require resident no-grad inference.");
        ArgumentNullException.ThrowIfNull(weight);
        int inputWidth = _shape[^1];
        if (weight.Rank != 2 || weight._shape[1] != inputWidth)
            throw new ArgumentException("Arc partial projection dimensions do not match.", nameof(weight));
        int rows = Numel / inputWidth;
        int outputWidth = weight._shape[0];
        int[] outputShape = (int[])_shape.Clone();
        outputShape[^1] = outputWidth;
        Tensor? fusedGemv = TryArcFusedInferenceGemv(weight, null, outputShape,
            relu: false, outputDType: TensorDType.Float32);
        if (fusedGemv is not null) return fusedGemv;
        Tensor? gemv = TryArcInferenceGemv(weight, null, null, outputShape,
            relu: false, outputDType: TensorDType.Float32);
        if (gemv is not null) return gemv;
        var lane = ArcLane;
        using var output = lane.Allocate(checked(rows * outputWidth));
        bool packed = false;
        if (DType != TensorDType.Float32 && weight.DType != TensorDType.Float32
            && lane.Options.PackedMatrixStorage
            && ArcXmxStorageOperand.CanRunAny(lane, rows, outputWidth, inputWidth, tb: true))
        {
            using var input = ArcMatrixOperand();
            using var weights = weight.ArcMatrixOperand(cacheWeightPanels: true);
            packed = ArcXmxStorageOperand.TryGemm(lane, input, weights, output,
                rows, outputWidth, inputWidth, tb: true);
        }
        if (!packed)
        {
            using var input = ArcUploadValues(true);
            using var weights = weight.ArcUploadValues(true);
            ArcMuonMath.Gemm(lane, input, weights, output, rows, outputWidth,
                inputWidth, tb: true,
                bf16: DType != TensorDType.Float32 && weight.DType != TensorDType.Float32 ? 3 : 0);
        }
        return ArcDeviceResult(output, outputShape, [this, weight], TensorDType.Float32);
    }

    /// <summary>
    /// Combines two FP32 column-shard products and adds the original bias once
    /// before publishing the model's output dtype.
    /// </summary>
    internal Tensor ArcInferenceSumPartials(
        float[] otherPartial,
        Tensor bias,
        TensorDType outputDType)
    {
        if (!ArcResident || DType != TensorDType.Float32
            || AutogradContext.IsRecordingEnabled)
            throw new InvalidOperationException("Arc partial sums require resident FP32 no-grad inference.");
        ArgumentNullException.ThrowIfNull(otherPartial);
        ArgumentNullException.ThrowIfNull(bias);
        int width = _shape[^1];
        if (otherPartial.Length != Numel || bias.Rank != 1 || bias.Numel != width)
            throw new ArgumentException("Arc partial sum dimensions do not match.");
        var lane = ArcLane;
        using var left = ArcUploadValues();
        using var right = lane.Upload(otherPartial);
        using var biasValues = bias.ArcUploadValues();
        using var sum = lane.Allocate(Numel);
        using var output = lane.Allocate(Numel);
        lane.Run("add", Numel, 0, left, right, sum, Numel);
        lane.Run("add_row_bias", Numel, 0, sum, biasValues, output, Numel, width);
        if (outputDType != TensorDType.Bfp8)
            return ArcDeviceResult(output, _shape, [this, bias], outputDType);
        Tensor result = FromStorageResult(
            TensorStorage.CreateDeviceBfp8Placeholder(Numel,
                bias.Bfp8Quantization ?? throw new InvalidOperationException(
                    "A BFP8 parallel projection requires a BFP8 bias.")),
            _shape,
            [this, bias]);
        try
        {
            result.PublishArcValues(output);
            ArcInferenceFrame.Current?.Add(result);
            return result;
        }
        catch
        {
            result.ReleaseArcReplica(preserve: false);
            throw;
        }
    }
}
