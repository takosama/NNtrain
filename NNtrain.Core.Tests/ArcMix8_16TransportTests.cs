using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMix8_16TransportTests
{
    [Fact]
    public void PackedReductionMatchesFloatScratchAndRemovesTwoKernelLaunches()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");
        const int elements = 1024;
        ushort[] contribution = Enumerable.Range(0, elements)
            .Select(i => TensorStorageCodec.EncodeBFloat16(0.0037f + i * 0.00031f))
            .ToArray();

        (float[] Values, long Launches) Run(bool direct)
        {
            using var execution = Tensor.BeginArcExecution(
                precision: TensorPrecisionMode.Mix8_16,
                options: new ArcExecutionOptions
                {
                    Mix8_16DirectGradientReduction = direct,
                });
            var tensor = new Tensor(new float[elements], [elements]);
            for (int i = 0; i < elements; i++)
                tensor.MutableGrad[i] = 0.12539f - i * 0.00023f;
            _ = tensor.ArcBFloat16Gradient();
            long before = Tensor.ArcLane.KernelLaunchCount;
            tensor.AccumulateArcBFloat16Gradient(contribution);
            long launches = Tensor.ArcLane.KernelLaunchCount - before;
            return (tensor.Grad.ToArray(), launches);
        }

        var oldPath = Run(false);
        var packedPath = Run(true);
        Assert.Equal(oldPath.Values, packedPath.Values);
        Assert.Equal(3, oldPath.Launches);
        Assert.Equal(1, packedPath.Launches);
    }

    [Fact]
    public void CrossGpuGradientUsesTwoBytesPerElementInEachDirection()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(0) || !Tensor.IsArcAvailable(1),
            "Two Intel Arc GPUs are required.");
        using var execution = Tensor.BeginArcInferenceExecution(
            [0, 1], TensorPrecisionMode.Mix8_16);
        const int elements = 1024;
        var source = new Tensor(new float[elements], [elements]);
        var destination = new Tensor(new float[elements], [elements]);
        ArcExecutionLane sourceLane, destinationLane;
        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, 1)))
        {
            for (int i = 0; i < elements; i++)
                source.MutableGrad[i] = 0.125f + i / 1024f;
            _ = source.ArcBFloat16Gradient();
            sourceLane = Tensor.ArcLane;
        }
        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, 0)))
        {
            _ = destination.ArcGradient();
            destinationLane = Tensor.ArcLane;
        }

        long d2hBefore = sourceLane.D2HBytes;
        long h2dBefore = destinationLane.H2DBytes;
        ushort[] packed;
        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, 1)))
            packed = source.CaptureArcBFloat16Gradient();
        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, 0)))
            destination.AccumulateArcBFloat16Gradient(packed);

        Assert.Equal(elements * sizeof(ushort), sourceLane.D2HBytes - d2hBefore);
        Assert.Equal(elements * sizeof(ushort), destinationLane.H2DBytes - h2dBefore);
        float[] expected = packed.Select(TensorStorageCodec.DecodeBFloat16).ToArray();
        Assert.Equal(expected, destination.Grad.ToArray());
    }
}
