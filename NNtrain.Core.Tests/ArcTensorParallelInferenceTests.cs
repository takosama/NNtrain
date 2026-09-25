using NNtrain;
using NNtrain.Arc;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class ArcTensorParallelInferenceTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Float32, 2e-3f)]
    [InlineData(TensorPrecisionMode.Mix16_32, 0.08f)]
    [InlineData(TensorPrecisionMode.Mix8_32, 0.12f)]
    [InlineData(TensorPrecisionMode.Mix8_16, 0.12f)]
    public void TwoArcLanesContributeAndMatchSingleArcLogits(
        TensorPrecisionMode precision, float tolerance)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(0) || !Tensor.IsArcAvailable(1),
            "Two Intel Arc OpenCL GPUs are required.");
        using var execution = Tensor.BeginArcInferenceExecution([0, 1], precision);
        var model = new GptRinWikiJp(64, 8, 32, 4, 64, 2,
            new Random(43), dropout: 0f, tieWordEmbeddings: true);
        model.to(precision, 32);
        model.eval();
        ArcExecutionLane first = (ArcExecutionLane)ExecutionSession.Current!
            .GetRequiredLane(ExecutionDeviceKind.Arc, 0);
        ArcExecutionLane second = (ArcExecutionLane)ExecutionSession.Current!
            .GetRequiredLane(ExecutionDeviceKind.Arc, 1);
        int[] prompt = [1, 5, 9, 13];
        float[] expected;
        using (AutogradContext.NoGrad())
        using (Tensor.BeginArcInferenceFrame())
            expected = model.forward(prompt, 1, prompt.Length).Data.ToArray();
        int[] singleGenerated = model.GenerateTokenIds(prompt, 2,
            temperature: 0f, topK: 1, stopTokenId: null);

        long firstKernels = first.KernelLaunchCount;
        long secondKernels = second.KernelLaunchCount;
        model.ArcTensorParallelEnabled = true;
        float[] actual;
        using (AutogradContext.NoGrad())
        using (Tensor.BeginArcInferenceFrame())
            actual = model.forward(prompt, 1, prompt.Length).Data.ToArray();

        Assert.Equal(expected.Length, actual.Length);
        Assert.True(first.KernelLaunchCount > firstKernels);
        Assert.True(second.KernelLaunchCount > secondKernels);
        Assert.All(actual, value => Assert.True(float.IsFinite(value)));
        float maximumDifference = expected.Zip(actual)
            .Max(pair => MathF.Abs(pair.First - pair.Second));
        Assert.InRange(maximumDifference, 0f, tolerance);
        int[] parallelGenerated = model.GenerateTokenIds(prompt, 2,
            temperature: 0f, topK: 1, stopTokenId: null);
        Assert.Equal(singleGenerated, parallelGenerated);
    }

    [Fact]
    public void TwoArcGenerationReusesShardsAndReleasesIntermediates()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(0) || !Tensor.IsArcAvailable(1),
            "Two Intel Arc OpenCL GPUs are required.");
        using var execution = Tensor.BeginArcInferenceExecution([0, 1]);
        var model = new GptRinWikiJp(32, 8, 16, 4, 32, 2,
            new Random(47), dropout: 0f, tieWordEmbeddings: true)
        {
            ArcTensorParallelEnabled = true,
        };
        ArcExecutionLane first = (ArcExecutionLane)ExecutionSession.Current!
            .GetRequiredLane(ExecutionDeviceKind.Arc, 0);
        ArcExecutionLane second = (ArcExecutionLane)ExecutionSession.Current!
            .GetRequiredLane(ExecutionDeviceKind.Arc, 1);
        int[] output = model.GenerateTokenIds([1, 2, 3], 2,
            temperature: 0f, topK: 1, stopTokenId: null);
        Assert.Equal(5, output.Length);
        long firstBytes = first.AllocatedBytes;
        long secondBytes = second.AllocatedBytes;
        _ = model.GenerateTokenIds([1, 2, 3], 2,
            temperature: 0f, topK: 1, stopTokenId: null);
        Assert.Equal(firstBytes, first.AllocatedBytes);
        Assert.Equal(secondBytes, second.AllocatedBytes);
        Assert.True(first.PeakAllocatedBytes >= firstBytes && first.PeakAllocatedBytes > 0);
        Assert.True(second.PeakAllocatedBytes >= secondBytes && second.PeakAllocatedBytes > 0);
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"Arc TP peak allocated bytes: device 0={first.PeakAllocatedBytes:N0}, " +
            $"device 1={second.PeakAllocatedBytes:N0}; " +
            $"retained: device 0={firstBytes:N0}, device 1={secondBytes:N0}");
    }

    [Fact]
    public void TensorParallelKeepsLogitsOnPrimaryWithSecondaryAmbientDevice()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(0) || !Tensor.IsArcAvailable(1),
            "Two Intel Arc OpenCL GPUs are required.");
        using var execution = Tensor.BeginArcInferenceExecution([0, 1]);
        using var secondary = TensorExecutionContext.Push(
            new TorchDevice(TensorDevice.Arc, 1));
        var model = new GptRinWikiJp(32, 8, 16, 4, 32, 1,
            new Random(61), dropout: 0f, tieWordEmbeddings: true)
        {
            ArcTensorParallelEnabled = true,
        };

        using (AutogradContext.NoGrad())
        using (Tensor.BeginArcInferenceFrame())
        {
            Tensor logits = model.forward([1, 2, 3], 1, 3);
            Assert.Equal(TensorDevice.Arc, logits.device.Type);
            Assert.Equal(0, logits.device.Index);
        }

        int[] generated = model.GenerateTokenIds([1, 2, 3], 1,
            temperature: 0f, topK: 1, stopTokenId: null);
        Assert.Equal(4, generated.Length);
        Assert.Equal(1, TensorExecutionContext.Device.Index);
    }

    [Fact]
    public void TwoArcMixed8RunsProductionHeadWidthAndAttentionLayout()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(0) || !Tensor.IsArcAvailable(1),
            "Two Intel Arc OpenCL GPUs are required.");
        using var execution = Tensor.BeginArcInferenceExecution([0, 1],
            TensorPrecisionMode.Mix8_32);
        var model = new GptRinWikiJp(257, 128, 512, 16, 1536, 1,
            new Random(53), dropout: 0f, tieWordEmbeddings: true)
        {
            ArcTensorParallelEnabled = true,
        };
        model.to(TensorPrecisionMode.Mix8_32, 32);
        int[] prompt = Enumerable.Range(0, 128).Select(i => i % 257).ToArray();
        int[] generated = model.GenerateTokenIds(prompt, 1,
            temperature: 0f, topK: 1, stopTokenId: null);
        Assert.Equal(prompt.Length + 1, generated.Length);
        Assert.InRange(generated[^1], 0, 256);
    }

    [Fact]
    public void OddHeadCountFailsPreflightBeforeDispatch()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(0) || !Tensor.IsArcAvailable(1),
            "Two Intel Arc OpenCL GPUs are required.");
        using var execution = Tensor.BeginArcInferenceExecution([0, 1]);
        var model = new GptRinWikiJp(32, 8, 12, 3, 24, 1,
            new Random(59), dropout: 0f)
        {
            ArcTensorParallelEnabled = true,
        };
        Assert.False(model.CanUseArcTensorParallel(out string reason));
        Assert.Contains("even number of attention heads", reason);
        Assert.Throws<NotSupportedException>(() => model.GenerateTokenIds(
            [1, 2], 1, temperature: 0f, topK: 1, stopTokenId: null));
    }
}
