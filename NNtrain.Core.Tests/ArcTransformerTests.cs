using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcTransformerTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void TiledStreamingAndChunkedLossMatchReferenceWithTails(TensorPrecisionMode precision)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc OpenCL GPU is unavailable.");
        (float Loss, float[][] Grad) Run(bool optimized)
        {
            using var scope = Tensor.BeginArcExecution(precision: precision,
                options: optimized ? new ArcExecutionOptions() : ArcExecutionOptions.Reference);
            var model = new GptRinWikiJp(37, 137, 8, 2, 13, 1, new Random(57), dropout: .1f, tieWordEmbeddings: true);
            model.to(precision, 32);
            int[] tokens = Enumerable.Range(0, 137).Select(i => i % 37).ToArray();
            int[] labels = Enumerable.Range(0, 137).Select(i => i % 11 == 0 ? -1 : (i + 1) % 37).ToArray();
            Tensor loss = model.forward_loss(tokens, labels, 1, 137);
            float value = loss.item();
            loss.BackwardAndRelease();
            Assert.Equal(optimized ? 4 + model.parameters().Sum(p => (long)p.T.StorageByteLength + p.T.Numel * 4L) : 0,
                Tensor.ArcLane.AllocatedBytes);
            return (value, model.parameters().Select(p => p.T.Grad.ToArray()).ToArray());
        }
        var reference = Run(false); var optimized = Run(true);
        Assert.InRange(MathF.Abs(reference.Loss - optimized.Loss), 0, 1e-4f);
        for (int p = 0; p < reference.Grad.Length; p++)
            for (int i = 0; i < reference.Grad[p].Length; i++)
                Assert.InRange(MathF.Abs(reference.Grad[p][i] - optimized.Grad[p][i]), 0, 1e-4f);
    }

    [Theory]
    [InlineData(2.5f)]
    [InlineData(0f)]
    public void FractionalNekoMuonMatchesCpu(float depth)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc OpenCL GPU is unavailable.");
        (float[] Values, NekoMuonState State) Run(bool arc)
        {
            using var scope = arc ? Tensor.BeginArcExecution() : TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu));
            var p = new Parameter(Enumerable.Range(0, 15).Select(i => .07f * i - .3f).ToArray(), [5, 3], "w", WeightDecayPolicy.Apply);
            var optimizer = new NekoMuon([p], new NekoMuonOptions { LearningRate = .02f, BetaFast = .9f, BetaSlow = .95f,
                NewtonSchulzDepthMode = NekoMuonNewtonSchulzDepthMode.Fixed, NewtonSchulzDepth = depth });
            for (int step = 0; step < 3; step++)
            {
                for (int i = 0; i < 15; i++) p.T.MutableGrad[i] = MathF.Sin(i * .37f + step);
                optimizer.step();
            }
            return (p.T.Data.ToArray(), optimizer.CaptureState());
        }
        var cpu = Run(false); var arc = Run(true);
        Assert.InRange(MathF.Abs(cpu.State.ParameterStates[0].Confidence - arc.State.ParameterStates[0].Confidence), 0, 1e-5f);
        for (int i = 0; i < 15; i++) Assert.InRange(MathF.Abs(cpu.Values[i] - arc.Values[i]), 0, 3e-5f);
    }

    [Theory]
    [InlineData(false, 3, 5)]
    [InlineData(true, 5, 3)]
    [InlineData(true, 3, 5)]
    public void OptimizersAndClipMatchCpu(bool muon, int rows, int cols)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc OpenCL GPU is unavailable.");
        (float[] Weights, float Norm) Run(bool arc)
        {
            using IDisposable scope = arc ? Tensor.BeginArcExecution() : TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu));
            var p = new Parameter(Enumerable.Range(0, rows * cols).Select(i => .07f * i - .3f).ToArray(), [rows, cols], "w", WeightDecayPolicy.Apply);
            var opt = muon ? (IOptimizer)new NekoMuon([p], new NekoMuonOptions {
                LearningRate = .02f, WeightDecay = .01f, BetaFast = .95f, Nesterov = true,
                NewtonSchulzDepthMode = NekoMuonNewtonSchulzDepthMode.Fixed, NewtonSchulzDepth = 5 })
                : new AdamW([p], new AdamWOptions { LearningRate = .02f, WeightDecay = .01f });
            float norm = 0;
            for (int step = 0; step < 3; step++)
            {
                for (int i = 0; i < rows * cols; i++) p.T.MutableGrad[i] = MathF.Sin(i * .39f + step);
                norm = nn.utils.clip_grad_norm_([p], .8f);
                opt.step();
                // Exercise serialization/restoration without changing the optimizer leaf order.
                opt.load_state_dict(opt.state_dict());
            }
            if (opt is IDisposable disposable) disposable.Dispose();
            return (p.T.Data.ToArray(), norm);
        }
        var cpu = Run(false); var arc = Run(true);
        Assert.InRange(MathF.Abs(cpu.Norm - arc.Norm), 0, 2e-6f);
        for (int i = 0; i < cpu.Weights.Length; i++) Assert.InRange(MathF.Abs(cpu.Weights[i] - arc.Weights[i]), 0, 3e-5f);
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void TrainingAccumulationAndGenerationKeepPrecisionAndReleaseMemory(TensorPrecisionMode precision)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc OpenCL GPU is unavailable.");
        using var scope = Tensor.BeginArcExecution(precision: precision);
        var model = new GptRinWikiJp(32, 8, 8, 2, 16, 2, new Random(37), dropout: .1f, tieWordEmbeddings: true);
        model.to(precision, 32).to("arc");
        Assert.All(model.parameters(), p => Assert.Equal(TensorDevice.Arc, p.T.device.Type));
        var parameters = model.parameters().ToArray();
        var optimizer = new AdamW(parameters, new AdamWOptions { LearningRate = .003f });
        float[] before = parameters[0].T.CaptureData(preferMaster: true);
        long retainedBytes = 0;
        for (int step = 0; step < 3; step++)
        {
            optimizer.zero_grad();
            for (int micro = 0; micro < 2; micro++)
            {
                Tensor loss = model.forward_loss([1,2,3,4], [2,3,4,5], 1, 4);
                Assert.True(float.IsFinite(loss.item()));
                loss.BackwardAndRelease([.5f]);
            }
            Assert.True(float.IsFinite(nn.utils.clip_grad_norm_(parameters, 1)));
            optimizer.step();
            long expected = 4 + parameters.Sum(p => (long)p.T.StorageByteLength + p.T.Numel * 16L);
            Assert.Equal(expected, Tensor.ArcLane.AllocatedBytes);
            retainedBytes = expected;
        }
        Assert.NotEqual(before, parameters[0].T.CaptureData(preferMaster: true));
        Assert.All(parameters, p => Assert.Equal(precision.ToStorageDType(), p.T.DType));
        Assert.Equal(4, model.GenerateTokenIds([1,2], 2, temperature: 0, stopTokenId: null).Length);
        Assert.True(model.IsTraining);
        Assert.Equal(retainedBytes, Tensor.ArcLane.AllocatedBytes);
        Assert.True(Tensor.ArcLane.KernelLaunchCount > 100);
    }

    [Fact]
    public void DeviceSyntaxDoesNotAliasCuda()
    {
        Assert.Equal(TensorDevice.Arc, TorchDevice.Parse("arc").Type);
        Assert.Equal(2, TorchDevice.Parse("arc:2").Index);
        Assert.Equal("arc:2", TorchDevice.Parse("arc:2").ToString());
        Assert.Equal(TensorDevice.Cuda, TorchDevice.Parse("cuda").Type);
    }

    [Fact]
    public void OpenClExecutesOnIntelArcAndReleasesBuffers()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc OpenCL GPU is unavailable.");
        using var lane = new ArcExecutionLane();
        float[] result = new float[3];
        lane.Run("copy_scale", 3, 0, ArcExecutionLane.In(new float[] { 1, 2, 3 }), ArcExecutionLane.InOut(result), 3, 2f, 0);
        Assert.Equal(new float[] { 2, 4, 6 }, result);
        Assert.Contains("Arc", lane.Device.Name);
        Assert.Equal(0, lane.AllocatedBytes);
        Assert.Equal(1, lane.KernelLaunchCount);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(.1f)]
    public void TransformerForwardBackwardMatchesCpu(float dropout)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc OpenCL GPU is unavailable.");
        (float Loss, float[][] Grad) Run(bool arc)
        {
            using IDisposable scope = arc ? Tensor.BeginArcExecution()
                : TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu));
            var model = new GptRinWikiJp(32, 8, 8, 2, 16, 2, new Random(37), dropout: dropout);
            Tensor loss = model.forward_loss([1, 2, 3, 4, 2, 4, 5, 7], [2, 3, 4, 5, 4, 5, 7, -1], 2, 4);
            float value = loss.Data[0];
            loss.BackwardAndRelease();
            return (value, model.parameters().Select(p => p.T.Grad.ToArray()).ToArray());
        }
        var cpu = Run(false); var arc = Run(true);
        Assert.InRange(MathF.Abs(cpu.Loss - arc.Loss), 0, 3e-5f);
        Assert.Equal(cpu.Grad.Length, arc.Grad.Length);
        for (int p = 0; p < cpu.Grad.Length; p++)
            for (int i = 0; i < cpu.Grad[p].Length; i++)
                Assert.InRange(MathF.Abs(cpu.Grad[p][i] - arc.Grad[p][i]), 0, 6e-5f);
    }
}
