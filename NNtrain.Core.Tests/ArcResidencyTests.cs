using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcResidencyTests
{
    [Fact]
    public void FailedBackwardReleasesEveryTransientAndPrecisionSwitchPreservesValues()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");
        using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_32);
        var model = new GptRinWikiJp(37, 8, 8, 2, 13, 2, new Random(11), dropout: .1f);
        model.to(TensorPrecisionMode.Mix8_32, 32);
        Tensor loss = model.forward_loss([1,2,3,4], [2,3,4,5], 1, 4);
        Parameter changed = model.parameters().First();
        changed.CompleteUpdate(); // Version change between forward and backward must reject the graph.
        Assert.Throws<InvalidOperationException>(() => loss.BackwardAndRelease());
        Assert.Equal(4 + model.parameters().Where(p => !ReferenceEquals(p, changed)).Sum(p => (long)p.T.StorageByteLength),
            Tensor.ArcLane.AllocatedBytes);
        var parameter = new Parameter([1,2,3,4], [2,2], "state", WeightDecayPolicy.Apply);
        var optimizer = new AdamW([parameter]);
        parameter.T.MutableGrad.Fill(1);
        optimizer.step();
        float[] master = parameter.T.CaptureData(true);
        parameter.T.ConvertStorageInPlace(TensorDType.BFloat16, preserveFloat32Master: true);
        Assert.Equal(master, parameter.T.CaptureData(true));
        Assert.Equal(new float[] {1,1,1,1}, parameter.T.Grad);
    }

    [Fact]
    public void NonFiniteDeviceQuantizationFailsInsteadOfSilentlyPublishingZeros()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");
        // Dispose may report the same invalid publication while attempting to preserve live values.
        var scope = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_32);
        try
        {
            var input = new Tensor([float.NaN, 1], [1,2]);
            Tensor result = input.To(TensorDType.Bfp8);
            Assert.Throws<ArithmeticException>(() => result.Data.ToArray());
        }
        finally { try { scope.Dispose(); } catch (AggregateException) { } }
    }
    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void WarmTrainingTransfersOnlyTokensTargetsAndScalarStatistics(TensorPrecisionMode precision)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");
        ArcExecutionLane lane;
        GptRinWikiJp model;
        NekoMuon muon;
        AdamW adam;
        using (Tensor.BeginArcExecution(precision: precision))
        {
            lane = Tensor.ArcLane;
            model = new GptRinWikiJp(37, 16, 8, 2, 13, 2, new Random(57), dropout: .1f, tieWordEmbeddings: true);
            model.to(precision, 32);
            muon = new NekoMuon(model.HiddenWeightParameters, new NekoMuonOptions {
                LearningRate = .001f, Nesterov = true, BetaFast = .95f,
                NewtonSchulzDepthMode = NekoMuonNewtonSchulzDepthMode.Fixed, NewtonSchulzDepth = 5 });
            adam = new AdamW(model.AuxiliaryParameters, new AdamWOptions { LearningRate = .0003f });
            int[] x = Enumerable.Range(0, 16).Select(i => i % 37).ToArray();
            int[] y = x.Select(i => (i + 1) % 37).ToArray();
            void Step()
            {
                muon.zero_grad(); adam.zero_grad();
                for (int micro = 0; micro < 2; micro++)
                {
                    Tensor loss = model.forward_loss(x, y, 1, 16);
                    Assert.True(float.IsFinite(loss.item()));
                    loss.BackwardAndRelease([.5f]);
                }
                Assert.True(float.IsFinite(nn.utils.clip_grad_norm_(model.parameters(), 1f)));
                muon.step(); adam.step();
            }
            Step();
            long retained = lane.AllocatedBytes, upload = lane.H2DBytes, download = lane.D2HBytes;
            for (int i = 0; i < 3; i++) Step();
            Assert.Equal(3 * 2 * 2 * 16 * 4, lane.H2DBytes - upload);
            // Two loss floats, norm+scale+numeric status, 4 confidence floats and an NS norm per matrix.
            Assert.Equal(3 * (2 * 4 + 12 + model.HiddenWeightParameters.Count * 20), lane.D2HBytes - download);
            Assert.Equal(retained, lane.AllocatedBytes);
            Assert.True(ArcTrainingMath.GradientsFinite(model.parameters()));
            // Checkpoint/state inspection must publish current device state, not initialization arrays.
            Assert.Contains(muon.CaptureState().ParameterStates, p => p.FastMoment.Any(v => v != 0));
            Assert.Contains(adam.CaptureState().ParameterStates, p => p.FirstMoment.Any(v => v != 0));
        }
        Assert.Equal(0, lane.AllocatedBytes);
        // Session teardown is an explicit handoff, so objects remain readable afterwards.
        Assert.All(model.parameters(), p => Assert.All(p.T.Data, x => Assert.True(float.IsFinite(x))));
        Assert.Contains(muon.CaptureState().ParameterStates, p => p.FastMoment.Any(v => v != 0));
    }

    [Fact]
    public void HostMutationZeroGradAndCpuTransitionInvalidateResidentCopies()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");
        using var scope = Tensor.BeginArcExecution();
        var p = new Parameter([1,2,3,4], [2,2], "w", WeightDecayPolicy.Apply);
        var opt = new AdamW([p]);
        for (int i = 0; i < 4; i++) p.T.MutableGrad[i] = i + 1;
        opt.step();
        float[] updated = p.T.CaptureData(true);
        using (var mutation = p.BeginUpdate()) mutation.Values.Fill(7);
        Assert.Equal(new float[] {7,7,7,7}, p.T.Data);
        p.T.ArcGradient();
        p.ZeroGrad();
        Assert.Equal(new float[4], p.T.Grad);
        p.T.MutableGrad.Fill(2);
        p.T.to(TensorDevice.Cpu);
        Assert.Equal(new float[] {2,2,2,2}, p.T.Grad);
        Assert.NotEqual(updated, p.T.Data);
        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu))) opt.step();
        Assert.All(p.T.Data, value => Assert.True(float.IsFinite(value) && value < 7));
    }
}
