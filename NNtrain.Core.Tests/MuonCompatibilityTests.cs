using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class MuonCompatibilityTests
{
    [Fact]
    public void FactorySelectsReferenceNesterovPolicy()
    {
        Parameter parameter = CreateParameter(
            [0.2f, -0.1f, 0.3f, 0.4f],
            [2, 2],
            WeightDecayPolicy.Apply);

        NekoMuon optimizer = Assert.IsType<NekoMuon>(optim.Muon(
            [parameter],
            lr: 0.02f,
            momentum: 0.8f,
            eps: 1e-6f,
            weight_decay: 0.03f,
            decay_1d: true));

        NekoMuonOptions options = optimizer.CaptureState().Options;
        Assert.Equal(0.02f, options.LearningRate);
        Assert.Equal(0.8f, options.BetaFast);
        Assert.Equal(0.8f, options.BetaSlow);
        Assert.True(options.Nesterov);
        Assert.Equal(0f, options.Rho);
        Assert.Equal(1e-6f, options.Epsilon);
        Assert.Equal(5, options.MaxNewtonSchulzSteps);
        Assert.Equal(1, options.NewtonSchulzInterval);
        Assert.Equal(
            NekoMuonNewtonSchulzDepthMode.Fixed,
            options.NewtonSchulzDepthMode);
        Assert.Equal(5f, options.NewtonSchulzDepth);
        Assert.Equal(0.03f, options.WeightDecay);
        Assert.True(options.Decay1D);
    }

    [Theory]
    [InlineData(-0.01f)]
    [InlineData(1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void OrdinaryPolicyRejectsInvalidMomentum(float momentum)
    {
        var optimizer = new NekoMuon(
            [CreateParameter([0f], [1], WeightDecayPolicy.Exclude)]);

        ArgumentOutOfRangeException error = Assert.Throws<
            ArgumentOutOfRangeException>(
                () => optimizer.SetOrdinaryMuonPolicy(momentum));

        Assert.Equal("momentum", error.ParamName);
    }

    [Fact]
    public void ReapplyingPolicyPreservesStepMomentsAndOtherSettings()
    {
        Parameter parameter = CreateParameter(
            [0.2f, -0.1f, 0.3f, 0.4f],
            [2, 2],
            WeightDecayPolicy.Apply);
        var optimizer = new NekoMuon(
            [parameter],
            new NekoMuonOptions
            {
                LearningRate = 0.017f,
                BetaFast = 0.4f,
                BetaSlow = 0.8f,
                Rho = 0.7f,
                Epsilon = 3e-6f,
                MaxNewtonSchulzSteps = 2,
                NewtonSchulzInterval = 3,
                WeightDecay = 0.025f,
                Decay1D = true,
            });
        new float[] { 0.7f, -0.2f, 0.1f, 0.5f }
            .AsSpan()
            .CopyTo(parameter.T.MutableGrad);
        optimizer.Step();
        NekoMuonState before = optimizer.CaptureState();

        optimizer.SetOrdinaryMuonPolicy();
        optimizer.SetOrdinaryMuonPolicy();

        NekoMuonState after = optimizer.CaptureState();
        Assert.Equal(before.Step, after.Step);
        AssertClose(
            before.ParameterStates[0].FastMoment,
            after.ParameterStates[0].FastMoment,
            0f);
        AssertClose(
            before.ParameterStates[0].SlowMoment,
            after.ParameterStates[0].SlowMoment,
            0f);
        Assert.Equal(
            before.ParameterStates[0].Confidence,
            after.ParameterStates[0].Confidence);
        Assert.Equal(before.Options.LearningRate, after.Options.LearningRate);
        Assert.Equal(before.Options.Epsilon, after.Options.Epsilon);
        Assert.Equal(before.Options.WeightDecay, after.Options.WeightDecay);
        Assert.Equal(before.Options.Decay1D, after.Options.Decay1D);
        Assert.Equal(0.95f, after.Options.BetaFast);
        Assert.Equal(0.95f, after.Options.BetaSlow);
        Assert.True(after.Options.Nesterov);
        Assert.Equal(0f, after.Options.Rho);
        Assert.Equal(5, after.Options.MaxNewtonSchulzSteps);
        Assert.Equal(1, after.Options.NewtonSchulzInterval);
        Assert.Equal(
            NekoMuonNewtonSchulzDepthMode.Fixed,
            after.Options.NewtonSchulzDepthMode);
        Assert.Equal(5f, after.Options.NewtonSchulzDepth);
    }

    [Fact]
    public void FactoryMatchesExplicitFixedFiveNekoMuonOnCpu()
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            float[] initial =
            [
                0.12f, -0.04f, 0.21f,
                -0.08f, 0.17f, 0.03f,
            ];
            Parameter factoryParameter = CreateParameter(
                initial.ToArray(),
                [2, 3],
                WeightDecayPolicy.Apply);
            Parameter explicitParameter = CreateParameter(
                initial.ToArray(),
                [2, 3],
                WeightDecayPolicy.Apply);
            const float LearningRate = 0.013f;
            const float Momentum = 0.82f;
            const float Epsilon = 2e-7f;
            const float WeightDecay = 0.021f;
            NekoMuon factory = Assert.IsType<NekoMuon>(optim.Muon(
                [factoryParameter],
                lr: LearningRate,
                momentum: Momentum,
                eps: Epsilon,
                weight_decay: WeightDecay));
            var explicitOptimizer = new NekoMuon(
                [explicitParameter],
                new NekoMuonOptions
                {
                    LearningRate = LearningRate,
                    BetaFast = Momentum,
                    BetaSlow = Momentum,
                    Nesterov = true,
                    Rho = 0f,
                    Epsilon = Epsilon,
                    MaxNewtonSchulzSteps = 5,
                    NewtonSchulzInterval = 1,
                    NewtonSchulzDepthMode =
                        NekoMuonNewtonSchulzDepthMode.Fixed,
                    NewtonSchulzDepth = 5f,
                    WeightDecay = WeightDecay,
                });
            float[][] gradients =
            [
                [0.7f, -0.2f, 0.1f, 0.5f, -0.3f, 0.4f],
                [-0.1f, 0.6f, -0.4f, 0.2f, 0.5f, -0.7f],
                [0.3f, 0.1f, -0.6f, 0.8f, -0.2f, 0.4f],
            ];

            foreach (float[] gradient in gradients)
            {
                gradient.AsSpan().CopyTo(factoryParameter.T.MutableGrad);
                gradient.AsSpan().CopyTo(explicitParameter.T.MutableGrad);
                factory.Step();
                explicitOptimizer.Step();
            }

            AssertClose(
                explicitParameter.T.Data,
                factoryParameter.T.Data,
                1e-7f);
            NekoMuonState expected = explicitOptimizer.CaptureState();
            NekoMuonState actual = factory.CaptureState();
            Assert.Equal(expected.Step, actual.Step);
            AssertClose(
                expected.ParameterStates[0].FastMoment,
                actual.ParameterStates[0].FastMoment,
                1e-7f);
            AssertClose(
                expected.ParameterStates[0].SlowMoment,
                actual.ParameterStates[0].SlowMoment,
                1e-7f);
            Assert.Equal(
                expected.ParameterStates[0].Confidence,
                actual.ParameterStates[0].Confidence);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void NesterovDirectionDiffersFromDirectMomentumAfterDirectionChange()
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            float[] initial = [0.04f, -0.02f, 0.03f, 0.01f];
            Parameter nesterovParameter = CreateParameter(
                initial.ToArray(), [2, 2], WeightDecayPolicy.Exclude);
            Parameter directParameter = CreateParameter(
                initial.ToArray(), [2, 2], WeightDecayPolicy.Exclude);
            NekoMuon nesterov = Assert.IsType<NekoMuon>(optim.Muon(
                [nesterovParameter],
                lr: 0.02f,
                momentum: 0.8f,
                nesterov: true,
                weight_decay: 0f));
            NekoMuon direct = Assert.IsType<NekoMuon>(optim.Muon(
                [directParameter],
                lr: 0.02f,
                momentum: 0.8f,
                nesterov: false,
                weight_decay: 0f));
            float[][] gradients =
            [
                [0.8f, 0.1f, -0.2f, 0.4f],
                [-0.3f, 0.9f, 0.5f, -0.1f],
            ];
            foreach (float[] gradient in gradients)
            {
                gradient.AsSpan().CopyTo(nesterovParameter.T.MutableGrad);
                gradient.AsSpan().CopyTo(directParameter.T.MutableGrad);
                nesterov.Step();
                direct.Step();
            }

            Assert.False(nesterovParameter.T.Data.SequenceEqual(
                directParameter.T.Data));
            AssertClose(
                nesterov.CaptureState().ParameterStates[0].FastMoment,
                direct.CaptureState().ParameterStates[0].FastMoment,
                1e-7f);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void NesterovPolicyPersistsThroughStateRestoreAndCanBeReasserted()
    {
        Parameter sourceParameter = CreateParameter(
            [0.1f, -0.2f, 0.3f, 0.4f],
            [2, 2],
            WeightDecayPolicy.Exclude);
        NekoMuon source = Assert.IsType<NekoMuon>(optim.Muon(
            [sourceParameter], momentum: 0.87f));
        new float[] { 0.2f, 0.5f, -0.4f, 0.7f }
            .AsSpan()
            .CopyTo(sourceParameter.T.MutableGrad);
        source.Step();
        NekoMuonState checkpoint = source.CaptureState();

        Parameter restoredParameter = CreateParameter(
            sourceParameter.T.Data.ToArray(),
            [2, 2],
            WeightDecayPolicy.Exclude);
        var restored = new NekoMuon([restoredParameter]);
        restored.RestoreState(checkpoint);

        Assert.True(restored.CaptureState().Options.Nesterov);
        restored.SetOrdinaryMuonPolicy();
        NekoMuonState reapplied = restored.CaptureState();
        Assert.True(reapplied.Options.Nesterov);
        Assert.Equal(0.95f, reapplied.Options.BetaFast);
        Assert.Equal(checkpoint.Step, reapplied.Step);
        AssertClose(
            checkpoint.ParameterStates[0].FastMoment,
            reapplied.ParameterStates[0].FastMoment,
            0f);
    }

    private static Parameter CreateParameter(
        float[] data,
        int[] shape,
        WeightDecayPolicy weightDecay)
        => new(data, shape, "weight", weightDecay);
}
