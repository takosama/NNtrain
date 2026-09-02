using NNtrain;
using Xunit;

public sealed class NekoMuonResumePolicyTests
{
    [Fact]
    public void OrdinaryMuonResumeReassertsFixedPolicyAndPreservesProgress()
    {
        NekoMuon optimizer = CreateOptimizer(
            NekoMuonNewtonSchulzDepthMode.Minimum,
            1.5f);
        optimizer.SetLearningRate(0.0125f);
        NekoMuonState before = optimizer.CaptureState();
        var config = new WikiTrainingConfiguration
        {
            Optimizer = WikiTrainingConfiguration.MuonOptimizer,
        };
        using var output = new StringWriter();

        WikiLanguageModelCommand.ApplyOrdinaryMuonPolicyAfterResume(
            config,
            optimizer,
            output);

        NekoMuonState after = optimizer.CaptureState();
        Assert.Equal(before.Step, after.Step);
        Assert.Equal(0.0125f, after.Options.LearningRate);
        Assert.Equal(0.95f, after.Options.BetaFast);
        Assert.Equal(0.95f, after.Options.BetaSlow);
        Assert.Equal(0f, after.Options.Rho);
        Assert.Equal(1, after.Options.NewtonSchulzInterval);
        Assert.Equal(5, after.Options.MaxNewtonSchulzSteps);
        Assert.Equal(
            NekoMuonNewtonSchulzDepthMode.Fixed,
            after.Options.NewtonSchulzDepthMode);
        Assert.Equal(5f, after.Options.NewtonSchulzDepth);
        Assert.Contains("runtime override", output.ToString());
    }

    [Fact]
    public void OrdinaryMuonResumeRejectsAnIncompatibleOptimizerTopology()
    {
        var config = new WikiTrainingConfiguration
        {
            Optimizer = WikiTrainingConfiguration.MuonOptimizer,
        };
        var optimizer = new AdamW(
            [new Parameter(
                [1f],
                [1],
                "bias",
                WeightDecayPolicy.Exclude)]);

        Assert.Throws<InvalidDataException>(
            () => WikiLanguageModelCommand
                .ApplyOrdinaryMuonPolicyAfterResume(
                    config,
                    optimizer,
                    TextWriter.Null));
    }

    [Fact]
    public void ResumeWithoutRuntimeOverridePreservesCheckpointPolicy()
    {
        NekoMuon optimizer = CreateOptimizer(
            NekoMuonNewtonSchulzDepthMode.Minimum,
            1.5f);
        var config = new WikiTrainingConfiguration();

        WikiLanguageModelCommand
            .ApplyNekoMuonNewtonSchulzDepthPolicyOverride(
                config,
                optimizer,
                TextWriter.Null);

        Assert.Equal(
            NekoMuonNewtonSchulzDepthMode.Minimum,
            optimizer.NewtonSchulzDepthMode);
        Assert.Equal(1.5f, optimizer.NewtonSchulzDepth);
    }

    [Fact]
    public void ResumeRuntimeOverrideReplacesAndPersistsCheckpointPolicy()
    {
        NekoMuon optimizer = CreateOptimizer(
            NekoMuonNewtonSchulzDepthMode.Minimum,
            1.5f);
        var config = new WikiTrainingConfiguration
        {
            NekoMuonNewtonSchulzDepthMode = "fixed",
            NekoMuonNewtonSchulzDepth = 5f,
        };
        using var output = new StringWriter();

        WikiLanguageModelCommand
            .ApplyNekoMuonNewtonSchulzDepthPolicyOverride(
                config,
                optimizer,
                output);

        Assert.Equal(
            NekoMuonNewtonSchulzDepthMode.Fixed,
            optimizer.NewtonSchulzDepthMode);
        Assert.Equal(5f, optimizer.NewtonSchulzDepth);
        Assert.Equal(
            NekoMuonNewtonSchulzDepthMode.Fixed,
            optimizer.CaptureState().Options.NewtonSchulzDepthMode);
        Assert.Equal(
            5f,
            optimizer.CaptureState().Options.NewtonSchulzDepth);
        Assert.Contains("runtime override", output.ToString());
    }

    [Fact]
    public void ExplicitAdaptiveRuntimePolicyClearsCheckpointFixedDepth()
    {
        NekoMuon optimizer = CreateOptimizer(
            NekoMuonNewtonSchulzDepthMode.Fixed,
            5f);
        var config = new WikiTrainingConfiguration
        {
            NekoMuonNewtonSchulzDepthMode = "adaptive",
        };

        WikiLanguageModelCommand
            .ApplyNekoMuonNewtonSchulzDepthPolicyOverride(
                config,
                optimizer,
                TextWriter.Null);

        Assert.Equal(
            NekoMuonNewtonSchulzDepthMode.Adaptive,
            optimizer.NewtonSchulzDepthMode);
        Assert.Equal(0f, optimizer.NewtonSchulzDepth);
    }

    private static NekoMuon CreateOptimizer(
        NekoMuonNewtonSchulzDepthMode mode,
        float depth)
        => new(
            [new Parameter(
                [1f, 0f, 0f, 1f],
                [2, 2],
                "matrix",
                WeightDecayPolicy.Apply)],
            new NekoMuonOptions
            {
                NewtonSchulzDepthMode = mode,
                NewtonSchulzDepth = depth,
            });
}
