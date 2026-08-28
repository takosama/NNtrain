using NNtrain;
using Xunit;

public sealed class NekoMuonResumePolicyTests
{
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
