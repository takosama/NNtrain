using System.Text.Json;
using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class NekoMuonStateTests
{
    [Fact]
    public void JsonRoundTripRestoresAnIdenticalContinuation()
    {
        Parameter sourceParameter = CreateParameter([1f, 2f], [1, 2]);
        var options = new NekoMuonOptions
        {
            LearningRate = 0.05f,
            BetaFast = 0.7f,
            BetaSlow = 0.8f,
            Rho = 0.6f,
            MaxNewtonSchulzSteps = 2,
            WeightDecay = 0f,
        };
        var source = new NekoMuon([sourceParameter], options);
        new float[] { 0.5f, -0.25f }
            .AsSpan()
            .CopyTo(sourceParameter.T.MutableGrad);
        source.Step();

        string json = JsonSerializer.Serialize(source.CaptureState());
        NekoMuonState restoredState =
            JsonSerializer.Deserialize<NekoMuonState>(json)!;
        Parameter restoredParameter = CreateParameter(
            sourceParameter.T.Data.ToArray(),
            [1, 2]);
        var restored = new NekoMuon([restoredParameter]);
        restored.RestoreState(restoredState);

        new float[] { -0.2f, 0.4f }
            .AsSpan()
            .CopyTo(sourceParameter.T.MutableGrad);
        new float[] { -0.2f, 0.4f }
            .AsSpan()
            .CopyTo(restoredParameter.T.MutableGrad);
        source.Step();
        restored.Step();

        AssertClose(sourceParameter.T.Data, restoredParameter.T.Data);
        NekoMuonState sourceState = source.CaptureState();
        NekoMuonState continuedState = restored.CaptureState();
        Assert.Equal(sourceState.Step, continuedState.Step);
        Assert.Equal(sourceState.Options, continuedState.Options);
        AssertClose(
            sourceState.ParameterStates[0].FastMoment,
            continuedState.ParameterStates[0].FastMoment);
        AssertClose(
            sourceState.ParameterStates[0].SlowMoment,
            continuedState.ParameterStates[0].SlowMoment);
        Assert.Equal(
            sourceState.ParameterStates[0].Confidence,
            continuedState.ParameterStates[0].Confidence);
    }

    [Fact]
    public void CaptureAndRestoreDoNotShareArrays()
    {
        Parameter parameter = CreateParameter([1f], [1]);
        var optimizer = new NekoMuon([parameter]);
        NekoMuonState snapshot = optimizer.CaptureState();
        var restored = new NekoMuon([CreateParameter([1f], [1])]);
        restored.RestoreState(snapshot);

        snapshot.ParameterStates[0].FastMoment[0] = 123f;
        snapshot.ParameterStates[0].SlowMoment[0] = 456f;
        snapshot.ParameterStates[0].Shape[0] = 9;

        NekoMuonParameterState state =
            restored.CaptureState().ParameterStates[0];
        Assert.Equal(0f, state.FastMoment[0]);
        Assert.Equal(0f, state.SlowMoment[0]);
        Assert.Equal([1], state.Shape);
    }

    [Fact]
    public void RestoreRejectsInvalidConfidenceAndTerminalStep()
    {
        var optimizer = new NekoMuon([CreateParameter([1f], [1])]);
        NekoMuonState captured = optimizer.CaptureState();
        NekoMuonParameterState invalidParameter =
            captured.ParameterStates[0] with { Confidence = float.NaN };

        Assert.Throws<ArgumentException>(
            () => optimizer.RestoreState(captured with
            {
                ParameterStates = [invalidParameter],
            }));
        Assert.Throws<ArgumentException>(
            () => optimizer.RestoreState(captured with
            {
                Step = int.MaxValue,
            }));
    }

    [Fact]
    public void NewtonSchulzDepthPolicyRoundTripsThroughCapturedState()
    {
        var source = new NekoMuon(
            [CreateParameter([1f, 2f], [1, 2])],
            new NekoMuonOptions
            {
                MaxNewtonSchulzSteps = 4,
            });
        source.SetNewtonSchulzDepthPolicy(
            NekoMuonNewtonSchulzDepthMode.Minimum,
            1.5f);

        NekoMuonState captured = source.CaptureState();
        var restored = new NekoMuon(
            [CreateParameter([1f, 2f], [1, 2])]);
        restored.RestoreState(captured);

        NekoMuonOptions options = restored.CaptureState().Options;
        Assert.Equal(
            NekoMuonNewtonSchulzDepthMode.Minimum,
            options.NewtonSchulzDepthMode);
        Assert.Equal(1.5f, options.NewtonSchulzDepth);
        Assert.Equal(4, options.MaxNewtonSchulzSteps);
    }

    private static Parameter CreateParameter(float[] data, int[] shape)
    {
        return new Parameter(
            data,
            shape,
            "weight",
            WeightDecayPolicy.Exclude);
    }
}
