using System.Text.Json;
using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class AdamWStateTests
{
    [Fact]
    public void CaptureStateContainsVersionOptionsAndOrderedMoments()
    {
        var first = CreateParameter([1f, 2f], [2], "first");
        var second = CreateParameter([3f], [1], "second");
        first.T.MutableGrad[0] = 0.5f;
        first.T.MutableGrad[1] = -0.25f;
        second.T.MutableGrad[0] = 2f;
        var options = new AdamWOptions
        {
            Beta1 = 0.5f,
            Beta2 = 0.25f,
            WeightDecay = 0f,
        };
        var optimizer = new AdamW([first, second], options);

        optimizer.Step();
        AdamWState state = optimizer.CaptureState();

        Assert.Equal(AdamWState.CurrentFormatVersion, state.FormatVersion);
        Assert.Equal(1, state.Step);
        Assert.Equal(options, state.Options);
        Assert.Collection(
            state.ParameterStates,
            parameterState =>
            {
                Assert.Equal(0, parameterState.Index);
                Assert.Equal("first", parameterState.Name);
                Assert.Equal([2], parameterState.Shape);
                AssertClose([0.25f, -0.125f], parameterState.FirstMoment);
                AssertClose(
                    [0.1875f, 0.046875f],
                    parameterState.SecondMoment);
            },
            parameterState =>
            {
                Assert.Equal(1, parameterState.Index);
                Assert.Equal("second", parameterState.Name);
                Assert.Equal([1], parameterState.Shape);
                AssertClose([1f], parameterState.FirstMoment);
                AssertClose([3f], parameterState.SecondMoment);
            });
    }

    [Fact]
    public void JsonRoundTripRestoresAnIdenticalContinuation()
    {
        var sourceParameter = CreateParameter([1f], [1], "weight");
        var options = new AdamWOptions
        {
            LearningRate = 0.05f,
            Beta1 = 0.7f,
            Beta2 = 0.8f,
            Epsilon = 1e-6f,
            WeightDecay = 0f,
        };
        var source = new AdamW([sourceParameter], options);
        sourceParameter.T.MutableGrad[0] = 0.5f;
        source.Step();

        string json = JsonSerializer.Serialize(source.CaptureState());
        AdamWState restoredState =
            JsonSerializer.Deserialize<AdamWState>(json)!;
        var restoredParameter = CreateParameter(
            sourceParameter.T.Data.ToArray(),
            [1],
            "weight");
        var restored = new AdamW(
            [restoredParameter],
            new AdamWOptions { LearningRate = 9f });
        restored.RestoreState(restoredState);

        sourceParameter.T.MutableGrad[0] = -0.25f;
        restoredParameter.T.MutableGrad[0] = -0.25f;
        source.Step();
        restored.Step();

        AssertClose(sourceParameter.T.Data, restoredParameter.T.Data);
        AdamWState continuedState = restored.CaptureState();
        Assert.Equal(2, continuedState.Step);
        Assert.Equal(options, continuedState.Options);
    }

    [Fact]
    public void CaptureAndRestoreDoNotShareMomentArrays()
    {
        var sourceParameter = CreateParameter([1f], [1], "weight");
        sourceParameter.T.MutableGrad[0] = 1f;
        var source = new AdamW([sourceParameter]);
        source.Step();
        AdamWState snapshot = source.CaptureState();

        var restoredParameter = CreateParameter(
            sourceParameter.T.Data.ToArray(),
            [1],
            "weight");
        var restored = new AdamW([restoredParameter]);
        restored.RestoreState(snapshot);
        snapshot.ParameterStates[0].FirstMoment[0] = 123f;
        snapshot.ParameterStates[0].SecondMoment[0] = 456f;

        AdamWState sourceState = source.CaptureState();
        AdamWState restoredState = restored.CaptureState();
        Assert.NotEqual(123f, sourceState.ParameterStates[0].FirstMoment[0]);
        Assert.NotEqual(456f, sourceState.ParameterStates[0].SecondMoment[0]);
        Assert.NotEqual(123f, restoredState.ParameterStates[0].FirstMoment[0]);
        Assert.NotEqual(456f, restoredState.ParameterStates[0].SecondMoment[0]);
    }

    [Fact]
    public void RestoreRejectsAnUnsupportedFormatVersion()
    {
        var optimizer = new AdamW(
            [CreateParameter([1f], [1], "weight")]);
        AdamWState state = optimizer.CaptureState() with
        {
            FormatVersion = AdamWState.CurrentFormatVersion + 1,
        };

        var exception = Assert.Throws<ArgumentException>(
            () => optimizer.RestoreState(state));

        Assert.Equal("state", exception.ParamName);
        Assert.Contains("format version", exception.Message);
    }

    [Fact]
    public void RestoreRejectsAnIncompatibleParameterShape()
    {
        var optimizer = new AdamW(
            [CreateParameter([1f], [1], "weight")]);
        AdamWState captured = optimizer.CaptureState();
        AdamWParameterState incompatible =
            captured.ParameterStates[0] with { Shape = [2] };
        AdamWState state = captured with
        {
            ParameterStates = [incompatible],
        };

        var exception = Assert.Throws<ArgumentException>(
            () => optimizer.RestoreState(state));

        Assert.Equal("state", exception.ParamName);
        Assert.Contains("incompatible shape", exception.Message);
    }

    private static Parameter CreateParameter(
        float[] data,
        int[] shape,
        string name)
    {
        return new Parameter(
            data,
            shape,
            name,
            WeightDecayPolicy.Exclude);
    }
}
