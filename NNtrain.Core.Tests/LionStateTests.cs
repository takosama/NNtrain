using System.Text.Json;
using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class LionStateTests
{
    [Fact]
    public void CaptureStateContainsVersionOptionsAndOrderedMomentum()
    {
        Parameter first = CreateParameter([1f, 2f], [2], "first");
        Parameter second = CreateParameter([3f], [1], "second");
        first.T.MutableGrad[0] = 0.5f;
        first.T.MutableGrad[1] = -0.25f;
        second.T.MutableGrad[0] = 2f;
        var options = new LionOptions
        {
            Beta1 = 0.5f,
            Beta2 = 0.25f,
            WeightDecay = 0f,
        };
        var optimizer = new Lion([first, second], options);

        optimizer.Step();
        LionState state = optimizer.CaptureState();

        Assert.Equal(LionState.CurrentFormatVersion, state.FormatVersion);
        Assert.Equal(1, state.Step);
        Assert.Equal(options, state.Options);
        Assert.Collection(
            state.ParameterStates,
            parameterState =>
            {
                Assert.Equal(0, parameterState.Index);
                Assert.Equal("first", parameterState.Name);
                Assert.Equal([2], parameterState.Shape);
                AssertClose([0.375f, -0.1875f], parameterState.Momentum);
            },
            parameterState =>
            {
                Assert.Equal(1, parameterState.Index);
                Assert.Equal("second", parameterState.Name);
                Assert.Equal([1], parameterState.Shape);
                AssertClose([1.5f], parameterState.Momentum);
            });
    }

    [Fact]
    public void JsonRoundTripRestoresAnIdenticalContinuation()
    {
        Parameter sourceParameter = CreateParameter([1f], [1], "weight");
        var options = new LionOptions
        {
            LearningRate = 0.05f,
            Beta1 = 0.7f,
            Beta2 = 0.8f,
            WeightDecay = 0f,
        };
        var source = new Lion([sourceParameter], options);
        sourceParameter.T.MutableGrad[0] = 0.5f;
        source.Step();

        string json = JsonSerializer.Serialize(source.CaptureState());
        LionState restoredState = JsonSerializer.Deserialize<LionState>(json)!;
        Parameter restoredParameter = CreateParameter(
            sourceParameter.T.Data.ToArray(),
            [1],
            "weight");
        var restored = new Lion(
            [restoredParameter],
            new LionOptions { LearningRate = 9f });
        restored.RestoreState(restoredState);

        sourceParameter.T.MutableGrad[0] = -0.25f;
        restoredParameter.T.MutableGrad[0] = -0.25f;
        source.Step();
        restored.Step();

        AssertClose(sourceParameter.T.Data, restoredParameter.T.Data);
        LionState continuedState = restored.CaptureState();
        Assert.Equal(2, continuedState.Step);
        Assert.Equal(options, continuedState.Options);
        AssertClose(
            source.CaptureState().ParameterStates[0].Momentum,
            continuedState.ParameterStates[0].Momentum);
    }

    [Fact]
    public void CaptureAndRestoreDoNotShareStateArrays()
    {
        Parameter sourceParameter = CreateParameter([1f], [1], "weight");
        sourceParameter.T.MutableGrad[0] = 1f;
        var source = new Lion([sourceParameter]);
        source.Step();
        LionState snapshot = source.CaptureState();

        Parameter restoredParameter = CreateParameter(
            sourceParameter.T.Data.ToArray(),
            [1],
            "weight");
        var restored = new Lion([restoredParameter]);
        restored.RestoreState(snapshot);
        snapshot.ParameterStates[0].Momentum[0] = 123f;
        snapshot.ParameterStates[0].Shape[0] = 9;

        LionState sourceState = source.CaptureState();
        LionState restoredState = restored.CaptureState();
        Assert.NotEqual(123f, sourceState.ParameterStates[0].Momentum[0]);
        Assert.NotEqual(123f, restoredState.ParameterStates[0].Momentum[0]);
        Assert.Equal([1], sourceState.ParameterStates[0].Shape);
        Assert.Equal([1], restoredState.ParameterStates[0].Shape);
    }

    [Fact]
    public void RestoreRejectsAnUnsupportedFormatVersion()
    {
        var optimizer = new Lion([CreateParameter([1f], [1], "weight")]);
        LionState state = optimizer.CaptureState() with
        {
            FormatVersion = LionState.CurrentFormatVersion + 1,
        };

        var exception = Assert.Throws<ArgumentException>(
            () => optimizer.RestoreState(state));

        Assert.Equal("state", exception.ParamName);
        Assert.Contains("format version", exception.Message);
    }

    [Fact]
    public void RestoreRejectsAnIncompatibleParameterShape()
    {
        var optimizer = new Lion([CreateParameter([1f], [1], "weight")]);
        LionState captured = optimizer.CaptureState();
        LionParameterState incompatible =
            captured.ParameterStates[0] with { Shape = [2] };
        LionState state = captured with
        {
            ParameterStates = [incompatible],
        };

        var exception = Assert.Throws<ArgumentException>(
            () => optimizer.RestoreState(state));

        Assert.Equal("state", exception.ParamName);
        Assert.Contains("incompatible shape", exception.Message);
    }

    [Fact]
    public void RestoreRejectsOptionsThatCouldCorruptParameters()
    {
        var optimizer = new Lion([CreateParameter([1f], [1], "weight")]);
        LionState state = optimizer.CaptureState() with
        {
            Options = new LionOptions { Beta1 = 1f },
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            () => optimizer.RestoreState(state));

        Assert.Equal("state", exception.ParamName);
    }

    [Fact]
    public void RestoreRejectsAStateThatCannotAdvanceAnotherStep()
    {
        var optimizer = new Lion([CreateParameter([1f], [1], "weight")]);
        LionState state = optimizer.CaptureState() with
        {
            Step = int.MaxValue,
        };

        var exception = Assert.Throws<ArgumentException>(
            () => optimizer.RestoreState(state));

        Assert.Equal("state", exception.ParamName);
        Assert.Contains("another optimizer step", exception.Message);
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
