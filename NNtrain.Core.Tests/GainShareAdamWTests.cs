using NNtrain;
using Xunit;

public sealed class GainShareAdamWTests
{
    [Fact]
    public void GainSharingFavorsAlignedGroupAndPreservesUpdateNorm()
    {
        Parameter first = CreateParameter(0f, WeightDecayPolicy.Exclude);
        Parameter second = CreateParameter(0f, WeightDecayPolicy.Exclude);
        first.T.MutableGrad[0] = 1f;
        second.T.MutableGrad[0] = 4f;
        var optimizer = new GainShareAdamW(
            [[first], [second]],
            new GainShareAdamWOptions
            {
                LearningRate = 1f,
                Beta1 = 0f,
                Beta2 = 0f,
                Epsilon = 1e-8f,
                Rho = 0.95f,
                Gamma = 1f,
                MinScale = 0.5f,
                MaxScale = 2f,
                WeightDecay = 0f,
            });

        optimizer.Step();

        float firstUpdate = -first.T.Data[0];
        float secondUpdate = -second.T.Data[0];
        Assert.True(secondUpdate > firstUpdate);
        Assert.InRange(
            firstUpdate * firstUpdate + secondUpdate * secondUpdate,
            1.9999f,
            2.0001f);
    }

    [Fact]
    public void WeightDecayUsesParameterMetadata()
    {
        Parameter decayed = CreateParameter(2f, WeightDecayPolicy.Apply);
        Parameter excluded = CreateParameter(2f, WeightDecayPolicy.Exclude);
        var optimizer = new GainShareAdamW(
            [[decayed, excluded]],
            new GainShareAdamWOptions
            {
                LearningRate = 0.1f,
                WeightDecay = 0.2f,
            });

        optimizer.Step();

        Assert.Equal(1.96f, decayed.T.Data[0], 5);
        Assert.Equal(2f, excluded.T.Data[0]);
    }

    [Fact]
    public void StateRoundTripContinuesWithTheSameUpdate()
    {
        Parameter sourceFirst = CreateParameter(
            1f,
            WeightDecayPolicy.Exclude);
        Parameter sourceSecond = CreateParameter(
            -1f,
            WeightDecayPolicy.Exclude);
        var source = new GainShareAdamW(
            [[sourceFirst], [sourceSecond]],
            CreateOptions());
        sourceFirst.T.MutableGrad[0] = 1f;
        sourceSecond.T.MutableGrad[0] = 2f;
        source.Step();
        GainShareAdamWState snapshot = source.CaptureState();

        Parameter restoredFirst = CreateParameter(
            sourceFirst.T.Data[0],
            WeightDecayPolicy.Exclude);
        Parameter restoredSecond = CreateParameter(
            sourceSecond.T.Data[0],
            WeightDecayPolicy.Exclude);
        var restored = new GainShareAdamW(
            [[restoredFirst], [restoredSecond]],
            CreateOptions());
        restored.RestoreState(snapshot);
        sourceFirst.T.MutableGrad[0] = -0.5f;
        sourceSecond.T.MutableGrad[0] = 0.25f;
        restoredFirst.T.MutableGrad[0] = -0.5f;
        restoredSecond.T.MutableGrad[0] = 0.25f;

        source.Step();
        restored.Step();

        Assert.Equal(sourceFirst.T.Data[0], restoredFirst.T.Data[0]);
        Assert.Equal(sourceSecond.T.Data[0], restoredSecond.T.Data[0]);
        Assert.Equal(
            source.CaptureState().GroupStates[0].AlignmentEma,
            restored.CaptureState().GroupStates[0].AlignmentEma);
    }

    [Fact]
    public void RejectsDuplicateAndEmptyGroups()
    {
        Parameter parameter = CreateParameter(
            1f,
            WeightDecayPolicy.Apply);

        Assert.Throws<ArgumentException>(
            () => new GainShareAdamW([[parameter], [parameter]]));
        Assert.Throws<ArgumentException>(
            () => new GainShareAdamW([Array.Empty<Parameter>()]));
    }

    [Fact]
    public void CapturedStateIsDefensiveAndRejectsIncompatibleGroups()
    {
        Parameter first = CreateParameter(1f, WeightDecayPolicy.Apply);
        Parameter second = CreateParameter(2f, WeightDecayPolicy.Apply);
        var optimizer = new GainShareAdamW([[first], [second]]);
        GainShareAdamWState snapshot = optimizer.CaptureState();

        snapshot.ParameterStates[0].FirstMoment[0] = 9f;
        snapshot.GroupStates[0].ParameterIndices[0] = 1;

        GainShareAdamWState current = optimizer.CaptureState();
        Assert.Equal(0f, current.ParameterStates[0].FirstMoment[0]);
        Assert.Equal([0], current.GroupStates[0].ParameterIndices);
        Assert.Throws<ArgumentException>(
            () => optimizer.RestoreState(snapshot));
        Assert.Throws<ArgumentException>(
            () => optimizer.RestoreState(current with
            {
                Step = int.MaxValue,
            }));
    }

    private static GainShareAdamWOptions CreateOptions()
        => new()
        {
            LearningRate = 0.01f,
            WeightDecay = 0f,
        };

    private static Parameter CreateParameter(
        float value,
        WeightDecayPolicy weightDecay)
        => new([value], [1], "weight", weightDecay);
}
