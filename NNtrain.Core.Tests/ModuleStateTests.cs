using NNtrain;
using Xunit;

public sealed class ModuleStateTests
{
    [Fact]
    public void CaptureAndRestoreRoundTripsAllParameters()
    {
        var module = new StateModule();
        ModuleState state = module.CaptureState();
        using (Tensor.DataMutation mutation = module.Weight.BeginUpdate())
            mutation.Values.Fill(9f);

        module.RestoreState(state);

        Assert.Equal([1f, 2f], module.Weight.T.Data);
    }

    [Fact]
    public void CapturedValuesDoNotAliasModelStorage()
    {
        var module = new StateModule();
        ModuleState state = module.CaptureState();

        state.Parameters[0].Values[0] = 7f;

        Assert.Equal(1f, module.Weight.T.Data[0]);
    }

    [Fact]
    public void RestoreValidatesEverySlotBeforeChangingTheModel()
    {
        var module = new StateModule();
        ModuleState state = module.CaptureState();
        state.Parameters[0].Values[0] = 8f;
        state.Parameters[1] = state.Parameters[1] with { Shape = [2] };

        Assert.Throws<ArgumentException>(() => module.RestoreState(state));
        Assert.Equal([1f, 2f], module.Weight.T.Data);
    }

    private sealed class StateModule : Module
    {
        internal StateModule()
        {
            Weight = RegisterParameter(
                new Parameter(
                    [1f, 2f],
                    [2],
                    "Weight",
                    WeightDecayPolicy.Apply));
            Bias = RegisterParameter(
                new Parameter(
                    [3f],
                    [1],
                    "Bias",
                    WeightDecayPolicy.Exclude));
        }

        internal Parameter Weight { get; }
        internal Parameter Bias { get; }
    }
}
