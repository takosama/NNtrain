using System.Text.Json;
using NNtrain;
using Xunit;

public sealed class ModuleStateDTypeTests
{
    [Fact]
    public void CaptureRecordsPhysicalDTypeAndExactFloat32MasterValues()
    {
        var module = new DTypeStateModule(TensorDType.Float16);

        ModuleParameterState state = Assert.Single(
            module.CaptureState().Parameters);

        Assert.Equal(TensorDType.Float16, state.DType);
        Assert.Equal([0.1f, -0.2f], state.Values);
        Assert.NotEqual(module.Weight.T.Data[0], state.Values[0]);
    }

    [Fact]
    public void RestoreConvertsFloat16StateToFloat32Target()
    {
        var module = new DTypeStateModule(TensorDType.Float32);
        ModuleState state = new(
            ModuleState.CurrentFormatVersion,
            [
                new ModuleParameterState(
                    0,
                    "Weight",
                    [2],
                    [0.1f, -0.2f],
                    TensorDType.Float16),
            ]);

        module.RestoreState(state);

        Assert.Equal(TensorDType.Float32, module.Weight.T.DType);
        Assert.Equal([0.1f, -0.2f], module.Weight.T.Data);
    }

    [Fact]
    public void RestoreConvertsFloat32StateToFloat16Target()
    {
        var module = new DTypeStateModule(TensorDType.Float16);
        ModuleState state = new(
            ModuleState.CurrentFormatVersion,
            [
                new ModuleParameterState(
                    0,
                    "Weight",
                    [2],
                    [MathF.PI, -0.33333334f],
                    TensorDType.Float32),
            ]);

        module.RestoreState(state);

        Assert.Equal(TensorDType.Float16, module.Weight.T.DType);
        Assert.Equal(
            [(float)(Half)MathF.PI, (float)(Half)(-0.33333334f)],
            module.Weight.T.Data);
        Assert.Equal(
            [MathF.PI, -0.33333334f],
            module.CaptureState().Parameters[0].Values);
    }

    [Fact]
    public void LegacyJsonWithoutDTypeDefaultsToFloat32()
    {
        const string json =
            "{\"Index\":0,\"Name\":\"Weight\",\"Shape\":[1]," +
            "\"Values\":[1.25]}";

        ModuleParameterState? state =
            JsonSerializer.Deserialize<ModuleParameterState>(json);

        Assert.NotNull(state);
        Assert.Equal(TensorDType.Float32, state.DType);
        Assert.Equal([1.25f], state.Values);
    }

    [Fact]
    public void RestoreRejectsUnsupportedDTypeBeforeChangingParameters()
    {
        var module = new DTypeStateModule(TensorDType.Float32);
        ModuleState state = new(
            ModuleState.CurrentFormatVersion,
            [
                new ModuleParameterState(
                    0,
                    "Weight",
                    [2],
                    [7f, 8f],
                    TensorDType.Float8E4M3Fn),
            ]);

        Assert.Throws<ArgumentException>(() => module.RestoreState(state));
        Assert.Equal([0.1f, -0.2f], module.Weight.T.Data);
    }

    [Fact]
    public void RestoreRejectsFloat32ValuesOutsideFloat16TargetRangeAtomically()
    {
        var module = new DTypeStateModule(TensorDType.Float16);
        ModuleState state = new(
            ModuleState.CurrentFormatVersion,
            [
                new ModuleParameterState(
                    0,
                    "Weight",
                    [2],
                    [float.MaxValue, -0.2f],
                    TensorDType.Float32),
            ]);
        ModuleState before = module.CaptureState();

        Assert.Throws<ArgumentException>(() => module.RestoreState(state));

        Assert.Equal(
            before.Parameters[0].Values,
            module.CaptureState().Parameters[0].Values);
        Assert.Equal([(float)(Half)0.1f, (float)(Half)(-0.2f)],
            module.Weight.T.Data);
    }

    private sealed class DTypeStateModule : Module
    {
        internal DTypeStateModule(TensorDType dtype)
            : base(dtype)
        {
            Weight = RegisterParameter(
                new Parameter(
                    [0.1f, -0.2f],
                    [2],
                    "Weight",
                    WeightDecayPolicy.Apply,
                    dtype));
        }

        internal Parameter Weight { get; }
    }
}
