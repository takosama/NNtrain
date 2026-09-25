using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMix8_16Tests
{
    [Fact]
    public void AdamWPublishesBFloat16GradientMasterAndMoments()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");
        var mixed = RunAdam(TensorPrecisionMode.Mix8_16);
        var baseline = RunAdam(TensorPrecisionMode.Mix8_32);

        Assert.All(mixed.Gradient, AssertBFloat16);
        Assert.All(mixed.Master, AssertBFloat16);
        Assert.All(mixed.FirstMoment, AssertBFloat16);
        Assert.All(mixed.SecondMoment, AssertBFloat16);
        Assert.Contains(mixed.Master.Zip(baseline.Master),
            pair => pair.First != pair.Second);
        Assert.Contains(mixed.Gradient.Zip(baseline.Gradient),
            pair => pair.First != pair.Second);
    }

    [Fact]
    public void MuonPublishesBFloat16GradientMasterAndMoments()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");
        using var execution = Tensor.BeginArcExecution(
            precision: TensorPrecisionMode.Mix8_16);
        Parameter parameter = NewParameter();
        var optimizer = new NekoMuon([parameter], new NekoMuonOptions
        {
            LearningRate = 0.003f,
            BetaFast = 0.8f,
            BetaSlow = 0.9f,
            Nesterov = true,
            NewtonSchulzDepthMode = NekoMuonNewtonSchulzDepthMode.Fixed,
            NewtonSchulzDepth = 2f,
        });
        FillGradient(parameter, 0);
        optimizer.step();

        Assert.All(parameter.T.Grad, AssertBFloat16);
        Assert.All(parameter.T.CaptureData(preferMaster: true), AssertBFloat16);
        NekoMuonParameterState state = optimizer.CaptureState().ParameterStates[0];
        Assert.All(state.FastMoment, AssertBFloat16);
        Assert.All(state.SlowMoment, AssertBFloat16);
    }

    [Fact]
    public void NesterovSingleReductionPreservesConfidenceAndNorm()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");
        using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_16);
        float[] direction = Enumerable.Range(0, 262_144)
            .Select(i => (i % 257 - 128) * 0.0001f).ToArray();
        var lane = Tensor.ArcLane;
        using var hat = lane.Upload(direction);

        float oldConfidence = ArcMuonMath.Confidence(lane, hat, hat,
            direction.Length, 1e-8f, out float oldSumSquares);
        float newSumSquares = ArcMuonMath.SumSquares(lane, hat, direction.Length);
        float newConfidence = ArcMuonMath.ConfidenceForIdenticalDirections(
            newSumSquares, 1e-8f);

        Assert.InRange(MathF.Abs(newConfidence - oldConfidence), 0f, 1e-5f);
        Assert.InRange(MathF.Abs(newSumSquares - oldSumSquares) / oldSumSquares, 0f, 1e-3f);
    }

    [Fact]
    public void AdamWRestoresRoundedWorkState()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");
        using var execution = Tensor.BeginArcExecution(
            precision: TensorPrecisionMode.Mix8_16);
        Parameter parameter = NewParameter();
        var optimizer = new AdamW([parameter], new AdamWOptions
        {
            LearningRate = 0.003f,
            WeightDecay = 0f,
        });
        FillGradient(parameter, 0);
        optimizer.step();
        AdamWState saved = optimizer.CaptureState();
        optimizer.RestoreState(saved);
        Assert.False(optimizer.CaptureState().Options.UseBFloat16FirstMoment);
        Assert.False(optimizer.CaptureState().Options.UseBFloat16SecondMoment);
        FillGradient(parameter, 1);
        optimizer.step();

        AdamWParameterState state = optimizer.CaptureState().ParameterStates[0];
        Assert.All(state.FirstMoment, AssertBFloat16);
        Assert.All(state.SecondMoment, AssertBFloat16);
        Assert.All(parameter.T.CaptureData(preferMaster: true), AssertBFloat16);
    }

    [Fact]
    public void InitialMasterIsRoundedBeforeFirstUpdate()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");
        using var execution = Tensor.BeginArcExecution(
            precision: TensorPrecisionMode.Mix8_16);
        Parameter parameter = NewParameter();
        float[] original = parameter.T.CaptureData(preferMaster: true);
        using ArcExecutionLane.ArcBuffer master = parameter.T.ArcBFloat16Master().Borrow();
        Assert.Equal(parameter.T.Numel * sizeof(ushort), master.ByteLength);
        var packed = new ushort[parameter.T.Numel];
        Tensor.ArcLane.ReadRaw(master, packed);
        float[] rounded = packed.Select(bits => BitConverter.UInt32BitsToSingle((uint)bits << 16)).ToArray();

        Assert.Contains(original.Zip(rounded), pair => pair.First != pair.Second);
        Assert.All(rounded, AssertBFloat16);
    }

    [Fact]
    public void GradientAndOptimizerMomentsUseTwoBytesPerValueOnArc()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");
        using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_16);
        Parameter parameter = NewParameter();
        FillGradient(parameter, 0);
        using ArcExecutionLane.ArcBuffer gradient = parameter.T.ArcBFloat16Gradient().Borrow();
        Assert.Equal(parameter.T.Numel * sizeof(ushort), gradient.ByteLength);

        using var cache = new ArcOptimizerStateCache();
        var firstMoment = new float[parameter.T.Numel];
        ArcExecutionLane.ArcBuffer moment = cache.GetBFloat16(firstMoment);
        Assert.Equal(parameter.T.Numel * sizeof(ushort), moment.ByteLength);
    }

    private static (float[] Gradient, float[] Master,
        float[] FirstMoment, float[] SecondMoment) RunAdam(
        TensorPrecisionMode precision)
    {
        using var execution = Tensor.BeginArcExecution(precision: precision);
        Parameter parameter = NewParameter();
        var optimizer = new AdamW([parameter], new AdamWOptions
        {
            LearningRate = 0.003f,
            WeightDecay = 0f,
        });
        FillGradient(parameter, 0);
        optimizer.step();
        AdamWParameterState state = optimizer.CaptureState().ParameterStates[0];
        return (parameter.T.Grad.ToArray(),
            parameter.T.CaptureData(preferMaster: true),
            state.FirstMoment, state.SecondMoment);
    }

    private static Parameter NewParameter()
    {
        float[] values = Enumerable.Range(0, 16)
            .Select(i => 0.1234567f + i * 0.013579f).ToArray();
        var parameter = new Parameter(values, [4, 4], "mixed", WeightDecayPolicy.Apply);
        parameter.T.ConvertStorageInPlace(TensorDType.Bfp8,
            Bfp8QuantizationDescriptor.Block(32), preserveFloat32Master: true);
        return parameter;
    }

    private static void FillGradient(Parameter parameter, int step)
    {
        for (int i = 0; i < parameter.T.Numel; i++)
            parameter.T.MutableGrad[i] = 0.035679f + 0.011319f * i + 0.00473f * step;
    }

    private static void AssertBFloat16(float value)
    {
        Assert.True(float.IsFinite(value));
        Assert.Equal(value, TensorStorageCodec.RoundToBFloat16(value));
    }
}
