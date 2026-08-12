using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class NekoMuonTests
{
    [Fact]
    public void ConfidenceControlsFractionalNewtonSchulzDepth()
    {
        Parameter parameter = CreateParameter(
            [0f, 0f, 0f, 0f],
            [2, 2],
            WeightDecayPolicy.Exclude);
        new float[] { 1f, 0f, 0f, 1f }
            .AsSpan()
            .CopyTo(parameter.T.MutableGrad);
        var optimizer = new NekoMuon(
            [parameter],
            new NekoMuonOptions
            {
                LearningRate = 1f,
                BetaFast = 0f,
                BetaSlow = 0f,
                Rho = 0.5f,
                Epsilon = 1e-12f,
                MaxNewtonSchulzSteps = 1,
                NewtonSchulzInterval = 1,
                WeightDecay = 0f,
            });

        optimizer.Step();

        AssertClose(
            [-0.9068203f, 0f, 0f, -0.9068203f],
            parameter.T.Data,
            2e-5f);
        Assert.Equal(0.5f, optimizer.CaptureState()
            .ParameterStates[0].Confidence);
    }

    [Fact]
    public void TallMatricesAreTransposedAndReceiveMuonFinalScale()
    {
        Parameter parameter = CreateParameter(
            [0f, 0f],
            [2, 1],
            WeightDecayPolicy.Exclude);
        new float[] { 1f, 0f }
            .AsSpan()
            .CopyTo(parameter.T.MutableGrad);
        var optimizer = new NekoMuon(
            [parameter],
            new NekoMuonOptions
            {
                LearningRate = 1f,
                BetaFast = 0f,
                BetaSlow = 0f,
                Rho = 0f,
                Epsilon = 1e-12f,
                MaxNewtonSchulzSteps = 1,
                NewtonSchulzInterval = 1,
                WeightDecay = 0f,
            });

        optimizer.Step();

        AssertClose([-0.9913637f, 0f], parameter.T.Data, 2e-5f);
    }

    [Fact]
    public void FastAndSlowBiasCorrectionProduceTheFirstGradient()
    {
        Parameter parameter = CreateParameter(
            [0f, 0f],
            [1, 2],
            WeightDecayPolicy.Exclude);
        new float[] { 3f, 4f }
            .AsSpan()
            .CopyTo(parameter.T.MutableGrad);
        var optimizer = new NekoMuon(
            [parameter],
            new NekoMuonOptions
            {
                LearningRate = 0.1f,
                BetaFast = 0.5f,
                BetaSlow = 0.75f,
                Rho = 0.9f,
                MaxNewtonSchulzSteps = 1,
                WeightDecay = 0f,
            });

        optimizer.Step();
        NekoMuonParameterState state =
            optimizer.CaptureState().ParameterStates[0];

        AssertClose([1.5f, 2f], state.FastMoment);
        AssertClose([0.75f, 1f], state.SlowMoment);
        Assert.InRange(state.Confidence, 0.09999f, 0.10001f);
    }

    [Fact]
    public void PersistenceReducesConfidenceWhenFastDepartsFromSlow()
    {
        Parameter parameter = CreateParameter(
            [0f],
            [1],
            WeightDecayPolicy.Exclude);
        var optimizer = new NekoMuon(
            [parameter],
            new NekoMuonOptions
            {
                LearningRate = 0.01f,
                BetaFast = 0.5f,
                BetaSlow = 0.75f,
                Rho = 0.9f,
                Epsilon = 1e-12f,
                MaxNewtonSchulzSteps = 1,
                WeightDecay = 0f,
            });

        parameter.T.MutableGrad[0] = 1f;
        optimizer.Step();
        parameter.T.MutableGrad[0] = -1f;
        optimizer.Step();

        float confidence = optimizer.CaptureState()
            .ParameterStates[0].Confidence;
        Assert.InRange(confidence, 0.12599f, 0.12601f);
    }

    [Fact]
    public void WeightDecayUsesParameterMetadata()
    {
        Parameter decayed = CreateParameter(
            [2f],
            [1],
            WeightDecayPolicy.Apply);
        Parameter excluded = CreateParameter(
            [2f],
            [1],
            WeightDecayPolicy.Exclude);
        var optimizer = new NekoMuon(
            [decayed, excluded],
            new NekoMuonOptions
            {
                LearningRate = 0.1f,
                WeightDecay = 0.2f,
            });

        optimizer.Step();

        AssertClose([1.96f], decayed.T.Data);
        AssertClose([2f], excluded.T.Data);
    }

    [Fact]
    public void BlockedSimdNewtonSchulzMatchesScalarPath()
    {
        const int Rows = 8;
        const int Columns = 12;
        float[] initial = Enumerable.Range(0, Rows * Columns)
            .Select(index => 0.01f * MathF.Cos(index * 0.17f))
            .ToArray();
        float[] gradient = Enumerable.Range(0, Rows * Columns)
            .Select(index => MathF.Sin((index + 1) * 0.11f))
            .ToArray();
        Parameter scalar = CreateParameter(
            (float[])initial.Clone(),
            [Rows, Columns],
            WeightDecayPolicy.Exclude);
        Parameter simd = CreateParameter(
            (float[])initial.Clone(),
            [Rows, Columns],
            WeightDecayPolicy.Exclude);
        gradient.AsSpan().CopyTo(scalar.T.MutableGrad);
        gradient.AsSpan().CopyTo(simd.T.MutableGrad);
        var options = new NekoMuonOptions
        {
            LearningRate = 0.01f,
            BetaFast = 0f,
            BetaSlow = 0f,
            Rho = 0f,
            Epsilon = 1e-12f,
            MaxNewtonSchulzSteps = 2,
            NewtonSchulzInterval = 1,
            WeightDecay = 0f,
        };
        bool previousSimd = Tensor.SimdEnabled;
        int previousParallelism = Tensor.MaxDegreeOfParallelism;

        try
        {
            Tensor.MaxDegreeOfParallelism = 1;
            Tensor.SimdEnabled = false;
            new NekoMuon([scalar], options).Step();
            Tensor.SimdEnabled = true;
            new NekoMuon([simd], options).Step();

            AssertClose(scalar.T.Data, simd.T.Data, 2e-5f);
        }
        finally
        {
            Tensor.SimdEnabled = previousSimd;
            Tensor.MaxDegreeOfParallelism = previousParallelism;
        }
    }

    [Fact]
    public void NewtonSchulzRunsOnlyAtConfiguredInterval()
    {
        Parameter parameter = CreateParameter(
            [0f, 0f, 0f, 0f],
            [2, 2],
            WeightDecayPolicy.Exclude);
        var optimizer = new NekoMuon(
            [parameter],
            new NekoMuonOptions
            {
                LearningRate = 0.01f,
                BetaFast = 0f,
                BetaSlow = 0f,
                Rho = 0f,
                MaxNewtonSchulzSteps = 1,
                NewtonSchulzInterval = 5,
                WeightDecay = 0f,
            })
        {
            ProfilingEnabled = true,
        };

        for (int step = 1; step <= 5; step++)
        {
            new float[] { 1f, 0f, 0f, 1f }
                .AsSpan()
                .CopyTo(parameter.T.MutableGrad);
            float before = parameter.T.Data[0];

            optimizer.Step();

            Assert.NotEqual(before, parameter.T.Data[0]);
            if (step < 5)
                Assert.Equal(0d, optimizer.LastStepProfile.FirstGramMilliseconds);
            else
                Assert.True(optimizer.LastStepProfile.FirstGramMilliseconds > 0d);
        }
    }

    private static Parameter CreateParameter(
        float[] data,
        int[] shape,
        WeightDecayPolicy weightDecay)
    {
        return new Parameter(data, shape, "weight", weightDecay);
    }
}
