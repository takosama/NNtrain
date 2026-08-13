using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class OptimizerFloat16Tests
{
    [Fact]
    public void ParameterKeepsExactFloat32MasterBehindFloat16Storage()
    {
        float[] initial = [1.0001f, -0.33337f, 0.10003f];
        var parameter = new Parameter(
            initial,
            [initial.Length],
            "weight",
            WeightDecayPolicy.Exclude,
            TensorDType.Float16);

        Assert.Equal(TensorDType.Float16, parameter.T.DType);
        AssertClose(initial, parameter.T.CaptureData(preferMaster: true));
        AssertShadowMatchesMaster(parameter);
        Assert.NotEqual(initial[0], parameter.T.Data[0]);
    }

    [Fact]
    public void AdamWUpdatesFloat32MasterAndPublishesFloat16OncePerStep()
    {
        const int length = 16;
        float[] initial = Enumerable.Range(0, length)
            .Select(index => (index - 7) * 0.01337f)
            .ToArray();
        (Parameter reference, Parameter half) = CreatePair(
            initial,
            [length]);
        var options = new AdamWOptions
        {
            LearningRate = 0.01f,
            Beta1 = 0.8f,
            Beta2 = 0.9f,
            Epsilon = 1e-8f,
            WeightDecay = 0.02f,
            UseBFloat16FirstMoment = true,
            UseBFloat16SecondMoment = true,
        };
        var referenceOptimizer = new AdamW([reference], options);
        var halfOptimizer = new AdamW([half], options);
        long originalVersion = half.T.DataVersion;

        for (int step = 0; step < 4; step++)
        {
            float[] gradient = Enumerable.Range(0, length)
                .Select(index => MathF.Sin(index * 0.17f + step * 0.23f))
                .ToArray();
            SetGradient(reference, gradient);
            SetGradient(half, gradient);

            referenceOptimizer.Step();
            halfOptimizer.Step();

            AssertMasterMatchesReference(reference, half);
            AssertShadowMatchesMaster(half);
        }

        Assert.Equal(originalVersion + 4, half.T.DataVersion);
        AdamWState referenceState = referenceOptimizer.CaptureState();
        AdamWState halfState = halfOptimizer.CaptureState();
        Assert.True(halfState.Options.UseBFloat16FirstMoment);
        Assert.True(halfState.Options.UseBFloat16SecondMoment);
        AssertClose(
            referenceState.ParameterStates[0].FirstMoment,
            halfState.ParameterStates[0].FirstMoment);
        AssertClose(
            referenceState.ParameterStates[0].SecondMoment,
            halfState.ParameterStates[0].SecondMoment);
    }

    [Fact]
    public void ParallelAdamWSynchronizesMultiChunkFloat16ParameterOnce()
    {
        const int length = 131_073;
        float[] initial = Enumerable.Range(0, length)
            .Select(index => (index % 31 - 15) * 0.00103f)
            .ToArray();
        var parameter = new Parameter(
            initial,
            [length],
            "weight",
            WeightDecayPolicy.Exclude,
            TensorDType.Float16);
        for (int index = 0; index < length; index++)
            parameter.T.MutableGrad[index] = (index % 13 - 6) * 0.002f;
        var optimizer = new AdamW(
            [parameter],
            new AdamWOptions
            {
                LearningRate = 0.01f,
                WeightDecay = 0f,
            });
        long originalVersion = parameter.T.DataVersion;

        optimizer.Step();

        Assert.Equal(originalVersion + 1, parameter.T.DataVersion);
        AssertShadowMatchesMaster(parameter);
    }

    [Fact]
    public void LionUpdatesFloat32MasterAndPublishesFloat16()
    {
        float[] initial = Enumerable.Range(0, 16)
            .Select(index => (index - 8) * 0.02117f)
            .ToArray();
        (Parameter reference, Parameter half) = CreatePair(initial, [16]);
        var options = new LionOptions
        {
            LearningRate = 0.003f,
            Beta1 = 0.8f,
            Beta2 = 0.9f,
            WeightDecay = 0.01f,
        };
        var referenceOptimizer = new Lion([reference], options);
        var halfOptimizer = new Lion([half], options);
        long originalVersion = half.T.DataVersion;

        for (int step = 0; step < 3; step++)
        {
            float[] gradient = Enumerable.Range(0, 16)
                .Select(index => MathF.Cos(index * 0.11f + step))
                .ToArray();
            SetGradient(reference, gradient);
            SetGradient(half, gradient);
            referenceOptimizer.Step();
            halfOptimizer.Step();
        }

        AssertMasterMatchesReference(reference, half);
        AssertShadowMatchesMaster(half);
        Assert.Equal(originalVersion + 3, half.T.DataVersion);
        AssertClose(
            referenceOptimizer.CaptureState().ParameterStates[0].Momentum,
            halfOptimizer.CaptureState().ParameterStates[0].Momentum);
    }

    [Fact]
    public void NekoMuonUpdatesFloat32MasterAndPublishesFloat16()
    {
        float[] initial = Enumerable.Range(0, 8)
            .Select(index => (index - 3) * 0.01731f)
            .ToArray();
        (Parameter reference, Parameter half) = CreatePair(initial, [2, 4]);
        var options = new NekoMuonOptions
        {
            LearningRate = 0.005f,
            BetaFast = 0.5f,
            BetaSlow = 0.75f,
            Rho = 0.5f,
            Epsilon = 1e-7f,
            MaxNewtonSchulzSteps = 2,
            NewtonSchulzInterval = 1,
            WeightDecay = 0.01f,
        };
        var referenceOptimizer = new NekoMuon([reference], options);
        var halfOptimizer = new NekoMuon([half], options);
        long originalVersion = half.T.DataVersion;

        for (int step = 0; step < 2; step++)
        {
            float[] gradient = Enumerable.Range(0, 8)
                .Select(index => MathF.Sin(index * 0.31f + step * 0.7f))
                .ToArray();
            SetGradient(reference, gradient);
            SetGradient(half, gradient);
            referenceOptimizer.Step();
            halfOptimizer.Step();
        }

        AssertMasterMatchesReference(reference, half, 2e-6f);
        AssertShadowMatchesMaster(half);
        Assert.Equal(originalVersion + 2, half.T.DataVersion);
        NekoMuonState referenceState = referenceOptimizer.CaptureState();
        NekoMuonState halfState = halfOptimizer.CaptureState();
        AssertClose(
            referenceState.ParameterStates[0].FastMoment,
            halfState.ParameterStates[0].FastMoment);
        AssertClose(
            referenceState.ParameterStates[0].SlowMoment,
            halfState.ParameterStates[0].SlowMoment);
    }

    [Fact]
    public void GainShareAdamWUpdatesFloat32MastersAndPublishesFloat16()
    {
        float[] firstInitial = [0.10123f, -0.20457f, 0.30991f, -0.40321f];
        float[] secondInitial = [-0.50123f, 0.60457f, -0.70991f, 0.80321f];
        (Parameter firstReference, Parameter firstHalf) = CreatePair(
            firstInitial,
            [4],
            "first");
        (Parameter secondReference, Parameter secondHalf) = CreatePair(
            secondInitial,
            [4],
            "second");
        var options = new GainShareAdamWOptions
        {
            LearningRate = 0.005f,
            Beta1 = 0.8f,
            Beta2 = 0.9f,
            Rho = 0.5f,
            Gamma = 0.75f,
            WeightDecay = 0.01f,
        };
        var referenceOptimizer = new GainShareAdamW(
            [[firstReference], [secondReference]],
            options);
        var halfOptimizer = new GainShareAdamW(
            [[firstHalf], [secondHalf]],
            options);
        long firstVersion = firstHalf.T.DataVersion;
        long secondVersion = secondHalf.T.DataVersion;

        for (int step = 0; step < 3; step++)
        {
            float[] firstGradient = Enumerable.Range(0, 4)
                .Select(index => (index + 1) * (step + 1) * 0.1f)
                .ToArray();
            float[] secondGradient = Enumerable.Range(0, 4)
                .Select(index => (4 - index) * (step + 1) * -0.07f)
                .ToArray();
            SetGradient(firstReference, firstGradient);
            SetGradient(firstHalf, firstGradient);
            SetGradient(secondReference, secondGradient);
            SetGradient(secondHalf, secondGradient);
            referenceOptimizer.Step();
            halfOptimizer.Step();
        }

        AssertMasterMatchesReference(firstReference, firstHalf, 2e-6f);
        AssertMasterMatchesReference(secondReference, secondHalf, 2e-6f);
        AssertShadowMatchesMaster(firstHalf);
        AssertShadowMatchesMaster(secondHalf);
        Assert.Equal(firstVersion + 3, firstHalf.T.DataVersion);
        Assert.Equal(secondVersion + 3, secondHalf.T.DataVersion);
    }

    [Fact]
    public void CompositeOptimizerPublishesEachFloat16ParameterOnce()
    {
        var adamParameter = new Parameter(
            [0.12345f],
            [1],
            "adam",
            WeightDecayPolicy.Exclude,
            TensorDType.Float16);
        var lionParameter = new Parameter(
            [-0.54321f],
            [1],
            "lion",
            WeightDecayPolicy.Exclude,
            TensorDType.Float16);
        adamParameter.T.MutableGrad[0] = 0.5f;
        lionParameter.T.MutableGrad[0] = -0.25f;
        long adamVersion = adamParameter.T.DataVersion;
        long lionVersion = lionParameter.T.DataVersion;
        IOptimizer optimizer = new CompositeOptimizer(
            new AdamW(
                [adamParameter],
                new AdamWOptions
                {
                    LearningRate = 0.01f,
                    WeightDecay = 0f,
                }),
            new Lion(
                [lionParameter],
                new LionOptions
                {
                    LearningRate = 0.01f,
                    WeightDecay = 0f,
                }));

        optimizer.Step();

        Assert.Equal(adamVersion + 1, adamParameter.T.DataVersion);
        Assert.Equal(lionVersion + 1, lionParameter.T.DataVersion);
        AssertShadowMatchesMaster(adamParameter);
        AssertShadowMatchesMaster(lionParameter);
    }

    [Fact]
    public void CompositeOptimizerRejectsCrossOptimizerDoubleRegistration()
    {
        var parameter = new Parameter(
            [1f],
            [1],
            "shared",
            WeightDecayPolicy.Exclude,
            TensorDType.Float16);

        ArgumentException error = Assert.Throws<ArgumentException>(
            () => new CompositeOptimizer(
                new AdamW([parameter]),
                new Lion([parameter])));

        Assert.Contains("updated twice", error.Message);
    }

    private static (Parameter Reference, Parameter Half) CreatePair(
        float[] initial,
        int[] shape,
        string name = "weight")
        => (
            new Parameter(
                initial,
                shape,
                name,
                WeightDecayPolicy.Apply),
            new Parameter(
                initial,
                shape,
                name,
                WeightDecayPolicy.Apply,
                TensorDType.Float16));

    private static void SetGradient(Parameter parameter, float[] gradient)
        => gradient.AsSpan().CopyTo(parameter.T.MutableGrad);

    private static void AssertMasterMatchesReference(
        Parameter reference,
        Parameter half,
        float tolerance = 1e-6f)
        => AssertClose(
            reference.T.Data,
            half.T.CaptureData(preferMaster: true),
            tolerance);

    private static void AssertShadowMatchesMaster(Parameter parameter)
    {
        float[] master = parameter.T.CaptureData(preferMaster: true);
        Assert.Equal(master.Length, parameter.T.Numel);
        for (int index = 0; index < master.Length; index++)
        {
            Assert.Equal(
                (float)(Half)master[index],
                parameter.T.Data[index]);
        }
    }
}
