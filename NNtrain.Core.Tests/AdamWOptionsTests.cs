using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class AdamWOptionsTests
{
    public static TheoryData<AdamWOptions> InvalidOptions => new()
    {
        new AdamWOptions { LearningRate = 0f },
        new AdamWOptions { LearningRate = float.NaN },
        new AdamWOptions { Beta1 = -0.1f },
        new AdamWOptions { Beta1 = 1f },
        new AdamWOptions { Beta2 = -0.1f },
        new AdamWOptions { Beta2 = 1f },
        new AdamWOptions { Epsilon = 0f },
        new AdamWOptions { Epsilon = float.PositiveInfinity },
        new AdamWOptions { WeightDecay = -0.1f },
        new AdamWOptions { WeightDecay = float.NaN },
    };

    [Fact]
    public void DefaultsUseConfiguredAdamWHyperparameters()
    {
        var options = new AdamWOptions();

        Assert.Equal(1e-3f, options.LearningRate);
        Assert.Equal(0.9f, options.Beta1);
        Assert.Equal(0.999f, options.Beta2);
        Assert.Equal(1e-8f, options.Epsilon);
        Assert.Equal(5e-2f, options.WeightDecay);
        Assert.False(options.Decay1D);
        Assert.False(options.UseBFloat16FirstMoment);
        Assert.False(options.UseBFloat16SecondMoment);
    }

    [Fact]
    public void ConstructorAcceptsOnlyParametersAndOptions()
    {
        var constructor = Assert.Single(typeof(AdamW).GetConstructors());
        var constructorParameters = constructor.GetParameters();

        Assert.Equal(
            [typeof(IEnumerable<Parameter>), typeof(AdamWOptions)],
            constructorParameters.Select(parameter => parameter.ParameterType));
        Assert.True(constructorParameters[1].HasDefaultValue);
        Assert.Null(constructorParameters[1].DefaultValue);
    }

    [Fact]
    public void CustomLearningRateAndEpsilonControlTheUpdate()
    {
        var parameter = new Parameter(
            [1f],
            [1],
            "weight",
            WeightDecayPolicy.Exclude);
        parameter.T.MutableGrad[0] = 2f;
        var optimizer = new AdamW(
            [parameter],
            new AdamWOptions
            {
                LearningRate = 0.4f,
                Epsilon = 2f,
                WeightDecay = 0f,
            });

        optimizer.Step();

        AssertClose([0.8f], parameter.T.Data, 2e-6f);
    }

    [Theory]
    [MemberData(nameof(InvalidOptions))]
    public void ConstructorRejectsOptionsThatCouldCorruptParameters(
        AdamWOptions options)
    {
        var parameter = new Parameter(
            [1f],
            [1],
            "weight",
            WeightDecayPolicy.Exclude);

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            () => new AdamW([parameter], options));

        Assert.Equal("options", exception.ParamName);
    }

    [Fact]
    public void SimdApproximationRemainsCloseAcrossManySteps()
    {
        bool previousSimd = Tensor.SimdEnabled;
        int previousParallelism = Tensor.MaxDegreeOfParallelism;
        try
        {
            const int length = 16_384;
            float[] initial = Enumerable.Range(0, length)
                .Select(index => (index % 97 - 48) * 0.001f)
                .ToArray();
            var scalarParameter = new Parameter(
                initial,
                [length],
                "weight",
                WeightDecayPolicy.Apply);
            var simdParameter = new Parameter(
                initial,
                [length],
                "weight",
                WeightDecayPolicy.Apply);
            var options = new AdamWOptions
            {
                LearningRate = 3e-4f,
                Beta1 = 0.9f,
                Beta2 = 0.95f,
                WeightDecay = 0.01f,
            };
            var scalarOptimizer = new AdamW([scalarParameter], options);
            var simdOptimizer = new AdamW([simdParameter], options);

            for (int step = 0; step < 200; step++)
            {
                Span<float> scalarGradient =
                    scalarParameter.T.MutableGrad;
                Span<float> simdGradient = simdParameter.T.MutableGrad;
                for (int index = 0; index < length; index++)
                {
                    float gradient = MathF.Sin(
                        index * 0.017f + step * 0.031f) * 0.1f;
                    scalarGradient[index] = gradient;
                    simdGradient[index] = gradient;
                }

                Tensor.SimdEnabled = false;
                Tensor.MaxDegreeOfParallelism = 1;
                scalarOptimizer.Step();
                Tensor.SimdEnabled = true;
                Tensor.MaxDegreeOfParallelism = 0;
                simdOptimizer.Step();
            }

            AssertClose(
                scalarParameter.T.Data,
                simdParameter.T.Data,
                5e-5f);
            AdamWState scalarState = scalarOptimizer.CaptureState();
            AdamWState simdState = simdOptimizer.CaptureState();
            AssertClose(
                scalarState.ParameterStates[0].FirstMoment,
                simdState.ParameterStates[0].FirstMoment,
                5e-6f);
            AssertClose(
                scalarState.ParameterStates[0].SecondMoment,
                simdState.ParameterStates[0].SecondMoment,
                5e-6f);
        }
        finally
        {
            Tensor.SimdEnabled = previousSimd;
            Tensor.MaxDegreeOfParallelism = previousParallelism;
        }
    }

    [Fact]
    public void BFloat16SecondMomentStaysCloseAndProducesPortableState()
    {
        bool previousSimd = Tensor.SimdEnabled;
        int previousParallelism = Tensor.MaxDegreeOfParallelism;
        try
        {
            const int length = 16_384;
            float[] initial = Enumerable.Range(0, length)
                .Select(index => (index % 97 - 48) * 0.001f)
                .ToArray();
            var floatParameter = new Parameter(
                initial,
                [length],
                "weight",
                WeightDecayPolicy.Apply);
            var bfloatParameter = new Parameter(
                initial,
                [length],
                "weight",
                WeightDecayPolicy.Apply);
            var floatOptimizer = new AdamW(
                [floatParameter],
                new AdamWOptions
                {
                    LearningRate = 3e-4f,
                    Beta1 = 0.9f,
                    Beta2 = 0.95f,
                    WeightDecay = 0.01f,
                });
            var bfloatOptimizer = new AdamW(
                [bfloatParameter],
                new AdamWOptions
                {
                    LearningRate = 3e-4f,
                    Beta1 = 0.9f,
                    Beta2 = 0.95f,
                    WeightDecay = 0.01f,
                    UseBFloat16SecondMoment = true,
                });
            Tensor.SimdEnabled = true;
            Tensor.MaxDegreeOfParallelism = 0;

            for (int step = 0; step < 200; step++)
            {
                Span<float> floatGradient = floatParameter.T.MutableGrad;
                Span<float> bfloatGradient = bfloatParameter.T.MutableGrad;
                for (int index = 0; index < length; index++)
                {
                    float gradient = MathF.Sin(
                        index * 0.017f + step * 0.031f) * 0.1f;
                    floatGradient[index] = gradient;
                    bfloatGradient[index] = gradient;
                }
                floatOptimizer.Step();
                bfloatOptimizer.Step();
            }

            AssertClose(
                floatParameter.T.Data,
                bfloatParameter.T.Data,
                4e-3f);
            AdamWState state = bfloatOptimizer.CaptureState();
            Assert.True(state.Options.UseBFloat16SecondMoment);
            Assert.Equal(length, state.ParameterStates[0].SecondMoment.Length);
            Assert.All(
                state.ParameterStates[0].SecondMoment,
                value => Assert.True(float.IsFinite(value) && value >= 0f));

            var restoredParameter = new Parameter(
                bfloatParameter.T.Data.ToArray(),
                [length],
                "weight",
                WeightDecayPolicy.Apply);
            var restored = new AdamW([restoredParameter]);
            restored.RestoreState(state);
            Assert.True(restored.CaptureState().Options
                .UseBFloat16SecondMoment);
        }
        finally
        {
            Tensor.SimdEnabled = previousSimd;
            Tensor.MaxDegreeOfParallelism = previousParallelism;
        }
    }

    [Fact]
    public void BFloat16MomentsStayCloseAndResumeFromPortableState()
    {
        bool previousSimd = Tensor.SimdEnabled;
        int previousParallelism = Tensor.MaxDegreeOfParallelism;
        try
        {
            const int length = 16_384;
            float[] initial = Enumerable.Range(0, length)
                .Select(index => (index % 113 - 56) * 0.001f)
                .ToArray();
            var floatParameter = new Parameter(
                initial,
                [length],
                "weight",
                WeightDecayPolicy.Apply);
            var bfloatParameter = new Parameter(
                initial,
                [length],
                "weight",
                WeightDecayPolicy.Apply);
            var floatOptimizer = new AdamW(
                [floatParameter],
                new AdamWOptions
                {
                    LearningRate = 3e-4f,
                    Beta1 = 0.9f,
                    Beta2 = 0.95f,
                    WeightDecay = 0.01f,
                });
            var bfloatOptimizer = new AdamW(
                [bfloatParameter],
                new AdamWOptions
                {
                    LearningRate = 3e-4f,
                    Beta1 = 0.9f,
                    Beta2 = 0.95f,
                    WeightDecay = 0.01f,
                    UseBFloat16FirstMoment = true,
                    UseBFloat16SecondMoment = true,
                });
            Tensor.SimdEnabled = true;
            Tensor.MaxDegreeOfParallelism = 0;

            for (int step = 0; step < 300; step++)
            {
                Span<float> floatGradient = floatParameter.T.MutableGrad;
                Span<float> bfloatGradient = bfloatParameter.T.MutableGrad;
                for (int index = 0; index < length; index++)
                {
                    float gradient = MathF.Sin(
                        index * 0.017f + step * 0.031f) * 0.1f;
                    floatGradient[index] = gradient;
                    bfloatGradient[index] = gradient;
                }
                floatOptimizer.Step();
                bfloatOptimizer.Step();
            }

            AssertClose(
                floatParameter.T.Data,
                bfloatParameter.T.Data,
                2e-4f);
            AdamWState state = bfloatOptimizer.CaptureState();
            Assert.True(state.Options.UseBFloat16FirstMoment);
            Assert.True(state.Options.UseBFloat16SecondMoment);
            Assert.Equal(length, state.ParameterStates[0].FirstMoment.Length);
            Assert.Equal(length, state.ParameterStates[0].SecondMoment.Length);

            var restoredParameter = new Parameter(
                bfloatParameter.T.Data.ToArray(),
                [length],
                "weight",
                WeightDecayPolicy.Apply);
            var restored = new AdamW([restoredParameter]);
            restored.RestoreState(state);
            Span<float> sourceGradient = bfloatParameter.T.MutableGrad;
            Span<float> restoredGradient = restoredParameter.T.MutableGrad;
            for (int index = 0; index < length; index++)
            {
                float gradient = MathF.Cos(index * 0.013f) * 0.07f;
                sourceGradient[index] = gradient;
                restoredGradient[index] = gradient;
            }

            bfloatOptimizer.Step();
            restored.Step();

            Assert.Equal(
                bfloatParameter.T.Data.ToArray(),
                restoredParameter.T.Data.ToArray());
            AdamWParameterState resumedState = bfloatOptimizer
                .CaptureState()
                .ParameterStates[0];
            AdamWParameterState restoredState = restored
                .CaptureState()
                .ParameterStates[0];
            Assert.Equal(
                resumedState.FirstMoment,
                restoredState.FirstMoment);
            Assert.Equal(
                resumedState.SecondMoment,
                restoredState.SecondMoment);
        }
        finally
        {
            Tensor.SimdEnabled = previousSimd;
            Tensor.MaxDegreeOfParallelism = previousParallelism;
        }
    }
}
