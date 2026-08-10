using System.Runtime.Intrinsics;
using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class TensorSimdCollection
{
    public const string Name = "Tensor SIMD configuration";
}

[Collection(TensorSimdCollection.Name)]
public sealed class TensorSimdTests
{
    [Fact]
    public void SimdIsEnabledByDefaultAndCanBeDisabled()
    {
        bool previous = Tensor.SimdEnabled;
        try
        {
            Assert.True(Tensor.SimdEnabled);

            Tensor.SimdEnabled = false;

            Assert.False(Tensor.SimdEnabled);
        }
        finally
        {
            Tensor.SimdEnabled = previous;
        }
    }

    [Fact]
    public void ScalarAndSimdElementwisePathsProduceEquivalentGradients()
    {
        bool previous = Tensor.SimdEnabled;
        try
        {
            ElementwiseResult scalar = RunElementwise(useSimd: false);
            ElementwiseResult simd = RunElementwise(useSimd: true);

            AssertClose(scalar.Output, simd.Output, 2e-5f);
            AssertClose(scalar.LeftGradient, simd.LeftGradient, 2e-5f);
            AssertClose(scalar.RightGradient, simd.RightGradient, 2e-5f);
        }
        finally
        {
            Tensor.SimdEnabled = previous;
        }
    }

    [Fact]
    public void ScalarAndSimdTransformerKernelsProduceEquivalentResults()
    {
        bool previous = Tensor.SimdEnabled;
        try
        {
            KernelResult scalar = RunTransformerKernels(useSimd: false);
            KernelResult simd = RunTransformerKernels(useSimd: true);

            AssertClose(scalar.Output, simd.Output, 2e-4f);
            AssertClose(scalar.InputGradient, simd.InputGradient, 4e-4f);
            AssertClose(scalar.WeightGradient, simd.WeightGradient, 4e-4f);
            AssertClose(scalar.BiasGradient, simd.BiasGradient, 4e-4f);
            AssertClose(scalar.GammaGradient, simd.GammaGradient, 4e-4f);
            AssertClose(scalar.BetaGradient, simd.BetaGradient, 4e-4f);
        }
        finally
        {
            Tensor.SimdEnabled = previous;
        }
    }

    [Fact]
    public void ScalarAndVector256BatchedBackwardProduceEquivalentResults()
    {
        bool previous = Tensor.SimdEnabled;
        try
        {
            BatchedKernelResult scalar = RunBatchedKernels(useSimd: false);
            BatchedKernelResult vector256 = RunBatchedKernels(useSimd: true);

            AssertClose(scalar.Output, vector256.Output, 2e-4f);
            AssertClose(
                scalar.QueryGradient,
                vector256.QueryGradient,
                4e-4f);
            AssertClose(
                scalar.KeyGradient,
                vector256.KeyGradient,
                4e-4f);
            AssertClose(
                scalar.ValueGradient,
                vector256.ValueGradient,
                4e-4f);
        }
        finally
        {
            Tensor.SimdEnabled = previous;
        }
    }

    [Fact]
    public void ScalarAndVector256ParallelAdamWProduceEquivalentStates()
    {
        bool previous = Tensor.SimdEnabled;
        try
        {
            AdamResult scalar = RunAdam(useSimd: false);
            AdamResult vector256 = RunAdam(useSimd: true);

            AssertClose(scalar.Parameters, vector256.Parameters, 3e-5f);
            AssertClose(scalar.FirstMoments, vector256.FirstMoments, 2e-6f);
            AssertClose(scalar.SecondMoments, vector256.SecondMoments, 2e-6f);
        }
        finally
        {
            Tensor.SimdEnabled = previous;
        }
    }

    [Fact]
    public void ParallelWorkerCountCanBeAutomaticOrExplicitlyLimited()
    {
        int previous = Tensor.MaxDegreeOfParallelism;
        try
        {
            Tensor.MaxDegreeOfParallelism = 3;

            Assert.Equal(3, Tensor.MaxDegreeOfParallelism);
            Assert.Equal(3, Tensor.EffectiveMaxDegreeOfParallelism);

            Tensor.MaxDegreeOfParallelism = 0;
            Assert.Equal(
                Environment.ProcessorCount,
                Tensor.EffectiveMaxDegreeOfParallelism);
            Assert.Throws<ArgumentOutOfRangeException>(
                () => Tensor.MaxDegreeOfParallelism = -1);
        }
        finally
        {
            Tensor.MaxDegreeOfParallelism = previous;
        }
    }

    [Fact]
    public void LargeCrossEntropyUsesMemoryBoundedSimdBackward()
    {
        bool previous = Tensor.SimdEnabled;
        try
        {
            Tensor.SimdEnabled = true;
            const int rows = 257;
            const int columns = 4096;
            var logits = new Tensor(
                new float[rows * columns],
                [rows, columns]);
            int[] labels = Enumerable.Range(0, rows)
                .Select(row => row % columns)
                .ToArray();

            Tensor loss = logits.CrossEntropyWithLogits(labels);
            loss.Backward();

            AssertClose(MathF.Log(columns), loss.Data[0], 2e-5f);
            float nonTarget = 1f / (rows * columns);
            float target = nonTarget - 1f / rows;
            AssertClose(target, logits.Grad[0], 2e-6f);
            AssertClose(nonTarget, logits.Grad[1], 2e-6f);
            int lastOffset = (rows - 1) * columns;
            AssertClose(
                target,
                logits.Grad[lastOffset + labels[^1]],
                2e-6f);
        }
        finally
        {
            Tensor.SimdEnabled = previous;
        }
    }

    [Fact]
    public void ScalarAndVector256ParallelLionProduceEquivalentStates()
    {
        bool previous = Tensor.SimdEnabled;
        try
        {
            LionResult scalar = RunLion(useSimd: false);
            LionResult vector256 = RunLion(useSimd: true);

            AssertClose(scalar.Parameters, vector256.Parameters, 2e-6f);
            AssertClose(scalar.Momenta, vector256.Momenta, 2e-6f);
        }
        finally
        {
            Tensor.SimdEnabled = previous;
        }
    }

    [Fact]
    public void ScalarAndVector256ParallelNekoMuonProduceEquivalentStates()
    {
        bool previous = Tensor.SimdEnabled;
        try
        {
            NekoMuonResult scalar = RunNekoMuon(useSimd: false);
            NekoMuonResult vector256 = RunNekoMuon(useSimd: true);

            AssertClose(scalar.Parameters, vector256.Parameters, 3e-5f);
            AssertClose(scalar.FastMoments, vector256.FastMoments, 2e-6f);
            AssertClose(scalar.SlowMoments, vector256.SlowMoments, 2e-6f);
            AssertClose(scalar.Confidences, vector256.Confidences, 2e-6f);
        }
        finally
        {
            Tensor.SimdEnabled = previous;
        }
    }

    [Fact]
    public void ScalarAndVector256GptTrainingStepsProduceEquivalentResults()
    {
        bool previous = Tensor.SimdEnabled;
        try
        {
            GptStepResult scalar = RunGptTrainingStep(useSimd: false);
            GptStepResult vector256 = RunGptTrainingStep(useSimd: true);

            AssertClose([scalar.Loss], [vector256.Loss], 3e-4f);
            AssertClose(scalar.Logits, vector256.Logits, 4e-4f);
            AssertClose(scalar.Gradients, vector256.Gradients, 6e-4f);
            AssertClose(scalar.Parameters, vector256.Parameters, 6e-4f);
        }
        finally
        {
            Tensor.SimdEnabled = previous;
        }
    }

    [Fact]
    public void SingleAndMultiThreadedGptTrainingStepsAreEquivalent()
    {
        int previous = Tensor.MaxDegreeOfParallelism;
        try
        {
            GptStepResult single = RunGptTrainingStep(
                useSimd: true,
                maxDegreeOfParallelism: 1);
            GptStepResult parallel = RunGptTrainingStep(
                useSimd: true,
                maxDegreeOfParallelism: 4);

            AssertClose([single.Loss], [parallel.Loss], 3e-4f);
            AssertClose(single.Logits, parallel.Logits, 4e-4f);
            AssertClose(single.Gradients, parallel.Gradients, 6e-4f);
            AssertClose(single.Parameters, parallel.Parameters, 6e-4f);
        }
        finally
        {
            Tensor.MaxDegreeOfParallelism = previous;
        }
    }

    [Fact]
    public void ScalarAndVector256GainShareAdamWProduceEquivalentStates()
    {
        bool previous = Tensor.SimdEnabled;
        try
        {
            GainShareResult scalar = RunGainShare(useSimd: false);
            GainShareResult vector256 = RunGainShare(useSimd: true);

            AssertClose(scalar.Parameters, vector256.Parameters, 2e-5f);
            AssertClose(scalar.FirstMoments, vector256.FirstMoments, 2e-6f);
            AssertClose(scalar.SecondMoments, vector256.SecondMoments, 2e-6f);
            AssertClose(scalar.AlignmentEmas, vector256.AlignmentEmas, 2e-5f);
        }
        finally
        {
            Tensor.SimdEnabled = previous;
        }
    }

    private static ElementwiseResult RunElementwise(bool useSimd)
    {
        Tensor.SimdEnabled = useSimd;
        int length = Vector256<float>.Count * 2 + 3;
        float[] leftValues = Enumerable.Range(0, length)
            .Select(index => 0.25f + index * 0.01f)
            .ToArray();
        float[] rightValues = Enumerable.Range(0, length)
            .Select(index => 1.5f + index * 0.02f)
            .ToArray();
        var left = Tensor.From1D(leftValues);
        var right = Tensor.From1D(rightValues);

        Tensor output =
            ((left + right) * (left - right))
            / (right + Tensor.Scalar(2f));
        output.Sum().Backward();

        return new ElementwiseResult(
            output.Data.ToArray(),
            left.Grad.ToArray(),
            right.Grad.ToArray());
    }

    private static KernelResult RunTransformerKernels(bool useSimd)
    {
        Tensor.SimdEnabled = useSimd;
        const int rows = 3;
        const int inputWidth = 32;
        const int outputWidth = 24;

        var input = new Tensor(
            Enumerable.Range(0, rows * inputWidth)
                .Select(index => (index % 13 - 6) * 0.03f)
                .ToArray(),
            [rows, inputWidth]);
        var weight = new Tensor(
            Enumerable.Range(0, inputWidth * outputWidth)
                .Select(index => (index % 17 - 8) * 0.02f)
                .ToArray(),
            [inputWidth, outputWidth]);
        var bias = Tensor.From1D(
            Enumerable.Range(0, outputWidth)
                .Select(index => (index - 12) * 0.01f)
                .ToArray());
        var gamma = Tensor.From1D(
            Enumerable.Range(0, outputWidth)
                .Select(index => 0.8f + index * 0.01f)
                .ToArray());
        var beta = Tensor.From1D(new float[outputWidth]);
        var lossWeight = new Tensor(
            Enumerable.Range(0, rows * outputWidth)
                .Select(index => (index % 7 - 3) * 0.1f)
                .ToArray(),
            [rows, outputWidth]);

        Tensor output = input
            .MatMul(weight)
            .AddRowWise(bias)
            .Relu()
            .LayerNormLastDim(gamma, beta)
            .LogSoftmaxLastDim();
        (output * lossWeight).Sum().Backward();

        return new KernelResult(
            output.Data.ToArray(),
            input.Grad.ToArray(),
            weight.Grad.ToArray(),
            bias.Grad.ToArray(),
            gamma.Grad.ToArray(),
            beta.Grad.ToArray());
    }

    private static BatchedKernelResult RunBatchedKernels(bool useSimd)
    {
        Tensor.SimdEnabled = useSimd;
        const int batch = 3;
        const int sequence = 4;
        const int width = 16;
        int length = batch * sequence * width;
        var query = new Tensor(
            Enumerable.Range(0, length)
                .Select(index => (index % 19 - 9) * 0.015f)
                .ToArray(),
            [batch, sequence, width]);
        var key = new Tensor(
            Enumerable.Range(0, length)
                .Select(index => (index % 23 - 11) * 0.012f)
                .ToArray(),
            [batch, sequence, width]);
        var value = new Tensor(
            Enumerable.Range(0, length)
                .Select(index => (index % 17 - 8) * 0.02f)
                .ToArray(),
            [batch, sequence, width]);
        var lossWeight = new Tensor(
            Enumerable.Range(0, length)
                .Select(index => (index % 13 - 6) * 0.03f)
                .ToArray(),
            [batch, sequence, width]);

        Tensor attention =
            (query.BatchedMatMulTransposedRight(key)
                * Tensor.Scalar(0.25f))
            .SoftmaxLastDim();
        Tensor output = attention.BatchedMatMul(value);
        (output * lossWeight).Sum().Backward();

        return new BatchedKernelResult(
            output.Data.ToArray(),
            query.Grad.ToArray(),
            key.Grad.ToArray(),
            value.Grad.ToArray());
    }

    private static AdamResult RunAdam(bool useSimd)
    {
        Tensor.SimdEnabled = useSimd;
        const int parameterLength = 20_000;
        var parameters = new[]
        {
            new Parameter(
                Enumerable.Range(0, parameterLength)
                    .Select(index => (index % 31 - 15) * 0.01f)
                    .ToArray(),
                [parameterLength],
                "first",
                WeightDecayPolicy.Apply),
            new Parameter(
                Enumerable.Range(0, parameterLength)
                    .Select(index => (index % 23 - 11) * 0.013f)
                    .ToArray(),
                [parameterLength],
                "second",
                WeightDecayPolicy.Exclude),
        };
        for (int parameter = 0; parameter < parameters.Length; parameter++)
        {
            Span<float> gradient = parameters[parameter].T.MutableGrad;
            for (int index = 0; index < gradient.Length; index++)
            {
                gradient[index] =
                    (index % 17 - 8) * (0.002f + parameter * 0.001f);
            }
        }

        var optimizer = new AdamW(
            parameters,
            new AdamWOptions
            {
                LearningRate = 0.003f,
                WeightDecay = 0.01f,
                Decay1D = true,
            });
        optimizer.Step();
        optimizer.Step();
        AdamWState state = optimizer.CaptureState();

        return new AdamResult(
            parameters.SelectMany(parameter => parameter.T.Data).ToArray(),
            state.ParameterStates
                .SelectMany(parameter => parameter.FirstMoment)
                .ToArray(),
            state.ParameterStates
                .SelectMany(parameter => parameter.SecondMoment)
                .ToArray());
    }

    private static LionResult RunLion(bool useSimd)
    {
        Tensor.SimdEnabled = useSimd;
        const int parameterLength = 20_000;
        var parameters = new[]
        {
            new Parameter(
                Enumerable.Range(0, parameterLength)
                    .Select(index => (index % 31 - 15) * 0.01f)
                    .ToArray(),
                [parameterLength],
                "first",
                WeightDecayPolicy.Apply),
            new Parameter(
                Enumerable.Range(0, parameterLength)
                    .Select(index => (index % 23 - 11) * 0.013f)
                    .ToArray(),
                [parameterLength],
                "second",
                WeightDecayPolicy.Exclude),
        };
        for (int parameter = 0; parameter < parameters.Length; parameter++)
        {
            Span<float> gradient = parameters[parameter].T.MutableGrad;
            for (int index = 0; index < gradient.Length; index++)
            {
                gradient[index] =
                    (index % 17 - 8) * (0.002f + parameter * 0.001f);
            }
        }

        var optimizer = new Lion(
            parameters,
            new LionOptions
            {
                LearningRate = 0.003f,
                WeightDecay = 0.01f,
                Decay1D = true,
            });
        optimizer.Step();
        optimizer.Step();
        LionState state = optimizer.CaptureState();

        return new LionResult(
            parameters.SelectMany(parameter => parameter.T.Data).ToArray(),
            state.ParameterStates
                .SelectMany(parameter => parameter.Momentum)
                .ToArray());
    }

    private static NekoMuonResult RunNekoMuon(bool useSimd)
    {
        Tensor.SimdEnabled = useSimd;
        const int rows = 32;
        const int columns = 1024;
        int parameterLength = rows * columns;
        var parameters = new[]
        {
            new Parameter(
                Enumerable.Range(0, parameterLength)
                    .Select(index => (index % 31 - 15) * 0.01f)
                    .ToArray(),
                [rows, columns],
                "first",
                WeightDecayPolicy.Apply),
            new Parameter(
                Enumerable.Range(0, parameterLength)
                    .Select(index => (index % 23 - 11) * 0.013f)
                    .ToArray(),
                [rows, columns],
                "second",
                WeightDecayPolicy.Exclude),
        };
        for (int parameter = 0; parameter < parameters.Length; parameter++)
        {
            Span<float> gradient = parameters[parameter].T.MutableGrad;
            for (int index = 0; index < gradient.Length; index++)
            {
                gradient[index] =
                    (index % 17 - 8) * (0.002f + parameter * 0.001f);
            }
        }

        var optimizer = new NekoMuon(
            parameters,
            new NekoMuonOptions
            {
                LearningRate = 0.003f,
                MaxNewtonSchulzSteps = 3,
                WeightDecay = 0.01f,
            });
        optimizer.Step();
        NekoMuonState state = optimizer.CaptureState();

        return new NekoMuonResult(
            parameters.SelectMany(parameter => parameter.T.Data).ToArray(),
            state.ParameterStates
                .SelectMany(parameter => parameter.FastMoment)
                .ToArray(),
            state.ParameterStates
                .SelectMany(parameter => parameter.SlowMoment)
                .ToArray(),
            state.ParameterStates
                .Select(parameter => parameter.Confidence)
                .ToArray());
    }

    private static GptStepResult RunGptTrainingStep(
        bool useSimd,
        int maxDegreeOfParallelism = 0)
    {
        Tensor.SimdEnabled = useSimd;
        Tensor.MaxDegreeOfParallelism = maxDegreeOfParallelism;
        var model = new GptRinWikiJp(
            vocabularySize: BpeTokenizer.BaseVocabularySize,
            contextLength: 4,
            dModel: 16,
            numHeads: 4,
            dHidden: 32,
            numLayers: 1,
            rng: new Random(29),
            dropout: 0f);
        var optimizer = new CompositeOptimizer(
            new NekoMuon(
                model.HiddenWeightParameters,
                new NekoMuonOptions
                {
                    LearningRate = 3e-4f,
                    MaxNewtonSchulzSteps = 3,
                }),
            new AdamW(
                model.AuxiliaryParameters,
                new AdamWOptions { LearningRate = 3e-4f }));
        int[] input = [1, 4, 5, 6, 1, 7, 8, 9];
        int[] targets = [4, 5, 6, 2, 7, 8, 9, 2];

        optimizer.ZeroGrad();
        Tensor logits = model.Forward(input, 2, 4);
        Tensor loss = logits.CrossEntropyWithLogits(targets);
        loss.Backward();
        float[] gradients = model.Parameters()
            .SelectMany(parameter => parameter.T.Grad)
            .ToArray();
        optimizer.Step();

        return new GptStepResult(
            loss.Data[0],
            logits.Data.ToArray(),
            gradients,
            model.Parameters()
                .SelectMany(parameter => parameter.T.Data)
                .ToArray());
    }

    private static GainShareResult RunGainShare(bool useSimd)
    {
        Tensor.SimdEnabled = useSimd;
        const int length = 32 * 1024;
        var parameters = new[]
        {
            new Parameter(
                Enumerable.Range(0, length)
                    .Select(index => (index % 29 - 14) * 0.01f)
                    .ToArray(),
                [32, 1024],
                "first",
                WeightDecayPolicy.Apply),
            new Parameter(
                Enumerable.Range(0, length)
                    .Select(index => (index % 19 - 9) * 0.015f)
                    .ToArray(),
                [32, 1024],
                "second",
                WeightDecayPolicy.Exclude),
        };
        for (int parameter = 0; parameter < parameters.Length; parameter++)
        {
            Span<float> gradient = parameters[parameter].T.MutableGrad;
            for (int index = 0; index < gradient.Length; index++)
            {
                gradient[index] =
                    (index % 17 - 8) * (0.002f + parameter * 0.001f);
            }
        }

        var optimizer = new GainShareAdamW(
            [[parameters[0]], [parameters[1]]],
            new GainShareAdamWOptions
            {
                LearningRate = 3e-4f,
                WeightDecay = 5e-4f,
            });
        optimizer.Step();
        GainShareAdamWState state = optimizer.CaptureState();

        return new GainShareResult(
            parameters.SelectMany(parameter => parameter.T.Data).ToArray(),
            state.ParameterStates
                .SelectMany(parameter => parameter.FirstMoment)
                .ToArray(),
            state.ParameterStates
                .SelectMany(parameter => parameter.SecondMoment)
                .ToArray(),
            state.GroupStates
                .Select(group => (float)(group.AlignmentEma ?? 0d))
                .ToArray());
    }

    private sealed record ElementwiseResult(
        float[] Output,
        float[] LeftGradient,
        float[] RightGradient);

    private sealed record KernelResult(
        float[] Output,
        float[] InputGradient,
        float[] WeightGradient,
        float[] BiasGradient,
        float[] GammaGradient,
        float[] BetaGradient);

    private sealed record BatchedKernelResult(
        float[] Output,
        float[] QueryGradient,
        float[] KeyGradient,
        float[] ValueGradient);

    private sealed record AdamResult(
        float[] Parameters,
        float[] FirstMoments,
        float[] SecondMoments);

    private sealed record LionResult(
        float[] Parameters,
        float[] Momenta);

    private sealed record NekoMuonResult(
        float[] Parameters,
        float[] FastMoments,
        float[] SlowMoments,
        float[] Confidences);

    private sealed record GptStepResult(
        float Loss,
        float[] Logits,
        float[] Gradients,
        float[] Parameters);

    private sealed record GainShareResult(
        float[] Parameters,
        float[] FirstMoments,
        float[] SecondMoments,
        float[] AlignmentEmas);
}
