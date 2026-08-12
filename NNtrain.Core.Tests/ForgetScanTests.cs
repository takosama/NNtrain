using NNtrain;
using Xunit;

[Collection(TensorSimdCollection.Name)]
public sealed class ForgetScanTests
{
    [Fact]
    public void ForwardAndBackwardMatchSequentialReferenceAndFiniteDifferences()
    {
        const int batch = 1;
        const int sequence = 4;
        const int width = 2;
        bool previousSimd = Tensor.SimdEnabled;
        int previousParallelism = Tensor.MaxDegreeOfParallelism;
        try
        {
            Tensor.SimdEnabled = false;
            Tensor.MaxDegreeOfParallelism = 1;
            float[] values = Enumerable.Range(0, batch * sequence * 3 * width)
                .Select(index => (index - 11) * 0.035f)
                .ToArray();
            float[] seed = Enumerable.Range(0, batch * sequence * width)
                .Select(index => (index - 3) * 0.07f)
                .ToArray();
            var projected = new Tensor(
                values,
                [batch, sequence, 3 * width]);

            Tensor output = projected.FusedForgetScan();
            output.Backward(seed);

            TensorCharacterizationTests.AssertClose(
                Reference(values, batch, sequence, width),
                output.Data,
                2e-6f);
            AssertGradient(
                values,
                projected.Grad,
                candidate => Objective(
                    candidate,
                    seed,
                    batch,
                    sequence,
                    width));
        }
        finally
        {
            Tensor.SimdEnabled = previousSimd;
            Tensor.MaxDegreeOfParallelism = previousParallelism;
        }
    }

    [Fact]
    public void ScalarSimdParallelAndNoGradPathsMatch()
    {
        bool previousSimd = Tensor.SimdEnabled;
        int previousParallelism = Tensor.MaxDegreeOfParallelism;
        try
        {
            ScanResult scalar = Run(useSimd: false, maxDegreeOfParallelism: 1);
            ScanResult simd = Run(useSimd: true, maxDegreeOfParallelism: 1);
            ScanResult parallel = Run(useSimd: true, maxDegreeOfParallelism: 0);

            AssertClose(scalar, simd, 3e-5f);
            AssertClose(simd, parallel, 3e-5f);

            Tensor.SimdEnabled = true;
            Tensor.MaxDegreeOfParallelism = 0;
            var projected = new Tensor(
                CreateValues(batch: 3, sequence: 17, width: 16),
                [3, 17, 48]);
            Tensor detached;
            using (AutogradContext.NoGrad())
                detached = projected.FusedForgetScan();
            TensorCharacterizationTests.AssertClose(
                parallel.Output,
                detached.Data,
                3e-5f);
        }
        finally
        {
            Tensor.SimdEnabled = previousSimd;
            Tensor.MaxDegreeOfParallelism = previousParallelism;
        }
    }

    [Fact]
    public void FutureProjectionDoesNotChangePastMemory()
    {
        const int sequence = 6;
        const int width = 4;
        float[] firstValues = CreateValues(1, sequence, width);
        float[] secondValues = (float[])firstValues.Clone();
        for (int index = 4 * 3 * width; index < secondValues.Length; index++)
            secondValues[index] += 0.8f;
        var first = new Tensor(firstValues, [1, sequence, 3 * width]);
        var second = new Tensor(secondValues, [1, sequence, 3 * width]);

        Tensor firstOutput = first.FusedForgetScan();
        Tensor secondOutput = second.FusedForgetScan();

        TensorCharacterizationTests.AssertClose(
            firstOutput.Data.Take(4 * width),
            secondOutput.Data.Take(4 * width),
            2e-6f);
    }

    [Fact]
    public void ExtremeGateLogitsRemainFinite()
    {
        const int sequence = 5;
        const int width = 16;
        var values = Enumerable.Range(0, sequence * 3 * width)
            .Select(index => index % 2 == 0 ? -100f : 100f)
            .ToArray();
        var projected = new Tensor(values, [1, sequence, 3 * width]);

        Tensor output = projected.FusedForgetScan();
        output.Sum().Backward();

        Assert.All(output.Data, value => Assert.True(float.IsFinite(value)));
        Assert.All(projected.Grad, value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void SimdGateApproximationsMatchScalarAcrossWideLogitRange()
    {
        const int batch = 2;
        const int sequence = 31;
        const int width = 20;
        bool previousSimd = Tensor.SimdEnabled;
        int previousParallelism = Tensor.MaxDegreeOfParallelism;
        try
        {
            float[] values = Enumerable.Range(
                    0,
                    batch * sequence * 3 * width)
                .Select(index => ((index * 1543) % 4001 - 2000) * 0.01f)
                .ToArray();

            ScanResult Evaluate(bool useSimd)
            {
                Tensor.SimdEnabled = useSimd;
                Tensor.MaxDegreeOfParallelism = 1;
                var projected = new Tensor(
                    values,
                    [batch, sequence, 3 * width]);
                Tensor output = projected.FusedForgetScan();
                output.Sum().Backward();
                return new ScanResult(
                    output.Data.ToArray(),
                    projected.Grad.ToArray());
            }

            AssertClose(Evaluate(false), Evaluate(true), 4e-5f);
        }
        finally
        {
            Tensor.SimdEnabled = previousSimd;
            Tensor.MaxDegreeOfParallelism = previousParallelism;
        }
    }

    [Fact]
    public void ForgetScanGptTrainsGeneratesAndSeparatesOptimizerParameters()
    {
        var model = new ForgetScanGpt(
            vocabularySize: BpeTokenizer.BaseVocabularySize,
            contextLength: 4,
            modelWidth: 8,
            hiddenWidth: 16,
            numLayers: 2,
            random: new Random(7));
        int[] input = [1, 4, 5, 6, 1, 7, 8, 9];
        int[] targets = [4, 5, 6, 2, 7, 8, 9, 2];

        Tensor logits = model.Forward(input, batchSize: 2, sequenceLength: 4);
        Tensor loss = logits.CrossEntropyWithLogits(targets);
        loss.Backward();
        int[] generated = model.GenerateTokenIds(
            [BpeTokenizer.BosTokenId],
            maxNewTokens: 3,
            temperature: 0f,
            stopTokenId: null,
            random: new Random(11));

        Assert.Equal<int>(
            [8, BpeTokenizer.BaseVocabularySize],
            logits.Shape);
        Assert.True(float.IsFinite(loss.Data[0]));
        Assert.All(logits.Data, value => Assert.True(float.IsFinite(value)));
        Assert.All(
            model.Parameters().SelectMany(parameter => parameter.T.Grad),
            value => Assert.True(float.IsFinite(value)));
        Assert.Equal(4, generated.Length);
        Assert.True(model.IsTraining);

        Parameter[] all = model.Parameters().ToArray();
        Parameter[] hidden = model.HiddenWeightParameters.ToArray();
        Parameter[] auxiliary = model.AuxiliaryParameters.ToArray();
        Assert.NotEmpty(hidden);
        Assert.NotEmpty(auxiliary);
        Assert.All(hidden, parameter => Assert.True(parameter.T.Rank >= 2));
        Assert.Empty(hidden.Intersect(auxiliary));
        Assert.Equal(all.Length, hidden.Length + auxiliary.Length);
    }

    [Fact]
    public void ForgetScanGptLogitsAreCausal()
    {
        var model = new ForgetScanGpt(
            BpeTokenizer.BaseVocabularySize,
            contextLength: 6,
            modelWidth: 8,
            hiddenWidth: 16,
            numLayers: 1,
            random: new Random(5));
        int[] first = [1, 4, 5, 6, 7, 8];
        int[] second = [1, 4, 5, 21, 22, 23];

        Tensor firstLogits = model.Forward(first, 1, 6);
        Tensor secondLogits = model.Forward(second, 1, 6);

        TensorCharacterizationTests.AssertClose(
            firstLogits.Data.Take(3 * BpeTokenizer.BaseVocabularySize),
            secondLogits.Data.Take(3 * BpeTokenizer.BaseVocabularySize),
            3e-5f);
    }

    private static ScanResult Run(bool useSimd, int maxDegreeOfParallelism)
    {
        const int batch = 3;
        const int sequence = 17;
        const int width = 16;
        Tensor.SimdEnabled = useSimd;
        Tensor.MaxDegreeOfParallelism = maxDegreeOfParallelism;
        var projected = new Tensor(
            CreateValues(batch, sequence, width),
            [batch, sequence, 3 * width]);
        Tensor output = projected.FusedForgetScan();
        output.Sum().Backward();
        return new ScanResult(
            output.Data.ToArray(),
            projected.Grad.ToArray());
    }

    private static float[] CreateValues(int batch, int sequence, int width)
        => Enumerable.Range(0, batch * sequence * 3 * width)
            .Select(index => (index % 31 - 15) * 0.025f)
            .ToArray();

    private static float[] Reference(
        float[] projected,
        int batch,
        int sequence,
        int width)
    {
        int channels = 3 * width;
        var output = new float[batch * sequence * width];
        for (int batchIndex = 0; batchIndex < batch; batchIndex++)
        {
            for (int time = 0; time < sequence; time++)
            {
                int projectedOffset =
                    (batchIndex * sequence + time) * channels;
                int outputOffset =
                    (batchIndex * sequence + time) * width;
                int previousOffset = outputOffset - width;
                for (int channel = 0; channel < width; channel++)
                {
                    float f = Sigmoid(projected[projectedOffset + channel]);
                    float i = Sigmoid(
                        projected[projectedOffset + width + channel]);
                    float v = MathF.Tanh(
                        projected[projectedOffset + 2 * width + channel]);
                    float previous = time == 0
                        ? 0f
                        : output[previousOffset + channel];
                    output[outputOffset + channel] = f * previous + i * v;
                }
            }
        }
        return output;
    }

    private static float Objective(
        float[] projected,
        float[] seed,
        int batch,
        int sequence,
        int width)
        => Reference(projected, batch, sequence, width)
            .Zip(seed, (value, gradient) => value * gradient)
            .Sum();

    private static void AssertGradient(
        float[] values,
        IEnumerable<float> analytical,
        Func<float[], float> function)
    {
        const float epsilon = 1e-3f;
        var numerical = new float[values.Length];
        for (int index = 0; index < values.Length; index++)
        {
            var plus = (float[])values.Clone();
            var minus = (float[])values.Clone();
            plus[index] += epsilon;
            minus[index] -= epsilon;
            numerical[index] =
                (function(plus) - function(minus)) / (2f * epsilon);
        }
        TensorCharacterizationTests.AssertClose(
            numerical,
            analytical,
            3e-4f);
    }

    private static void AssertClose(
        ScanResult expected,
        ScanResult actual,
        float tolerance)
    {
        TensorCharacterizationTests.AssertClose(
            expected.Output,
            actual.Output,
            tolerance);
        TensorCharacterizationTests.AssertClose(
            expected.ProjectedGradient,
            actual.ProjectedGradient,
            tolerance);
    }

    private static float Sigmoid(float value)
        => 1f / (1f + MathF.Exp(-value));

    private sealed record ScanResult(
        float[] Output,
        float[] ProjectedGradient);
}
