using NNtrain;
using Xunit;

public sealed class FrogetMemoryV2Tests
{
    [Fact]
    public void SingleTokenImplementsStableDeltaMemoryAndReadout()
    {
        float valueLogit = 0.5f * MathF.Log(3f);
        var projected = new Tensor(
            [
                2f,       // q
                3f,       // k
                valueLogit,
                0f,       // gate sigmoid = 0.5
                0f,       // beta sigmoid = 0.5
            ],
            [1, 1, 5]);

        Tensor output = projected.FrogetMemoryV2(
            keyWidth: 1,
            valueWidth: 1,
            retentionFloor: 0f);

        // v=0.5, g=0.5, write=(1-g)*beta=0.25.
        // M=0.25*0.5*3=0.375 and r=M*q=0.75.
        Assert.Equal(0.75f, output.Data[0], precision: 5);
    }

    [Fact]
    public void IntegratedTensorBackwardMatchesFiniteDifferences()
    {
        const int keyWidth = 2;
        const int valueWidth = 2;
        const int sequence = 3;
        const int projectionWidth = 2 * keyWidth + 3 * valueWidth;
        var random = new Random(123);
        float[] values = Enumerable.Range(0, sequence * projectionWidth)
            .Select(_ => (float)(random.NextDouble() * 0.8 - 0.4))
            .ToArray();
        float[] upstream = Enumerable.Range(0, sequence * valueWidth)
            .Select(index => 0.2f + 0.1f * index)
            .ToArray();

        var projected = new Tensor(
            values,
            [1, sequence, projectionWidth]);
        Tensor output = projected.FrogetMemoryV2(
            keyWidth,
            valueWidth,
            retentionFloor: 0.3f);
        output.Backward(upstream);

        const float epsilon = 1e-3f;
        for (int index = 0; index < values.Length; index++)
        {
            float original = values[index];
            values[index] = original + epsilon;
            float positive = Evaluate(
                values,
                upstream,
                sequence,
                projectionWidth,
                keyWidth,
                valueWidth);
            values[index] = original - epsilon;
            float negative = Evaluate(
                values,
                upstream,
                sequence,
                projectionWidth,
                keyWidth,
                valueWidth);
            values[index] = original;
            float numerical = (positive - negative) / (2f * epsilon);

            Assert.InRange(
                MathF.Abs(projected.Grad[index] - numerical),
                0f,
                2e-3f);
        }
    }

    [Fact]
    public void FutureProjectionCannotChangePastRecall()
    {
        const int keyWidth = 2;
        const int valueWidth = 2;
        const int projectionWidth = 2 * keyWidth + 3 * valueWidth;
        float[] firstValues = Enumerable.Range(0, 4 * projectionWidth)
            .Select(index => 0.01f * (index + 1))
            .ToArray();
        float[] secondValues = (float[])firstValues.Clone();
        for (int index = 3 * projectionWidth;
            index < secondValues.Length;
            index++)
        {
            secondValues[index] += 3f;
        }

        Tensor first = new Tensor(firstValues, [1, 4, projectionWidth])
            .FrogetMemoryV2(keyWidth, valueWidth, 0.4f);
        Tensor second = new Tensor(secondValues, [1, 4, projectionWidth])
            .FrogetMemoryV2(keyWidth, valueWidth, 0.4f);

        Assert.Equal(
            first.Data.Take(3 * valueWidth),
            second.Data.Take(3 * valueWidth));
    }

    [Fact]
    public void ScalarAndParallelSimdForwardBackwardMatch()
    {
        bool previousSimd = Tensor.SimdEnabled;
        int previousParallelism = Tensor.MaxDegreeOfParallelism;
        try
        {
            const int batch = 2;
            const int sequence = 5;
            const int keyWidth = 9;
            const int valueWidth = 4;
            const int projectionWidth = 2 * keyWidth + 3 * valueWidth;
            var random = new Random(321);
            float[] values = Enumerable.Range(
                    0,
                    batch * sequence * projectionWidth)
                .Select(_ => (float)(random.NextDouble() - 0.5))
                .ToArray();
            float[] upstream = Enumerable.Range(
                    0,
                    batch * sequence * valueWidth)
                .Select(_ => (float)(random.NextDouble() - 0.5))
                .ToArray();

            (float[] Output, float[] Gradient) EvaluateMode(
                bool simd,
                int parallelism)
            {
                Tensor.SimdEnabled = simd;
                Tensor.MaxDegreeOfParallelism = parallelism;
                var input = new Tensor(
                    values,
                    [batch, sequence, projectionWidth]);
                Tensor output = input.FrogetMemoryV2(
                    keyWidth,
                    valueWidth,
                    0.35f);
                output.Backward(upstream);
                return (output.Data.ToArray(), input.Grad.ToArray());
            }

            (float[] scalarOutput, float[] scalarGradient) =
                EvaluateMode(simd: false, parallelism: 1);
            (float[] simdOutput, float[] simdGradient) =
                EvaluateMode(simd: true, parallelism: 0);

            AssertClose(scalarOutput, simdOutput, 2e-5f);
            AssertClose(scalarGradient, simdGradient, 3e-5f);
        }
        finally
        {
            Tensor.SimdEnabled = previousSimd;
            Tensor.MaxDegreeOfParallelism = previousParallelism;
        }
    }

    [Fact]
    public void GptSchedulesShortToLongMemoryAndTrains()
    {
        var model = new FrogetMemoryV2Gpt(
            vocabularySize: BpeTokenizer.BaseVocabularySize,
            contextLength: 4,
            modelWidth: 8,
            hiddenWidth: 16,
            numLayers: 3,
            keyWidth: 3,
            valueWidth: 4,
            retentionMinimum: 0.2f,
            retentionMaximum: 0.8f,
            random: new Random(17));

        Assert.Equal(0.2f, model.Layers[0].RetentionFloor, precision: 6);
        Assert.Equal(0.5f, model.Layers[1].RetentionFloor, precision: 6);
        Assert.Equal(0.8f, model.Layers[2].RetentionFloor, precision: 6);

        int[] tokens =
        [
            BpeTokenizer.BosTokenId,
            BpeTokenizer.ByteTokenOffset + 1,
            BpeTokenizer.ByteTokenOffset + 2,
            BpeTokenizer.EosTokenId,
        ];
        Tensor logits = model.Forward(tokens, batchSize: 1, sequenceLength: 4);
        Tensor loss = logits.CrossEntropyWithLogits(
        [
            tokens[1],
            tokens[2],
            tokens[3],
            tokens[0],
        ]);
        loss.Backward();

        Assert.Equal(
            4 * BpeTokenizer.BaseVocabularySize,
            logits.Numel);
        Assert.True(float.IsFinite(loss.Data[0]));
        Assert.NotEmpty(model.HiddenWeightParameters);
        Assert.NotEmpty(model.AuxiliaryParameters);
        Assert.Contains(
            model.Parameters(),
            parameter => parameter.T.Grad.Any(gradient => gradient != 0f));

        int[] generated = model.GenerateTokenIds(
            [BpeTokenizer.BosTokenId],
            maxNewTokens: 2,
            temperature: 0f,
            topK: 1,
            stopTokenId: null,
            random: new Random(19));
        Assert.Equal(3, generated.Length);
        Assert.True(model.IsTraining);
    }

    private static float Evaluate(
        float[] values,
        float[] upstream,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth)
    {
        using (AutogradContext.NoGrad())
        {
            Tensor output = new Tensor(
                values,
                [1, sequence, projectionWidth])
                .FrogetMemoryV2(keyWidth, valueWidth, 0.3f);
            float result = 0f;
            for (int index = 0; index < output.Numel; index++)
                result += output.Data[index] * upstream[index];
            return result;
        }
    }

    private static void AssertClose(
        IReadOnlyList<float> expected,
        IReadOnlyList<float> actual,
        float tolerance)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int index = 0; index < expected.Count; index++)
        {
            Assert.InRange(
                MathF.Abs(expected[index] - actual[index]),
                0f,
                tolerance);
        }
    }
}
