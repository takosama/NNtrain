using NNtrain;
using Xunit;

public sealed class ForgetMemoryDRNTests
{
    [Fact]
    public void ProducesExpectedShapeForArbitraryDimensions()
    {
        const int batch = 3;
        const int sequence = 5;
        const int keyWidth = 4;
        const int valueWidth = 7;
        const int projectionWidth = 2 * keyWidth + 3 * valueWidth;
        var projected = new Tensor(
            new float[batch * sequence * projectionWidth],
            [batch, sequence, projectionWidth]);

        Tensor output = projected.ForgetMemoryDRN(
            keyWidth,
            valueWidth,
            retentionFloor: 0.99f);

        Assert.Equal<int>([batch, sequence, valueWidth], output.Shape);
    }

    [Fact]
    public void ReadsBeforeWritingAndNormalizesQueryAndKey()
    {
        const float raw = 1.25f;
        float valueRaw = MathF.Atanh(0.5f);
        float[] token =
        [
            raw, 0f,       // q
            raw, 0f,       // k
            valueRaw,      // v
            20f,           // f ~= 1
            20f,           // beta ~= 1
        ];
        var values = new float[token.Length * 2];
        token.CopyTo(values, 0);
        token.CopyTo(values, token.Length);

        Tensor output = new Tensor(values, [1, 2, token.Length])
            .ForgetMemoryDRN(2, 1, retentionFloor: 0.99f);

        Assert.Equal(0f, output.Data[0]);
        float tanh = MathF.Tanh(raw);
        float normalizedSquared = tanh * tanh / (tanh * tanh + 1e-8f);
        float sigmoid20 = 1f / (1f + MathF.Exp(-20f));
        float expectedSecond =
            sigmoid20 * 0.5f * normalizedSquared * normalizedSquared;
        Assert.Equal(expectedSecond, output.Data[1], precision: 5);
    }

    [Fact]
    public void RepeatedWritesConvergeWithoutUnboundedReadGain()
    {
        const int sequence = 256;
        const float raw = 1.5f;
        const float expectedValue = 0.5f;
        const int projectionWidth = 5;
        float valueRaw = MathF.Atanh(expectedValue);
        var values = new float[sequence * projectionWidth];
        for (int time = 0; time < sequence; time++)
        {
            int offset = time * projectionWidth;
            values[offset] = raw;
            values[offset + 1] = raw;
            values[offset + 2] = valueRaw;
            values[offset + 3] = 20f;
            values[offset + 4] = 0f;
        }

        Tensor output = new Tensor(values, [1, sequence, projectionWidth])
            .ForgetMemoryDRN(1, 1, retentionFloor: 0f);

        Assert.InRange(output.Data[^1], 0.499f, 0.501f);
        Assert.All(
            output.Data,
            value => Assert.InRange(MathF.Abs(value), 0f, 0.501f));
    }

    [Fact]
    public void IntegratedBackwardMatchesFiniteDifferences()
    {
        const int keyWidth = 3;
        const int valueWidth = 2;
        const int sequence = 4;
        const int projectionWidth = 2 * keyWidth + 3 * valueWidth;
        var random = new Random(913);
        float[] values = Enumerable.Range(0, sequence * projectionWidth)
            .Select(_ => (float)(random.NextDouble() * 0.8 - 0.4))
            .ToArray();
        float[] upstream = Enumerable.Range(0, sequence * valueWidth)
            .Select(index => 0.1f + 0.06f * index)
            .ToArray();

        var projected = new Tensor(values, [1, sequence, projectionWidth]);
        Tensor output = projected.ForgetMemoryDRN(
            keyWidth,
            valueWidth,
            retentionFloor: 0.8f);
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
                3e-3f);
        }
    }

    [Fact]
    public void LongZeroSequenceStaysFinite()
    {
        const int sequence = 4096;
        const int keyWidth = 4;
        const int valueWidth = 3;
        const int projectionWidth = 2 * keyWidth + 3 * valueWidth;
        using (AutogradContext.NoGrad())
        {
            Tensor output = new Tensor(
                    new float[sequence * projectionWidth],
                    [1, sequence, projectionWidth])
                .ForgetMemoryDRN(keyWidth, valueWidth, 0.99f);

            Assert.All(
                output.Data,
                value => Assert.True(float.IsFinite(value)));
        }
    }

    [Fact]
    public void CudaForwardBackwardMatchesCpu()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int previousDeviceIndex = Tensor.CudaDeviceIndex;
        try
        {
            const int batch = 2;
            const int sequence = 4;
            const int keyWidth = 3;
            const int valueWidth = 2;
            const int projectionWidth = 2 * keyWidth + 3 * valueWidth;
            var random = new Random(294);
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

            (float[] Output, float[] Gradient) EvaluateDevice(
                TensorDevice device)
            {
                Tensor.ExecutionDevice = device;
                Tensor.CudaDeviceIndex = 0;
                var input = new Tensor(
                    values,
                    [batch, sequence, projectionWidth]);
                Tensor result = input.ForgetMemoryDRN(
                    keyWidth,
                    valueWidth,
                    0.35f);
                result.Backward(upstream);
                return (result.Data.ToArray(), input.Grad.ToArray());
            }

            (float[] cpuOutput, float[] cpuGradient) =
                EvaluateDevice(TensorDevice.Cpu);
            (float[] cudaOutput, float[] cudaGradient) =
                EvaluateDevice(TensorDevice.Cuda);

            AssertClose(cpuOutput, cudaOutput, 3e-5f);
            AssertClose(cpuGradient, cudaGradient, 6e-5f);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndex = previousDeviceIndex;
        }
    }

    [Fact]
    public void GptUsesDrnInEveryLayerAndTrains()
    {
        var model = new ForgetMemoryDRNGpt(
            vocabularySize: BpeTokenizer.BaseVocabularySize,
            contextLength: 4,
            modelWidth: 8,
            hiddenWidth: 16,
            numLayers: 2,
            keyWidth: 3,
            valueWidth: 4,
            random: new Random(17));

        Assert.True(model.UseDrn);
        Assert.False(model.UseV3);
        Assert.All(model.Layers, layer => Assert.True(layer.UseDrn));

        int[] tokens =
        [
            BpeTokenizer.BosTokenId,
            BpeTokenizer.ByteTokenOffset + 1,
            BpeTokenizer.ByteTokenOffset + 2,
            BpeTokenizer.EosTokenId,
        ];
        Tensor logits = model.Forward(tokens, 1, tokens.Length);
        Tensor loss = logits.CrossEntropyWithLogits(
            [tokens[1], tokens[2], tokens[3], tokens[0]]);
        loss.Backward();

        Assert.True(float.IsFinite(loss.item()));
        Assert.Contains(
            model.Parameters(),
            parameter => parameter.T.Grad.Any(value => value != 0f));
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
                .ForgetMemoryDRN(keyWidth, valueWidth, 0.8f);
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
