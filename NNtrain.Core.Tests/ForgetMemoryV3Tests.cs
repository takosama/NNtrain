using NNtrain;
using Xunit;

public sealed class ForgetMemoryV3Tests
{
    [Fact]
    public void SingleTokenUsesIndependentWriteAndUnitKey()
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

        Tensor output = projected.ForgetMemoryV3(
            keyWidth: 1,
            valueWidth: 1,
            retentionFloor: 0f);

        float keyTanh = MathF.Tanh(3f);
        float key = keyTanh / MathF.Sqrt(keyTanh * keyTanh + 1e-6f);
        // M0=0, so g does not suppress the beta=0.5 write of v=0.5.
        float expected = 0.25f * key * MathF.Tanh(2f);
        Assert.Equal(expected, output.Data[0], precision: 5);
    }

    [Fact]
    public void IntegratedTensorBackwardMatchesFiniteDifferences()
    {
        const int keyWidth = 3;
        const int valueWidth = 2;
        const int sequence = 3;
        const int projectionWidth = 2 * keyWidth + 3 * valueWidth;
        var random = new Random(731);
        float[] values = Enumerable.Range(0, sequence * projectionWidth)
            .Select(_ => (float)(random.NextDouble() * 0.8 - 0.4))
            .ToArray();
        float[] upstream = Enumerable.Range(0, sequence * valueWidth)
            .Select(index => 0.15f + 0.07f * index)
            .ToArray();

        var projected = new Tensor(values, [1, sequence, projectionWidth]);
        Tensor output = projected.ForgetMemoryV3(
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
                3e-3f);
        }
    }

    [Fact]
    public void ZeroKeyProducesFiniteOutputAndGradients()
    {
        const int keyWidth = 3;
        const int valueWidth = 2;
        const int projectionWidth = 2 * keyWidth + 3 * valueWidth;
        var projected = new Tensor(
            new float[2 * projectionWidth],
            [1, 2, projectionWidth]);

        Tensor output = projected.ForgetMemoryV3(
            keyWidth,
            valueWidth,
            retentionFloor: 0.99f);
        output.Backward(Enumerable.Repeat(1f, output.Numel).ToArray());

        Assert.All(output.Data, value => Assert.True(float.IsFinite(value)));
        Assert.All(projected.Grad, value => Assert.True(float.IsFinite(value)));
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
            var random = new Random(982);
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
                Tensor result = input.ForgetMemoryV3(
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
            AssertClose(cpuGradient, cudaGradient, 5e-5f);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndex = previousDeviceIndex;
        }
    }

    [Fact]
    public void GptUsesV3InEveryLayer()
    {
        var model = new ForgetMemoryV3Gpt(
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

        Assert.True(model.UseV3);
        Assert.All(model.Layers, layer => Assert.True(layer.UseV3));
        Assert.Equal(0.2f, model.Layers[0].RetentionFloor, precision: 6);
        Assert.Equal(0.5f, model.Layers[1].RetentionFloor, precision: 6);
        Assert.Equal(0.8f, model.Layers[2].RetentionFloor, precision: 6);
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
                .ForgetMemoryV3(keyWidth, valueWidth, 0.3f);
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
