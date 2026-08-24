using NNtrain;
using Xunit;

public sealed class ForgetMemoryV2Tests
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

        Tensor output = projected.ForgetMemoryV2(
            keyWidth: 1,
            valueWidth: 1,
            retentionFloor: 0f);

        // K=1, so k~=tanh(3), q~=tanh(2). With v=0.5, g=0.5,
        // and write=(1-g)*beta=0.25, M=0.125*k~ and r=M*q~.
        float expected = 0.125f * MathF.Tanh(3f) * MathF.Tanh(2f);
        Assert.Equal(expected, output.Data[0], precision: 5);
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
        Tensor output = projected.ForgetMemoryV2(
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
            .ForgetMemoryV2(keyWidth, valueWidth, 0.4f);
        Tensor second = new Tensor(secondValues, [1, 4, projectionWidth])
            .ForgetMemoryV2(keyWidth, valueWidth, 0.4f);

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
                Tensor output = input.ForgetMemoryV2(
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
            var random = new Random(456);
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
                Tensor.CudaDeviceIndices = device == TensorDevice.Cuda
                    && Tensor.CudaDeviceCount >= 2
                        ? [0, 1]
                        : [0];
                var input = new Tensor(
                    values,
                    [batch, sequence, projectionWidth],
                    dtype: TensorDType.BFloat16);
                Tensor output = input.ForgetMemoryV2(
                    keyWidth,
                    valueWidth,
                    0.35f);
                output.Backward(upstream);
                return (output.Data.ToArray(), input.Grad.ToArray());
            }

            (float[] cpuOutput, float[] cpuGradient) =
                EvaluateDevice(TensorDevice.Cpu);
            (float[] cudaOutput, float[] cudaGradient) =
                EvaluateDevice(TensorDevice.Cuda);

            AssertClose(cpuOutput, cudaOutput, 2e-4f);
            AssertClose(cpuGradient, cudaGradient, 3e-5f);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndex = previousDeviceIndex;
        }
    }

    [Fact]
    public void BFloat16GptTrainsWithCudaForgetMemory()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndex = 0;
            var model = new ForgetMemoryV2Gpt(
                vocabularySize: BpeTokenizer.BaseVocabularySize,
                contextLength: 3,
                modelWidth: 8,
                hiddenWidth: 12,
                numLayers: 1,
                keyWidth: 2,
                valueWidth: 2,
                dropout: 0f,
                random: new Random(91),
                dtype: TensorDType.BFloat16);
            int[] tokens =
            [
                BpeTokenizer.BosTokenId,
                BpeTokenizer.ByteTokenOffset + 1,
                BpeTokenizer.EosTokenId,
            ];

            Tensor logits = model.Forward(tokens, 1, tokens.Length);
            Tensor loss = logits.CrossEntropyWithLogits(
                [tokens[1], tokens[2], tokens[0]]);
            loss.Backward();
            IOptimizer optimizer = optim.Composite(
                optim.NekoMuon(
                    model.HiddenWeightParameters,
                    lr: 0.01f,
                    newton_schulz_interval: 5),
                optim.AdamW(
                    model.AuxiliaryParameters,
                    lr: 0.01f,
                    bf16_first_moment: true,
                    bf16_second_moment: true));
            optimizer.step();

            Assert.Equal(TensorDType.BFloat16, model.DType);
            Assert.Equal(TensorDType.BFloat16, logits.DType);
            Assert.True(float.IsFinite(loss.item()));
            Assert.Contains(
                model.Parameters(),
                parameter => parameter.T.Data.All(float.IsFinite));
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void GptSchedulesShortToLongMemoryAndTrains()
    {
        var model = new ForgetMemoryV2Gpt(
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
                .ForgetMemoryV2(keyWidth, valueWidth, 0.3f);
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
