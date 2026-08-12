using NNtrain;
using Xunit;

[Collection(TensorSimdCollection.Name)]
public sealed class HyenaGptTests
{
    [Fact]
    public void FusedHyenaMatchesReferenceAndFiniteDifferenceGradients()
    {
        const int batch = 1;
        const int sequence = 3;
        const int width = 2;
        float[] projectedValues = Enumerable.Range(0, batch * sequence * 3 * width)
            .Select(index => (index - 8) * 0.035f)
            .ToArray();
        float[] shortValues = Enumerable.Range(0, 3 * 3 * width)
            .Select(index => 0.15f + index * 0.012f)
            .ToArray();
        float[] filterValues =
            [0.2f, -0.1f, 0.08f, 0.04f, -0.03f, 0.06f];
        float[] diagonalValues = [0.7f, -0.35f];
        float[] seed = [0.4f, -0.2f, 0.1f, 0.3f, -0.5f, 0.25f];
        var projected = new Tensor(
            projectedValues,
            [batch, sequence, 3 * width]);
        var shortFilter = new Tensor(shortValues, [3, 3 * width]);
        var longFilter = new Tensor(filterValues, [sequence, width]);
        var diagonal = new Tensor(diagonalValues, [width]);

        Tensor result = projected.FusedCausalHyenaOrder2(
            shortFilter,
            longFilter,
            diagonal);
        result.Backward(seed);

        float[] expected = ReferenceHyena(
            projectedValues,
            shortValues,
            filterValues,
            diagonalValues,
            batch,
            sequence,
            width);
        TensorCharacterizationTests.AssertClose(expected, result.Data, 1e-5f);
        AssertGradient(
            projectedValues,
            projected.Grad,
            values => Objective(
                values,
                shortValues,
                filterValues,
                diagonalValues,
                seed,
                batch,
                sequence,
                width));
        AssertGradient(
            shortValues,
            shortFilter.Grad,
            values => Objective(
                projectedValues,
                values,
                filterValues,
                diagonalValues,
                seed,
                batch,
                sequence,
                width));
        AssertGradient(
            filterValues,
            longFilter.Grad,
            values => Objective(
                projectedValues,
                shortValues,
                values,
                diagonalValues,
                seed,
                batch,
                sequence,
                width));
        AssertGradient(
            diagonalValues,
            diagonal.Grad,
            values => Objective(
                projectedValues,
                shortValues,
                filterValues,
                values,
                seed,
                batch,
                sequence,
                width));
    }

    [Fact]
    public void FusedHyenaIsCausal()
    {
        const int sequence = 4;
        const int width = 2;
        var projected = new Tensor(
            Enumerable.Range(0, sequence * 3 * width)
                .Select(index => (index + 1) * 0.02f)
                .ToArray(),
            [1, sequence, 3 * width]);
        var shortFilter = new Tensor(
            Enumerable.Repeat(0.2f, 3 * 3 * width).ToArray(),
            [3, 3 * width]);
        var longFilter = new Tensor(
            Enumerable.Repeat(0.1f, sequence * width).ToArray(),
            [sequence, width]);
        var diagonal = Tensor.From1D([0.5f, 0.25f]);
        Tensor result = projected.FusedCausalHyenaOrder2(
            shortFilter,
            longFilter,
            diagonal);
        var seed = new float[sequence * width];
        seed[width] = 1f;
        seed[width + 1] = -0.5f;

        result.Backward(seed);

        Assert.All(
            projected.Grad.Skip(2 * 3 * width),
            gradient => Assert.Equal(0f, gradient));
    }

    [Fact]
    public void ParallelFusedHyenaMatchesSingleThreadedKernel()
    {
        int previousParallelism = Tensor.MaxDegreeOfParallelism;
        try
        {
            HyenaKernelResult single = RunHyenaKernel(maxDegreeOfParallelism: 1);
            HyenaKernelResult parallel = RunHyenaKernel(maxDegreeOfParallelism: 0);

            TensorCharacterizationTests.AssertClose(
                single.Output,
                parallel.Output,
                2e-5f);
            TensorCharacterizationTests.AssertClose(
                single.ProjectedGradient,
                parallel.ProjectedGradient,
                2e-5f);
            TensorCharacterizationTests.AssertClose(
                single.ShortFilterGradient,
                parallel.ShortFilterGradient,
                2e-5f);
            TensorCharacterizationTests.AssertClose(
                single.LongFilterGradient,
                parallel.LongFilterGradient,
                2e-5f);
            TensorCharacterizationTests.AssertClose(
                single.DiagonalGradient,
                parallel.DiagonalGradient,
                2e-5f);
        }
        finally
        {
            Tensor.MaxDegreeOfParallelism = previousParallelism;
        }
    }

    [Fact]
    public void SimdFusedHyenaMatchesScalarKernel()
    {
        int previousParallelism = Tensor.MaxDegreeOfParallelism;
        bool previousSimd = Tensor.SimdEnabled;
        try
        {
            HyenaKernelResult scalar = RunHyenaKernel(
                maxDegreeOfParallelism: 1,
                simdEnabled: false);
            HyenaKernelResult simd = RunHyenaKernel(
                maxDegreeOfParallelism: 1,
                simdEnabled: true);

            TensorCharacterizationTests.AssertClose(
                scalar.Output,
                simd.Output,
                2e-5f);
            TensorCharacterizationTests.AssertClose(
                scalar.ProjectedGradient,
                simd.ProjectedGradient,
                2e-5f);
            TensorCharacterizationTests.AssertClose(
                scalar.ShortFilterGradient,
                simd.ShortFilterGradient,
                2e-5f);
            TensorCharacterizationTests.AssertClose(
                scalar.LongFilterGradient,
                simd.LongFilterGradient,
                2e-5f);
            TensorCharacterizationTests.AssertClose(
                scalar.DiagonalGradient,
                simd.DiagonalGradient,
                2e-5f);
        }
        finally
        {
            Tensor.MaxDegreeOfParallelism = previousParallelism;
            Tensor.SimdEnabled = previousSimd;
        }
    }

    [Fact]
    public void FftHyenaMatchesDirectScalarSimdAndParallelKernels()
    {
        int previousParallelism = Tensor.MaxDegreeOfParallelism;
        bool previousSimd = Tensor.SimdEnabled;
        try
        {
            HyenaKernelResult direct = RunHyenaKernel(
                maxDegreeOfParallelism: 1,
                simdEnabled: false,
                convolutionAlgorithm: HyenaConvolutionAlgorithm.Direct);
            HyenaKernelResult fftScalar = RunHyenaKernel(
                maxDegreeOfParallelism: 1,
                simdEnabled: false,
                convolutionAlgorithm: HyenaConvolutionAlgorithm.Fft);
            HyenaKernelResult fftSimd = RunHyenaKernel(
                maxDegreeOfParallelism: 1,
                simdEnabled: true,
                convolutionAlgorithm: HyenaConvolutionAlgorithm.Fft);
            HyenaKernelResult fftParallel = RunHyenaKernel(
                maxDegreeOfParallelism: 0,
                simdEnabled: true,
                convolutionAlgorithm: HyenaConvolutionAlgorithm.Fft);
            HyenaKernelResult directLong = RunHyenaKernel(
                maxDegreeOfParallelism: 0,
                simdEnabled: true,
                convolutionAlgorithm: HyenaConvolutionAlgorithm.Direct,
                batch: 1,
                sequence: 257,
                width: 8);
            HyenaKernelResult fftLong = RunHyenaKernel(
                maxDegreeOfParallelism: 0,
                simdEnabled: true,
                convolutionAlgorithm: HyenaConvolutionAlgorithm.Fft,
                batch: 1,
                sequence: 257,
                width: 8);

            AssertHyenaKernelClose(direct, fftScalar, 3e-4f);
            AssertHyenaKernelClose(direct, fftSimd, 3e-4f);
            AssertHyenaKernelClose(fftSimd, fftParallel, 3e-4f);
            AssertHyenaKernelClose(directLong, fftLong, 2e-3f);
        }
        finally
        {
            Tensor.MaxDegreeOfParallelism = previousParallelism;
            Tensor.SimdEnabled = previousSimd;
        }
    }

    [Fact]
    public void NoGradFusedHyenaMatchesRecordedForward()
    {
        const int batch = 2;
        const int sequence = 7;
        const int width = 16;
        var projected = new Tensor(
            Enumerable.Range(0, batch * sequence * 3 * width)
                .Select(index => (index % 23 - 11) * 0.007f)
                .ToArray(),
            [batch, sequence, 3 * width]);
        var shortFilter = new Tensor(
            Enumerable.Range(0, 3 * 3 * width)
                .Select(index => (index % 13 - 6) * 0.009f)
                .ToArray(),
            [3, 3 * width]);
        var longFilter = new Tensor(
            Enumerable.Range(0, sequence * width)
                .Select(index => (index % 17 - 8) * 0.005f)
                .ToArray(),
            [sequence, width]);
        var diagonal = new Tensor(
            Enumerable.Range(0, width)
                .Select(index => 0.1f + index * 0.01f)
                .ToArray(),
            [width]);

        Tensor recorded = projected.FusedCausalHyenaOrder2(
            shortFilter,
            longFilter,
            diagonal);
        Tensor detached;
        using (AutogradContext.NoGrad())
        {
            detached = projected.FusedCausalHyenaOrder2(
                shortFilter,
                longFilter,
                diagonal,
                HyenaConvolutionAlgorithm.Fft);
        }

        TensorCharacterizationTests.AssertClose(
            recorded.Data,
            detached.Data,
            2e-5f);
    }

    [Fact]
    public void SinBackwardMatchesCosine()
    {
        var input = Tensor.From1D([-0.7f, 0f, 0.9f]);
        Tensor output = input.Sin();

        output.Backward([0.2f, -0.4f, 0.6f]);

        TensorCharacterizationTests.AssertClose(
            new[] { -0.7f, 0f, 0.9f }.Select(MathF.Sin),
            output.Data);
        TensorCharacterizationTests.AssertClose(
            new[]
            {
                0.2f * MathF.Cos(-0.7f),
                -0.4f,
                0.6f * MathF.Cos(0.9f),
            },
            input.Grad);
    }

    [Fact]
    public void ForwardBackwardAndGenerationProduceFiniteValues()
    {
        var model = new HyenaGpt(
            vocabularySize: BpeTokenizer.BaseVocabularySize,
            contextLength: 4,
            modelWidth: 8,
            hiddenWidth: 16,
            numLayers: 1,
            random: new Random(7),
            filterWidth: 8);
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

        Assert.Equal<int>([8, BpeTokenizer.BaseVocabularySize], logits.Shape);
        Assert.True(float.IsFinite(loss.Data[0]));
        Assert.All(logits.Data, value => Assert.True(float.IsFinite(value)));
        Assert.All(
            model.Parameters().SelectMany(parameter => parameter.T.Grad),
            value => Assert.True(float.IsFinite(value)));
        Assert.Equal(4, generated.Length);
        Assert.True(model.IsTraining);
    }

    [Fact]
    public void SeparatesHyenaMatricesFromAuxiliaryParameters()
    {
        var model = new HyenaGpt(
            BpeTokenizer.BaseVocabularySize,
            contextLength: 4,
            modelWidth: 8,
            hiddenWidth: 16,
            numLayers: 2,
            random: new Random(3),
            filterWidth: 8);

        Parameter[] all = model.Parameters().ToArray();
        Parameter[] hidden = model.HiddenWeightParameters.ToArray();
        Parameter[] auxiliary = model.AuxiliaryParameters.ToArray();

        Assert.NotEmpty(hidden);
        Assert.NotEmpty(auxiliary);
        Assert.All(hidden, parameter => Assert.True(parameter.T.Rank >= 2));
        Assert.Empty(hidden.Intersect(auxiliary));
        Assert.Equal(all.Length, hidden.Length + auxiliary.Length);
    }

    private static float Objective(
        float[] projected,
        float[] shortFilter,
        float[] longFilter,
        float[] diagonal,
        float[] seed,
        int batch,
        int sequence,
        int width)
    {
        float[] output = ReferenceHyena(
            projected,
            shortFilter,
            longFilter,
            diagonal,
            batch,
            sequence,
            width);
        return output.Zip(seed, (value, gradient) => value * gradient).Sum();
    }

    private static HyenaKernelResult RunHyenaKernel(
        int maxDegreeOfParallelism,
        bool simdEnabled = true,
        HyenaConvolutionAlgorithm convolutionAlgorithm =
            HyenaConvolutionAlgorithm.Auto,
        int batch = 4,
        int sequence = 32,
        int width = 16)
    {
        Tensor.MaxDegreeOfParallelism = maxDegreeOfParallelism;
        Tensor.SimdEnabled = simdEnabled;
        var projected = new Tensor(
            Enumerable.Range(0, batch * sequence * 3 * width)
                .Select(index => (index % 29 - 14) * 0.006f)
                .ToArray(),
            [batch, sequence, 3 * width]);
        var shortFilter = new Tensor(
            Enumerable.Range(0, 3 * 3 * width)
                .Select(index => (index % 17 - 8) * 0.01f)
                .ToArray(),
            [3, 3 * width]);
        var longFilter = new Tensor(
            Enumerable.Range(0, sequence * width)
                .Select(index => (index % 13 - 6) * 0.008f)
                .ToArray(),
            [sequence, width]);
        var diagonal = new Tensor(
            Enumerable.Range(0, width)
                .Select(index => 0.2f + index * 0.01f)
                .ToArray(),
            [width]);
        Tensor output = projected.FusedCausalHyenaOrder2(
            shortFilter,
            longFilter,
            diagonal,
            convolutionAlgorithm);

        output.Sum().Backward();

        return new HyenaKernelResult(
            output.Data.ToArray(),
            projected.Grad.ToArray(),
            shortFilter.Grad.ToArray(),
            longFilter.Grad.ToArray(),
            diagonal.Grad.ToArray());
    }

    private static void AssertHyenaKernelClose(
        HyenaKernelResult expected,
        HyenaKernelResult actual,
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
        TensorCharacterizationTests.AssertClose(
            expected.ShortFilterGradient,
            actual.ShortFilterGradient,
            tolerance);
        TensorCharacterizationTests.AssertClose(
            expected.LongFilterGradient,
            actual.LongFilterGradient,
            tolerance);
        TensorCharacterizationTests.AssertClose(
            expected.DiagonalGradient,
            actual.DiagonalGradient,
            tolerance);
    }

    private static float[] ReferenceHyena(
        float[] projected,
        float[] shortFilter,
        float[] longFilter,
        float[] diagonal,
        int batch,
        int sequence,
        int width)
    {
        int channels = 3 * width;
        var shortOutput = new float[projected.Length];
        var gated = new float[batch * sequence * width];
        var output = new float[gated.Length];
        for (int batchIndex = 0; batchIndex < batch; batchIndex++)
        {
            for (int time = 0; time < sequence; time++)
            {
                int shortOffset = (batchIndex * sequence + time) * channels;
                for (int tap = 0; tap <= Math.Min(2, time); tap++)
                {
                    int inputOffset =
                        (batchIndex * sequence + time - tap) * channels;
                    for (int channel = 0; channel < channels; channel++)
                    {
                        shortOutput[shortOffset + channel] +=
                            projected[inputOffset + channel]
                            * shortFilter[tap * channels + channel];
                    }
                }

                int outputOffset = (batchIndex * sequence + time) * width;
                for (int channel = 0; channel < width; channel++)
                {
                    gated[outputOffset + channel] =
                        shortOutput[shortOffset + width + channel]
                        * shortOutput[shortOffset + 2 * width + channel];
                    float convolved =
                        diagonal[channel] * gated[outputOffset + channel];
                    for (int lag = 0; lag <= time; lag++)
                    {
                        int source =
                            (batchIndex * sequence + time - lag) * width;
                        convolved += gated[source + channel]
                            * longFilter[lag * width + channel];
                    }
                    output[outputOffset + channel] =
                        shortOutput[shortOffset + channel] * convolved;
                }
            }
        }
        return output;
    }

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
            tolerance: 2e-3f);
    }

    private sealed record HyenaKernelResult(
        float[] Output,
        float[] ProjectedGradient,
        float[] ShortFilterGradient,
        float[] LongFilterGradient,
        float[] DiagonalGradient);
}
