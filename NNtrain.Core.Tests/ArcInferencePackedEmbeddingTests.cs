using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcInferencePackedEmbeddingTests
{
    [Theory]
    [InlineData(TensorDType.Float32, TensorPrecisionMode.Float32, 0)]
    [InlineData(TensorDType.Float32, TensorPrecisionMode.Float32, 7)]
    [InlineData(TensorDType.BFloat16, TensorPrecisionMode.Mix16_32, 0)]
    [InlineData(TensorDType.BFloat16, TensorPrecisionMode.Mix16_32, 7)]
    [InlineData(TensorDType.Bfp8, TensorPrecisionMode.Mix8_16, 0)]
    [InlineData(TensorDType.Bfp8, TensorPrecisionMode.Mix8_16, 7)]
    public void PackedLookupMatchesOnePublicationReferenceExactly(
        TensorDType storage, TensorPrecisionMode precision, int positionOffset)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int width = 79, tokenRows = 131, positionRows = 32;
        int[] ids = positionOffset == 0 ? [1, 53, 129] : [129];
        float[] tokenValues = Values(tokenRows * width, 17);
        float[] positionValues = Values(positionRows * width, 31);

        float[] Run(bool packed)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision,
                options: new ArcExecutionOptions { InferencePackedEmbedding = packed });
            Tensor tokens = Make(tokenValues, [tokenRows, width], storage);
            Tensor positions = Make(positionValues, [positionRows, width], storage);
            float[] output;
            using (AutogradContext.NoGrad())
            {
                if (packed)
                    output = tokens.ArcInferenceEmbeddingWithPositions(
                        positions, ids, positionOffset).Data.ToArray();
                else
                {
                    int[] padded = new int[positionOffset + ids.Length];
                    ids.CopyTo(padded, positionOffset);
                    float[] reference = tokens.EmbeddingLookupWithPositions(
                        positions, padded, 1, padded.Length).Data.ToArray();
                    output = reference.AsSpan(positionOffset * width, ids.Length * width).ToArray();
                }
            }
            Assert.Equal(packed, Tensor.ArcLane.KernelTimings.ContainsKey(
                "arc_inference_embedding_packed"));
            Tensor.ArcLane.CheckNumericStatus();
            return output;
        }

        float[] reference = Run(false);
        float[] candidate = Run(true);
        Assert.Equal(reference.Select(BitConverter.SingleToInt32Bits),
            candidate.Select(BitConverter.SingleToInt32Bits));
    }

    private static Tensor Make(float[] source, int[] shape, TensorDType storage)
    {
        var tensor = new Tensor((float[])source.Clone(), shape);
        if (storage == TensorDType.Bfp8)
            tensor.ConvertStorageInPlace(storage, Bfp8QuantizationDescriptor.Block(32));
        else if (storage == TensorDType.BFloat16)
            tensor.ConvertStorageInPlace(storage);
        return tensor;
    }

    private static float[] Values(int count, int seed)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = MathF.Sin(i * .017f + seed * .13f) * .05f;
        return values;
    }
}
