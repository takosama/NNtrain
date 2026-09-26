using Xunit;

namespace NNtrain.Core.Tests;

public sealed class ArcQwenQuantizedEmbeddingTests
{
    [Theory]
    [InlineData(Qwen2Gguf.Q4KType)]
    [InlineData(Qwen2Gguf.Q6KType)]
    public void LookupDecodesSelectedRowsWithoutReuploadingWeight(uint ggmlType)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");

        const int vocabulary = 3, width = 256;
        int[] tokenIds = [2, 0, 1, 2];
        byte[] payload = ggmlType == Qwen2Gguf.Q4KType
            ? Q4Payload(vocabulary) : Q6Payload(vocabulary);
        float[] decoded = ggmlType == Qwen2Gguf.Q4KType
            ? GgufQ4K.Dequantize(payload, vocabulary * width)
            : GgufQ6K.Dequantize(payload, vocabulary * width);

        using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Float32);
        using var matrix = new ArcQuantizedMatrix(payload, ggmlType, vocabulary, width);
        Assert.Equal(payload.Length, matrix.StorageBytes);

        using (AutogradContext.NoGrad())
        {
            Tensor result = matrix.LookupEmbedding(tokenIds, 2, 2);
            Assert.Equal([2, 2, width], result.Shape);
            float[] actual = result.Data.ToArray();
            for (int position = 0; position < tokenIds.Length; ++position)
                for (int column = 0; column < width; ++column)
                    Assert.InRange(
                        MathF.Abs(actual[position * width + column]
                            - decoded[tokenIds[position] * width + column]),
                        0f, 1e-5f);

            long uploadedAfterFirst = Tensor.ArcLane.H2DBytes;
            matrix.LookupEmbedding(tokenIds, 2, 2).Data.ToArray();
            Assert.Equal(uploadedAfterFirst + tokenIds.Length * sizeof(int),
                Tensor.ArcLane.H2DBytes);
        }

        string kernel = ggmlType == Qwen2Gguf.Q4KType
            ? "qwen_embedding_q4_k" : "qwen_embedding_q6_k";
        Assert.Contains(kernel, Tensor.ArcLane.KernelTimings.Keys);
    }

    private static byte[] Q4Payload(int rows)
    {
        var payload = new byte[rows * GgufQ4K.BlockBytes];
        for (int row = 0; row < rows; ++row)
        {
            int start = row * GgufQ4K.BlockBytes;
            WriteHalfOne(payload, start);
            for (int i = 0; i < 8; ++i)
                payload[start + 4 + i] = (byte)(i + row + 1);
            for (int i = 0; i < 128; ++i)
                payload[start + 16 + i] = (byte)(((i + row) & 15)
                    | (((i + row + 3) & 15) << 4));
        }
        return payload;
    }

    private static byte[] Q6Payload(int rows)
    {
        var payload = new byte[rows * GgufQ6K.BlockBytes];
        for (int row = 0; row < rows; ++row)
        {
            int start = row * GgufQ6K.BlockBytes;
            for (int i = 0; i < 128; ++i)
                payload[start + i] = unchecked((byte)(i * 17 + row * 11));
            for (int i = 0; i < 64; ++i)
                payload[start + 128 + i] = unchecked((byte)(i * 37 + row * 13));
            for (int i = 0; i < 16; ++i)
                payload[start + 192 + i] = unchecked((byte)(sbyte)((i + row) % 5 - 2));
            WriteHalfOne(payload, start + 208);
        }
        return payload;
    }

    private static void WriteHalfOne(byte[] payload, int offset)
    {
        payload[offset] = 0;
        payload[offset + 1] = 0x3c;
    }
}
