using NNtrain;
using Xunit;

public sealed class TensorFloat16StorageCodecTests
{
    [Fact]
    public void DecodeMatchesRuntimeForEveryBinary16BitPattern()
    {
        var source = new Half[ushort.MaxValue + 1];
        var actual = new float[source.Length];
        for (int bits = 0; bits < source.Length; bits++)
            source[bits] = BitConverter.UInt16BitsToHalf((ushort)bits);

        TensorStorageCodec.DecodeFloat16(source, actual);

        for (int bits = 0; bits < source.Length; bits++)
        {
            float expected = (float)source[bits];
            Assert.Equal(
                BitConverter.SingleToUInt32Bits(expected),
                BitConverter.SingleToUInt32Bits(actual[bits]));
        }
    }

    [Fact]
    public void EncodeMatchesRuntimeAtIeeeBoundariesAndRoundTies()
    {
        float minimumSubnormal = (float)BitConverter.UInt16BitsToHalf(0x0001);
        float minimumNormal = (float)BitConverter.UInt16BitsToHalf(0x0400);
        float maximumFinite = (float)BitConverter.UInt16BitsToHalf(0x7BFF);
        float[] source =
        [
            0f,
            -0f,
            minimumSubnormal,
            -minimumSubnormal,
            minimumNormal,
            -minimumNormal,
            maximumFinite,
            -maximumFinite,
            float.PositiveInfinity,
            float.NegativeInfinity,
            float.NaN,
            1f + MathF.Pow(2f, -11f),
            1f + 3f * MathF.Pow(2f, -11f),
        ];
        var actual = new Half[source.Length];

        TensorStorageCodec.EncodeFloat16(source, actual);

        for (int index = 0; index < source.Length; index++)
        {
            ushort expectedBits =
                BitConverter.HalfToUInt16Bits((Half)source[index]);
            ushort actualBits = BitConverter.HalfToUInt16Bits(actual[index]);
            if (float.IsNaN(source[index]))
            {
                Assert.True(Half.IsNaN(actual[index]));
            }
            else
            {
                Assert.Equal(expectedBits, actualBits);
            }
        }
    }

    [Fact]
    public void VectorizedEncodeMatchesRuntimeForNormalValuesAndMixedFallbacks()
    {
        var random = new Random(7319);
        var source = new float[131_079];
        for (int index = 0; index < source.Length; index++)
        {
            source[index] = index % 17 switch
            {
                0 => 0f,
                1 => -0f,
                2 => float.PositiveInfinity,
                3 => float.NegativeInfinity,
                4 => float.NaN,
                5 => (float)BitConverter.UInt16BitsToHalf(0x0001),
                6 => -(float)BitConverter.UInt16BitsToHalf(0x0001),
                _ => (float)((random.NextDouble() * 2d - 1d) * 65_500d),
            };
        }
        var actual = new Half[source.Length];

        TensorStorageCodec.EncodeFloat16(source, actual);

        for (int index = 0; index < source.Length; index++)
        {
            Half expected = (Half)source[index];
            if (Half.IsNaN(expected))
                Assert.True(Half.IsNaN(actual[index]));
            else
                Assert.Equal(
                    BitConverter.HalfToUInt16Bits(expected),
                    BitConverter.HalfToUInt16Bits(actual[index]));
        }
    }

    [Fact]
    public void TorchFactoriesExposeFloat16DType()
    {
        Tensor fromData = torch.tensor(
            [1.25f, -2.5f],
            [2],
            dtype: torch.float16);
        Tensor zeros = torch.zeros([2, 3], dtype: torch.half);
        Tensor scalar = torch.scalar(0.5f, dtype: torch.float16);

        Assert.Equal(TensorDType.Float16, fromData.DType);
        Assert.Equal(TensorDType.Float16, zeros.DType);
        Assert.Equal(TensorDType.Float16, scalar.DType);
        Assert.Equal(2 * sizeof(ushort), fromData.StorageByteLength);
    }

    [Fact]
    public void Float16StorageUsesTwoBytesPerElementWithoutHiddenMirror()
    {
        var tensor = new Tensor(
            Enumerable.Range(0, 257).Select(static value => (float)value).ToArray(),
            [257],
            dtype: TensorDType.Float16);

        Assert.Equal(257 * sizeof(ushort), tensor.StorageByteLength);
        Assert.Equal(TensorDType.Float16, tensor.DType);
        Assert.Equal(TensorDType.Float32, tensor.ComputeDType);
        Assert.Equal(TensorDType.Float32, tensor.AccumulationDType);
    }

    [Fact]
    public void ReservedLowBitDTypesFailBeforeAllocatingStorage()
    {
        TensorDType[] reserved =
        [
            TensorDType.Float8E4M3Fn,
            TensorDType.Float8E5M2,
            TensorDType.Float4,
            TensorDType.Float2,
            TensorDType.Ternary1Bit58,
        ];

        foreach (TensorDType dtype in reserved)
        {
            NotSupportedException exception = Assert.Throws<NotSupportedException>(
                () => new Tensor([1f], [1], dtype: dtype));
            Assert.Contains("reserved", exception.Message);
        }
    }
}
