using System.Runtime.InteropServices;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35VisionPreprocessorTests
{
    [Fact]
    public void PatchesFollowMergedBlockOrderAndNormalizeRgbChannels()
    {
        byte[] rgb = new byte[4 * 4 * 3];
        for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
        {
            int pixel = (y * 4 + x) * 3;
            rgb[pixel] = (byte)(y * 4 + x);
            rgb[pixel + 1] = 255;
        }

        Qwen35VisionInput input = Qwen35VisionPreprocessor.Prepare(rgb, 4, 4,
            patchSize: 1, mergeSize: 2, maximumPixels: 16);

        Assert.Equal(4, input.GridHeight);
        Assert.Equal(4, input.GridWidth);
        Assert.Equal(4 * 4 * 3, input.Patches.Length);
        int[] redOrder = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
        for (int patch = 0; patch < redOrder.Length; patch++)
        {
            AssertClose(redOrder[patch] * (2f / 255f) - 1f, input.Patches[patch * 3]);
            Assert.Equal(1f, input.Patches[patch * 3 + 1]);
            Assert.Equal(-1f, input.Patches[patch * 3 + 2]);
        }
    }

    [Fact]
    public void EachPatchIsChannelThenRowThenColumn()
    {
        byte[] rgb = new byte[4 * 4 * 3];
        for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
        {
            int pixel = (y * 4 + x) * 3;
            rgb[pixel] = (byte)(y * 4 + x);
            rgb[pixel + 1] = 64;
            rgb[pixel + 2] = 192;
        }

        Qwen35VisionInput input = Qwen35VisionPreprocessor.Prepare(rgb, 4, 4,
            patchSize: 2, mergeSize: 2, maximumPixels: 16);

        Assert.Equal(2, input.GridHeight);
        Assert.Equal(2, input.GridWidth);
        Assert.Equal(4 * 3 * 2 * 2, input.Patches.Length);
        int[][] redPatches = [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]];
        for (int patch = 0; patch < 4; patch++)
        {
            int start = patch * 12;
            for (int pixel = 0; pixel < 4; pixel++)
                AssertClose(redPatches[patch][pixel] * (2f / 255f) - 1f, input.Patches[start + pixel]);
            for (int pixel = 4; pixel < 8; pixel++)
                AssertClose(64 * (2f / 255f) - 1f, input.Patches[start + pixel]);
            for (int pixel = 8; pixel < 12; pixel++)
                AssertClose(192 * (2f / 255f) - 1f, input.Patches[start + pixel]);
        }
    }

    [Fact]
    public void SmartResizeUpscalesSmallImagesToModelMinimumWhileKeepingAspect()
    {
        byte[] rgb = SolidRgb(64, 32, 128);

        Qwen35VisionInput input = Qwen35VisionPreprocessor.Prepare(rgb, 64, 32);

        Assert.Equal(12, input.GridHeight); // 192 / 16
        Assert.Equal(24, input.GridWidth);  // 384 / 16
        Assert.Equal(12 * 24 * 3 * 16 * 16, input.Patches.Length);
        Assert.All(input.Patches, value => AssertClose(128 * (2f / 255f) - 1f, value));
    }

    [Fact]
    public void SmartResizeDownscalesToPixelCapOnFactorAlignedAspect()
    {
        byte[] rgb = SolidRgb(96, 48, 200);

        Qwen35VisionInput input = Qwen35VisionPreprocessor.Prepare(rgb, 96, 48,
            patchSize: 4, mergeSize: 2, maximumPixels: 1536);

        Assert.Equal(6, input.GridHeight);  // 24 / 4
        Assert.Equal(12, input.GridWidth);  // 48 / 4
        Assert.True((long)input.GridHeight * input.GridWidth * 4 * 4 <= 1536);
        Assert.Equal(2.0, (double)input.GridWidth / input.GridHeight);
        Assert.All(input.Patches, value => AssertClose(200 * (2f / 255f) - 1f, value));
    }

    [Fact]
    public void BicubicDownscaleAntialiasesAlternatingPixels()
    {
        byte[] rgb = new byte[64 * 64 * 3];
        for (int y = 0; y < 64; y++)
        for (int x = 0; x < 64; x++)
        {
            byte value = (byte)(((x + y) & 1) == 0 ? 0 : 255);
            int pixel = (y * 64 + x) * 3;
            rgb[pixel] = value;
            rgb[pixel + 1] = value;
            rgb[pixel + 2] = value;
        }

        Qwen35VisionInput input = Qwen35VisionPreprocessor.Prepare(rgb, 64, 64,
            patchSize: 1, mergeSize: 2, maximumPixels: 32 * 32);

        Assert.Equal(32, input.GridHeight);
        Assert.Equal(32, input.GridWidth);
        Assert.All(input.Patches, value => Assert.InRange(value, -0.1f, 0.1f));
    }

    [Fact]
    public void PrepareRejectsInconsistentShapesAndCaps()
    {
        Assert.Throws<ArgumentException>(() => Qwen35VisionPreprocessor.Prepare(new byte[11], 2, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => Qwen35VisionPreprocessor.Prepare(new byte[3], 0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Qwen35VisionPreprocessor.Prepare(new byte[3], 1, 1,
            patchSize: 16, mergeSize: 2, maximumPixels: 100));
        Assert.Throws<ArgumentOutOfRangeException>(() => Qwen35VisionPreprocessor.Prepare(new byte[201 * 3], 201, 1));
    }

    [Theory]
    [InlineData(4096, 3072)]
    [InlineData(1200, 1600)]
    [InlineData(1025, 513)]
    [InlineData(768, 768)]
    public void ParallelRowsPreserveSingleWorkerPatchBits(int width, int height)
    {
        byte[] rgb = new byte[checked(width * height * 3)];
        new Random(42).NextBytes(rgb);
        Qwen35VisionInput sequential = Qwen35VisionPreprocessor.Prepare(rgb, width, height,
            patchSize: 16, mergeSize: 2, maximumPixels: 768 * 768, maximumDegreeOfParallelism: 1);
        Qwen35VisionInput parallel = Qwen35VisionPreprocessor.Prepare(rgb, width, height,
            patchSize: 16, mergeSize: 2, maximumPixels: 768 * 768, maximumDegreeOfParallelism: 8);
        Assert.Equal(sequential.GridHeight, parallel.GridHeight);
        Assert.Equal(sequential.GridWidth, parallel.GridWidth);
        Assert.True(MemoryMarshal.Cast<float, int>(sequential.Patches.AsSpan()).SequenceEqual(
            MemoryMarshal.Cast<float, int>(parallel.Patches.AsSpan())), "Parallel preprocessing changed patch bits.");
    }

    private static byte[] SolidRgb(int width, int height, byte value)
    {
        byte[] rgb = new byte[width * height * 3];
        Array.Fill(rgb, value);
        return rgb;
    }

    private static void AssertClose(float expected, float actual)
        => Assert.InRange(actual, expected - 1e-5f, expected + 1e-5f);
}
