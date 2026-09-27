using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35InferenceHalfScaleTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EveryHalfEncodingPreservesFiniteBitsAndNanClassification(bool nativeHalf)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, ExperimentalOptimizationKernels = true,
            Qwen35NativeHalfScale = nativeHalf
        });
        const int count = 65536;
        ushort[] encoded = Enumerable.Range(0, count).Select(i => (ushort)i).ToArray();
        using ArcBuffer input = lane.UploadRaw(encoded);
        using ArcBuffer output = lane.Upload(Enumerable.Repeat(float.NaN, count + 2).ToArray());
        lane.Run("q35l_half_convert_probe", count, 128, input, output, count);
        var actual = new float[count + 2];
        lane.Read(output, actual);
        for (int bits = 0; bits < count; bits++)
        {
            float expected = (float)BitConverter.UInt16BitsToHalf((ushort)bits);
            if (float.IsNaN(expected))
                Assert.True(float.IsNaN(actual[bits]), $"Half 0x{bits:X4} must remain NaN.");
            else
                Assert.True(BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(actual[bits]),
                    $"Half 0x{bits:X4}, native={nativeHalf}: expected 0x{BitConverter.SingleToInt32Bits(expected):X8}, actual 0x{BitConverter.SingleToInt32Bits(actual[bits]):X8}");
        }
        // Exact-bit comparisons above include both signed zeros, infinities,
        // all positive/negative subnormals and the normal/subnormal boundaries.
        Assert.True(float.IsNaN(actual[count]) && float.IsNaN(actual[count + 1]));
        Assert.Contains("q35l_half_convert_probe", lane.KernelTimings.Keys);
    }
}
