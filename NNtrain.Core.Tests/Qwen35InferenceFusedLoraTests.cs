using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35InferenceFusedLoraTests
{
    [Theory]
    [InlineData(32)]
    [InlineData(64)]
    [InlineData(128)]
    public void FusedProjectionMatchesSeparateBaseAndLoraBitwise(int workgroup)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, Qwen35ProjectionWorkgroupSize = workgroup
        });
        Assert.SkipWhen(!(lane.Options.XmxMatrices && lane.Device.SupportsXmx
            && lane.Device.MinimumSubgroupSize == 16
            && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups")), "Intel SG16 is required.");
        var random = new Random(975 + workgroup);
        float[] Values(int length) => Enumerable.Range(0, length)
            .Select(_ => (float)(random.NextDouble() * 0.8 - 0.4)).ToArray();
        foreach (var quant in new[]
        {
            (Name: "q4_k", Bytes: 144, ScaleOffset: 0, Minimum: true),
            (Name: "q5_k", Bytes: 176, ScaleOffset: 0, Minimum: true),
            (Name: "q6_k", Bytes: 210, ScaleOffset: 208, Minimum: false),
            (Name: "iq2_s", Bytes: 82, ScaleOffset: 0, Minimum: false),
            (Name: "iq3_s", Bytes: 110, ScaleOffset: 0, Minimum: false)
        })
        foreach (var shape in new[]
        {
            (Rows: 1, Input: 256, Output: 5, Rank: 3),
            (Rows: 2, Input: 512, Output: 7, Rank: 8),
            (Rows: 1, Input: 5120, Output: 33, Rank: 8),
            (Rows: 2, Input: 17408, Output: 9, Rank: 3)
        })
        {
            int blocks = shape.Input / 256 * shape.Output;
            var encoded = new byte[blocks * quant.Bytes];
            random.NextBytes(encoded);
            for (int block = 0; block < blocks; block++)
            {
                int start = block * quant.Bytes;
                WriteHalf(encoded, start + quant.ScaleOffset, (Half)((block % 7 + 1) / 8192f));
                if (quant.Minimum) WriteHalf(encoded, start + 2, (Half)((block % 3 + 1) / 16384f));
            }
            int count = shape.Rows * shape.Output;
            // Odd output counts leave unused subgroups in the last work-group.
            float[] sentinel = Enumerable.Repeat(float.NaN, count + 2).ToArray();
            using ArcBuffer x = lane.Upload(Values(shape.Rows * shape.Input));
            using ArcBuffer weight = lane.UploadRaw(encoded), bias = lane.Upload(Values(shape.Output));
            using ArcBuffer z = lane.Upload(Values(shape.Rows * shape.Rank));
            using ArcBuffer b = lane.Upload(Values(shape.Output * shape.Rank));
            using ArcBuffer baseline = lane.Upload(sentinel), fused = lane.Upload(sentinel);
            int outputsPerWorkgroup = workgroup / 16;
            long global = ((long)count + outputsPerWorkgroup - 1) / outputsPerWorkgroup * workgroup;
            foreach (float scale in new[] { 2f, -0.75f, 0f })
            {
                long uploaded = lane.H2DBytes, downloaded = lane.D2HBytes, allocated = lane.AllocatedBytes;
                lane.Run($"q35l_{quant.Name}_sg16", global, workgroup,
                    x, weight, bias, baseline, shape.Rows, shape.Input, shape.Output);
                lane.Run("q35l_lora_b", count, 0, z, b, baseline,
                    shape.Rows, shape.Output, shape.Rank, scale);
                lane.Run($"q35l_{quant.Name}_sg16_lora", global, workgroup,
                    x, weight, bias, fused, shape.Rows, shape.Input, shape.Output,
                    z, b, shape.Rank, scale);
                Assert.Equal(uploaded, lane.H2DBytes);
                Assert.Equal(downloaded, lane.D2HBytes);
                Assert.Equal(allocated, lane.AllocatedBytes);
                var expected = new float[count + 2];
                var actual = new float[count + 2];
                lane.Read(baseline, expected);
                lane.Read(fused, actual);
                for (int i = 0; i < count; i++)
                {
                    Assert.True(float.IsFinite(actual[i]));
                    Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                        $"{quant.Name}, WG {workgroup}, shape {shape}, scale {scale}, index {i}: expected {expected[i]:R}, actual {actual[i]:R}");
                }
                for (int i = count; i < count + 2; i++)
                    Assert.True(float.IsNaN(expected[i]) && float.IsNaN(actual[i]));
            }
        }
    }

    private static void WriteHalf(byte[] bytes, int offset, Half value)
    {
        ushort bits = BitConverter.HalfToUInt16Bits(value);
        bytes[offset] = (byte)bits;
        bytes[offset + 1] = (byte)(bits >> 8);
    }
}
