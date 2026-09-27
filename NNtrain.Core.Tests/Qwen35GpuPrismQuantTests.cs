using System.Text.Json;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35GpuPrismQuantTests
{
    [Theory]
    [InlineData(142u, "pq2_0", 34)]
    [InlineData(143u, "ptq1_0", 28)]
    public void PackedDecodeAndProjectionMatchAuthorNativeReference(uint type, string quant, int blockBytes)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using Stream stream = typeof(Qwen35GpuPrismQuantTests).Assembly.GetManifestResourceStream(
            "NNtrain.Core.Tests.Fixtures.PrismQuantReference.prism-quant-blocks.json")!;
        using JsonDocument fixture = JsonDocument.Parse(stream);
        var cases = fixture.RootElement.GetProperty("cases").EnumerateArray()
            .Where(item => item.GetProperty("type").GetUInt32() == type)
            .Select(item => (Encoded: Convert.FromBase64String(item.GetProperty("encoded").GetString()!),
                Decoded: item.GetProperty("decoded").EnumerateArray().Select(value => value.GetSingle()).ToArray())).ToArray();
        Assert.NotEmpty(cases);
        Assert.All(cases, c => { Assert.Equal(blockBytes, c.Encoded.Length); Assert.Equal(128, c.Decoded.Length); });
        foreach (int group in new[] { 32, 64, 128 })
        {
            using var lane = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true, Qwen35ProjectionWorkgroupSize = group });
            if (group == 32)
            {
                using ArcBuffer encoded = lane.UploadRaw(cases.SelectMany(c => c.Encoded).ToArray());
                foreach (int row in Enumerable.Range(0, cases.Length))
                {
                    using ArcBuffer id = lane.UploadRaw(new[] { row });
                    using ArcBuffer result = lane.Upload(Enumerable.Repeat(float.NaN, 130).ToArray());
                    lane.Run($"q35l_{quant}_embedding", 128, 0, encoded, id, result, 128);
                    var values = new float[130]; lane.Read(result, values);
                    Assert.True(float.IsNaN(values[128]) && float.IsNaN(values[129]));
                    for (int i = 0; i < 128; i++) Assert.Equal(cases[row].Decoded[i], values[i]);
                }
            }
            foreach (var shape in new[] { (Rows: 2, Width: 128), (Rows: 1, Width: 384), (Rows: 2, Width: 5120), (Rows: 1, Width: 17408) })
            {
                const int outputs = 7;
                int blocks = shape.Width / 128;
                var encoded = new byte[outputs * blocks * blockBytes];
                var weights = new float[outputs * shape.Width];
                for (int block = 0; block < outputs * blocks; block++)
                {
                    var c = cases[(block * 37 + 11) % cases.Length];
                    c.Encoded.CopyTo(encoded, block * blockBytes); c.Decoded.CopyTo(weights, block * 128);
                }
                CheckProjection(lane, quant, encoded, weights, shape.Rows, shape.Width, outputs, group);
            }
        }
    }

    [Fact]
    public void Bf16ProjectionsRemainPackedAndMatchFloatReference()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true });
        const int width = 5120, outputs = 48;
        var bytes = new byte[width * outputs * 2]; var decoded = new float[width * outputs];
        for (int i = 0; i < decoded.Length; i++)
        {
            ushort bits = (ushort)(BitConverter.SingleToUInt32Bits(MathF.Sin(i * .31f) * .05f) >> 16);
            bytes[2 * i] = (byte)bits; bytes[2 * i + 1] = (byte)(bits >> 8);
            decoded[i] = BitConverter.UInt32BitsToSingle((uint)bits << 16);
        }
        CheckProjection(lane, "bf16", bytes, decoded, 2, width, outputs, 32);
    }

    private static void CheckProjection(ArcExecutionLane lane, string quant, byte[] encoded, float[] weights,
        int rows, int width, int outputs, int group)
    {
        float[] input = Enumerable.Range(0, rows * width).Select(i => MathF.Cos(i * .071f) * .025f).ToArray();
        float[] bias = Enumerable.Range(0, outputs).Select(i => (i % 5 - 2) * .007f).ToArray();
        var expected = new double[rows * outputs]; var tolerance = new double[expected.Length];
        for (int row = 0; row < rows; row++)
        for (int output = 0; output < outputs; output++)
        {
            double sum = bias[output], absolute = Math.Abs(sum);
            for (int i = 0; i < width; i++)
            {
                double product = (double)input[row * width + i] * weights[output * width + i];
                sum += product; absolute += Math.Abs(product);
            }
            expected[row * outputs + output] = sum;
            tolerance[row * outputs + output] = 1e-6 + absolute * 4e-6;
        }
        using ArcBuffer x = lane.Upload(input), w = lane.UploadRaw(encoded), b = lane.Upload(bias);
        Assert.Equal(encoded.Length, w.ByteLength);
        bool sg16 = lane.Device.MinimumSubgroupSize == 16 && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups");
        foreach (string variant in sg16 ? new[] { "reference", "coop64", "sg16" } : new[] { "reference", "coop64" })
        {
            using ArcBuffer output = lane.Upload(Enumerable.Repeat(float.NaN, expected.Length + 2).ToArray());
            long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
            if (variant == "sg16") lane.Run($"q35l_{quant}_{variant}", ((long)expected.Length + group / 16 - 1) / (group / 16) * group,
                group, x, w, b, output, rows, width, outputs);
            else if (variant == "coop64") lane.Run2D($"q35l_{quant}_{variant}", outputs * 64, rows, 64, 1,
                x, w, b, output, rows, width, outputs);
            else lane.Run($"q35l_{quant}_{variant}", expected.Length, 0, x, w, b, output, rows, width, outputs);
            Assert.Equal(uploads, lane.H2DBytes); Assert.Equal(downloads, lane.D2HBytes);
            var actual = new float[expected.Length + 2]; lane.Read(output, actual);
            Assert.True(float.IsNaN(actual[^1]) && float.IsNaN(actual[^2]));
            for (int i = 0; i < expected.Length; i++)
                Assert.True(double.IsFinite(actual[i]) && Math.Abs(actual[i] - expected[i]) <= tolerance[i],
                    $"{quant}/{variant} WG{group} width{width} output{i}: {actual[i]} vs {expected[i]}, tolerance {tolerance[i]}");
        }
    }
}
