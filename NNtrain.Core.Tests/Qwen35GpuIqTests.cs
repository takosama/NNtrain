using System.Text.Json;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35GpuIqTests
{
    [Theory]
    [InlineData(Qwen35Gguf.Q5KType, "q5_k", 176)]
    [InlineData(Qwen35Gguf.IQ2SType, "iq2_s", 82)]
    [InlineData(Qwen35Gguf.IQ3SType, "iq3_s", 110)]
    public void EncodedEmbeddingAndGemvMatchIndependentNativeGgml(uint type, string quant, int blockBytes)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true });
        using Stream stream = typeof(Qwen35GpuIqTests).Assembly.GetManifestResourceStream(
            "NNtrain.Core.Tests.Fixtures.IqQuantReference.iq-quant-blocks.json")!;
        using JsonDocument fixture = JsonDocument.Parse(stream);
        var cases = fixture.RootElement.GetProperty("cases").EnumerateArray()
            .Where(item => item.GetProperty("type").GetUInt32() == type)
            .Select(item => (
                Encoded: Convert.FromBase64String(item.GetProperty("encoded").GetString()!),
                Decoded: item.GetProperty("decoded").EnumerateArray().Select(value => value.GetSingle()).ToArray()))
            .ToArray();
        Assert.NotEmpty(cases);
        Assert.All(cases, item => { Assert.Equal(blockBytes, item.Encoded.Length); Assert.Equal(256, item.Decoded.Length); });

        // Each fixture block is an embedding row. Checking every row exercises
        // every IQ lookup-table entry, all signs and FP16 edge cases.
        using (ArcBuffer encoded = lane.UploadRaw(cases.SelectMany(item => item.Encoded).ToArray()))
        {
            Assert.Equal((long)cases.Length * blockBytes, encoded.ByteLength);
            for (int row = 0; row < cases.Length; row++)
            {
                using ArcBuffer id = lane.UploadRaw(new[] { row });
                using ArcBuffer result = lane.Upload(Enumerable.Repeat(float.NaN, 258).ToArray());
                long uploads = lane.H2DBytes, allocated = lane.AllocatedBytes;
                lane.Run($"q35l_{quant}_embedding", 256, 0, encoded, id, result, 256);
                Assert.Equal(uploads, lane.H2DBytes);
                Assert.Equal(allocated, lane.AllocatedBytes);
                var values = new float[258];
                lane.Read(result, values);
                Assert.True(float.IsNaN(values[256]) && float.IsNaN(values[257]));
                for (int i = 0; i < 256; i++)
                {
                    double expected = cases[row].Decoded[i];
                    Assert.True(float.IsFinite(values[i]));
                    Assert.InRange(Math.Abs(values[i] - expected), 0, 1e-10 + Math.Abs(expected) * 2e-7);
                }
            }
        }

        bool subgroup = lane.Options.XmxMatrices && lane.Device.SupportsXmx
            && lane.Device.MinimumSubgroupSize == 16 && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups");
        string[] variants = subgroup ? ["reference", "coop64", "sg16"] : ["reference", "coop64"];
        foreach (var shape in new[] { (Rows: 3, Input: 256, Output: cases.Length), (Rows: 1, Input: 5120, Output: 7), (Rows: 1, Input: 17408, Output: 5) })
        {
            int blocks = shape.Input / 256;
            byte[] payload = new byte[shape.Output * blocks * blockBytes];
            float[] decoded = new float[shape.Output * shape.Input];
            for (int block = 0; block < shape.Output * blocks; block++)
            {
                var item = cases[block % cases.Length];
                item.Encoded.CopyTo(payload, block * blockBytes);
                item.Decoded.CopyTo(decoded, block * 256);
            }
            float[] input = Enumerable.Range(0, shape.Rows * shape.Input).Select(i => MathF.Sin(i * 0.37f) * 0.125f).ToArray();
            float[] bias = Enumerable.Range(0, shape.Output).Select(i => (i % 7 - 3) * 0.01f).ToArray();
            int count = shape.Rows * shape.Output;
            var expected = new double[count];
            var tolerances = new double[count];
            for (int row = 0; row < shape.Rows; row++)
            for (int output = 0; output < shape.Output; output++)
            {
                double sum = bias[output], absoluteSum = Math.Abs(sum);
                for (int i = 0; i < shape.Input; i++)
                {
                    double product = (double)input[row * shape.Input + i] * decoded[output * shape.Input + i];
                    sum += product;
                    absoluteSum += Math.Abs(product);
                }
                expected[row * shape.Output + output] = sum;
                tolerances[row * shape.Output + output] = 2e-6 + absoluteSum * 3e-6;
            }
            using ArcBuffer x = lane.Upload(input), w = lane.UploadRaw(payload), b = lane.Upload(bias);
            foreach (string variant in variants)
            {
                using ArcBuffer result = lane.Upload(Enumerable.Repeat(float.NaN, count + 2).ToArray());
                long uploads = lane.H2DBytes, downloads = lane.D2HBytes, allocated = lane.AllocatedBytes;
                string kernel = $"q35l_{quant}_{variant}";
                if (variant == "sg16")
                    lane.Run(kernel, ((long)count + 1) / 2 * 32, 32, x, w, b, result, shape.Rows, shape.Input, shape.Output);
                else if (variant == "coop64")
                    lane.Run2D(kernel, (long)shape.Output * 64, shape.Rows, 64, 1, x, w, b, result, shape.Rows, shape.Input, shape.Output);
                else
                    lane.Run(kernel, count, 0, x, w, b, result, shape.Rows, shape.Input, shape.Output);
                Assert.Equal(uploads, lane.H2DBytes);
                Assert.Equal(downloads, lane.D2HBytes);
                Assert.Equal(allocated, lane.AllocatedBytes);
                var actual = new float[count + 2];
                lane.Read(result, actual);
                Assert.True(float.IsNaN(actual[count]) && float.IsNaN(actual[count + 1]));
                for (int i = 0; i < count; i++)
                {
                    Assert.True(float.IsFinite(actual[i]));
                    Assert.True(Math.Abs(actual[i] - expected[i]) <= tolerances[i],
                        $"{quant}/{variant} width={shape.Input}, index={i}: {actual[i]} vs {expected[i]}, tolerance={tolerances[i]}");
                }
            }
        }
    }
}
