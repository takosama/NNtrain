using System.Text.Json;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35TrainingProjectionTests
{
    [Theory]
    [InlineData(Qwen2Gguf.Q4KType, "q4_k", 144)]
    [InlineData(Qwen2Gguf.Q6KType, "q6_k", 210)]
    [InlineData(Qwen35Gguf.Q5KType, "q5_k", 176)]
    [InlineData(Qwen35Gguf.IQ2SType, "iq2_s", 82)]
    [InlineData(Qwen35Gguf.IQ3SType, "iq3_s", 110)]
    public void EncodedTransposeAndSplitReductionMatchIndependentDecodedWeights(uint type, string quant, int blockBytes)
    {
        using ArcExecutionLane lane = CreateLane();
        const int rows = 2, tile = 1024, output = tile + 7, splits = 2;
        foreach (int input in new[] { 256, 512 })
        {
            (byte[] encoded, float[] decoded) = Weights(type, input * output, blockBytes);
            float[] gradient = Values(rows * output, 91 + input);
            float[] initial = Values(rows * input, 101);
            using ArcBuffer dy = lane.Upload(gradient), w = lane.UploadRaw(encoded);
            using ArcBuffer partial = lane.Upload(Enumerable.Repeat(float.NaN, rows * splits * input + 2).ToArray());
            using ArcBuffer dx = lane.Upload(initial.Concat(new[] { float.NaN, float.NaN }).ToArray());
            long uploads = lane.H2DBytes, downloads = lane.D2HBytes, allocation = lane.AllocatedBytes;
            lane.Run("q35t_xpose_" + quant, rows * splits * input, 128,
                dy, w, partial, rows, input, output, splits, tile);
            lane.Run("q35t_xpose_reduce", rows * input, 128, partial, dx, rows, input, splits);
            Assert.Equal(uploads, lane.H2DBytes);
            Assert.Equal(downloads, lane.D2HBytes);
            Assert.Equal(allocation, lane.AllocatedBytes);
            Assert.Equal(encoded.Length, w.ByteLength);
            float[] actualPartial = Read(lane, partial, rows * splits * input + 2);
            float[] actual = Read(lane, dx, rows * input + 2);
            Assert.True(float.IsNaN(actualPartial[^1]) && float.IsNaN(actualPartial[^2]));
            Assert.True(float.IsNaN(actual[^1]) && float.IsNaN(actual[^2]));
            var sums = new double[rows * input];
            var magnitudes = new double[sums.Length];
            for (int row = 0; row < rows; ++row)
                for (int j = 0; j < input; ++j)
                {
                    for (int split = 0; split < splits; ++split)
                    {
                        double sum = 0, magnitude = 0;
                        for (int o = split * tile; o < Math.Min(output, (split + 1) * tile); ++o)
                        {
                            double product = (double)gradient[row * output + o] * decoded[o * input + j];
                            sum += product;
                            magnitude += Math.Abs(product);
                        }
                        Close(sum, actualPartial[(row * splits + split) * input + j], 2e-6 + magnitude * 5e-6);
                        sums[row * input + j] += sum;
                        magnitudes[row * input + j] += magnitude;
                    }
                    int index = row * input + j;
                    Close(initial[index] + sums[index], actual[index], 2e-6 + magnitudes[index] * 5e-6);
                }
            // The reducer must accumulate into an existing input gradient.
            lane.Run("q35t_xpose_reduce", rows * input, 128, partial, dx, rows, input, splits);
            actual = Read(lane, dx, rows * input + 2);
            for (int i = 0; i < sums.Length; ++i)
                Close(initial[i] + 2 * sums[i], actual[i], 4e-6 + magnitudes[i] * 1e-5);
            Assert.True(float.IsNaN(actual[^1]) && float.IsNaN(actual[^2]));
        }
    }

    [Fact]
    public void LoraForwardAndAllAdjointsMatchCpuAndAccumulateWithNonzeroB()
    {
        using ArcExecutionLane lane = CreateLane();
        const int rows = 2, input = 7, output = 5, rank = 3;
        var options = new Qwen35LoraOptions { Rank = rank, Alpha = 4.5f };
        using var adapter = new Qwen35LoraMatrix(lane, input, output, options, new Random(11));
        float[] a = Values(rank * input, 13), b = Values(output * rank, 17);
        adapter.RestoreState([a, b, new float[a.Length], new float[b.Length], new float[a.Length], new float[b.Length]], training: true);
        float[] inputValues = Values(rows * input, 19), baseOutput = Values(rows * output, 23);
        float[] upstream = Values(rows * output, 29), initialDx = Values(rows * input, 31);
        using ArcBuffer x = lane.Upload(inputValues), y = lane.Upload(baseOutput), dy = lane.Upload(upstream);
        using ArcBuffer dx = lane.Upload(initialDx);
        long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
        using ArcBuffer z = adapter.Forward(x, y, rows);
        adapter.Backward(x, z, dy, dx, rows);
        Assert.Equal(uploads, lane.H2DBytes);
        Assert.Equal(downloads, lane.D2HBytes);
        double scale = options.Alpha / rank;
        var expectedZ = new double[rows * rank];
        var expectedDz = new double[rows * rank];
        var expectedY = baseOutput.Select(v => (double)v).ToArray();
        var expectedDa = new double[a.Length];
        var expectedDb = new double[b.Length];
        var expectedDx = new double[inputValues.Length];
        for (int t = 0; t < rows; ++t)
            for (int r = 0; r < rank; ++r)
            {
                for (int j = 0; j < input; ++j)
                    expectedZ[t * rank + r] += (double)inputValues[t * input + j] * a[r * input + j];
                for (int o = 0; o < output; ++o)
                    expectedDz[t * rank + r] += scale * upstream[t * output + o] * b[o * rank + r];
            }
        for (int t = 0; t < rows; ++t)
            for (int r = 0; r < rank; ++r)
            {
                for (int o = 0; o < output; ++o)
                {
                    expectedY[t * output + o] += scale * expectedZ[t * rank + r] * b[o * rank + r];
                    expectedDb[o * rank + r] += scale * upstream[t * output + o] * expectedZ[t * rank + r];
                }
                for (int j = 0; j < input; ++j)
                {
                    expectedDa[r * input + j] += expectedDz[t * rank + r] * inputValues[t * input + j];
                    expectedDx[t * input + j] += expectedDz[t * rank + r] * a[r * input + j];
                }
            }
        Close(expectedZ, Read(lane, z, expectedZ.Length));
        Close(expectedY, Read(lane, y, expectedY.Length));
        float[][] gradients = adapter.ReadGradients();
        Close(expectedDa, gradients[0]);
        Close(expectedDb, gradients[1]);
        Close(expectedDx.Select((g, i) => g + initialDx[i]).ToArray(), Read(lane, dx, expectedDx.Length));
        adapter.Backward(x, z, dy, dx, rows);
        gradients = adapter.ReadGradients();
        Close(expectedDa.Select(g => g * 2).ToArray(), gradients[0]);
        Close(expectedDb.Select(g => g * 2).ToArray(), gradients[1]);
        Close(expectedDx.Select((g, i) => g * 2 + initialDx[i]).ToArray(), Read(lane, dx, expectedDx.Length));
        adapter.ZeroGrad();
        adapter.Backward(x, z, dy, null, rows);
        gradients = adapter.ReadGradients();
        Close(expectedDa, gradients[0]);
        Close(expectedDb, gradients[1]);
        Assert.Equal(a, Read(lane, adapter.A, a.Length));
        Assert.Equal(b, Read(lane, adapter.B, b.Length));
        Assert.Equal(inputValues, Read(lane, x, inputValues.Length));
    }

    [Fact]
    public void ZeroInitializedBPreservesBaseOutputAndFirstStepOnlyTrainsB()
    {
        using ArcExecutionLane lane = CreateLane();
        const int rows = 2, input = 7, output = 5, rank = 3;
        using var adapter = new Qwen35LoraMatrix(lane, input, output, new() { Rank = rank, Alpha = 6 }, new Random(41));
        float[] baseOutput = Values(rows * output, 43);
        using ArcBuffer x = lane.Upload(Values(rows * input, 47)), y = lane.Upload(baseOutput);
        using ArcBuffer dy = lane.Upload(Values(rows * output, 53)), dx = lane.Upload(new float[rows * input]);
        using ArcBuffer z = adapter.Forward(x, y, rows);
        Assert.Equal(baseOutput, Read(lane, y, baseOutput.Length));
        adapter.Backward(x, z, dy, dx, rows);
        float[][] gradient = adapter.ReadGradients();
        Assert.All(gradient[0], value => Assert.Equal(0f, value));
        Assert.Contains(gradient[1], value => Math.Abs(value) > 1e-6);
        Assert.All(Read(lane, dx, rows * input), value => Assert.Equal(0f, value));
    }

    private static (byte[] Encoded, float[] Decoded) Weights(uint type, int elements, int blockBytes)
    {
        if (type is Qwen2Gguf.Q4KType or Qwen2Gguf.Q6KType)
        {
            // Independent existing CPU codecs decode the same randomized block bytes.
            byte[] payload = new byte[elements / 256 * blockBytes];
            new Random(211 + (int)type).NextBytes(payload);
            for (int block = 0; block < elements / 256; ++block)
            {
                ushort scale = block % 9 == 0 ? (ushort)0x8001
                    : BitConverter.HalfToUInt16Bits((Half)((block % 7 + 1) / 8192f));
                int start = block * blockBytes;
                if (type == Qwen2Gguf.Q4KType)
                {
                    WriteHalf(payload, start, scale);
                    WriteHalf(payload, start + 2, BitConverter.HalfToUInt16Bits((Half)((block % 3 + 1) / 16384f)));
                }
                else WriteHalf(payload, start + 208, scale);
            }
            return (payload, type == Qwen2Gguf.Q4KType
                ? GgufQ4K.Dequantize(payload, elements) : GgufQ6K.Dequantize(payload, elements));
        }
        // Q5/IQ expected floats were produced by native ggml, not NNtrain decoders.
        using Stream stream = typeof(Qwen35TrainingProjectionTests).Assembly.GetManifestResourceStream(
            "NNtrain.Core.Tests.Fixtures.IqQuantReference.iq-quant-blocks.json")!;
        using JsonDocument fixture = JsonDocument.Parse(stream);
        var cases = fixture.RootElement.GetProperty("cases").EnumerateArray()
            .Where(item => item.GetProperty("type").GetUInt32() == type)
            .Select(item => (
                Encoded: Convert.FromBase64String(item.GetProperty("encoded").GetString()!),
                Decoded: item.GetProperty("decoded").EnumerateArray().Select(value => value.GetSingle()).ToArray()))
            .ToArray();
        Assert.NotEmpty(cases);
        var encoded = new byte[elements / 256 * blockBytes];
        var decoded = new float[elements];
        for (int block = 0; block < elements / 256; ++block)
        {
            var item = cases[block % cases.Length];
            Assert.Equal(blockBytes, item.Encoded.Length);
            Assert.Equal(256, item.Decoded.Length);
            item.Encoded.CopyTo(encoded, block * blockBytes);
            item.Decoded.CopyTo(decoded, block * 256);
        }
        return (encoded, decoded);
    }

    private static void WriteHalf(byte[] bytes, int offset, ushort bits)
    {
        bytes[offset] = (byte)bits;
        bytes[offset + 1] = (byte)(bits >> 8);
    }

    private static ArcExecutionLane CreateLane()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        return new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true,
            Qwen35TrainingKernels = true,
            BufferPoolBytes = 4 * 1024 * 1024
        });
    }

    private static float[] Values(int length, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length).Select(_ => (float)(random.NextDouble() * .4 - .2)).ToArray();
    }

    private static float[] Read(ArcExecutionLane lane, ArcBuffer buffer, int count)
    {
        var result = new float[count];
        lane.Read(buffer, result);
        return result;
    }

    private static void Close(double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i) Close(expected[i], actual[i], 2e-6 + Math.Abs(expected[i]) * 3e-6);
    }

    private static void Close(double expected, float actual, double tolerance)
        => Assert.True(float.IsFinite(actual) && Math.Abs(expected - actual) <= tolerance,
            $"Expected {expected:R}, actual {actual:R}, tolerance {tolerance:R}.");
}
