using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;
using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

/// <summary>
/// Expected values come from official gguf-py and NumPy, never the project's
/// decoder. The checked-in fixture also exercises the official writer's layout.
/// </summary>
public sealed class QwenGgufIndependentReferenceTests
{
    private static string FixturePath(string name) => Path.Combine(
        AppContext.BaseDirectory, "Fixtures", "QwenGgufReference", name);

    [Fact]
    public void OfficialGgufPyFixtureDecodesBitExactlyIncludingHalfSubnormals()
    {
        using JsonDocument document = ReadFixture();
        using var reader = new GgufReader(FixturePath("reference.gguf"));
        Assert.Equal(3u, reader.Version);
        Assert.Equal(6, reader.Tensors.Count);

        foreach (JsonElement reference in document.RootElement.GetProperty("tensors").EnumerateArray())
        {
            string name = reference.GetProperty("name").GetString()!;
            GgufTensorInfo tensor = reader.GetTensor(name);
            Assert.Equal(reference.GetProperty("type").GetUInt32(), tensor.Type);
            Assert.Equal(reference.GetProperty("shape").EnumerateArray()
                .Select(value => value.GetUInt64()), tensor.Shape);
            float[] expected = ReadFloat32(reference, "decodedFloat32Base64");
            float[] actual = Qwen2Gguf.ReadTensor(reader, tensor);
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(BitConverter.SingleToInt32Bits(expected[i])
                    == BitConverter.SingleToInt32Bits(actual[i]),
                    $"gguf-py mismatch in {name}[{i}]: expected {expected[i]:R}, actual {actual[i]:R}.");
            }
        }
    }

    [Theory]
    [InlineData(false, false, false)]
    [InlineData(true, false, false)]
    [InlineData(true, true, false)]
    [InlineData(true, true, true)]
    public void ArcQuantizedLinearMatchesIndependentFloat64Reference(
        bool fast, bool subgroup, bool subgroupPrefill)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(0), "Intel Arc GPU is required.");
        using JsonDocument document = ReadFixture();
        JsonElement root = document.RootElement;
        int inputWidth = root.GetProperty("inputWidth").GetInt32();
        int outputWidth = root.GetProperty("outputWidth").GetInt32();
        float[] allInput = ReadFloat32(root, "inputFloat32Base64");
        using var reader = new GgufReader(FixturePath("reference.gguf"));
        using var execution = Tensor.BeginArcExecution(0, TensorPrecisionMode.Float32,
            new ArcExecutionOptions
            {
                QwenQuantizedLinearFast = fast,
                QwenQuantizedLinearSubgroup = subgroup,
                QwenQuantizedSubgroupPrefill = subgroupPrefill,
            });
        using var noGrad = AutogradContext.NoGrad();

        foreach (JsonElement reference in root.GetProperty("tensors").EnumerateArray())
        {
            uint type = reference.GetProperty("type").GetUInt32();
            if (type is not (Qwen2Gguf.Q4KType or Qwen2Gguf.Q6KType)) continue;
            string name = reference.GetProperty("name").GetString()!;
            GgufTensorInfo tensor = reader.GetTensor(name);
            byte[] payload = reader.ReadTensorBytes(tensor, reference.GetProperty("payloadBytes").GetInt32());
            double[] expected = ReadFloat64(reference, "linearFloat64Base64");
            float[] biasValues = ReadFloat32(reference, "biasFloat32Base64");
            double maximum = expected.Select(Math.Abs).Max();
            // Accommodates FP32 accumulation/reduction order. The subnormal-only
            // matrices use zero bias, so an underflow-to-zero regression fails.
            double tolerance = maximum * 5e-6 + 1e-9;
            using var matrix = new ArcQuantizedMatrix(payload, type, outputWidth, inputWidth);
            foreach (int rows in new[] { 1, 3, 5 })
            {
                using var frame = Tensor.BeginArcInferenceFrame();
                var input = new Tensor(allInput.AsSpan(0, rows * inputWidth).ToArray(), [rows, inputWidth]);
                var bias = new Tensor(biasValues, [outputWidth]);
                float[] actual = matrix.Forward(input, bias).Data.ToArray();
                Assert.Equal(rows * outputWidth, actual.Length);
                for (int i = 0; i < actual.Length; i++)
                {
                    Assert.True(float.IsFinite(actual[i]));
                    Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                        $"Independent linear mismatch in {name}, rows={rows}, index={i}, "
                        + $"fast={fast}, subgroup={subgroup}: expected {expected[i]:R}, "
                        + $"actual {actual[i]:R}, tolerance {tolerance:R}.");
                }
            }
        }
        Tensor.ArcLane.Synchronize();
        Tensor.ArcLane.CheckNumericStatus();
    }

    private static JsonDocument ReadFixture()
    {
        JsonDocument document = JsonDocument.Parse(File.ReadAllText(FixturePath("expected.json")));
        string hash = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(FixturePath("reference.gguf"))));
        Assert.Equal(document.RootElement.GetProperty("ggufSha256").GetString(), hash);
        return document;
    }

    private static float[] ReadFloat32(JsonElement value, string property)
    {
        byte[] bytes = Convert.FromBase64String(value.GetProperty(property).GetString()!);
        var result = new float[bytes.Length / 4];
        for (int i = 0; i < result.Length; i++)
            result[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(i * 4, 4)));
        return result;
    }

    private static double[] ReadFloat64(JsonElement value, string property)
    {
        byte[] bytes = Convert.FromBase64String(value.GetProperty(property).GetString()!);
        var result = new double[bytes.Length / 8];
        for (int i = 0; i < result.Length; i++)
            result[i] = BitConverter.Int64BitsToDouble(BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(i * 8, 8)));
        return result;
    }
}
