using System.Buffers.Binary;
using System.Text;
using NNtrain;
using Xunit;

public sealed class SafeTensorFloat16Tests
{
    [Fact]
    public void Float16DescriptorOffsetsPayloadAndBoundariesAreStandard()
    {
        float[] values =
        [
            (float)BitConverter.UInt16BitsToHalf(0x0000),
            (float)BitConverter.UInt16BitsToHalf(0x8000),
            (float)BitConverter.UInt16BitsToHalf(0x0001),
            (float)BitConverter.UInt16BitsToHalf(0x03ff),
            (float)BitConverter.UInt16BitsToHalf(0x0400),
            (float)BitConverter.UInt16BitsToHalf(0x7bff),
            (float)BitConverter.UInt16BitsToHalf(0xfbff),
        ];
        ModuleState state = new(
            ModuleState.CurrentFormatVersion,
            [
                new ModuleParameterState(
                    0,
                    "boundary",
                    [values.Length],
                    values,
                    TensorDType.Float16),
            ]);
        string path = TemporaryPath("f16-format");

        try
        {
            safetensors.torch.save_file(state, path);
            byte[] file = File.ReadAllBytes(path);
            (int headerLength, string header, int dataStart) =
                ReadHeader(file);

            Assert.Equal(0, headerLength % 8);
            Assert.Contains("\"dtype\":\"F16\"", header);
            Assert.Contains($"\"shape\":[{values.Length}]", header);
            Assert.Contains(
                $"\"data_offsets\":[0,{values.Length * sizeof(ushort)}]",
                header);
            Assert.Equal(
                dataStart + values.Length * sizeof(ushort),
                file.Length);

            for (int index = 0; index < values.Length; index++)
            {
                ushort actualBits = BinaryPrimitives.ReadUInt16LittleEndian(
                    file.AsSpan(
                        dataStart + index * sizeof(ushort),
                        sizeof(ushort)));
                Assert.Equal(
                    BitConverter.HalfToUInt16Bits((Half)values[index]),
                    actualBits);
            }

            ModuleState restored = safetensors.torch.load_file(path);
            ModuleParameterState parameter = Assert.Single(restored.Parameters);
            Assert.Equal(TensorDType.Float16, parameter.DType);
            Assert.Equal(values.Length, parameter.Values.Length);
            for (int index = 0; index < values.Length; index++)
            {
                Assert.Equal(
                    BitConverter.SingleToUInt32Bits(values[index]),
                    BitConverter.SingleToUInt32Bits(parameter.Values[index]));
            }
        }
        finally
        {
            DeleteIfExists(path);
        }
    }

    [Fact]
    public void MixedFloat16AndFloat32ParametersHaveContiguousByteOffsets()
    {
        ModuleState state = new(
            ModuleState.CurrentFormatVersion,
            [
                new ModuleParameterState(
                    0,
                    "half.first",
                    [3],
                    [0.1f, -2f, 3.25f],
                    TensorDType.Float16),
                new ModuleParameterState(
                    1,
                    "single",
                    [2],
                    [MathF.PI, -123.5f],
                    TensorDType.Float32),
                new ModuleParameterState(
                    2,
                    "half.last",
                    [2],
                    [0.00006103515625f, -0f],
                    TensorDType.Float16),
            ]);
        string path = TemporaryPath("mixed-format");

        try
        {
            torch.save_safetensors(state, path);
            byte[] file = File.ReadAllBytes(path);
            (_, string header, int dataStart) = ReadHeader(file);

            Assert.Contains("nntrain.module_state.mixed.v1", header);
            Assert.Contains("\"data_offsets\":[0,6]", header);
            Assert.Contains("\"data_offsets\":[6,14]", header);
            Assert.Contains("\"data_offsets\":[14,18]", header);
            Assert.Equal(dataStart + 18, file.Length);

            ModuleState restored = torch.load_safetensors(path);
            Assert.Equal(
                [
                    TensorDType.Float16,
                    TensorDType.Float32,
                    TensorDType.Float16,
                ],
                restored.Parameters.Select(parameter => parameter.DType));
            Assert.Equal(
                state.Parameters[0].Values.Select(value => (float)(Half)value),
                restored.Parameters[0].Values);
            Assert.Equal(
                state.Parameters[1].Values,
                restored.Parameters[1].Values);
            Assert.Equal(
                state.Parameters[2].Values.Select(value => (float)(Half)value),
                restored.Parameters[2].Values);
        }
        finally
        {
            DeleteIfExists(path);
        }
    }

    [Fact]
    public void LegacyFloat32StateKeepsOriginalDescriptorAndRoundTrips()
    {
        ModuleState state = new(
            ModuleState.CurrentFormatVersion,
            [new ModuleParameterState(0, "weight", [2], [1.5f, -2f])]);
        string path = TemporaryPath("legacy-f32");

        try
        {
            safetensors.torch.save_file(state, path);
            byte[] file = File.ReadAllBytes(path);
            (_, string header, int dataStart) = ReadHeader(file);

            Assert.Contains("nntrain.module_state.f32.v1", header);
            Assert.Contains("\"dtype\":\"F32\"", header);
            Assert.Contains("\"data_offsets\":[0,8]", header);
            Assert.Equal(dataStart + 8, file.Length);

            ModuleParameterState restored = Assert.Single(
                safetensors.torch.load_file(path).Parameters);
            Assert.Equal(TensorDType.Float32, restored.DType);
            Assert.Equal([1.5f, -2f], restored.Values);
        }
        finally
        {
            DeleteIfExists(path);
        }
    }

    private static (int HeaderLength, string Header, int DataStart)
        ReadHeader(byte[] file)
    {
        ulong encodedLength = BinaryPrimitives.ReadUInt64LittleEndian(
            file.AsSpan(0, sizeof(long)));
        int headerLength = checked((int)encodedLength);
        string header = Encoding.UTF8.GetString(
            file,
            sizeof(long),
            headerLength);
        return (
            headerLength,
            header,
            checked(sizeof(long) + headerLength));
    }

    private static string TemporaryPath(string prefix)
        => Path.Combine(
            Path.GetTempPath(),
            $"nntrain-{prefix}-{Guid.NewGuid():N}.safetensors");

    private static void DeleteIfExists(string path)
    {
        if (File.Exists(path))
            File.Delete(path);
    }
}
