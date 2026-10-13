using System.Buffers.Binary;
using System.Text;
using NNtrain;
using Xunit;

public sealed class SafeTensorWriterCompatibilityTests
{
    [Theory]
    [InlineData(TensorDType.Float32)]
    [InlineData(TensorDType.Float16)]
    [InlineData(TensorDType.BFloat16)]
    public void StateWriterPreservesWireFormatAcrossChunkBoundary(
        TensorDType dtype)
    {
        // Fixed wire encodings keep this independent of the shared writer
        // and exercise signed zero and both directions of BF16 tie rounding.
        uint[] float32Bits =
            [0x00000000, 0x80000000, 0x3f800000, 0xbf800000,
             0x3fc00000, 0x3f808000, 0x3f818000];
        ushort[] float16Bits =
            [0x0000, 0x8000, 0x3c00, 0xbc00, 0x3e00, 0x3c04, 0x3c0c];
        ushort[] bfloat16Bits =
            [0x0000, 0x8000, 0x3f80, 0xbf80, 0x3fc0, 0x3f80, 0x3f82];
        const int Count = 4099;
        float[] values = Enumerable.Range(0, Count)
            .Select(index => BitConverter.UInt32BitsToSingle(
                float32Bits[index % float32Bits.Length]))
            .ToArray();
        ModuleState state = new(
            ModuleState.CurrentFormatVersion,
            [
                new ModuleParameterState(1, "last", [1], [1f], dtype),
                new ModuleParameterState(0, "first", [Count], values, dtype),
            ]);
        string path = Path.Combine(
            Path.GetTempPath(),
            $"nntrain-writer-compatibility-{Guid.NewGuid():N}.safetensors");
        try
        {
            safetensors.torch.save_file(state, path);
            byte[] file = File.ReadAllBytes(path);
            int elementSize = dtype == TensorDType.Float32 ? 4 : 2;
            string descriptor = dtype switch
            {
                TensorDType.Float32 => "F32",
                TensorDType.Float16 => "F16",
                _ => "BF16",
            };
            string format = dtype == TensorDType.Float32
                ? "nntrain.module_state.f32.v1"
                : "nntrain.module_state.mixed.v1";
            int firstEnd = Count * elementSize;
            int dataLength = firstEnd + elementSize;
            string json = "{\"__metadata__\":{\"format\":\"" + format +
                "\",\"module_state_format_version\":\"" +
                ModuleState.CurrentFormatVersion + "\"}," +
                "\"00000000:first\":{\"dtype\":\"" + descriptor +
                "\",\"shape\":[4099],\"data_offsets\":[0," + firstEnd + "]}," +
                "\"00000001:last\":{\"dtype\":\"" + descriptor +
                "\",\"shape\":[1],\"data_offsets\":[" + firstEnd + "," +
                dataLength + "]}}";
            byte[] expectedJson = Encoding.UTF8.GetBytes(json);
            int headerLength = (expectedJson.Length + 7) & ~7;
            int dataStart = sizeof(long) + headerLength;
            Assert.Equal(
                (ulong)headerLength,
                BinaryPrimitives.ReadUInt64LittleEndian(file));
            Assert.Equal(
                expectedJson,
                file.AsSpan(sizeof(long), expectedJson.Length).ToArray());
            Assert.All(
                file.AsSpan(sizeof(long) + expectedJson.Length,
                    headerLength - expectedJson.Length).ToArray(),
                value => Assert.Equal((byte)' ', value));
            Assert.Equal(dataStart + dataLength, file.Length);

            for (int index = 0; index <= Count; index++)
            {
                int patternIndex = index == Count
                    ? 2
                    : index % float32Bits.Length;
                ReadOnlySpan<byte> scalar = file.AsSpan(
                    dataStart + index * elementSize, elementSize);
                if (dtype == TensorDType.Float32)
                {
                    Assert.Equal(float32Bits[patternIndex],
                        BinaryPrimitives.ReadUInt32LittleEndian(scalar));
                }
                else
                {
                    ushort[] expectedBits = dtype == TensorDType.Float16
                        ? float16Bits
                        : bfloat16Bits;
                    Assert.Equal(expectedBits[patternIndex],
                        BinaryPrimitives.ReadUInt16LittleEndian(scalar));
                }
            }
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Fact]
    public void MixedStateKeepsEmptyTensorOffsetsAndScalarShape()
    {
        ModuleState state = new(
            ModuleState.CurrentFormatVersion,
            [
                new ModuleParameterState(2, "scalar", [], [-1f], TensorDType.Float16),
                new ModuleParameterState(0, "empty", [0, 3], [], TensorDType.BFloat16),
                new ModuleParameterState(1, "full", [1], [1f], TensorDType.Float32),
            ]);
        string path = Path.Combine(
            Path.GetTempPath(),
            $"nntrain-writer-mixed-{Guid.NewGuid():N}.safetensors");
        try
        {
            safetensors.torch.save_file(state, path);
            byte[] file = File.ReadAllBytes(path);
            const string ExpectedHeader =
                "{\"__metadata__\":{\"format\":\"nntrain.module_state.mixed.v1\"," +
                "\"module_state_format_version\":\"1\"}," +
                "\"00000000:empty\":{\"dtype\":\"BF16\",\"shape\":[0,3],\"data_offsets\":[0,0]}," +
                "\"00000001:full\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}," +
                "\"00000002:scalar\":{\"dtype\":\"F16\",\"shape\":[],\"data_offsets\":[4,6]}}";
            int headerLength = (ExpectedHeader.Length + 7) & ~7;
            Assert.Equal(
                (ulong)headerLength,
                BinaryPrimitives.ReadUInt64LittleEndian(file));
            Assert.Equal(
                ExpectedHeader.PadRight(headerLength, ' '),
                Encoding.UTF8.GetString(file, sizeof(long), headerLength));
            Assert.Equal(
                new byte[] { 0x00, 0x00, 0x80, 0x3f, 0x00, 0xbc },
                file.AsSpan(sizeof(long) + headerLength).ToArray());
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }
}
