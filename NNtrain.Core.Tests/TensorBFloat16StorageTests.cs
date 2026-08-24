using NNtrain;
using Xunit;

public sealed class TensorBFloat16StorageTests
{
    [Fact]
    public void StorageUsesTwoBytesAndRoundTripsBFloat16()
    {
        var tensor = new Tensor(
            [1f, -2.5f, 1.00390625f, float.PositiveInfinity],
            [4],
            dtype: TensorDType.BFloat16);

        Assert.Equal(TensorDType.BFloat16, tensor.DType);
        Assert.Equal(8, GetStorageByteLength(tensor));
        Assert.Equal(1f, tensor.Data[0]);
        Assert.Equal(-2.5f, tensor.Data[1]);
        Assert.Equal(1f, tensor.Data[2]);
        Assert.Equal(float.PositiveInfinity, tensor.Data[3]);
    }

    [Fact]
    public void SafeTensorsUsesStandardBFloat16Payload()
    {
        float[] values = [1.00390625f, -2.5f, 3.1415927f];
        ModuleState state = new(
            ModuleState.CurrentFormatVersion,
            [
                new ModuleParameterState(
                    0,
                    "bf16",
                    [values.Length],
                    values,
                    TensorDType.BFloat16),
            ]);
        string path = Path.Combine(
            Path.GetTempPath(),
            $"nntrain-bf16-{Guid.NewGuid():N}.safetensors");

        try
        {
            safetensors.torch.save_file(state, path);
            string header = System.Text.Encoding.UTF8.GetString(
                File.ReadAllBytes(path));
            Assert.Contains("\"dtype\":\"BF16\"", header);

            ModuleState restored = safetensors.torch.load_file(path);
            ModuleParameterState parameter = Assert.Single(restored.Parameters);
            Assert.Equal(TensorDType.BFloat16, parameter.DType);
            Assert.Equal(
                values.Select(QuantizeBFloat16),
                parameter.Values);
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    private static float QuantizeBFloat16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
        return BitConverter.UInt32BitsToSingle((rounded >> 16) << 16);
    }

    private static int GetStorageByteLength(Tensor tensor)
    {
        System.Reflection.PropertyInfo property = typeof(Tensor).GetProperty(
            "StorageByteLength",
            System.Reflection.BindingFlags.Instance |
            System.Reflection.BindingFlags.NonPublic)!;
        return (int)property.GetValue(tensor)!;
    }
}
