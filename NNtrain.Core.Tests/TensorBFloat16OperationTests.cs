using NNtrain;
using Xunit;

public sealed class TensorBFloat16OperationTests
{
    [Fact]
    public void ElementwiseOperationsReturnBFloat16RoundedResults()
    {
        var left = new Tensor(
            [1.001f, -2.003f],
            [2],
            dtype: TensorDType.BFloat16);
        var right = new Tensor(
            [0.501f, 3.007f],
            [2],
            dtype: TensorDType.BFloat16);

        Tensor sum = left + right;
        Tensor product = left * right;

        Assert.Equal(TensorDType.BFloat16, sum.DType);
        Assert.Equal(TensorDType.BFloat16, product.DType);
        Assert.Equal(QuantizeBFloat16(QuantizeBFloat16(1.001f)
            + QuantizeBFloat16(0.501f)), sum.Data[0]);
        Assert.Equal(QuantizeBFloat16(QuantizeBFloat16(-2.003f)
            * QuantizeBFloat16(3.007f)), product.Data[1]);
    }

    [Fact]
    public void LinearLastDimReturnsBFloat16StorageForBFloat16Inputs()
    {
        var input = new Tensor(
            [1.001f, -2.003f],
            [1, 2],
            dtype: TensorDType.BFloat16);
        var weight = new Tensor(
            [0.501f, -0.250f],
            [1, 2],
            dtype: TensorDType.BFloat16);
        var bias = new Tensor(
            [0.125f],
            [1],
            dtype: TensorDType.BFloat16);

        Tensor output = InvokeLinearLastDim(input, weight, bias);

        Assert.Equal(TensorDType.BFloat16, output.DType);
        Assert.Equal(QuantizeBFloat16(output.Data[0]), output.Data[0]);
    }

    [Fact]
    public void ToCudaMarksTensorDeviceAndCanReturnToCpu()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int previousDeviceIndex = Tensor.CudaDeviceIndex;
        try
        {
            Tensor.CudaDeviceIndex = 0;
            var tensor = new Tensor(
                [1.25f, -0.5f],
                [2],
                dtype: TensorDType.BFloat16);

            tensor.To(TensorDevice.Cuda);
            Assert.Equal(TensorDevice.Cuda, tensor.Device);

            tensor.To(TensorDevice.Cpu);
            Assert.Equal(TensorDevice.Cpu, tensor.Device);
            Assert.Equal(1.25f, tensor.Data[0]);
            Assert.Equal(-0.5f, tensor.Data[1]);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndex = previousDeviceIndex;
        }
    }

    private static Tensor InvokeLinearLastDim(
        Tensor input,
        Tensor weight,
        Tensor bias)
    {
        System.Reflection.MethodInfo method = typeof(Tensor).GetMethod(
            "LinearLastDim",
            System.Reflection.BindingFlags.Instance |
            System.Reflection.BindingFlags.NonPublic)!;
        return (Tensor)method.Invoke(input, [weight, bias, false])!;
    }

    private static float QuantizeBFloat16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
        return BitConverter.UInt32BitsToSingle((rounded >> 16) << 16);
    }
}
