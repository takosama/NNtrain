using NNtrain.Runtime.Execution;

namespace NNtrain;

partial class Tensor
{
    private Tensor ToCuda(TensorDType dtype)
    {
        int deviceIndex = CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        nint stream = accelerator.DefaultStream;
        Tensor result = (DType, dtype) switch
        {
            (TensorDType.Float32, TensorDType.BFloat16) =>
                ConvertCudaFloat32ToBFloat16(
                    deviceIndex, accelerator),
            (TensorDType.BFloat16, TensorDType.Float32) =>
                ConvertCudaBFloat16ToFloat32(
                    deviceIndex, accelerator),
            (TensorDType.Float32, TensorDType.Bfp8) =>
                ConvertCudaFloat32ToBfp8(
                    deviceIndex, accelerator, stream),
            (TensorDType.BFloat16, TensorDType.Bfp8) =>
                ConvertCudaBFloat16ToBfp8(
                    deviceIndex, accelerator, stream),
            (TensorDType.Bfp8, TensorDType.Float32) =>
                ConvertCudaBfp8ToFloat32(
                    deviceIndex, accelerator, stream),
            (TensorDType.Bfp8, TensorDType.BFloat16) =>
                ConvertCudaBfp8ToBFloat16(
                    deviceIndex, accelerator, stream),
            _ => throw new InvalidOperationException(
                $"CUDA dtype conversion {DType} -> {dtype} is not implemented."),
        };

        if (AutogradContext.IsRecordingEnabled)
        {
            result.Node.BackwardAction = () =>
            {
                if (TryAccumulateBFloat16GradientRangeCuda(
                        result,
                        this,
                        sourceOffset: 0,
                        destinationOffset: 0,
                        length: Numel))
                {
                    return;
                }
                CudaTensorNative.Accumulate(
                    deviceIndex,
                    result.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    Numel);
                MarkCudaGradientMutated(deviceIndex);
            };
        }
        return result;
    }

    private Tensor ConvertCudaFloat32ToBFloat16(
        int deviceIndex,
        NativeCudaDevice accelerator)
    {
        NativeCudaBuffer<ushort> output =
            RentCudaBFloat16Buffer(deviceIndex, Numel);
        try
        {
            CudaTensorNative.EncodeBFloat16(
                deviceIndex,
                EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                output.NativePtr,
                Numel);
            return FromCudaResult(
                output,
                deviceIndex,
                _shape,
                [this],
                TensorDType.BFloat16);
        }
        catch
        {
            ReturnCudaBFloat16Buffer(accelerator, output);
            throw;
        }
    }

    private Tensor ConvertCudaBFloat16ToFloat32(
        int deviceIndex,
        NativeCudaDevice accelerator)
    {
        NativeCudaBuffer<float> output = RentCudaFloatBuffer(deviceIndex, Numel);
        try
        {
            CudaTensorNative.DecodeBFloat16(
                deviceIndex,
                EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                output.NativePtr,
                Numel);
            return FromCudaResult(
                output,
                deviceIndex,
                _shape,
                [this],
                TensorDType.Float32);
        }
        catch
        {
            ReturnCudaFloatBuffer(accelerator, output);
            throw;
        }
    }

    private Tensor ConvertCudaFloat32ToBfp8(
        int deviceIndex,
        NativeCudaDevice accelerator,
        nint stream)
    {
        Bfp8QuantizationDescriptor descriptor =
            SelectExplicitBfp8ConversionDescriptor();
        using CudaBfp8OwnedBuffers output = CudaBfp8OwnedBuffers.Allocate(
            accelerator, Numel, descriptor);
        CudaBfp8Native.QuantizeFloat32(
            deviceIndex,
            EnsureCudaFloat32Buffer(deviceIndex),
            output.Payload,
            output.Scales,
            descriptor,
            stream);
        return FromCudaBfp8Result(
            output, deviceIndex, _shape, [this]);
    }

    private Tensor ConvertCudaBFloat16ToBfp8(
        int deviceIndex,
        NativeCudaDevice accelerator,
        nint stream)
    {
        Bfp8QuantizationDescriptor descriptor =
            SelectExplicitBfp8ConversionDescriptor();
        using CudaBfp8OwnedBuffers output = CudaBfp8OwnedBuffers.Allocate(
            accelerator, Numel, descriptor);
        CudaBfp8Native.QuantizeBFloat16(
            deviceIndex,
            EnsureCudaBFloat16Buffer(deviceIndex),
            output.Payload,
            output.Scales,
            descriptor,
            stream);
        return FromCudaBfp8Result(
            output, deviceIndex, _shape, [this]);
    }

    private Tensor ConvertCudaBfp8ToFloat32(
        int deviceIndex,
        NativeCudaDevice accelerator,
        nint stream)
    {
        CudaBfp8BufferView source = EnsureCudaBfp8Buffer(deviceIndex);
        NativeCudaBuffer<float> output = RentCudaFloatBuffer(deviceIndex, Numel);
        try
        {
            CudaBfp8Native.DequantizeFloat32(
                deviceIndex,
                source.Payload,
                source.Scales,
                output,
                source.Descriptor,
                stream);
            return FromCudaResult(
                output,
                deviceIndex,
                _shape,
                [this],
                TensorDType.Float32);
        }
        catch
        {
            ReturnCudaFloatBuffer(accelerator, output);
            throw;
        }
    }

    private Tensor ConvertCudaBfp8ToBFloat16(
        int deviceIndex,
        NativeCudaDevice accelerator,
        nint stream)
    {
        CudaBfp8BufferView source = EnsureCudaBfp8Buffer(deviceIndex);
        NativeCudaBuffer<ushort> output =
            RentCudaBFloat16Buffer(deviceIndex, Numel);
        try
        {
            CudaBfp8Native.DequantizeBFloat16(
                deviceIndex,
                source.Payload,
                source.Scales,
                output,
                source.Descriptor,
                stream);
            return FromCudaResult(
                output,
                deviceIndex,
                _shape,
                [this],
                TensorDType.BFloat16);
        }
        catch
        {
            ReturnCudaBFloat16Buffer(accelerator, output);
            throw;
        }
    }

    private static Bfp8QuantizationDescriptor
        SelectExplicitBfp8ConversionDescriptor()
        => TensorExecutionContext.ActivePrecisionPolicy?.Mode
            == PrecisionMode.Mix8_32
            ? Bfp8QuantizationDescriptor.Mix8_32
            : Bfp8QuantizationDescriptor.TensorWide;
}
