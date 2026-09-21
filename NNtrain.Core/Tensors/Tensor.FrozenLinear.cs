namespace NNtrain;

partial class Tensor
{
    // LoRA base branch: preserve dX, omit dW/db and their persistent buffers.
    internal Tensor LinearLastDimFrozen(Tensor weight, Tensor bias, bool applyRelu)
    {
        if (DType is not (TensorDType.Float32 or TensorDType.BFloat16 or TensorDType.Bfp8))
            throw new NotSupportedException("Frozen LoRA linear supports float32/mix16_32/mix8_32.");
        return LinearLastDimCore(weight, bias, applyRelu, false, false, parameterGradients: false);
    }

    private void BackwardFrozenLinear(Tensor result, Tensor weight, bool applyRelu, long version)
    {
        int inputWidth = weight._shape[1], outputWidth = weight._shape[0], rows = Numel / inputWidth;
        if (weight.DataVersion != version) throw new InvalidOperationException("Frozen linear weight changed before backward.");
        if (result.Device == TensorDevice.Cuda)
        {
            if (result.DType == TensorDType.Bfp8)
            {
                CudaBfp8Gemm.LinearBackwardInputFrozen(
                    this, weight, result, rows, inputWidth, outputWidth, applyRelu);
                return;
            }
            int device = CudaDeviceIndex;
            var accelerator = ForgetMemoryV2Cuda.GetAccelerator(device);
            if (result.DType == TensorDType.BFloat16)
            {
                var gradient = RentCudaBFloat16Buffer(device, result.Numel);
                try
                {
                    CudaTensorNative.LinearEncodeBFloat16(device,
                        result.EnsureCudaGradientBuffer(device).NativePtr,
                        applyRelu ? result.EnsureCudaBFloat16Buffer(device).NativePtr : nint.Zero,
                        gradient.NativePtr, result.Numel, applyRelu);
                    CudaBlas.LinearBackwardInputBFloat16(accelerator, device, gradient,
                        weight.EnsureCudaBFloat16Buffer(device), EnsureCudaGradientBuffer(device),
                        rows, inputWidth, outputWidth);
                }
                finally { ReturnCudaBFloat16Buffer(accelerator, gradient); }
            }
            else
            {
                var gradient = result.EnsureCudaGradientBuffer(device);
                if (applyRelu)
                    CudaTensorNative.LinearMask(device, result.EnsureCudaFloat32Buffer(device).NativePtr,
                        gradient.NativePtr, result.Numel, true);
                CudaBlas.LinearBackwardInput(accelerator, device, gradient,
                    weight.EnsureCudaFloat32Buffer(device), EnsureCudaGradientBuffer(device),
                    rows, inputWidth, outputWidth, false);
            }
            MarkCudaGradientMutated(device);
            return;
        }
        EnsureGradientBuffer();
        RunBatches(rows, (long)inputWidth * outputWidth, row =>
        {
            for (int column = 0; column < outputWidth; column++)
            {
                int index = row * outputWidth + column;
                if (applyRelu && result._data[index] <= 0) continue;
                float gradient = result._grad[index];
                if (DType == TensorDType.BFloat16) gradient = TensorStorageCodec.RoundToBFloat16(gradient);
                AddScaledValues(_grad, row * inputWidth, weight._data, column * inputWidth, gradient, inputWidth);
            }
        });
    }
}
