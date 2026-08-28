namespace NNtrain;

partial class Tensor
{
    private Tensor BroadcastAddCuda(Tensor addend, int repeatLength)
    {
        if (DType != addend.DType)
        {
            throw new InvalidOperationException(
                "CUDA indexed broadcast addition requires matching physical " +
                "storage types. Convert both operands explicitly.");
        }
        int deviceIndex = CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        nint stream = accelerator.DefaultStream;
        Tensor result;
        if (DType == TensorDType.Bfp8)
        {
            Bfp8QuantizationDescriptor descriptor =
                SelectBfp8ResultDescriptor(this, addend);
            using CudaBfp8BFloat16Lease inputDecoded =
                AcquireCudaBfp8BFloat16Buffer(deviceIndex);
            using CudaBfp8BFloat16Lease addendDecoded =
                addend.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
            NativeCudaBuffer<ushort> encodedOutput =
                RentCudaBFloat16Buffer(deviceIndex, Numel);
            using CudaBfp8OwnedBuffers output = CudaBfp8OwnedBuffers.Allocate(
                accelerator, Numel, descriptor);
            try
            {
                CudaPublicOpsNative.BroadcastAdd(
                    deviceIndex,
                    inputDecoded.Buffer.NativePtr,
                    addendDecoded.Buffer.NativePtr,
                    encodedOutput.NativePtr,
                    Numel,
                    repeatLength,
                    bfloat16: true,
                    stream);
                CudaBfp8Native.QuantizeBFloat16(
                    deviceIndex,
                    encodedOutput,
                    output.Payload,
                    output.Scales,
                    descriptor,
                    stream);
                result = FromCudaBfp8Result(
                    output, deviceIndex, _shape, [this, addend]);
            }
            finally
            {
                ReturnCudaBFloat16Buffer(accelerator, encodedOutput);
            }
        }
        else if (DType == TensorDType.BFloat16)
        {
            NativeCudaBuffer<ushort> output =
                RentCudaBFloat16Buffer(deviceIndex, Numel);
            try
            {
                CudaPublicOpsNative.BroadcastAdd(
                    deviceIndex,
                    EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                    addend.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                    output.NativePtr,
                    Numel,
                    repeatLength,
                    bfloat16: true,
                    stream);
                result = FromCudaResult(
                    output,
                    deviceIndex,
                    _shape,
                    [this, addend],
                    TensorDType.BFloat16);
            }
            catch
            {
                ReturnCudaBFloat16Buffer(accelerator, output);
                throw;
            }
        }
        else if (DType == TensorDType.Float32)
        {
            NativeCudaBuffer<float> output =
                RentCudaFloatBuffer(deviceIndex, Numel);
            try
            {
                CudaPublicOpsNative.BroadcastAdd(
                    deviceIndex,
                    EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                    addend.EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                    output.NativePtr,
                    Numel,
                    repeatLength,
                    bfloat16: false,
                    stream);
                result = FromCudaResult(
                    output,
                    deviceIndex,
                    _shape,
                    [this, addend],
                    TensorDType.Float32);
            }
            catch
            {
                ReturnCudaFloatBuffer(accelerator, output);
                throw;
            }
        }
        else
        {
            ThrowIfCudaHostFallback("Indexed broadcast addition");
            throw new InvalidOperationException(
                "CUDA indexed broadcast addition did not select a dtype.");
        }

        if (AutogradContext.IsRecordingEnabled)
        {
            result.Node.BackwardAction = () =>
            {
                if (DType == TensorDType.BFloat16
                    && TensorExecutionContext.UsesBFloat16GradientStorage
                    && TryBroadcastAddBackwardBFloat16Direct(
                        result,
                        addend,
                        repeatLength,
                        deviceIndex,
                        accelerator))
                {
                    return;
                }
                CudaPublicOpsNative.BroadcastAddBackward(
                    deviceIndex,
                    result.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    addend.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    Numel,
                    repeatLength,
                    accelerator.DefaultStream);
                MarkCudaGradientMutated(deviceIndex);
                addend.MarkCudaGradientMutated(deviceIndex);
            };
        }
        return result;
    }

    private bool TryBroadcastAddBackwardBFloat16Direct(
        Tensor output,
        Tensor addend,
        int repeatLength,
        int deviceIndex,
        NativeCudaDevice accelerator)
    {
        bool inputBorrowed = TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? inputExisting);
        bool addendBorrowed = addend.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? addendExisting);
        if ((!inputBorrowed && HasGradientBuffer)
            || (!addendBorrowed && addend.HasGradientBuffer))
        {
            return false;
        }
        bool outputBorrowed = output.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? outputExisting);
        NativeCudaBuffer<ushort>? rentedOutput = null;
        NativeCudaBuffer<ushort>? rentedInput = null;
        NativeCudaBuffer<ushort>? rentedAddend = null;
        NativeCudaBuffer<ushort> outputGradient = outputBorrowed
            ? outputExisting!
            : rentedOutput = RentCudaBFloat16Buffer(deviceIndex, output.Numel);
        NativeCudaBuffer<ushort> inputGradient = inputBorrowed
            ? inputExisting!
            : rentedInput = RentCudaBFloat16Buffer(deviceIndex, Numel);
        NativeCudaBuffer<ushort> addendGradient = addendBorrowed
            ? addendExisting!
            : rentedAddend = RentCudaBFloat16Buffer(deviceIndex, addend.Numel);
        try
        {
            if (!outputBorrowed)
            {
                CudaTensorNative.EncodeBFloat16(
                    deviceIndex,
                    output.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    outputGradient.NativePtr,
                    output.Numel);
            }
            if (!inputBorrowed)
                inputGradient.MemSetToZero();
            if (!addendBorrowed)
                addendGradient.MemSetToZero();
            CudaPublicOpsNative.BroadcastAddBackwardBFloat16Gradient(
                deviceIndex,
                outputGradient.NativePtr,
                inputGradient.NativePtr,
                addendGradient.NativePtr,
                Numel,
                repeatLength,
                accelerator.DefaultStream);
            if (!inputBorrowed)
            {
                AdoptCudaBFloat16GradientBuffer(inputGradient, deviceIndex);
                rentedInput = null;
            }
            if (!addendBorrowed)
            {
                addend.AdoptCudaBFloat16GradientBuffer(
                    addendGradient,
                    deviceIndex);
                rentedAddend = null;
            }
            return true;
        }
        finally
        {
            if (rentedOutput is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedOutput);
            if (rentedInput is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedInput);
            if (rentedAddend is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedAddend);
        }
    }

    private Tensor CausalMaskCuda(
        int rows,
        int columns,
        float fillValue)
    {
        int deviceIndex = CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        nint stream = accelerator.DefaultStream;
        Tensor result;
        if (DType == TensorDType.Bfp8)
        {
            Bfp8QuantizationDescriptor descriptor =
                SelectBfp8ResultDescriptor(this);
            using CudaBfp8BFloat16Lease decoded =
                AcquireCudaBfp8BFloat16Buffer(deviceIndex);
            NativeCudaBuffer<ushort> encodedOutput =
                RentCudaBFloat16Buffer(deviceIndex, Numel);
            using CudaBfp8OwnedBuffers output = CudaBfp8OwnedBuffers.Allocate(
                accelerator, Numel, descriptor);
            try
            {
                CudaPublicOpsNative.CausalMask(
                    deviceIndex,
                    decoded.Buffer.NativePtr,
                    encodedOutput.NativePtr,
                    Numel,
                    rows,
                    columns,
                    fillValue,
                    bfloat16: true,
                    stream);
                CudaBfp8Native.QuantizeBFloat16(
                    deviceIndex,
                    encodedOutput,
                    output.Payload,
                    output.Scales,
                    descriptor,
                    stream);
                result = FromCudaBfp8Result(
                    output, deviceIndex, _shape, [this]);
            }
            finally
            {
                ReturnCudaBFloat16Buffer(accelerator, encodedOutput);
            }
        }
        else if (DType == TensorDType.BFloat16)
        {
            NativeCudaBuffer<ushort> output =
                RentCudaBFloat16Buffer(deviceIndex, Numel);
            try
            {
                CudaPublicOpsNative.CausalMask(
                    deviceIndex,
                    EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                    output.NativePtr,
                    Numel,
                    rows,
                    columns,
                    fillValue,
                    bfloat16: true,
                    stream);
                result = FromCudaResult(
                    output, deviceIndex, _shape, [this], TensorDType.BFloat16);
            }
            catch
            {
                ReturnCudaBFloat16Buffer(accelerator, output);
                throw;
            }
        }
        else if (DType == TensorDType.Float32)
        {
            NativeCudaBuffer<float> output =
                RentCudaFloatBuffer(deviceIndex, Numel);
            try
            {
                CudaPublicOpsNative.CausalMask(
                    deviceIndex,
                    EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                    output.NativePtr,
                    Numel,
                    rows,
                    columns,
                    fillValue,
                    bfloat16: false,
                    stream);
                result = FromCudaResult(
                    output, deviceIndex, _shape, [this], TensorDType.Float32);
            }
            catch
            {
                ReturnCudaFloatBuffer(accelerator, output);
                throw;
            }
        }
        else
        {
            ThrowIfCudaHostFallback(nameof(CausalMask));
            throw new InvalidOperationException(
                "CUDA causal mask did not select a supported dtype.");
        }

        if (AutogradContext.IsRecordingEnabled)
        {
            result.Node.BackwardAction = () =>
            {
                if (DType == TensorDType.BFloat16
                    && TensorExecutionContext.UsesBFloat16GradientStorage
                    && TryCausalMaskBackwardBFloat16Direct(
                        result,
                        rows,
                        columns,
                        deviceIndex,
                        accelerator))
                {
                    return;
                }
                CudaPublicOpsNative.CausalMaskBackward(
                    deviceIndex,
                    result.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    Numel,
                    rows,
                    columns,
                    accelerator.DefaultStream);
                MarkCudaGradientMutated(deviceIndex);
            };
        }
        return result;
    }

    private bool TryCausalMaskBackwardBFloat16Direct(
        Tensor output,
        int rows,
        int columns,
        int deviceIndex,
        NativeCudaDevice accelerator)
    {
        bool inputBorrowed = TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? inputExisting);
        if (!inputBorrowed && HasGradientBuffer)
            return false;
        bool outputBorrowed = output.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? outputExisting);
        NativeCudaBuffer<ushort>? rentedOutput = null;
        NativeCudaBuffer<ushort>? rentedInput = null;
        NativeCudaBuffer<ushort> outputGradient = outputBorrowed
            ? outputExisting!
            : rentedOutput = RentCudaBFloat16Buffer(deviceIndex, output.Numel);
        NativeCudaBuffer<ushort> inputGradient = inputBorrowed
            ? inputExisting!
            : rentedInput = RentCudaBFloat16Buffer(deviceIndex, Numel);
        try
        {
            if (!outputBorrowed)
            {
                CudaTensorNative.EncodeBFloat16(
                    deviceIndex,
                    output.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    outputGradient.NativePtr,
                    output.Numel);
            }
            if (!inputBorrowed)
                inputGradient.MemSetToZero();
            CudaPublicOpsNative.CausalMaskBackwardBFloat16Gradient(
                deviceIndex,
                outputGradient.NativePtr,
                inputGradient.NativePtr,
                Numel,
                rows,
                columns,
                accelerator.DefaultStream);
            if (!inputBorrowed)
            {
                AdoptCudaBFloat16GradientBuffer(inputGradient, deviceIndex);
                rentedInput = null;
            }
            return true;
        }
        finally
        {
            if (rentedOutput is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedOutput);
            if (rentedInput is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedInput);
        }
    }

    private Tensor SoftmaxCuda(bool logSoftmax)
    {
        int columns = _shape[^1];
        int rows = Numel / columns;
        int deviceIndex = CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        nint stream = accelerator.DefaultStream;
        CudaSoftmaxContext? context = AutogradContext.IsRecordingEnabled
            ? new CudaSoftmaxContext(
                accelerator,
                RentCudaFloatBuffer(deviceIndex, Numel))
            : null;
        Tensor result;
        try
        {
            if (DType == TensorDType.Bfp8)
            {
                Bfp8QuantizationDescriptor descriptor =
                    SelectBfp8ResultDescriptor(this);
                using CudaBfp8BFloat16Lease decoded =
                    AcquireCudaBfp8BFloat16Buffer(deviceIndex);
                NativeCudaBuffer<ushort> encodedOutput =
                    RentCudaBFloat16Buffer(deviceIndex, Numel);
                using CudaBfp8OwnedBuffers output = CudaBfp8OwnedBuffers.Allocate(
                    accelerator, Numel, descriptor);
                try
                {
                    Launch(decoded.Buffer.NativePtr, encodedOutput.NativePtr, true);
                    CudaBfp8Native.QuantizeBFloat16(
                        deviceIndex,
                        encodedOutput,
                        output.Payload,
                        output.Scales,
                        descriptor,
                        stream);
                    result = FromCudaBfp8Result(
                        output, deviceIndex, _shape, [this]);
                }
                finally
                {
                    ReturnCudaBFloat16Buffer(accelerator, encodedOutput);
                }
            }
            else if (DType == TensorDType.BFloat16)
            {
                NativeCudaBuffer<ushort> output =
                    RentCudaBFloat16Buffer(deviceIndex, Numel);
                try
                {
                    Launch(
                        EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                        output.NativePtr,
                        true);
                    result = FromCudaResult(
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
            else if (DType == TensorDType.Float32)
            {
                NativeCudaBuffer<float> output =
                    RentCudaFloatBuffer(deviceIndex, Numel);
                try
                {
                    Launch(
                        EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                        output.NativePtr,
                        false);
                    result = FromCudaResult(
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
            else
            {
                ThrowIfCudaHostFallback(
                    logSoftmax ? nameof(LogSoftmaxLastDim) : nameof(SoftmaxLastDim));
                throw new InvalidOperationException(
                    "CUDA softmax did not select a supported dtype.");
            }
        }
        catch
        {
            context?.Dispose();
            throw;
        }

        if (context is not null)
        {
            AutogradLease<CudaSoftmaxContext> lease =
                AutogradLease<CudaSoftmaxContext>.Own(
                    context,
                    AutogradLeaseMetadata.CudaOwned(
                        deviceIndex,
                        DType,
                        DataVersion),
                    static saved => saved.Dispose());
            result.Node.SetBackward(lease, savedContext =>
            {
                if (DType == TensorDType.BFloat16
                    && TensorExecutionContext.UsesBFloat16GradientStorage
                    && TrySoftmaxBackwardBFloat16Direct(
                        result,
                        savedContext,
                        rows,
                        columns,
                        logSoftmax,
                        deviceIndex,
                        accelerator))
                {
                    return;
                }
                CudaPublicOpsNative.SoftmaxBackward(
                    deviceIndex,
                    savedContext.Probabilities.NativePtr,
                    result.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    rows,
                    columns,
                    logSoftmax,
                    accelerator.DefaultStream);
                MarkCudaGradientMutated(deviceIndex);
            });
        }
        return result;

        void Launch(nint input, nint output, bool bfloat16)
        {
            CudaPublicOpsNative.Softmax(
                deviceIndex,
                input,
                output,
                context?.Probabilities.NativePtr ?? nint.Zero,
                rows,
                columns,
                logSoftmax,
                bfloat16,
                stream);
        }
    }

    private bool TrySoftmaxBackwardBFloat16Direct(
        Tensor output,
        CudaSoftmaxContext context,
        int rows,
        int columns,
        bool logSoftmax,
        int deviceIndex,
        NativeCudaDevice accelerator)
    {
        bool inputBorrowed = TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? inputExisting);
        if (!inputBorrowed && HasGradientBuffer)
            return false;
        bool outputBorrowed = output.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? outputExisting);
        NativeCudaBuffer<ushort>? rentedOutput = null;
        NativeCudaBuffer<ushort>? rentedInput = null;
        NativeCudaBuffer<ushort> outputGradient = outputBorrowed
            ? outputExisting!
            : rentedOutput = RentCudaBFloat16Buffer(deviceIndex, output.Numel);
        NativeCudaBuffer<ushort> inputGradient = inputBorrowed
            ? inputExisting!
            : rentedInput = RentCudaBFloat16Buffer(deviceIndex, Numel);
        try
        {
            if (!outputBorrowed)
            {
                CudaTensorNative.EncodeBFloat16(
                    deviceIndex,
                    output.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    outputGradient.NativePtr,
                    output.Numel);
            }
            if (!inputBorrowed)
                inputGradient.MemSetToZero();
            CudaPublicOpsNative.SoftmaxBackwardBFloat16Gradient(
                deviceIndex,
                context.Probabilities.NativePtr,
                outputGradient.NativePtr,
                inputGradient.NativePtr,
                rows,
                columns,
                logSoftmax,
                accelerator.DefaultStream);
            if (!inputBorrowed)
            {
                AdoptCudaBFloat16GradientBuffer(inputGradient, deviceIndex);
                rentedInput = null;
            }
            return true;
        }
        finally
        {
            if (rentedOutput is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedOutput);
            if (rentedInput is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedInput);
        }
    }
}

internal sealed class CudaSoftmaxContext : IDisposable
{
    private readonly NativeCudaDevice _accelerator;
    private int _disposed;

    internal CudaSoftmaxContext(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> probabilities)
    {
        _accelerator = accelerator;
        Probabilities = probabilities;
    }

    internal NativeCudaBuffer<float> Probabilities { get; }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
            Tensor.ReturnCudaFloatBuffer(_accelerator, Probabilities);
    }
}
