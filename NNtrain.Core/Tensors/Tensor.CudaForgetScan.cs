namespace NNtrain;

partial class Tensor
{
    private Tensor ForgetScanCuda(
        int batch,
        int sequence,
        int width)
    {
        int deviceIndex = CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        nint stream = accelerator.DefaultStream;
        int outputLength = checked(batch * sequence * width);
        CudaForgetScanContext? context = AutogradContext.IsRecordingEnabled
            ? CudaForgetScanContext.Allocate(
                accelerator,
                deviceIndex,
                outputLength)
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
                    RentCudaBFloat16Buffer(deviceIndex, outputLength);
                using CudaBfp8OwnedBuffers output =
                    CudaBfp8OwnedBuffers.Allocate(
                        accelerator,
                        outputLength,
                        descriptor);
                try
                {
                    LaunchForgetScan(
                        decoded.Buffer.NativePtr,
                        encodedOutput.NativePtr,
                        context,
                        batch,
                        sequence,
                        width,
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
                        output,
                        deviceIndex,
                        [batch, sequence, width],
                        [this]);
                }
                finally
                {
                    ReturnCudaBFloat16Buffer(accelerator, encodedOutput);
                }
            }
            else if (DType == TensorDType.BFloat16)
            {
                NativeCudaBuffer<ushort> output =
                    RentCudaBFloat16Buffer(deviceIndex, outputLength);
                try
                {
                    LaunchForgetScan(
                        EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                        output.NativePtr,
                        context,
                        batch,
                        sequence,
                        width,
                        bfloat16: true,
                        stream);
                    result = FromCudaResult(
                        output,
                        deviceIndex,
                        [batch, sequence, width],
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
                    RentCudaFloatBuffer(deviceIndex, outputLength);
                try
                {
                    LaunchForgetScan(
                        EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                        output.NativePtr,
                        context,
                        batch,
                        sequence,
                        width,
                        bfloat16: false,
                        stream);
                    result = FromCudaResult(
                        output,
                        deviceIndex,
                        [batch, sequence, width],
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
                ThrowIfCudaHostFallback(nameof(FusedForgetScan));
                throw new InvalidOperationException(
                    "ForgetScan CUDA dispatch did not select a supported dtype.");
            }
        }
        catch
        {
            context?.Dispose();
            throw;
        }

        if (context is not null)
        {
            AutogradLease<CudaForgetScanContext> lease =
                AutogradLease<CudaForgetScanContext>.Own(
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
                    && TryForgetScanBackwardBFloat16Direct(
                        result,
                        savedContext,
                        batch,
                        sequence,
                        width,
                        deviceIndex,
                        accelerator))
                {
                    return;
                }
                CudaPublicOpsNative.ForgetScanBackward(
                    deviceIndex,
                    result.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    savedContext.Memory.NativePtr,
                    savedContext.Forget.NativePtr,
                    savedContext.Input.NativePtr,
                    savedContext.Value.NativePtr,
                    EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    batch,
                    sequence,
                    width,
                    accelerator.DefaultStream);
                MarkCudaGradientMutated(deviceIndex);
            });
        }
        return result;

        void LaunchForgetScan(
            nint projected,
            nint output,
            CudaForgetScanContext? saved,
            int batchSize,
            int sequenceLength,
            int modelWidth,
            bool bfloat16,
            nint computeStream)
        {
            CudaPublicOpsNative.ForgetScan(
                deviceIndex,
                projected,
                output,
                saved?.Memory.NativePtr ?? nint.Zero,
                saved?.Forget.NativePtr ?? nint.Zero,
                saved?.Input.NativePtr ?? nint.Zero,
                saved?.Value.NativePtr ?? nint.Zero,
                batchSize,
                sequenceLength,
                modelWidth,
                saved is not null,
                bfloat16,
                computeStream);
        }
    }

    private bool TryForgetScanBackwardBFloat16Direct(
        Tensor output,
        CudaForgetScanContext context,
        int batch,
        int sequence,
        int width,
        int deviceIndex,
        NativeCudaDevice accelerator)
    {
        bool projectedBorrowed = TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? projectedExisting);
        if (!projectedBorrowed && HasGradientBuffer)
            return false;
        bool outputBorrowed = output.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? outputExisting);
        NativeCudaBuffer<ushort>? rentedOutput = null;
        NativeCudaBuffer<ushort>? rentedProjected = null;
        NativeCudaBuffer<ushort> outputGradient = outputBorrowed
            ? outputExisting!
            : rentedOutput = RentCudaBFloat16Buffer(deviceIndex, output.Numel);
        NativeCudaBuffer<ushort> projectedGradient = projectedBorrowed
            ? projectedExisting!
            : rentedProjected = RentCudaBFloat16Buffer(deviceIndex, Numel);
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
            if (!projectedBorrowed)
                projectedGradient.MemSetToZero();
            CudaPublicOpsNative.ForgetScanBackwardBFloat16Gradient(
                deviceIndex,
                outputGradient.NativePtr,
                context.Memory.NativePtr,
                context.Forget.NativePtr,
                context.Input.NativePtr,
                context.Value.NativePtr,
                projectedGradient.NativePtr,
                batch,
                sequence,
                width,
                accelerator.DefaultStream);
            if (!projectedBorrowed)
            {
                AdoptCudaBFloat16GradientBuffer(
                    projectedGradient,
                    deviceIndex);
                rentedProjected = null;
            }
            return true;
        }
        finally
        {
            if (rentedOutput is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedOutput);
            if (rentedProjected is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedProjected);
        }
    }
}

internal sealed class CudaForgetScanContext : IDisposable
{
    private readonly NativeCudaDevice _accelerator;
    private int _disposed;

    private CudaForgetScanContext(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> memory,
        NativeCudaBuffer<float> forget,
        NativeCudaBuffer<float> input,
        NativeCudaBuffer<float> value)
    {
        _accelerator = accelerator;
        Memory = memory;
        Forget = forget;
        Input = input;
        Value = value;
    }

    internal NativeCudaBuffer<float> Memory { get; }
    internal NativeCudaBuffer<float> Forget { get; }
    internal NativeCudaBuffer<float> Input { get; }
    internal NativeCudaBuffer<float> Value { get; }

    internal static CudaForgetScanContext Allocate(
        NativeCudaDevice accelerator,
        int deviceIndex,
        int length)
    {
        NativeCudaBuffer<float>? memory = null;
        NativeCudaBuffer<float>? forget = null;
        NativeCudaBuffer<float>? input = null;
        NativeCudaBuffer<float>? value = null;
        try
        {
            memory = Tensor.RentCudaFloatBuffer(deviceIndex, length);
            forget = Tensor.RentCudaFloatBuffer(deviceIndex, length);
            input = Tensor.RentCudaFloatBuffer(deviceIndex, length);
            value = Tensor.RentCudaFloatBuffer(deviceIndex, length);
            return new CudaForgetScanContext(
                accelerator,
                memory,
                forget,
                input,
                value);
        }
        catch (Exception allocationFailure)
        {
            var releases = new List<Action>(4);
            if (memory is not null)
                releases.Add(() => Tensor.ReturnCudaFloatBuffer(accelerator, memory));
            if (forget is not null)
                releases.Add(() => Tensor.ReturnCudaFloatBuffer(accelerator, forget));
            if (input is not null)
                releases.Add(() => Tensor.ReturnCudaFloatBuffer(accelerator, input));
            if (value is not null)
                releases.Add(() => Tensor.ReturnCudaFloatBuffer(accelerator, value));
            try
            {
                CudaResourceCleanup.RunAll(
                    "CUDA ForgetScan allocation rollback failed.",
                    releases);
            }
            catch (Exception cleanupFailure)
            {
                throw new AggregateException(
                    allocationFailure,
                    cleanupFailure);
            }
            throw;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        CudaResourceCleanup.RunAll(
            "CUDA ForgetScan context cleanup failed.",
            () => Tensor.ReturnCudaFloatBuffer(_accelerator, Memory),
            () => Tensor.ReturnCudaFloatBuffer(_accelerator, Forget),
            () => Tensor.ReturnCudaFloatBuffer(_accelerator, Input),
            () => Tensor.ReturnCudaFloatBuffer(_accelerator, Value));
    }
}
