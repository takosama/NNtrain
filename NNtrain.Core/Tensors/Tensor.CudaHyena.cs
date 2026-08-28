namespace NNtrain;

partial class Tensor
{
    private Tensor HyenaCuda(
        Tensor shortFilter,
        Tensor longFilter,
        Tensor diagonalBias,
        int batch,
        int sequence,
        int width,
        bool parallelLong)
    {
        if (shortFilter.DType != DType
            || longFilter.DType != DType
            || diagonalBias.DType != DType)
        {
            throw new InvalidOperationException(
                "CUDA Hyena requires matching physical storage types. " +
                "Convert all operands explicitly before dispatch.");
        }

        int deviceIndex = CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        nint stream = accelerator.DefaultStream;
        int channels = checked(3 * width);
        int outputLength = checked(batch * sequence * width);
        int shortLength = checked(batch * sequence * channels);
        CudaHyenaContext context = CudaHyenaContext.Allocate(
            accelerator,
            deviceIndex,
            shortLength,
            outputLength,
            parallelLong);
        Tensor result;
        try
        {
            if (DType == TensorDType.Bfp8)
            {
                Bfp8QuantizationDescriptor descriptor =
                    SelectBfp8ResultDescriptor(
                        this,
                        shortFilter,
                        longFilter,
                        diagonalBias);
                using CudaBfp8BFloat16Lease projectedDecoded =
                    AcquireCudaBfp8BFloat16Buffer(deviceIndex);
                using CudaBfp8BFloat16Lease shortDecoded =
                    shortFilter.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
                using CudaBfp8BFloat16Lease longDecoded =
                    longFilter.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
                using CudaBfp8BFloat16Lease diagonalDecoded =
                    diagonalBias.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
                NativeCudaBuffer<ushort> encodedOutput =
                    RentCudaBFloat16Buffer(deviceIndex, outputLength);
                using CudaBfp8OwnedBuffers output =
                    CudaBfp8OwnedBuffers.Allocate(
                        accelerator,
                        outputLength,
                        descriptor);
                try
                {
                    LaunchHyena(
                        projectedDecoded.Buffer.NativePtr,
                        shortDecoded.Buffer.NativePtr,
                        longDecoded.Buffer.NativePtr,
                        diagonalDecoded.Buffer.NativePtr,
                        encodedOutput.NativePtr,
                        bfloat16: true);
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
                        [this, shortFilter, longFilter, diagonalBias]);
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
                    LaunchHyena(
                        EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                        shortFilter.EnsureCudaBFloat16Buffer(deviceIndex)
                            .NativePtr,
                        longFilter.EnsureCudaBFloat16Buffer(deviceIndex)
                            .NativePtr,
                        diagonalBias.EnsureCudaBFloat16Buffer(deviceIndex)
                            .NativePtr,
                        output.NativePtr,
                        bfloat16: true);
                    result = FromCudaResult(
                        output,
                        deviceIndex,
                        [batch, sequence, width],
                        [this, shortFilter, longFilter, diagonalBias],
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
                    LaunchHyena(
                        EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                        shortFilter.EnsureCudaFloat32Buffer(deviceIndex)
                            .NativePtr,
                        longFilter.EnsureCudaFloat32Buffer(deviceIndex)
                            .NativePtr,
                        diagonalBias.EnsureCudaFloat32Buffer(deviceIndex)
                            .NativePtr,
                        output.NativePtr,
                        bfloat16: false);
                    result = FromCudaResult(
                        output,
                        deviceIndex,
                        [batch, sequence, width],
                        [this, shortFilter, longFilter, diagonalBias],
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
                ThrowIfCudaHostFallback(nameof(FusedCausalHyenaOrder2));
                throw new InvalidOperationException(
                    "Hyena CUDA dispatch did not select a supported dtype.");
            }
        }
        catch
        {
            context.Dispose();
            throw;
        }

        if (!AutogradContext.IsRecordingEnabled)
        {
            context.Dispose();
            return result;
        }

        AutogradLease<CudaHyenaContext> lease =
            AutogradLease<CudaHyenaContext>.Own(
                context,
                AutogradLeaseMetadata.CudaOwned(
                    deviceIndex,
                    result.DType,
                    DataVersion),
                static saved => saved.Dispose());
        result.Node.SetBackward(lease, savedContext => HyenaBackwardCuda(
            result,
            shortFilter,
            longFilter,
            diagonalBias,
            savedContext,
            batch,
            sequence,
            width));
        return result;

        void LaunchHyena(
            nint projected,
            nint shortWeights,
            nint longWeights,
            nint diagonal,
            nint output,
            bool bfloat16)
        {
            CudaPublicOpsNative.Hyena(
                deviceIndex,
                projected,
                shortWeights,
                longWeights,
                diagonal,
                output,
                context.ShortOutput.NativePtr,
                context.Gated.NativePtr,
                context.Convolved.NativePtr,
                batch,
                sequence,
                width,
                bfloat16,
                parallelLong,
                stream);
        }
    }

    private void HyenaBackwardCuda(
        Tensor output,
        Tensor shortFilter,
        Tensor longFilter,
        Tensor diagonalBias,
        CudaHyenaContext context,
        int batch,
        int sequence,
        int width)
    {
        int deviceIndex = CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        int channels = checked(3 * width);
        int shortLength = checked(batch * sequence * channels);
        int outputLength = checked(batch * sequence * width);
        NativeCudaBuffer<float> shortGradient =
            RentCudaFloatBuffer(deviceIndex, shortLength);
        NativeCudaBuffer<float> convolutionGradient =
            RentCudaFloatBuffer(deviceIndex, outputLength);
        NativeCudaBuffer<float> gatedGradient =
            RentCudaFloatBuffer(deviceIndex, outputLength);
        try
        {
            nint stream = accelerator.DefaultStream;
            if (DType == TensorDType.BFloat16
                && TensorExecutionContext.UsesBFloat16GradientStorage
                && TryHyenaBackwardBFloat16Direct(
                    output,
                    shortFilter,
                    longFilter,
                    diagonalBias,
                    context,
                    shortGradient,
                    convolutionGradient,
                    gatedGradient,
                    batch,
                    sequence,
                    width,
                    deviceIndex,
                    accelerator,
                    stream))
            {
                return;
            }
            if (DType == TensorDType.Bfp8)
            {
                using CudaBfp8BFloat16Lease projectedDecoded =
                    AcquireCudaBfp8BFloat16Buffer(deviceIndex);
                using CudaBfp8BFloat16Lease shortDecoded =
                    shortFilter.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
                using CudaBfp8BFloat16Lease longDecoded =
                    longFilter.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
                using CudaBfp8BFloat16Lease diagonalDecoded =
                    diagonalBias.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
                LaunchBackward(
                    projectedDecoded.Buffer.NativePtr,
                    shortDecoded.Buffer.NativePtr,
                    longDecoded.Buffer.NativePtr,
                    diagonalDecoded.Buffer.NativePtr,
                    bfloat16: true,
                    stream);
            }
            else if (DType == TensorDType.BFloat16)
            {
                LaunchBackward(
                    EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                    shortFilter.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                    longFilter.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                    diagonalBias.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                    bfloat16: true,
                    stream);
            }
            else
            {
                LaunchBackward(
                    EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                    shortFilter.EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                    longFilter.EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                    diagonalBias.EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                    bfloat16: false,
                    stream);
            }

            MarkCudaGradientMutated(deviceIndex);
            shortFilter.MarkCudaGradientMutated(deviceIndex);
            longFilter.MarkCudaGradientMutated(deviceIndex);
            diagonalBias.MarkCudaGradientMutated(deviceIndex);
        }
        finally
        {
            ReturnCudaFloatBuffer(accelerator, shortGradient);
            ReturnCudaFloatBuffer(accelerator, convolutionGradient);
            ReturnCudaFloatBuffer(accelerator, gatedGradient);
        }

        void LaunchBackward(
            nint projected,
            nint shortWeights,
            nint longWeights,
            nint diagonal,
            bool bfloat16,
            nint stream)
        {
            CudaPublicOpsNative.HyenaBackward(
                deviceIndex,
                projected,
                shortWeights,
                longWeights,
                diagonal,
                output.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                context.ShortOutput.NativePtr,
                context.Gated.NativePtr,
                context.Convolved.NativePtr,
                EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                shortFilter.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                longFilter.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                diagonalBias.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                shortGradient.NativePtr,
                convolutionGradient.NativePtr,
                gatedGradient.NativePtr,
                batch,
                sequence,
                width,
                bfloat16,
                context.ParallelLong,
                bfloat16Gradient: false,
                stream);
        }
    }

    private bool TryHyenaBackwardBFloat16Direct(
        Tensor output,
        Tensor shortFilter,
        Tensor longFilter,
        Tensor diagonalBias,
        CudaHyenaContext context,
        NativeCudaBuffer<float> shortGradient,
        NativeCudaBuffer<float> convolutionGradient,
        NativeCudaBuffer<float> gatedGradient,
        int batch,
        int sequence,
        int width,
        int deviceIndex,
        NativeCudaDevice accelerator,
        nint stream)
    {
        bool projectedBorrowed = TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? projectedExisting);
        bool shortBorrowed = shortFilter.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? shortExisting);
        bool longBorrowed = longFilter.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? longExisting);
        bool diagonalBorrowed = diagonalBias.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? diagonalExisting);
        if ((!projectedBorrowed && HasGradientBuffer)
            || (!shortBorrowed && shortFilter.HasGradientBuffer)
            || (!longBorrowed && longFilter.HasGradientBuffer)
            || (!diagonalBorrowed && diagonalBias.HasGradientBuffer))
        {
            return false;
        }

        bool outputBorrowed = output.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? outputExisting);
        NativeCudaBuffer<ushort>? rentedOutput = null;
        NativeCudaBuffer<ushort>? rentedProjected = null;
        NativeCudaBuffer<ushort>? rentedShort = null;
        NativeCudaBuffer<ushort>? rentedLong = null;
        NativeCudaBuffer<ushort>? rentedDiagonal = null;
        NativeCudaBuffer<ushort> outputGradient = outputBorrowed
            ? outputExisting!
            : rentedOutput = RentCudaBFloat16Buffer(deviceIndex, output.Numel);
        NativeCudaBuffer<ushort> projectedGradient = projectedBorrowed
            ? projectedExisting!
            : rentedProjected = RentCudaBFloat16Buffer(deviceIndex, Numel);
        NativeCudaBuffer<ushort> shortFilterGradient = shortBorrowed
            ? shortExisting!
            : rentedShort = RentCudaBFloat16Buffer(
                deviceIndex, shortFilter.Numel);
        NativeCudaBuffer<ushort> longFilterGradient = longBorrowed
            ? longExisting!
            : rentedLong = RentCudaBFloat16Buffer(
                deviceIndex, longFilter.Numel);
        NativeCudaBuffer<ushort> diagonalGradient = diagonalBorrowed
            ? diagonalExisting!
            : rentedDiagonal = RentCudaBFloat16Buffer(
                deviceIndex, diagonalBias.Numel);
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
            if (!shortBorrowed)
                shortFilterGradient.MemSetToZero();
            if (!longBorrowed)
                longFilterGradient.MemSetToZero();
            if (!diagonalBorrowed)
                diagonalGradient.MemSetToZero();
            CudaPublicOpsNative.HyenaBackward(
                deviceIndex,
                EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                shortFilter.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                longFilter.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                diagonalBias.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr,
                outputGradient.NativePtr,
                context.ShortOutput.NativePtr,
                context.Gated.NativePtr,
                context.Convolved.NativePtr,
                projectedGradient.NativePtr,
                shortFilterGradient.NativePtr,
                longFilterGradient.NativePtr,
                diagonalGradient.NativePtr,
                shortGradient.NativePtr,
                convolutionGradient.NativePtr,
                gatedGradient.NativePtr,
                batch,
                sequence,
                width,
                bfloat16: true,
                context.ParallelLong,
                bfloat16Gradient: true,
                stream);
            if (!projectedBorrowed)
            {
                AdoptCudaBFloat16GradientBuffer(
                    projectedGradient, deviceIndex);
                rentedProjected = null;
            }
            if (!shortBorrowed)
            {
                shortFilter.AdoptCudaBFloat16GradientBuffer(
                    shortFilterGradient, deviceIndex);
                rentedShort = null;
            }
            if (!longBorrowed)
            {
                longFilter.AdoptCudaBFloat16GradientBuffer(
                    longFilterGradient, deviceIndex);
                rentedLong = null;
            }
            if (!diagonalBorrowed)
            {
                diagonalBias.AdoptCudaBFloat16GradientBuffer(
                    diagonalGradient, deviceIndex);
                rentedDiagonal = null;
            }
            return true;
        }
        finally
        {
            if (rentedOutput is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedOutput);
            if (rentedProjected is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedProjected);
            if (rentedShort is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedShort);
            if (rentedLong is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedLong);
            if (rentedDiagonal is not null)
                ReturnCudaBFloat16Buffer(accelerator, rentedDiagonal);
        }
    }
}

internal sealed class CudaHyenaContext : IDisposable
{
    private readonly NativeCudaDevice _accelerator;
    private int _disposed;

    private CudaHyenaContext(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> shortOutput,
        NativeCudaBuffer<float> gated,
        NativeCudaBuffer<float> convolved,
        bool parallelLong)
    {
        _accelerator = accelerator;
        ShortOutput = shortOutput;
        Gated = gated;
        Convolved = convolved;
        ParallelLong = parallelLong;
    }

    internal NativeCudaBuffer<float> ShortOutput { get; }
    internal NativeCudaBuffer<float> Gated { get; }
    internal NativeCudaBuffer<float> Convolved { get; }
    internal bool ParallelLong { get; }

    internal static CudaHyenaContext Allocate(
        NativeCudaDevice accelerator,
        int deviceIndex,
        int shortLength,
        int outputLength,
        bool parallelLong)
    {
        NativeCudaBuffer<float>? shortOutput = null;
        NativeCudaBuffer<float>? gated = null;
        NativeCudaBuffer<float>? convolved = null;
        try
        {
            shortOutput = Tensor.RentCudaFloatBuffer(deviceIndex, shortLength);
            gated = Tensor.RentCudaFloatBuffer(deviceIndex, outputLength);
            convolved = Tensor.RentCudaFloatBuffer(deviceIndex, outputLength);
            return new CudaHyenaContext(
                accelerator,
                shortOutput,
                gated,
                convolved,
                parallelLong);
        }
        catch (Exception allocationFailure)
        {
            var releases = new List<Action>(3);
            if (shortOutput is not null)
                releases.Add(() => Tensor.ReturnCudaFloatBuffer(accelerator, shortOutput));
            if (gated is not null)
                releases.Add(() => Tensor.ReturnCudaFloatBuffer(accelerator, gated));
            if (convolved is not null)
                releases.Add(() => Tensor.ReturnCudaFloatBuffer(accelerator, convolved));
            try
            {
                CudaResourceCleanup.RunAll(
                    "CUDA Hyena allocation rollback failed.",
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
            "CUDA Hyena context cleanup failed.",
            () => Tensor.ReturnCudaFloatBuffer(_accelerator, ShortOutput),
            () => Tensor.ReturnCudaFloatBuffer(_accelerator, Gated),
            () => Tensor.ReturnCudaFloatBuffer(_accelerator, Convolved));
    }
}
