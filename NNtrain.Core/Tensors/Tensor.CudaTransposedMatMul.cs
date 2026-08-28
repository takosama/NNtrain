namespace NNtrain;

partial class Tensor
{
    private Tensor MatMulTransposedRightCuda(
        Tensor other,
        int batch,
        int m,
        int k,
        int n,
        int[] outputShape)
    {
        int deviceIndex = CudaDeviceIndex;
        Tensor result;
        if (DType == TensorDType.Bfp8)
        {
            Bfp8QuantizationDescriptor descriptor =
                SelectBfp8ResultDescriptor(this, other);
            using CudaBfp8OwnedBuffers output =
                CudaBfp8Gemm.MatMulTransposedRightForward(
                    this,
                    other,
                    descriptor,
                    batch,
                    m,
                    k,
                    n);
            result = FromCudaBfp8Result(
                output,
                deviceIndex,
                outputShape,
                [this, other]);
        }
        else
        {
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            if (DType == TensorDType.BFloat16)
            {
                NativeCudaBuffer<ushort> output =
                    RentCudaBFloat16Buffer(
                        deviceIndex,
                        checked(batch * m * n));
                try
                {
                    CudaBlas.MatMulTransposedRightForwardBFloat16(
                        accelerator,
                        deviceIndex,
                        EnsureCudaBFloat16Buffer(deviceIndex),
                        other.EnsureCudaBFloat16Buffer(deviceIndex),
                        output,
                        batch,
                        m,
                        k,
                        n);
                    result = FromCudaResult(
                        output,
                        deviceIndex,
                        outputShape,
                        [this, other],
                        TensorDType.BFloat16);
                }
                catch
                {
                    ReturnCudaBFloat16Buffer(accelerator, output);
                    throw;
                }
            }
            else
            {
                NativeCudaBuffer<float> output = RentCudaFloatBuffer(
                    deviceIndex,
                    checked(batch * m * n));
                try
                {
                    CudaBlas.MatMulTransposedRightForward(
                        accelerator,
                        deviceIndex,
                        EnsureCudaFloat32Buffer(deviceIndex),
                        other.EnsureCudaFloat32Buffer(deviceIndex),
                        output,
                        batch,
                        m,
                        k,
                        n);
                    result = FromCudaResult(
                        output,
                        deviceIndex,
                        outputShape,
                        [this, other],
                        TensorDType.Float32);
                }
                catch
                {
                    ReturnCudaFloatBuffer(accelerator, output);
                    throw;
                }
            }
        }

        if (AutogradContext.IsRecordingEnabled)
        {
            result.Node.BackwardAction = () =>
                MatMulTransposedRightBackwardCuda(
                    other,
                    result,
                    batch,
                    m,
                    k,
                    n);
        }
        return result;
    }

    private void MatMulTransposedRightBackwardCuda(
        Tensor other,
        Tensor output,
        int batch,
        int m,
        int k,
        int n)
    {
        if (DType == TensorDType.Bfp8)
        {
            CudaBfp8Gemm.MatMulTransposedRightBackward(
                this,
                other,
                output,
                batch,
                m,
                k,
                n);
            return;
        }

        int deviceIndex = CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        if (DType == TensorDType.BFloat16)
        {
            if (TensorExecutionContext.UsesBFloat16GradientStorage)
            {
                bool sameParent = ReferenceEquals(this, other);
                using CudaBFloat16GradientSource outputGradient =
                    CudaBFloat16GradientSource.Acquire(
                        output,
                        deviceIndex);
                using CudaPureBFloat16GradientTarget leftGradient =
                    CudaPureBFloat16GradientTarget.Acquire(
                        this,
                        deviceIndex);
                using CudaPureBFloat16GradientTarget? rightGradient =
                    sameParent
                        ? null
                        : CudaPureBFloat16GradientTarget.Acquire(
                            other,
                            deviceIndex);

                CudaBlas
                    .MatMulTransposedRightBackwardInputBFloat16Accumulate(
                        accelerator,
                        deviceIndex,
                        other.EnsureCudaBFloat16Buffer(deviceIndex),
                        outputGradient.Buffer,
                        leftGradient.Buffer,
                        batch,
                        m,
                        k,
                        n,
                        leftGradient.HasValue);
                leftGradient.MarkFullContributionWritten();

                CudaPureBFloat16GradientTarget effectiveRight =
                    rightGradient ?? leftGradient;
                CudaBlas
                    .MatMulTransposedRightBackwardWeightBFloat16Direct(
                        accelerator,
                        deviceIndex,
                        EnsureCudaBFloat16Buffer(deviceIndex),
                        outputGradient.Buffer,
                        effectiveRight.Buffer,
                        batch,
                        m,
                        k,
                        n,
                        effectiveRight.HasValue);
                effectiveRight.MarkFullContributionWritten();

                leftGradient.Commit();
                rightGradient?.Commit();
                return;
            }

            NativeCudaBuffer<ushort>? rentedGradient = null;
            bool borrowed = output.TryGetCudaBFloat16GradientBuffer(
                deviceIndex,
                out NativeCudaBuffer<ushort>? directGradient);
            NativeCudaBuffer<ushort> encodedGradient = borrowed
                ? directGradient!
                : rentedGradient = RentCudaBFloat16Buffer(
                    deviceIndex,
                    output.Numel);
            NativeCudaBuffer<ushort>? directInputGradient = null;
            try
            {
                if (!borrowed)
                {
                    CudaTensorNative.EncodeBFloat16(
                        deviceIndex,
                        output.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                        encodedGradient.NativePtr,
                        output.Numel);
                }

                bool publishDirectInput =
                    TensorExecutionContext.UsesBFloat16GradientStorage
                    && !HasGradientBuffer;
                if (publishDirectInput)
                {
                    directInputGradient = RentCudaBFloat16Buffer(
                        deviceIndex,
                        checked(batch * m * k));
                    CudaBlas
                        .MatMulTransposedRightBackwardInputBFloat16Direct(
                            accelerator,
                            deviceIndex,
                            other.EnsureCudaBFloat16Buffer(deviceIndex),
                            encodedGradient,
                            directInputGradient,
                            batch,
                            m,
                            k,
                            n);
                    CudaBlas.MatMulTransposedRightBackwardWeightBFloat16(
                        accelerator,
                        deviceIndex,
                        EnsureCudaBFloat16Buffer(deviceIndex),
                        encodedGradient,
                        other.EnsureCudaGradientBuffer(deviceIndex),
                        batch,
                        m,
                        k,
                        n);
                    AdoptCudaBFloat16GradientBuffer(
                        directInputGradient,
                        deviceIndex);
                    directInputGradient = null;
                    other.MarkCudaGradientMutated(deviceIndex);
                }
                else
                {
                    CudaBlas.MatMulTransposedRightBackwardBFloat16(
                        accelerator,
                        deviceIndex,
                        EnsureCudaBFloat16Buffer(deviceIndex),
                        other.EnsureCudaBFloat16Buffer(deviceIndex),
                        encodedGradient,
                        EnsureCudaGradientBuffer(deviceIndex),
                        other.EnsureCudaGradientBuffer(deviceIndex),
                        batch,
                        m,
                        k,
                        n);
                    MarkCudaGradientMutated(deviceIndex);
                    other.MarkCudaGradientMutated(deviceIndex);
                }
            }
            finally
            {
                if (directInputGradient is not null)
                    ReturnCudaBFloat16Buffer(accelerator, directInputGradient);
                if (rentedGradient is not null)
                    ReturnCudaBFloat16Buffer(accelerator, rentedGradient);
            }
            return;
        }

        CudaBlas.MatMulTransposedRightBackward(
            accelerator,
            deviceIndex,
            EnsureCudaFloat32Buffer(deviceIndex),
            other.EnsureCudaFloat32Buffer(deviceIndex),
            output.EnsureCudaGradientBuffer(deviceIndex),
            EnsureCudaGradientBuffer(deviceIndex),
            other.EnsureCudaGradientBuffer(deviceIndex),
            batch,
            m,
            k,
            n);
        MarkCudaGradientMutated(deviceIndex);
        other.MarkCudaGradientMutated(deviceIndex);
    }
}
