
namespace NNtrain;

internal static partial class TensorCudaKernels
{
    internal static NativeCudaBuffer<float> MatMulForwardResident(
        Tensor left,
        Tensor right,
        int batch,
        int m,
        int k,
        int n)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaFloatBuffer(
            deviceIndex, checked(batch * m * n));
        CudaBlas.MatMulForward(
            accelerator,
            deviceIndex,
            left.EnsureCudaFloat32Buffer(deviceIndex),
            right.EnsureCudaFloat32Buffer(deviceIndex),
            output,
            batch,
            m,
            k,
            n);
        return output;
    }

    internal static NativeCudaBuffer<ushort>
        MatMulForwardBFloat16Resident(
            Tensor left,
            Tensor right,
            int batch,
            int m,
            int k,
            int n)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaBFloat16Buffer(
            deviceIndex, checked(batch * m * n));
        CudaBlas.MatMulForwardBFloat16(
            accelerator,
            deviceIndex,
            left.EnsureCudaBFloat16Buffer(deviceIndex),
            right.EnsureCudaBFloat16Buffer(deviceIndex),
            output,
            batch,
            m,
            k,
            n);
        return output;
    }

    internal static void MatMulBackwardResident(
        Tensor left,
        Tensor right,
        Tensor output,
        int batch,
        int m,
        int k,
        int n)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        CudaBlas.MatMulBackward(
            accelerator,
            deviceIndex,
            left.EnsureCudaFloat32Buffer(deviceIndex),
            right.EnsureCudaFloat32Buffer(deviceIndex),
            output.EnsureCudaGradientBuffer(deviceIndex),
            left.EnsureCudaGradientBuffer(deviceIndex),
            right.EnsureCudaGradientBuffer(deviceIndex),
            batch,
            m,
            k,
            n);
        left.MarkCudaGradientMutated(deviceIndex);
        right.MarkCudaGradientMutated(deviceIndex);
    }

    internal static void MatMulBackwardBFloat16Resident(
        Tensor left,
        Tensor right,
        Tensor output,
        int batch,
        int m,
        int k,
        int n)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);

        if (TensorExecutionContext.UsesBFloat16GradientStorage)
        {
            bool sameParent = ReferenceEquals(left, right);
            using CudaBFloat16GradientSource outputGradient =
                CudaBFloat16GradientSource.Acquire(output, deviceIndex);
            using CudaPureBFloat16GradientTarget leftGradient =
                CudaPureBFloat16GradientTarget.Acquire(left, deviceIndex);
            using CudaPureBFloat16GradientTarget? rightGradient = sameParent
                ? null
                : CudaPureBFloat16GradientTarget.Acquire(
                    right,
                    deviceIndex);

            CudaBlas.MatMulBackwardLeftBFloat16Direct(
                accelerator,
                deviceIndex,
                right.EnsureCudaBFloat16Buffer(deviceIndex),
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
            CudaBlas.MatMulBackwardRightBFloat16Direct(
                accelerator,
                deviceIndex,
                left.EnsureCudaBFloat16Buffer(deviceIndex),
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

        int length = checked(batch * m * n);
        var encodedGradient = Tensor.RentCudaBFloat16Buffer(deviceIndex, length);
        try
        {
            CudaTensorNative.EncodeBFloat16(
                deviceIndex,
                output.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                encodedGradient.NativePtr,
                length);

            CudaBlas.MatMulBackwardBFloat16(
                accelerator,
                deviceIndex,
                left.EnsureCudaBFloat16Buffer(deviceIndex),
                right.EnsureCudaBFloat16Buffer(deviceIndex),
                encodedGradient,
                left.EnsureCudaGradientBuffer(deviceIndex),
                right.EnsureCudaGradientBuffer(deviceIndex),
                batch,
                m,
                k,
                n);
            left.MarkCudaGradientMutated(deviceIndex);
            right.MarkCudaGradientMutated(deviceIndex);
        }
        finally
        {
            Tensor.ReturnCudaBFloat16Buffer(accelerator, encodedGradient);
        }
    }
}
