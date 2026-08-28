using System.Runtime.ExceptionServices;

namespace NNtrain;

partial class Tensor
{
    private Tensor ForgetMemoryBfp8Cuda(
        int batch,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        bool useV3,
        bool useDrn)
    {
        Bfp8QuantizationDescriptor outputDescriptor =
            SelectBfp8ResultDescriptor(this);
        CudaBfp8ForgetMemoryResidentContext forward =
            CudaOperationProfiler.IsEnabled
                ? CudaOperationProfiler.Measure(
                    "forward.forget_memory",
                    () => CudaBfp8ForgetMemory.ForwardResident(
                        this,
                        outputDescriptor,
                        batch,
                        sequence,
                        projectionWidth,
                        keyWidth,
                        valueWidth,
                        retentionFloor,
                        useV3,
                        useDrn))
                : CudaBfp8ForgetMemory.ForwardResident(
                    this,
                    outputDescriptor,
                    batch,
                    sequence,
                    projectionWidth,
                    keyWidth,
                    valueWidth,
                    retentionFloor,
                    useV3,
                    useDrn);

        Tensor result;
        try
        {
            using CudaBfp8OwnedBuffers output =
                forward.DetachEncodedOutput();
            result = FromCudaBfp8Result(
                output,
                forward.DeviceIndex,
                [batch, sequence, valueWidth],
                [this]);
        }
        catch (Exception conversionFailure)
        {
            try
            {
                forward.Dispose();
            }
            catch (Exception cleanupFailure)
            {
                throw new AggregateException(
                    "BFP8 ForgetMemory result construction and "
                    + "saved-context cleanup failed.",
                    conversionFailure,
                    cleanupFailure);
            }
            ExceptionDispatchInfo.Capture(conversionFailure).Throw();
            throw;
        }

        if (!AutogradContext.IsRecordingEnabled)
        {
            if (!CudaInferenceScope.TrackResource(forward))
                forward.Dispose();
            return result;
        }

        int deviceIndex = forward.DeviceIndex;
        AutogradLease<CudaBfp8ForgetMemoryResidentContext> lease =
            AutogradLease<CudaBfp8ForgetMemoryResidentContext>.Own(
                forward,
                AutogradLeaseMetadata.CudaOwned(
                    deviceIndex,
                    TensorDType.Bfp8,
                    DataVersion),
                static saved => saved.Dispose());
        result.Node.SetBackward(lease, savedContext =>
        {
            if (CudaOperationProfiler.IsEnabled)
            {
                CudaOperationProfiler.Measure(
                    "backward.forget_memory",
                    () => CudaBfp8ForgetMemory.BackwardResident(
                        this,
                        result,
                        savedContext,
                        batch,
                        sequence,
                        projectionWidth,
                        keyWidth,
                        valueWidth,
                        retentionFloor,
                        useV3,
                        useDrn));
            }
            else
            {
                CudaBfp8ForgetMemory.BackwardResident(
                    this,
                    result,
                    savedContext,
                    batch,
                    sequence,
                    projectionWidth,
                    keyWidth,
                    valueWidth,
                    retentionFloor,
                    useV3,
                    useDrn);
            }
        });
        return result;
    }

    private Tensor ForgetMemoryBfp8ContinueCuda(
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        float[] state,
        bool useV3,
        bool useDrn)
    {
        Bfp8QuantizationDescriptor outputDescriptor =
            SelectBfp8ResultDescriptor(this);
        using CudaBfp8OwnedBuffers output =
            CudaBfp8ForgetMemory.ForwardContinue(
                this,
                outputDescriptor,
                sequence,
                projectionWidth,
                keyWidth,
                valueWidth,
                retentionFloor,
                state,
                useV3,
                useDrn);
        return FromCudaBfp8Result(
            output,
            CudaDeviceIndex,
            [1, sequence, valueWidth],
            [this]);
    }

    private Tensor ForgetMemoryBfp8ContinueCuda(
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        NativeCudaBuffer<float> state,
        bool useV3,
        bool useDrn)
    {
        Bfp8QuantizationDescriptor outputDescriptor =
            SelectBfp8ResultDescriptor(this);
        using CudaBfp8OwnedBuffers output =
            CudaBfp8ForgetMemory.ForwardContinue(
                this,
                outputDescriptor,
                sequence,
                projectionWidth,
                keyWidth,
                valueWidth,
                retentionFloor,
                state,
                useV3,
                useDrn);
        return FromCudaBfp8Result(
            output,
            CudaDeviceIndex,
            [1, sequence, valueWidth],
            [this]);
    }
}
