namespace NNtrain;

public partial class Tensor
{
    private static Tensor FromCudaBfp8Result(
        CudaBfp8OwnedBuffers buffers,
        int deviceIndex,
        int[] shape,
        Tensor[] parents)
    {
        ArgumentNullException.ThrowIfNull(buffers);
        Bfp8QuantizationDescriptor descriptor = buffers.Descriptor;
        var detached = buffers.Detach();
        try
        {
            return FromCudaBfp8Result(
                detached.Payload,
                detached.Scales,
                descriptor,
                deviceIndex,
                shape,
                parents);
        }
        catch
        {
            detached.Payload.Dispose();
            detached.Scales.Dispose();
            throw;
        }
    }

    private static Tensor FromCudaBfp8Result(
        NativeCudaBuffer<sbyte> payload,
        NativeCudaBuffer<float> scales,
        Bfp8QuantizationDescriptor descriptor,
        int deviceIndex,
        int[] shape,
        Tensor[] parents)
    {
        ArgumentNullException.ThrowIfNull(payload);
        ArgumentNullException.ThrowIfNull(scales);
        ArgumentNullException.ThrowIfNull(descriptor);
        var result = new Tensor(
            TensorStorage.CreateDeviceBfp8Placeholder(
                checked((int)payload.Length),
                descriptor),
            shape,
            parents,
            cudaResult: true);
        result.AdoptCudaBfp8Buffers(
            payload,
            scales,
            descriptor,
            deviceIndex);
        if (!AutogradContext.IsRecordingEnabled)
            CudaInferenceScope.Track(result, deviceIndex);
        return result;
    }

    private static Bfp8QuantizationDescriptor SelectBfp8ResultDescriptor(
        params Tensor[] operands)
    {
        ArgumentNullException.ThrowIfNull(operands);
        Bfp8QuantizationDescriptor? block = null;
        foreach (Tensor operand in operands)
        {
            if (operand.DType != TensorDType.Bfp8)
            {
                throw new InvalidOperationException(
                    "A BFP8 CUDA GEMM requires BFP8 operands.");
            }

            Bfp8QuantizationDescriptor descriptor = operand.Bfp8Quantization
                ?? throw new InvalidOperationException(
                    "BFP8 tensor storage has no quantization descriptor.");
            if (descriptor.Granularity == Bfp8ScaleGranularity.Block)
            {
                if (block is not null && block != descriptor)
                {
                    throw new InvalidOperationException(
                        "BFP8 GEMM operands must use the same block scale contract.");
                }
                block = descriptor;
            }
        }
        return block ?? Bfp8QuantizationDescriptor.TensorWide;
    }
}
