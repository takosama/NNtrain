using NNtrain.Cuda.Interop;

namespace NNtrain;

internal readonly record struct CudaTopKSelection(
    CudaTopKCandidate[] Candidates);

public partial class Tensor
{
    /// <summary>
    /// Reduces one CUDA-resident logits row to at most 64 stable candidates.
    /// Only the final value/index pairs cross to the host; Float32 and BF16
    /// storage are read directly, while BFP8 uses its device BF16 decode.
    /// </summary>
    internal CudaTopKSelection ReadCudaTopK(
        int offset,
        int count,
        int k,
        int deviceIndex = -1)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        if (offset > Numel - count)
            throw new ArgumentOutOfRangeException(nameof(count));
        if (k <= 0 || k > Math.Min(64, count))
        {
            throw new ArgumentOutOfRangeException(
                nameof(k),
                k,
                "CUDA top-K supports 1 through min(64, count).");
        }
        if (Device != TensorDevice.Cuda)
        {
            throw new InvalidOperationException(
                "CUDA top-K requires a CUDA-resident tensor.");
        }

        int resolvedDevice = ResolveCudaDeviceIndex(deviceIndex);
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(resolvedDevice);
        accelerator.Bind();
        int reductionBlocks = Math.Min(
            1024,
            Math.Max(1, (count + 255) / 256));
        // A candidate is exactly two 32-bit words.  Borrowing the established
        // transient float pool avoids a cudaMalloc/cudaFree pair per generated
        // token both with and without an ExecutionSession lane.
        NativeCudaBuffer<float> workspace = RentCudaFloatBuffer(
            resolvedDevice,
            checked(reductionBlocks * k * 2));
        NativeCudaBuffer<float>? output = null;
        CudaBfp8BFloat16Lease? bfp8Lease = null;
        try
        {
            output = RentCudaFloatBuffer(resolvedDevice, checked(k * 2));
            int status;
            if (DType == TensorDType.BFloat16)
            {
                NativeCudaBuffer<ushort> values =
                    EnsureCudaBFloat16Buffer(resolvedDevice);
                status = CudaNativeGateway.TensorTopKBFloat16(
                    resolvedDevice,
                    values.NativePtr,
                    offset,
                    count,
                    k,
                    workspace.NativePtr,
                    reductionBlocks,
                    output.NativePtr);
            }
            else if (DType == TensorDType.Bfp8)
            {
                bfp8Lease = AcquireCudaBfp8BFloat16Buffer(resolvedDevice);
                status = CudaNativeGateway.TensorTopKBFloat16(
                    resolvedDevice,
                    bfp8Lease.Buffer.NativePtr,
                    offset,
                    count,
                    k,
                    workspace.NativePtr,
                    reductionBlocks,
                    output.NativePtr);
            }
            else
            {
                NativeCudaBuffer<float> values =
                    EnsureCudaFloat32Buffer(resolvedDevice);
                status = CudaNativeGateway.TensorTopKFloat32(
                    resolvedDevice,
                    values.NativePtr,
                    offset,
                    count,
                    k,
                    workspace.NativePtr,
                    reductionBlocks,
                    output.NativePtr);
            }
            NativeCudaRuntime.Check(status, "CUDA vocabulary top-K");

            var rawCandidates = new float[checked(k * 2)];
            output.CopyToCPU(rawCandidates);
            var candidates = new CudaTopKCandidate[k];
            for (int index = 0; index < candidates.Length; index++)
            {
                candidates[index] = new CudaTopKCandidate
                {
                    Index = BitConverter.SingleToInt32Bits(
                        rawCandidates[index * 2]),
                    Value = rawCandidates[index * 2 + 1],
                };
                if ((uint)candidates[index].Index >= (uint)count)
                {
                    throw new InvalidOperationException(
                        "CUDA top-K returned an invalid logical index.");
                }
            }
            return new CudaTopKSelection(candidates);
        }
        finally
        {
            bfp8Lease?.Dispose();
            if (output is not null)
                ReturnCudaFloatBuffer(accelerator, output);
            ReturnCudaFloatBuffer(accelerator, workspace);
        }
    }
}
