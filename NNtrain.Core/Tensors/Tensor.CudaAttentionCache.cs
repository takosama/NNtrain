namespace NNtrain;

partial class Tensor
{
    internal void PrefillMultiHeadAttentionCache(
        NativeCudaBuffer<ushort> keyCache,
        NativeCudaBuffer<ushort> valueCache,
        int sequence,
        int cacheCapacity,
        int modelWidth)
    {
        if (ExecutionDevice != TensorDevice.Cuda
            || DType != TensorDType.BFloat16)
        {
            throw new InvalidOperationException(
                "Attention cache prefill requires CUDA BF16 execution.");
        }
        int deviceIndex = CudaDeviceIndex;
        CudaFlashAttention.PrefillCacheBFloat16(
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex),
            EnsureCudaBFloat16Buffer(deviceIndex),
            keyCache,
            valueCache,
            sequence,
            cacheCapacity,
            modelWidth);
    }

    internal Tensor FusedMultiHeadAttentionIncremental(
        NativeCudaBuffer<ushort> keyCache,
        NativeCudaBuffer<ushort> valueCache,
        int position,
        int cacheCapacity,
        int modelWidth,
        int numHeads)
    {
        ArgumentNullException.ThrowIfNull(keyCache);
        ArgumentNullException.ThrowIfNull(valueCache);
        if (ExecutionDevice != TensorDevice.Cuda
            || DType != TensorDType.BFloat16)
        {
            throw new InvalidOperationException(
                "Incremental attention requires CUDA BF16 execution.");
        }
        if (Rank is not 2 and not 3 || _shape[^1] != 3 * modelWidth)
        {
            throw new InvalidOperationException(
                "Incremental attention input must contain one fused QKV row.");
        }
        int rows = Numel / _shape[^1];
        if (rows != 1)
        {
            throw new InvalidOperationException(
                "Incremental attention accepts exactly one token.");
        }
        if (keyCache.Length != checked(cacheCapacity * modelWidth)
            || valueCache.Length != checked(cacheCapacity * modelWidth))
        {
            throw new ArgumentException(
                "K/V cache shape does not match its declared capacity.");
        }

        int deviceIndex = CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<ushort> projected =
            EnsureCudaBFloat16Buffer(deviceIndex);
        NativeCudaBuffer<ushort> output =
            RentCudaBFloat16Buffer(deviceIndex, modelWidth);
        try
        {
            CudaFlashAttention.IncrementalBFloat16(
                accelerator,
                projected,
                keyCache,
                valueCache,
                output,
                position,
                cacheCapacity,
                modelWidth,
                numHeads);
            int[] outputShape = Rank == 3
                ? [1, 1, modelWidth]
                : [1, modelWidth];
            return FromCudaResult(
                output,
                deviceIndex,
                outputShape,
                [],
                TensorDType.BFloat16);
        }
        catch
        {
            ReturnCudaBFloat16Buffer(accelerator, output);
            throw;
        }
    }
}
