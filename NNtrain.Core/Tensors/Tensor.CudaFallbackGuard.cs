namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// CUDA execution must never continue through a managed implementation
    /// that reads the host storage directly. CUDA-produced tensors may have no
    /// current host replica, so such a fallback is both a hidden transfer risk
    /// and, more importantly, a stale-data correctness bug.
    /// </summary>
    private static void ThrowIfCudaHostFallback(string operation)
    {
        if (ExecutionDevice != TensorDevice.Cuda)
            return;

        throw new PlatformNotSupportedException(
            $"{operation} has no resident CUDA implementation for this " +
            "shape/dtype combination; implicit CPU fallback is forbidden.");
    }
}
