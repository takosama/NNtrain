using System.Runtime.InteropServices;
using System.Text;
using NNtrain.Cuda.Execution;

namespace NNtrain.Cuda.Interop;

public static partial class CudaNativeGateway
{

    private const ulong KnownCapabilityMask =
        (ulong)(CudaKernelFeature.TensorCores |
            CudaKernelFeature.BFloat16 |
            CudaKernelFeature.FlashAttention |
            CudaKernelFeature.FusedLayerNorm |
            CudaKernelFeature.ForgetMemory |
            CudaKernelFeature.BlockReducedMuon |
            CudaKernelFeature.AsynchronousGradientReduction |
            CudaKernelFeature.CudaGraphs |
            CudaKernelFeature.Bfp8Quantization |
            CudaKernelFeature.Int8TensorCores |
            CudaKernelFeature.FusedFirstOrderOptimizers);

    public static int DeviceCount(out int count)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.DeviceCount(out count),
            CudaNativeOperation.DeviceCount,
            device: -1);
    }

    public static int DeviceName(
        int device,
        StringBuilder destination,
        int capacity)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.DeviceName(device, destination, capacity),
            CudaNativeOperation.DeviceName,
            device);
    }

    public static int SetDevice(int device)
    {
        EnsureCompatibleAbi();
        unchecked
        {
            _threadSetDeviceCallCount++;
        }
        return Complete(
            NativeMethods.SetDevice(device),
            CudaNativeOperation.SetDevice,
            device);
    }

    public static int Synchronize(int device)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Synchronize(device),
            CudaNativeOperation.Synchronize,
            device);
    }

    public static int CanAccessPeer(
        int device,
        int peerDevice,
        out int canAccess)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.CanAccessPeer(device, peerDevice, out canAccess),
            CudaNativeOperation.PeerAccess,
            device);
    }

    public static int KernelCapabilities(
        int device,
        out CudaKernelCapabilities capabilities)
    {
        EnsureCompatibleAbi();
        int status = Complete(
            NativeMethods.CapabilityBitmap(
                device,
                out ulong bitmap,
                out int major,
                out int minor),
            CudaNativeOperation.Capabilities,
            device);
        capabilities = new CudaKernelCapabilities(
            major,
            minor,
            (CudaKernelFeature)(bitmap & KnownCapabilityMask));
        return status;
    }

    /// <summary>
    /// Returns the minimum native capability bits advertised by a device
    /// before a training-kernel operation should be selected.
    /// </summary>
    public static CudaKernelFeature RequiredFeatures(
        CudaNativeOperation operation) => operation switch
    {
        CudaNativeOperation.LayerNormForward or
        CudaNativeOperation.LayerNormBackward or
        CudaNativeOperation.ResidualDropoutLayerNormForward or
        CudaNativeOperation.ResidualDropoutLayerNormBackward =>
            CudaKernelFeature.FusedLayerNorm,

        CudaNativeOperation.LayerNormForwardBFloat16 or
        CudaNativeOperation.LayerNormBackwardBFloat16 or
        CudaNativeOperation.ResidualDropoutLayerNormForwardBFloat16 or
        CudaNativeOperation.ResidualDropoutLayerNormBackwardBFloat16 or
        CudaNativeOperation
            .ResidualDropoutLayerNormBackwardBFloat16BranchGradient or
        CudaNativeOperation
            .ResidualDropoutLayerNormBackwardBFloat16IoGradient =>
            CudaKernelFeature.FusedLayerNorm |
            CudaKernelFeature.BFloat16,

        CudaNativeOperation.FlashAttentionForward or
        CudaNativeOperation.FlashAttentionBackward =>
            CudaKernelFeature.FlashAttention,

        CudaNativeOperation.FlashAttentionForwardBFloat16 or
        CudaNativeOperation.FlashAttentionBackwardBFloat16 or
        CudaNativeOperation.FlashAttentionIncrementalBFloat16 or
        CudaNativeOperation.FlashAttentionPrefillCacheBFloat16 =>
            CudaKernelFeature.FlashAttention |
            CudaKernelFeature.BFloat16,

        CudaNativeOperation.FlashAttentionForwardBFloat16TensorCore or
        CudaNativeOperation.FlashAttentionForwardBFloat16TensorCoreSync or
        CudaNativeOperation.FlashAttentionBackwardBFloat16TensorCore or
        CudaNativeOperation
            .FlashAttentionBackwardBFloat16TensorCoreParallelDkv or
        CudaNativeOperation
            .FlashAttentionBackwardBFloat16TensorCoreBFloat16Gradient or
        CudaNativeOperation
            .FlashAttentionBackwardBFloat16TensorCoreBFloat16IoGradient or
        CudaNativeOperation
            .FlashAttentionBackwardBFloat16TensorCoreBFloat16IoGradientSync or
        CudaNativeOperation
            .FlashAttentionBackwardBFloat16TensorCoreBfp8Output =>
            CudaKernelFeature.FlashAttention |
            CudaKernelFeature.BFloat16 |
            CudaKernelFeature.TensorCores |
            (operation == CudaNativeOperation
                .FlashAttentionBackwardBFloat16TensorCoreBfp8Output
                    ? CudaKernelFeature.Bfp8Quantization
                    : CudaKernelFeature.None),

        CudaNativeOperation.ForgetMemoryForward or
        CudaNativeOperation.ForgetMemoryBackward =>
            CudaKernelFeature.ForgetMemory,

        CudaNativeOperation.ForgetMemoryForwardBFloat16TensorCore =>
            CudaKernelFeature.ForgetMemory |
            CudaKernelFeature.BFloat16 |
            CudaKernelFeature.TensorCores,

        CudaNativeOperation.Bfp8GradientQuantize or
        CudaNativeOperation.Bfp8GradientQuantizeAccumulate or
        CudaNativeOperation.Bfp8GradientSquaredSum or
        CudaNativeOperation.Bfp8GradientScale =>
            CudaKernelFeature.Bfp8Quantization,

        CudaNativeOperation.Bfp8GradientReduce or
        CudaNativeOperation.Bfp8GradientBroadcast =>
            CudaKernelFeature.Bfp8Quantization |
            CudaKernelFeature.AsynchronousGradientReduction,

        CudaNativeOperation.NekoMuonMomentsStatsCompact or
        CudaNativeOperation.NekoMuonMomentsStatsCompactFinite =>
            CudaKernelFeature.BlockReducedMuon,

        CudaNativeOperation.GraphBeginCapture or
        CudaNativeOperation.GraphEndCapture or
        CudaNativeOperation.GraphInstantiate or
        CudaNativeOperation.GraphLaunch or
        CudaNativeOperation.GraphDestroy or
        CudaNativeOperation.GraphExecutableDestroy or
        CudaNativeOperation.GraphRngStep or
        CudaNativeOperation.GraphCounterSet or
        CudaNativeOperation.GraphCounterAdvance or
        CudaNativeOperation.GraphDropoutForward or
        CudaNativeOperation.GraphAddDropoutForward or
        CudaNativeOperation.GraphDropoutBackward or
        CudaNativeOperation.GraphAddDropoutBackward =>
            CudaKernelFeature.CudaGraphs,

        CudaNativeOperation.EmbeddingBackwardBFloat16Gradient or
        CudaNativeOperation.EmbeddingPositionsBackwardBFloat16Gradient or
        CudaNativeOperation.DropoutBackwardBFloat16Gradient or
        CudaNativeOperation.AddDropoutBackwardBFloat16Gradient or
        CudaNativeOperation.LinearBiasBackwardBFloat16Gradient or
        CudaNativeOperation.BFloat16GradientSquaredSum or
        CudaNativeOperation.BFloat16GradientScale =>
            CudaKernelFeature.BFloat16,

        CudaNativeOperation.TensorPrimitiveBFloat16 or
        CudaNativeOperation.OptimizerBFloat16 =>
            CudaKernelFeature.BFloat16,

        CudaNativeOperation.OptimizerBfp8 =>
            CudaKernelFeature.Bfp8Quantization,

        CudaNativeOperation.OptimizerNekoMuon =>
            CudaKernelFeature.BlockReducedMuon,

        CudaNativeOperation.OptimizerNekoMuonBFloat16 =>
            CudaKernelFeature.BlockReducedMuon |
            CudaKernelFeature.BFloat16 |
            CudaKernelFeature.TensorCores,

        CudaNativeOperation.GradientCollective =>
            CudaKernelFeature.AsynchronousGradientReduction,

        CudaNativeOperation.GradientCollectiveBFloat16 =>
            CudaKernelFeature.AsynchronousGradientReduction |
            CudaKernelFeature.BFloat16,

        _ => CudaKernelFeature.None,
    };

    private static partial class NativeMethods
    {

        [DllImport(LibraryName, EntryPoint = "nntrain_capability_bitmap",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CapabilityBitmap(
            int device,
            out ulong bitmap,
            out int computeCapabilityMajor,
            out int computeCapabilityMinor);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_device_count",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DeviceCount(out int count);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_device_name",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DeviceName(
            int device,
            StringBuilder destination,
            int capacity);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_set_device",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int SetDevice(int device);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_synchronize",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Synchronize(int device);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_can_access_peer",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CanAccessPeer(
            int device,
            int peerDevice,
            out int canAccess);
    }
}
