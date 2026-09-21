using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

public static partial class CudaNativeGateway
{
    public static int DrnChunkParallelForward(int device, nint projected, nint output,
        nint checkpoints, nint state, nint ping, nint pong, nint mismatch,
        int batch, int sequence, int width, int key, int value, int passes, float retentionFloor, nint stream)
    {
        EnsureDrnRetentionFloorAbi(retentionFloor);
        return Complete(DrnChunkNative.ParallelForwardWithFloor(projected, output, checkpoints, state, ping, pong, mismatch,
            batch, sequence, width, key, value, passes, retentionFloor, stream), CudaNativeOperation.ForgetMemoryForward, device);
    }

    public static int DrnChunkForward(int device, nint projected, nint output, nint checkpoints,
        nint state, int batch, int sequence, int width, int key, int value, float retentionFloor, nint stream)
    {
        EnsureDrnRetentionFloorAbi(retentionFloor);
        return Complete(DrnChunkNative.ForwardWithFloor(projected, output, checkpoints, state,
            batch, sequence, width, key, value, retentionFloor, stream), CudaNativeOperation.ForgetMemoryForward, device);
    }

    public static int DrnChunkBackward(int device, nint projected, nint dp, nint dy,
        nint checkpoints, nint stateGradient, nint adjoints, int batch, int sequence,
        int width, int key, int value, float retentionFloor, nint stream)
    {
        EnsureDrnRetentionFloorAbi(retentionFloor);
        return Complete(DrnChunkNative.BackwardWithFloor(projected, dp, dy, checkpoints, stateGradient,
            adjoints, batch, sequence, width, key, value, retentionFloor, stream), CudaNativeOperation.ForgetMemoryBackward, device);
    }

    public static int DrnChunkParallelForward(int device, nint projected, nint output,
        nint checkpoints, nint state, nint ping, nint pong, nint mismatch,
        int batch, int sequence, int width, int key, int value, int passes, nint stream)
    {
        EnsureMinimumAbiMinor(CudaAbiVersion.DrnChunkBackwardMinor, "DRN parallel chunk forward");
        return Complete(DrnChunkNative.ParallelForward(projected,output,checkpoints,state,ping,pong,mismatch,
            batch,sequence,width,key,value,passes,stream),CudaNativeOperation.ForgetMemoryForward,device);
    }
    public static int DrnChunkForward(int device, nint projected, nint output, nint checkpoints,
        nint state, int batch, int sequence, int width, int key, int value, nint stream)
    {
        EnsureMinimumAbiMinor(CudaAbiVersion.DrnChunkBackwardMinor, "DRN chunk forward");
        return Complete(DrnChunkNative.Forward(projected, output, checkpoints, state,
            batch, sequence, width, key, value, stream), CudaNativeOperation.ForgetMemoryForward, device);
    }

    public static int DrnChunkBackward(int device, nint projected, nint dp, nint dy,
        nint checkpoints, nint stateGradient, nint adjoints, int batch, int sequence,
        int width, int key, int value, nint stream)
    {
        EnsureMinimumAbiMinor(CudaAbiVersion.DrnChunkBackwardMinor, "DRN chunk backward");
        return Complete(DrnChunkNative.Backward(projected, dp, dy, checkpoints, stateGradient,
            adjoints, batch, sequence, width, key, value, stream), CudaNativeOperation.ForgetMemoryBackward, device);
    }

    private static class DrnChunkNative
    {
        [DllImport(LibraryName, EntryPoint = "nntrain_drn_chunk_parallel_forward_with_floor", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ParallelForwardWithFloor(nint projected, nint output, nint checkpoints, nint state,
            nint ping, nint pong, nint mismatch, int batch, int sequence, int width, int key, int value, int passes, float retentionFloor, nint stream);
        [DllImport(LibraryName, EntryPoint = "nntrain_drn_chunk_forward_with_floor", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ForwardWithFloor(nint projected, nint output, nint checkpoints, nint state,
            int batch, int sequence, int width, int key, int value, float retentionFloor, nint stream);
        [DllImport(LibraryName, EntryPoint = "nntrain_drn_chunk_backward_with_floor", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BackwardWithFloor(nint projected, nint dp, nint dy, nint checkpoints,
            nint stateGradient, nint adjoints, int batch, int sequence, int width, int key, int value, float retentionFloor, nint stream);
        [DllImport(LibraryName, EntryPoint = "nntrain_drn_chunk_parallel_forward", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ParallelForward(nint projected,nint output,nint checkpoints,nint state,
            nint ping,nint pong,nint mismatch,int batch,int sequence,int width,int key,int value,int passes,nint stream);
        [DllImport(LibraryName, EntryPoint = "nntrain_drn_chunk_forward", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Forward(nint projected, nint output, nint checkpoints, nint state,
            int batch, int sequence, int width, int key, int value, nint stream);
        [DllImport(LibraryName, EntryPoint = "nntrain_drn_chunk_backward", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Backward(nint projected, nint dp, nint dy, nint checkpoints,
            nint stateGradient, nint adjoints, int batch, int sequence, int width, int key, int value, nint stream);
    }
}
