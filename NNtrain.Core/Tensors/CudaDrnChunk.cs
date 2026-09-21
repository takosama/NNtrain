using NNtrain.Cuda.Interop;

namespace NNtrain;

internal static class CudaDrnChunk
{
    internal const int Size = 16;
    internal static bool CanUse(int batch, int sequence, int key, int value)
    {
        CudaDispatchPolicy policy = CudaDispatchPolicy.Current;
        return AutogradContext.IsRecordingEnabled && !policy.DisableDrnChunkBackward
            && !policy.DisableDrnStateRecomputation && !policy.DisableTensorCoreForgetMemory
            && batch > 0 && sequence >= 64 && key is 16 or 32 && value is > 0 and <= 128
            && value % 16 == 0
            && (long)batch * Count(sequence) * key * value * sizeof(float) <= 8L * 1024 * 1024
            && CudaNativeGateway.AbiVersion.Minor >= CudaAbiVersion.DrnRetentionFloorMinor;
    }
    internal static int Count(int sequence) => checked((sequence - 1) / Size + 1);
    internal static int CheckpointLength(int batch, int sequence, int key, int value)
        => checked(batch * Count(sequence) * key * value);
    internal static int AdjointLength(int batch, int sequence, int key, int value)
        => checked(batch * (Count(sequence) + 1) * key * value);

    internal static void Forward(NativeCudaDevice device, NativeCudaBuffer<ushort> projected,
        NativeCudaBuffer<ushort> output, NativeCudaBuffer<float> checkpoints, NativeCudaBuffer<float> state,
        int batch, int sequence, int width, int key, int value, float retentionFloor = 0f)
    {
        device.Bind();
        if (sequence >= 512 && !CudaDispatchPolicy.Current.DisableDrnChunkParallelForward)
        {
            using var ping = device.Allocate1D<float>(checkpoints.Length, NNtrain.Cuda.Memory.CudaMemoryKind.Transient);
            using var pong = device.Allocate1D<float>(checkpoints.Length, NNtrain.Cuda.Memory.CudaMemoryKind.Transient);
            using var mismatch = device.Allocate1D<int>(1, NNtrain.Cuda.Memory.CudaMemoryKind.Transient);
            ParallelForward(device,projected,output,checkpoints,state,ping,pong,mismatch,batch,sequence,width,key,value,
                retentionFloor: retentionFloor);
            if (!TensorExecutionContext.TryGetCudaStreamLane(device.Index, out _)) device.Synchronize();
            return;
        }
        NativeCudaRuntime.Check(CudaNativeGateway.DrnChunkForward(device.Index, projected.NativePtr,
            output.NativePtr, checkpoints.NativePtr, state.NativePtr, batch, sequence, width, key, value,
            retentionFloor, device.DefaultStream), "DRN checkpoint forward");
    }

    internal static void ParallelForward(NativeCudaDevice device, NativeCudaBuffer<ushort> projected,
        NativeCudaBuffer<ushort> output, NativeCudaBuffer<float> checkpoints, NativeCudaBuffer<float> state,
        NativeCudaBuffer<float> ping, NativeCudaBuffer<float> pong, NativeCudaBuffer<int> mismatch,
        int batch, int sequence, int width, int key, int value, int passes = 8, float retentionFloor = 0f)
    {
        device.Bind();
        NativeCudaRuntime.Check(CudaNativeGateway.DrnChunkParallelForward(device.Index,projected.NativePtr,
            output.NativePtr,checkpoints.NativePtr,state.NativePtr,ping.NativePtr,pong.NativePtr,mismatch.NativePtr,
            batch,sequence,width,key,value,passes,retentionFloor,device.DefaultStream),"DRN parallel chunk forward");
    }

    internal static void Backward(NativeCudaDevice device, NativeCudaBuffer<ushort> projected,
        NativeCudaBuffer<float> dp, NativeCudaBuffer<float> dy, NativeCudaBuffer<float> checkpoints,
        NativeCudaBuffer<float> stateGradient, NativeCudaBuffer<float> adjoints,
        int batch, int sequence, int width, int key, int value, float retentionFloor = 0f)
    {
        device.Bind();
        NativeCudaRuntime.Check(CudaNativeGateway.DrnChunkBackward(device.Index, projected.NativePtr,
            dp.NativePtr, dy.NativePtr, checkpoints.NativePtr, stateGradient.NativePtr, adjoints.NativePtr,
            batch, sequence, width, key, value, retentionFloor, device.DefaultStream), "DRN chunk fused backward");
    }
}
