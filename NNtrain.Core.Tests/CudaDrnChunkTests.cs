using NNtrain;
using Xunit;

public sealed class CudaDrnChunkTests
{
    [Theory]
    [InlineData(1, 1, 16, 16)]
    [InlineData(2, 17, 16, 32)]
    [InlineData(2, 65, 32, 32)]
    [InlineData(8, 257, 32, 32)]
    [InlineData(4, 1025, 32, 32)]
    [InlineData(3, 129, 16, 128)]
    public void CheckpointsAndFusedBackwardMatchReference(int batch, int sequence, int key, int value)
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA is unavailable.");
        using var scope = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cuda, 0));
        using var policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults);
        var device = ForgetMemoryV2Cuda.GetAccelerator(0);
        int width = 2*key + 3*value, matrix = key*value;
        var rng = new Random(8301);
        float[] Values(int length) => Enumerable.Range(0,length).Select(_ => (float)(rng.NextDouble()-.5)*.6f).ToArray();
        using var p = device.Allocate1D(Values(batch*sequence*width).Select(TensorStorageCodec.EncodeBFloat16).ToArray());
        using var oldOutput = device.Allocate1D<ushort>(batch*sequence*value);
        using var newOutput = device.Allocate1D<ushort>(oldOutput.Length);
        using var history = device.Allocate1D<float>(batch*sequence*matrix);
        using var oldState = device.Allocate1D<float>(batch*matrix);
        using var newState = device.Allocate1D<float>(oldState.Length);
        using var checkpoints = device.Allocate1D<float>(CudaDrnChunk.CheckpointLength(batch,sequence,key,value));
        using var adjoints = device.Allocate1D<float>(CudaDrnChunk.AdjointLength(batch,sequence,key,value));
        oldState.MemSetToZero(); newState.MemSetToZero();
        Assert.True(CudaForgetMemoryTensorCore.TryForward(device,p,oldOutput,history,oldState,batch,sequence,width,key,value,.37f,2));
        CudaDrnChunk.Forward(device,p,newOutput,checkpoints,newState,batch,sequence,width,key,value,retentionFloor:.37f);
        var expectedOutput = new ushort[oldOutput.Length]; var actualOutput = new ushort[newOutput.Length];
        oldOutput.CopyToCPU(expectedOutput); newOutput.CopyToCPU(actualOutput);
        Assert.Equal(expectedOutput, actualOutput);
        Compare(oldState,newState,0);
        var full = new float[history.Length]; var boundaries = new float[checkpoints.Length];
        history.CopyToCPU(full); checkpoints.CopyToCPU(boundaries);
        for (int b=0;b<batch;b++)
        for (int c=0;c<CudaDrnChunk.Count(sequence);c++)
        for (int i=0;i<matrix;i++)
            Assert.Equal(c == 0 ? 0f : full[(b*sequence+c*CudaDrnChunk.Size-1)*matrix+i],
                boundaries[(b*CudaDrnChunk.Count(sequence)+c)*matrix+i]);
        using var dy = device.Allocate1D(Values(oldOutput.Length));
        float[] initialDp = Values(p.Length), terminal = Values(oldState.Length);
        using var oldDp = device.Allocate1D(initialDp); using var newDp = device.Allocate1D(initialDp);
        using var oldDs = device.Allocate1D(terminal); using var newDs = device.Allocate1D(terminal);
        using var previous = device.Allocate1D<float>(oldState.Length);
        CudaForgetMemoryNative.Backward(device,0,p.NativePtr,oldDp.NativePtr,dy.NativePtr,
            history.NativePtr,oldDs.NativePtr,previous.NativePtr,batch,sequence,width,key,value,.37f,2,true);
        CudaDrnChunk.Backward(device,p,newDp,dy,checkpoints,newDs,adjoints,batch,sequence,width,key,value,retentionFloor:.37f);
        device.Synchronize();
        Compare(oldDp,newDp,6e-5f); Compare(oldDs,newDs,6e-5f);
    }

    [Fact]
    public void UnsupportedShapesAndExplicitFallbackDoNotSelectChunkKernel()
    {
        using var policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults);
        Assert.False(CudaDrnChunk.CanUse(8,2048,17,32));
        Assert.False(CudaDrnChunk.CanUse(8,2048,32,19));
        Assert.False(CudaDrnChunk.CanUse(2,1,32,32));
        Assert.False(CudaDrnChunk.CanUse(8,2048,32,128)); // Per-layer boundary budget.
        using var disabled = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults with { DisableDrnChunkBackward = true });
        Assert.False(CudaDrnChunk.CanUse(8,2048,32,32));
    }

    [Theory]
    [InlineData(false, 8, 0f)]
    [InlineData(true, 2, 0f)]
    [InlineData(false, 8, .37f)]
    [InlineData(false, 8, .99f)]
    [InlineData(true, 2, .37f)]
    [InlineData(true, 2, .99f)]
    public void ParallelForwardChecksExactnessAndResetsBetweenCalls(bool slowDecay, int passes, float retentionFloor)
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA is unavailable.");
        using var scope = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cuda, 0));
        using var policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults);
        var device = ForgetMemoryV2Cuda.GetAccelerator(0);
        const int batch=2, sequence=513, key=32, value=32, width=160;
        using var p=device.Allocate1D<ushort>(batch*sequence*width);
        using var oldOutput=device.Allocate1D<ushort>(batch*sequence*value);
        using var newOutput=device.Allocate1D<ushort>(oldOutput.Length);
        using var history=device.Allocate1D<float>(batch*sequence*key*value);
        using var oldState=device.Allocate1D<float>(batch*key*value);
        using var newState=device.Allocate1D<float>(oldState.Length);
        using var checkpoints=device.Allocate1D<float>(CudaDrnChunk.CheckpointLength(batch,sequence,key,value));
        using var ping=device.Allocate1D<float>(checkpoints.Length);
        using var pong=device.Allocate1D<float>(checkpoints.Length);
        using var mismatch=device.Allocate1D<int>(1);
        for(int iteration=0;iteration<3;iteration++) {
            var rng=new Random(910+iteration);
            float[] values=Enumerable.Range(0,p.Length).Select(_=>(float)(rng.NextDouble()-.5)*.6f).ToArray();
            if(slowDecay) for(int t=0;t<batch*sequence;t++) for(int v=0;v<value;v++) {
                values[t*width+2*key+value+v]=8f;
                values[t*width+2*key+2*value+v]=-8f;
            }
            p.CopyFromCPU(values.Select(TensorStorageCodec.EncodeBFloat16).ToArray());
            oldState.MemSetToZero(); newState.MemSetToZero();
            Assert.True(CudaForgetMemoryTensorCore.TryForward(device,p,oldOutput,history,oldState,batch,sequence,width,key,value,retentionFloor,2));
            CudaDrnChunk.ParallelForward(device,p,newOutput,checkpoints,newState,ping,pong,mismatch,
                batch,sequence,width,key,value,passes,retentionFloor);
            var expected=new ushort[oldOutput.Length]; var actual=new ushort[newOutput.Length];
            oldOutput.CopyToCPU(expected); newOutput.CopyToCPU(actual);
            Assert.Equal(expected,actual); Compare(oldState,newState,0f);
            var flag=new int[1]; mismatch.CopyToCPU(flag);
            if(slowDecay) Assert.Equal(1,flag[0]);
            else if(retentionFloor == 0f) Assert.Equal(0,flag[0]);
            // Higher retention can legitimately require exact serial fallback.
            // The bitwise output/state comparisons above remain mandatory.
            else Assert.InRange(flag[0],0,1);
        }
    }

    private static void Compare(NativeCudaBuffer<float> a, NativeCudaBuffer<float> b, float tolerance)
    {
        var expected = new float[a.Length]; var actual = new float[b.Length];
        a.CopyToCPU(expected); b.CopyToCPU(actual);
        for (int i=0;i<actual.Length;i++) {
            Assert.True(float.IsFinite(actual[i]));
            Assert.InRange(Math.Abs(actual[i]-expected[i]),0f,tolerance);
        }
    }
}
