using System.Diagnostics;
using System.Text.Json;

namespace NNtrain.Benchmarks;

internal static class DrnChunkProbe
{
    internal static void Run(string outputPath, bool parallel = false, bool slowDecay = false, int passes = 8)
    {
        if (File.Exists(outputPath)) throw new IOException("Use a new result path.");
        if (!Tensor.IsCudaAvailable(0)) throw new InvalidOperationException("CUDA0 required.");
        using var scope = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cuda, 0));
        using var policy = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults);
        var device = ForgetMemoryV2Cuda.GetAccelerator(0);
        var results = new List<object>();
        foreach (var (batch, sequence, key, value) in new[]
            { (8,2048,32,32), (16,512,16,16), (2,33,16,32), (2,129,32,128), (2,1,32,32) })
        {
            int width = 2*key + 3*value;
            var rng = new Random(8301);
            float[] Values(int length) => Enumerable.Range(0, length)
                .Select(_ => (float)(rng.NextDouble() - .5) * .6f).ToArray();
            float[] projected = Values(batch*sequence*width);
            if (slowDecay)
                for (int t=0;t<batch*sequence;t++) for(int v=0;v<value;v++) {
                    projected[t*width+2*key+value+v]=8f;
                    projected[t*width+2*key+2*value+v]=-8f;
                }
            using var p = device.Allocate1D(projected.Select(TensorStorageCodec.EncodeBFloat16).ToArray());
            using var oldOutput = device.Allocate1D<ushort>(batch*sequence*value);
            using var newOutput = device.Allocate1D<ushort>(oldOutput.Length);
            using var history = device.Allocate1D<float>(batch*sequence*key*value);
            using var oldState = device.Allocate1D<float>(batch*key*value);
            using var newState = device.Allocate1D<float>(oldState.Length);
            using var checkpoints = device.Allocate1D<float>(CudaDrnChunk.CheckpointLength(batch,sequence,key,value));
            using var adjoints = device.Allocate1D<float>(CudaDrnChunk.AdjointLength(batch,sequence,key,value));
            using var ping = parallel ? device.Allocate1D<float>(checkpoints.Length) : null;
            using var pong = parallel ? device.Allocate1D<float>(checkpoints.Length) : null;
            using var mismatch = parallel ? device.Allocate1D<int>(1) : null;
            using var dy = device.Allocate1D(Values(oldOutput.Length));
            float[] initialDp = Values(p.Length), initialDs = Values(oldState.Length);
            using var oldDp = device.Allocate1D(initialDp);
            using var newDp = device.Allocate1D(initialDp);
            using var oldDs = device.Allocate1D(initialDs);
            using var newDs = device.Allocate1D(initialDs);
            using var previous = device.Allocate1D<float>(oldState.Length);
            int preparedLength = CudaForgetMemoryNative.PreparedBackwardScratchLength(batch,sequence,key,value);
            using var prepared = preparedLength > 0 ? device.Allocate1D<float>(preparedLength) : null;
            void OldForward()
            {
                oldState.MemSetToZero();
                if (!CudaForgetMemoryTensorCore.TryForward(device,p,oldOutput,history,oldState,batch,sequence,width,key,value,.37f,2))
                    throw new InvalidOperationException("Expected Tensor Core reference.");
            }
            void NewForward()
            {
                newState.MemSetToZero();
                if (parallel) CudaDrnChunk.ParallelForward(device,p,newOutput,checkpoints,newState,ping!,pong!,mismatch!,batch,sequence,width,key,value,passes,.37f);
                else CudaDrnChunk.Forward(device,p,newOutput,checkpoints,newState,batch,sequence,width,key,value,.37f);
            }
            void OldBackward() => CudaForgetMemoryNative.Backward(device,0,p.NativePtr,oldDp.NativePtr,dy.NativePtr,
                history.NativePtr,oldDs.NativePtr,previous.NativePtr,batch,sequence,width,key,value,.37f,2,true,prepared?.NativePtr??0);
            void NewBackward() => CudaDrnChunk.Backward(device,p,newDp,dy,checkpoints,newDs,adjoints,batch,sequence,width,key,value,.37f);
            OldForward(); NewForward(); OldBackward(); NewBackward(); device.Synchronize();
            var expectedOutput = new ushort[oldOutput.Length]; var actualOutput = new ushort[newOutput.Length];
            oldOutput.CopyToCPU(expectedOutput); newOutput.CopyToCPU(actualOutput);
            if (!expectedOutput.SequenceEqual(actualOutput)) throw new InvalidOperationException("Forward is not bit exact.");
            double Compare(NativeCudaBuffer<float> a, NativeCudaBuffer<float> b)
            {
                var x = new float[a.Length]; var y = new float[b.Length]; a.CopyToCPU(x); b.CopyToCPU(y);
                double maximum = 0;
                for (int i=0;i<x.Length;i++) {
                    if (!float.IsFinite(y[i])) throw new InvalidOperationException("Nonfinite result.");
                    maximum = Math.Max(maximum, Math.Abs((double)x[i]-y[i]));
                }
                return maximum;
            }
            double stateError = Compare(oldState,newState), gradientError = Compare(oldDp,newDp), adjointError = Compare(oldDs,newDs);
            if (stateError != 0 || gradientError > 6e-5 || adjointError > 6e-5)
                throw new InvalidOperationException($"K{key} V{value} T{sequence}: state={stateError}, gradient={gradientError}, adjoint={adjointError}.");
            double Measure(Action action)
            {
                for(int i=0;i<5;i++) action(); device.Synchronize(); var timer=Stopwatch.StartNew();
                for(int i=0;i<20;i++) action(); device.Synchronize(); return timer.Elapsed.TotalMilliseconds/20;
            }
            double oldForwardMs=Measure(OldForward), newForwardMs=Measure(NewForward);
            double oldBackwardMs=Measure(() => { OldForward(); oldDp.MemSetToZero(); oldDs.MemSetToZero(); OldBackward(); });
            double newBackwardMs=Measure(() => { newDp.MemSetToZero(); newDs.MemSetToZero(); NewBackward(); });
            var fallback = new int[1]; mismatch?.CopyToCPU(fallback);
            var result = new {Batch=batch,Sequence=sequence,Key=key,Value=value,ForwardBitExact=true,
                Parallel=parallel,Passes=passes,SlowDecay=slowDecay,UsedFallback=fallback[0]!=0,
                StateError=stateError,GradientError=gradientError,AdjointError=adjointError,
                OldForwardMs=oldForwardMs,NewForwardMs=newForwardMs,OldRecomputeBackwardMs=oldBackwardMs,NewFusedBackwardMs=newBackwardMs,
                HistoryBytes=(long)history.Length*4,CheckpointBytes=(long)checkpoints.Length*4,AdjointBytes=(long)adjoints.Length*4};
            results.Add(result); Console.WriteLine(JsonSerializer.Serialize(result));
        }
        using var file = new FileStream(outputPath,FileMode.CreateNew);
        JsonSerializer.Serialize(file,results,new JsonSerializerOptions{WriteIndented=true});
    }
}
