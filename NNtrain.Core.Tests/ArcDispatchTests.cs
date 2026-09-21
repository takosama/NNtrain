using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcDispatchTests
{
    [Fact]
    public void DeferredReleaseHasBoundedPhysicalLifetimeAndProfilesWithoutReadback()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        // Exercise the legacy budget-triggered fence, not early same-queue
        // retired reuse (covered separately by ArcRetiredBufferReuseTests).
        using var lane=new ArcExecutionLane(options:new(){BufferPoolBytes=0,DeferredReleaseBytes=4096,DetailedProfiling=true,ReuseRetiredBuffers=false});
        lane.DetailedProfiler!.Phase="test";
        using var result=lane.Upload(new float[131]);
        using var one=lane.Upload(Enumerable.Repeat(1f,131).ToArray());
        lane.ResetDetailedProfile();
        for(int i=0;i<30;i++){
            // Build temporaries on the device: explicit blocking uploads would
            // finish earlier commands before the retirement budget is reached.
            using var tmp=lane.Allocate(131);
            lane.Run("copy_scale",131,0,one,tmp,131,(float)i,0);
            lane.Run("copy_scale",131,0,tmp,result,131,1f,1);
            tmp.Dispose();
            Assert.InRange(lane.RetiredBytes,0,4096);
        }
        Assert.True(lane.RetiredBytes>0);
        Assert.Throws<InvalidOperationException>(lane.ResetDetailedProfile);
        Assert.Equal(0,lane.D2HBytes);
        float[] values=new float[131];lane.Read(result,values);
        Assert.All(values,v=>Assert.Equal(435f,v));Assert.Equal(0,lane.RetiredBytes);
        var entries=lane.DetailedProfiler.Snapshot();
        Assert.Equal(60,entries.Where(e=>e.Kind=="gpu-kernel").Sum(e=>e.Count));
        Assert.Contains(entries,e=>e.Kind=="synchronize"&&e.Detail.EndsWith("pool-budget-release"));
        Assert.Equal(524,entries.Where(e=>e.Kind=="D2H").Sum(e=>e.Bytes));
        Assert.Equal(0,entries.Where(e=>e.Kind=="H2D").Sum(e=>e.Bytes));
        lane.ResetDetailedProfile();Assert.Empty(lane.DetailedProfiler.Snapshot());
    }
    [Fact]
    public void ProfileIncludesStagedHostResultsAndAllocationUploads()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var lane = new ArcExecutionLane(options: new() { DetailedProfiling = true });
        float[] left = [1, 2, 3], right = [4, 5, 6], result = new float[3];
        lane.Run("add", 3, 0, ArcExecutionLane.In(left), ArcExecutionLane.In(right), ArcExecutionLane.Out(result), 3);
        Assert.Equal(new float[] { 5, 7, 9 }, result);
        var entries = lane.DetailedProfiler!.Snapshot();
        Assert.Equal(lane.H2DBytes, entries.Where(e => e.Kind == "H2D").Sum(e => e.Bytes));
        Assert.Equal(lane.D2HBytes, entries.Where(e => e.Kind == "D2H").Sum(e => e.Bytes));
        Assert.Equal(24, lane.H2DBytes); Assert.Equal(12, lane.D2HBytes);
    }
    [Theory]
    [InlineData(0)]
    [InlineData(1048576)]
    public void QueuedPoolReuseCopiesAndFailureKeepOwnership(long pool)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var lane = new ArcExecutionLane(options:new(){BufferPoolBytes=pool});
        const int n=131;
        using var one=lane.Upload(Enumerable.Repeat(1f,n).ToArray());using var result=lane.Upload(new float[n]);
        long uploaded=lane.H2DBytes;
        for(int i=0;i<300;i++){
            using var tmp=lane.Allocate(n);using var copy=lane.Allocate(n);
            lane.Run("copy_scale",n,0,one,tmp,n,(float)i,0);
            lane.CopyBytes(tmp,copy,0,0,n*4);
            lane.Run("add",n,0,copy,result,result,n);
        }
        // Invalid dispatch must fence already queued work without losing events.
        Assert.Throws<ArgumentException>(()=>lane.Run("copy_scale",n,0,one,result,"invalid"));
        float[] actual=new float[n];lane.Read(result,actual);
        Assert.All(actual,x=>Assert.Equal(44850f,x));
        Assert.Equal(uploaded,lane.H2DBytes);Assert.Equal(n*4,lane.D2HBytes);
        Assert.Equal(n*8,lane.AllocatedBytes);Assert.Equal(600,lane.KernelLaunchCount);
        Assert.True(lane.KernelMilliseconds>0);
    }
}
