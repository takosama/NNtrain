using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcXmxTests
{
    [Theory]
    [InlineData(false,false,35,67,65)]
    [InlineData(false,true,35,67,65)]
    [InlineData(true,false,35,67,65)]
    [InlineData(true,true,35,67,65)]
    [InlineData(false,true,7,19,2061)]
    public void XmxCoversTransposeTailsGateBiasAndSplitK(bool ta,bool tb,int m,int n,int k)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx, "Intel matrix extension is required.");
        using var lane = new ArcExecutionLane();
        float[] av=Enumerable.Range(0,m*k).Select(i=>(i%17-8)/16f).ToArray();
        float[] bv=Enumerable.Range(0,k*n).Select(i=>(i%19-9)/16f).ToArray();
        float[] bias=Enumerable.Range(0,n).Select(i=>(i%7-3)/16f).ToArray();
        using var a=lane.Upload(av);using var b=lane.Upload(bv);using var c=lane.Upload(Enumerable.Repeat(.125f,m*n).ToArray());using var bs=lane.Upload(bias);
        long uploads=lane.H2DBytes,downloads=lane.D2HBytes;
        ArcMuonMath.Gemm(lane,a,b,c,m,n,k,ta,tb,3,true,bs,true,a,1);
        Assert.Equal(uploads,lane.H2DBytes);Assert.Equal(downloads,lane.D2HBytes);
        float[] actual=new float[m*n];lane.Read(c,actual);
        Assert.Contains("gemm_xmx",lane.KernelTimings.Keys);
        for(int row=0;row<m;row++)for(int col=0;col<n;col++){
            float expected=.125f+bias[col];
            for(int j=0;j<k;j++)expected+=Math.Max(0,av[ta?j*m+row:row*k+j])*bv[tb?col*k+j:j*n+col];
            Assert.Equal(Math.Max(0,expected),actual[row*n+col]);
        }
    }
}
