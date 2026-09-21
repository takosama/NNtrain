using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcWideGemmTests
{
    [Theory]
    [InlineData(false,false,0,0)]
    [InlineData(true,false,0,1)]
    [InlineData(false,true,1,2)]
    [InlineData(true,true,2,1)]
    [InlineData(false,true,3,0)]
    public void WideRegisterTilesKeepOrderedFloat32Math(bool ta,bool tb,int bf16,int gate)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int m=95,n=137,k=67;
        float[] av=Enumerable.Range(0,m*k).Select(i=>MathF.Sin(i*.317f)*.01f).ToArray();
        float[] bv=Enumerable.Range(0,k*n).Select(i=>MathF.Cos(i*.167f)*.01f).ToArray();
        float[] Run(bool wide){
            using var lane=new ArcExecutionLane(options:new(){WideMatrices=wide,XmxMatrices=false});
            using var a=lane.Upload(av);using var b=lane.Upload(bv);using var c=lane.Upload(Enumerable.Repeat(-.0001f,m*n).ToArray());
            using var bias=lane.Upload(Enumerable.Range(0,n).Select(i=>i*.00001f).ToArray());
            ArcMuonMath.Gemm(lane,a,b,c,m,n,k,ta,tb,bf16,true,bias,true,gate==1?a:b,gate);
            float[] actual=new float[m*n];lane.Read(c,actual);
            Assert.Contains(wide?"gemm_wide":"gemm_tiled",lane.KernelTimings.Keys);
            return actual;
        }
        Assert.Equal(Run(false),Run(true));
    }
}
