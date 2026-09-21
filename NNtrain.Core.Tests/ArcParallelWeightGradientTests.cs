using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcParallelWeightGradientTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void ParallelKMatchesSequentialWithAccumulationGateAndTails(int gate)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(),"Intel Arc is required.");
        const int m=95,n=137,k=4101;
        var av=Enumerable.Range(0,m*k).Select(i=>MathF.Sin(i*.019f)*.02f).ToArray();
        var bv=Enumerable.Range(0,n*k).Select(i=>MathF.Cos(i*.037f)*.03f).ToArray();
        float[] Run(bool parallel){
            using var lane=new ArcExecutionLane(options:new(){ParallelWeightGradients=parallel});
            using var a=lane.Upload(av);using var b=lane.Upload(bv);using var c=lane.Upload(Enumerable.Repeat(.001f,m*n).ToArray());
            long uploaded=lane.H2DBytes;
            for(int step=0;step<3;step++)ArcMuonMath.Gemm(lane,a,b,c,m,n,k,ta:true,accumulate:true,gate:gate==1?a:b,gateOperand:gate);
            Assert.Equal(uploaded,lane.H2DBytes);Assert.Equal(0,lane.D2HBytes);
            float[] output=new float[m*n];lane.Read(c,output);
            Assert.Contains(parallel?"gemm_split":"gemm_wide",lane.KernelTimings.Keys);
            return output;
        }
        var reference=Run(false);var actual=Run(true);
        for(int i=0;i<actual.Length;i++)Assert.InRange(MathF.Abs(reference[i]-actual[i]),0,1e-5f);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DeepGemmMatchesOrderedBaselineWithTails(bool transpose)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(),"Intel Arc is required.");
        const int m=1025,n=67,k=97;
        var av=Enumerable.Range(0,m*k).Select(i=>MathF.Sin(i*.019f)*.02f).ToArray();
        var bv=Enumerable.Range(0,n*k).Select(i=>MathF.Cos(i*.037f)*.03f).ToArray();
        float[] Run(bool deep){
            using var lane=new ArcExecutionLane(options:new(){DeepMatrices=deep,XmxMatrices=false});
            using var a=lane.Upload(av);using var b=lane.Upload(bv);using var c=lane.Upload(new float[m*n]);
            ArcMuonMath.Gemm(lane,a,b,c,m,n,k,ta:transpose,accumulate:true,gate:a,gateOperand:1);
            float[] output=new float[m*n];lane.Read(c,output);
            Assert.Contains(deep?"gemm_deep":"gemm_wide",lane.KernelTimings.Keys);
            return output;
        }
        Assert.Equal(Run(false),Run(true));
    }
}
