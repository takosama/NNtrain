using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcXmxTuningTests
{
    [Theory]
    [InlineData(ArcXmxGemmMode.Narrow)]
    [InlineData(ArcXmxGemmMode.Wide)]
    [InlineData(ArcXmxGemmMode.Auto)]
    public void PairedSlmMatchesLegacyWithTailsTransposesAndEpilogues(ArcXmxGemmMode mode)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx, "Intel XMX is required.");
        foreach (var (m,n,k,ta,tb) in new[] { (137,131,67,false,false), (137,71,65,false,true),
            (67,71,4101,true,false), (67,131,67,true,true), (257,2049,35,false,true) })
        foreach (int gateOperand in new[] { 0, 1, 2 })
        {
            float[] av=Enumerable.Range(0,m*k).Select(i=>(i%17-8)/32f).ToArray();
            float[] bv=Enumerable.Range(0,k*n).Select(i=>(i%19-9)/32f).ToArray();
            float[] Run(ArcXmxGemmMode selected)
            {
                // Compare the SLM plans, not the production direct-panel route
                // which takes precedence over XmxGemmMode when enabled.
                using var lane=new ArcExecutionLane(options:new(){XmxGemmMode=selected,DirectXmxMatrices=false});
                using var a=lane.Upload(av); using var b=lane.Upload(bv);
                using var c=lane.Upload(Enumerable.Repeat(.125f,m*n).ToArray());
                using var bias=lane.Upload(Enumerable.Range(0,n).Select(i=>(i%7-3)/32f).ToArray());
                long h2d=lane.H2DBytes,d2h=lane.D2HBytes;
                for(int repeat=0;repeat<2;repeat++)
                    ArcMuonMath.Gemm(lane,a,b,c,m,n,k,ta,tb,3,true,gateOperand==0?null:bias,
                        gateOperand!=0,gateOperand==2?b:a,gateOperand);
                Assert.Equal(h2d,lane.H2DBytes); Assert.Equal(d2h,lane.D2HBytes);
                float[] output=new float[m*n];lane.Read(c,output);return output;
            }
            Assert.Equal(Run(ArcXmxGemmMode.Legacy),Run(mode));
        }
    }

    [Theory]
    [InlineData(16384,1536,false,true,"gemm_xmx_wide_nt",256,128)]
    [InlineData(1536,512,true,false,"gemm_xmx_wide_tn",256,128)]
    [InlineData(512,1536,true,false,"gemm_xmx_wide_tn",256,128)]
    [InlineData(512,512,false,false,"gemm_xmx_narrow_nn",128,64)]
    [InlineData(11500,512,true,false,"gemm_xmx_wide_tn",256,128)]
    [InlineData(512,11500,false,true,"gemm_xmx_wide_nt",256,128)]
    [InlineData(128,64,false,false,"gemm_xmx_narrow_nn",128,64)]
    [InlineData(127,64,false,false,"gemm_xmx",128,64)]
    [InlineData(512,63,false,false,"gemm_xmx",128,64)]
    public void AutomaticTileSelectionIsShapeBounded(int m,int n,bool ta,bool tb,string kernel,int rows,int columns)
    {
        Assert.Equal((kernel,rows,columns),ArcMuonMath.SelectXmxPlan(ArcXmxGemmMode.Auto,16,m,n,ta,tb));
        Assert.Equal(("gemm_xmx",128,64),ArcMuonMath.SelectXmxPlan(ArcXmxGemmMode.Auto,8,m,n,ta,tb));
        Assert.Equal(("gemm_xmx",128,64),ArcMuonMath.SelectXmxPlan(ArcXmxGemmMode.Legacy,16,m,n,ta,tb));
    }

    [Theory]
    [InlineData(ArcXmxGemmMode.Narrow)]
    [InlineData(ArcXmxGemmMode.Wide)]
    public void PanelRoundingMatchesLegacyWithoutHostPacking(ArcXmxGemmMode mode)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx, "Intel XMX is required.");
        const int m=257,n=131,k=67;
        float[] a=Enumerable.Range(0,m*k).Select(i=>(i%31-15)/127f).ToArray();
        float[] b=Enumerable.Range(0,k*n).Select(i=>(i%37-18)/137f).ToArray();
        foreach(bool ta in new[]{false,true}) foreach(bool tb in new[]{false,true})
        {
            float[] Run(ArcXmxGemmMode selected)
            {
                using var lane=new ArcExecutionLane(options:new(){XmxGemmMode=selected,DirectXmxMatrices=false});
                using var aa=lane.Upload(a);using var bb=lane.Upload(b);using var cc=lane.Allocate(m*n);
                long up=lane.H2DBytes,down=lane.D2HBytes;
                ArcMuonMath.Gemm(lane,aa,bb,cc,m,n,k,ta,tb,3);
                Assert.Equal(up,lane.H2DBytes);Assert.Equal(down,lane.D2HBytes);
                float[] result=new float[m*n];lane.Read(cc,result);
                var plan=ArcMuonMath.SelectXmxPlan(selected,lane.Device.MinimumSubgroupSize,m,n,ta,tb);
                var resources=lane.GetKernelResources(plan.Kernel);
                Assert.True(resources.MaximumWorkGroupSize >= (ulong)(lane.Device.MinimumSubgroupSize*plan.Rows/8));
                Assert.True(resources.LocalMemoryBytes > 0);
                Assert.Contains(plan.Kernel,lane.KernelTimings.Keys);
                return result;
            }
            Assert.Equal(Run(ArcXmxGemmMode.Legacy),Run(mode));
        }
    }

    [Theory]
    [InlineData(ArcXmxGemmMode.Narrow)]
    [InlineData(ArcXmxGemmMode.Wide)]
    public void LongReductionWithVaryingExponentsMatchesReferenceTolerance(ArcXmxGemmMode mode)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx, "Intel XMX is required.");
        const int m=129,n=131,k=4101;
        float[] a=Enumerable.Range(0,m*k).Select(i=>MathF.ScaleB(MathF.Sin(i*.019f)*.1f,-(i%8))).ToArray();
        float[] b=Enumerable.Range(0,k*n).Select(i=>MathF.ScaleB(MathF.Cos(i*.037f)*.1f,-(i%7))).ToArray();
        foreach(bool ta in new[]{false,true}) foreach(bool tb in new[]{false,true})
        {
            float[] Run(ArcXmxGemmMode selected)
            {
                using var lane=new ArcExecutionLane(options:new(){XmxGemmMode=selected,DirectXmxMatrices=false});
                using var aa=lane.Upload(a);using var bb=lane.Upload(b);
                using var cc=lane.Upload(Enumerable.Repeat(.001f,m*n).ToArray());
                for(int step=0;step<2;step++)ArcMuonMath.Gemm(lane,aa,bb,cc,m,n,k,ta,tb,3,accumulate:true);
                float[] result=new float[m*n];lane.Read(cc,result);return result;
            }
            var expected=Run(ArcXmxGemmMode.Legacy);var actual=Run(mode);
            // Same existing absolute tolerance as the Arc mixed-matrix CPU reference tests.
            for(int i=0;i<actual.Length;i++)Assert.InRange(MathF.Abs(expected[i]-actual[i]),0,1e-5f);
        }
    }
}
