using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcExpandedTileTests
{
    [Theory]
    [InlineData(16384,1536,512,false,true,false,"16x64_wg16")]
    [InlineData(8192,512,1536,false,false,true,"16x64_wg16")]
    [InlineData(1536,512,16384,true,false,true,"32x32_wg16")]
    [InlineData(512,11500,512,false,true,false,"16x64_wg4")]
    [InlineData(512,512,512,false,false,false,"16x32_wg16")]
    [InlineData(512,512,11500,false,false,true,"8x32_wg16")]
    [InlineData(512,512,8192,true,false,false,"16x32_wg16")]
    public void TileSelectionKeepsSmallAndLongReductionFallbacks(int m,int n,int k,bool ta,bool tb,bool accumulate,string suffix)
    {
        var plan=ArcXmxStorageOperand.SelectPanelTile(true,m,n,k,ta,tb,accumulate);
        Assert.Equal("gemm_xmx_direct_block_"+suffix,plan.Kernel);
    }

    [Theory]
    [InlineData(8193,512,79,false,true,TensorDType.BFloat16)]
    [InlineData(8193,512,79,false,true,TensorDType.Bfp8)]
    [InlineData(513,513,4097,true,false,TensorDType.BFloat16)]
    [InlineData(513,513,4097,true,false,TensorDType.Bfp8)]
    [InlineData(512,4097,79,false,true,TensorDType.BFloat16)]
    [InlineData(512,4097,79,false,true,TensorDType.Bfp8)]
    public void PackedDispatchPreservesEveryBitWithTailsAndRepeatedAccumulation(int m,int n,int k,bool ta,bool tb,TensorDType dtype)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(),"Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx||ArcDevices.Enumerate()[0].MinimumSubgroupSize!=16,"SG16 XMX is required.");
        float[] Run(bool enabled)
        {
            using var scope=Tensor.BeginArcExecution(precision:TensorPrecisionMode.Mix8_32,options:new(){ExpandedXmxTiles=enabled});
            var lane=Tensor.ArcLane;
            Tensor Make(int length,int seed){var t=new Tensor(Enumerable.Range(0,length).Select(i=>MathF.Sin(i*.017f+seed)*.013f).ToArray(),[length]);
                t.ConvertStorageInPlace(dtype,dtype==TensorDType.Bfp8?Bfp8QuantizationDescriptor.Block(32):null);return t;}
            var x=Make(m*k,1);var w=Make(n*k,3);
            using var a=x.ArcMatrixOperand();using var b=w.ArcMatrixOperand();
            using var output=lane.Upload(Enumerable.Repeat(.00017f,m*n).ToArray());
            long h2d=lane.H2DBytes,d2h=lane.D2HBytes;
            for(int i=0;i<2;i++)Assert.True(ArcXmxStorageOperand.TryGemm(lane,a,b,output,m,n,k,ta,tb,accumulate:true));
            lane.Synchronize();Assert.Equal(h2d,lane.H2DBytes);Assert.Equal(d2h,lane.D2HBytes);
            Assert.Contains(ArcXmxStorageOperand.SelectPanelTile(enabled,m,n,k,ta,tb,true).Kernel,lane.KernelTimings.Keys);
            float[] result=new float[m*n];lane.Read(output,result);return result;
        }
        Assert.Equal(Run(false),Run(true));
    }
}
