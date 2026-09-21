using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcTiledAttentionTests
{
    [Theory]
    [InlineData(false, 31, 65)]
    [InlineData(true, 31, 65)]
    [InlineData(false, 32, 137)]
    [InlineData(true, 32, 137)]
    [InlineData(false, 16, 67)]
    [InlineData(true, 16, 67)]
    [InlineData(false, 65, 67)]
    [InlineData(true, 65, 67)]
    public void CompactBackwardTilesPreserveExactValuesAndAccumulatedGradients(bool causal, int headWidth, int sequence)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int batch = 2, heads = 3;
        int width = heads * headWidth;
        var values = Enumerable.Range(0, batch * sequence * width * 3).Select(i => MathF.Sin(i * .013f) * .1f).ToArray();
        var seed = Enumerable.Range(0, batch * sequence * width).Select(i => MathF.Cos(i * .017f) * .01f).ToArray();
        (float[] Value, float[] Grad) Run(bool compact, bool unroll = false)
        {
            using var scope = Tensor.BeginArcExecution(options: new() { CompactAttentionTiles = compact, UnrolledAttentionTiles = unroll, PanelAttention = unroll });
            var x = new Tensor((float[])values.Clone(), [batch, sequence, width * 3]);
            x.ConvertStorageInPlace(TensorDType.BFloat16);
            var y = x.FusedMultiHeadAttention(heads, causal);
            float[] output = y.Data.ToArray();
            y.BackwardAndRelease(seed);
            x.FusedMultiHeadAttention(heads, causal).BackwardAndRelease(seed);
            return (output, x.Grad.ToArray());
        }
        var expected = Run(false); var actual = Run(true);
        Assert.Equal(expected.Value, actual.Value);
        Assert.Equal(expected.Grad, actual.Grad);
        var unrolled = Run(true, true);
        Assert.Equal(expected.Value, unrolled.Value);
        Assert.Equal(expected.Grad, unrolled.Grad);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HeadBatchWorkspacePreservesValuesAndGradients(bool causal)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int batch=3,heads=3,width=15,sequence=513;
        var values=Enumerable.Range(0,batch*sequence*width*3).Select(i=>MathF.Sin(i*.013f)*.1f).ToArray();
        var seed=Enumerable.Range(0,batch*sequence*width).Select(i=>MathF.Cos(i*.017f)*.01f).ToArray();
        (float[] Value,float[] Grad,long Launches) Run(int budget){
            using var scope=Tensor.BeginArcExecution(options:new(){AttentionWorkspaceMiB=budget});
            var x=new Tensor((float[])values.Clone(),[batch,sequence,width*3]);
            x.ConvertStorageInPlace(TensorDType.BFloat16);
            var y=x.FusedMultiHeadAttention(heads,causal);
            float[] output=y.Data.ToArray();y.BackwardAndRelease(seed);
            return(output,x.Grad.ToArray(),Tensor.ArcLane.KernelLaunchCount);
        }
        var small=Run(8);var large=Run(128);
        Assert.Equal(small.Value,large.Value);Assert.Equal(small.Grad,large.Grad);
        Assert.True(large.Launches<small.Launches);
    }
    [Theory]
    [InlineData(5,65)]
    [InlineData(33,137)]
    [InlineData(65,67)]
    public void CausalBoundsMatchUnboundedBatchedForNarrowAndWideHeads(int headWidth,int sequence)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(),"Intel Arc is required.");
        const int batch=2,heads=3;
        int width=headWidth*heads;
        float[] values=Enumerable.Range(0,batch*sequence*width*3).Select(i=>MathF.Sin(i*.013f)*.1f).ToArray();
        float[] seed=Enumerable.Range(0,batch*sequence*width).Select(i=>MathF.Cos(i*.017f)*.01f).ToArray();
        (float[] Value,float[] Grad) Run(bool bounds){
            using var scope=Tensor.BeginArcExecution(options:new(){CausalAttentionBounds=bounds});
            var x=new Tensor((float[])values.Clone(),[batch,sequence,3*width]);
            var y=x.FusedMultiHeadAttention(heads,causal:true);
            float[] output=y.Data.ToArray();y.BackwardAndRelease(seed);
            return(output,x.Grad.ToArray());
        }
        var expected=Run(false);var actual=Run(true);
        Assert.Equal(expected.Value,actual.Value);Assert.Equal(expected.Grad,actual.Grad);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void MultipleBatchesHeadTilesAndOddHeadWidthMatchStreaming(bool causal, bool bf16)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int batch=3,heads=3,width=15,sequence=1025;
        var values=Enumerable.Range(0,batch*sequence*3*width).Select(i=>MathF.Sin(i*.017f)*.1f).ToArray();
        var seed=Enumerable.Range(0,batch*sequence*width).Select(i=>MathF.Cos(i*.031f)*.01f).ToArray();
        (float[] Output,float[] Grad) Run(bool batched){
            using var scope=Tensor.BeginArcExecution(options:new(){BatchedAttention=batched});
            var x=new Tensor((float[])values.Clone(),[batch,sequence,3*width]);
            if(bf16)x.ConvertStorageInPlace(TensorDType.BFloat16);
            var y=x.FusedMultiHeadAttention(heads,causal);
            float[] output=y.Data.ToArray();y.BackwardAndRelease(seed);
            return(output,x.Grad.ToArray());
        }
        var expected=Run(false);var actual=Run(true);
        for(int i=0;i<actual.Output.Length;i++)Assert.InRange(MathF.Abs(expected.Output[i]-actual.Output[i]),0,1e-5f);
        for(int i=0;i<actual.Grad.Length;i++)Assert.InRange(MathF.Abs(expected.Grad[i]-actual.Grad[i]),0,1e-5f);
    }

    [Fact]
    public void BlockReducedNormGradientsMatchSerialWithRowAndColumnTails()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int rows=1027,width=37;
        var values=Enumerable.Range(0,rows*width).Select(i=>MathF.Sin(i*.017f)).ToArray();
        var seed=Enumerable.Range(0,rows*width).Select(i=>MathF.Cos(i*.031f)*.001f).ToArray();
        (float[] Gamma,float[] Beta) Run(bool parallel){
            using var scope=Tensor.BeginArcExecution(options:new(){ParallelReductions=parallel});
            var x=new Tensor((float[])values.Clone(),[rows,width]);
            var g=new Tensor(Enumerable.Repeat(1f,width).ToArray(),[width]);var b=new Tensor(new float[width],[width]);
            x.LayerNormLastDim(g,b).BackwardAndRelease(seed);
            return(g.Grad.ToArray(),b.Grad.ToArray());
        }
        var expected=Run(false);var actual=Run(true);
        for(int i=0;i<width;i++){
            Assert.InRange(MathF.Abs(expected.Gamma[i]-actual.Gamma[i]),0,1e-5f);
            Assert.InRange(MathF.Abs(expected.Beta[i]-actual.Beta[i]),0,1e-5f);
        }
    }
}
