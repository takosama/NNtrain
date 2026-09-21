using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcFusedBfp8LinearTests
{
    private static void RequireXmx()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");
    }
    [Theory]
    [InlineData(4099,64,79,32,false,TensorPrecisionMode.Mix8_32)]
    [InlineData(4099,64,79,32,true,TensorPrecisionMode.Mix8_32)]
    [InlineData(4096,64,128,128,true,TensorPrecisionMode.Mix8_32)]
    [InlineData(4096,65,128,32,true,TensorPrecisionMode.Mix8_32)]
    [InlineData(4096,64,128,32,false,TensorPrecisionMode.Mix16_32)]
    [InlineData(513,64,128,32,false,TensorPrecisionMode.Mix8_32)]
    public void ForwardAndAccumulatedGradientsKeepTheirExactBits(int rows,int n,int k,int block,bool relu,TensorPrecisionMode precision)
    {
        RequireXmx();
        float[][] Run(bool enabled)
        {
            using var execution=Tensor.BeginArcExecution(precision:precision,options:new(){FusedBfp8Linear=enabled,ExpandedXmxTiles=false});
            Tensor Make(int[] shape,int seed){var t=new Tensor(Enumerable.Range(0,shape.Aggregate(1,(a,b)=>a*b)).Select(i=>MathF.Sin(i*.013f+seed)*.031f).ToArray(),shape);
                t.ConvertStorageInPlace(precision.ToStorageDType(),precision==TensorPrecisionMode.Mix8_32?Bfp8QuantizationDescriptor.Block(block):null);return t;}
            var x=Make([rows,k],1);var w=Make([n,k],2);var b=Make([n],3);float[] values=[];
            for(int i=0;i<2;i++){
                var y=x.LinearLastDim(w,b,applyRelu:relu);values=y.Data.ToArray();
                long downloads=Tensor.ArcLane.D2HBytes;
                y.BackwardAndRelease(Enumerable.Range(0,rows*n).Select(j=>MathF.Cos(j*.017f+i)*.003f).ToArray());
                Tensor.ArcLane.Synchronize();Assert.Equal(downloads,Tensor.ArcLane.D2HBytes);
            }
            bool eligible=enabled&&rows>=4096&&n%32==0&&block==32&&precision==TensorPrecisionMode.Mix8_32;
            Assert.Equal(eligible,Tensor.ArcLane.KernelTimings.ContainsKey("gemm_xmx_bfp8_epilogue_16x32"));
            Tensor.ArcLane.CheckNumericStatus();
            return [values,x.Grad.ToArray(),w.Grad.ToArray(),b.Grad.ToArray()];
        }
        var old=Run(false);var candidate=Run(true);
        for(int i=0;i<old.Length;i++)Assert.Equal(old[i].Select(BitConverter.SingleToInt32Bits),candidate[i].Select(BitConverter.SingleToInt32Bits));
    }
    [Theory]
    [InlineData(64, false)]
    [InlineData(512, true)]
    public void StreamedForwardPreservesOutputAcrossRowSlicesAndTail(int n, bool expanded)
    {
        RequireXmx();
        const int rows=32771,k=1536;
        float[] Run(bool enabled)
        {
            using var execution=Tensor.BeginArcExecution(precision:TensorPrecisionMode.Mix8_32,options:new(){FusedBfp8Linear=enabled,ExpandedXmxTiles=enabled&&expanded});
            Tensor Make(int[] shape) {var t=new Tensor(Enumerable.Range(0,shape.Aggregate(1,(a,b)=>a*b)).Select(i=>(i%31-15)*.0007f).ToArray(),shape);
                t.ConvertStorageInPlace(TensorDType.Bfp8,Bfp8QuantizationDescriptor.Block(32));return t;}
            var x=Make([rows,k]);var w=Make([n,k]);var b=Make([n]);
            Assert.False(ArcXmxStorageOperand.CanRun(Tensor.ArcLane,rows,n,k));
            var y=x.LinearLastDim(w,b,applyRelu:true);
            float[] result=y.Data.ToArray();
            Tensor.ArcLane.CheckNumericStatus();
            Assert.Equal(enabled,Tensor.ArcLane.KernelTimings.ContainsKey(expanded?"gemm_xmx_bfp8_epilogue_16x64":"gemm_xmx_bfp8_epilogue_16x32"));
            return result;
        }
        Assert.Equal(Run(false).Select(BitConverter.SingleToInt32Bits),Run(true).Select(BitConverter.SingleToInt32Bits));
    }
    [Fact]
    public void NonfiniteFusedOutputRaisesStatusInsteadOfCommitting()
    {
        RequireXmx();
        using var execution=Tensor.BeginArcExecution(precision:TensorPrecisionMode.Mix8_32,options:new(){FusedBfp8Linear=true});
        var x=new Tensor(Enumerable.Repeat(1e20f,4096*64).ToArray(),[4096,64]);x.ConvertStorageInPlace(TensorDType.Bfp8,Bfp8QuantizationDescriptor.Block(32));
        var w=new Tensor(Enumerable.Repeat(1e20f,64*64).ToArray(),[64,64]);w.ConvertStorageInPlace(TensorDType.Bfp8,Bfp8QuantizationDescriptor.Block(32));
        var b=new Tensor(new float[64],[64]);b.ConvertStorageInPlace(TensorDType.Bfp8,Bfp8QuantizationDescriptor.Block(32));
        var output=x.LinearLastDim(w,b,applyRelu:false);
        try { Assert.Throws<ArithmeticException>(Tensor.ArcLane.CheckNumericStatus); }
        finally { output.ReleaseArcGraph(); }
    }

    [Theory]
    [InlineData(4096, 512, false)]
    [InlineData(4099, 512, true)]
    [InlineData(4096, 1536, true)]
    public void ExpandedTileAtFourThousandRowsPreservesOutputAndAccumulatedGradients(int rows, int n, bool relu)
    {
        RequireXmx();
        const int k = 79;
        float[][] Run(bool expanded)
        {
            using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_32,
                options: new() { FusedBfp8Linear = true, ExpandedXmxTiles = expanded });
            Tensor Make(int[] shape, int seed)
            {
                var tensor = new Tensor(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b))
                    .Select(i => MathF.Sin(i * .013f + seed) * .031f).ToArray(), shape);
                tensor.ConvertStorageInPlace(TensorDType.Bfp8, Bfp8QuantizationDescriptor.Block(32));
                return tensor;
            }
            var x = Make([rows, k], 1);
            var w = Make([n, k], 2);
            var b = Make([n], 3);
            float[] values = [];
            for (int iteration = 0; iteration < 2; iteration++)
            {
                var y = x.LinearLastDim(w, b, applyRelu: relu);
                values = y.Data.ToArray();
                long downloads = Tensor.ArcLane.D2HBytes;
                y.BackwardAndRelease(Enumerable.Range(0, rows * n)
                    .Select(j => MathF.Cos(j * .017f + iteration) * .003f).ToArray());
                Tensor.ArcLane.Synchronize();
                Assert.Equal(downloads, Tensor.ArcLane.D2HBytes);
            }
            Assert.Contains(expanded ? "gemm_xmx_bfp8_epilogue_16x64" : "gemm_xmx_bfp8_epilogue_16x32",
                Tensor.ArcLane.KernelTimings.Keys);
            Tensor.ArcLane.CheckNumericStatus();
            return [values, x.Grad.ToArray(), w.Grad.ToArray(), b.Grad.ToArray()];
        }
        var baseline = Run(false);
        var candidate = Run(true);
        for (int i = 0; i < baseline.Length; i++)
            Assert.Equal(baseline[i].Select(BitConverter.SingleToInt32Bits), candidate[i].Select(BitConverter.SingleToInt32Bits));
    }
}
