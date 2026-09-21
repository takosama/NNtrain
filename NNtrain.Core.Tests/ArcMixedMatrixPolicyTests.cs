using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMixedMatrixPolicyTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Float32,false)]
    [InlineData(TensorPrecisionMode.Mix16_32,false)]
    [InlineData(TensorPrecisionMode.Mix16_32,true)]
    [InlineData(TensorPrecisionMode.Mix8_32,false)]
    [InlineData(TensorPrecisionMode.Mix8_32,true)]
    public void LinearUsesPolicyOperandsButKeepsFp32GradientAccumulation(TensorPrecisionMode precision,bool relu)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int rows=137,ni=65,no=71;
        using var scope=Tensor.BeginArcExecution(precision:precision);
        Tensor Make(int[] shape,int seed){
            var t=new Tensor(Enumerable.Range(0,shape.Aggregate(1,(a,b)=>a*b)).Select(i=>MathF.Sin(i*.013f+seed)*.031f).ToArray(),shape);
            t.ConvertStorageInPlace(precision.ToStorageDType(),precision==TensorPrecisionMode.Mix8_32?Bfp8QuantizationDescriptor.Mix8_32:null);
            return t;
        }
        var x=Make([rows,ni],1);var w=Make([no,ni],2);var b=Make([no],3);
        bool mixed=precision!=TensorPrecisionMode.Float32;
        float R(float v)=>mixed?TensorStorageCodec.RoundToBFloat16(v):v;
        var xv=x.Data.ToArray().Select(R).ToArray();var wv=w.Data.ToArray().Select(R).ToArray();
        float[] dx=new float[rows*ni],dw=new float[no*ni],db=new float[no];
        for(int step=0;step<2;step++){
            var y=x.LinearLastDim(w,b,applyRelu:relu);
            float[] output=y.Data.ToArray();
            var seed=Enumerable.Range(0,rows*no).Select(i=>MathF.Cos(i*.017f+step)*.013f).ToArray();
            for(int row=0;row<rows;row++)for(int col=0;col<no;col++){
                float g=R(relu&&output[row*no+col]<=0?0:seed[row*no+col]);db[col]+=g;
                for(int k=0;k<ni;k++){
                    dx[row*ni+k]=MathF.FusedMultiplyAdd(g,wv[col*ni+k],dx[row*ni+k]);
                    dw[col*ni+k]=MathF.FusedMultiplyAdd(g,xv[row*ni+k],dw[col*ni+k]);
                }
            }
            y.BackwardAndRelease(seed);
        }
        void Equal(float[] expected,float[] actual){for(int i=0;i<actual.Length;i++)Assert.InRange(MathF.Abs(expected[i]-actual[i]),0,1e-5f);}
        Equal(dx,x.Grad.ToArray());Equal(dw,w.Grad.ToArray());Equal(db,b.Grad.ToArray());
        Assert.Contains(w.Grad.ToArray(),v=>v!=TensorStorageCodec.RoundToBFloat16(v));
        Assert.Equal(mixed&&ArcDevices.Enumerate()[0].SupportsXmx,
            Tensor.ArcLane.KernelTimings.Keys.Any(name => name.StartsWith("gemm_xmx", StringComparison.Ordinal)));
    }

    [Fact]
    public void Bf16SplitKPreservesFp32DestinationAndTails()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx,"Intel XMX is required.");
        const int m=65,n=71,k=4101;
        var av=Enumerable.Range(0,m*k).Select(i=>(i%7-3)/16f).ToArray();
        var bv=Enumerable.Range(0,n*k).Select(i=>(i%5-2)/16f).ToArray();
        float[] Run(bool parallel){
            using var lane=new ArcExecutionLane(options:new(){ParallelWeightGradients=parallel});
            using var a=lane.Upload(av);using var b=lane.Upload(bv);using var c=lane.Upload(Enumerable.Repeat(.125f,m*n).ToArray());
            for(int step=0;step<2;step++)ArcMuonMath.Gemm(lane,a,b,c,m,n,k,ta:true,bf16:3,accumulate:true);
            float[] values=new float[m*n];lane.Read(c,values);return values;
        }
        Assert.Equal(Run(false),Run(true));
    }
}
