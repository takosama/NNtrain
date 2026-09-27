using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35TrainingOptimizationTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OptimizedUpdatePreservesResponseLossGradientsAndAdam(bool cooperative)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        using var fixture = Qwen35ResidentModelTests.CreateFixture(tiedOutput: false);
        using var reference = Qwen35QuantizedModel.Load(fixture.Path, [0], options: new()
        {
            LoraTraining=true,TrainingBatchGradientNorm=false,TrainingCooperativeLora=false,
            TrainingCooperativeDelta=false,TrainingResponseOnlyHead=false,TrainingTransposeRows=1,TrainingForwardRows=1
        });
        using var optimized = Qwen35QuantizedModel.Load(fixture.Path, [0], options: new()
        {
            LoraTraining=true,TrainingBatchGradientNorm=true,TrainingCooperativeLora=cooperative,
            TrainingCooperativeDelta=cooperative,TrainingResponseOnlyHead=true,TrainingTransposeRows=16,
            TrainingNormSplits=cooperative?8:1
        });
        var options = new Qwen35LoraOptions { Rank=3,Alpha=6,IncludeOutput=true,Seed=57,GradientClip=1 };
        reference.AttachLora(options); optimized.AttachLora(options);
        int[] tokens=[1,2,3,0,2,1];
        foreach(int responseStart in new[]{4,1,2})
        {
            var a=reference.TrainLora(tokens,responseStart);
            var b=optimized.TrainLora(tokens,responseStart);
            Close(a.Loss,b.Loss,cooperative?3e-5:1e-7);
            Close(a.GradientNorm,b.GradientNorm,cooperative?3e-4:2e-6);
            Assert.Equal(a.SupervisedTokens,b.SupervisedTokens);
            Assert.Equal(reference.ResidentWeightBytes,optimized.ResidentWeightBytes);
            foreach(var pair in reference.LoraMatrices)
            {
                var left=pair.Value.ReadState();var right=optimized.LoraMatrices[pair.Key].ReadState();
                for(int field=0;field<left.Length;field++)
                    for(int i=0;i<left[field].Length;i++) Close(left[field][i],right[field][i],cooperative?3e-5:2e-6);
            }
        }
    }

    [Fact]
    public void CooperativeLowRankAdjointsMatchSerialForRaggedShapesAndNonzeroB()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0,"Intel Arc GPU is required.");
        using var lane=new ArcExecutionLane(0,new(){Qwen35InferenceKernelsOnly=true,Qwen35TrainingKernels=true});
        const int rows=5,input=513,output=259,rank=3;
        var random=new Random(42);
        float[] Values(int n)=>Enumerable.Range(0,n).Select(_=>(float)(random.NextDouble()*.4-.2)).ToArray();
        using ArcBuffer x=lane.Upload(Values(rows*input)),a=lane.Upload(Values(input*rank)),dy=lane.Upload(Values(rows*output)),b=lane.Upload(Values(output*rank));
        using ArcBuffer z=lane.Allocate(rows*rank),zc=lane.Allocate(rows*rank),dz=lane.Allocate(rows*rank),dzc=lane.Allocate(rows*rank);
        lane.Run("q35l_lora_a",rows*rank,0,x,a,z,rows,input,rank);
        lane.Run("q35l_lora_a_coop",rows*rank*128,128,x,a,zc,rows,input,rank);
        lane.Run("q35t_lora_dz",rows*rank,0,dy,b,dz,rows,output,rank,2f);
        lane.Run("q35t_lora_dz_coop",rows*rank*128,128,dy,b,dzc,rows,output,rank,2f);
        var expected=new float[rows*rank];var actual=new float[rows*rank];
        foreach(var pair in new[]{(z,zc),(dz,dzc)}){
            lane.Read(pair.Item1,expected);lane.Read(pair.Item2,actual);
            for(int i=0;i<expected.Length;i++)Close(expected[i],actual[i],2e-6);
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(32)]
    public void BatchedNormPreservesOffsetsAndNonfiniteDetection(int splits)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0,"Intel Arc GPU is required.");
        using var lane=new ArcExecutionLane(0,new(){Qwen35InferenceKernelsOnly=true,Qwen35TrainingKernels=true});
        float[] values=Enumerable.Range(0,513).Select(i=>(i%17-8)*.03f).ToArray();
        using ArcBuffer x=lane.Upload(values),result=lane.Upload(Enumerable.Repeat(-7f,splits+2).ToArray()),single=lane.Allocate(1);
        lane.Run("q35t_norm2",128,128,x,single,values.Length);
        lane.Run("q35t_norm2_split",128*splits,128,x,result,values.Length,1,splits);
        var expected=new float[1];var actual=new float[splits+2];lane.Read(single,expected);lane.Read(result,actual);
        Assert.Equal(-7f,actual[0]);Assert.Equal(-7f,actual[^1]);
        Close(expected[0],actual.Skip(1).Take(splits).Sum(v=>(double)v),1e-6);
        values[500]=float.NaN;lane.Write(x,values);
        lane.Run("q35t_norm2_split",128*splits,128,x,result,values.Length,1,splits);
        lane.Read(result,actual);Assert.Contains(actual,float.IsPositiveInfinity);
    }

    private static void Close(double expected,double actual,double tolerance)
        => Assert.True(double.IsFinite(actual)&&Math.Abs(expected-actual)<=tolerance*(1+Math.Abs(expected)),
            $"Expected {expected:R}, actual {actual:R}, tolerance {tolerance:R}");
}
