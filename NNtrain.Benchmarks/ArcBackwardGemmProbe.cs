using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;

namespace NNtrain.Benchmarks;

internal static class ArcBackwardGemmProbe
{
    internal static void Run(string path)
    {
        path=Path.GetFullPath(path);
        if(File.Exists(path))throw new IOException("Probe output must be a new file.");
        var results=new List<object>();
        foreach(string mode in new[]{"wide","deep","split"}){
            using var lane=new ArcExecutionLane(options:new(){XmxMatrices=false,DeepMatrices=mode!="wide",ParallelWeightGradients=mode=="split"});
            foreach(var (m,n,k,ta,tb) in new[]{(16384,512,1536,false,false),(1536,512,16384,true,false),
                (16384,1536,512,false,false),(512,1536,16384,true,false),(512,512,16384,true,false),
                (512,512,11500,false,false),(11500,512,512,true,false)}){
                using var a=lane.Upload(Enumerable.Range(0,m*k).Select(i=>MathF.Sin(i*.01f)*.01f).ToArray());
                using var b=lane.Upload(Enumerable.Range(0,k*n).Select(i=>TensorStorageCodec.RoundToBFloat16(MathF.Cos(i*.01f)*.01f)).ToArray());
                using var c=lane.Upload(new float[m*n]);
                var times=new List<double>();
                for(int i=0;i<13;i++){
                    long start=Stopwatch.GetTimestamp();ArcMuonMath.Gemm(lane,a,b,c,m,n,k,ta,tb,accumulate:ta);lane.Synchronize();
                    if(i>=3)times.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                }
                times.Sort();double p50=(times[4]+times[5])/2;
                Console.WriteLine($"{mode} {m}x{n}x{k} ta={ta} tb={tb}: {p50:F4} ms");
                results.Add(new{Mode=mode,M=m,N=n,K=k,Ta=ta,Tb=tb,P50Ms=p50,Samples=times});
            }
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file=new FileStream(path,FileMode.CreateNew);
        JsonSerializer.Serialize(file,new{Timestamp=DateTimeOffset.UtcNow,Note="FP32 gradient x BF16-representable RHS, 3 warmup/10 timed dispatch+finish; no host transfers; bounded split workspace allocation included.",Results=results},new JsonSerializerOptions{WriteIndented=true});
    }
}
