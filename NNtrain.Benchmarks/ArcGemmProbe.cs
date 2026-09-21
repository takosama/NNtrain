using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;

namespace NNtrain.Benchmarks;

internal static class ArcGemmProbe
{
    internal static void Run(string path)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("GEMM result must be a new file.");
        var results = new List<object>();
        foreach (bool xmx in new[] { false, true })
        {
            using var lane = new ArcExecutionLane(options: new() { XmxMatrices = xmx });
            if (xmx && !lane.Device.SupportsXmx) throw new NotSupportedException("XMX is unavailable.");
            foreach (var shape in new[] { (512,512,512,false,true), (512,512,512,false,false),
                (512,512,1536,false,true), (512,1536,512,false,false), (16384,1536,512,false,true), (128,11500,512,false,true) })
            {
                var (m,n,k,ta,tb) = shape;
                float[] av = Enumerable.Range(0,m*k).Select(i => (i%17-8)/32f).ToArray();
                float[] bv = Enumerable.Range(0,k*n).Select(i => (i%19-9)/32f).ToArray();
                using var a = lane.Upload(av); using var b = lane.Upload(bv); using var c = lane.Allocate(m*n);
                var times = new List<double>();
                for (int i=0;i<13;i++)
                {
                    long start = Stopwatch.GetTimestamp();
                    ArcMuonMath.Gemm(lane,a,b,c,m,n,k,ta,tb,bf16:3);
                    lane.Synchronize();
                    if(i>=3)times.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                }
                times.Sort(); double p50 = (times[4]+times[5])/2;
                Console.WriteLine($"{(xmx?"XMX":"FP32 tile")} {m}x{n}x{k} ta={ta} tb={tb}: {p50:F4} ms");
                results.Add(new { Xmx=xmx,M=m,N=n,K=k,Ta=ta,Tb=tb,P50Ms=p50,Samples=times, Device=lane.Device.Name, lane.Device.DriverVersion });
            }
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path,FileMode.CreateNew);
        JsonSerializer.Serialize(file,new { Timestamp=DateTimeOffset.UtcNow,Note="BF16-rounded operands; FP32 accumulation; 3 warmup, 10 measurements; blocking dispatch wall time; allocation and transfers excluded.",Results=results },new JsonSerializerOptions {WriteIndented=true});
    }
}
