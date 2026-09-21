using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;

namespace NNtrain.Benchmarks;

internal static class ArcBfp8EpilogueProbe
{
    internal static void Run(string path)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Use a new probe output path.");
        using var lane = new ArcExecutionLane();
        if (!lane.Device.SupportsXmx || lane.Device.MinimumSubgroupSize != 16)
            throw new NotSupportedException("This candidate requires SG16 XMX.");
        var results = new List<object>();
        foreach (var (m,n,k) in new[] { (137,64,79), (256,96,128), (4096,1536,512), (4096,512,1536), (16384,1536,512), (16384,512,1536), (65536,1536,512) })
        foreach (bool relu in new[] { false, true })
        {
            int ac = ((m+7)/8)*((k+15)/16)*128, bc=((n+15)/16)*((k+15)/16)*256;
            ushort[] Bits(int length, int seed) => Enumerable.Range(0,length).Select(i =>
                TensorStorageCodec.EncodeBFloat16(((i*13+seed)%61-30)*.00091f)).ToArray();
            using var a=lane.UploadRaw(Bits(ac,7)); using var b=lane.UploadRaw(Bits(bc,11));
            using var bias=lane.Upload(Enumerable.Range(0,n).Select(i=>(i%19-9)*.00037f).ToArray());
            using var raw=lane.Allocate(checked(m*n));
            using var packed=lane.AllocateBytes(checked(m*n));using var scales=lane.Allocate(checked(m*n/32));
            using var status=lane.UploadRaw(new int[1]);
            sbyte[]? expected=null;float[]? expectedScales=null;
            foreach(var (columns,rows,sg) in new[]{(0,16,16),(32,16,16),(64,16,16),(64,32,4),(64,32,8),(128,16,4),(128,16,8)})
            {
                if(columns!=0&&n%columns!=0)continue;
                bool fused=columns!=0;
                string kernel = $"gemm_xmx_bfp8_epilogue_{rows}x{columns}" + (sg == 16 ? "" : $"_wg{sg}");
                void Dispatch()
                {
                    if(fused) lane.Run2D(kernel,n/(long)columns*16,((m+(long)rows*sg-1)/(rows*sg))*sg,16,sg,
                        a,b,packed,scales,status,bias,m,n,k,relu?1:0,0);
                    else {
                        lane.Run2D("gemm_xmx_direct_block_16x32_wg16",n/32L*16,((m+255L)/256)*16,16,16,
                            a,b,raw,bias,a,m,n,k,0,1,3,0,1,relu?1:0,0,0,k);
                        lane.Run("resident_bfp8_quad4_32",m*n/8L,256,raw,packed,scales,status,m*n,32);
                    }
                }
                Dispatch();var bytes=new sbyte[m*n];var scale=new float[m*n/32];var flags=new int[1];
                lane.ReadRaw(packed,bytes);lane.Read(scales,scale);lane.ReadRaw(status,flags);
                if(flags[0]!=0)throw new ArithmeticException("Nonfinite GEMM epilogue.");
                if(!fused){expected=bytes;expectedScales=scale;}
                else if(!expected!.SequenceEqual(bytes)||!expectedScales!.Select(BitConverter.SingleToInt32Bits).SequenceEqual(scale.Select(BitConverter.SingleToInt32Bits)))
                    throw new ArithmeticException($"Fused epilogue is not bit exact: {m}x{n}x{k} relu={relu}.");
                var wall=new List<double>();var gpu=new List<double>();long h2d=lane.H2DBytes,d2h=lane.D2HBytes;
                for(int it=0;it<10;it++){
                    double before=lane.KernelMilliseconds;long start=Stopwatch.GetTimestamp();Dispatch();lane.Synchronize();
                    if(it>=2){wall.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);gpu.Add(lane.KernelMilliseconds-before);}
                }
                if(h2d!=lane.H2DBytes||d2h!=lane.D2HBytes)throw new InvalidOperationException("Unexpected transfer in benchmark.");
                static double Median(List<double> values)=>(values.Order().ElementAt(3)+values.Order().ElementAt(4))*.5;
                var resources = lane.GetKernelResources(fused ? kernel : "gemm_xmx_direct_block_16x32_wg16");
                Console.WriteLine($"GEMM {m}x{n}x{k} relu={relu} fused={rows}x{columns}/wg{sg}: GPU={Median(gpu):F4} ms, wall={Median(wall):F4} ms spill={resources.SpillMemoryBytes} (bit-exact)");
                results.Add(new{M=m,N=n,K=k,Relu=relu,Fused=fused,Columns=columns,Rows=rows,Subgroups=sg,Resources=resources,GpuP50Ms=Median(gpu),P50Ms=Median(wall),GpuSamples=gpu,Samples=wall});
            }
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file=new FileStream(path,FileMode.CreateNew);
        JsonSerializer.Serialize(file,new{lane.Device.Name,lane.Device.DriverVersion,Results=results},new JsonSerializerOptions{WriteIndented=true});
    }
}
