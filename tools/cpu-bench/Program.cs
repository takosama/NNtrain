using NNtrain;
using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Reflection;
using System.Text.Json;

// Deliberately no CLI, model loader, ExecutionSession or backend capability probe.
// Tensor's ambient default is CPU; selecting CPU does not activate a GPU lane.
Tensor.ExecutionDevice = TensorDevice.Cpu;
Tensor.MaxDegreeOfParallelism = 1;
Tensor.Float16NativeEnabled = false;
Process.GetCurrentProcess().PriorityClass = ProcessPriorityClass.BelowNormal;
Console.WriteLine(JsonSerializer.Serialize(new { Environment.ProcessorCount, Avx = Avx.IsSupported, Avx2 = Avx2.IsSupported, Fma = Fma.IsSupported, Avx512 = Avx512F.IsSupported }));
var rng = new Random(5700);
float[] Input(int n) => Enumerable.Range(0,n).Select(_ => (float)(rng.NextDouble()*2-1)).ToArray();
var measurements = new List<object>();
if (args.Contains("--verify"))
{
    double maxForward = 0, maxGradient = 0;
    int cases = 0;
    foreach (string operation in new[] {"rms", "silu"})
    foreach (var dtype in new[] {TensorDType.Float32, TensorDType.Float16, TensorDType.BFloat16, TensorDType.Bfp8})
    foreach (int width in new[] {1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 257, 2048})
    {
        int rows = 3;
        float[] data = Input(rows*width), weights = Input(operation == "rms" ? width : rows*width), seed = Input(rows*width);
        (float[] y,float[] dx,float[] dw) Run(bool simd)
        {
            Tensor.SimdEnabled = simd;
            var x = new Tensor(data,[rows,width],dtype:dtype);
            var w = new Tensor(weights,operation == "rms" ? [width] : [rows,width],dtype:dtype);
            var y = operation == "rms" ? x.RmsNormLastDim(w) : x.SiluMultiply(w);
            y.Backward(seed);
            return (y.Data.ToArray(),x.Grad.ToArray(),w.Grad.ToArray());
        }
        var reference = Run(false); var actual = Run(true);
        void Check(float[] a,float[] b, bool grad)
        {
            for(int i=0;i<a.Length;i++)
            {
                double error = Math.Abs(a[i]-b[i]);
                if (grad) maxGradient=Math.Max(maxGradient,error); else maxForward=Math.Max(maxForward,error);
                double tolerance = dtype == TensorDType.Float32 ? 3e-5 : dtype == TensorDType.Float16 ? 0.003 : dtype == TensorDType.BFloat16 ? 0.02 : 0.03;
                if (!float.IsFinite(b[i]) || error > tolerance*(1+Math.Abs(a[i])))
                    throw new Exception($"{operation} mismatch {dtype}/{width}/{i}: {a[i]} vs {b[i]}");
            }
        }
        Check(reference.y,actual.y,false); Check(reference.dx,actual.dx,true); Check(reference.dw,actual.dw,true);
        // Finite differences of an independent double-precision RMS formula.
        if(dtype==TensorDType.Float32)
        {
            double Loss(float[] values)
            {
                double loss=0;
                if(operation=="silu")
                {
                    for(int i=0;i<values.Length;i++) loss+=values[i]/(1+Math.Exp(-values[i]))*weights[i]*seed[i];
                    return loss;
                }
                for(int r=0;r<rows;r++)
                {
                    double sum=0; for(int i=0;i<width;i++) sum+=(double)values[r*width+i]*values[r*width+i];
                    double inv=1/Math.Sqrt(sum/width+1e-6);
                    for(int i=0;i<width;i++) loss+=values[r*width+i]*inv*weights[i]*seed[r*width+i];
                }
                return loss;
            }
            for(int i=0;i<Math.Min(data.Length,10);i++)
            {
                float saved=data[i]; data[i]=saved+0.001f; double plus=Loss(data);
                data[i]=saved-0.001f; double minus=Loss(data); double delta=(saved+0.001f)-(saved-0.001f); data[i]=saved;
                double numerical=(plus-minus)/delta;
                if(Math.Abs(numerical-actual.dx[i])>0.002*(1+Math.Abs(numerical))) throw new Exception("Finite difference failed");
            }
        }
        using (AutogradContext.NoGrad())
        {
            var x=new Tensor(data,[rows,width],dtype:dtype); var w=new Tensor(weights,operation == "rms" ? [width] : [rows,width],dtype:dtype);
            Check(actual.y,(operation=="rms" ? x.RmsNormLastDim(w) : x.SiluMultiply(w)).Data.ToArray(),false);
        }
        cases++;
    }
    Tensor.SimdEnabled=true;
    // Check overflow/underflow and exceptional-value semantics in every SIMD lane.
    float[] extremes=[float.NaN,float.PositiveInfinity,float.NegativeInfinity,0f,-0f,100f,-100f,88f,-88f,1e-30f,-1e-30f,10f,-10f,80f,-80f,1f];
    extremes=extremes.Concat(extremes).ToArray();
    float[] Special(bool simd)
    {
        Tensor.SimdEnabled=simd;
        using var ng=AutogradContext.NoGrad();
        var x=new Tensor(extremes,[extremes.Length]);
        return x.SiluMultiply(new Tensor(Enumerable.Repeat(1f,extremes.Length).ToArray(),[extremes.Length])).Data.ToArray();
    }
    var specialRef=Special(false); var specialActual=Special(true);
    for(int i=0;i<extremes.Length;i++)
    {
        if(float.IsNaN(specialRef[i]) ? !float.IsNaN(specialActual[i]) : float.IsInfinity(specialRef[i]) ? specialRef[i]!=specialActual[i] : Math.Abs(specialRef[i]-specialActual[i])>1e-5*(1+Math.Abs(specialRef[i])))
            throw new Exception("SiLU exceptional semantics mismatch");
    }
    float[] Alias(bool simd)
    {
        Tensor.SimdEnabled=simd;
        var x=new Tensor(Enumerable.Range(0,33).Select(i=>(i-16)*0.25f).ToArray(),[33]);
        x.SiluMultiply(x).Backward(Enumerable.Repeat(1f,33).ToArray());
        return x.Grad.ToArray();
    }
    var aliasRef=Alias(false); var aliasActual=Alias(true);
    for(int i=0;i<aliasRef.Length;i++) if(Math.Abs(aliasRef[i]-aliasActual[i])>1e-5*(1+Math.Abs(aliasRef[i]))) throw new Exception("Aliased gradient mismatch");
    int codecVectors=0;
    var field=typeof(Tensor).GetProperty("_data",BindingFlags.NonPublic|BindingFlags.Instance)!;
    var load=field.PropertyType.GetMethod("LoadVector256",BindingFlags.NonPublic|BindingFlags.Instance)!;
    foreach(int length in new[] {8,9,17,33,129,257})
    foreach(var descriptor in new[] {Bfp8QuantizationDescriptor.TensorWide,Bfp8QuantizationDescriptor.Block(1),Bfp8QuantizationDescriptor.Block(3),Bfp8QuantizationDescriptor.Block(7),Bfp8QuantizationDescriptor.Block(8),Bfp8QuantizationDescriptor.Block(9),Bfp8QuantizationDescriptor.Block(31),Bfp8QuantizationDescriptor.Mix8_32})
    {
        var x=Tensor.FromBfp8(Input(length),[length],descriptor);
        foreach(bool simd in new[] {false,true})
        {
            Tensor.SimdEnabled=simd;
            for(int offset=0;offset<=length-8;offset++)
            {
                var vector=(Vector256<float>)load.Invoke(field.GetValue(x),[offset])!;
                for(int lane=0;lane<8;lane++)
                    if(BitConverter.SingleToInt32Bits(vector.GetElement(lane))!=BitConverter.SingleToInt32Bits(x.Data[offset+lane]))
                        throw new Exception($"BFP8 decode differs at {length}/{descriptor}/{offset}/{lane}");
                codecVectors++;
            }
        }
    }
    Tensor.SimdEnabled=true;
    Console.WriteLine(JsonSerializer.Serialize(new {verification="PASS",cases,codecVectors,maxForward,maxGradient}));
    return;
}
foreach (var dtype in args.Contains("--profile") ? new[] {TensorDType.Float32,TensorDType.Bfp8} : args.Contains("--matrix") ? new[] {TensorDType.Float32,TensorDType.Float16,TensorDType.BFloat16,TensorDType.Bfp8} : new[] {TensorDType.Float32})
foreach (int width in args.Contains("--profile") ? new[] {256,1024} : args.Contains("--matrix") ? new[] {1,7,8,9,33,257,2048} : new[] {32,257,2048})
foreach (int rows in args.Contains("--profile") ? new[] {1,8} : new[] {8})
{
    var x = new Tensor(Input(rows*width), [rows,width],dtype:dtype);
    var w = new Tensor(Input(width), [width],dtype:dtype);
    var u = new Tensor(Input(rows*width), [rows,width],dtype:dtype);
    using var noGrad = AutogradContext.NoGrad();
    var operations = new List<(string,Func<Tensor>)> { ("rms",()=>x.RmsNormLastDim(w)), ("silu",()=>x.SiluMultiply(u)) };
    if(args.Contains("--profile"))
    {
        var matrix=new Tensor(Input(width*width),[width,width],dtype:dtype);
        operations.Add(("matmul",()=>x.MatMulTransposedRight(matrix)));
    }
    int repeats = args.Contains("--profile") ? 10 : 100;
    foreach (var (name,run) in operations)
    {
        for(int j=0;j<(args.Contains("--profile") ? 3 : 30);j++) run();
        var times = new List<double>(); var allocations = new List<long>();
        for(int j=0;j<7;j++)
        {
            long a=GC.GetAllocatedBytesForCurrentThread(); var sw=Stopwatch.StartNew();
            for(int k=0;k<repeats;k++) run();
            sw.Stop(); times.Add(sw.Elapsed.TotalMicroseconds/repeats); allocations.Add((GC.GetAllocatedBytesForCurrentThread()-a)/repeats);
        }
        double mean=times.Average();
        measurements.Add(new {name,dtype=dtype.ToString(),width,rows,meanUs=mean,sdUs=Math.Sqrt(times.Select(t=>(t-mean)*(t-mean)).Average()),minUs=times.Min(),allocatedBytes=allocations.Min(),samplesUs=times});
    }
}
Console.WriteLine(JsonSerializer.Serialize(measurements, new JsonSerializerOptions {WriteIndented=true}));
