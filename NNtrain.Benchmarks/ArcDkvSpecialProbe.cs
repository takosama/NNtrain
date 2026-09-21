using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Benchmarks;

internal static class ArcDkvSpecialProbe
{
    private enum Candidate { Reference, K32Q32, K64Q32, K32Q64, K64Q16, Heads16K64Q32, RegisterOwner, Register8, Register4, Register2, Swizzle16, Swizzle8, Swizzle4, Unroll1, Unroll4, Unroll8, Unroll16, K16Q16, K16Q32, K16Q64, Rq8, Rq16, Rq32, Rq16W128, BlockSlm, BlockSlmK32Q16, BlockSlmK64Q16, BlockSlmK64Q32, BlockSlmK128Q16, Split2, Split4 }

    internal static void Run(string path, bool narrowOnly = false, bool registerQueryOnly = false, bool splitOnly = false)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Probe output must be a new file.");
        using var lane = new ArcExecutionLane(options: new() { ExperimentalOptimizationKernels = registerQueryOnly || splitOnly });
        Candidate[] candidates = splitOnly ? [Candidate.Reference, Candidate.BlockSlm, Candidate.Split2, Candidate.Split4]
            : registerQueryOnly ? [Candidate.Reference, Candidate.K32Q32, Candidate.K64Q32, Candidate.Rq8, Candidate.Rq16, Candidate.Rq32, Candidate.Rq16W128, Candidate.BlockSlm, Candidate.BlockSlmK32Q16, Candidate.BlockSlmK64Q16, Candidate.BlockSlmK64Q32, Candidate.BlockSlmK128Q16]
            : narrowOnly ? [Candidate.Reference, Candidate.K32Q32, Candidate.K16Q16, Candidate.K16Q32, Candidate.K16Q64]
            : Enum.GetValues<Candidate>().Where(c => c <= Candidate.Unroll16).ToArray();
        Validate(lane, candidates);
        const int batch = 16, heads = 16, d = 32, sequence = 1024, first = 0, count = 4;
        const int width = heads * d;
        float[] qkvValues = Values(batch * sequence * width * 3, .013f, .13f);
        float[] dyValues = Values(batch * sequence * width, .017f, .03f);
        float[] initial = Enumerable.Range(0, qkvValues.Length).Select(i => .001f + (i % 7) * .0003f).ToArray();
        using var qkv = lane.Upload(qkvValues); using var dy = lane.Upload(dyValues);
        var results = new List<object>();
        foreach (bool causal in new[] { true, false })
        {
            var (probabilities, derivatives) = Scores(count, sequence, causal);
            using var p = lane.Upload(probabilities); using var ds = lane.Upload(derivatives);
            foreach (Candidate candidate in candidates)
            {
                using var dx = lane.Upload(initial);
                var (kernel, keys) = Plan(candidate, causal);
                var resources = lane.GetKernelResources(kernel);
                var wall = new List<double>(); var gpu = new List<double>();
                long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
                for (int i = 0; i < 13; i++)
                {
                    double gpuStart = lane.KernelMilliseconds;
                    long start = Stopwatch.GetTimestamp();
                    Dispatch(lane, candidate, causal, qkv, dy, p, ds, dx, sequence, width, heads, first, count);
                    lane.Synchronize();
                    if (i >= 3) { wall.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds); gpu.Add(lane.KernelMilliseconds - gpuStart); }
                }
                if (h2d != lane.H2DBytes || d2h != lane.D2HBytes)
                    throw new InvalidOperationException("DKV crossed the host/device boundary.");
                double Median(List<double> values) { var sorted = values.Order().ToArray(); return (sorted[4] + sorted[5]) / 2; }
                Console.WriteLine($"DKV {candidate}, causal={causal}: {Median(wall):F4} ms wall, {Median(gpu):F4} ms GPU, SLM={resources.LocalMemoryBytes}, spill={resources.SpillMemoryBytes}");
                results.Add(new { Candidate = candidate.ToString(), Causal = causal, Kernel = kernel, KeyTile = keys,
                    Batch = batch, Heads = heads, Sequence = sequence, HeadWidth = d, First = first, Count = count,
                    P50Ms = Median(wall), GpuP50Ms = Median(gpu), Samples = wall, GpuSamples = gpu,
                    Resources = resources, Device = lane.Device.Name, lane.Device.DriverVersion });
            }
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new { Timestamp = DateTimeOffset.UtcNow,
            SplitReduction = splitOnly, SplitAbsoluteTolerance = splitOnly ? 1e-6 : 0,
            Note = "FP32 DKV; resident inputs; nonzero accumulation. Unsplit candidates preserve FMA order and compare exact bits; split candidates change association and use the stated absolute tolerance, without changing existing tests. 3 warmup + 10 measured launches. Full-array baseline comparison first at T128/T1024, dense/causal, head slices crossing batches, two accumulated launches, and a non-finite causal case.",
            Results = results }, new JsonSerializerOptions { WriteIndented = true });
    }

    private static (string Kernel, int Keys) Plan(Candidate candidate, bool causal)
    {
        string suffix = causal ? "causal" : "dense";
        return candidate switch {
            Candidate.Reference => ("attention_dkv_fused_candidate", 32),
            Candidate.K32Q32 => ($"attention_dkv_d32_k32_q32_{suffix}", 32),
            Candidate.K64Q32 => ($"attention_dkv_d32_k64_q32_{suffix}", 64),
            Candidate.K32Q64 => ($"attention_dkv_d32_k32_q64_{suffix}", 32),
            Candidate.K64Q16 => ($"attention_dkv_d32_k64_q16_{suffix}", 64),
            Candidate.Heads16K64Q32 => ($"attention_dkv_d32_h16_k64_q32_{suffix}", 64),
            Candidate.RegisterOwner => ($"attention_dkv_owner_{suffix}", 16),
            Candidate.Register8 => ($"attention_dkv_reg8_{suffix}", 32),
            Candidate.Register4 => ($"attention_dkv_reg4_{suffix}", 32),
            Candidate.Register2 => ($"attention_dkv_reg2_{suffix}", 32),
            Candidate.Swizzle16 => ($"attention_dkv_swizzle16_{suffix}", 32),
            Candidate.Swizzle8 => ($"attention_dkv_swizzle8_{suffix}", 32),
            Candidate.Swizzle4 => ($"attention_dkv_swizzle4_{suffix}", 32),
            Candidate.Unroll1 => ($"attention_dkv_unroll1_{suffix}", 32),
            Candidate.Unroll4 => ($"attention_dkv_unroll4_{suffix}", 32),
            Candidate.Unroll8 => ($"attention_dkv_unroll8_{suffix}", 32),
            Candidate.Unroll16 => ($"attention_dkv_unroll16_{suffix}", 32),
            Candidate.K16Q16 => ($"attention_dkv_d32_k16_q16_{suffix}", 16),
            Candidate.K16Q32 => ($"attention_dkv_d32_k16_q32_{suffix}", 16),
            Candidate.K16Q64 => ($"attention_dkv_d32_k16_q64_{suffix}", 16),
            Candidate.Rq8 => ($"attention_dkv_rq8_{suffix}", 32),
            Candidate.Rq16 => ($"attention_dkv_rq16_{suffix}", 32),
            Candidate.Rq32 => ($"attention_dkv_rq32_{suffix}", 32),
            Candidate.Rq16W128 => ($"attention_dkv_rq16_w128_{suffix}", 32),
            Candidate.BlockSlm => ($"attention_dkv_block_slm_{suffix}", 32),
            Candidate.BlockSlmK32Q16 => ($"attention_dkv_block_slm_k32q16_{suffix}", 32),
            Candidate.BlockSlmK64Q16 => ($"attention_dkv_block_slm_k64q16_{suffix}", 64),
            Candidate.BlockSlmK64Q32 => ($"attention_dkv_block_slm_k64q32_{suffix}", 64),
            Candidate.BlockSlmK128Q16 => ($"attention_dkv_block_slm_k128q16_{suffix}", 128),
            Candidate.Split2 or Candidate.Split4 => ("attention_dkv_split_partial", 32),
            _ => throw new ArgumentOutOfRangeException(nameof(candidate)),
        };
    }

    private static void Dispatch(ArcExecutionLane lane, Candidate candidate, bool causal,
        ArcBuffer qkv, ArcBuffer dy, ArcBuffer p, ArcBuffer ds, ArcBuffer dx,
        int sequence, int width, int heads, int first, int count)
    {
        if (width != heads * 32 || sequence % 64 != 0 || (candidate == Candidate.Heads16K64Q32 && heads != 16))
            throw new ArgumentException("Specialized DKV requires D32 and sequence aligned to64; fixed-head kernel requires H16.");
        var (kernel, keys) = Plan(candidate, causal);
        if (candidate is Candidate.Split2 or Candidate.Split4)
        {
            int splits = candidate == Candidate.Split2 ? 2 : 4;
            using var parts = lane.Allocate(checked(count*splits*sequence*64));
            lane.Run3D(kernel,16,sequence/32L*16,count*splits,16,16,1,
                qkv,dy,p,ds,parts,sequence,width,heads,first,count,splits,causal?1:0);
            lane.Run("attention_dkv_split_finish",count*sequence*32L,256,
                parts,dx,sequence,width,heads,first,count,splits);
            return;
        }
        int rows = candidate switch { Candidate.Register8 or Candidate.Swizzle8 or Candidate.Rq16W128 => 8, Candidate.Register4 or Candidate.Swizzle4 => 4, Candidate.Register2 => 2, _ => 16 };
        lane.Run3D(kernel, 16, (sequence / keys) * (long)rows, count, 16, rows, 1,
            qkv, dy, p, ds, dx, sequence, width, heads, first, causal ? 1 : 0);
    }

    private static void Validate(ArcExecutionLane lane, Candidate[] candidates)
    {
        foreach (int sequence in new[] { 128, 1024 })
        foreach (int heads in new[] { 3, 16 })
        foreach (bool causal in new[] { false, true })
            Compare(lane, sequence, heads, causal, false, candidates);
        Compare(lane, 128, 3, true, true, candidates);
        Console.WriteLine("DKV full-array validation passed: T128/T1024, causal/dense, partial batch/head slices, two accumulations, and non-finite causal start.");
    }

    private static void Compare(ArcExecutionLane lane, int sequence, int heads, bool causal, bool poisonInput, Candidate[] candidates)
    {
        const int batch = 2;
        int width = heads * 32, first = heads == 3 ? 1 : 13, count = heads == 3 ? 4 : 8;
        float[] qkvValues = Values(batch * sequence * width * 3, .013f, .13f);
        float[] dyValues = Values(batch * sequence * width, .017f, .03f);
        if (poisonInput)
        {
            // Reference higher32 key tiles never evaluate these early queries.
            // A64-key implementation must not introduce extra zero*NaN FMAs.
            qkvValues[first * 32] = float.NaN;
            dyValues[3 * width + first * 32 + 1] = float.NaN;
        }
        var (probabilities, derivatives) = Scores(count, sequence, causal);
        float[] initial = Enumerable.Range(0, qkvValues.Length).Select(i => .001f + (i % 7) * .0003f).ToArray();
        using var qkv = lane.Upload(qkvValues); using var dy = lane.Upload(dyValues);
        using var p = lane.Upload(probabilities); using var ds = lane.Upload(derivatives);
        float[]? expected = null;
        foreach (Candidate candidate in candidates)
        {
            if (candidate == Candidate.Heads16K64Q32 && heads != 16) continue;
            using var dx = lane.Upload(initial);
            long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
            for (int repeat = 0; repeat < 2; repeat++)
                Dispatch(lane, candidate, causal, qkv, dy, p, ds, dx, sequence, width, heads, first, count);
            if (h2d != lane.H2DBytes || d2h != lane.D2HBytes)
                throw new InvalidOperationException("DKV validation encountered an implicit transfer.");
            float[] actual = new float[initial.Length]; lane.Read(dx, actual);
            if (expected is null) expected = actual;
            else for (int i = 0; i < actual.Length; i++)
            {
                bool equal = float.IsNaN(expected[i]) ? float.IsNaN(actual[i])
                    : BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]);
                if (candidate is Candidate.Split2 or Candidate.Split4 && float.IsFinite(expected[i]) && float.IsFinite(actual[i]))
                    equal = MathF.Abs(expected[i]-actual[i]) <= 1e-6f;
                if (!equal) throw new InvalidOperationException($"DKV {candidate}: T{sequence}, H{heads}, causal={causal}, poison={poisonInput}, element {i}: {expected[i]} != {actual[i]}.");
            }
        }
    }

    private static float[] Values(int length, float frequency, float scale)
        => Enumerable.Range(0, length).Select(i => MathF.Sin(i * frequency) * scale).ToArray();

    private static (float[] P, float[] Ds) Scores(int count, int sequence, bool causal)
    {
        var p = new float[checked(count * sequence * sequence)]; var ds = new float[p.Length];
        for (int head = 0; head < count; head++)
        for (int query = 0; query < sequence; query++)
        for (int key = 0; key < sequence; key++)
        {
            int i = (head * sequence + query) * sequence + key;
            if (!causal || key <= query)
            {
                p[i] = ((key + head) % 5 + 1f) / sequence;
                ds[i] = MathF.Sin(i * .019f) * .003f;
            }
            else if (key >= ((query / 64) + 1) * 64)
                p[i] = ds[i] = float.NaN; // Outside the producer's published causal band.
        }
        return (p, ds);
    }
}
