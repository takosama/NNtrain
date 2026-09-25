using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Benchmarks;

// Resident-input A/B for the physical-BF16 T2048/D32 mix8_16 attention path.
// This probes individual kernels; a full training update decides dispatch.
internal static class ArcOverallAttentionProbe
{
    private const int Sequence = 2048;
    private const int Heads = 16;
    private const int Width = 512;
    private const int First = 0;
    private const int Causal = 2;
    private const int QkvElements = Sequence * 3 * Width;
    private const int ActivationElements = Sequence * Width;
    private const int Warmups = 3;
    private const int Samples = 15;

    private const string FreshSoftmax = "attention_probabilities";
    private const string NativeFreshSoftmax = "attention_probabilities_native_exp_2048";
    private const string SavedSoftmax = "attention_probabilities_saved_direct_2048";
    private const string NativeSavedSoftmax = "attention_probabilities_saved_native_exp_2048";
    private const string Pv = "attention_mix8_16_bf16_pv_d32_2048";
    private const string PvM128 = "attention_mix8_16_bf16_pv_m128_d32_2048";
    private const string PvPackedB = "attention_mix8_16_bf16_pv_bpair_d32_2048";
    private const string DpDs = "attention_mix8_16_bf16_dpds_d32_2048";
    private const string DpDsK32 = "attention_mix8_16_bf16_dpds_k32_d32_2048";
    private const string Dq = "attention_mix8_16_bf16_dq_packed_2048";
    private const string DqM128 = "attention_mix8_16_bf16_dq_packed_m128_2048";
    private const string DqPackedB = "attention_mix8_16_bf16_dq_packed_bpair_2048";
    private const string DqPackedAb = "attention_mix8_16_bf16_dq_packed_ab_2048";
    private const string Dkv = "attention_mix8_16_bf16_dkv_packed_2048";
    private const string DkvSlmUint = "attention_mix8_16_bf16_dkv_packed_slm_uint_2048";

    private sealed record Difference(long BitMismatches, double MaxAbsoluteError, double RelativeRms);
    private sealed record Timing(double GpuP50Ms, double WallP50Ms,
        double[] GpuSamplesMs, double[] WallSamplesMs);
    private sealed record PairTiming(Timing Baseline, Timing Candidate,
        double GpuReductionPercent, double WallReductionPercent);

    internal static void Run(string path, int count = 2)
    {
        if (count is < 1 or > 4)
            throw new ArgumentOutOfRangeException(nameof(count), "Use 1 to 4 resident heads.");
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Probe output must be a new file.");
        using var lane = new ArcExecutionLane(options: new() { ExperimentalOptimizationKernels = true });
        if (!lane.Device.SupportsXmx || lane.Device.MinimumSubgroupSize != 16
            || !lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroup_local_block_io"))
            throw new NotSupportedException("T2048 mix8_16 candidates require SG16, Intel XMX and subgroup local block I/O.");

        int rows = checked(count * Sequence), scoreElements = checked(rows * Sequence);
        var results = new List<object>();
        var rawScores = new float[scoreElements];
        for (int i = 0; i < rawScores.Length; ++i)
            rawScores[i] = ((i * 17L & 1023) - 512) * .0002f;
        var qkvValues = new ushort[QkvElements];
        for (int i = 0; i < qkvValues.Length; ++i)
            qkvValues[i] = Bf16(((i * 13L & 1023) - 512) * .00015f);
        var dyValues = new float[ActivationElements];
        for (int i = 0; i < dyValues.Length; ++i)
            dyValues[i] = ((i * 19L & 1023) - 512) * .0001f;
        var dxInitial = new float[QkvElements];
        for (int i = 0; i < dxInitial.Length; ++i)
            dxInitial[i] = .001f + i % 7 * .0003f;
        var pvInitial = new float[ActivationElements];
        for (int i = 0; i < pvInitial.Length; ++i)
            pvInitial[i] = .002f + i % 11 * .0001f;
        var deltaValues = new float[rows];
        for (int i = 0; i < deltaValues.Length; ++i)
            deltaValues[i] = ((i * 29L & 127) - 64) * .00001f;

        using var sourceScores = lane.Upload(rawScores);
        using var referenceScores = lane.Upload(rawScores);
        using var candidateScores = lane.Upload(rawScores);
        using var referenceStats = lane.Allocate(rows * 2);
        using var candidateStats = lane.Allocate(rows * 2);
        using var probabilities = lane.Allocate(scoreElements);
        using var qkv = lane.UploadRaw(qkvValues);
        using var dy = lane.Upload(dyValues);
        using var delta = lane.Upload(deltaValues);
        using var sourcePv = lane.Upload(pvInitial);
        using var referencePv = lane.Upload(pvInitial);
        using var candidatePv = lane.Upload(pvInitial);
        using var packedBPv = lane.Upload(pvInitial);
        using var referencePacked = lane.Allocate(scoreElements);
        using var candidatePacked = lane.Allocate(scoreElements);
        using var sourcePacked = lane.Allocate(scoreElements);
        using var sourceDx = lane.Upload(dxInitial);
        using var referenceDx = lane.Upload(dxInitial);
        using var candidateDx = lane.Upload(dxInitial);
        using var packedBDx = lane.Upload(dxInitial);
        using var packedAbDx = lane.Upload(dxInitial);

        void Copy(ArcBuffer source, ArcBuffer target, int elements)
            => lane.CopyBytes(source, target, 0, 0, checked(elements * sizeof(float)));
        float[] Read(ArcBuffer buffer, int elements)
        {
            var values = new float[elements];
            lane.Read(buffer, values);
            return values;
        }
        int[] ReadPacked(ArcBuffer buffer)
        {
            var values = new int[scoreElements];
            lane.ReadRaw(buffer, values);
            return values;
        }
        void Probability(string kernel, ArcBuffer scores, ArcBuffer stats, bool saved)
            => lane.Run(kernel, rows * 64L, 64, scores, stats,
                Sequence, Width, Heads, First, Causal, saved ? 1 : 0);
        void RunPv(string kernel, ArcBuffer output, int rowsPerGroup)
            => lane.Run3D(kernel, 16, Sequence / (long)rowsPerGroup * 16, count,
                16, 16, 1, probabilities, qkv, output,
                Sequence, Width, Heads, First);
        void RunDpDs(string kernel, ArcBuffer packed)
            => lane.Run3D(kernel, Sequence / 64L * 16, Sequence / 64L * 16,
                count, 16, 16, 1, dy, qkv, packed, delta,
                Sequence, Width, Heads, First);
        void RunDq(string kernel, ArcBuffer dx, int rowsPerGroup)
            => lane.Run3D(kernel, 16, Sequence / (long)rowsPerGroup * 16, count,
                16, 16, 1, sourcePacked, qkv, dx,
                Sequence, Width, Heads, First);
        void RunDkv(string kernel, ArcBuffer dx)
            => lane.Run3D(kernel, 16, Sequence / 32L * 16, count,
                16, 16, 1, qkv, dy, sourcePacked, dx,
                Sequence, Width, Heads, First);
        void Record(string stage, string baseline, string candidate,
            Dictionary<string, Difference> differences,
            Action resetBaseline, Action dispatchBaseline,
            Action resetCandidate, Action dispatchCandidate)
        {
            var resources = new[] { lane.GetKernelResources(baseline), lane.GetKernelResources(candidate) };
            PairTiming timing = MeasurePair(lane, resetBaseline, dispatchBaseline,
                resetCandidate, dispatchCandidate);
            Console.WriteLine($"{stage}: baseline GPU {timing.Baseline.GpuP50Ms:F4} ms, "
                + $"candidate GPU {timing.Candidate.GpuP50Ms:F4} ms "
                + $"({timing.GpuReductionPercent:+0.00;-0.00;0.00}%), "
                + $"wall {timing.Baseline.WallP50Ms:F4}->{timing.Candidate.WallP50Ms:F4} ms, "
                + $"bit mismatches {differences.Values.Sum(d => d.BitMismatches)}");
            results.Add(new { Stage = stage, BaselineKernel = baseline, CandidateKernel = candidate,
                Differences = differences, Timing = timing, Resources = resources });
        }

        // A fresh reference pass supplies both the saved row statistics and
        // the exact probability input shared by PV and dP/dS probes.
        Probability(FreshSoftmax, referenceScores, referenceStats, saved: false);
        Probability(NativeFreshSoftmax, candidateScores, candidateStats, saved: false);
        lane.Synchronize();
        var freshDifferences = new Dictionary<string, Difference>
        {
            ["scores"] = Compare(Read(referenceScores, scoreElements), Read(candidateScores, scoreElements)),
            ["stats"] = Compare(Read(referenceStats, rows * 2), Read(candidateStats, rows * 2)),
        };
        Copy(referenceScores, probabilities, scoreElements);
        lane.Synchronize();
        Record("softmax-fresh", FreshSoftmax, NativeFreshSoftmax, freshDifferences,
            () => Copy(sourceScores, referenceScores, scoreElements),
            () => Probability(FreshSoftmax, referenceScores, referenceStats, saved: false),
            () => Copy(sourceScores, candidateScores, scoreElements),
            () => Probability(NativeFreshSoftmax, candidateScores, candidateStats, saved: false));

        Copy(sourceScores, referenceScores, scoreElements);
        Copy(sourceScores, candidateScores, scoreElements);
        lane.Synchronize();
        Probability(SavedSoftmax, referenceScores, referenceStats, saved: true);
        Probability(NativeSavedSoftmax, candidateScores, referenceStats, saved: true);
        lane.Synchronize();
        Record("softmax-saved", SavedSoftmax, NativeSavedSoftmax,
            new() { ["scores"] = Compare(Read(referenceScores, scoreElements), Read(candidateScores, scoreElements)) },
            () => Copy(sourceScores, referenceScores, scoreElements),
            () => Probability(SavedSoftmax, referenceScores, referenceStats, saved: true),
            () => Copy(sourceScores, candidateScores, scoreElements),
            () => Probability(NativeSavedSoftmax, candidateScores, referenceStats, saved: true));

        RunPv(Pv, referencePv, 64);
        RunPv(PvM128, candidatePv, 128);
        lane.Synchronize();
        Record("PV", Pv, PvM128,
            new() { ["output"] = Compare(Read(referencePv, ActivationElements), Read(candidatePv, ActivationElements)) },
            () => Copy(sourcePv, referencePv, ActivationElements), () => RunPv(Pv, referencePv, 64),
            () => Copy(sourcePv, candidatePv, ActivationElements), () => RunPv(PvM128, candidatePv, 128));

        Copy(sourcePv, referencePv, ActivationElements);
        Copy(sourcePv, packedBPv, ActivationElements);
        lane.Synchronize();
        RunPv(Pv, referencePv, 64);
        RunPv(PvPackedB, packedBPv, 64);
        lane.Synchronize();
        Record("PV-packed-B-SLM", Pv, PvPackedB,
            new() { ["output"] = Compare(Read(referencePv, ActivationElements), Read(packedBPv, ActivationElements)) },
            () => Copy(sourcePv, referencePv, ActivationElements), () => RunPv(Pv, referencePv, 64),
            () => Copy(sourcePv, packedBPv, ActivationElements), () => RunPv(PvPackedB, packedBPv, 64));

        Copy(probabilities, referencePacked, scoreElements);
        Copy(probabilities, candidatePacked, scoreElements);
        lane.Synchronize();
        RunDpDs(DpDs, referencePacked);
        RunDpDs(DpDsK32, candidatePacked);
        lane.Synchronize();
        var dpDsDifference = ComparePacked(ReadPacked(referencePacked), ReadPacked(candidatePacked));
        Copy(referencePacked, sourcePacked, scoreElements);
        lane.Synchronize();
        Record("dP+dS", DpDs, DpDsK32,
            new() { ["packed-P-dS"] = dpDsDifference },
            () => Copy(probabilities, referencePacked, scoreElements), () => RunDpDs(DpDs, referencePacked),
            () => Copy(probabilities, candidatePacked, scoreElements), () => RunDpDs(DpDsK32, candidatePacked));

        Copy(sourceDx, referenceDx, QkvElements);
        Copy(sourceDx, candidateDx, QkvElements);
        lane.Synchronize();
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            RunDq(Dq, referenceDx, 64);
            RunDq(DqM128, candidateDx, 128);
        }
        lane.Synchronize();
        Record("dQ", Dq, DqM128,
            new() { ["dx-two-accumulations"] = Compare(Read(referenceDx, QkvElements), Read(candidateDx, QkvElements)) },
            () => Copy(sourceDx, referenceDx, QkvElements), () => RunDq(Dq, referenceDx, 64),
            () => Copy(sourceDx, candidateDx, QkvElements), () => RunDq(DqM128, candidateDx, 128));

        Copy(sourceDx, referenceDx, QkvElements);
        Copy(sourceDx, packedBDx, QkvElements);
        lane.Synchronize();
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            RunDq(Dq, referenceDx, 64);
            RunDq(DqPackedB, packedBDx, 64);
        }
        lane.Synchronize();
        Record("dQ-packed-B-SLM", Dq, DqPackedB,
            new() { ["dx-two-accumulations"] = Compare(Read(referenceDx, QkvElements), Read(packedBDx, QkvElements)) },
            () => Copy(sourceDx, referenceDx, QkvElements), () => RunDq(Dq, referenceDx, 64),
            () => Copy(sourceDx, packedBDx, QkvElements), () => RunDq(DqPackedB, packedBDx, 64));

        Copy(sourceDx, referenceDx, QkvElements);
        Copy(sourceDx, packedAbDx, QkvElements);
        lane.Synchronize();
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            RunDq(Dq, referenceDx, 64);
            RunDq(DqPackedAb, packedAbDx, 64);
        }
        lane.Synchronize();
        Record("dQ-packed-A+B-SLM", Dq, DqPackedAb,
            new() { ["dx-two-accumulations"] = Compare(Read(referenceDx, QkvElements), Read(packedAbDx, QkvElements)) },
            () => Copy(sourceDx, referenceDx, QkvElements), () => RunDq(Dq, referenceDx, 64),
            () => Copy(sourceDx, packedAbDx, QkvElements), () => RunDq(DqPackedAb, packedAbDx, 64));

        Copy(sourceDx, referenceDx, QkvElements);
        Copy(sourceDx, candidateDx, QkvElements);
        lane.Synchronize();
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            RunDkv(Dkv, referenceDx);
            RunDkv(DkvSlmUint, candidateDx);
        }
        lane.Synchronize();
        Record("dK+dV", Dkv, DkvSlmUint,
            new() { ["dx-two-accumulations"] = Compare(Read(referenceDx, QkvElements), Read(candidateDx, QkvElements)) },
            () => Copy(sourceDx, referenceDx, QkvElements), () => RunDkv(Dkv, referenceDx),
            () => Copy(sourceDx, candidateDx, QkvElements), () => RunDkv(DkvSlmUint, candidateDx));

        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new
        {
            Timestamp = DateTimeOffset.UtcNow, Device = lane.Device.Name, lane.Device.DriverVersion,
            Sequence, Heads, Width, ResidentHeadCount = count, First, Causal, Warmups, Samples,
            Note = "Physical BF16 QKV, packed BF16 P/dS, T2048/D32. Correctness compares complete stage outputs; "
                + "dQ and dK/dV compare two accumulations. Resets are synchronized outside each timed window. "
                + "Order alternates baseline/candidate by round. No host/device transfers inside timings. "
                + "Packed mismatches count 32-bit words; numeric error covers causal entries only.",
            Results = results,
        }, new JsonSerializerOptions { WriteIndented = true });
    }

    private static PairTiming MeasurePair(ArcExecutionLane lane,
        Action resetBaseline, Action dispatchBaseline,
        Action resetCandidate, Action dispatchCandidate)
    {
        var baselineGpu = new List<double>(Samples);
        var baselineWall = new List<double>(Samples);
        var candidateGpu = new List<double>(Samples);
        var candidateWall = new List<double>(Samples);
        long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
        void Sample(Action reset, Action dispatch, List<double> gpu, List<double> wall, bool measured)
        {
            reset();
            lane.Synchronize();
            double beforeGpu = lane.KernelMilliseconds;
            long started = Stopwatch.GetTimestamp();
            dispatch();
            lane.Synchronize();
            if (measured)
            {
                gpu.Add(lane.KernelMilliseconds - beforeGpu);
                wall.Add(Stopwatch.GetElapsedTime(started).TotalMilliseconds);
            }
        }
        for (int round = 0; round < Warmups + Samples; ++round)
        {
            bool measured = round >= Warmups;
            if ((round & 1) == 0)
            {
                Sample(resetBaseline, dispatchBaseline, baselineGpu, baselineWall, measured);
                Sample(resetCandidate, dispatchCandidate, candidateGpu, candidateWall, measured);
            }
            else
            {
                Sample(resetCandidate, dispatchCandidate, candidateGpu, candidateWall, measured);
                Sample(resetBaseline, dispatchBaseline, baselineGpu, baselineWall, measured);
            }
        }
        if (lane.H2DBytes != h2d || lane.D2HBytes != d2h)
            throw new InvalidOperationException("Timed attention probe crossed the host/device boundary.");
        var baseline = new Timing(Median(baselineGpu), Median(baselineWall),
            baselineGpu.ToArray(), baselineWall.ToArray());
        var candidate = new Timing(Median(candidateGpu), Median(candidateWall),
            candidateGpu.ToArray(), candidateWall.ToArray());
        return new PairTiming(baseline, candidate,
            100.0 * (baseline.GpuP50Ms - candidate.GpuP50Ms) / baseline.GpuP50Ms,
            100.0 * (baseline.WallP50Ms - candidate.WallP50Ms) / baseline.WallP50Ms);
    }

    private static Difference Compare(float[] expected, float[] actual)
    {
        if (expected.Length != actual.Length) throw new InvalidOperationException("Attention output length mismatch.");
        long mismatches = 0;
        double maxAbs = 0, squareError = 0, squareReference = 0;
        for (int i = 0; i < expected.Length; ++i)
        {
            float left = expected[i], right = actual[i];
            if (!float.IsFinite(left) || !float.IsFinite(right))
                throw new InvalidOperationException($"Nonfinite attention output at element {i}.");
            if (BitConverter.SingleToInt32Bits(left) != BitConverter.SingleToInt32Bits(right)) ++mismatches;
            double error = (double)right - left;
            squareError += error * error;
            squareReference += (double)left * left;
            maxAbs = Math.Max(maxAbs, Math.Abs(error));
        }
        return new Difference(mismatches, maxAbs, Math.Sqrt(squareError / Math.Max(squareReference, double.Epsilon)));
    }

    private static Difference ComparePacked(int[] expected, int[] actual)
    {
        if (expected.Length != actual.Length) throw new InvalidOperationException("Packed attention output length mismatch.");
        long mismatches = 0;
        double maxAbs = 0, squareError = 0, squareReference = 0;
        for (int i = 0; i < expected.Length; ++i)
        {
            if (expected[i] != actual[i]) ++mismatches;
            int query = i / Sequence % Sequence, key = i % Sequence;
            if (key > query) continue;
            uint leftBits = unchecked((uint)expected[i]), rightBits = unchecked((uint)actual[i]);
            for (int half = 0; half < 2; ++half)
            {
                int shift = half * 16;
                float left = BitConverter.Int32BitsToSingle(unchecked((int)((leftBits >> shift) << 16)));
                float right = BitConverter.Int32BitsToSingle(unchecked((int)((rightBits >> shift) << 16)));
                if (!float.IsFinite(left) || !float.IsFinite(right))
                    throw new InvalidOperationException($"Nonfinite packed attention output at element {i}, half {half}.");
                double error = (double)right - left;
                squareError += error * error;
                squareReference += (double)left * left;
                maxAbs = Math.Max(maxAbs, Math.Abs(error));
            }
        }
        return new Difference(mismatches, maxAbs, Math.Sqrt(squareError / Math.Max(squareReference, double.Epsilon)));
    }

    private static ushort Bf16(float value)
    {
        uint bits = unchecked((uint)BitConverter.SingleToInt32Bits(value));
        return (ushort)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
    }

    private static double Median(List<double> values)
    {
        double[] sorted = values.Order().ToArray();
        return sorted[sorted.Length / 2];
    }
}
