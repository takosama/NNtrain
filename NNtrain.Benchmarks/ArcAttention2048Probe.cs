using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Benchmarks;

// A narrow, resident-input probe for the actual T2048/D32/H16 attention tile.
// The complete training probe is still required before changing dispatch defaults.
internal static class ArcAttention2048Probe
{
    private const int Sequence = 2048;
    private const int Heads = 16;
    private const int HeadWidth = 32;
    private const int Width = Heads * HeadWidth;
    private const int Causal = 2;
    private const string BaselineDp = "attention_fp32_dp_d32_aligned";
    private const string FusedDpDsKernel = "attention_dp_ds_two_pass_d32_2048_candidate";
    private const string BaselineDkv = "attention_dkv_block_slm_causal";

    private sealed record Difference(long MismatchedBits, double MaxAbsoluteError);
    private sealed record Timing(double WallP50Ms, double GpuP50Ms,
        double[] WallSamplesMs, double[] GpuSamplesMs);

    internal static void Run(string path, int count)
    {
        if (count is < 1 or > 8)
            throw new ArgumentOutOfRangeException(nameof(count), "Use 1 to 8 resident heads.");
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Probe output must be a new file.");

        using var lane = new ArcExecutionLane(options: new() { ExperimentalOptimizationKernels = true });
        if (!lane.Device.SupportsXmx || lane.Device.MinimumSubgroupSize != 16
            || !lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroup_local_block_io"))
            throw new NotSupportedException("T2048 attention candidates require SG16, Intel XMX and subgroup local block I/O.");
        int rows = checked(count * Sequence);
        int elements = checked(rows * Sequence);
        var results = new List<object>();

        float[] raw = new float[elements];
        float[] dpInput = new float[elements];
        for (int i = 0; i < elements; ++i)
        {
            raw[i] = ((i * 17L & 1023) - 512) * .0002f;
            dpInput[i] = ((i * 31L & 2047) - 1024) * .0001f;
        }
        using var sourceScores = lane.Upload(raw);
        using var referenceScores = lane.Upload(raw);
        using var candidateScores = lane.Upload(raw);
        using var slmScores = lane.Upload(raw);
        using var directScores = lane.Upload(raw);
        using var referenceStats = lane.Allocate(count * Sequence * 2);
        using var candidateStats = lane.Allocate(count * Sequence * 2);
        using var slmStats = lane.Allocate(count * Sequence * 2);
        using var sourceDp = lane.Upload(dpInput);
        using var referenceDp = lane.Upload(dpInput);
        using var candidateDp = lane.Upload(dpInput);
        using var slmDp = lane.Upload(dpInput);
        using var fixedDp = lane.Upload(dpInput);
        using var prefixDp = lane.Upload(dpInput);
        using var prefix4Dp = lane.Upload(dpInput);
        using var prefix4SubgroupDp = lane.Upload(dpInput);
        using var prefix8Dp = lane.Upload(dpInput);
        using var prefix16Dp = lane.Upload(dpInput);

        void Probability(string kernel, ArcBuffer scores, ArcBuffer stats, bool saved)
            => lane.Run(kernel, rows * 64L, 64, scores, stats,
                Sequence, Width, Heads, 0, Causal, saved ? 1 : 0);
        void Derivative(string kernel, ArcBuffer dp)
            => lane.Run(kernel, rows * 64L, 64, referenceScores, dp,
                Sequence, Width, Heads, Causal);
        void Reset(ArcBuffer source, ArcBuffer target, int floats)
            => lane.CopyBytes(source, target, 0, 0, checked(floats * sizeof(float)));
        float[] Read(ArcBuffer buffer, int length)
        {
            var values = new float[length];
            lane.Read(buffer, values);
            return values;
        }
        void Record(string stage, string candidate, string[] kernels, Difference difference,
            Action reset, Action dispatch)
        {
            Timing timing = Measure(lane, reset, dispatch);
            Console.WriteLine($"{stage} {candidate}: GPU p50 {timing.GpuP50Ms:F4} ms, "
                + $"wall p50 {timing.WallP50Ms:F4} ms, bit mismatches {difference.MismatchedBits}, "
                + $"max abs {difference.MaxAbsoluteError:G6}");
            results.Add(new
            {
                Stage = stage, Candidate = candidate, Kernels = kernels,
                Timing = timing, Difference = difference,
                Resources = kernels.Select(kernel => new { Kernel = kernel, Values = lane.GetKernelResources(kernel) }).ToArray(),
            });
        }

        Probability("attention_probabilities", referenceScores, referenceStats, saved: false);
        Probability("attention_probabilities_register_2048", candidateScores, candidateStats, saved: false);
        Probability("attention_probabilities_slm_2048", slmScores, slmStats, saved: false);
        Difference fresh = Compare(Read(referenceScores, elements), Read(candidateScores, elements));
        Difference freshSlm = Compare(Read(referenceScores, elements), Read(slmScores, elements));
        Difference freshStats = Compare(Read(referenceStats, count * Sequence * 2),
            Read(candidateStats, count * Sequence * 2));
        Difference freshSlmStats = Compare(Read(referenceStats, count * Sequence * 2),
            Read(slmStats, count * Sequence * 2));
        Console.WriteLine($"fresh stats: bit mismatches {freshStats.MismatchedBits}, max abs {freshStats.MaxAbsoluteError:G6}");
        Console.WriteLine($"SLM fresh stats: bit mismatches {freshSlmStats.MismatchedBits}, max abs {freshSlmStats.MaxAbsoluteError:G6}");
        Record("probabilities-fresh", "reference", ["attention_probabilities"], new Difference(0, 0),
            () => Reset(sourceScores, referenceScores, elements),
            () => Probability("attention_probabilities", referenceScores, referenceStats, saved: false));
        Record("probabilities-fresh", "row-register", ["attention_probabilities_register_2048"], fresh,
            () => Reset(sourceScores, candidateScores, elements),
            () => Probability("attention_probabilities_register_2048", candidateScores, candidateStats, saved: false));
        Record("probabilities-fresh", "slm-cache", ["attention_probabilities_slm_2048"], freshSlm,
            () => Reset(sourceScores, slmScores, elements),
            () => Probability("attention_probabilities_slm_2048", slmScores, slmStats, saved: false));

        Reset(sourceScores, referenceScores, elements);
        Reset(sourceScores, candidateScores, elements);
        Reset(sourceScores, slmScores, elements);
        Reset(sourceScores, directScores, elements);
        Probability("attention_probabilities", referenceScores, referenceStats, saved: true);
        Probability("attention_probabilities_register_2048", candidateScores, candidateStats, saved: true);
        Probability("attention_probabilities_slm_2048", slmScores, slmStats, saved: true);
        Probability("attention_probabilities_saved_direct_2048", directScores, referenceStats, saved: true);
        Difference savedProbability = Compare(Read(referenceScores, elements), Read(candidateScores, elements));
        Difference savedSlm = Compare(Read(referenceScores, elements), Read(slmScores, elements));
        Difference savedDirect = Compare(Read(referenceScores, elements), Read(directScores, elements));
        Record("probabilities-saved", "reference", ["attention_probabilities"], new Difference(0, 0),
            () => Reset(sourceScores, referenceScores, elements),
            () => Probability("attention_probabilities", referenceScores, referenceStats, saved: true));
        Record("probabilities-saved", "row-register", ["attention_probabilities_register_2048"], savedProbability,
            () => Reset(sourceScores, candidateScores, elements),
            () => Probability("attention_probabilities_register_2048", candidateScores, candidateStats, saved: true));
        Record("probabilities-saved", "slm-cache", ["attention_probabilities_slm_2048"], savedSlm,
            () => Reset(sourceScores, slmScores, elements),
            () => Probability("attention_probabilities_slm_2048", slmScores, slmStats, saved: true));
        Record("probabilities-saved", "direct", ["attention_probabilities_saved_direct_2048"], savedDirect,
            () => Reset(sourceScores, directScores, elements),
            () => Probability("attention_probabilities_saved_direct_2048", directScores, referenceStats, saved: true));

        Reset(sourceScores, referenceScores, elements);
        Probability("attention_probabilities", referenceScores, referenceStats, saved: false);
        Derivative("attention_derivatives", referenceDp);
        Derivative("attention_derivatives_register_2048", candidateDp);
        Derivative("attention_derivatives_slm_2048", slmDp);
        Derivative("attention_derivatives_fixed_2048", fixedDp);
        Derivative("attention_derivatives_prefix_2048", prefixDp);
        Derivative("attention_derivatives_prefix4_2048", prefix4Dp);
        Derivative("attention_derivatives_prefix4_sg16_2048", prefix4SubgroupDp);
        Derivative("attention_derivatives_prefix8_2048", prefix8Dp);
        Derivative("attention_derivatives_prefix16_2048", prefix16Dp);
        Difference derivative = Compare(Read(referenceDp, elements), Read(candidateDp, elements));
        Difference derivativeSlm = Compare(Read(referenceDp, elements), Read(slmDp, elements));
        Difference derivativeFixed = Compare(Read(referenceDp, elements), Read(fixedDp, elements));
        Difference derivativePrefix = Compare(Read(referenceDp, elements), Read(prefixDp, elements));
        Difference derivativePrefix4 = Compare(Read(referenceDp, elements), Read(prefix4Dp, elements));
        Difference derivativePrefix4Subgroup = Compare(Read(referenceDp, elements), Read(prefix4SubgroupDp, elements));
        Difference derivativePrefix8 = Compare(Read(referenceDp, elements), Read(prefix8Dp, elements));
        Difference derivativePrefix16 = Compare(Read(referenceDp, elements), Read(prefix16Dp, elements));
        Record("derivative", "reference", ["attention_derivatives"], new Difference(0, 0),
            () => Reset(sourceDp, referenceDp, elements),
            () => Derivative("attention_derivatives", referenceDp));
        Record("derivative", "row-register", ["attention_derivatives_register_2048"], derivative,
            () => Reset(sourceDp, candidateDp, elements),
            () => Derivative("attention_derivatives_register_2048", candidateDp));
        Record("derivative", "slm-cache", ["attention_derivatives_slm_2048"], derivativeSlm,
            () => Reset(sourceDp, slmDp, elements),
            () => Derivative("attention_derivatives_slm_2048", slmDp));
        Record("derivative", "fixed-shape", ["attention_derivatives_fixed_2048"], derivativeFixed,
            () => Reset(sourceDp, fixedDp, elements),
            () => Derivative("attention_derivatives_fixed_2048", fixedDp));
        Record("derivative", "first-pair-register", ["attention_derivatives_prefix_2048"], derivativePrefix,
            () => Reset(sourceDp, prefixDp, elements),
            () => Derivative("attention_derivatives_prefix_2048", prefixDp));
        Record("derivative", "first-four-pairs-register", ["attention_derivatives_prefix4_2048"], derivativePrefix4,
            () => Reset(sourceDp, prefix4Dp, elements),
            () => Derivative("attention_derivatives_prefix4_2048", prefix4Dp));
        Record("derivative", "first-four-pairs-subgroup", ["attention_derivatives_prefix4_sg16_2048"], derivativePrefix4Subgroup,
            () => Reset(sourceDp, prefix4SubgroupDp, elements),
            () => Derivative("attention_derivatives_prefix4_sg16_2048", prefix4SubgroupDp));
        Record("derivative", "first-eight-pairs-register", ["attention_derivatives_prefix8_2048"], derivativePrefix8,
            () => Reset(sourceDp, prefix8Dp, elements),
            () => Derivative("attention_derivatives_prefix8_2048", prefix8Dp));
        Record("derivative", "first-sixteen-pairs-register", ["attention_derivatives_prefix16_2048"], derivativePrefix16,
            () => Reset(sourceDp, prefix16Dp, elements),
            () => Derivative("attention_derivatives_prefix16_2048", prefix16Dp));

        float[] qkvInput = new float[Sequence * Width * 3];
        float[] dyInput = new float[Sequence * Width];
        for (int i = 0; i < qkvInput.Length; ++i)
            qkvInput[i] = ((i * 13L & 1023) - 512) * .00015f;
        for (int i = 0; i < dyInput.Length; ++i)
            dyInput[i] = ((i * 19L & 1023) - 512) * .0001f;
        using var qkv = lane.Upload(qkvInput);
        using var dy = lane.Upload(dyInput);
        using var sourceDs = lane.Upload(Enumerable.Repeat(.125f, elements).ToArray());
        using var referenceDs = lane.Allocate(elements);
        using var candidateDs = lane.Allocate(elements);
        void SeparateDpDs()
        {
            lane.Run3D(BaselineDp, Sequence / 64L * 16, Sequence / 64L * 16,
                count, 16, 16, 1, dy, qkv, referenceDs, Sequence, Sequence, HeadWidth,
                Width, 1, 0, Sequence * Width, HeadWidth, 0,
                1, 3 * Width, 0, Sequence * 3 * Width, HeadWidth, 2 * Width,
                Sequence, 1, Sequence * Sequence, 0, 0, 0,
                Heads, 0, 0, 1);
            lane.Run("attention_derivatives", rows * 64L, 64,
                referenceScores, referenceDs, Sequence, Width, Heads, Causal);
        }
        void FusedDpDs()
            => lane.Run2D(FusedDpDsKernel, Sequence / 16L * 256, count, 256, 1,
                qkv, dy, referenceScores, candidateDs, Sequence, Width, Heads, 0, Causal);
        Reset(sourceDs, referenceDs, elements);
        Reset(sourceDs, candidateDs, elements);
        SeparateDpDs();
        FusedDpDs();
        Difference dpDs = Compare(Read(referenceDs, elements), Read(candidateDs, elements));
        Record("dP+dS", "reference", [BaselineDp, "attention_derivatives"], new Difference(0, 0),
            () => Reset(sourceDs, referenceDs, elements), SeparateDpDs);
        Record("dP+dS", "two-pass-fused", [FusedDpDsKernel], dpDs,
            () => Reset(sourceDs, candidateDs, elements), FusedDpDs);

        float[] initialDx = new float[qkvInput.Length];
        for (int i = 0; i < initialDx.Length; ++i)
            initialDx[i] = .001f + i % 7 * .0003f;
        using var sourceDx = lane.Upload(initialDx);
        using var referenceDx = lane.Upload(initialDx);
        using var candidateDx = lane.Upload(initialDx);
        void Dkv(string kernel, ArcBuffer dx)
            => lane.Run3D(kernel, 16, Sequence / 32L * 16, count, 16, 16, 1,
                qkv, dy, referenceScores, referenceDs, dx,
                Sequence, Width, Heads, 0, 1);
        // Restore the reference dS after the timed comparisons above.
        Reset(sourceDs, referenceDs, elements);
        SeparateDpDs();
        Dkv(BaselineDkv, referenceDx);
        Record("dK+dV", "reference", [BaselineDkv], new Difference(0, 0),
            () => Reset(sourceDx, referenceDx, initialDx.Length),
            () => Dkv(BaselineDkv, referenceDx));
        foreach (int pitch in new[] { 32, 33 })
        {
            string kernel = $"attention_dkv_slm_qmajor{pitch}_causal";
            Reset(sourceDx, referenceDx, initialDx.Length);
            Dkv(BaselineDkv, referenceDx);
            Reset(sourceDx, candidateDx, initialDx.Length);
            Dkv(kernel, candidateDx);
            Difference difference = Compare(Read(referenceDx, initialDx.Length),
                Read(candidateDx, initialDx.Length));
            Record("dK+dV", $"query-major-pitch-{pitch}", [kernel], difference,
                () => Reset(sourceDx, candidateDx, initialDx.Length),
                () => Dkv(kernel, candidateDx));
        }
        foreach (int unroll in new[] { 4, 8, 16 })
        {
            string kernel = $"attention_dkv_block_slm_2048_unroll{unroll}_causal";
            Reset(sourceDx, candidateDx, initialDx.Length);
            Dkv(kernel, candidateDx);
            Difference difference = Compare(Read(referenceDx, initialDx.Length),
                Read(candidateDx, initialDx.Length));
            Record("dK+dV", $"inner-unroll-{unroll}", [kernel], difference,
                () => Reset(sourceDx, candidateDx, initialDx.Length),
                () => Dkv(kernel, candidateDx));
        }

        // These two products share the production [head, query, key] matrix
        // and the interleaved QKV layout. PV overwrites its output; dQ adds
        // into a pre-existing Q gradient. Compare untouched channels as well.
        using var sourcePv = lane.Upload(dyInput);
        using var referencePv = lane.Upload(dyInput);
        using var candidatePv = lane.Upload(dyInput);
        void Product(string kernel, int tile, bool backward, ArcBuffer destination)
        {
            ArcBuffer matrix = backward ? referenceDs : referenceScores;
            int outputComponents = backward ? 3 : 1;
            lane.Run3D(kernel, 16, Sequence / (long)tile * 16, count, 16, 16, 1,
                matrix, qkv, destination, Sequence, HeadWidth, Sequence,
                Sequence, 1, Sequence * Sequence, 0, 0, 0,
                3 * Width, 1, 0, Sequence * 3 * Width, HeadWidth,
                (backward ? 1 : 2) * Width,
                outputComponents * Width, 1, 0,
                Sequence * outputComponents * Width, HeadWidth, 0,
                Heads, 0, backward ? 1 : 0, Causal);
        }
        foreach (bool backward in new[] { false, true })
        {
            string operation = backward ? "dQ" : "PV";
            string suffix = backward ? "dq" : "pv";
            string baseline = $"attention_fp32_{suffix}_d32_aligned";
            ArcBuffer source = backward ? sourceDx : sourcePv;
            ArcBuffer reference = backward ? referenceDx : referencePv;
            ArcBuffer candidate = backward ? candidateDx : candidatePv;
            int outputLength = backward ? initialDx.Length : dyInput.Length;
            int repeats = backward ? 2 : 1;
            Reset(source, reference, outputLength);
            for (int i = 0; i < repeats; ++i) Product(baseline, 64, backward, reference);
            float[] expected = Read(reference, outputLength);
            Record(operation, "reference", [baseline], new Difference(0, 0),
                () => Reset(source, reference, outputLength),
                () => Product(baseline, 64, backward, reference));
            foreach (int tile in new[] { 32, 64 })
            {
                string kernel = $"attention_fp32_{suffix}_d32_t2048_m{tile}k{tile}";
                Reset(source, candidate, outputLength);
                for (int i = 0; i < repeats; ++i) Product(kernel, tile, backward, candidate);
                Difference difference = Compare(expected, Read(candidate, outputLength));
                Record(operation, $"m{tile}k{tile}", [kernel], difference,
                    () => Reset(source, candidate, outputLength),
                    () => Product(kernel, tile, backward, candidate));
            }
        }

        const string FusedBackwardKernel = "attention_dp_ds_dq4_t2048_candidate";
        const string BaselineDq = "attention_fp32_dq_d32_aligned";
        void SeparateBackward()
        {
            SeparateDpDs();
            Product(BaselineDq, 64, backward: true, referenceDx);
        }
        void FusedBackward()
            => lane.Run3D(FusedBackwardKernel, 32, Sequence, count, 32, 4, 1,
                qkv, dy, referenceScores, candidateDs, candidateDx,
                Sequence, Width, Heads, 0, Causal,
                new LocalMemory(4 * Sequence * sizeof(float)));
        Reset(sourceDs, referenceDs, elements);
        Reset(sourceDs, candidateDs, elements);
        Reset(sourceDx, referenceDx, initialDx.Length);
        Reset(sourceDx, candidateDx, initialDx.Length);
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            SeparateBackward();
            FusedBackward();
        }
        Difference backwardDs = Compare(Read(referenceDs, elements), Read(candidateDs, elements));
        Difference backwardDq = Compare(Read(referenceDx, initialDx.Length),
            Read(candidateDx, initialDx.Length));
        Difference backwardCombined = new(backwardDs.MismatchedBits + backwardDq.MismatchedBits,
            Math.Max(backwardDs.MaxAbsoluteError, backwardDq.MaxAbsoluteError));
        Console.WriteLine($"fused dP+dS+dQ: dS bit mismatches {backwardDs.MismatchedBits}, "
            + $"dQ bit mismatches {backwardDq.MismatchedBits}");
        Record("dP+dS+dQ", "reference", [BaselineDp, "attention_derivatives", BaselineDq],
            new Difference(0, 0),
            () => { Reset(sourceDs, referenceDs, elements); Reset(sourceDx, referenceDx, initialDx.Length); },
            SeparateBackward);
        Record("dP+dS+dQ", "fused-r4", [FusedBackwardKernel], backwardCombined,
            () => { Reset(sourceDs, candidateDs, elements); Reset(sourceDx, candidateDx, initialDx.Length); },
            FusedBackward);

        const string FusedForwardKernel = "attention_prob_pv_fused_t2048_r4_candidate";
        const string BaselinePv = "attention_fp32_pv_d32_aligned";
        using var fusedStats = lane.Allocate(count * Sequence * 2);
        void FusedForward()
            => lane.Run3D(FusedForwardKernel, 32, Sequence / 4L * 4, count,
                32, 4, 1, qkv, sourceScores, candidatePv, fusedStats,
                Sequence, Width, Heads, 0, 1);
        void SeparateForward()
        {
            Probability("attention_probabilities", referenceScores, referenceStats, saved: false);
            Product(BaselinePv, 64, backward: false, referencePv);
        }
        Reset(sourceScores, referenceScores, elements);
        Reset(sourcePv, referencePv, dyInput.Length);
        Reset(sourcePv, candidatePv, dyInput.Length);
        SeparateForward();
        FusedForward();
        Difference fusedPv = Compare(Read(referencePv, dyInput.Length),
            Read(candidatePv, dyInput.Length));
        Difference fusedStatsDifference = Compare(Read(referenceStats, count * Sequence * 2),
            Read(fusedStats, count * Sequence * 2));
        Console.WriteLine($"fused forward stats: bit mismatches {fusedStatsDifference.MismatchedBits}, "
            + $"max abs {fusedStatsDifference.MaxAbsoluteError:G6}");
        Record("probability+PV", "reference", ["attention_probabilities", BaselinePv],
            new Difference(0, 0),
            () => { Reset(sourceScores, referenceScores, elements); Reset(sourcePv, referencePv, dyInput.Length); },
            SeparateForward);
        Record("probability+PV", "fused-r4", [FusedForwardKernel], fusedPv,
            () => Reset(sourcePv, candidatePv, dyInput.Length), FusedForward);

        // Backward needs P even when forward's fused kernel keeps it in SLM.
        // Time its saved-stat regeneration as a second training-compatible
        // result, with output and P compared against the ordinary pair.
        Reset(sourceScores, referenceScores, elements);
        Reset(sourcePv, referencePv, dyInput.Length);
        SeparateForward();
        Reset(sourceScores, candidateScores, elements);
        Reset(sourcePv, candidatePv, dyInput.Length);
        FusedForward();
        Probability("attention_probabilities_saved_direct_2048", candidateScores,
            fusedStats, saved: true);
        Difference fusedP = Compare(Read(referenceScores, elements),
            Read(candidateScores, elements));
        Difference materialized = new(fusedPv.MismatchedBits + fusedP.MismatchedBits,
            Math.Max(fusedPv.MaxAbsoluteError, fusedP.MaxAbsoluteError));
        Record("probability+PV+P-materialization", "reference",
            ["attention_probabilities", BaselinePv], new Difference(0, 0),
            () => { Reset(sourceScores, referenceScores, elements); Reset(sourcePv, referencePv, dyInput.Length); },
            SeparateForward);
        Record("probability+PV+P-materialization", "fused-r4+saved-direct",
            [FusedForwardKernel, "attention_probabilities_saved_direct_2048"], materialized,
            () => { Reset(sourceScores, candidateScores, elements); Reset(sourcePv, candidatePv, dyInput.Length); },
            () => { FusedForward(); Probability("attention_probabilities_saved_direct_2048",
                candidateScores, fusedStats, saved: true); });

        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new
        {
            Timestamp = DateTimeOffset.UtcNow, Device = lane.Device.Name,
            lane.Device.DriverVersion, Sequence, Heads, HeadWidth, ResidentHeadCount = count,
            CausalMode = Causal, FreshStatsDifference = freshStats,
            SlmFreshStatsDifference = freshSlmStats,
            Note = "Resident FP32 attention internals at T2048/D32/H16. Three warmup plus ten timed samples; "
                + "device-side reset excluded, both GPU event time and launch+sync wall time reported. "
                + "Full output compared before timing. No transfers inside timings. Training throughput must be checked separately.",
            Results = results,
        }, new JsonSerializerOptions { WriteIndented = true });
    }

    private static Timing Measure(ArcExecutionLane lane, Action reset, Action dispatch)
    {
        var wall = new List<double>(10);
        var gpu = new List<double>(10);
        long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
        for (int sample = 0; sample < 13; ++sample)
        {
            reset();
            lane.Synchronize();
            double beforeGpu = lane.KernelMilliseconds;
            long start = Stopwatch.GetTimestamp();
            dispatch();
            lane.Synchronize();
            if (sample >= 3)
            {
                wall.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                gpu.Add(lane.KernelMilliseconds - beforeGpu);
            }
        }
        if (lane.H2DBytes != h2d || lane.D2HBytes != d2h)
            throw new InvalidOperationException("Timed attention kernel crossed the host/device boundary.");
        return new Timing(Median(wall), Median(gpu), wall.ToArray(), gpu.ToArray());
    }

    private static Difference Compare(float[] expected, float[] actual)
    {
        if (expected.Length != actual.Length) throw new InvalidOperationException("Attention output length mismatch.");
        long mismatches = 0;
        double maxAbs = 0;
        for (int i = 0; i < expected.Length; ++i)
        {
            float left = expected[i], right = actual[i];
            if (!float.IsFinite(left) || !float.IsFinite(right))
                throw new InvalidOperationException($"Nonfinite attention output at element {i}.");
            if (BitConverter.SingleToInt32Bits(left) != BitConverter.SingleToInt32Bits(right))
                ++mismatches;
            maxAbs = Math.Max(maxAbs, Math.Abs((double)left - right));
        }
        return new Difference(mismatches, maxAbs);
    }

    private static double Median(List<double> samples)
    {
        double[] sorted = samples.Order().ToArray();
        return (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) * .5;
    }
}
