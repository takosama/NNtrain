using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.RegularExpressions;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace Iq2sProductionBench;

internal static class TransposeBenchmark
{
    internal static object Run(ArcExecutionLane lane, InputSource source, int rows, int samples, int warmup,
        int cpuPoints, IReadOnlyList<(int Rows, int Columns)> tiles, int panelColumns, int scratchMiB, bool fallback)
    {
        int inputWidth = source.Input, outputWidth = source.Output, count = checked(rows * inputWidth);
        int seed = 481008 + rows + inputWidth + outputWidth;
        var random = new Random(seed);
        float[] values = Enumerable.Range(0, checked(rows * outputWidth)).Select(_ => random.NextSingle() * 4 - 2).ToArray();
        if (fallback) { values[0] = 100000; values[Math.Min(values.Length - 1, 8)] = -100000; }
        ReferenceData.Point[] reference = Reference(values, source.Payload, rows, inputWidth, outputWidth, cpuPoints);
        using ArcBuffer dy = lane.Upload(values), weight = lane.UploadRaw(source.Payload), dx = lane.Allocate(count + 2);
        lane.Synchronize();
        int splits = checked((outputWidth + 1023) / 1024);
        long scratchBudget = checked((long)scratchMiB * 1024 * 1024), bytesPerRow = checked((long)splits * inputWidth * 4);
        int chunkRows = Math.Min(rows, Math.Max(1, checked((int)(scratchBudget / bytesPerRow))));
        if (chunkRows < rows && chunkRows >= 8) chunkRows = chunkRows / 8 * 8;
        Console.WriteLine($"transpose {source.Name}: rows={rows}, input={inputWidth}, output={outputWidth}, chunkRows={chunkRows}");

        void Original()
        {
            for (int first = 0; first < rows; first += chunkRows)
            {
                int amount = Math.Min(chunkRows, rows - first);
                using ArcBuffer partial = lane.Allocate(checked(amount * splits * inputWidth));
                if (first == 0 && amount == rows)
                {
                    lane.Run("q35a_zero", count, 0, dx, count);
                    Run(dy, partial, dx, amount);
                }
                else
                {
                    using ArcBuffer inputSlice = lane.Allocate(checked(amount * outputWidth)), outputSlice = lane.Allocate(checked(amount * inputWidth));
                    lane.CopyBytes(dy, inputSlice, checked(first * outputWidth * 4), 0, checked(amount * outputWidth * 4));
                    lane.Run("q35a_zero", (long)amount * inputWidth, 0, outputSlice, checked(amount * inputWidth));
                    Run(inputSlice, partial, outputSlice, amount);
                    lane.CopyBytes(outputSlice, dx, 0, checked(first * inputWidth * 4), checked(amount * inputWidth * 4));
                }
            }
            void Run(ArcBuffer input, ArcBuffer partial, ArcBuffer output, int amount)
            {
                lane.Run("q35t_xpose_iq2_s_vec8_rows8", ((long)amount + 7) / 8 * splits * (inputWidth / 8), 32,
                    input, weight, partial, amount, inputWidth, outputWidth, splits, 1024);
                lane.Run("q35t_xpose_reduce", (long)amount * inputWidth, 0, partial, output, amount, inputWidth, splits);
            }
        }

        var variants = new List<Variant> { new("original-transpose-vec8-rows8-reduce", Original) };
        foreach (var tile in tiles)
            variants.Add(new($"tiled-transpose-{tile.Rows}x{tile.Columns}-panel{panelColumns}", () =>
                lane.Iq2TiledBackward(dy, weight, dx, rows, inputWidth, outputWidth, tile.Rows, tile.Columns, panelColumns,
                    addToOutput: false, workspaceBudgetBytes: scratchBudget)));
        float[] poison = Enumerable.Repeat(float.NaN, count + 2).ToArray();
        float[] ReadResult(Action action)
        {
            lane.WriteRaw(dx, poison); lane.Synchronize(); action(); lane.Synchronize();
            var result = new float[count + 2]; lane.Read(dx, result); return result;
        }
        float[] original = ReadResult(Original);
        var originalValidation = ReferenceData.Validate(original, count, reference, null);
        foreach (Variant variant in variants)
        {
            variant.Validation = ReferenceData.Validate(ReadResult(variant.Run), count, reference, original);
            Console.WriteLine($" validated {variant.Name}: CPU L2={variant.Validation.RelativeL2VsDouble:E4}, original L2={variant.Validation.RelativeL2VsOriginal:E4}");
        }
        for (int round = 0; round < warmup; round++)
            foreach (Variant variant in round % 2 == 0 ? variants : variants.AsEnumerable().Reverse()) { variant.Run(); lane.Synchronize(); }
        for (int round = 0; round < samples; round++)
        foreach (Variant variant in round % 2 == 0 ? variants : variants.AsEnumerable().Reverse())
        {
            var before = new Dictionary<string, double>(lane.KernelTimings);
            double gpuBefore = lane.KernelMilliseconds;
            long allocations = lane.AllocationCount, poolHits = lane.PoolHits, launches = lane.KernelLaunchCount;
            long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
            var watch = Stopwatch.StartNew(); variant.Run(); lane.Synchronize(); watch.Stop();
            var byKernel = lane.KernelTimings.Where(pair => pair.Value - before.GetValueOrDefault(pair.Key) > 0)
                .ToDictionary(pair => pair.Key, pair => pair.Value - before.GetValueOrDefault(pair.Key));
            double gpuTime = lane.KernelMilliseconds - gpuBefore;
            if (lane.H2DBytes != h2d || lane.D2HBytes != d2h) throw new InvalidOperationException("Measured transpose used host transfers.");
            if (!double.IsFinite(gpuTime) || gpuTime <= 0) throw new ArithmeticException("Missing transpose GPU timings.");
            variant.Samples.Add(new(round, gpuTime, watch.Elapsed.TotalMilliseconds, byKernel,
                lane.AllocationCount - allocations, lane.PoolHits - poolHits, lane.KernelLaunchCount - launches));
        }
        foreach (Variant variant in variants)
            variant.PostTimingValidation = ReferenceData.Validate(ReadResult(variant.Run), count, reference, original);
        var summaries = variants.Select(variant => new { variant.Name, variant.Validation, variant.PostTimingValidation,
            gpuMilliseconds = Stats.Of(variant.Samples.Select(sample => sample.GpuMilliseconds)),
            wallMilliseconds = Stats.Of(variant.Samples.Select(sample => sample.WallMilliseconds)), samples = variant.Samples }).ToArray();
        foreach (var summary in summaries) Console.WriteLine($" {summary.Name}: GPU median {summary.gpuMilliseconds.Median:F6} ms, wall median {summary.wallMilliseconds.Median:F6} ms");
        return new { direction = "transpose", operation = "dX[M,K] = dY_FP32[M,N] * W_IQ2S[N,K] (overwrite output)",
            name = source.Name, rows, inputWidth, outputWidth, source = source.Provenance,
            upstreamGradient = new { type = "FP32", seed, sha256 = Convert.ToHexString(SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()))).ToLowerInvariant(), fallback },
            original = new { splits, reductionTile = 1024, rowTile = 8, scratchBudgetBytes = scratchBudget, chunkRows,
                note = "Original split/row8/reduce kernels, bounded row chunks, device copies included in host wall; GPU metric sums kernel events only (excludes device copies). Original output zero included; candidate overwrites." },
            originalValidation, variants = summaries };
    }

    private static ReferenceData.Point[] Reference(float[] dy, byte[] packed, int rows, int inputWidth, int outputWidth, int requested)
    {
        string source = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "qwen35_iq.cl"));
        string initializer = source.Split("q35l_iq2s_grid[1024]", 2)[1].Split("};", 2)[0];
        ulong[] grid = Regex.Matches(initializer, @"0x([0-9a-fA-F]{16})[uU][lL]")
            .Select(match => Convert.ToUInt64(match.Groups[1].Value, 16)).ToArray();
        if (grid.Length != 1024) throw new InvalidDataException("Invalid IQ2 lookup table.");
        int total = checked(rows * inputWidth), pointCount = Math.Min(total, Math.Max(4, requested));
        var indices = new HashSet<int> { 0, inputWidth - 1, (rows - 1) * inputWidth, total - 1 };
        var random = new Random(9381008);
        if (pointCount == total) for (int index = 0; index < total; index++) indices.Add(index);
        else while (indices.Count < pointCount) indices.Add(random.Next(total));
        var result = new List<ReferenceData.Point>();
        foreach (var columnPoints in indices.GroupBy(index => index % inputWidth))
        {
            int column = columnPoints.Key, group = column % 256 / 32, octet = column / 8 % 4, within = column % 8;
            var decoded = new float[outputWidth];
            for (int n = 0; n < outputWidth; n++)
            {
                int offset = checked((n * (inputWidth / 256) + column / 256) * 82);
                float d = (float)BitConverter.UInt16BitsToHalf((ushort)(packed[offset] | packed[offset + 1] << 8));
                int code = packed[offset + 2 + group * 4 + octet] | (((packed[offset + 66 + group] >> (octet * 2)) & 3) << 8);
                int q = (int)((grid[code] >> (within * 8)) & 255);
                int scale = (packed[offset + 74 + group] >> (octet / 2 * 4)) & 15;
                bool negative = ((packed[offset + 34 + group * 4 + octet] >> within) & 1) != 0;
                decoded[n] = ((d * (0.5f + scale)) * 0.25f) * q * (negative ? -1f : 1f);
            }
            foreach (int index in columnPoints)
            {
                double sum = 0, absolute = 0;
                int row = index / inputWidth;
                for (int n = 0; n < outputWidth; n++) { double product = (double)dy[row * outputWidth + n] * decoded[n]; sum += product; absolute += Math.Abs(product); }
                result.Add(new(index, sum, absolute));
            }
        }
        return result.OrderBy(point => point.Index).ToArray();
    }
}
