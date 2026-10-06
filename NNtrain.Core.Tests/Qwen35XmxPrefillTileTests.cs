using System.Diagnostics;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35XmxPrefillTileTests(ITestOutputHelper output)
{
    private static readonly (string Suffix, int Rows)[] Variants =
    [ ("m64_k64", 64), ("m64_k64_slm", 64), ("k64", 128), ("k64_slm", 128),
      ("m64_k64_ab_slm", 64), ("k64_ab_slm", 128) ];
    private static readonly (string Suffix, int Rows, int Columns)[] PackedVariants =
    [ ("m256_n32", 256, 32), ("m128_n128", 128, 128),
      ("m128_n32", 128, 32), ("m64_n128", 64, 128) ];

    [Theory]
    [InlineData(256, 32)]
    [InlineData(128, 128)]
    [InlineData(128, 32)]
    [InlineData(64, 128)]
    public void PackedTilesAndFallbackOwnEveryDestination(int tileRows, int tileColumns)
    {
        foreach (int rows in new[] { 17, 65, 129, 257 })
        foreach (int cols in new[] { 5, 33, 65, 129 })
        {
            var normal = new int[rows * cols]; var fallback = new int[normal.Length];
            for (int rowBase = 0; rowBase < rows; rowBase += tileRows)
            for (int colBase = 0; colBase < cols; colBase += tileColumns)
            for (int subgroup = 0; subgroup < tileRows / 8; subgroup++)
            {
                for (int lane = 0; lane < 16; lane++)
                for (int tile = 0; tile < tileColumns / 16; tile++)
                for (int r = 0; r < 8; r++)
                {
                    int row = rowBase + subgroup * 8 + r, col = colBase + tile * 16 + lane;
                    if (row < rows && col < cols) normal[row * cols + col]++;
                }
                for (int item = subgroup; item < tileRows * tileColumns; item += tileRows / 8)
                {
                    int row = rowBase + item / tileColumns, col = colBase + item % tileColumns;
                    if (row < rows && col < cols) fallback[row * cols + col]++;
                }
            }
            Assert.All(normal, count => Assert.Equal(1, count));
            Assert.All(fallback, count => Assert.Equal(1, count));
        }
    }

    [Theory]
    [InlineData("iq2_s", 82)]
    [InlineData("iq3_s", 110)]
    public void PackedReuseTilesRetainProjectionAndFallback(string quant, int blockBytes)
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_XMX_PACKED_TILES_GPU") != "1", "Opt-in Arc packed tile correctness test.");
        using var lane = CreateLane();
        foreach (var shape in new[] { (Rows: 65, Input: 512, Output: 65), (Rows: 257, Input: 5120, Output: 129) })
        foreach (bool rangeFallback in new[] { false, true })
        {
            var random = new Random(shape.Input + blockBytes);
            float[] values = Enumerable.Range(0, shape.Rows * shape.Input).Select(_ => random.NextSingle() * 4 - 2).ToArray();
            if (rangeFallback) values[0] = 70000f;
            using ArcBuffer x = lane.Upload(values), w = lane.UploadRaw(Payload(random, quant, blockBytes, shape.Input, shape.Output));
            using ArcBuffer b = lane.Upload(Enumerable.Range(0, shape.Output).Select(i => (i % 9 - 4) * .03f).ToArray());
            int inputCount = (shape.Rows + 7) / 8 * 8 * shape.Input, weightCount = (shape.Output + 15) / 16 * 16 * shape.Input;
            using ArcBuffer high = lane.AllocateBytes(inputCount * 2), low = lane.AllocateBytes(inputCount * 2);
            using ArcBuffer grid = lane.AllocateBytes(weightCount * 2), scales = lane.Allocate(weightCount / 16);
            using ArcBuffer status = lane.Allocate(1), expectedBuffer = lane.Allocate(shape.Rows * shape.Output);
            lane.Run("q35a_zero", 1, 0, status, 1);
            lane.Run("q35l_prefill_xmx_pack_input_f16x2", inputCount, 0, x, high, low, status, shape.Rows, shape.Input);
            lane.Run($"q35l_prefill_xmx_pack_{quant}_factored", (long)shape.Input * shape.Output / 8, 0, w, grid, scales, shape.Input, shape.Output);
            lane.Run($"q35l_{quant}_sg16", ((long)shape.Rows * shape.Output + 1) / 2 * 32, 32,
                x, w, b, expectedBuffer, shape.Rows, shape.Input, shape.Output);
            var expected = new float[shape.Rows * shape.Output]; lane.Read(expectedBuffer, expected);
            foreach (var variant in PackedVariants)
            {
                using ArcBuffer result = lane.Upload(Enumerable.Repeat(float.NaN, expected.Length + 2).ToArray());
                RunPackedTile(lane, quant, variant, shape.Rows, shape.Input, shape.Output, high, low, grid, scales, b, result, x, w, status);
                var actual = new float[expected.Length + 2]; lane.Read(result, actual);
                Assert.True(float.IsNaN(actual[^1]) && float.IsNaN(actual[^2]));
                double error = 0, reference = 0;
                for (int i = 0; i < expected.Length; i++)
                {
                    Assert.True(float.IsFinite(actual[i]));
                    error += Math.Pow(actual[i] - expected[i], 2); reference += (double)expected[i] * expected[i];
                }
                double relative = Math.Sqrt(error / Math.Max(reference, 1e-30));
                output.WriteLine($"packed {quant} {shape} {variant.Suffix} fallback={rangeFallback}: L2={relative:E6}");
                Assert.True(relative < 5e-5, $"packed {quant} {variant.Suffix} L2={relative:E6}");
            }
            // The on-the-fly alternative uses row-major converted inputs.
            lane.Run("q35l_prefill_xmx_input_f16x2", values.Length, 0, x, high, low, status, values.Length);
            using ArcBuffer onFly = lane.Upload(Enumerable.Repeat(float.NaN, expected.Length + 2).ToArray());
            lane.Run2D($"q35l_prefill_xmx_{quant}_factored_m256_n32", ((long)shape.Output + 31) / 32 * 16,
                ((long)shape.Rows + 255) / 256 * 32, 16, 32,
                high, low, w, b, onFly, shape.Rows, shape.Input, shape.Output, x, status);
            var onFlyActual = new float[expected.Length + 2]; lane.Read(onFly, onFlyActual);
            Assert.True(float.IsNaN(onFlyActual[^1]) && float.IsNaN(onFlyActual[^2]));
            double onFlyError = 0, onFlyReference = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(float.IsFinite(onFlyActual[i]));
                onFlyError += Math.Pow(onFlyActual[i] - expected[i], 2); onFlyReference += (double)expected[i] * expected[i];
            }
            double onFlyRelative = Math.Sqrt(onFlyError / Math.Max(onFlyReference, 1e-30));
            output.WriteLine($"onfly {quant} {shape} m256_n32 fallback={rangeFallback}: L2={onFlyRelative:E6}");
            Assert.True(onFlyRelative < 5e-5, $"onfly {quant} m256_n32 L2={onFlyRelative:E6}");
        }
    }

    [Fact]
    public void ActualIq2PackedTileShapeReportsSpeedIncludingPacking()
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_XMX_PACKED_TILES_BENCH") != "1", "Opt-in Arc packed tile timing benchmark.");
        using var lane = CreateLane();
        foreach (var shape in new[] { (Rows: 128, Input: 5120, Output: 17408), (Rows: 256, Input: 17408, Output: 5120),
            (Rows: 512, Input: 5120, Output: 17408) })
        {
            var random = new Random(shape.Input);
            using ArcBuffer x = lane.Upload(Enumerable.Range(0, shape.Rows * shape.Input).Select(_ => random.NextSingle() * 2 - 1).ToArray());
            using ArcBuffer w = lane.UploadRaw(Payload(random, "iq2_s", 82, shape.Input, shape.Output)), b = lane.Upload(new float[shape.Output]);
            int inputCount = (shape.Rows + 7) / 8 * 8 * shape.Input, weightCount = (shape.Output + 15) / 16 * 16 * shape.Input;
            using ArcBuffer high = lane.AllocateBytes(inputCount * 2), low = lane.AllocateBytes(inputCount * 2);
            using ArcBuffer grid = lane.AllocateBytes(weightCount * 2), scales = lane.Allocate(weightCount / 16);
            using ArcBuffer status = lane.Allocate(1), result = lane.Allocate(shape.Rows * shape.Output);
            foreach (var variant in new[] { (Suffix: "reference", Rows: 128, Columns: 64) }.Concat(PackedVariants))
            {
                string kernel = "q35l_prefill_xmx_packed_factored" + (variant.Suffix == "reference" ? "" : "_" + variant.Suffix);
                output.WriteLine(lane.GetKernelResources(kernel).ToString());
                void Run()
                {
                    lane.Run("q35a_zero", 1, 0, status, 1);
                    lane.Run("q35l_prefill_xmx_pack_input_f16x2", inputCount, 0, x, high, low, status, shape.Rows, shape.Input);
                    lane.Run("q35l_prefill_xmx_pack_iq2_s_factored", (long)shape.Input * shape.Output / 8, 0, w, grid, scales, shape.Input, shape.Output);
                    RunPackedTile(lane, "iq2_s", variant, shape.Rows, shape.Input, shape.Output, high, low, grid, scales, b, result, x, w, status);
                }
                Run(); lane.Synchronize();
                var times = new List<double>();
                for (int i = 0; i < 3; i++)
                {
                    var watch = Stopwatch.StartNew(); Run(); lane.Synchronize(); times.Add(watch.Elapsed.TotalMilliseconds);
                }
                output.WriteLine($"packed iq2_s {shape} {variant.Suffix}: {times.Order().ElementAt(1):F3} ms");
            }
            foreach (bool wide in new[] { false, true })
            {
                string kernel = "q35l_prefill_xmx_iq2_s_factored" + (wide ? "_m256_n32" : "");
                output.WriteLine(lane.GetKernelResources(kernel).ToString());
                void Run()
                {
                    lane.Run("q35a_zero", 1, 0, status, 1);
                    lane.Run("q35l_prefill_xmx_input_f16x2", shape.Rows * shape.Input, 0, x, high, low, status, shape.Rows * shape.Input);
                    int bm = wide ? 256 : 128, bn = wide ? 32 : 64;
                    lane.Run2D(kernel, ((long)shape.Output + bn - 1) / bn * 16,
                        ((long)shape.Rows + bm - 1) / bm * (bm / 8), 16, bm / 8,
                        high, low, w, b, result, shape.Rows, shape.Input, shape.Output, x, status);
                }
                Run(); lane.Synchronize();
                var times = new List<double>();
                for (int i = 0; i < 3; i++)
                {
                    var watch = Stopwatch.StartNew(); Run(); lane.Synchronize(); times.Add(watch.Elapsed.TotalMilliseconds);
                }
                output.WriteLine($"onfly iq2_s {shape} {(wide ? "m256_n32" : "reference")}: {times.Order().ElementAt(1):F3} ms");
            }
        }
    }

    private static void RunPackedTile(ArcExecutionLane lane, string quant, (string Suffix, int Rows, int Columns) variant,
        int rows, int input, int output, ArcBuffer high, ArcBuffer low, ArcBuffer grid, ArcBuffer scales,
        ArcBuffer bias, ArcBuffer result, ArcBuffer values, ArcBuffer packed, ArcBuffer status)
    {
        string name = "q35l_prefill_xmx_packed_factored" + (variant.Suffix == "reference" ? "" : "_" + variant.Suffix);
        lane.Run2D(name, ((long)output + variant.Columns - 1) / variant.Columns * 16,
            ((long)rows + variant.Rows - 1) / variant.Rows * (variant.Rows / 8), 16, variant.Rows / 8,
            high, low, grid, scales, bias, result, rows, input, output, values, packed, status, quant == "iq2_s" ? 0 : 1);
    }

    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    public void PackedARowPairsMatchTheDpasShort8Operand(int tileRows)
    {
        var halves = new ushort[tileRows * 64];
        var written = new bool[halves.Length];
        for (int row = 0; row < tileRows; row++)
        for (int k = 0; k < 64; k++)
        {
            int destination = ((k / 16) * (tileRows / 8) + row / 8) * 128
                + row % 8 / 2 * 32 + k % 16 * 2 + row % 2;
            Assert.False(written[destination]); written[destination] = true;
            halves[destination] = (ushort)(row * 64 + k);
        }
        Assert.All(written, Assert.True);
        for (int subgroup = 0; subgroup < tileRows / 8; subgroup++)
        for (int part = 0; part < 4; part++)
        for (int lane = 0; lane < 16; lane++)
        for (int r = 0; r < 8; r++)
        {
            int word = (part * (tileRows / 8) + subgroup) * 64 + r / 2 * 16 + lane;
            ushort value = halves[word * 2 + r % 2];
            Assert.Equal((ushort)((subgroup * 8 + r) * 64 + part * 16 + lane), value);
        }
    }

    [Fact]
    public void K64PanelProducerAndXmxConsumerCoverEveryHalfWord()
    {
        // CPU contract for the on-chip representation: each uint contains
        // consecutive K halves; a DPAS B operand reads eight vectors of 16.
        var panel = new uint[64 * 32];
        var written = new bool[panel.Length];
        for (int col = 0; col < 64; col++)
        for (int octet = 0; octet < 8; octet++)
        for (int pair = 0; pair < 4; pair++)
        {
            int p = octet * 4 + pair;
            int destination = ((col / 16) * 4 + p / 8) * 128 + p % 8 * 16 + col % 16;
            Assert.False(written[destination]);
            written[destination] = true;
            panel[destination] = (uint)(col * 64 + p * 2) | (uint)(col * 64 + p * 2 + 1) << 16;
        }
        Assert.All(written, Assert.True);
        for (int tile = 0; tile < 4; tile++)
        for (int part = 0; part < 4; part++)
        for (int lane = 0; lane < 16; lane++)
        for (int vector = 0; vector < 8; vector++)
        {
            uint packed = panel[(tile * 4 + part) * 128 + vector * 16 + lane];
            int col = tile * 16 + lane, k = part * 16 + vector * 2;
            Assert.Equal((uint)(col * 64 + k), packed & 0xffff);
            Assert.Equal((uint)(col * 64 + k + 1), packed >> 16);
        }
        Assert.Equal(32 * 1024, (64 * 64 * 2 + 64 * 32 * 4) * 2);
    }

    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    public void NormalAndFallbackTilesOwnExactlyTheirRows(int tileRows)
    {
        foreach (int rows in new[] { 1, 17, 63, 64, 65, 127, 129 })
        foreach (int cols in new[] { 1, 5, 63, 64, 65, 129 })
        {
            var normal = new int[rows * cols];
            var fallback = new int[normal.Length];
            for (int rowBase = 0; rowBase < rows; rowBase += tileRows)
            for (int colBase = 0; colBase < cols; colBase += 64)
            for (int subgroup = 0; subgroup < tileRows / 8; subgroup++)
            {
                for (int lane = 0; lane < 16; lane++)
                for (int tile = 0; tile < 4; tile++)
                for (int r = 0; r < 8; r++)
                {
                    int row = rowBase + subgroup * 8 + r, col = colBase + tile * 16 + lane;
                    if (row < rows && col < cols) normal[row * cols + col]++;
                }
                // Only lane zero publishes the subgroup reduction.
                for (int item = subgroup; item < tileRows * 64; item += tileRows / 8)
                {
                    int row = rowBase + item / 64, col = colBase + item % 64;
                    if (row < rows && col < cols) fallback[row * cols + col]++;
                }
            }
            Assert.All(normal, count => Assert.Equal(1, count));
            Assert.All(fallback, count => Assert.Equal(1, count));
        }
    }

    [Theory]
    [InlineData("iq2_s", 82)]
    [InlineData("iq3_s", 110)]
    [InlineData("q4_k", 144)]
    public void K64TilesRetainProjectionAndFallbackWithoutTailWrites(string quant, int blockBytes)
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_XMX_TILES_GPU") != "1", "Opt-in Arc tile correctness test.");
        using var lane = CreateLane();
        foreach (var shape in new[] { (Rows: 17, Input: 256, Output: 5), (Rows: 65, Input: 512, Output: 65),
            (Rows: 129, Input: 5120, Output: 129) })
        foreach (bool rangeFallback in new[] { false, true })
        {
            var random = new Random(shape.Input + blockBytes);
            float[] values = Enumerable.Range(0, shape.Rows * shape.Input).Select(_ => random.NextSingle() * 4 - 2).ToArray();
            if (rangeFallback) values[0] = 70000f;
            using ArcBuffer x = lane.Upload(values), w = lane.UploadRaw(Payload(random, quant, blockBytes, shape.Input, shape.Output));
            using ArcBuffer b = lane.Upload(Enumerable.Range(0, shape.Output).Select(i => (i % 9 - 4) * .03f).ToArray());
            using ArcBuffer high = lane.AllocateBytes(values.Length * sizeof(ushort)), low = lane.AllocateBytes(values.Length * sizeof(ushort));
            using ArcBuffer status = lane.Allocate(1), expectedBuffer = lane.Allocate(shape.Rows * shape.Output);
            lane.Run("q35a_zero", 1, 0, status, 1);
            lane.Run("q35l_prefill_xmx_input_f16x2", values.Length, 0, x, high, low, status, values.Length);
            lane.Run($"q35l_{quant}_sg16", ((long)shape.Rows * shape.Output + 1) / 2 * 32, 32,
                x, w, b, expectedBuffer, shape.Rows, shape.Input, shape.Output);
            var expected = new float[shape.Rows * shape.Output]; lane.Read(expectedBuffer, expected);
            foreach (var variant in Variants)
            {
                using ArcBuffer result = lane.Upload(Enumerable.Repeat(float.NaN, expected.Length + 2).ToArray());
                RunTile(lane, quant, variant, shape.Rows, shape.Input, shape.Output, high, low, w, b, result, x, status);
                var actual = new float[expected.Length + 2]; lane.Read(result, actual);
                Assert.True(float.IsNaN(actual[^1]) && float.IsNaN(actual[^2]));
                double error = 0, reference = 0;
                for (int i = 0; i < expected.Length; i++)
                {
                    Assert.True(float.IsFinite(actual[i]));
                    error += Math.Pow(actual[i] - expected[i], 2); reference += (double)expected[i] * expected[i];
                }
                double relative = Math.Sqrt(error / Math.Max(reference, 1e-30));
                output.WriteLine($"{quant} {shape} {variant.Suffix} fallback={rangeFallback}: L2={relative:E6}");
                Assert.True(relative < 5e-5, $"{quant} {variant.Suffix} relative L2={relative:E6}");
            }
        }
    }

    [Fact]
    public void ActualIq2FfnShapeReportsK64TileSpeed()
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_XMX_TILES_BENCH") != "1", "Opt-in Arc tile timing benchmark.");
        using var lane = CreateLane();
        foreach (var shape in new[] { (Rows: 128, Input: 5120, Output: 17408), (Rows: 256, Input: 17408, Output: 5120) })
        {
            var random = new Random(shape.Input);
            using ArcBuffer x = lane.Upload(Enumerable.Range(0, shape.Rows * shape.Input).Select(_ => random.NextSingle() * 2 - 1).ToArray());
            using ArcBuffer w = lane.UploadRaw(Payload(random, "iq2_s", 82, shape.Input, shape.Output)), b = lane.Upload(new float[shape.Output]);
            using ArcBuffer high = lane.AllocateBytes(shape.Rows * shape.Input * 2), low = lane.AllocateBytes(shape.Rows * shape.Input * 2);
            using ArcBuffer status = lane.Allocate(1), result = lane.Allocate(shape.Rows * shape.Output);
            foreach (var variant in new[] { (Suffix: "reference", Rows: 128) }.Concat(Variants))
            {
                string kernel = "q35l_prefill_xmx_iq2_s_f16x2" + (variant.Suffix == "reference" ? "" : "_" + variant.Suffix);
                output.WriteLine(lane.GetKernelResources(kernel).ToString());
                void Run()
                {
                    lane.Run("q35a_zero", 1, 0, status, 1);
                    lane.Run("q35l_prefill_xmx_input_f16x2", shape.Rows * shape.Input, 0, x, high, low, status, shape.Rows * shape.Input);
                    RunTile(lane, "iq2_s", variant, shape.Rows, shape.Input, shape.Output, high, low, w, b, result, x, status);
                }
                Run(); lane.Synchronize();
                var times = new List<double>();
                for (int i = 0; i < 3; i++)
                {
                    var watch = Stopwatch.StartNew(); Run(); lane.Synchronize(); times.Add(watch.Elapsed.TotalMilliseconds);
                }
                output.WriteLine($"iq2_s {shape} {variant.Suffix}: {times.Order().ElementAt(1):F3} ms");
            }
        }
    }

    private static ArcExecutionLane CreateLane()
    {
        var devices = ArcDevices.Enumerate();
        Assert.SkipWhen(devices.Count == 0, "Intel Arc is required.");
        ArcDeviceInfo device = devices[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_khr_fp16"), "Arc SG16/F16 XMX is required.");
        return new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true,
            Qwen35TrainingKernels = false, BufferPoolBytes = 64 * 1024 * 1024,
            ExperimentalOptimizationKernels = true });
    }

    private static void RunTile(ArcExecutionLane lane, string quant, (string Suffix, int Rows) variant,
        int rows, int input, int output, ArcBuffer high, ArcBuffer low, ArcBuffer weights, ArcBuffer bias,
        ArcBuffer result, ArcBuffer values, ArcBuffer status)
    {
        string name = $"q35l_prefill_xmx_{quant}_f16x2" + (variant.Suffix == "reference" ? "" : "_" + variant.Suffix);
        lane.Run2D(name, ((long)output + 63) / 64 * 16, ((long)rows + variant.Rows - 1) / variant.Rows * (variant.Rows / 8),
            16, variant.Rows / 8, high, low, weights, bias, result, rows, input, output, values, status);
    }

    private static byte[] Payload(Random random, string quant, int blockBytes, int input, int output)
    {
        int blocks = checked(input * output / 256);
        var result = new byte[checked(blocks * blockBytes)]; random.NextBytes(result);
        for (int block = 0; block < blocks; block++)
        {
            WriteHalf(block * blockBytes, (Half)((block % 7 + 1) / 8192f));
            if (quant == "q4_k") WriteHalf(block * blockBytes + 2, (Half)((block % 3 + 1) / 16384f));
        }
        return result;
        void WriteHalf(int offset, Half value)
        {
            ushort bits = BitConverter.HalfToUInt16Bits(value); result[offset] = (byte)bits; result[offset + 1] = (byte)(bits >> 8);
        }
    }
}
