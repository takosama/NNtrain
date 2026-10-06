using System.Diagnostics;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35XmxPrefillLinearTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData("iq2_s", 82)]
    [InlineData("iq3_s", 110)]
    [InlineData("q4_k", 144)]
    public void OutOfF16RangeActivationUsesDeviceOnlyExactFallback(string quant, int blockBytes)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true });
        const int rows = 17, inputWidth = 256, outputWidth = 65, count = rows * outputWidth;
        var random = new Random(blockBytes + 95);
        var payload = Payload(random, quant, blockBytes, inputWidth, outputWidth);
        float[] values = Enumerable.Range(0, rows * inputWidth).Select(_ => random.NextSingle() * 2 - 1).ToArray();
        values[0] = 100_000f; values[8] = -100_000f;
        using ArcBuffer x = lane.Upload(values), w = lane.UploadRaw(payload), b = lane.Upload(new float[outputWidth]);
        using ArcBuffer high = lane.AllocateBytes(values.Length * 2), low = lane.AllocateBytes(values.Length * 2), status = lane.Allocate(1);
        using ArcBuffer reference = lane.Allocate(count), actual = lane.Allocate(count);
        lane.Run("q35a_zero", 1, 0, status, 1);
        lane.Run($"q35l_{quant}_sg16", ((long)count + 1) / 2 * 32, 32, x, w, b, reference, rows, inputWidth, outputWidth);
        lane.Run("q35l_prefill_xmx_input_f16x2", values.Length, 0, x, high, low, status, values.Length);
        lane.Run2D($"q35l_prefill_xmx_{quant}_f16x2", ((long)outputWidth + 63) / 64 * 16,
            ((long)rows + 127) / 128 * 16, 16, 16, high, low, w, b, actual, rows, inputWidth, outputWidth, x, status);
        float[] expectedValues = new float[count], actualValues = new float[count];
        int[] flags = new int[1]; lane.ReadRaw(status, flags); lane.Read(reference, expectedValues); lane.Read(actual, actualValues);
        Assert.Equal(1, flags[0]);
        for (int i = 0; i < count; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(expectedValues[i]), BitConverter.SingleToInt32Bits(actualValues[i]));
        if (quant != "q4_k")
        {
            lane.Run2D($"q35l_prefill_xmx_{quant}_factored", ((long)outputWidth + 63) / 64 * 16,
                ((long)rows + 127) / 128 * 16, 16, 16, high, low, w, b, actual, rows, inputWidth, outputWidth, x, status);
            lane.Read(actual, actualValues);
            for (int i = 0; i < count; i++)
                Assert.Equal(BitConverter.SingleToInt32Bits(expectedValues[i]), BitConverter.SingleToInt32Bits(actualValues[i]));
        }
    }

    [Theory]
    [InlineData("iq2_s", 82)]
    [InlineData("iq3_s", 110)]
    [InlineData("q4_k", 144)]
    public void TwoComponentXmxRetainsQuantizedProjectionWithinTolerance(string quant, int blockBytes)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true, Qwen35TrainingKernels = false, ExperimentalOptimizationKernels = true });
        Assert.SkipWhen(!lane.Device.SupportsXmx || lane.Device.MinimumSubgroupSize != 16
            || !lane.Device.Extensions.Split(' ').Contains("cl_khr_fp16"), "Arc SG16/F16 XMX is required.");
        foreach (var shape in new[] { (Rows: 1, Input: 256, Output: 5), (Rows: 17, Input: 512, Output: 65),
            (Rows: 129, Input: 5120, Output: 129), (Rows: 256, Input: 17408, Output: 65) })
        {
            var random = new Random(shape.Input + blockBytes);
            var payload = Payload(random, quant, blockBytes, shape.Input, shape.Output);
            float[] values = Enumerable.Range(0, checked(shape.Rows * shape.Input))
                .Select(_ => random.NextSingle() * 4 - 2).ToArray();
            float[] bias = Enumerable.Range(0, shape.Output).Select(i => (i % 9 - 4) * .03f).ToArray();
            using ArcBuffer x = lane.Upload(values), w = lane.UploadRaw(payload), b = lane.Upload(bias);
            int count = checked(shape.Rows * shape.Output), inputCount = values.Length;
            using ArcBuffer high = lane.AllocateBytes(inputCount * sizeof(ushort)), low = lane.AllocateBytes(inputCount * sizeof(ushort));
            using ArcBuffer status = lane.Allocate(1);
            lane.Run("q35a_zero", 1, 0, status, 1);
            using ArcBuffer expectedBuffer = lane.Allocate(count), resultBuffer = lane.Upload(Enumerable.Repeat(float.NaN, count + 2).ToArray());
            lane.Run($"q35l_{quant}_sg16", ((long)count + 1) / 2 * 32, 32,
                x, w, b, expectedBuffer, shape.Rows, shape.Input, shape.Output);
            lane.Run("q35l_prefill_xmx_input_f16x2", inputCount, 0, x, high, low, status, inputCount);
            lane.Run2D($"q35l_prefill_xmx_{quant}_f16x2", ((long)shape.Output + 63) / 64 * 16,
                ((long)shape.Rows + 127) / 128 * 16, 16, 16,
                high, low, w, b, resultBuffer, shape.Rows, shape.Input, shape.Output, x, status);
            float[] expected = new float[count], actual = new float[count + 2];
            lane.Read(expectedBuffer, expected); lane.Read(resultBuffer, actual);
            Assert.True(float.IsNaN(actual[count]) && float.IsNaN(actual[count + 1]));
            double error2 = 0, reference2 = 0, maximum = 0;
            for (int i = 0; i < count; ++i)
            {
                Assert.True(float.IsFinite(actual[i]));
                double difference = actual[i] - expected[i];
                error2 += difference * difference; reference2 += (double)expected[i] * expected[i];
                maximum = Math.Max(maximum, Math.Abs(difference));
            }
            double relative = Math.Sqrt(error2 / Math.Max(reference2, 1e-30));
            output.WriteLine($"{quant} {shape}: relative L2={relative:E6}, max={maximum:E6}");
            Assert.True(relative < 5e-5, $"{quant} {shape}: relative L2 {relative:E6}");
            if (quant is "iq2_s" or "iq3_s")
            {
                lane.Run2D($"q35l_prefill_xmx_{quant}_factored", ((long)shape.Output + 63) / 64 * 16,
                    ((long)shape.Rows + 127) / 128 * 16, 16, 16, high, low, w, b, resultBuffer,
                    shape.Rows, shape.Input, shape.Output, x, status);
                lane.Read(resultBuffer, actual); error2 = 0; maximum = 0;
                for (int i = 0; i < count; i++)
                {
                    Assert.True(float.IsFinite(actual[i]));
                    double difference = actual[i] - expected[i]; error2 += difference * difference; maximum = Math.Max(maximum, Math.Abs(difference));
                }
                relative = Math.Sqrt(error2 / Math.Max(reference2, 1e-30));
                output.WriteLine($"factored {quant} {shape}: relative L2={relative:E6}, max={maximum:E6}");
                Assert.True(relative < 5e-5, $"factored {quant} {shape}: relative L2 {relative:E6}");
            }
            int packedInputCount = checked((shape.Rows + 7) / 8 * 8 * shape.Input);
            int packedWeightCount = checked((shape.Output + 15) / 16 * 16 * shape.Input);
            using ArcBuffer packedHigh = lane.AllocateBytes(packedInputCount * sizeof(ushort)), packedLow = lane.AllocateBytes(packedInputCount * sizeof(ushort));
            using ArcBuffer weightHigh = lane.AllocateBytes(packedWeightCount * sizeof(ushort)), weightLow = lane.AllocateBytes(packedWeightCount * sizeof(ushort));
            lane.Run("q35l_prefill_xmx_pack_input_f16x2", packedInputCount, 0, x, packedHigh, packedLow, status, shape.Rows, shape.Input);
            lane.Run($"q35l_prefill_xmx_pack_{quant}_f16x2", (long)shape.Input * shape.Output / 8, 0, w, weightHigh, weightLow, shape.Input, shape.Output);
            lane.Run2D("q35l_prefill_xmx_packed_f16x2", ((long)shape.Output + 63) / 64 * 16, ((long)shape.Rows + 127) / 128 * 16,
                16, 16, packedHigh, packedLow, weightHigh, weightLow, b, resultBuffer, shape.Rows, shape.Input, shape.Output, x, w, status,
                quant == "iq2_s" ? 0 : quant == "iq3_s" ? 1 : 2);
            lane.Read(resultBuffer, actual);
            error2 = 0; maximum = 0;
            for (int i = 0; i < count; i++)
            {
                Assert.True(float.IsFinite(actual[i]));
                double difference = actual[i] - expected[i]; error2 += difference * difference; maximum = Math.Max(maximum, Math.Abs(difference));
            }
            relative = Math.Sqrt(error2 / Math.Max(reference2, 1e-30));
            output.WriteLine($"packed {quant} {shape}: relative L2={relative:E6}, max={maximum:E6}");
            Assert.True(relative < 5e-5, $"packed {quant} {shape}: relative L2 {relative:E6}");
            if (quant is "iq2_s" or "iq3_s")
            {
                using ArcBuffer grid = lane.AllocateBytes(packedWeightCount * sizeof(ushort));
                using ArcBuffer scales = lane.Allocate(packedWeightCount / 16);
                foreach (var variant in new[] { (Int8: false, BSlm: false), (Int8: true, BSlm: false), (Int8: false, BSlm: true) })
                {
                    bool int8Grid = variant.Int8;
                    lane.Run($"q35l_prefill_xmx_pack_{quant}_factored" + (int8Grid ? "_i8" : ""), (long)shape.Input * shape.Output / 8, 0,
                        w, grid, scales, shape.Input, shape.Output);
                    lane.Run2D("q35l_prefill_xmx_packed_factored" + (int8Grid ? "_i8" : variant.BSlm ? "_bslm" : ""), ((long)shape.Output + 63) / 64 * 16,
                        ((long)shape.Rows + 127) / 128 * 16, 16, 16, packedHigh, packedLow, grid, scales, b, resultBuffer,
                        shape.Rows, shape.Input, shape.Output, x, w, status, quant == "iq2_s" ? 0 : 1);
                    lane.Read(resultBuffer, actual); error2 = 0; maximum = 0;
                    for (int i = 0; i < count; i++)
                    {
                        Assert.True(float.IsFinite(actual[i]));
                        double difference = actual[i] - expected[i]; error2 += difference * difference; maximum = Math.Max(maximum, Math.Abs(difference));
                    }
                    relative = Math.Sqrt(error2 / Math.Max(reference2, 1e-30));
                    output.WriteLine($"packed factored {variant} {quant} {shape}: relative L2={relative:E6}, max={maximum:E6}");
                    Assert.True(relative < 5e-5, $"packed factored {quant} {shape}: relative L2 {relative:E6}");
                }
            }
        }
    }

    [Fact]
    public void ResidentGrid3RetainsProjectionAndFusedLoraIncludingRangeFallback()
    {
        using var lane = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true, ExperimentalOptimizationKernels = true });
        foreach (var shape in new[] { (Rows: 1, Input: 256, Output: 5, Range: false), (Rows: 17, Input: 512, Output: 65, Range: false),
            (Rows: 129, Input: 5120, Output: 129, Range: false), (Rows: 17, Input: 512, Output: 65, Range: true) })
        {
            var random = new Random(shape.Input + shape.Rows);
            float[] values = Enumerable.Range(0, shape.Rows * shape.Input).Select(_ => random.NextSingle() * 2 - 1).ToArray();
            bool outOfRange = shape.Range;
            if (outOfRange) { values[0] = 100000; values[8] = -100000; }
            int colBlocks = (shape.Output + 15) / 16, packedInputs = (shape.Rows + 7) / 8 * 8 * shape.Input;
            using ArcBuffer x = lane.Upload(values), w = lane.UploadRaw(Payload(random, "iq2_s", 82, shape.Input, shape.Output));
            using ArcBuffer bias = lane.Upload(Enumerable.Range(0, shape.Output).Select(_ => random.NextSingle()).ToArray());
            using ArcBuffer grid = lane.AllocateBytes(shape.Input / 32 * colBlocks * 48 * 4);
            using ArcBuffer scales = lane.AllocateBytes(shape.Input / 32 * colBlocks * 16);
            using ArcBuffer d = lane.AllocateBytes(shape.Input / 256 * colBlocks * 16 * 2);
            using ArcBuffer high = lane.AllocateBytes(packedInputs * 2), low = lane.AllocateBytes(packedInputs * 2), status = lane.Allocate(1);
            int count = shape.Rows * shape.Output;
            using ArcBuffer expected = lane.Allocate(count), actual = lane.Allocate(count);
            lane.Run("q35l_prefill_xmx_pack_iq2_s_grid3", (long)shape.Input / 32 * shape.Output, 0,
                w, grid, scales, d, shape.Input, shape.Output);
            long global = ((long)count + lane.Options.Qwen35ProjectionWorkgroupSize / 16 - 1)
                / (lane.Options.Qwen35ProjectionWorkgroupSize / 16) * lane.Options.Qwen35ProjectionWorkgroupSize;
            lane.Run("q35l_iq2_s_sg16", global, lane.Options.Qwen35ProjectionWorkgroupSize, x, w, bias, expected, shape.Rows, shape.Input, shape.Output);
            lane.Run("q35l_iq2_s_grid3_sg16", global, lane.Options.Qwen35ProjectionWorkgroupSize, x, grid, scales, d, bias, actual, shape.Rows, shape.Input, shape.Output);
            float[] expectedValues = new float[count], actualValues = new float[count];
            lane.Read(expected, expectedValues); lane.Read(actual, actualValues);
            Assert.Equal(expectedValues.Select(BitConverter.SingleToInt32Bits), actualValues.Select(BitConverter.SingleToInt32Bits));
            lane.Run2D("q35l_iq2_s_grid3_coalesced", ((long)shape.Output + 15) / 16 * 16, (long)shape.Rows * 16,
                16, 16, x, grid, scales, d, bias, actual, shape.Rows, shape.Input, shape.Output);
            lane.Read(actual, actualValues);
            Assert.Equal(expectedValues.Select(BitConverter.SingleToInt32Bits), actualValues.Select(BitConverter.SingleToInt32Bits));
            foreach (bool paired in new[] { false, true })
            {
                lane.Run("q35l_iq2_s_grid3_sg16_" + (paired ? "pair" : "fast"), paired ? ((long)shape.Rows * ((shape.Output + 1) / 2) + 1) / 2 * 32 : global,
                    paired ? 32 : lane.Options.Qwen35ProjectionWorkgroupSize, x, grid, scales, d, bias, actual, shape.Rows, shape.Input, shape.Output);
                lane.Read(actual, actualValues);
                Assert.Equal(expectedValues.Select(BitConverter.SingleToInt32Bits), actualValues.Select(BitConverter.SingleToInt32Bits));
            }
            const int rank = 3;
            using ArcBuffer z = lane.Upload(Enumerable.Range(0, shape.Rows * rank).Select(_ => random.NextSingle()).ToArray());
            using ArcBuffer b = lane.Upload(Enumerable.Range(0, shape.Output * rank).Select(_ => random.NextSingle()).ToArray());
            lane.Run("q35l_iq2_s_sg16_lora", global, lane.Options.Qwen35ProjectionWorkgroupSize,
                x, w, bias, expected, shape.Rows, shape.Input, shape.Output, z, b, rank, 0.75f);
            lane.Run("q35l_iq2_s_grid3_sg16_lora", global, lane.Options.Qwen35ProjectionWorkgroupSize,
                x, grid, scales, d, bias, actual, shape.Rows, shape.Input, shape.Output, z, b, rank, 0.75f);
            lane.Read(expected, expectedValues); lane.Read(actual, actualValues);
            Assert.Equal(expectedValues.Select(BitConverter.SingleToInt32Bits), actualValues.Select(BitConverter.SingleToInt32Bits));
            lane.Run2D("q35l_iq2_s_grid3_coalesced_lora", ((long)shape.Output + 15) / 16 * 16, (long)shape.Rows * 16,
                16, 16, x, grid, scales, d, bias, actual, shape.Rows, shape.Input, shape.Output, z, b, rank, 0.75f);
            lane.Read(actual, actualValues);
            Assert.Equal(expectedValues.Select(BitConverter.SingleToInt32Bits), actualValues.Select(BitConverter.SingleToInt32Bits));
            foreach (bool paired in new[] { false, true })
            {
                lane.Run("q35l_iq2_s_grid3_sg16_" + (paired ? "pair_lora" : "fast_lora"), paired ? ((long)shape.Rows * ((shape.Output + 1) / 2) + 1) / 2 * 32 : global,
                    paired ? 32 : lane.Options.Qwen35ProjectionWorkgroupSize, x, grid, scales, d, bias, actual,
                    shape.Rows, shape.Input, shape.Output, z, b, rank, 0.75f);
                lane.Read(actual, actualValues);
                Assert.Equal(expectedValues.Select(BitConverter.SingleToInt32Bits), actualValues.Select(BitConverter.SingleToInt32Bits));
            }
            lane.Run("q35l_iq2_s_sg16", global, lane.Options.Qwen35ProjectionWorkgroupSize, x, w, bias, expected, shape.Rows, shape.Input, shape.Output);
            lane.Run("q35a_zero", 1, 0, status, 1);
            lane.Run("q35l_prefill_xmx_pack_input_f16x2", packedInputs, 0, x, high, low, status, shape.Rows, shape.Input);
            foreach (string kernel in new[] { "q35l_prefill_xmx_iq2_s_grid3", "q35l_prefill_xmx_iq2_s_grid3_bslm" })
            {
                lane.Run2D(kernel, ((long)shape.Output + 63) / 64 * 16, ((long)shape.Rows + 127) / 128 * 16,
                    16, 16, high, low, grid, scales, d, bias, actual, shape.Rows, shape.Input, shape.Output, x, status);
                lane.Read(expected, expectedValues); lane.Read(actual, actualValues);
                double reference2 = 0, error2 = 0;
                for (int i = 0; i < count; ++i)
                {
                    Assert.True(float.IsFinite(actualValues[i]));
                    double difference = actualValues[i] - expectedValues[i]; error2 += difference * difference;
                    reference2 += (double)expectedValues[i] * expectedValues[i];
                }
                double relative = Math.Sqrt(error2 / Math.Max(reference2, 1e-30));
                output.WriteLine($"{kernel} {shape} fallback={outOfRange}: relative L2={relative:E6}");
                Assert.True(relative < 5e-5);
                if (outOfRange) Assert.Equal(expectedValues.Select(BitConverter.SingleToInt32Bits), actualValues.Select(BitConverter.SingleToInt32Bits));
            }
            int[] residentBits = actualValues.Select(BitConverter.SingleToInt32Bits).ToArray();
            lane.Run2D("q35l_prefill_xmx_iq2_s_gguf_bslm", ((long)shape.Output + 63) / 64 * 16,
                ((long)shape.Rows + 127) / 128 * 16, 16, 16, high, low, w, bias, actual,
                shape.Rows, shape.Input, shape.Output, x, status);
            lane.Read(actual, actualValues);
            Assert.Equal(residentBits, actualValues.Select(BitConverter.SingleToInt32Bits));
            output.WriteLine($"GGUF BSLM {shape}: bitwise identical to resident BSLM");
        }
    }

    [Fact]
    public void OriginalGgufBslmReportsBoundedSpeed()
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_XMX_PREFILL_BENCH") != "1", "Opt-in GPU timing benchmark.");
        using var lane = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true,
            ExperimentalOptimizationKernels = true, BufferPoolBytes = 128 * 1024 * 1024 });
        output.WriteLine($"resources GGUF BSLM: {lane.GetKernelResources("q35l_prefill_xmx_iq2_s_gguf_bslm")}");
        foreach (var shape in new[] { (Rows: 128, Input: 5120, Output: 17408),
            (Rows: 256, Input: 17408, Output: 5120), (Rows: 512, Input: 5120, Output: 17408) })
        {
            var random = new Random(shape.Input + 82);
            using ArcBuffer x = lane.Upload(Enumerable.Range(0, shape.Rows * shape.Input).Select(_ => random.NextSingle() * 2 - 1).ToArray());
            using ArcBuffer w = lane.UploadRaw(Payload(random, "iq2_s", 82, shape.Input, shape.Output));
            using ArcBuffer b = lane.Upload(new float[shape.Output]), result = lane.Allocate(shape.Rows * shape.Output);
            using ArcBuffer high = lane.AllocateBytes(shape.Rows * shape.Input * 2), low = lane.AllocateBytes(shape.Rows * shape.Input * 2);
            using ArcBuffer status = lane.Allocate(1);
            int columns = (shape.Output + 15) / 16 * 16;
            using ArcBuffer grid = lane.AllocateBytes(shape.Input / 32 * columns * 12);
            using ArcBuffer scales = lane.AllocateBytes(shape.Input / 32 * columns);
            using ArcBuffer d = lane.AllocateBytes(shape.Input / 256 * columns * 2);
            lane.Run("q35l_prefill_xmx_pack_iq2_s_grid3", (long)shape.Input / 32 * shape.Output, 0,
                w, grid, scales, d, shape.Input, shape.Output);
            foreach (bool resident in new[] { false, true })
            {
                void Run()
                {
                    lane.Run("q35a_zero", 1, 0, status, 1);
                    lane.Run("q35l_prefill_xmx_pack_input_f16x2", shape.Rows * shape.Input, 0,
                        x, high, low, status, shape.Rows, shape.Input);
                    long globalX = ((long)shape.Output + 63) / 64 * 16, globalY = ((long)shape.Rows + 127) / 128 * 16;
                    if (resident) lane.Run2D("q35l_prefill_xmx_iq2_s_grid3_bslm", globalX, globalY, 16, 16,
                        high, low, grid, scales, d, b, result, shape.Rows, shape.Input, shape.Output, x, status);
                    else lane.Run2D("q35l_prefill_xmx_iq2_s_gguf_bslm", globalX, globalY, 16, 16,
                        high, low, w, b, result, shape.Rows, shape.Input, shape.Output, x, status);
                }
                Run(); lane.Synchronize();
                var times = new List<double>();
                for (int iteration = 0; iteration < 5; ++iteration)
                {
                    var watch = Stopwatch.StartNew(); Run(); lane.Synchronize(); times.Add(watch.Elapsed.TotalMilliseconds);
                }
                output.WriteLine($"IQ2 BSLM {shape} resident={resident}: {times.Order().ElementAt(2):F3} ms");
            }
        }
    }

    [Theory]
    [InlineData("iq2_s", 82)]
    [InlineData("iq3_s", 110)]
    [InlineData("q4_k", 144)]
    public void ActualFfnShapeReportsXmxSpeed(string quant, int blockBytes)
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_XMX_PREFILL_BENCH") != "1", "Opt-in GPU timing benchmark.");
        using var lane = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true, BufferPoolBytes = 128 * 1024 * 1024, ExperimentalOptimizationKernels = true });
        output.WriteLine($"resources base: {lane.GetKernelResources($"q35l_prefill_xmx_{quant}_f16x2")}");
        if (quant != "q4_k") output.WriteLine($"resources factored: {lane.GetKernelResources($"q35l_prefill_xmx_{quant}_factored")}");
        if (quant == "iq2_s")
        {
            output.WriteLine($"resources resident grid3: {lane.GetKernelResources("q35l_prefill_xmx_iq2_s_grid3")}");
            output.WriteLine($"resources resident grid3 BSLM: {lane.GetKernelResources("q35l_prefill_xmx_iq2_s_grid3_bslm")}");
            output.WriteLine($"resources packed factored BSLM: {lane.GetKernelResources("q35l_prefill_xmx_packed_factored_bslm")}");
        }
        foreach (var shape in new[] { (Rows: 128, Input: 5120, Output: 17408), (Rows: 256, Input: 17408, Output: 5120),
            (Rows: 512, Input: 5120, Output: 17408) })
        {
            var random = new Random(shape.Input + blockBytes);
            var payload = Payload(random, quant, blockBytes, shape.Input, shape.Output);
            using ArcBuffer x = lane.Upload(Enumerable.Range(0, shape.Rows * shape.Input).Select(_ => random.NextSingle() * 2 - 1).ToArray());
            using ArcBuffer w = lane.UploadRaw(payload), b = lane.Upload(new float[shape.Output]);
            using ArcBuffer high = lane.AllocateBytes(shape.Rows * shape.Input * 2), low = lane.AllocateBytes(shape.Rows * shape.Input * 2);
            using ArcBuffer status = lane.Allocate(1);
            using ArcBuffer weightHigh = lane.AllocateBytes(shape.Input * shape.Output * 2), weightLow = lane.AllocateBytes(shape.Input * shape.Output * 2);
            using ArcBuffer grid = lane.AllocateBytes(shape.Input * shape.Output * 2), scales = lane.Allocate(shape.Input * shape.Output / 16);
            using ArcBuffer gridInt8 = lane.AllocateBytes(shape.Input * shape.Output);
            int colBlocks = (shape.Output + 15) / 16;
            using ArcBuffer grid3 = lane.AllocateBytes(shape.Input / 32 * colBlocks * 48 * 4);
            using ArcBuffer scale3 = lane.AllocateBytes(shape.Input / 32 * colBlocks * 16);
            using ArcBuffer blockD3 = lane.AllocateBytes(shape.Input / 256 * colBlocks * 16 * 2);
            if (quant == "iq2_s")
                lane.Run("q35l_prefill_xmx_pack_iq2_s_grid3", (long)shape.Input / 32 * shape.Output, 0,
                    w, grid3, scale3, blockD3, shape.Input, shape.Output);
            using ArcBuffer result = lane.Allocate(shape.Rows * shape.Output);
            void Run(int mode)
            {
                if (mode is 1 or 3)
                {
                    lane.Run("q35a_zero", 1, 0, status, 1);
                    lane.Run("q35l_prefill_xmx_input_f16x2", shape.Rows * shape.Input, 0, x, high, low, status, shape.Rows * shape.Input);
                    lane.Run2D($"q35l_prefill_xmx_{quant}_" + (mode == 3 && quant != "q4_k" ? "factored" : "f16x2"), ((long)shape.Output + 63) / 64 * 16,
                        ((long)shape.Rows + 127) / 128 * 16, 16, 16, high, low, w, b, result, shape.Rows, shape.Input, shape.Output, x, status);
                }
                else if (mode is 6 or 7)
                {
                    lane.Run("q35a_zero", 1, 0, status, 1);
                    lane.Run("q35l_prefill_xmx_pack_input_f16x2", shape.Rows * shape.Input, 0, x, high, low, status, shape.Rows, shape.Input);
                    lane.Run2D("q35l_prefill_xmx_iq2_s_grid3" + (mode == 7 ? "_bslm" : ""), ((long)shape.Output + 63) / 64 * 16,
                        ((long)shape.Rows + 127) / 128 * 16, 16, 16, high, low, grid3, scale3, blockD3, b, result,
                        shape.Rows, shape.Input, shape.Output, x, status);
                }
                else if (mode is 4 or 5 or 8 && quant != "q4_k")
                {
                    lane.Run("q35a_zero", 1, 0, status, 1);
                    lane.Run("q35l_prefill_xmx_pack_input_f16x2", shape.Rows * shape.Input, 0, x, high, low, status, shape.Rows, shape.Input);
                    ArcBuffer selectedGrid = mode == 5 ? gridInt8 : grid;
                    lane.Run($"q35l_prefill_xmx_pack_{quant}_factored" + (mode == 5 ? "_i8" : ""), (long)shape.Input * shape.Output / 8, 0, w, selectedGrid, scales, shape.Input, shape.Output);
                    lane.Run2D("q35l_prefill_xmx_packed_factored" + (mode == 5 ? "_i8" : mode == 8 ? "_bslm" : ""), ((long)shape.Output + 63) / 64 * 16,
                        ((long)shape.Rows + 127) / 128 * 16, 16, 16, high, low, selectedGrid, scales, b, result,
                        shape.Rows, shape.Input, shape.Output, x, w, status, quant == "iq2_s" ? 0 : 1);
                }
                else if (mode == 2)
                {
                    lane.Run("q35a_zero", 1, 0, status, 1);
                    lane.Run("q35l_prefill_xmx_pack_input_f16x2", shape.Rows * shape.Input, 0, x, high, low, status, shape.Rows, shape.Input);
                    lane.Run($"q35l_prefill_xmx_pack_{quant}_f16x2", (long)shape.Input * shape.Output / 8, 0, w, weightHigh, weightLow, shape.Input, shape.Output);
                    lane.Run2D("q35l_prefill_xmx_packed_f16x2", ((long)shape.Output + 63) / 64 * 16,
                        ((long)shape.Rows + 127) / 128 * 16, 16, 16, high, low, weightHigh, weightLow, b, result,
                        shape.Rows, shape.Input, shape.Output, x, w, status, quant == "iq2_s" ? 0 : quant == "iq3_s" ? 1 : 2);
                }
                else lane.Run($"q35l_prefill_linear_{quant}_rows4", ((((long)shape.Rows + 3) / 4 * shape.Output + 1) / 2) * 32, 32,
                    x, w, b, result, shape.Rows, shape.Input, shape.Output);
            }
            foreach (int mode in quant == "q4_k" ? new[] { 0, 1, 2 } : quant == "iq2_s" ? new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 } : new[] { 0, 1, 2, 3, 4, 5, 8 })
            {
                Run(mode); lane.Synchronize();
                var times = new List<double>();
                for (int iteration = 0; iteration < 3; iteration++)
                {
                    var watch = Stopwatch.StartNew(); Run(mode); lane.Synchronize(); times.Add(watch.Elapsed.TotalMilliseconds);
                }
                output.WriteLine($"{quant} {shape} mode={mode}: {times.Order().ElementAt(1):F3} ms");
            }
            if (quant == "iq2_s" && shape.Rows == 128)
            {
                foreach (int resident in new[] { 0, 1, 2, 3, 4 })
                {
                    void Gemv()
                    {
                        int group = lane.Options.Qwen35ProjectionWorkgroupSize;
                        long global = ((long)shape.Output + group / 16 - 1) / (group / 16) * group;
                        if (resident == 3) lane.Run("q35l_iq2_s_grid3_sg16_pair", (((long)shape.Output + 1) / 2 + 1) / 2 * 32,
                            32, x, grid3, scale3, blockD3, b, result, 1, shape.Input, shape.Output);
                        else if (resident == 4) lane.Run("q35l_iq2_s_grid3_sg16_fast", global, group,
                            x, grid3, scale3, blockD3, b, result, 1, shape.Input, shape.Output);
                        else if (resident == 2) lane.Run2D("q35l_iq2_s_grid3_coalesced", ((long)shape.Output + 15) / 16 * 16, 16,
                            16, 16, x, grid3, scale3, blockD3, b, result, 1, shape.Input, shape.Output);
                        else if (resident == 1) lane.Run("q35l_iq2_s_grid3_sg16", global, group, x, grid3, scale3, blockD3, b, result, 1, shape.Input, shape.Output);
                        else lane.Run("q35l_iq2_s_sg16", global, group, x, w, b, result, 1, shape.Input, shape.Output);
                    }
                    Gemv(); lane.Synchronize(); var times = new List<double>();
                    for (int iteration = 0; iteration < 5; ++iteration)
                    {
                        var watch = Stopwatch.StartNew(); Gemv(); lane.Synchronize(); times.Add(watch.Elapsed.TotalMilliseconds);
                    }
                    output.WriteLine($"GEMV resident={resident}: {times.Order().ElementAt(2):F3} ms");
                }
            }
        }
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
