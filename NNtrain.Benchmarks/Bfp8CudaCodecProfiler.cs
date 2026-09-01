using System.Diagnostics;
using System.Text.Json;

namespace NNtrain.Benchmarks;

internal static class Bfp8CudaCodecProfiler
{
    internal static void Run(
        int warmup,
        int iterations,
        int length,
        int blockSize,
        string? outputPath = null)
    {
        if (!Tensor.IsCudaAvailable())
            throw new InvalidOperationException("CUDA is not available.");
        ArgumentOutOfRangeException.ThrowIfNegative(warmup);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(iterations);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockSize);
        if (blockSize >= length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(blockSize),
                blockSize,
                "The codec benchmark requires block-scaled BFP8 storage.");
        }

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            NativeCudaDevice device = ForgetMemoryV2Cuda.GetAccelerator(0);
            Bfp8QuantizationDescriptor descriptor =
                Bfp8QuantizationDescriptor.Block(blockSize);
            var source = new float[length];
            for (int index = 0; index < source.Length; index++)
            {
                source[index] =
                    MathF.Sin(index * 0.00391f) * 0.73f
                    + MathF.Cos(index * 0.00173f) * 0.19f;
            }

            using NativeCudaBuffer<float> floatInput =
                device.Allocate1D(source);
            using NativeCudaBuffer<float> floatOutput =
                device.Allocate1D<float>(length);
            using NativeCudaBuffer<ushort> bfloat16 =
                device.Allocate1D<ushort>(length);
            using NativeCudaBuffer<sbyte> payload =
                device.Allocate1D<sbyte>(length);
            using NativeCudaBuffer<float> scales = device.Allocate1D<float>(
                descriptor.GetScaleCount(length));
            nint stream = device.DefaultStream;

            // Seed a valid encoded tensor and a BF16 source before measuring.
            CudaBfp8Native.QuantizeFloat32(
                0, floatInput, payload, scales, descriptor, stream);
            CudaBfp8Native.DequantizeBFloat16(
                0, payload, scales, bfloat16, descriptor, stream);
            device.Synchronize();

            CodecMeasurement[] measurements =
            [
                Measure(
                    "quantize-f32",
                    warmup,
                    iterations,
                    device,
                    () => CudaBfp8Native.QuantizeFloat32(
                        0, floatInput, payload, scales, descriptor, stream)),
                Measure(
                    "dequantize-f32",
                    warmup,
                    iterations,
                    device,
                    () => CudaBfp8Native.DequantizeFloat32(
                        0, payload, scales, floatOutput, descriptor, stream)),
                Measure(
                    "roundtrip-f32",
                    warmup,
                    iterations,
                    device,
                    () => CudaBfp8Native.QuantizeFloat32Roundtrip(
                        0, floatInput, payload, scales, descriptor, stream)),
                Measure(
                    "quantize-bf16",
                    warmup,
                    iterations,
                    device,
                    () => CudaBfp8Native.QuantizeBFloat16(
                        0, bfloat16, payload, scales, descriptor, stream)),
                Measure(
                    "dequantize-bf16",
                    warmup,
                    iterations,
                    device,
                    () => CudaBfp8Native.DequantizeBFloat16(
                        0, payload, scales, bfloat16, descriptor, stream)),
            ];

            var result = new CodecBenchmarkResult(
                Schema: "nntrain.bfp8-codec-benchmark/v1",
                TimestampUtc: DateTimeOffset.UtcNow,
                Device: device.Name,
                Length: length,
                BlockSize: blockSize,
                ScaleCount: descriptor.GetScaleCount(length),
                Warmup: warmup,
                Iterations: iterations,
                Measurements: measurements);
            string json = JsonSerializer.Serialize(
                result,
                new JsonSerializerOptions { WriteIndented = true });
            Console.WriteLine(json);
            if (!string.IsNullOrWhiteSpace(outputPath))
            {
                string fullPath = Path.GetFullPath(outputPath);
                string? directory = Path.GetDirectoryName(fullPath);
                if (!string.IsNullOrEmpty(directory))
                    Directory.CreateDirectory(directory);
                File.WriteAllText(fullPath, json);
            }
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private static CodecMeasurement Measure(
        string operation,
        int warmup,
        int iterations,
        NativeCudaDevice device,
        Action launch)
    {
        for (int index = 0; index < warmup; index++)
        {
            launch();
            device.Synchronize();
        }

        var samples = new double[iterations];
        for (int index = 0; index < iterations; index++)
        {
            long started = Stopwatch.GetTimestamp();
            launch();
            device.Synchronize();
            samples[index] = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
        }
        Array.Sort(samples);
        int p95Index = Math.Min(
            samples.Length - 1,
            (int)Math.Ceiling(samples.Length * 0.95d) - 1);
        return new CodecMeasurement(
            operation,
            samples.Average(),
            samples[samples.Length / 2],
            samples[p95Index],
            samples[0]);
    }

    private sealed record CodecBenchmarkResult(
        string Schema,
        DateTimeOffset TimestampUtc,
        string Device,
        int Length,
        int BlockSize,
        int ScaleCount,
        int Warmup,
        int Iterations,
        CodecMeasurement[] Measurements);

    private sealed record CodecMeasurement(
        string Operation,
        double MeanMilliseconds,
        double P50Milliseconds,
        double P95Milliseconds,
        double MinimumMilliseconds);
}
