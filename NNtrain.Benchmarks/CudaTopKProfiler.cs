using System.Diagnostics;
using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;

namespace NNtrain.Benchmarks;

internal static class CudaTopKProfiler
{
    internal static void Run(int warmup, int iterations, int vocabulary)
    {
        if (!Tensor.IsCudaAvailable())
            throw new InvalidOperationException("CUDA is required for this benchmark.");
        ArgumentOutOfRangeException.ThrowIfNegative(warmup);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(iterations);
        if (vocabulary < 64)
            throw new ArgumentOutOfRangeException(nameof(vocabulary));

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            using var execution = new ExecutionSession(new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(0),
                Precision = PrecisionPolicy.BFloat16,
            });
            execution.AttachLane(CudaExecutionLaneFactory.Create(0));
            using IDisposable executionScope = execution.Enter();

            float[] values = Enumerable.Range(0, vocabulary)
                .Select(index => MathF.Sin(index * 0.013f) * 4f
                    + MathF.Cos(index * 0.0037f))
                .ToArray();
            values[3] = 12f;
            values[vocabulary / 2] = 12f;
            values[^1] = 12f;
            var logits = new Tensor(
                values,
                [vocabulary],
                dtype: TensorDType.BFloat16);
            try
            {
                logits.to(new TorchDevice(TensorDevice.Cuda, 0));
                NativeCudaBuffer<ushort> deviceValues =
                    logits.EnsureCudaBFloat16Buffer(0);

                Console.WriteLine(
                    $"CUDA vocabulary sampling BF16 [device=0, " +
                    $"vocab={vocabulary}, warmup={warmup}, " +
                    $"iterations={iterations}]");
                Benchmark(1, warmup, iterations, logits, deviceValues);
                Benchmark(40, warmup, iterations, logits, deviceValues);
            }
            finally
            {
                logits.InvalidateCudaBuffers();
            }
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }

    private static void Benchmark(
        int k,
        int warmup,
        int iterations,
        Tensor logits,
        NativeCudaBuffer<ushort> deviceValues)
    {
        int[] NativeSelection()
            => logits.ReadCudaTopK(0, logits.Numel, k, 0)
                .Candidates
                .Select(candidate => candidate.Index)
                .ToArray();

        var encoded = new ushort[logits.Numel];
        var decoded = new float[logits.Numel];
        int[] LegacySelection()
        {
            deviceValues.CopyToCPU(encoded);
            TensorStorageCodec.DecodeBFloat16(encoded, decoded);
            return Enumerable.Range(0, decoded.Length)
                .OrderByDescending(index => decoded[index])
                .ThenBy(index => index)
                .Take(k)
                .ToArray();
        }

        for (int index = 0; index < warmup; index++)
        {
            _ = NativeSelection();
            _ = LegacySelection();
        }

        Measurement legacy = Measure(LegacySelection, iterations);
        Measurement native = Measure(NativeSelection, iterations);
        bool matches = legacy.Last.SequenceEqual(native.Last);
        long legacyD2H = legacy.Transfer.DeviceToHostBytes / iterations;
        long nativeD2H = native.Transfer.DeviceToHostBytes / iterations;
        double transferReduction = legacyD2H / (double)nativeD2H;
        Console.WriteLine(
            $"  topK={k}: legacy full-D2H+CPU p50=" +
            $"{Median(legacy.Samples):F3} ms, D2H={legacyD2H} B/call; " +
            $"CUDA two-stage p50={Median(native.Samples):F3} ms, " +
            $"D2H={nativeD2H} B/call; speedup=" +
            $"{Median(legacy.Samples) / Median(native.Samples):F2}x, " +
            $"transfer={transferReduction:F1}x smaller, " +
            $"numeric-match={matches}");
        if (!matches)
            throw new InvalidOperationException("CUDA top-K did not match the reference.");
    }

    private static Measurement Measure(Func<int[]> action, int iterations)
    {
        var samples = new double[iterations];
        int[] last = [];
        NativeCudaTransferTelemetry before = NativeCudaRuntime.TransferTelemetry;
        for (int index = 0; index < iterations; index++)
        {
            long start = Stopwatch.GetTimestamp();
            last = action();
            samples[index] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }
        NativeCudaTransferTelemetry transfer =
            NativeCudaRuntime.TransferTelemetry - before;
        return new Measurement(samples, transfer, last);
    }

    private static double Median(double[] values)
    {
        double[] ordered = values.Order().ToArray();
        return ordered[ordered.Length / 2];
    }

    private readonly record struct Measurement(
        double[] Samples,
        NativeCudaTransferTelemetry Transfer,
        int[] Last);
}
