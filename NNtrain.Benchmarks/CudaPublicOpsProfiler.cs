using System.Diagnostics;
using NNtrain.Runtime.Execution;

namespace NNtrain.Benchmarks;

internal static class CudaPublicOpsProfiler
{
    internal static void Run(int warmup, int iterations, int length)
    {
        if (!Tensor.IsCudaAvailable())
            throw new InvalidOperationException("CUDA is required.");
        ArgumentOutOfRangeException.ThrowIfNegative(warmup);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(iterations);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);

        TensorDevice previous = Tensor.ExecutionDevice;
        Tensor.ExecutionDevice = TensorDevice.Cuda;
        string deviceName = Tensor.ExecutionDeviceName;
        Tensor.ExecutionDevice = previous;

        float[] leftValues = Pattern(length, 0.002f, 257);
        float[] rightValues = Pattern(length, 0.003f, 251);
        double cpu = MeasureElementwise(
            TensorDevice.Cpu, leftValues, rightValues, warmup, iterations);
        double cuda = MeasureElementwise(
            TensorDevice.Cuda, leftValues, rightValues, warmup, iterations);

        const int sequence = 512;
        const int width = 64;
        int channels = 3 * width;
        float[] projectedValues = Pattern(sequence * channels, 0.002f, 263);
        float[] shortValues = Pattern(3 * channels, 0.003f, 127);
        float[] longValues = Pattern(sequence * width, 0.002f, 257);
        float[] diagonalValues = Pattern(width, 0.004f, 61);
        (double direct, double parallel) = MeasureHyena(
            projectedValues,
            shortValues,
            longValues,
            diagonalValues,
            warmup,
            Math.Max(3, iterations / 5));

        Console.WriteLine(
            $"CUDA public-op microbenchmark: {deviceName}");
        Console.WriteLine(
            $"conditions: BF16, no-grad, length={length:N0}, " +
            $"warmup={warmup}, measured={iterations}, synchronized batches");
        Console.WriteLine(
            $"elementwise add+GELU+tanh+mean: CPU {cpu:F3} ms, " +
            $"CUDA {cuda:F3} ms, speedup {cpu / cuda:F2}x");
        Console.WriteLine(
            $"Hyena [1,{sequence},{width}], measured={Math.Max(3, iterations / 5)}: " +
            $"direct {direct:F3} ms, parallel-long {parallel:F3} ms, " +
            $"speedup {direct / parallel:F2}x");
    }

    private static double MeasureElementwise(
        TensorDevice device,
        float[] leftValues,
        float[] rightValues,
        int warmup,
        int iterations)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = device;
            Tensor left = new(
                leftValues, [leftValues.Length], dtype: TensorDType.BFloat16);
            Tensor right = new(
                rightValues, [rightValues.Length], dtype: TensorDType.BFloat16);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.BFloat16);
            using IDisposable noGrad = AutogradContext.NoGrad();
            if (device == TensorDevice.Cuda)
            {
                _ = left.EnsureCudaBFloat16Buffer(0);
                _ = right.EnsureCudaBFloat16Buffer(0);
                using CudaInferenceScope inference = CudaInferenceScope.Begin();
                return MeasureCuda(
                    () => _ = (left + right).Gelu().Tanh().Mean(),
                    warmup,
                    iterations);
            }
            return MeasureCpu(
                () => _ = (left + right).Gelu().Tanh().Mean(),
                warmup,
                iterations);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static (double Direct, double Parallel) MeasureHyena(
        float[] projectedValues,
        float[] shortValues,
        float[] longValues,
        float[] diagonalValues,
        int warmup,
        int iterations)
    {
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor projected = new(
                projectedValues, [1, 512, 192], dtype: TensorDType.BFloat16);
            Tensor shortFilter = new(
                shortValues, [3, 192], dtype: TensorDType.BFloat16);
            Tensor longFilter = new(
                longValues, [512, 64], dtype: TensorDType.BFloat16);
            Tensor diagonal = new(
                diagonalValues, [64], dtype: TensorDType.BFloat16);
            using IDisposable policy = TensorExecutionContext.PushPrecisionPolicy(
                PrecisionPolicy.BFloat16);
            using IDisposable noGrad = AutogradContext.NoGrad();
            foreach (Tensor tensor in new[]
            {
                projected, shortFilter, longFilter, diagonal,
            })
            {
                _ = tensor.EnsureCudaBFloat16Buffer(0);
            }
            using CudaInferenceScope inference = CudaInferenceScope.Begin();
            double direct = MeasureCuda(
                () => _ = projected.FusedCausalHyenaOrder2(
                    shortFilter,
                    longFilter,
                    diagonal,
                    HyenaConvolutionAlgorithm.Direct),
                warmup,
                iterations);
            double parallel = MeasureCuda(
                () => _ = projected.FusedCausalHyenaOrder2(
                    shortFilter,
                    longFilter,
                    diagonal,
                    HyenaConvolutionAlgorithm.Fft),
                warmup,
                iterations);
            return (direct, parallel);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    private static double MeasureCuda(
        Action action,
        int warmup,
        int iterations)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(0);
        for (int index = 0; index < warmup; index++)
            action();
        accelerator.Synchronize();
        var stopwatch = Stopwatch.StartNew();
        for (int index = 0; index < iterations; index++)
            action();
        accelerator.Synchronize();
        stopwatch.Stop();
        return stopwatch.Elapsed.TotalMilliseconds / iterations;
    }

    private static double MeasureCpu(
        Action action,
        int warmup,
        int iterations)
    {
        for (int index = 0; index < warmup; index++)
            action();
        var stopwatch = Stopwatch.StartNew();
        for (int index = 0; index < iterations; index++)
            action();
        stopwatch.Stop();
        return stopwatch.Elapsed.TotalMilliseconds / iterations;
    }

    private static float[] Pattern(int length, float scale, int period)
        => Enumerable.Range(0, length)
            .Select(index => ((index % period) - period / 2) * scale)
            .ToArray();
}
