using System.Diagnostics;
using NNtrain;

namespace NNtrain.Benchmarks;

internal static class GpuPrimitiveProfiler
{
    internal static void Run(int warmup, int iterations)
    {
        if (!Tensor.IsCudaAvailable())
            throw new InvalidOperationException("CUDA is unavailable.");
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            Console.WriteLine(
                $"GPU primitive benchmark: {Tensor.ExecutionDeviceName}; " +
                $"warmup={warmup}, iterations={iterations}, storage=BF16, " +
                "accumulation=FP32");
            BenchmarkForgetMemory(warmup, iterations);
            BenchmarkMatMul(warmup, iterations);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }

    private static void BenchmarkForgetMemory(int warmup, int iterations)
    {
        const int batch = 96;
        const int sequence = 512;
        const int keyWidth = 16;
        const int valueWidth = 16;
        const int projectionWidth = 2 * keyWidth + 3 * valueWidth;
        var projected = new Tensor(
            MakeValues(batch * sequence * projectionWidth, 0.013f),
            [batch, sequence, projectionWidth],
            dtype: TensorDType.BFloat16);
        projected.To(TensorDevice.Cuda);

        double scalar = MeasureCuda(
            warmup,
            iterations,
            () => projected.ForgetMemoryV2(
                keyWidth, valueWidth, retentionFloor: 0.5f),
            CudaDispatchPolicy.Defaults with
            {
                DisableTensorCoreForgetMemory = true,
            });
        double tensorCore = MeasureCuda(
            warmup,
            iterations,
            () => projected.ForgetMemoryV2(
                keyWidth, valueWidth, retentionFloor: 0.5f),
            CudaDispatchPolicy.Defaults);
        Console.WriteLine(
            $"ForgetMemory forward [{batch},{sequence},K{keyWidth},V{valueWidth}]: " +
            $"native CUDA scalar {scalar:F3} ms, tiled WMMA {tensorCore:F3} ms, " +
            $"speedup {scalar / tensorCore:F2}x");
    }

    private static void BenchmarkMatMul(int warmup, int iterations)
    {
        const int m = 512;
        const int k = 384;
        const int n = 384;
        float[] leftValues = MakeValues(m * k, 0.017f);
        float[] rightValues = MakeValues(k * n, 0.019f);
        var gpuLeft = new Tensor(
            leftValues, [m, k], dtype: TensorDType.BFloat16);
        var gpuRight = new Tensor(
            rightValues, [k, n], dtype: TensorDType.BFloat16);
        gpuLeft.To(TensorDevice.Cuda);
        gpuRight.To(TensorDevice.Cuda);
        double gpu = MeasureCuda(
            warmup,
            iterations,
            () => gpuLeft.MatMul(gpuRight));

        Tensor.ExecutionDevice = TensorDevice.Cpu;
        var cpuLeft = new Tensor(
            leftValues, [m, k], dtype: TensorDType.BFloat16);
        var cpuRight = new Tensor(
            rightValues, [k, n], dtype: TensorDType.BFloat16);
        double cpu = MeasureCpu(
            Math.Min(warmup, 1),
            iterations,
            () => cpuLeft.MatMul(cpuRight));
        Tensor.ExecutionDevice = TensorDevice.Cuda;
        Console.WriteLine(
            $"MatMul forward [{m},{k}]x[{k},{n}]: CPU {cpu:F3} ms, " +
            $"CUDA BF16 Tensor Core {gpu:F3} ms, speedup {cpu / gpu:F2}x");

        const int batch = 32;
        const int batchM = 128;
        const int batchK = 64;
        const int batchN = 64;
        float[] batchLeftValues = MakeValues(batch * batchM * batchK, 0.021f);
        float[] batchRightValues = MakeValues(batch * batchK * batchN, 0.023f);
        var gpuBatchLeft = new Tensor(
            batchLeftValues,
            [batch, batchM, batchK],
            dtype: TensorDType.BFloat16);
        var gpuBatchRight = new Tensor(
            batchRightValues,
            [batch, batchK, batchN],
            dtype: TensorDType.BFloat16);
        gpuBatchLeft.To(TensorDevice.Cuda);
        gpuBatchRight.To(TensorDevice.Cuda);
        double batchedGpu = MeasureCuda(
            warmup,
            iterations,
            () => gpuBatchLeft.BatchedMatMul(gpuBatchRight));
        Tensor.ExecutionDevice = TensorDevice.Cpu;
        var cpuBatchLeft = new Tensor(
            batchLeftValues,
            [batch, batchM, batchK],
            dtype: TensorDType.BFloat16);
        var cpuBatchRight = new Tensor(
            batchRightValues,
            [batch, batchK, batchN],
            dtype: TensorDType.BFloat16);
        double batchedCpu = MeasureCpu(
            Math.Min(warmup, 1),
            iterations,
            () => cpuBatchLeft.BatchedMatMul(cpuBatchRight));
        Tensor.ExecutionDevice = TensorDevice.Cuda;
        Console.WriteLine(
            $"BatchedMatMul forward [{batch},{batchM},{batchK}]x" +
            $"[{batch},{batchK},{batchN}]: CPU {batchedCpu:F3} ms, " +
            $"CUDA strided BF16 Tensor Core {batchedGpu:F3} ms, " +
            $"speedup {batchedCpu / batchedGpu:F2}x");
    }

    private static double MeasureCuda(
        int warmup,
        int iterations,
        Func<Tensor> operation,
        CudaDispatchPolicy? dispatchPolicy = null)
    {
        using IDisposable dispatch = CudaDispatchPolicy.Push(
            dispatchPolicy ?? CudaDispatchPolicy.Current);
        for (int index = 0; index < warmup; ++index)
            RunCuda(operation);
        var samples = new double[iterations];
        for (int index = 0; index < iterations; ++index)
        {
            long start = Stopwatch.GetTimestamp();
            RunCuda(operation);
            samples[index] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }
        return samples.Average();
    }

    private static void RunCuda(Func<Tensor> operation)
    {
        using var noGrad = AutogradContext.NoGrad();
        using var scope = CudaInferenceScope.Begin();
        Tensor result = operation();
        ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
        GC.KeepAlive(result);
    }

    private static double MeasureCpu(
        int warmup,
        int iterations,
        Func<Tensor> operation)
    {
        using var noGrad = AutogradContext.NoGrad();
        for (int index = 0; index < warmup; ++index)
            GC.KeepAlive(operation());
        var samples = new double[iterations];
        for (int index = 0; index < iterations; ++index)
        {
            long start = Stopwatch.GetTimestamp();
            Tensor result = operation();
            samples[index] = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            GC.KeepAlive(result);
        }
        return samples.Average();
    }

    private static float[] MakeValues(int length, float scale)
        => Enumerable.Range(0, length)
            .Select(index => MathF.Sin(index * scale) * 0.25f)
            .ToArray();
}
