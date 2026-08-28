using System.Diagnostics;
using NNtrain.Cuda.Quantization;

namespace NNtrain.Benchmarks;

internal static class Bfp8CudaGemmProfiler
{
    internal static void Run(
        int warmup,
        int iterations,
        int m = 256,
        int k = 512,
        int n = 256)
    {
        if (!Tensor.IsCudaAvailable())
            throw new InvalidOperationException("CUDA is not available.");
        ArgumentOutOfRangeException.ThrowIfNegative(warmup);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(iterations);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(m);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(n);

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            float[] leftValues = Values(m * k, 5);
            float[] rightValues = Values(k * n, 37);

            Profile(
                "tensor-wide/int8",
                Tensor.FromBfp8(
                    leftValues,
                    [m, k],
                    Bfp8QuantizationDescriptor.TensorWide),
                Tensor.FromBfp8(
                    rightValues,
                    [k, n],
                    Bfp8QuantizationDescriptor.TensorWide),
                warmup,
                iterations,
                m,
                k,
                n);
            Profile(
                "block128/bf16-fallback",
                Tensor.FromBfp8(
                    leftValues,
                    [m, k],
                    Bfp8QuantizationDescriptor.Mix8_32),
                Tensor.FromBfp8(
                    rightValues,
                    [k, n],
                    Bfp8QuantizationDescriptor.Mix8_32),
                warmup,
                iterations,
                m,
                k,
                n);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private static void Profile(
        string name,
        Tensor left,
        Tensor right,
        int warmup,
        int iterations,
        int m,
        int k,
        int n)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(0);
        left.to(new TorchDevice(TensorDevice.Cuda, 0));
        right.to(new TorchDevice(TensorDevice.Cuda, 0));
        using IDisposable noGrad = AutogradContext.NoGrad();

        for (int index = 0; index < warmup; index++)
            RunOnce(left, right, accelerator);
        NativeCudaTransferTelemetry transfersBefore =
            NativeCudaRuntime.TransferTelemetry;
        CudaBfp8GemmTelemetrySnapshot routesBefore =
            CudaBfp8GemmTelemetry.Snapshot;
        var samples = new double[iterations];
        for (int index = 0; index < iterations; index++)
        {
            long started = Stopwatch.GetTimestamp();
            RunOnce(left, right, accelerator);
            samples[index] = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
        }
        Array.Sort(samples);
        NativeCudaTransferTelemetry transfers =
            NativeCudaRuntime.TransferTelemetry - transfersBefore;
        CudaBfp8GemmTelemetrySnapshot routes =
            CudaBfp8GemmTelemetry.Snapshot - routesBefore;
        double p50 = samples[samples.Length / 2];
        double average = samples.Average();
        Console.WriteLine(
            $"BFP8 GEMM {name}: shape=[{m},{k}]x[{k},{n}], " +
            $"warmup={warmup}, iterations={iterations}, " +
            $"p50={p50:F3} ms, mean={average:F3} ms, " +
            $"int8={routes.Int8TensorCoreExecutions}, " +
            $"bf16={routes.BFloat16FallbackExecutions}, " +
            $"H2D={transfers.HostToDeviceBytes} B, " +
            $"D2H={transfers.DeviceToHostBytes} B");

        left.InvalidateCudaBuffers();
        right.InvalidateCudaBuffers();
    }

    private static void RunOnce(
        Tensor left,
        Tensor right,
        NativeCudaDevice accelerator)
    {
        Tensor output = left.MatMul(right);
        accelerator.Synchronize();
        output.InvalidateCudaBuffers();
    }

    private static float[] Values(int length, int offset)
        => Enumerable.Range(0, length)
            .Select(index =>
                MathF.Sin((index + offset) * 0.017f) * 0.31f
                + MathF.Cos((index + offset) * 0.011f) * 0.09f)
            .ToArray();
}
