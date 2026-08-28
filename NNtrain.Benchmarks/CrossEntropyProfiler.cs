using System.Diagnostics;

namespace NNtrain.Benchmarks;

/// <summary>
/// Isolates the production language-model loss shape without constructing a
/// second transformer graph. Logits and their BF16 gradient share one device
/// buffer, matching the destructive one-shot loss-head contract.
/// </summary>
internal static class CrossEntropyProfiler
{
    internal static void Run(int warmup, int iterations)
    {
        if (!Tensor.IsCudaAvailable())
            throw new InvalidOperationException("CUDA is not available.");
        ArgumentOutOfRangeException.ThrowIfNegative(warmup);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(iterations);

        const int batch = 36;
        const int sequence = 512;
        const int columns = 11_500;
        const int rows = batch * sequence;
        const int length = rows * columns;
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(0);
            using NativeCudaBuffer<ushort> logits =
                accelerator.Allocate1D<ushort>(length);
            using NativeCudaBuffer<int> labels =
                accelerator.Allocate1D(new int[rows]);
            using NativeCudaBuffer<float> maxima =
                accelerator.Allocate1D<float>(rows);
            using NativeCudaBuffer<float> inverseSums =
                accelerator.Allocate1D<float>(rows);
            using NativeCudaBuffer<float> rowLosses =
                accelerator.Allocate1D<float>(rows);
            using NativeCudaBuffer<float> loss =
                accelerator.Allocate1D<float>(1);
            using NativeCudaBuffer<float> upstream =
                accelerator.Allocate1D([1f]);

            var forward = new double[iterations];
            var backward = new double[iterations];
            NativeCudaTransferTelemetry transfersBefore =
                NativeCudaRuntime.TransferTelemetry;
            NativeCudaAllocationTelemetry allocationsBefore =
                NativeCudaRuntime.AllocationTelemetry;
            for (int iteration = -warmup;
                 iteration < iterations;
                 iteration++)
            {
                logits.MemSetToZero();
                accelerator.Synchronize();
                long started = Stopwatch.GetTimestamp();
                CudaTensorNative.CrossEntropy(
                    0,
                    logits.NativePtr,
                    labels.NativePtr,
                    maxima.NativePtr,
                    inverseSums.NativePtr,
                    rowLosses.NativePtr,
                    loss.NativePtr,
                    rows,
                    columns,
                    Tensor.DefaultCrossEntropyIgnoreIndex,
                    rows,
                    smoothing: 0f,
                    bfloat16: true);
                accelerator.Synchronize();
                double forwardMilliseconds = Stopwatch
                    .GetElapsedTime(started).TotalMilliseconds;

                started = Stopwatch.GetTimestamp();
                CudaTensorNative.CrossEntropyBackwardBFloat16Output(
                    0,
                    logits.NativePtr,
                    maxima.NativePtr,
                    inverseSums.NativePtr,
                    labels.NativePtr,
                    logits.NativePtr,
                    upstream.NativePtr,
                    length,
                    columns,
                    Tensor.DefaultCrossEntropyIgnoreIndex,
                    rows,
                    smoothing: 0f);
                accelerator.Synchronize();
                double backwardMilliseconds = Stopwatch
                    .GetElapsedTime(started).TotalMilliseconds;
                if (iteration >= 0)
                {
                    forward[iteration] = forwardMilliseconds;
                    backward[iteration] = backwardMilliseconds;
                }
            }

            Array.Sort(forward);
            Array.Sort(backward);
            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - transfersBefore;
            NativeCudaAllocationTelemetry allocations =
                NativeCudaRuntime.AllocationTelemetry - allocationsBefore;
            Console.WriteLine(
                $"Cross entropy BF16 direct: shape=[{batch},{sequence}," +
                $"{columns}], rows={rows:N0}, values={length:N0}, " +
                $"warmup={warmup}, iterations={iterations}");
            Console.WriteLine(
                $"  forward p50={forward[iterations / 2]:F3} ms, " +
                $"mean={forward.Average():F3} ms");
            Console.WriteLine(
                $"  backward/in-place p50={backward[iterations / 2]:F3} ms, " +
                $"mean={backward.Average():F3} ms");
            Console.WriteLine(
                $"  H2D/D2H={transfers.HostToDeviceBytes}/" +
                $"{transfers.DeviceToHostBytes} B, malloc/free=" +
                $"{allocations.AllocationCount}/{allocations.FreeCount}");
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }
}
