using System.Diagnostics;

namespace NNtrain.Benchmarks;

internal static class EmbeddingBackwardProfiler
{
    internal static void Run(int warmup, int iterations)
    {
        if (!Tensor.IsCudaAvailable())
            throw new InvalidOperationException("CUDA is not available.");
        ArgumentOutOfRangeException.ThrowIfNegative(warmup);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(iterations);

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            const int batch = 36;
            const int sequence = 512;
            const int width = 1024;
            const int tokenRows = 32_000;
            int positionCount = batch * sequence;
            int length = positionCount * width;
            int[] indices = Enumerable.Range(0, positionCount)
                .Select(position => position % 5 == 0
                    ? position % 127
                    : (position * 7_919 + position / sequence * 17)
                        % tokenRows)
                .ToArray();
            float[] outputGradient = Enumerable.Range(0, length)
                .Select(index =>
                    MathF.Sin((index + 11) * 0.00071f) * 0.013f)
                .ToArray();
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(0);
            using NativeCudaBuffer<int> indicesBuffer =
                accelerator.Allocate1D(indices);
            using NativeCudaBuffer<float> outputGradientBuffer =
                accelerator.Allocate1D(outputGradient);
            using NativeCudaBuffer<float> tokenGradient =
                accelerator.Allocate1D<float>(tokenRows * width);
            using NativeCudaBuffer<float> positionGradient =
                accelerator.Allocate1D<float>(sequence * width);
            int workspaceInts =
                CudaEmbeddingBackwardDispatcher.GetWorkspaceIntCount(
                    positionCount);
            using NativeCudaBuffer<int> workspace =
                accelerator.Allocate1D<int>(workspaceInts);

            ProfileResult legacy = Measure(
                warmup,
                iterations,
                accelerator,
                tokenGradient,
                positionGradient,
                () => CudaTensorNative.EmbeddingPositionsBackward(
                    0,
                    indicesBuffer.NativePtr,
                    outputGradientBuffer.NativePtr,
                    tokenGradient.NativePtr,
                    positionGradient.NativePtr,
                    length,
                    sequence,
                    width));
            ProfileResult reduced = Measure(
                warmup,
                iterations,
                accelerator,
                tokenGradient,
                positionGradient,
                () => CudaTensorNative.EmbeddingPositionsBackwardReduced(
                    0,
                    indicesBuffer.NativePtr,
                    outputGradientBuffer.NativePtr,
                    tokenGradient.NativePtr,
                    positionGradient.NativePtr,
                    workspace.NativePtr,
                    workspaceInts,
                    length,
                    sequence,
                    width));

            long legacyAtomics = checked((long)length * 2);
            long reducedGradientAtomics = 0;
            Console.WriteLine(
                "Embedding backward: " +
                $"shape=[{batch},{sequence},{width}], tokenRows={tokenRows}, " +
                $"warmup={warmup}, iterations={iterations}");
            Console.WriteLine(
                $"  legacy atomic: p50={Percentile50(legacy.Samples):F3} ms, " +
                $"mean={legacy.Samples.Average():F3} ms, " +
                $"table atomicAdd/call={legacyAtomics:N0}, " +
                $"H2D/D2H={legacy.Transfers.HostToDeviceBytes}/" +
                $"{legacy.Transfers.DeviceToHostBytes} B, " +
                $"malloc/free={legacy.Allocations.AllocationCount}/" +
                $"{legacy.Allocations.FreeCount}");
            Console.WriteLine(
                $"  owner reduced: p50={Percentile50(reduced.Samples):F3} ms, " +
                $"mean={reduced.Samples.Average():F3} ms, " +
                $"table atomicAdd/call={reducedGradientAtomics}, " +
                $"hash atomic lower-bound/call={positionCount * 2:N0}, " +
                $"workspace={workspaceInts * sizeof(int) / 1024.0:F1} KiB, " +
                $"H2D/D2H={reduced.Transfers.HostToDeviceBytes}/" +
                $"{reduced.Transfers.DeviceToHostBytes} B, " +
                $"malloc/free={reduced.Allocations.AllocationCount}/" +
                $"{reduced.Allocations.FreeCount}");
            Console.WriteLine(
                $"  speedup={Percentile50(legacy.Samples) / Percentile50(reduced.Samples):F2}x, " +
                $"gradient atomic reduction=100.00%");
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private static ProfileResult Measure(
        int warmup,
        int iterations,
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> tokenGradient,
        NativeCudaBuffer<float> positionGradient,
        Action launch)
    {
        var samples = new double[iterations];
        NativeCudaTransferTelemetry transfersBefore =
            NativeCudaRuntime.TransferTelemetry;
        NativeCudaAllocationTelemetry allocationsBefore =
            NativeCudaRuntime.AllocationTelemetry;
        for (int iteration = -warmup; iteration < iterations; iteration++)
        {
            tokenGradient.MemSetToZero();
            positionGradient.MemSetToZero();
            accelerator.Synchronize();
            long started = Stopwatch.GetTimestamp();
            launch();
            accelerator.Synchronize();
            double elapsed =
                Stopwatch.GetElapsedTime(started).TotalMilliseconds;
            if (iteration >= 0)
                samples[iteration] = elapsed;
        }
        Array.Sort(samples);
        return new ProfileResult(
            samples,
            NativeCudaRuntime.TransferTelemetry - transfersBefore,
            NativeCudaRuntime.AllocationTelemetry - allocationsBefore);
    }

    private static double Percentile50(double[] sorted)
        => sorted[sorted.Length / 2];

    private readonly record struct ProfileResult(
        double[] Samples,
        NativeCudaTransferTelemetry Transfers,
        NativeCudaAllocationTelemetry Allocations);
}
