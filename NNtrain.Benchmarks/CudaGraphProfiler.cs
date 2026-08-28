using System.Diagnostics;
using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Interop;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain.Benchmarks;

internal static class CudaGraphProfiler
{
    internal static void Run(
        int warmup,
        int iterations,
        int operationsPerReplay,
        int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(warmup);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(iterations);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(operationsPerReplay);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        if (CudaNativeGateway.DeviceCount(out int deviceCount) != 0
            || deviceCount == 0)
        {
            throw new InvalidOperationException("CUDA is required.");
        }

        using CudaExecutionLane lane = CudaExecutionLaneFactory.Create(0);
        CudaMemoryLease output = ExecutionLaneResources.Attach(
            lane,
            lane.Memory.Allocate(
                checked((nuint)length * sizeof(float)),
                CudaMemoryKind.Persistent));
        CudaMemoryLease input = ExecutionLaneResources.Attach(
            lane,
            lane.Memory.Allocate(
                checked((nuint)length * sizeof(float)),
                CudaMemoryKind.Persistent));
        Check(CudaNativeGateway.MemsetAsync(
            lane.DeviceIndex,
            input.Pointer,
            0,
            checked((nuint)length * sizeof(float)),
            lane.ComputeStreamHandle));
        CudaGraphRngState rng = CudaGraphRngState.Create(lane);
        CudaGraphExecutable graph = CudaGraphExecutable.Capture(
            lane,
            () =>
            {
                rng.EnqueueAdvance();
                for (int operation = 0;
                     operation < operationsPerReplay;
                     operation++)
                {
                    rng.EnqueueDropoutForwardFloat32(
                        input.Pointer,
                        output.Pointer,
                        length,
                        dropoutProbability: 0.5f,
                        operationSeed: checked((ulong)(1701 + operation)));
                }
            });

        double[] direct = Measure(
            warmup,
            iterations,
            () =>
            {
                rng.EnqueueAdvance();
                for (int operation = 0;
                     operation < operationsPerReplay;
                     operation++)
                {
                    rng.EnqueueDropoutForwardFloat32(
                        input.Pointer,
                        output.Pointer,
                        length,
                        dropoutProbability: 0.5f,
                        operationSeed: checked((ulong)(1701 + operation)));
                }
                lane.SynchronizeComputeStream();
            });
        double[] replay = Measure(
            warmup,
            iterations,
            () =>
            {
                graph.Launch();
                lane.SynchronizeComputeStream();
            });

        double directP50 = Median(direct);
        double replayP50 = Median(replay);
        Console.WriteLine(
            $"CUDA Graph native [device=0, mask={length}, " +
            $"ops/replay={operationsPerReplay}, warmup={warmup}, " +
            $"iterations={iterations}]: direct p50={directP50:F3} ms, " +
            $"graph p50={replayP50:F3} ms, speedup=" +
            $"{directP50 / replayP50:F2}x");
    }

    private static void Check(int status)
    {
        if (status != 0)
        {
            throw new InvalidOperationException(
                $"CUDA Graph benchmark setup failed with CUDA error " +
                $"{status}: {CudaNativeGateway.ErrorString(status)}");
        }
    }

    private static double[] Measure(
        int warmup,
        int iterations,
        Action action)
    {
        var samples = new double[iterations];
        for (int run = -warmup; run < iterations; run++)
        {
            long start = Stopwatch.GetTimestamp();
            action();
            if (run >= 0)
            {
                samples[run] = Stopwatch.GetElapsedTime(start)
                    .TotalMilliseconds;
            }
        }
        return samples;
    }

    private static double Median(double[] values)
    {
        double[] ordered = values.Order().ToArray();
        return ordered[ordered.Length / 2];
    }
}
