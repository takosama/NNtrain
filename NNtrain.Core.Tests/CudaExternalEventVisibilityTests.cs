using System.Diagnostics;
using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Interop;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaExternalEventVisibilityTests
{
    [Fact]
    public void CapturedExternalEventBecomesHostVisibleBeforeGraphTail()
    {
        if (Tensor.CudaDeviceCount == 0
            || CudaDispatchPolicy.Current.DisableExternalGradientReadyEvents
            || CudaNativeGateway.AbiVersion.Minor
                < CudaAbiVersion.ExternalGradientReadyEventMinor)
        {
            return;
        }

        const int valueCount = 8 * 1024 * 1024;
        const int prefixPasses = 2;
        const int suffixPasses = 384;
        CudaExecutionLane lane = CudaExecutionLaneFactory.Create(0);
        using var execution = new ExecutionSession(
            new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(0),
                Precision = PrecisionPolicy.Mix16_32,
            },
            [lane]);
        using IDisposable executionScope = execution.Enter();
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(0);
        using NativeCudaBuffer<float> values =
            accelerator.Allocate1D<float>(valueCount);
        using NativeCudaBuffer<double> squaredSum =
            accelerator.Allocate1D<double>(1);
        values.MemSetToZero();
        squaredSum.MemSetToZero();
        lane.SynchronizeComputeStream();
        nint midpoint = CudaGradientBuckets.CreateReadyEvent(accelerator, 0);
        try
        {
            using CudaGraphExecutable graph = CudaGraphExecutable.Capture(
                lane,
                () =>
                {
                    SubmitSquaredSumPasses(
                        values, squaredSum, prefixPasses);
                    CudaGradientBuckets.RecordReadyExternal(
                        0, accelerator, midpoint);
                    SubmitSquaredSumPasses(
                        values, squaredSum, suffixPasses);
                });

            long started = Stopwatch.GetTimestamp();
            graph.Launch();
            Assert.Equal(
                0,
                NativeCudaRuntime.EventSynchronizeNative(0, midpoint));
            double midpointMilliseconds = Stopwatch.GetElapsedTime(started)
                .TotalMilliseconds;
            lane.SynchronizeComputeStream();
            double totalMilliseconds = Stopwatch.GetElapsedTime(started)
                .TotalMilliseconds;
            double tailMilliseconds = totalMilliseconds - midpointMilliseconds;

            Assert.True(
                tailMilliseconds >= 5d,
                $"External event was not host-visible before the graph tail: "
                + $"midpoint={midpointMilliseconds:F3} ms, "
                + $"total={totalMilliseconds:F3} ms, "
                + $"tail={tailMilliseconds:F3} ms.");
            Assert.True(
                midpointMilliseconds < totalMilliseconds * 0.5d,
                $"External event surfaced too close to the graph tail: "
                + $"midpoint={midpointMilliseconds:F3} ms, "
                + $"total={totalMilliseconds:F3} ms.");
        }
        finally
        {
            CudaGradientBuckets.DestroyEvent(accelerator, 0, midpoint);
        }
    }

    private static void SubmitSquaredSumPasses(
        NativeCudaBuffer<float> values,
        NativeCudaBuffer<double> squaredSum,
        int count)
    {
        for (int pass = 0; pass < count; pass++)
        {
            CudaTensorNative.SquaredSum(
                0,
                values.NativePtr,
                values.Length,
                squaredSum.NativePtr);
        }
    }
}
