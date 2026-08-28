using System.Diagnostics;
using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

[Collection(TensorSimdCollection.Name)]
public sealed class CudaOptimizerBarrierCoalescingTests
{
    [Fact]
    public void BatchQueuesReadbacksThenSynchronizesEachDeviceOnce()
    {
        var order = new List<string>();
        CudaOptimizerSynchronizationTelemetrySnapshot before =
            CudaOptimizerSynchronizationTelemetry.Snapshot;
        using CudaOptimizerStepBatch.Scope batch =
            CudaOptimizerStepBatch.EnterForTesting(
                [0, 1],
                (device, _) => order.Add($"sync:{device}"));

        CudaOptimizerStepBatch.CompleteAfterSynchronization(
            [0, 1],
            "neko",
            () => order.Add("read:neko"),
            () => order.Add("final:neko"));
        CudaOptimizerStepBatch.CompleteAfterSynchronization(
            [0, 1],
            "adam",
            () => order.Add("read:adam"),
            () => order.Add("final:adam"));

        Assert.Empty(order);
        batch.Complete();

        Assert.Equal(
            [
                "read:neko",
                "read:adam",
                "sync:0",
                "sync:1",
                "final:neko",
                "final:adam",
            ],
            order);
        CudaOptimizerSynchronizationTelemetrySnapshot telemetry =
            CudaOptimizerSynchronizationTelemetry.Snapshot - before;
        Assert.Equal(2, telemetry.LogicalBarrierRequests);
        Assert.Equal(4, telemetry.RequestedDeviceSynchronizations);
        Assert.Equal(2, telemetry.DeferredBarrierRequests);
        Assert.Equal(2, telemetry.PhysicalComputeStreamSynchronizations);
        Assert.Equal(1, telemetry.BatchStarts);
        Assert.Equal(1, telemetry.BatchCompletions);
    }

    [Fact]
    public void FailureDrainsAndFinalizesAlreadyQueuedChildren()
    {
        var order = new List<string>();
        var primary = new InvalidOperationException("child failed");
        CudaOptimizerSynchronizationTelemetrySnapshot before =
            CudaOptimizerSynchronizationTelemetry.Snapshot;
        using CudaOptimizerStepBatch.Scope batch =
            CudaOptimizerStepBatch.EnterForTesting(
                [0, 1],
                (device, _) => order.Add($"sync:{device}"));
        CudaOptimizerStepBatch.CompleteAfterSynchronization(
            [0, 1],
            "first child",
            () => order.Add("read"),
            () => order.Add("final"));

        Exception drained = batch.DrainAfterFailure(primary);

        Assert.Same(primary, drained);
        Assert.Equal(
            ["read", "sync:0", "sync:1", "final"],
            order);
        CudaOptimizerSynchronizationTelemetrySnapshot telemetry =
            CudaOptimizerSynchronizationTelemetry.Snapshot - before;
        Assert.Equal(1, telemetry.FailureDrains);
        Assert.Equal(2, telemetry.PhysicalComputeStreamSynchronizations);
    }

    [Fact]
    public void NestedCompositeScopeLeavesBarrierWithOuterOwner()
    {
        var synchronized = new List<int>();
        using CudaOptimizerStepBatch.Scope outer =
            CudaOptimizerStepBatch.EnterForTesting(
                [0],
                (device, _) => synchronized.Add(device));
        using CudaOptimizerStepBatch.Scope inner =
            CudaOptimizerStepBatch.Enter([0]);
        CudaOptimizerStepBatch.CompleteAfterSynchronization(
            [0], "nested", queueReadback: null, finalize: static () => { });

        inner.Complete();
        Assert.Empty(synchronized);
        outer.Complete();
        Assert.Equal([0], synchronized);
    }

    [Fact]
    public void CompositeNekoAndAdamUseOnePhysicalCudaBarrierPerStep()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(
                    PrecisionPolicy.Mix8_32);

            BenchmarkResult standalone = RunMix8(composite: false);
            BenchmarkResult coalesced = RunMix8(composite: true);

            Assert.Equal(2 * standalone.Iterations,
                standalone.Telemetry.LogicalBarrierRequests);
            Assert.Equal(2 * standalone.Iterations,
                standalone.Telemetry.PhysicalComputeStreamSynchronizations);
            Assert.Equal(0, standalone.Telemetry.DeferredBarrierRequests);
            Assert.Equal(2 * coalesced.Iterations,
                coalesced.Telemetry.LogicalBarrierRequests);
            Assert.Equal(coalesced.Iterations,
                coalesced.Telemetry.PhysicalComputeStreamSynchronizations);
            Assert.Equal(2 * coalesced.Iterations,
                coalesced.Telemetry.DeferredBarrierRequests);
            Assert.Equal(coalesced.Iterations,
                coalesced.Telemetry.BatchCompletions);

            Console.WriteLine(
                $"optimizer barrier benchmark: standalone " +
                $"p50={standalone.P50Milliseconds:F3} ms, " +
                $"mean={standalone.MeanMilliseconds:F3} ms, sync=" +
                $"{standalone.Telemetry.PhysicalComputeStreamSynchronizations}; " +
                $"composite p50={coalesced.P50Milliseconds:F3} ms, " +
                $"mean={coalesced.MeanMilliseconds:F3} ms, sync=" +
                $"{coalesced.Telemetry.PhysicalComputeStreamSynchronizations}");
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void TwoGpuCompositeSynchronizesEachComputeStreamOnce()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0, 1];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(
                    PrecisionPolicy.Mix8_32);
            (Parameter nekoParameter, Parameter adamParameter) =
                CreateParameters(TensorPrecisionMode.Mix8_32);
            _ = nekoParameter.T.EnsureCudaBfp8Buffer(1);
            _ = adamParameter.T.EnsureCudaBfp8Buffer(1);
            var neko = CreateNeko(nekoParameter);
            var adam = CreateAdam(adamParameter);
            var composite = new CompositeOptimizer(neko, adam);
            try
            {
                composite.prepare();
                foreach (Parameter parameter
                    in new[] { nekoParameter, adamParameter })
                {
                    float[] gradient = Values(
                        parameter.T.Numel,
                        parameter.T.Numel + 17,
                        0.02f);
                    parameter.T.SetCudaGradient(gradient, 0);
                    parameter.T.SetCudaGradient(gradient, 1);
                    parameter.T.MarkCudaGradientsSynchronized([0, 1]);
                }
                CudaOptimizerSynchronizationTelemetrySnapshot before =
                    CudaOptimizerSynchronizationTelemetry.Snapshot;

                composite.step();

                CudaOptimizerSynchronizationTelemetrySnapshot telemetry =
                    CudaOptimizerSynchronizationTelemetry.Snapshot - before;
                Assert.Equal(2, telemetry.LogicalBarrierRequests);
                Assert.Equal(4, telemetry.RequestedDeviceSynchronizations);
                Assert.Equal(2, telemetry.DeferredBarrierRequests);
                Assert.Equal(2,
                    telemetry.PhysicalComputeStreamSynchronizations);
                Assert.Equal(1, telemetry.BatchCompletions);
            }
            finally
            {
                neko.DisposeCudaResources();
                adam.DisposeCudaResources();
                nekoParameter.T.InvalidateCudaBuffers();
                adamParameter.T.InvalidateCudaBuffers();
            }
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void ClipScaleReturnsWithoutTerminalBarrierAndHostReadIsSafe()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        var parameter = new Parameter(
            [1f, -1f],
            [2],
            "clip.weight",
            WeightDecayPolicy.Apply);
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
            parameter.T.SetCudaGradient([3f, 4f], 0);
            parameter.T.MarkCudaGradientsSynchronized([0]);
            CudaOptimizerSynchronizationTelemetrySnapshot before =
                CudaOptimizerSynchronizationTelemetry.Snapshot;

            float norm = nn.utils.clip_grad_norm_([parameter], max_norm: 1f);

            CudaOptimizerSynchronizationTelemetrySnapshot telemetry =
                CudaOptimizerSynchronizationTelemetry.Snapshot - before;
            Assert.InRange(norm, 4.999f, 5.001f);
            Assert.Equal(1, telemetry.ClipScaleBarriersElided);
            Assert.Equal(0,
                telemetry.PhysicalComputeStreamSynchronizations);
            // GradientBuffer performs the required stream synchronization
            // before exposing host data; the asynchronous scale is complete.
            float[] gradient = parameter.T.GradientBuffer;
            Assert.InRange(gradient[0], 0.599f, 0.601f);
            Assert.InRange(gradient[1], 0.799f, 0.801f);
        }
        finally
        {
            parameter.T.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.BFloat16)]
    [InlineData(TensorPrecisionMode.Bfp8)]
    public void CompositeBarrierCoalescesForOtherPrecisionContracts(
        TensorPrecisionMode precisionMode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            PrecisionPolicy policy = precisionMode switch
            {
                TensorPrecisionMode.BFloat16 => PrecisionPolicy.BFloat16,
                TensorPrecisionMode.Bfp8 => PrecisionPolicy.Bfp8,
                _ => throw new ArgumentOutOfRangeException(
                    nameof(precisionMode)),
            };
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(policy);
            (Parameter nekoParameter, Parameter adamParameter) =
                CreateParameters(precisionMode);
            var neko = CreateNeko(nekoParameter);
            var adam = CreateAdam(adamParameter);
            var composite = new CompositeOptimizer(neko, adam);
            try
            {
                composite.prepare();
                PublishGradients(
                    precisionMode,
                    nekoParameter,
                    adamParameter,
                    iteration: 0);
                CudaOptimizerSynchronizationTelemetrySnapshot before =
                    CudaOptimizerSynchronizationTelemetry.Snapshot;

                composite.step();

                CudaOptimizerSynchronizationTelemetrySnapshot telemetry =
                    CudaOptimizerSynchronizationTelemetry.Snapshot - before;
                Assert.Equal(2, telemetry.LogicalBarrierRequests);
                Assert.Equal(2, telemetry.DeferredBarrierRequests);
                Assert.Equal(1,
                    telemetry.PhysicalComputeStreamSynchronizations);
                Assert.Equal(1, telemetry.BatchCompletions);
            }
            finally
            {
                neko.DisposeCudaResources();
                adam.DisposeCudaResources();
                nekoParameter.T.InvalidateCudaBuffers();
                adamParameter.T.InvalidateCudaBuffers();
            }
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private static BenchmarkResult RunMix8(bool composite)
    {
        const int warmup = 3;
        const int iterations = 12;
        (Parameter nekoParameter, Parameter adamParameter) =
            CreateParameters(TensorPrecisionMode.Mix8_32);
        var neko = CreateNeko(nekoParameter);
        var adam = CreateAdam(adamParameter);
        var combined = new CompositeOptimizer(neko, adam);
        try
        {
            combined.prepare();
            for (int iteration = 0; iteration < warmup; iteration++)
            {
                PublishGradients(
                    TensorPrecisionMode.Mix8_32,
                    nekoParameter,
                    adamParameter,
                    iteration);
                if (composite)
                    combined.step();
                else
                {
                    neko.step();
                    adam.step();
                }
            }

            CudaOptimizerSynchronizationTelemetrySnapshot before =
                CudaOptimizerSynchronizationTelemetry.Snapshot;
            var elapsed = new double[iterations];
            for (int iteration = 0; iteration < iterations; iteration++)
            {
                PublishGradients(
                    TensorPrecisionMode.Mix8_32,
                    nekoParameter,
                    adamParameter,
                    warmup + iteration);
                long started = Stopwatch.GetTimestamp();
                if (composite)
                    combined.step();
                else
                {
                    neko.step();
                    adam.step();
                }
                elapsed[iteration] = Stopwatch.GetElapsedTime(started)
                    .TotalMilliseconds;
            }
            CudaOptimizerSynchronizationTelemetrySnapshot telemetry =
                CudaOptimizerSynchronizationTelemetry.Snapshot - before;
            Array.Sort(elapsed);
            return new BenchmarkResult(
                iterations,
                elapsed.Average(),
                elapsed[elapsed.Length / 2],
                telemetry);
        }
        finally
        {
            neko.DisposeCudaResources();
            adam.DisposeCudaResources();
            nekoParameter.T.InvalidateCudaBuffers();
            adamParameter.T.InvalidateCudaBuffers();
        }
    }

    private static (Parameter Neko, Parameter Adam) CreateParameters(
        TensorPrecisionMode precisionMode)
    {
        var neko = new Parameter(
            Values(48 * 64, 11, 0.08f),
            [48, 64],
            "hidden.weight",
            WeightDecayPolicy.Apply);
        var adam = new Parameter(
            Values(256, 29, 0.05f),
            [256],
            "norm.weight",
            WeightDecayPolicy.Apply);
        foreach (Parameter parameter in new[] { neko, adam })
        {
            switch (precisionMode)
            {
                case TensorPrecisionMode.BFloat16:
                    parameter.T.ConvertStorageInPlace(
                        TensorDType.BFloat16,
                        preserveFloat32Master: false);
                    _ = parameter.T.EnsureCudaBFloat16Buffer(0);
                    break;
                case TensorPrecisionMode.Bfp8:
                    parameter.T.ConvertStorageInPlace(
                        TensorDType.Bfp8,
                        Bfp8QuantizationDescriptor.TensorWide,
                        preserveFloat32Master: false);
                    _ = parameter.T.EnsureCudaBfp8Buffer(0);
                    break;
                case TensorPrecisionMode.Mix8_32:
                    parameter.T.ConvertStorageInPlace(
                        TensorDType.Bfp8,
                        Bfp8QuantizationDescriptor.Block(128),
                        preserveFloat32Master: true);
                    _ = parameter.T.EnsureCudaBfp8Buffer(0);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(
                        nameof(precisionMode));
            }
            parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
        }
        return (neko, adam);
    }

    private static NekoMuon CreateNeko(Parameter parameter)
        => new(
            [parameter],
            new NekoMuonOptions
            {
                LearningRate = 3e-4f,
                WeightDecay = 0.01f,
                MaxNewtonSchulzSteps = 5,
                NewtonSchulzInterval = 1,
                NewtonSchulzDepthMode =
                    NekoMuonNewtonSchulzDepthMode.Fixed,
                NewtonSchulzDepth = 5f,
            });

    private static AdamW CreateAdam(Parameter parameter)
        => new(
            [parameter],
            new AdamWOptions
            {
                LearningRate = 3e-4f,
                Beta1 = 0.9f,
                Beta2 = 0.95f,
                Epsilon = 1e-8f,
                WeightDecay = 0.01f,
            });

    private static void PublishGradients(
        TensorPrecisionMode precisionMode,
        Parameter neko,
        Parameter adam,
        int iteration)
    {
        SetGradient(neko, Values(
            neko.T.Numel,
            37 + iteration,
            0.025f), precisionMode);
        SetGradient(adam, Values(
            adam.T.Numel,
            71 + iteration,
            0.018f), precisionMode);
    }

    private static void SetGradient(
        Parameter parameter,
        float[] gradient,
        TensorPrecisionMode precisionMode)
    {
        if (precisionMode == TensorPrecisionMode.BFloat16)
        {
            // Exercise the same pure-BF16 publication path as autograd.
            // SetCudaGradient intentionally publishes an FP32 accumulator,
            // which the pure-BF16 optimizer must reject rather than silently
            // converting at the optimizer boundary.
            parameter.T.BackwardAndRelease(gradient);
            parameter.T.MarkCudaBFloat16GradientsSynchronized([0]);
            return;
        }
        parameter.T.SetCudaGradient(gradient, 0);
        if (precisionMode == TensorPrecisionMode.Bfp8)
            parameter.T.PublishCudaBfp8Gradient(0);
        else
            parameter.T.MarkCudaGradientsSynchronized([0]);
    }

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index => MathF.Sin((index + offset) * 0.173f) * scale)
            .ToArray();

    private sealed record BenchmarkResult(
        int Iterations,
        double MeanMilliseconds,
        double P50Milliseconds,
        CudaOptimizerSynchronizationTelemetrySnapshot Telemetry);
}
