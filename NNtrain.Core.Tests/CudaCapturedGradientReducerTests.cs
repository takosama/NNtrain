using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaCapturedGradientReducerTests
{
    private const float ClipNorm = 0.75f;

    [Theory]
    [InlineData(TensorPrecisionMode.BFloat16)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void Bf16TransportCapturedReplayMatchesNormalAndDoesNotRepack(
        TensorPrecisionMode mode)
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        PrecisionPolicy precision = mode == TensorPrecisionMode.BFloat16
            ? PrecisionPolicy.BFloat16
            : PrecisionPolicy.Mix8_32;
        WithTwoGpuSession(precision, (lanes) =>
        {
            Parameter[] parameters =
            [
                CreateBf16TransportParameter(mode, "captured.left", 257),
                CreateBf16TransportParameter(mode, "captured.right", 129),
            ];
            var reducer = new CudaBFloat16GradientAllReducePlan(
                parameters,
                [0, 1],
                new CudaDispatchPolicy { GradientBucketElements = 200 });
            CudaGraphExecutable[] graphs = [];
            try
            {
                float[][][] gradients = CreateGradients(parameters);
                graphs = CaptureBf16LocalPublication(
                    reducer, lanes, parameters, gradients);
                Assert.Equal(0, reducer.CompletedSteps);
                Assert.Equal(0, reducer.LastCompletedTransportBytes);
                long recordedPacks = reducer.ManagedLocalPackSubmissionCount;
                Assert.True(recordedPacks >= 4);

                FloatRun normal = RunBf16(
                    reducer,
                    lanes,
                    parameters,
                    gradients,
                    graphs: null);
                Assert.True(
                    reducer.ManagedLocalPackSubmissionCount > recordedPacks);

                long beforeReplay = reducer.ManagedLocalPackSubmissionCount;
                FloatRun captured = RunBf16(
                    reducer,
                    lanes,
                    parameters,
                    gradients,
                    graphs);
                Assert.Equal(
                    beforeReplay,
                    reducer.ManagedLocalPackSubmissionCount);

                AssertFloatRunsEqual(normal, captured, sizeof(double));
                Assert.Equal(
                    reducer.TransportBytesPerStep,
                    reducer.LastCompletedTransportBytes);

                // A duplicate device publication poisons only this generation;
                // Abort ends it and the already-instantiated graphs remain
                // reusable for the following step.
                long failedStep = PrepareBf16Step(
                    reducer, lanes, parameters, gradients);
                graphs[0].Launch();
                reducer.PublishCapturedDeviceGradientsForReplay(
                    failedStep, 0);
                Assert.Throws<InvalidOperationException>(() =>
                    reducer.PublishCapturedDeviceGradientsForReplay(
                        failedStep, 0));
                lanes[0].SynchronizeComputeStream();
                reducer.Abort(failedStep);

                long beforeRecovery =
                    reducer.ManagedLocalPackSubmissionCount;
                FloatRun recovered = RunBf16(
                    reducer,
                    lanes,
                    parameters,
                    gradients,
                    graphs);
                Assert.Equal(
                    beforeRecovery,
                    reducer.ManagedLocalPackSubmissionCount);
                AssertFloatRunsEqual(captured, recovered, sizeof(double));
                Assert.Equal(3, reducer.CompletedSteps);
            }
            finally
            {
                DisposeGraphs(graphs);
                reducer.Dispose();
                Invalidate(parameters);
            }
        });
    }

    [Fact]
    public void PureBfp8CapturedReplayMatchesNormalAndDoesNotRequantize()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithTwoGpuSession(PrecisionPolicy.Bfp8, (lanes) =>
        {
            Parameter[] parameters =
            [
                CreateBfp8Parameter("captured.bfp8.left", 257),
                CreateBfp8Parameter("captured.bfp8.right", 129),
            ];
            var reducer = new CudaBfp8GradientAllReducePlan(
                parameters, [0, 1]);
            CudaGraphExecutable[] graphs = [];
            try
            {
                float[][][] gradients = CreateGradients(parameters);
                graphs = CaptureBfp8LocalPublication(
                    reducer, lanes, parameters, gradients);
                Assert.Equal(0, reducer.CompletedSteps);
                Assert.Equal(0, reducer.LastCompletedTransportBytes);
                long recordedQuantizations =
                    reducer.ManagedLocalQuantizationSubmissionCount;
                Assert.Equal(4, recordedQuantizations);

                Bfp8Run normal = RunBfp8(
                    reducer,
                    lanes,
                    parameters,
                    gradients,
                    graphs: null);
                Assert.True(
                    reducer.ManagedLocalQuantizationSubmissionCount
                        > recordedQuantizations);

                long beforeReplay =
                    reducer.ManagedLocalQuantizationSubmissionCount;
                Bfp8Run captured = RunBfp8(
                    reducer,
                    lanes,
                    parameters,
                    gradients,
                    graphs);
                Assert.Equal(
                    beforeReplay,
                    reducer.ManagedLocalQuantizationSubmissionCount);

                AssertBfp8RunsEqual(normal, captured);
                Assert.Equal(
                    reducer.TransportBytesPerStep,
                    reducer.LastCompletedTransportBytes);

                long failedStep = PrepareBfp8Step(
                    reducer, lanes, parameters, gradients);
                graphs[0].Launch();
                reducer.PublishCapturedDeviceGradientsAfterReplay(
                    failedStep, 0);
                Assert.Throws<InvalidOperationException>(() =>
                    reducer.PublishCapturedDeviceGradientsAfterReplay(
                        failedStep, 0));
                lanes[0].SynchronizeComputeStream();
                reducer.Abort(failedStep);

                long beforeRecovery =
                    reducer.ManagedLocalQuantizationSubmissionCount;
                Bfp8Run recovered = RunBfp8(
                    reducer,
                    lanes,
                    parameters,
                    gradients,
                    graphs);
                Assert.Equal(
                    beforeRecovery,
                    reducer.ManagedLocalQuantizationSubmissionCount);
                AssertBfp8RunsEqual(captured, recovered);
                Assert.Equal(3, reducer.CompletedSteps);
            }
            finally
            {
                DisposeGraphs(graphs);
                reducer.Dispose();
                Invalidate(parameters);
            }
        });
    }

    private static CudaGraphExecutable[] CaptureBf16LocalPublication(
        CudaBFloat16GradientAllReducePlan reducer,
        CudaExecutionLane[] lanes,
        Parameter[] parameters,
        float[][][] gradients)
    {
        long stepId = PrepareBf16Step(
            reducer, lanes, parameters, gradients);
        var graphs = new CudaGraphExecutable[2];
        try
        {
            for (int device = 0; device < 2; device++)
            {
                int capturedDevice = device;
                using IDisposable recording =
                    reducer.BeginCapturedBackwardRecording(
                        stepId,
                        capturedDevice,
                        CudaCapturedBackwardRecordingMode.StreamCapture);
                graphs[capturedDevice] = CudaGraphExecutable.Capture(
                    lanes[capturedDevice],
                    () =>
                    {
                        foreach (Parameter parameter in parameters)
                        {
                            reducer.NotifyGradientReady(
                                parameter.T, capturedDevice, stepId);
                        }
                    });
            }
            reducer.DiscardCapturedBackwardRecordingStep(stepId);
            return graphs;
        }
        catch
        {
            DisposeGraphs(graphs);
            reducer.Abort(stepId);
            throw;
        }
    }

    private static CudaGraphExecutable[] CaptureBfp8LocalPublication(
        CudaBfp8GradientAllReducePlan reducer,
        CudaExecutionLane[] lanes,
        Parameter[] parameters,
        float[][][] gradients)
    {
        long stepId = PrepareBfp8Step(
            reducer, lanes, parameters, gradients);
        var graphs = new CudaGraphExecutable[2];
        try
        {
            for (int device = 0; device < 2; device++)
            {
                int capturedDevice = device;
                using IDisposable recording =
                    reducer.BeginCapturedBackwardRecording(
                        stepId, capturedDevice);
                graphs[capturedDevice] = CudaGraphExecutable.Capture(
                    lanes[capturedDevice],
                    () =>
                    {
                        foreach (Parameter parameter in parameters)
                        {
                            reducer.NotifyGradientReady(
                                parameter.T, capturedDevice, stepId);
                        }
                    });
            }
            reducer.DiscardCapturedBackwardRecordingStep(stepId);
            return graphs;
        }
        catch
        {
            DisposeGraphs(graphs);
            reducer.Abort(stepId);
            throw;
        }
    }

    private static FloatRun RunBf16(
        CudaBFloat16GradientAllReducePlan reducer,
        CudaExecutionLane[] lanes,
        Parameter[] parameters,
        float[][][] gradients,
        CudaGraphExecutable[]? graphs)
    {
        long stepId = PrepareBf16Step(
            reducer, lanes, parameters, gradients);
        NativeCudaTransferTelemetry before =
            NativeCudaRuntime.TransferTelemetry;
        NativeCudaTransferTelemetry gradientBefore =
            NativeCudaRuntime.GradientCollectiveTransferTelemetry;
        DeviceTransferSnapshot guarded;
        float norm;
        using (DeviceTransferGuard.EnterTrainingStep(2))
        {
            for (int device = 0; device < 2; device++)
            {
                lanes[device].ActivateComputeStream();
                if (graphs is null)
                {
                    foreach (Parameter parameter in parameters)
                    {
                        reducer.NotifyGradientReady(
                            parameter.T, device, stepId);
                    }
                }
                else
                {
                    graphs[device].Launch();
                    reducer.PublishCapturedDeviceGradientsForReplay(
                        stepId, device);
                }
            }
            reducer.Complete(stepId);
            if (reducer.UsesAsyncHostPipeline)
            {
                CudaGradientOverlapTelemetry timeline = Assert.IsType<
                    CudaGradientOverlapTelemetry>(
                        reducer.LastOverlapTelemetry);
                Assert.Equal(stepId, timeline.StepId);
                Assert.Equal(
                    reducer.BucketCount * 2,
                    timeline.ScheduledHostWorkCount);
                Assert.Equal(
                    timeline.ScheduledHostWorkCount,
                    timeline.CompletedHostWorkCount);
                Assert.Equal(0, timeline.FailedHostWorkCount);
                Assert.True(
                    timeline.CompleteFinishedMilliseconds
                        >= timeline.CompleteEnteredMilliseconds);
                bool expectedExternal = graphs is not null
                    && reducer.UsesExternalCapturedReadyEvents;
                Assert.True(
                    timeline.UsedExternalCapturedReadyEvents
                        == expectedExternal,
                    $"graphs={(graphs is null ? "eager" : "replay")}, " +
                    $"capability={reducer.UsesExternalCapturedReadyEvents}, " +
                    $"timeline={timeline.UsedExternalCapturedReadyEvents}");
            }
            norm = nn.utils.clip_grad_norm_(parameters, ClipNorm);
            guarded = Assert.IsType<DeviceTransferSnapshot>(
                DeviceTransferGuard.CurrentSnapshot);
        }
        NativeCudaTransferTelemetry transfers =
            NativeCudaRuntime.TransferTelemetry - before;
        NativeCudaTransferTelemetry gradientTransfers =
            NativeCudaRuntime.GradientCollectiveTransferTelemetry
                - gradientBefore;
        return new FloatRun(
            norm,
            guarded,
            transfers,
            gradientTransfers,
            ReadFloatReplicas(parameters));
    }

    private static Bfp8Run RunBfp8(
        CudaBfp8GradientAllReducePlan reducer,
        CudaExecutionLane[] lanes,
        Parameter[] parameters,
        float[][][] gradients,
        CudaGraphExecutable[]? graphs)
    {
        long stepId = PrepareBfp8Step(
            reducer, lanes, parameters, gradients);
        NativeCudaTransferTelemetry before =
            NativeCudaRuntime.TransferTelemetry;
        DeviceTransferSnapshot guarded;
        float norm;
        using (DeviceTransferGuard.EnterTrainingStep(2))
        {
            for (int device = 0; device < 2; device++)
            {
                lanes[device].ActivateComputeStream();
                if (graphs is null)
                {
                    foreach (Parameter parameter in parameters)
                    {
                        reducer.NotifyGradientReady(
                            parameter.T, device, stepId);
                    }
                }
                else
                {
                    graphs[device].Launch();
                    reducer.PublishCapturedDeviceGradientsAfterReplay(
                        stepId, device);
                }
            }
            reducer.Complete(stepId);
            norm = nn.utils.clip_grad_norm_(parameters, ClipNorm);
            guarded = Assert.IsType<DeviceTransferSnapshot>(
                DeviceTransferGuard.CurrentSnapshot);
        }
        NativeCudaTransferTelemetry transfers =
            NativeCudaRuntime.TransferTelemetry - before;
        return new Bfp8Run(
            norm,
            guarded,
            transfers,
            ReadBfp8Replicas(parameters));
    }

    private static long PrepareBf16Step(
        CudaBFloat16GradientAllReducePlan reducer,
        CudaExecutionLane[] lanes,
        Parameter[] parameters,
        float[][][] gradients)
    {
        long stepId = reducer.BeginStep();
        for (int device = 0; device < 2; device++)
        {
            lanes[device].ActivateComputeStream();
            reducer.BeginDeviceStep(stepId, device);
            for (int parameter = 0; parameter < parameters.Length; parameter++)
            {
                parameters[parameter].T.SetCudaGradient(
                    gradients[parameter][device], device);
            }
        }
        return stepId;
    }

    private static long PrepareBfp8Step(
        CudaBfp8GradientAllReducePlan reducer,
        CudaExecutionLane[] lanes,
        Parameter[] parameters,
        float[][][] gradients)
    {
        long stepId = reducer.BeginStep();
        for (int device = 0; device < 2; device++)
        {
            lanes[device].ActivateComputeStream();
            reducer.BeginDeviceStep(stepId, device);
            for (int parameter = 0; parameter < parameters.Length; parameter++)
            {
                parameters[parameter].T.SetCudaGradient(
                    gradients[parameter][device], device);
            }
        }
        return stepId;
    }

    private static Parameter CreateBf16TransportParameter(
        TensorPrecisionMode mode,
        string name,
        int length)
    {
        var parameter = new Parameter(
            Values(length, 7, 0.03f),
            [length],
            name,
            WeightDecayPolicy.Apply,
            mode == TensorPrecisionMode.BFloat16
                ? TensorDType.BFloat16
                : TensorDType.Float32);
        if (mode == TensorPrecisionMode.Mix8_32)
        {
            parameter.T.ConvertStorageInPlace(
                TensorDType.Bfp8,
                Bfp8QuantizationDescriptor.Block(128),
                preserveFloat32Master: true);
        }
        parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
        if (mode == TensorPrecisionMode.BFloat16)
            _ = parameter.T.EnsureCudaBFloat16Buffer(1);
        else
            _ = parameter.T.EnsureCudaBfp8Buffer(1);
        return parameter;
    }

    private static Parameter CreateBfp8Parameter(string name, int length)
    {
        var parameter = new Parameter(
            Values(length, 11, 0.025f),
            [length],
            name,
            WeightDecayPolicy.Apply);
        parameter.T.ConvertStorageInPlace(
            TensorDType.Bfp8,
            Bfp8QuantizationDescriptor.TensorWide,
            preserveFloat32Master: false);
        parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
        _ = parameter.T.EnsureCudaBfp8Buffer(1);
        return parameter;
    }

    private static float[][][] CreateGradients(Parameter[] parameters)
        => parameters.Select((parameter, index) => new[]
        {
            Values(parameter.T.Numel, 17 + index * 13, 0.11f),
            Values(parameter.T.Numel, 43 + index * 19, 0.07f),
        }).ToArray();

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index => index == length - 1
                ? scale * 3.25f
                : MathF.Sin((index + offset) * 0.173f) * scale)
            .ToArray();

    private static float[][][] ReadFloatReplicas(Parameter[] parameters)
        => parameters.Select(parameter => new[]
        {
            Read(parameter.T.EnsureCudaGradientBuffer(0)),
            Read(parameter.T.EnsureCudaGradientBuffer(1)),
        }).ToArray();

    private static Bfp8Replica[][] ReadBfp8Replicas(Parameter[] parameters)
        => parameters.Select(parameter => new[]
        {
            ReadBfp8(parameter.T, 0),
            ReadBfp8(parameter.T, 1),
        }).ToArray();

    private static float[] Read(NativeCudaBuffer<float> buffer)
    {
        var result = new float[buffer.Length];
        buffer.CopyToCPU(result);
        return result;
    }

    private static Bfp8Replica ReadBfp8(Tensor tensor, int device)
    {
        Assert.True(tensor.TryGetCudaBfp8GradientBuffer(
            device, out CudaBfp8BufferView view));
        var payload = new sbyte[tensor.Numel];
        var scales = new float[view.Scales.Length];
        view.Payload.CopyToCPU(payload);
        view.Scales.CopyToCPU(scales);
        return new Bfp8Replica(payload, scales);
    }

    private static void AssertFloatRunsEqual(
        FloatRun expected,
        FloatRun actual,
        long expectedD2HBytes)
    {
        Assert.InRange(MathF.Abs(expected.Norm - actual.Norm), 0f, 1e-6f);
        AssertTransfer(
            expected.Guard,
            expected.Transfers,
            expected.GradientTransfers,
            expectedD2HBytes);
        AssertTransfer(
            actual.Guard,
            actual.Transfers,
            actual.GradientTransfers,
            expectedD2HBytes);
        Assert.Equal(expected.Replicas.Length, actual.Replicas.Length);
        for (int parameter = 0;
            parameter < expected.Replicas.Length;
            parameter++)
        {
            Assert.Equal(
                expected.Replicas[parameter][0],
                expected.Replicas[parameter][1]);
            for (int device = 0; device < 2; device++)
            {
                Assert.Equal(
                    expected.Replicas[parameter][device],
                    actual.Replicas[parameter][device]);
            }
        }
    }

    private static void AssertBfp8RunsEqual(Bfp8Run expected, Bfp8Run actual)
    {
        Assert.InRange(MathF.Abs(expected.Norm - actual.Norm), 0f, 1e-6f);
        AssertTransfer(expected.Guard, expected.Transfers, 16);
        AssertTransfer(actual.Guard, actual.Transfers, 16);
        Assert.Equal(expected.Replicas.Length, actual.Replicas.Length);
        for (int parameter = 0;
            parameter < expected.Replicas.Length;
            parameter++)
        {
            Assert.Equal(
                expected.Replicas[parameter][0],
                expected.Replicas[parameter][1]);
            for (int device = 0; device < 2; device++)
            {
                Assert.Equal(
                    expected.Replicas[parameter][device],
                    actual.Replicas[parameter][device]);
            }
        }
    }

    private static void AssertTransfer(
        DeviceTransferSnapshot guard,
        NativeCudaTransferTelemetry telemetry,
        long expectedD2HBytes)
    {
        Assert.Equal(0, guard.HostToDeviceCopyCount);
        Assert.Equal(0, guard.HostToDeviceBytes);
        Assert.Equal(expectedD2HBytes, guard.DeviceToHostBytes);
        Assert.Equal(0, telemetry.HostToDeviceBytes);
        Assert.Equal(expectedD2HBytes, telemetry.DeviceToHostBytes);
    }

    private static void AssertTransfer(
        DeviceTransferSnapshot guard,
        NativeCudaTransferTelemetry telemetry,
        NativeCudaTransferTelemetry gradientTransfers,
        long expectedD2HBytes)
    {
        Assert.Equal(0, guard.HostToDeviceCopyCount);
        Assert.Equal(0, guard.HostToDeviceBytes);
        Assert.Equal(expectedD2HBytes, guard.DeviceToHostBytes);
        Assert.True(gradientTransfers.HostToDeviceBytes > 0);
        Assert.Equal(
            gradientTransfers.HostToDeviceBytes,
            telemetry.HostToDeviceBytes);
        Assert.Equal(
            expectedD2HBytes + gradientTransfers.DeviceToHostBytes,
            telemetry.DeviceToHostBytes);
    }

    private static void WithTwoGpuSession(
        PrecisionPolicy precision,
        Action<CudaExecutionLane[]> action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0, 1];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            CudaExecutionLane[] lanes =
            [
                CudaExecutionLaneFactory.Create(0),
                CudaExecutionLaneFactory.Create(1),
            ];
            using var session = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(0, 1),
                    Precision = precision,
                },
                lanes);
            using IDisposable execution = session.Enter();
            action(lanes);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    private static void DisposeGraphs(IEnumerable<CudaGraphExecutable?> graphs)
    {
        List<Exception>? failures = null;
        foreach (CudaGraphExecutable? graph in graphs)
        {
            if (graph is null)
                continue;
            try
            {
                graph.DisposeChecked();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        if (failures is not null)
            throw new AggregateException("CUDA graph cleanup failed.", failures);
    }

    private static void Invalidate(IEnumerable<Parameter> parameters)
    {
        foreach (Parameter parameter in parameters)
            parameter.T.InvalidateCudaBuffers();
    }

    private sealed record FloatRun(
        float Norm,
        DeviceTransferSnapshot Guard,
        NativeCudaTransferTelemetry Transfers,
        NativeCudaTransferTelemetry GradientTransfers,
        float[][][] Replicas);

    private sealed record Bfp8Run(
        float Norm,
        DeviceTransferSnapshot Guard,
        NativeCudaTransferTelemetry Transfers,
        Bfp8Replica[][] Replicas);

    private sealed class Bfp8Replica(
        sbyte[] payload,
        float[] scales) : IEquatable<Bfp8Replica>
    {
        internal sbyte[] Payload { get; } = payload;
        internal float[] Scales { get; } = scales;

        public bool Equals(Bfp8Replica? other)
            => other is not null
                && Payload.SequenceEqual(other.Payload)
                && Scales.SequenceEqual(other.Scales);

        public override bool Equals(object? obj)
            => obj is Bfp8Replica other && Equals(other);

        public override int GetHashCode() => 0;
    }
}
