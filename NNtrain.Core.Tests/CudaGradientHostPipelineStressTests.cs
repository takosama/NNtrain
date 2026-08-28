using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaGradientHostPipelineStressTests
{
    [Fact]
    public void CapturedAsyncHostPipelineSurvivesOneHundredStepLifecycles()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithTwoGpuSession(lanes =>
        {
            int workerBaseline =
                CudaBFloat16GradientAllReducePlan.ActiveHostWorkerCount;
            int completed = 0;
            int injectedFailures = 0;
            int earlyAborts = 0;
            int scheduledAborts = 0;

            for (int group = 0; group < 4; group++)
            {
                Parameter[] parameters = CreateParameters(group);
                long faultStep = 0;
                CudaBFloat16GradientAllReducePlan? reducer = null;
                CudaGraphExecutable[] graphs = [];
                try
                {
                    reducer = new CudaBFloat16GradientAllReducePlan(
                        parameters,
                        [0, 1],
                        new CudaDispatchPolicy
                        {
                            GradientBucketElements = 128,
                            GradientHostChunkElements = 64,
                        },
                        hostReductionFaultInjector:
                            (step, bucket, destination) =>
                                step == Volatile.Read(ref faultStep)
                                    && bucket == 0
                                    && destination == 0
                                        ? new TestHostReductionException(step)
                                        : null);

                    // This stress specifically covers the non-peer host path.
                    // Peer-capable test hosts exercise the existing P2P path in
                    // the regular captured reducer suite instead.
                    if (!reducer.UsesAsyncHostPipeline)
                        return;

                    Assert.True(reducer.UsesHostPipeline);
                    Assert.True(reducer.BucketCount >= 2);
                    Assert.True(SpinWait.SpinUntil(
                        () => CudaBFloat16GradientAllReducePlan
                            .ActiveHostWorkerCount == workerBaseline + 2,
                        TimeSpan.FromSeconds(5)));

                    float[][][] gradients = CreateGradients(parameters, group);
                    graphs = Capture(reducer, lanes, parameters, gradients);

                    for (int local = 1; local <= 25; local++)
                    {
                        int iteration = group * 25 + local;
                        long stepId = Prepare(
                            reducer, lanes, parameters, gradients);

                        if (iteration % 10 == 1)
                        {
                            lanes[0].ActivateComputeStream();
                            graphs[0].Launch();
                            reducer.PublishCapturedDeviceGradientsForReplay(
                                stepId, 0);
                            lanes[0].SynchronizeComputeStream();
                            reducer.Abort(stepId);
                            earlyAborts++;
                            continue;
                        }

                        if (iteration % 10 == 0)
                            Volatile.Write(ref faultStep, stepId);
                        else
                            Volatile.Write(ref faultStep, 0);

                        PublishBoth(reducer, lanes, graphs, stepId);
                        if (iteration % 10 == 2)
                        {
                            reducer.Abort(stepId);
                            scheduledAborts++;
                        }
                        else if (iteration % 10 == 0)
                        {
                            InvalidOperationException failure = Assert.Throws<
                                InvalidOperationException>(() =>
                                    reducer.Complete(stepId));
                            Assert.Contains(
                                "host reduction",
                                failure.Message,
                                StringComparison.OrdinalIgnoreCase);
                            injectedFailures++;
                        }
                        else
                        {
                            reducer.Complete(stepId);
                            completed++;
                            Assert.Equal(
                                reducer.TransportBytesPerStep,
                                reducer.LastCompletedTransportBytes);
                            AssertResidentNonZeroReplicas(parameters);
                        }
                    }

                    // Graphs no longer reference the reducer's events before
                    // the active-step Dispose case begins.
                    DisposeGraphs(graphs);
                    graphs = [];

                    long disposeStep = Prepare(
                        reducer, lanes, parameters, gradients);
                    Volatile.Write(
                        ref faultStep,
                        group == 3 ? disposeStep : 0);
                    for (int device = 0; device < 2; device++)
                    {
                        lanes[device].ActivateComputeStream();
                        foreach (Parameter parameter in parameters)
                        {
                            reducer.NotifyGradientReady(
                                parameter.T, device, disposeStep);
                        }
                    }
                    if (group == 3)
                    {
                        AggregateException disposeFailure = Assert.Throws<
                            AggregateException>(reducer.Dispose);
                        Assert.Contains(
                            disposeFailure.Flatten().InnerExceptions,
                            exception => exception.Message.Contains(
                                "host reduction",
                                StringComparison.OrdinalIgnoreCase));
                    }
                    else
                    {
                        reducer.Dispose();
                    }
                    reducer = null;
                }
                finally
                {
                    DisposeGraphs(graphs);
                    reducer?.Dispose();
                    foreach (Parameter parameter in parameters)
                        parameter.T.InvalidateCudaBuffers();
                }

                Assert.True(SpinWait.SpinUntil(
                    () => CudaBFloat16GradientAllReducePlan
                        .ActiveHostWorkerCount == workerBaseline,
                    TimeSpan.FromSeconds(5)));
            }

            Assert.Equal(70, completed);
            Assert.Equal(10, injectedFailures);
            Assert.Equal(10, earlyAborts);
            Assert.Equal(10, scheduledAborts);
            Assert.Equal(
                workerBaseline,
                CudaBFloat16GradientAllReducePlan.ActiveHostWorkerCount);
        });
    }

    private static CudaGraphExecutable[] Capture(
        CudaBFloat16GradientAllReducePlan reducer,
        CudaExecutionLane[] lanes,
        Parameter[] parameters,
        float[][][] gradients)
    {
        long stepId = Prepare(reducer, lanes, parameters, gradients);
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

    private static long Prepare(
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

    private static void PublishBoth(
        CudaBFloat16GradientAllReducePlan reducer,
        CudaExecutionLane[] lanes,
        CudaGraphExecutable[] graphs,
        long stepId)
    {
        for (int device = 0; device < 2; device++)
        {
            lanes[device].ActivateComputeStream();
            graphs[device].Launch();
            reducer.PublishCapturedDeviceGradientsForReplay(stepId, device);
        }
    }

    private static Parameter[] CreateParameters(int group)
        =>
        [
            CreateParameter($"host.stress.{group}.left", 257),
            CreateParameter($"host.stress.{group}.right", 129),
        ];

    private static Parameter CreateParameter(string name, int length)
    {
        var parameter = new Parameter(
            Values(length, 7, 0.03f),
            [length],
            name,
            WeightDecayPolicy.Apply,
            TensorDType.Float32);
        parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
        _ = parameter.T.EnsureCudaFloat32Buffer(1);
        return parameter;
    }

    private static float[][][] CreateGradients(
        Parameter[] parameters,
        int group)
        => parameters.Select((parameter, index) => new[]
        {
            Values(parameter.T.Numel, 17 + group * 5 + index * 13, 0.11f),
            Values(parameter.T.Numel, 43 + group * 7 + index * 19, 0.07f),
        }).ToArray();

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index => index == length - 1
                ? scale * 3.25f
                : MathF.Sin((index + offset) * 0.173f) * scale)
            .ToArray();

    private static void AssertResidentNonZeroReplicas(
        IEnumerable<Parameter> parameters)
    {
        foreach (Parameter parameter in parameters)
        {
            float[] first = Read(parameter.T.EnsureCudaGradientBuffer(0));
            float[] second = Read(parameter.T.EnsureCudaGradientBuffer(1));
            Assert.Equal(first, second);
            Assert.Contains(first, value => value != 0f);
        }
    }

    private static float[] Read(NativeCudaBuffer<float> buffer)
    {
        var result = new float[buffer.Length];
        buffer.CopyToCPU(result);
        return result;
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

    private static void WithTwoGpuSession(Action<CudaExecutionLane[]> action)
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
                    Precision = PrecisionPolicy.Mix16_32,
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

    private sealed class TestHostReductionException(long stepId)
        : Exception($"Injected host reduction failure for step {stepId}.");
}
