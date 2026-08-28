using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaPureBfp8GraphTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void GraphMatchesEagerAndKeepsPayloadScaleSlotsStable(
        int deviceCount)
    {
        if (Tensor.CudaDeviceCount < deviceCount)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            GptRinWikiJp eagerModel = CreateModel(2701, dropout: 0f,
                attachTrainingRandom: false);
            GptRinWikiJp graphModel = CreateModel(2701, dropout: 0f,
                attachTrainingRandom: true);
            int[] devices = Enumerable.Range(0, deviceCount).ToArray();
            int batchSize = deviceCount;
            int[] input = Enumerable.Range(1, batchSize * 4).ToArray();
            int[] target = Enumerable.Range(2, batchSize * 4).ToArray();

            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = devices;
            using var execution = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(devices),
                    Precision = PrecisionPolicy.Bfp8,
                },
                devices.Select(device =>
                    CudaExecutionLaneFactory.Create(device)).ToArray());
            using IDisposable sessionScope = execution.Enter();

            using var eagerEngine = new CudaDataParallelEngine(
                eagerModel,
                devices,
                new CudaAdaptiveShardingOptions { Enabled = false });
            eagerEngine.PrepareForTraining(batchSize);
            eagerModel.ZeroGrad();
            float eagerLoss = eagerEngine.ForwardBackward(
                input,
                target,
                batchSize,
                4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 0);
            float[][] eagerGradients = SnapshotGradients(eagerModel);

            using var graphEngine = new CudaDataParallelEngine(
                graphModel,
                devices,
                new CudaAdaptiveShardingOptions { Enabled = false });
            graphEngine.PrepareForTraining(batchSize);
            graphModel.ZeroGrad();
            NativeCudaTransferTelemetry firstTransfersBefore =
                NativeCudaRuntime.TransferTelemetry;
            float graphLoss = graphEngine.ForwardBackward(
                input,
                target,
                batchSize,
                4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 0);
            NativeCudaTransferTelemetry firstTransfers =
                NativeCudaRuntime.TransferTelemetry - firstTransfersBefore;
            float[][] graphGradients = SnapshotGradients(graphModel);
            GradientPointers[] fixedSlots = SnapshotGradientPointers(
                graphModel,
                devices);

            graphModel.ZeroGrad();
            NativeCudaAllocationTelemetry allocationsBefore =
                NativeCudaRuntime.AllocationTelemetry;
            NativeCudaTransferTelemetry hotTransfersBefore =
                NativeCudaRuntime.TransferTelemetry;
            float replayLoss = graphEngine.ForwardBackward(
                input,
                target,
                batchSize,
                4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 1);
            NativeCudaAllocationTelemetry allocations =
                NativeCudaRuntime.AllocationTelemetry - allocationsBefore;
            NativeCudaTransferTelemetry hotTransfers =
                NativeCudaRuntime.TransferTelemetry - hotTransfersBefore;
            GradientPointers[] replaySlots = SnapshotGradientPointers(
                graphModel,
                devices);
            CudaTrainingGraphTelemetry telemetry =
                graphEngine.TrainingGraphTelemetry;

            AssertClose(eagerLoss, graphLoss, 1e-5f);
            AssertClose(graphLoss, replayLoss, 1e-5f);
            AssertGradientClose(eagerGradients, graphGradients, 1e-5f);
            Assert.Equal(fixedSlots, replaySlots);
            Assert.Equal(1, telemetry.CaptureCount);
            Assert.Equal(2, telemetry.ReplayCount);
            Assert.Equal(0, telemetry.FallbackCount);
            Assert.Equal(1, telemetry.CachedCompiledPlanCount);
            Assert.True(telemetry.GraphPinnedBytes > 0);
            Assert.Equal(0, allocations.AllocationCount);
            Assert.Equal(0, allocations.FreeCount);
            Assert.Equal(2 * deviceCount,
                firstTransfers.HostToDeviceCopyCount);
            Assert.Equal(2 * deviceCount,
                hotTransfers.HostToDeviceCopyCount);
            int expectedReadbacks = deviceCount == 1 ? 3 : 5;
            long expectedReadbackBytes = deviceCount == 1
                ? 2L * sizeof(float) + sizeof(double)
                : 4L * sizeof(float) + sizeof(double);
            Assert.Equal(expectedReadbacks,
                firstTransfers.DeviceToHostCopyCount);
            Assert.Equal(expectedReadbacks,
                hotTransfers.DeviceToHostCopyCount);
            Assert.Equal(
                expectedReadbackBytes,
                firstTransfers.DeviceToHostBytes);
            Assert.Equal(
                expectedReadbackBytes,
                hotTransfers.DeviceToHostBytes);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void GraphDropoutUsesGlobalStepAndResumeIsExact()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            GptRinWikiJp model = CreateModel(
                2903,
                dropout: 0.25f,
                attachTrainingRandom: true);
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [0];
            using var execution = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(0),
                    Precision = PrecisionPolicy.Bfp8,
                },
                [CudaExecutionLaneFactory.Create(0)]);
            using IDisposable sessionScope = execution.Enter();
            using var engine = new CudaDataParallelEngine(model, [0]);
            int[] input = [1, 2, 3, 4];
            int[] target = [2, 3, 4, 5];

            model.ZeroGrad();
            float first = engine.ForwardBackward(
                input, target, 1, 4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 0);
            model.ZeroGrad();
            float second = engine.ForwardBackward(
                input, target, 1, 4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 1);
            float[][] secondGradients = SnapshotGradients(model);
            model.ZeroGrad();
            NativeCudaAllocationTelemetry allocationBefore =
                NativeCudaRuntime.AllocationTelemetry;
            float resumed = engine.ForwardBackward(
                input, target, 1, 4,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep: 1);
            NativeCudaAllocationTelemetry allocations =
                NativeCudaRuntime.AllocationTelemetry - allocationBefore;
            float[][] resumedGradients = SnapshotGradients(model);
            CudaTrainingGraphTelemetry telemetry = engine.TrainingGraphTelemetry;

            Assert.NotEqual(first, second);
            Assert.Equal(second, resumed);
            AssertGradientClose(secondGradients, resumedGradients, 0f);
            Assert.Equal(1, telemetry.CaptureCount);
            Assert.Equal(3, telemetry.ReplayCount);
            Assert.Equal(0, telemetry.FallbackCount);
            Assert.Equal(0, allocations.AllocationCount);
            Assert.Equal(0, allocations.FreeCount);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private static GptRinWikiJp CreateModel(
        int seed,
        float dropout,
        bool attachTrainingRandom)
    {
        var random = new CheckpointableRandom(seed);
        var model = new GptRinWikiJp(
            vocabularySize: 32,
            contextLength: 4,
            dModel: 8,
            numHeads: 2,
            dHidden: 16,
            numLayers: 1,
            rng: random,
            dropout: dropout,
            dtype: TensorDType.Float32,
            tieWordEmbeddings: true);
        random.BeginRuntime();
        if (attachTrainingRandom)
            model.AttachTrainingRandom(random);
        model.to(TensorPrecisionMode.Bfp8);
        return model;
    }

    private static float[][] SnapshotGradients(LanguageModel model)
        => model.Parameters()
            .Select(parameter => parameter.T.Grad.ToArray())
            .ToArray();

    private static GradientPointers[] SnapshotGradientPointers(
        LanguageModel model,
        IReadOnlyList<int> devices)
        => model.Parameters()
            .Select(parameter => parameter.T)
            .Distinct((IEqualityComparer<Tensor>)
                ReferenceEqualityComparer.Instance)
            .SelectMany(tensor => devices.Select(device =>
            {
                CudaBfp8BufferView view =
                    tensor.PrepareCudaBfp8GradientReplica(device);
                return new GradientPointers(
                    device,
                    view.Payload.NativePtr,
                    view.Scales.NativePtr,
                    checked((int)view.Payload.Length));
            }))
            .ToArray();

    private static void AssertGradientClose(
        IReadOnlyList<float[]> expected,
        IReadOnlyList<float[]> actual,
        float tolerance)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int tensor = 0; tensor < expected.Count; tensor++)
        {
            Assert.Equal(expected[tensor].Length, actual[tensor].Length);
            for (int index = 0; index < expected[tensor].Length; index++)
            {
                AssertClose(
                    expected[tensor][index],
                    actual[tensor][index],
                    tolerance);
            }
        }
    }

    private static void AssertClose(float expected, float actual, float tolerance)
        => Assert.InRange(MathF.Abs(expected - actual), 0f, tolerance);

    private readonly record struct GradientPointers(
        int Device,
        nint Payload,
        nint Scale,
        int Length);
}
