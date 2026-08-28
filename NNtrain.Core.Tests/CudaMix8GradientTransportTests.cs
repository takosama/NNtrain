using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaMix8GradientTransportTests
{
    [Fact]
    public void TwoGpuMix8UsesBf16TransportAndPublishesFp32BeforeOptimizer()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithMix8Cuda([0, 1], () =>
        {
            const int length = 257;
            Parameter parameter = CreateParameter(length, [0, 1]);
            var optimizer = new AdamW(
                [parameter],
                new AdamWOptions
                {
                    LearningRate = 1e-3f,
                    WeightDecay = 0f,
                });
            using var reducer = new CudaBFloat16GradientAllReducePlan(
                [parameter], [0, 1]);
            try
            {
                float[] first = Values(length, 3, 0.13f);
                float[] second = Values(length, 29, 0.07f);
                long stepId = reducer.BeginStep();
                reducer.BeginDeviceStep(stepId, 0);
                reducer.BeginDeviceStep(stepId, 1);
                parameter.T.SetCudaGradient(first, 0);
                parameter.T.SetCudaGradient(second, 1);

                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;
                NativeCudaTransferTelemetry collectiveBefore =
                    NativeCudaRuntime.GradientCollectiveTransferTelemetry;
                reducer.NotifyGradientReady(parameter.T, 0, stepId);
                reducer.NotifyGradientReady(parameter.T, 1, stepId);

                InvalidOperationException incomplete =
                    Assert.Throws<InvalidOperationException>(optimizer.step);
                Assert.Contains("incomplete reduction", incomplete.Message);
                Assert.Equal(0, optimizer.CaptureState().Step);

                reducer.Complete(stepId);
                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - before;
                NativeCudaTransferTelemetry collective =
                    NativeCudaRuntime.GradientCollectiveTransferTelemetry
                        - collectiveBefore;

                long expectedTransport =
                    2L * length * sizeof(ushort);
                long expectedHostPipelineBytes = reducer.UsesHostPipeline
                    ? expectedTransport
                    : 0L;
                Assert.Equal(
                    expectedHostPipelineBytes,
                    collective.HostToDeviceBytes);
                Assert.Equal(
                    expectedHostPipelineBytes,
                    collective.DeviceToHostBytes);
                Assert.Equal(
                    expectedHostPipelineBytes,
                    transfer.HostToDeviceBytes);
                Assert.Equal(
                    expectedHostPipelineBytes + sizeof(double),
                    transfer.DeviceToHostBytes);
                Assert.Equal(expectedTransport, reducer.TransportBytesPerStep);
                Assert.Equal(
                    expectedTransport,
                    reducer.LastCompletedTransportBytes);
                Assert.Equal(1, reducer.CompletedSteps);

                float[] primary = Read(
                    parameter.T.EnsureCudaGradientBuffer(0));
                float[] secondary = Read(
                    parameter.T.EnsureCudaGradientBuffer(1));
                Assert.Equal(primary, secondary);
                AssertClose(
                    Bf16Round(first).Zip(
                        Bf16Round(second),
                        static (left, right) => left + right).ToArray(),
                    primary,
                    1e-6f);

                CudaGradientCoherenceSnapshot snapshot =
                    parameter.T.GetCudaGradientCoherenceSnapshot();
                Assert.Equal(CudaGradientCoherenceKind.Reduced, snapshot.Kind);
                Assert.Equal([0, 1], snapshot.ReducedDevices);
                Assert.True(snapshot.ReductionStamp.IsValid);
                Assert.False(snapshot.PendingStamp.IsValid);

                optimizer.step();
                Assert.Equal(1, optimizer.CaptureState().Step);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void ForcedNoPeerHostPipelineReportsPhysicalChunksByDevice()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithMix8Cuda([0, 1], () =>
        {
            const int length = 257;
            const int chunkElements = 64;
            Parameter parameter = CreateParameter(length, [0, 1]);
            using var reducer = new CudaBFloat16GradientAllReducePlan(
                [parameter],
                [0, 1],
                new CudaDispatchPolicy
                {
                    GradientBucketElements = 512,
                    GradientHostChunkElements = chunkElements,
                },
                peerAccessProbe: static (_, _) => false);
            try
            {
                Assert.True(reducer.UsesHostPipeline);
                Assert.True(reducer.UsesAsyncHostPipeline);
                long stepId = reducer.BeginStep();
                reducer.BeginDeviceStep(stepId, 0);
                reducer.BeginDeviceStep(stepId, 1);
                parameter.T.SetCudaGradient(Values(length, 5, 0.11f), 0);
                parameter.T.SetCudaGradient(Values(length, 37, 0.09f), 1);

                NativeCudaTransferTelemetry totalBefore =
                    NativeCudaRuntime.TransferTelemetry;
                NativeCudaTransferTelemetry collectiveBefore =
                    NativeCudaRuntime.GradientCollectiveTransferTelemetry;
                using IDisposable guard =
                    DeviceTransferGuard.EnterTrainingStep(2);
                reducer.NotifyGradientReady(parameter.T, 0, stepId);
                reducer.NotifyGradientReady(parameter.T, 1, stepId);
                reducer.Complete(stepId);

                DeviceTransferSnapshot ordinary = Assert.NotNull(
                    DeviceTransferGuard.CurrentSnapshot);
                Assert.Equal(0, ordinary.HostToDeviceCopyCount);
                Assert.Equal(0, ordinary.HostToDeviceBytes);
                DeviceTransferTransportSnapshot? transport =
                    DeviceTransferGuard.GetCurrentTransportSnapshot(
                        DeviceTransferTransportCategory.GradientCollective);
                Assert.NotNull(transport);
                long chunksPerPipeline =
                    ((long)length - 1L) / chunkElements + 1L;
                long bytesPerPipeline =
                    (long)length * sizeof(ushort);
                NativeCudaTransferTelemetry total =
                    NativeCudaRuntime.TransferTelemetry - totalBefore;
                NativeCudaTransferTelemetry collective =
                    NativeCudaRuntime.GradientCollectiveTransferTelemetry
                        - collectiveBefore;
                Assert.Equal(
                    2L * chunksPerPipeline,
                    transport.Totals.HostToDeviceCopyCount);
                Assert.Equal(
                    2L * bytesPerPipeline,
                    transport.Totals.HostToDeviceBytes);
                Assert.Equal(
                    2L * chunksPerPipeline,
                    transport.Totals.DeviceToHostCopyCount);
                Assert.Equal(
                    2L * bytesPerPipeline,
                    transport.Totals.DeviceToHostBytes);
                Assert.Equal(
                    transport.Totals.HostToDeviceCopyCount,
                    collective.HostToDeviceCopyCount);
                Assert.Equal(
                    transport.Totals.HostToDeviceBytes,
                    collective.HostToDeviceBytes);
                Assert.Equal(
                    transport.Totals.DeviceToHostCopyCount,
                    collective.DeviceToHostCopyCount);
                Assert.Equal(
                    transport.Totals.DeviceToHostBytes,
                    collective.DeviceToHostBytes);
                Assert.Equal(
                    collective.HostToDeviceCopyCount,
                    total.HostToDeviceCopyCount);
                Assert.Equal(
                    collective.HostToDeviceBytes,
                    total.HostToDeviceBytes);
                Assert.True(
                    total.DeviceToHostCopyCount
                        >= collective.DeviceToHostCopyCount);
                Assert.True(
                    total.DeviceToHostBytes
                        >= collective.DeviceToHostBytes);
                Assert.Collection(
                    transport.Devices,
                    device0 => AssertDeviceTransport(
                        device0,
                        deviceIndex: 0,
                        chunksPerPipeline,
                        bytesPerPipeline),
                    device1 => AssertDeviceTransport(
                        device1,
                        deviceIndex: 1,
                        chunksPerPipeline,
                        bytesPerPipeline));
            }
            finally
            {
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void TwoGpuMix8EngineBuildsOnlyBf16GradientTransportPlan()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 8,
                numHeads: 2,
                dHidden: 16,
                numLayers: 1,
                rng: new Random(211),
                dropout: 0f,
                dtype: TensorDType.Float32);
            model.to(TensorPrecisionMode.Mix8_32, bfp8_block_size: 32);

            Tensor.CudaDeviceIndices = [0, 1];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using var engine = new CudaDataParallelEngine(
                model,
                [0, 1],
                new CudaAdaptiveShardingOptions { Enabled = false });

            engine.PrepareForTraining(batchSize: 2);

            Parameter[] parameters = model.Parameters().ToArray();
            long elements = parameters.Sum(
                static parameter => (long)parameter.T.Numel);
            Assert.NotEmpty(parameters);
            Assert.All(parameters, parameter =>
            {
                Assert.Equal(TensorDType.Bfp8, parameter.T.DType);
                Assert.Equal(
                    Bfp8ScaleGranularity.Block,
                    parameter.T.Bfp8Quantization?.Granularity);
            });
            Assert.Equal(1, engine.BFloat16GradientPlanBuildCount);
            Assert.Equal(0, engine.FlatGradientPlanBuildCount);
            Assert.False(engine.HasFlatGradientPlan);
            Assert.Equal(
                2L * elements * sizeof(ushort),
                engine.BFloat16GradientTransportBytesPerStep);
            Assert.Equal(0, engine.BFloat16GradientTransportCompletedSteps);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void TwoGpuMix8BackwardCompletesBf16TransportForEveryLeaf()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithMix8Cuda([0, 1], () =>
        {
            var model = new GptRinWikiJp(
                vocabularySize: 32,
                contextLength: 4,
                dModel: 8,
                numHeads: 2,
                dHidden: 16,
                numLayers: 1,
                rng: new Random(223),
                dropout: 0f,
                dtype: TensorDType.Float32);
            model.to(TensorPrecisionMode.Mix8_32, bfp8_block_size: 32);
            model.to(TensorDevice.Cuda);
            Parameter[] parameters = model.Parameters().ToArray();
            try
            {
                using var engine = new CudaDataParallelEngine(
                    model,
                    [0, 1],
                    new CudaAdaptiveShardingOptions { Enabled = false });
                model.zero_grad();

                float loss = engine.ForwardBackward(
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [2, 3, 4, 5, 6, 7, 8, 9],
                    batchSize: 2,
                    sequenceLength: 4);

                long elements = parameters.Sum(
                    static parameter => (long)parameter.T.Numel);
                Assert.True(float.IsFinite(loss));
                Assert.Equal(1, engine.BFloat16GradientPlanBuildCount);
                Assert.Equal(0, engine.FlatGradientPlanBuildCount);
                Assert.False(engine.HasFlatGradientPlan);
                Assert.Equal(1, engine.BFloat16GradientTransportCompletedSteps);
                Assert.Equal(
                    2L * elements * sizeof(ushort),
                    engine.LastBFloat16GradientTransportBytes);
                foreach (Parameter parameter in parameters)
                {
                    CudaGradientCoherenceSnapshot snapshot =
                        parameter.T.GetCudaGradientCoherenceSnapshot();
                    Assert.Equal(
                        CudaGradientCoherenceKind.Reduced,
                        snapshot.Kind);
                    Assert.Equal([0, 1], snapshot.ReducedDevices);
                    Assert.True(snapshot.ReductionStamp.IsValid);
                    Assert.Equal(
                        Read(parameter.T.EnsureCudaGradientBuffer(0)),
                        Read(parameter.T.EnsureCudaGradientBuffer(1)));
                }
            }
            finally
            {
                foreach (Parameter parameter in parameters)
                    parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    private static Parameter CreateParameter(
        int length,
        IReadOnlyList<int> devices)
    {
        var parameter = new Parameter(
            Values(length, 7, 0.05f),
            [length],
            "mix.weight",
            WeightDecayPolicy.Apply);
        parameter.T.ConvertStorageInPlace(
            TensorDType.Bfp8,
            Bfp8QuantizationDescriptor.Block(128),
            preserveFloat32Master: true);
        foreach (int device in devices)
            parameter.T.EnsureCudaBfp8Buffer(device);
        parameter.T.to(new TorchDevice(TensorDevice.Cuda, devices[0]));
        return parameter;
    }

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index => index == length - 1
                ? scale * 2.75f
                : MathF.Sin((index + offset) * 0.173f) * scale)
            .ToArray();

    private static float[] Bf16Round(float[] values)
    {
        var encoded = new ushort[values.Length];
        var decoded = new float[values.Length];
        TensorStorageCodec.EncodeBFloat16(values, encoded);
        TensorStorageCodec.DecodeBFloat16(encoded, decoded);
        return decoded;
    }

    private static float[] Read(NativeCudaBuffer<float> buffer)
    {
        var result = new float[buffer.Length];
        buffer.CopyToCPU(result);
        return result;
    }

    private static void AssertClose(
        IReadOnlyList<float> expected,
        IReadOnlyList<float> actual,
        float tolerance)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int index = 0; index < expected.Count; index++)
        {
            Assert.InRange(
                MathF.Abs(expected[index] - actual[index]),
                0f,
                tolerance);
        }
    }

    private static void AssertDeviceTransport(
        DeviceTransferDeviceSnapshot snapshot,
        int deviceIndex,
        long copyCount,
        long byteLength)
    {
        Assert.Equal(deviceIndex, snapshot.DeviceIndex);
        Assert.Equal(copyCount, snapshot.HostToDeviceCopyCount);
        Assert.Equal(byteLength, snapshot.HostToDeviceBytes);
        Assert.Equal(copyCount, snapshot.DeviceToHostCopyCount);
        Assert.Equal(byteLength, snapshot.DeviceToHostBytes);
    }

    private static void WithMix8Cuda(
        int[] devices,
        Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = devices;
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(
                    PrecisionPolicy.Mix8_32);
            action();
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }
}
