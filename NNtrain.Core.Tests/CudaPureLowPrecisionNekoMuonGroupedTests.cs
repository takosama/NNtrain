using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

[Collection(TensorSimdCollection.Name)]
public sealed class CudaPureLowPrecisionNekoMuonGroupedTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FixedFiveEqualShapesUseBatchedNs5AndMatchScalar(
        bool bfp8)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], bfp8, () =>
        {
            Scenario scalar = RunOneGpu(bfp8, disableBatching: true);
            Scenario grouped = RunOneGpu(bfp8, disableBatching: false);

            AssertSnapshotsEqual(scalar.Snapshots, grouped.Snapshots);
            Assert.Equal(3 * 8, scalar.Telemetry.ScalarDispatchCount);
            Assert.Equal(0, scalar.Telemetry.BatchedDispatchCount);
            Assert.Equal(3 * 8 * 15,
                scalar.Telemetry.GemmLaunchCount);
            Assert.Equal(0, grouped.Telemetry.ScalarDispatchCount);
            Assert.Equal(3, grouped.Telemetry.BatchedDispatchCount);
            Assert.Equal(3 * 15, grouped.Telemetry.GemmLaunchCount);
            Assert.Equal(
                scalar.Telemetry.LogicalMatrixCount,
                grouped.Telemetry.LogicalMatrixCount);
            Assert.Equal(0, grouped.HotTransfers.HostToDeviceBytes);
            Assert.Equal(bfp8 ? sizeof(int) : 0,
                grouped.HotTransfers.DeviceToHostBytes);
            Assert.Equal(0, grouped.HotAllocations.AllocationCount);
            Assert.Equal(0, grouped.HotAllocations.FreeCount);
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FixedFiveGroupedTwoGpuReplicasAndMomentsMatch(bool bfp8)
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithCuda([0, 1], bfp8, () =>
        {
            const int parameterCount = 4;
            const int rows = 48;
            const int columns = 64;
            Parameter[] parameters = Enumerable.Range(0, parameterCount)
                .Select(index => CreateParameter(
                    bfp8,
                    Values(rows * columns, 7 + index * 11, 0.18f),
                    [rows, columns],
                    $"pure-low.dp.{index}",
                    [0, 1]))
                .ToArray();
            var optimizer = new NekoMuon(
                parameters,
                FixedFiveOptions(),
                CudaDispatchPolicy.Defaults with
                {
                    NekoMuonBatchSize = parameterCount,
                });
            IDisposable reducer = bfp8
                ? new CudaBfp8GradientAllReducePlan(
                    parameters, [0, 1])
                : new CudaBFloat16GradientAllReducePlan(
                    parameters,
                    [0, 1],
                    useBFloat16GradientStorage: true);
            try
            {
                optimizer.prepare();
                NativeCudaTransferTelemetry hotTransfers = default;
                NativeCudaAllocationTelemetry hotAllocations = default;
                NekoMuonFixedNs5TelemetrySnapshot telemetry = default;
                for (int step = 0; step < 2; step++)
                {
                    ReduceGradients(reducer, parameters, bfp8, step);
                    NativeCudaTransferTelemetry transferBefore =
                        NativeCudaRuntime.TransferTelemetry;
                    NativeCudaAllocationTelemetry allocationBefore =
                        NativeCudaRuntime.AllocationTelemetry;
                    NekoMuonFixedNs5TelemetrySnapshot telemetryBefore =
                        NekoMuonFixedNs5Telemetry.Snapshot;
                    optimizer.step();
                    if (step == 1)
                    {
                        hotTransfers = NativeCudaRuntime.TransferTelemetry
                            - transferBefore;
                        hotAllocations = NativeCudaRuntime.AllocationTelemetry
                            - allocationBefore;
                        telemetry = NekoMuonFixedNs5Telemetry.Snapshot
                            - telemetryBefore;
                    }
                    optimizer.zero_grad();
                }

                Assert.Equal(0, telemetry.ScalarDispatchCount);
                Assert.Equal(2, telemetry.BatchedDispatchCount);
                Assert.Equal(parameterCount * 2,
                    telemetry.LogicalMatrixCount);
                Assert.Equal(2 * 15, telemetry.GemmLaunchCount);
                Assert.Equal(0, hotTransfers.HostToDeviceBytes);
                Assert.Equal(bfp8 ? 2 * sizeof(int) : 0,
                    hotTransfers.DeviceToHostBytes);
                Assert.Equal(0, hotAllocations.AllocationCount);
                Assert.Equal(0, hotAllocations.FreeCount);
                for (int index = 0; index < parameters.Length; index++)
                {
                    Assert.Equal(
                        ReadData(parameters[index], bfp8, 0),
                        ReadData(parameters[index], bfp8, 1));
                    Assert.Equal(
                        ReadFast(optimizer, index, bfp8, 0),
                        ReadFast(optimizer, index, bfp8, 1));
                    Assert.Equal(
                        ReadSlow(optimizer, index, bfp8, 0),
                        ReadSlow(optimizer, index, bfp8, 1));
                    Assert.False(
                        parameters[index].T.HasCudaMasterFloat32Buffer(0));
                    Assert.False(
                        parameters[index].T.HasCudaMasterFloat32Buffer(1));
                }
            }
            finally
            {
                reducer.Dispose();
                optimizer.DisposeCudaResources();
                foreach (Parameter parameter in parameters)
                    parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    private static Scenario RunOneGpu(bool bfp8, bool disableBatching)
    {
        const int parameterCount = 8;
        const int rows = 48;
        const int columns = 64;
        Parameter[] parameters = Enumerable.Range(0, parameterCount)
            .Select(index => CreateParameter(
                bfp8,
                Values(rows * columns, 7 + index * 11, 0.18f),
                [rows, columns],
                $"pure-low.{index}",
                [0]))
            .ToArray();
        var optimizer = new NekoMuon(
            parameters,
            FixedFiveOptions(),
            CudaDispatchPolicy.Defaults with
            {
                DisableBatchedNekoMuon = disableBatching,
                NekoMuonBatchSize = parameterCount,
            });
        try
        {
            optimizer.prepare();
            NekoMuonFixedNs5TelemetrySnapshot before =
                NekoMuonFixedNs5Telemetry.Snapshot;
            NativeCudaTransferTelemetry hotTransfers = default;
            NativeCudaAllocationTelemetry hotAllocations = default;
            for (int step = 0; step < 3; step++)
            {
                foreach ((Parameter parameter, int index) in parameters
                    .Select((parameter, index) => (parameter, index)))
                {
                    PublishOneGpuGradient(
                        parameter,
                        bfp8,
                        Values(
                            parameter.T.Numel,
                            31 + index * 17 + step * 7,
                            0.035f));
                }
                NativeCudaTransferTelemetry transferBefore =
                    NativeCudaRuntime.TransferTelemetry;
                NativeCudaAllocationTelemetry allocationBefore =
                    NativeCudaRuntime.AllocationTelemetry;
                optimizer.step();
                if (step == 2)
                {
                    hotTransfers = NativeCudaRuntime.TransferTelemetry
                        - transferBefore;
                    hotAllocations = NativeCudaRuntime.AllocationTelemetry
                        - allocationBefore;
                }
                optimizer.zero_grad();
            }
            NekoMuonFixedNs5TelemetrySnapshot telemetry =
                NekoMuonFixedNs5Telemetry.Snapshot - before;
            PrecisionSnapshot[] snapshots = parameters
                .Select((parameter, index) => new PrecisionSnapshot(
                    ReadData(parameter, bfp8, 0),
                    ReadFast(optimizer, index, bfp8, 0),
                    ReadSlow(optimizer, index, bfp8, 0)))
                .ToArray();
            Assert.All(parameters, parameter => Assert.False(
                parameter.T.HasCudaMasterFloat32Buffer(0)));
            return new Scenario(
                snapshots,
                telemetry,
                hotTransfers,
                hotAllocations);
        }
        finally
        {
            optimizer.DisposeCudaResources();
            foreach (Parameter parameter in parameters)
                parameter.T.InvalidateCudaBuffers();
        }
    }

    private static void PublishOneGpuGradient(
        Parameter parameter,
        bool bfp8,
        float[] gradient)
    {
        if (!bfp8)
        {
            parameter.T.BackwardAndRelease(gradient);
            return;
        }
        parameter.T.SetCudaGradient(gradient, 0);
        _ = parameter.T.PublishCudaBfp8Gradient(0);
    }

    private static void ReduceGradients(
        IDisposable reducer,
        IReadOnlyList<Parameter> parameters,
        bool bfp8,
        int step)
    {
        switch (reducer)
        {
            case CudaBfp8GradientAllReducePlan bfp8Reducer:
            {
                long stepId = bfp8Reducer.BeginStep();
                try
                {
                    for (int device = 0; device < 2; device++)
                    {
                        bfp8Reducer.BeginDeviceStep(stepId, device);
                        for (int index = 0; index < parameters.Count; index++)
                        {
                            Parameter parameter = parameters[index];
                            parameter.T.SetCudaGradient(
                                Values(
                                    parameter.T.Numel,
                                    41 + index * 13 + device * 19
                                        + step * 7,
                                    0.025f),
                                device);
                            bfp8Reducer.NotifyGradientReady(
                                parameter.T,
                                device,
                                stepId);
                        }
                    }
                    bfp8Reducer.Complete(stepId);
                }
                catch
                {
                    bfp8Reducer.Abort(stepId);
                    throw;
                }
                break;
            }
            case CudaBFloat16GradientAllReducePlan bfloat16Reducer:
            {
                long stepId = bfloat16Reducer.BeginStep();
                try
                {
                    for (int device = 0; device < 2; device++)
                    {
                        bfloat16Reducer.BeginDeviceStep(stepId, device);
                        for (int index = 0; index < parameters.Count; index++)
                        {
                            Parameter parameter = parameters[index];
                            Assert.True(parameter.T
                                .TryGetCudaBFloat16GradientBuffer(
                                    device,
                                    out NativeCudaBuffer<ushort>? buffer));
                            ushort[] encoded = Values(
                                    parameter.T.Numel,
                                    41 + index * 13 + device * 19
                                        + step * 7,
                                    0.025f)
                                .Select(TensorStorageCodec.EncodeBFloat16)
                                .ToArray();
                            buffer!.CopyFromCPU(encoded);
                            buffer.MarkGradientStorageDirty();
                            bfloat16Reducer.NotifyGradientReady(
                                parameter.T,
                                device,
                                stepId);
                        }
                    }
                    bfloat16Reducer.Complete(stepId);
                }
                catch
                {
                    bfloat16Reducer.Abort(stepId);
                    throw;
                }
                break;
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(reducer));
        }
        Assert.Equal(bfp8,
            parameters.All(parameter =>
                parameter.T.HasAuthoritativeCudaBfp8Gradient));
    }

    private static Parameter CreateParameter(
        bool bfp8,
        float[] values,
        int[] shape,
        string name,
        int[] devices)
    {
        var parameter = new Parameter(
            values,
            shape,
            name,
            WeightDecayPolicy.Apply,
            bfp8 ? TensorDType.Float32 : TensorDType.BFloat16);
        if (bfp8)
        {
            parameter.T.ConvertStorageInPlace(
                TensorDType.Bfp8,
                Bfp8QuantizationDescriptor.TensorWide,
                preserveFloat32Master: false);
        }
        foreach (int device in devices)
        {
            if (bfp8)
                _ = parameter.T.EnsureCudaBfp8Buffer(device);
            else
                _ = parameter.T.EnsureCudaBFloat16Buffer(device);
        }
        parameter.T.to(new TorchDevice(TensorDevice.Cuda, devices[0]));
        return parameter;
    }

    private static float[] ReadData(
        Parameter parameter,
        bool bfp8,
        int device)
        => bfp8
            ? Read(parameter.T.EnsureCudaBfp8Buffer(device))
            : Read(parameter.T.EnsureCudaBFloat16Buffer(device));

    private static float[] ReadFast(
        NekoMuon optimizer,
        int parameterIndex,
        bool bfp8,
        int device)
        => bfp8
            ? Read(optimizer.GetCudaBfp8Moments(
                parameterIndex, device).Fast)
            : Read(optimizer.GetCudaBFloat16Moments(
                parameterIndex, device).Fast);

    private static float[] ReadSlow(
        NekoMuon optimizer,
        int parameterIndex,
        bool bfp8,
        int device)
        => bfp8
            ? Read(optimizer.GetCudaBfp8Moments(
                parameterIndex, device).Slow)
            : Read(optimizer.GetCudaBFloat16Moments(
                parameterIndex, device).Slow);

    private static float[] Read(NativeCudaBuffer<ushort> buffer)
    {
        var encoded = new ushort[buffer.Length];
        buffer.CopyToCPU(encoded);
        return encoded.Select(TensorStorageCodec.DecodeBFloat16).ToArray();
    }

    private static float[] Read(CudaBfp8BufferView view)
    {
        var payload = new sbyte[view.Payload.Length];
        var scales = new float[view.Scales.Length];
        view.Payload.CopyToCPU(payload);
        view.Scales.CopyToCPU(scales);
        var decoded = new float[payload.Length];
        Bfp8QuantizationCodec.Default.Decode(
            payload,
            scales,
            view.Descriptor,
            decoded);
        return decoded;
    }

    private static void AssertSnapshotsEqual(
        IReadOnlyList<PrecisionSnapshot> expected,
        IReadOnlyList<PrecisionSnapshot> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int index = 0; index < expected.Count; index++)
        {
            AssertClose(
                expected[index].Data,
                actual[index].Data,
                8e-4f);
            Assert.Equal(expected[index].Fast, actual[index].Fast);
            Assert.Equal(expected[index].Slow, actual[index].Slow);
        }
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

    private static NekoMuonOptions FixedFiveOptions() => new()
    {
        LearningRate = 0.002f,
        BetaFast = 0.8f,
        BetaSlow = 0.95f,
        Rho = 0.7f,
        Epsilon = 1e-6f,
        MaxNewtonSchulzSteps = 5,
        NewtonSchulzInterval = 1,
        NewtonSchulzDepthMode = NekoMuonNewtonSchulzDepthMode.Fixed,
        NewtonSchulzDepth = 5f,
        WeightDecay = 0.01f,
    };

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index => index == length - 1
                ? scale * 2.75f
                : MathF.Sin((index + offset) * 0.173f) * scale)
            .ToArray();

    private static void WithCuda(
        int[] devices,
        bool bfp8,
        Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = devices;
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(
                    bfp8
                        ? PrecisionPolicy.Bfp8
                        : PrecisionPolicy.BFloat16);
            action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private sealed record PrecisionSnapshot(
        float[] Data,
        float[] Fast,
        float[] Slow);

    private sealed record Scenario(
        PrecisionSnapshot[] Snapshots,
        NekoMuonFixedNs5TelemetrySnapshot Telemetry,
        NativeCudaTransferTelemetry HotTransfers,
        NativeCudaAllocationTelemetry HotAllocations);
}
