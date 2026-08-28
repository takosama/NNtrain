using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaPureBFloat16OptimizerStateTests
{
    [Fact]
    public void AdamWUsesRoundedBFloat16MomentsAndHasNoWarmStepTransfer()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            const int length = 257;
            AdamWOptions options = AdamOptions();
            Parameter parameter = CreateParameter(
                Values(length, 7, 0.7f), [length], "bf16.adam");
            var optimizer = new AdamW([parameter], options);
            try
            {
                ushort[] initial = Read(
                    parameter.T.EnsureCudaBFloat16Buffer(0));
                parameter.T.BackwardAndRelease(
                    Values(length, 29, 0.13f));
                Assert.True(parameter.T.TryGetCudaBFloat16GradientBuffer(
                    0, out NativeCudaBuffer<ushort>? gradientBuffer));
                ushort[] gradient = Read(gradientBuffer!);

                optimizer.step();

                (ushort[] expectedData, ushort[] expectedFirst,
                    ushort[] expectedSecond) = AdamReferenceStep(
                        initial,
                        gradient,
                        new ushort[length],
                        new ushort[length],
                        options,
                        step: 1);
                Assert.Equal(
                    expectedData,
                    Read(parameter.T.EnsureCudaBFloat16Buffer(0)));
                (NativeCudaBuffer<short> first,
                    NativeCudaBuffer<short> second) =
                    optimizer.GetCudaBFloat16Moments(0, 0);
                Assert.Equal(expectedFirst, Read(first));
                Assert.Equal(expectedSecond, Read(second));
                Assert.False(parameter.T.HasCudaMasterFloat32Buffer(0));

                optimizer.zero_grad();
                parameter.T.BackwardAndRelease(
                    Values(length, 61, 0.09f));
                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;

                optimizer.step();

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - before;
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(0, transfer.DeviceToHostBytes);
                Assert.False(parameter.T.HasCudaMasterFloat32Buffer(0));
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void NekoMuonFixedFiveKeepsBFloat16StateAndTransfersNoHotPayload()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Parameter parameter = CreateParameter(
                Values(128, 11, 0.55f), [8, 16], "bf16.neko");
            var optimizer = new NekoMuon([parameter], FixedFiveOptions());
            try
            {
                parameter.T.BackwardAndRelease(Values(128, 37, 0.08f));
                optimizer.step();
                optimizer.zero_grad();
                parameter.T.BackwardAndRelease(Values(128, 71, 0.06f));
                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;

                optimizer.step();

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - before;
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(0, transfer.DeviceToHostBytes);
                (NativeCudaBuffer<ushort> fast,
                    NativeCudaBuffer<ushort> slow) =
                    optimizer.GetCudaBFloat16Moments(0, 0);
                Assert.Equal(parameter.T.Numel, fast.Length);
                Assert.Equal(parameter.T.Numel, slow.Length);
                Assert.Contains(Read(fast), value => value != 0);
                Assert.Contains(Read(slow), value => value != 0);
                Assert.All(
                    Read(parameter.T.EnsureCudaBFloat16Buffer(0))
                        .Select(TensorStorageCodec.DecodeBFloat16),
                    value => Assert.True(float.IsFinite(value)));
                Assert.False(parameter.T.HasCudaMasterFloat32Buffer(0));
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void AdamWTwoGpuReplicasAndBFloat16MomentsRemainIdentical()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithCuda([0, 1], () =>
        {
            Parameter parameter = CreateParameter(
                Values(257, 3, 0.65f), [257], "bf16.adam.dp", [0, 1]);
            var optimizer = new AdamW([parameter], AdamOptions());
            using var reducer = new CudaBFloat16GradientAllReducePlan(
                [parameter],
                [0, 1],
                new CudaDispatchPolicy { GradientBucketElements = 512 },
                useBFloat16GradientStorage: true);
            try
            {
                ReduceGradient(
                    reducer,
                    parameter,
                    Values(257, 17, 0.11f),
                    Values(257, 43, 0.07f));
                optimizer.step();
                optimizer.zero_grad();
                ReduceGradient(
                    reducer,
                    parameter,
                    Values(257, 59, 0.08f),
                    Values(257, 89, 0.05f));
                Assert.True(parameter.T.TryGetCudaBFloat16GradientBuffer(
                    0, out NativeCudaBuffer<ushort>? primaryGradient));
                Assert.True(parameter.T.TryGetCudaBFloat16GradientBuffer(
                    1, out NativeCudaBuffer<ushort>? secondaryGradient));
                Assert.Equal(Read(primaryGradient!), Read(secondaryGradient!));
                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;

                optimizer.step();

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - before;
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(0, transfer.DeviceToHostBytes);
                Assert.Equal(
                    Read(parameter.T.EnsureCudaBFloat16Buffer(0)),
                    Read(parameter.T.EnsureCudaBFloat16Buffer(1)));
                var primary = optimizer.GetCudaBFloat16Moments(0, 0);
                var secondary = optimizer.GetCudaBFloat16Moments(0, 1);
                Assert.Equal(Read(primary.First), Read(secondary.First));
                Assert.Equal(Read(primary.Second), Read(secondary.Second));
                Assert.False(parameter.T.HasCudaMasterFloat32Buffer(0));
                Assert.False(parameter.T.HasCudaMasterFloat32Buffer(1));
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void NekoMuonCheckpointRoundTripRestoresBFloat16Authority()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            NekoMuonOptions options = FixedFiveOptions();
            Parameter source = CreateParameter(
                Values(128, 5, 0.5f), [8, 16], "bf16.neko.resume");
            var uninterrupted = new NekoMuon([source], options);
            NekoMuon? resumed = null;
            Parameter? restored = null;
            try
            {
                source.T.BackwardAndRelease(Values(128, 23, 0.08f));
                uninterrupted.step();
                float[] restoredData = source.T.Data.ToArray();
                using var checkpoint = new MemoryStream();
                OptimizerStateStream.SaveStateBinary(
                    uninterrupted, checkpoint);

                restored = CreateParameter(
                    restoredData, [8, 16], "bf16.neko.resume");
                resumed = new NekoMuon([restored], options);
                checkpoint.Position = 0;
                OptimizerStateStream.LoadStateBinary(resumed, checkpoint);

                float[] nextGradient = Values(128, 53, 0.06f);
                uninterrupted.zero_grad();
                resumed.zero_grad();
                source.T.BackwardAndRelease(nextGradient);
                restored.T.BackwardAndRelease(nextGradient);
                uninterrupted.step();
                resumed.step();

                Assert.Equal(
                    Read(source.T.EnsureCudaBFloat16Buffer(0)),
                    Read(restored.T.EnsureCudaBFloat16Buffer(0)));
                var expected = uninterrupted.GetCudaBFloat16Moments(0, 0);
                var actual = resumed.GetCudaBFloat16Moments(0, 0);
                Assert.Equal(Read(expected.Fast), Read(actual.Fast));
                Assert.Equal(Read(expected.Slow), Read(actual.Slow));
                Assert.False(source.T.HasCudaMasterFloat32Buffer(0));
                Assert.False(restored.T.HasCudaMasterFloat32Buffer(0));
            }
            finally
            {
                resumed?.DisposeCudaResources();
                uninterrupted.DisposeCudaResources();
                restored?.T.InvalidateCudaBuffers();
                source.T.InvalidateCudaBuffers();
            }
        });
    }

    private static AdamWOptions AdamOptions() => new()
    {
        LearningRate = 0.003f,
        Beta1 = 0.8f,
        Beta2 = 0.91f,
        Epsilon = 1e-6f,
        WeightDecay = 0.02f,
        Decay1D = true,
    };

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

    private static Parameter CreateParameter(
        float[] values,
        int[] shape,
        string name,
        int[]? devices = null)
    {
        var parameter = new Parameter(
            values,
            shape,
            name,
            WeightDecayPolicy.Apply,
            TensorDType.BFloat16);
        foreach (int device in devices ?? [0])
            _ = parameter.T.EnsureCudaBFloat16Buffer(device);
        parameter.T.to(new TorchDevice(
            TensorDevice.Cuda, devices?[0] ?? 0));
        return parameter;
    }

    private static (ushort[] Data, ushort[] First, ushort[] Second)
        AdamReferenceStep(
            ushort[] data,
            ushort[] gradient,
            ushort[] first,
            ushort[] second,
            AdamWOptions options,
            int step)
    {
        float bc1 = 1f - MathF.Pow(options.Beta1, step);
        float bc2 = 1f - MathF.Pow(options.Beta2, step);
        float sqrtBc2 = MathF.Sqrt(bc2);
        float updateScale = options.LearningRate * sqrtBc2 / bc1;
        float scaledEpsilon = options.Epsilon * sqrtBc2;
        var nextData = new ushort[data.Length];
        var nextFirst = new ushort[data.Length];
        var nextSecond = new ushort[data.Length];
        for (int index = 0; index < data.Length; index++)
        {
            float g = TensorStorageCodec.DecodeBFloat16(gradient[index]);
            float m = MathF.FusedMultiplyAdd(
                options.Beta1,
                TensorStorageCodec.DecodeBFloat16(first[index]),
                (1f - options.Beta1) * g);
            float v = MathF.FusedMultiplyAdd(
                options.Beta2,
                TensorStorageCodec.DecodeBFloat16(second[index]),
                (1f - options.Beta2) * g * g);
            nextFirst[index] = TensorStorageCodec.EncodeBFloat16(m);
            nextSecond[index] = TensorStorageCodec.EncodeBFloat16(v);
            m = TensorStorageCodec.DecodeBFloat16(nextFirst[index]);
            v = TensorStorageCodec.DecodeBFloat16(nextSecond[index]);
            float parameter = TensorStorageCodec.DecodeBFloat16(data[index]);
            parameter *= 1f - options.LearningRate * options.WeightDecay;
            parameter -= updateScale * m
                / (MathF.Sqrt(v) + scaledEpsilon);
            nextData[index] = TensorStorageCodec.EncodeBFloat16(parameter);
        }
        return (nextData, nextFirst, nextSecond);
    }

    private static void ReduceGradient(
        CudaBFloat16GradientAllReducePlan reducer,
        Parameter parameter,
        float[] first,
        float[] second)
    {
        long stepId = reducer.BeginStep();
        try
        {
            Publish(reducer, stepId, 0, parameter, first);
            Publish(reducer, stepId, 1, parameter, second);
            reducer.Complete(stepId);
        }
        catch
        {
            reducer.Abort(stepId);
            throw;
        }
    }

    private static void Publish(
        CudaBFloat16GradientAllReducePlan reducer,
        long stepId,
        int deviceIndex,
        Parameter parameter,
        float[] values)
    {
        reducer.BeginDeviceStep(stepId, deviceIndex);
        Assert.True(parameter.T.TryGetCudaBFloat16GradientBuffer(
            deviceIndex, out NativeCudaBuffer<ushort>? buffer));
        var encoded = new ushort[values.Length];
        TensorStorageCodec.EncodeBFloat16(values, encoded);
        buffer!.CopyFromCPU(encoded);
        buffer.MarkGradientStorageDirty();
        reducer.NotifyGradientReady(parameter.T, deviceIndex, stepId);
    }

    private static ushort[] Read(NativeCudaBuffer<ushort> buffer)
    {
        var values = new ushort[buffer.Length];
        buffer.CopyToCPU(values);
        return values;
    }

    private static ushort[] Read(NativeCudaBuffer<short> buffer)
    {
        var signed = new short[buffer.Length];
        buffer.CopyToCPU(signed);
        return signed.Select(value => unchecked((ushort)value)).ToArray();
    }

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index => MathF.Sin((index + offset) * 0.173f) * scale)
            .ToArray();

    private static void WithCuda(int[] devices, Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = devices;
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(
                    PrecisionPolicy.BFloat16);
            action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }
}
