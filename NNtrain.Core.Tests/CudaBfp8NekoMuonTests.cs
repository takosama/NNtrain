using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaBfp8NekoMuonTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(33)]
    [InlineData(257)]
    public void OneGpuMomentsAndParameterRemainTensorWideBfp8(int length)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Parameter parameter = CreateParameter(
                Values(length, 3, 0.7f), [1, length], "hidden");
            NekoMuonOptions options = Options(runNewtonSchulz: false);
            var optimizer = new NekoMuon([parameter], options);
            try
            {
                Bfp8EncodedStorage initial = Read(
                    parameter.T.EnsureCudaBfp8Buffer(0));
                Bfp8EncodedStorage gradient = PublishGradient(
                    parameter, Values(length, 19, 0.12f), 0);

                optimizer.step();

                float[] decodedGradient = Decode(gradient);
                Bfp8EncodedStorage expectedFast = Encode(decodedGradient
                    .Select(value => (1f - options.BetaFast) * value)
                    .ToArray());
                Bfp8EncodedStorage expectedSlow = Encode(decodedGradient
                    .Select(value => (1f - options.BetaSlow) * value)
                    .ToArray());
                var moments = optimizer.GetCudaBfp8Moments(0, 0);
                AssertEncoded(expectedFast, Read(moments.Fast));
                AssertEncoded(expectedSlow, Read(moments.Slow));
                Assert.True(ReadScale(moments.Fast) > 0f);
                Assert.True(ReadScale(moments.Slow) > 0f);
                Assert.False(parameter.T.HasCudaMasterFloat32Buffer(0));

                float[] expectedData = Decode(initial);
                float[] fast = Decode(expectedFast);
                double normSquared = decodedGradient.Sum(
                    value => (double)value * value);
                float inverseNorm = 1f
                    / ((float)Math.Sqrt(normSquared) + options.Epsilon);
                float correction = 1f - options.BetaFast;
                for (int index = 0; index < expectedData.Length; index++)
                {
                    expectedData[index] *= 1f
                        - options.LearningRate * options.WeightDecay;
                    expectedData[index] -= options.LearningRate
                        * (fast[index] / correction)
                        * inverseNorm;
                }
                float[] actualData = Decode(Read(
                    parameter.T.EnsureCudaBfp8Buffer(0)));
                float tolerance = MathF.Max(
                    2e-5f,
                    ReadScale(parameter.T.EnsureCudaBfp8Buffer(0)) * 1.5f);
                AssertClose(expectedData, actualData, tolerance);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void FixedFiveStepNewtonSchulzPublishesFiniteBfp8State()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Parameter parameter = CreateParameter(
                Values(128, 5, 0.5f), [8, 16], "hidden");
            NekoMuonOptions options = Options(runNewtonSchulz: true) with
            {
                MaxNewtonSchulzSteps = 5,
                NewtonSchulzDepthMode =
                    NekoMuonNewtonSchulzDepthMode.Fixed,
                NewtonSchulzDepth = 5f,
            };
            var optimizer = new NekoMuon([parameter], options);
            try
            {
                PublishGradient(parameter, Values(128, 37, 0.08f), 0);
                optimizer.step();
                Assert.All(
                    Decode(Read(parameter.T.EnsureCudaBfp8Buffer(0))),
                    value => Assert.True(float.IsFinite(value)));
                var moments = optimizer.GetCudaBfp8Moments(0, 0);
                Assert.True(ReadScale(moments.Fast) > 0f);
                Assert.True(ReadScale(moments.Slow) > 0f);
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
    public void FixedFiveWarmStepReadsOnlyOneFiniteScalarRegardlessOfLeaves()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Parameter[] parameters =
            [
                CreateParameter(
                    Values(128, 7, 0.5f), [8, 16], "hidden.0"),
                CreateParameter(
                    Values(96, 17, 0.4f), [8, 12], "hidden.1"),
            ];
            NekoMuonOptions options = Options(runNewtonSchulz: true) with
            {
                MaxNewtonSchulzSteps = 5,
                NewtonSchulzDepthMode =
                    NekoMuonNewtonSchulzDepthMode.Fixed,
                NewtonSchulzDepth = 5f,
            };
            var optimizer = new NekoMuon(parameters, options);
            try
            {
                for (int index = 0; index < parameters.Length; index++)
                {
                    PublishGradient(
                        parameters[index],
                        Values(parameters[index].T.Numel, 31 + index, 0.07f),
                        0);
                }
                optimizer.step();
                optimizer.zero_grad();
                for (int index = 0; index < parameters.Length; index++)
                {
                    PublishGradient(
                        parameters[index],
                        Values(parameters[index].T.Numel, 51 + index, 0.05f),
                        0);
                }
                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;

                optimizer.step();

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - before;
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(sizeof(int), transfer.DeviceToHostBytes);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                foreach (Parameter parameter in parameters)
                    parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void QuantizedMomentsDriveFiniteStatisticsAcrossSparseSteps()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            const int length = 128;
            Parameter parameter = CreateParameter(
                Values(length, 7, 0.4f), [8, 16], "hidden.sparse");
            NekoMuonOptions options = Options(runNewtonSchulz: true) with
            {
                MaxNewtonSchulzSteps = 5,
                NewtonSchulzDepthMode =
                    NekoMuonNewtonSchulzDepthMode.Fixed,
                NewtonSchulzDepth = 5f,
            };
            var optimizer = new NekoMuon([parameter], options);
            try
            {
                for (int step = 0; step < 24; step++)
                {
                    if (step != 0)
                        optimizer.zero_grad();
                    float[] gradient = new float[length];
                    if ((step & 1) == 0)
                    {
                        for (int index = 0; index < gradient.Length - 1;
                            index++)
                        {
                            gradient[index] = 0.002f * MathF.Sin(
                                (step + 1) * (index + 3) * 0.07f);
                        }
                        gradient[^1] = (step & 2) == 0 ? 0.9f : -0.9f;
                    }
                    PublishGradient(parameter, gradient, 0);
                    optimizer.step();

                    float[] values = Decode(Read(
                        parameter.T.EnsureCudaBfp8Buffer(0)));
                    Assert.All(values,
                        value => Assert.True(float.IsFinite(value)));
                    Assert.True(values.Max(MathF.Abs) < 2f);
                    var moments = optimizer.GetCudaBfp8Moments(0, 0);
                    Assert.True(ReadScale(moments.Fast) > 0f);
                    Assert.True(ReadScale(moments.Slow) > 0f);
                    NekoMuonDiagnostics diagnostics =
                        optimizer.GetDiagnostics();
                    Assert.InRange(diagnostics.MinimumConfidence, 0f, 1f);
                    Assert.InRange(diagnostics.MeanConfidence, 0f, 1f);
                    Assert.InRange(diagnostics.MaximumConfidence, 0f, 1f);
                    Assert.False(parameter.T.HasCudaMasterFloat32Buffer(0));
                }
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void WarmTwoGpuStepHasNoPayloadTransferAndReplicasMatch()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithCuda([0, 1], () =>
        {
            Parameter parameter = CreateParameter(
                Values(129, 7, 0.6f), [3, 43], "hidden", [0, 1]);
            var optimizer = new NekoMuon(
                [parameter], Options(runNewtonSchulz: false));
            using var reducer = new CudaBfp8GradientAllReducePlan(
                [parameter], [0, 1]);
            try
            {
                ReduceGradient(
                    reducer,
                    parameter,
                    Values(129, 13, 0.09f),
                    Values(129, 31, 0.07f));
                optimizer.step();
                optimizer.zero_grad();
                ReduceGradient(
                    reducer,
                    parameter,
                    Values(129, 41, 0.06f),
                    Values(129, 59, 0.05f));
                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;

                optimizer.step();

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - before;
                Assert.Equal(0, transfer.HostToDeviceBytes);
                // Four FP32 statistics plus one finite-status scalar per GPU.
                Assert.Equal(2 * (4 * sizeof(float) + sizeof(int)),
                    transfer.DeviceToHostBytes);
                AssertEncoded(
                    Read(parameter.T.EnsureCudaBfp8Buffer(0)),
                    Read(parameter.T.EnsureCudaBfp8Buffer(1)));
                var primary = optimizer.GetCudaBfp8Moments(0, 0);
                var secondary = optimizer.GetCudaBfp8Moments(0, 1);
                AssertEncoded(Read(primary.Fast), Read(secondary.Fast));
                AssertEncoded(Read(primary.Slow), Read(secondary.Slow));
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
    public void ScratchIsMaximumLeafBoundAndAllNativeStateIsFreed()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            int[] lengths = [5, 257, 31];
            Parameter[] parameters = lengths.Select((length, index) =>
                CreateParameter(
                    Values(length, 3 + index * 17, 0.4f),
                    [1, length],
                    $"hidden.{index}"))
                .ToArray();
            var optimizer = new NekoMuon(
                parameters, Options(runNewtonSchulz: false));
            try
            {
                foreach ((Parameter parameter, int index)
                    in parameters.Select((parameter, index) =>
                        (parameter, index)))
                {
                    PublishGradient(
                        parameter,
                        Values(parameter.T.Numel, 11 + index * 19, 0.05f),
                        0);
                }
                NativeCudaAllocationTelemetry before =
                    NativeCudaRuntime.AllocationTelemetry;

                optimizer.step();

                NativeCudaAllocationTelemetry allocated =
                    NativeCudaRuntime.AllocationTelemetry - before;
                int maximumLength = lengths.Max();
                int maximumGramLength = 1;
                long baseScratch = checked((
                    2L * maximumLength + 2L * maximumGramLength)
                    * optimizer.CudaBatchCapacity * sizeof(float));
                long bfp8Scratch = 4L * maximumLength * sizeof(float);
                long stateBytes = lengths.Sum(length =>
                    2L * (length + sizeof(float)) + 5L * sizeof(float));
                long statsBatchBytes = lengths.Length * (
                    IntPtr.Size + 4L * sizeof(float));
                long expectedBytes = baseScratch + bfp8Scratch + stateBytes
                    + statsBatchBytes + sizeof(int);
                Assert.Equal(
                    6L * lengths.Length + 11,
                    allocated.AllocationCount);
                Assert.Equal(expectedBytes, allocated.AllocationBytes);

                NativeCudaAllocationTelemetry beforeDispose =
                    NativeCudaRuntime.AllocationTelemetry;
                optimizer.DisposeCudaResources();
                NativeCudaAllocationTelemetry freed =
                    NativeCudaRuntime.AllocationTelemetry - beforeDispose;
                Assert.Equal(allocated.AllocationCount, freed.FreeCount);
                Assert.Equal(allocated.AllocationBytes, freed.FreeBytes);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                foreach (Parameter parameter in parameters)
                    parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void BinaryCheckpointResumeMatchesUninterruptedNextStep()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            NekoMuonOptions options = Options(runNewtonSchulz: false);
            Parameter source = CreateParameter(
                Values(129, 3, 0.55f), [3, 43], "hidden");
            var uninterrupted = new NekoMuon([source], options);
            NekoMuon? resumed = null;
            Parameter? restored = null;
            try
            {
                PublishGradient(source, Values(129, 23, 0.08f), 0);
                uninterrupted.step();
                float[] restoredData = source.T.Data.ToArray();
                using var checkpoint = new MemoryStream();
                OptimizerStateStream.SaveStateBinary(
                    uninterrupted, checkpoint);

                restored = CreateParameter(
                    restoredData, [3, 43], "hidden");
                resumed = new NekoMuon([restored], options);
                checkpoint.Position = 0;
                OptimizerStateStream.LoadStateBinary(resumed, checkpoint);

                float[] nextGradient = Values(129, 47, 0.06f);
                uninterrupted.zero_grad();
                resumed.zero_grad();
                PublishGradient(source, nextGradient, 0);
                PublishGradient(restored, nextGradient, 0);
                uninterrupted.step();
                resumed.step();

                AssertEncoded(
                    Read(source.T.EnsureCudaBfp8Buffer(0)),
                    Read(restored.T.EnsureCudaBfp8Buffer(0)));
                var expected = uninterrupted.GetCudaBfp8Moments(0, 0);
                var actual = resumed.GetCudaBfp8Moments(0, 0);
                AssertEncoded(Read(expected.Fast), Read(actual.Fast));
                AssertEncoded(Read(expected.Slow), Read(actual.Slow));
                Assert.Equal(
                    uninterrupted.CaptureState().ParameterStates[0].Confidence,
                    resumed.CaptureState().ParameterStates[0].Confidence);
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

    private static NekoMuonOptions Options(bool runNewtonSchulz) => new()
    {
        LearningRate = 0.002f,
        BetaFast = 0.8f,
        BetaSlow = 0.95f,
        Rho = 0.7f,
        Epsilon = 1e-6f,
        MaxNewtonSchulzSteps = 5,
        NewtonSchulzInterval = runNewtonSchulz ? 1 : 100,
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
            WeightDecayPolicy.Apply);
        parameter.T.ConvertStorageInPlace(
            TensorDType.Bfp8,
            Bfp8QuantizationDescriptor.TensorWide,
            preserveFloat32Master: false);
        foreach (int device in devices ?? [0])
            parameter.T.EnsureCudaBfp8Buffer(device);
        parameter.T.to(new TorchDevice(TensorDevice.Cuda, devices?[0] ?? 0));
        return parameter;
    }

    private static Bfp8EncodedStorage PublishGradient(
        Parameter parameter,
        float[] values,
        int device)
    {
        parameter.T.SetCudaGradient(values, device);
        return Read(parameter.T.PublishCudaBfp8Gradient(device));
    }

    private static void ReduceGradient(
        CudaBfp8GradientAllReducePlan reducer,
        Parameter parameter,
        float[] first,
        float[] second)
    {
        long stepId = reducer.BeginStep();
        reducer.BeginDeviceStep(stepId, 0);
        reducer.BeginDeviceStep(stepId, 1);
        parameter.T.SetCudaGradient(first, 0);
        reducer.NotifyGradientReady(parameter.T, 0, stepId);
        parameter.T.SetCudaGradient(second, 1);
        reducer.NotifyGradientReady(parameter.T, 1, stepId);
        reducer.Complete(stepId);
    }

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index => index == length - 1
                ? scale * 2.75f
                : MathF.Sin((index + offset) * 0.173f) * scale)
            .ToArray();

    private static Bfp8EncodedStorage Encode(float[] values)
        => Bfp8QuantizationCodec.Default.Encode(
            values,
            Bfp8QuantizationDescriptor.TensorWide);

    private static float[] Decode(Bfp8EncodedStorage encoded)
    {
        var result = new float[encoded.Count];
        Bfp8QuantizationCodec.Default.Decode(
            encoded.Payload.Span,
            encoded.Scales.Span,
            encoded.Descriptor,
            result);
        return result;
    }

    private static Bfp8EncodedStorage Read(CudaBfp8BufferView view)
    {
        var payload = new sbyte[view.Payload.Length];
        var scales = new float[view.Scales.Length];
        view.Payload.CopyToCPU(payload);
        view.Scales.CopyToCPU(scales);
        return new Bfp8EncodedStorage(payload, scales, view.Descriptor);
    }

    private static float ReadScale(CudaBfp8BufferView view)
    {
        var scale = new float[1];
        view.Scales.CopyToCPU(scale);
        return scale[0];
    }

    private static void AssertEncoded(
        Bfp8EncodedStorage expected,
        Bfp8EncodedStorage actual)
    {
        Assert.Equal(expected.Descriptor, actual.Descriptor);
        Assert.Equal(expected.Payload.ToArray(), actual.Payload.ToArray());
        Assert.Equal(expected.Scales.ToArray(), actual.Scales.ToArray());
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
                    PrecisionPolicy.Bfp8);
            action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }
}
