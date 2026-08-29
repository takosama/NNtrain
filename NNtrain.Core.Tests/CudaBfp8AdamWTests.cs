using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaBfp8AdamWTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(257)]
    [InlineData(515)]
    public void OneGpuUpdateKeepsParameterGradientAndMomentsPureBfp8(
        int length)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            float[] values = Values(length, 7, 0.8f);
            float[] gradient = Values(length, 31, 0.17f);
            Parameter parameter = CreateParameter(values, [length], "weight");
            var options = Options();
            var optimizer = new AdamW([parameter], options);
            try
            {
                Bfp8EncodedStorage initial = Read(
                    parameter.T.EnsureCudaBfp8Buffer(0));
                parameter.T.SetCudaGradient(gradient, 0);
                Bfp8EncodedStorage encodedGradient = Read(
                    parameter.T.PublishCudaBfp8Gradient(0));

                optimizer.step();

                (Bfp8EncodedStorage expectedData,
                    Bfp8EncodedStorage expectedFirst,
                    Bfp8EncodedStorage expectedSecond) = ReferenceStep(
                        initial,
                        encodedGradient,
                        Zero(length),
                        Zero(length),
                        options,
                        step: 1);
                AssertEncoded(
                    expectedData,
                    Read(parameter.T.EnsureCudaBfp8Buffer(0)));
                (CudaBfp8BufferView first, CudaBfp8BufferView second) =
                    optimizer.GetCudaBfp8Moments(0, 0);
                AssertEncoded(expectedFirst, Read(first));
                AssertEncoded(expectedSecond, Read(second));
                Assert.Equal(
                    Bfp8QuantizationDescriptor.TensorWide,
                    first.Descriptor);
                Assert.Equal(
                    Bfp8QuantizationDescriptor.TensorWide,
                    second.Descriptor);
                Assert.True(ReadScale(first) > 0f);
                Assert.True(ReadScale(second) > 0f);
                Assert.False(parameter.T.HasCudaMasterFloat32Buffer(0));
                Assert.True(parameter.T.HasAuthoritativeCudaBfp8Gradient);
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void WarmStepTransfersOnlyOneFiniteStatusScalarPerGpu()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Parameter parameter = CreateParameter(
                Values(257, 5, 0.6f), [257], "weight");
            var optimizer = new AdamW([parameter], Options());
            try
            {
                PublishGradient(parameter, Values(257, 13, 0.12f), 0);
                optimizer.step();
                optimizer.zero_grad();
                PublishGradient(parameter, Values(257, 29, 0.08f), 0);
                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;
                NativeCudaAllocationTelemetry allocationBefore =
                    NativeCudaRuntime.AllocationTelemetry;

                optimizer.step();

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - before;
                NativeCudaAllocationTelemetry allocation =
                    NativeCudaRuntime.AllocationTelemetry - allocationBefore;
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(sizeof(int), transfer.DeviceToHostBytes);
                Assert.Equal(0, allocation.AllocationCount);
                Assert.Equal(0, allocation.FreeCount);
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
    public void SparseMomentAfterOutlierUsesQuantizationFloorAndCannotExplode()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            const int length = 257;
            Parameter parameter = CreateParameter(
                Enumerable.Repeat(0.02f, length).ToArray(),
                [length],
                "sparse.weight");
            var options = new AdamWOptions
            {
                LearningRate = 3e-4f,
                Beta1 = 0.9f,
                Beta2 = 0.999f,
                Epsilon = 1e-8f,
                WeightDecay = 0f,
                Decay1D = true,
            };
            var optimizer = new AdamW([parameter], options);
            try
            {
                // The final outlier sets the tensor-wide second-moment scale.
                // Other first moments remain representable while their
                // squared moments quantize to code zero.  The legacy path
                // divided these first moments by epsilon on the zero-gradient
                // step and moved weights by O(10^2).
                float[] firstGradient = Enumerable.Repeat(0.01f, length)
                    .ToArray();
                firstGradient[^1] = 1f;
                PublishGradient(parameter, firstGradient, 0);
                optimizer.step();

                float maximumBeforeSparseStep = Decode(Read(
                    parameter.T.EnsureCudaBfp8Buffer(0)))
                    .Max(MathF.Abs);
                optimizer.zero_grad();
                PublishGradient(parameter, new float[length], 0);
                NativeCudaTransferTelemetry transferBefore =
                    NativeCudaRuntime.TransferTelemetry;

                optimizer.step();

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - transferBefore;
                float[] actual = Decode(Read(
                    parameter.T.EnsureCudaBfp8Buffer(0)));
                var moments = optimizer.GetCudaBfp8Moments(0, 0);
                Bfp8EncodedStorage first = Read(moments.First);
                Bfp8EncodedStorage second = Read(moments.Second);
                Assert.Equal(0, second.Payload.Span[0]);
                Assert.NotEqual(0, first.Payload.Span[0]);
                Assert.All(actual, value => Assert.True(float.IsFinite(value)));
                Assert.True(actual.Max(MathF.Abs) < 0.1f);
                Assert.True(MathF.Abs(actual.Max(MathF.Abs)
                    - maximumBeforeSparseStep) < 0.02f);
                Assert.True(second.Scales.Span[0] > 0f);
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(sizeof(int), transfer.DeviceToHostBytes);
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
    public void TwoGpuUpdateUsesIdenticalBfp8ReplicasAndNoPayloadTransfer()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithCuda([0, 1], () =>
        {
            Parameter parameter = CreateParameter(
                Values(257, 3, 0.7f), [257], "weight", [0, 1]);
            var optimizer = new AdamW([parameter], Options());
            using var reducer = new CudaBfp8GradientAllReducePlan(
                [parameter], [0, 1]);
            try
            {
                long stepId = reducer.BeginStep();
                reducer.BeginDeviceStep(stepId, 0);
                reducer.BeginDeviceStep(stepId, 1);
                parameter.T.SetCudaGradient(Values(257, 17, 0.11f), 0);
                reducer.NotifyGradientReady(parameter.T, 0, stepId);
                parameter.T.SetCudaGradient(Values(257, 43, 0.09f), 1);
                reducer.NotifyGradientReady(parameter.T, 1, stepId);
                reducer.Complete(stepId);
                optimizer.step();
                optimizer.zero_grad();

                long warmStepId = reducer.BeginStep();
                reducer.BeginDeviceStep(warmStepId, 0);
                reducer.BeginDeviceStep(warmStepId, 1);
                parameter.T.SetCudaGradient(Values(257, 53, 0.07f), 0);
                reducer.NotifyGradientReady(parameter.T, 0, warmStepId);
                parameter.T.SetCudaGradient(Values(257, 71, 0.06f), 1);
                reducer.NotifyGradientReady(parameter.T, 1, warmStepId);
                reducer.Complete(warmStepId);
                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;

                optimizer.step();

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - before;
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(2 * sizeof(int), transfer.DeviceToHostBytes);
                AssertEncoded(
                    Read(parameter.T.EnsureCudaBfp8Buffer(0)),
                    Read(parameter.T.EnsureCudaBfp8Buffer(1)));
                var primary = optimizer.GetCudaBfp8Moments(0, 0);
                var secondary = optimizer.GetCudaBfp8Moments(0, 1);
                AssertEncoded(Read(primary.First), Read(secondary.First));
                AssertEncoded(Read(primary.Second), Read(secondary.Second));
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
    public void MultiTensorPlanUsesOnlyScalarReductionStateAndFreesAll()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            int[] lengths = [5, 257, 31];
            Parameter[] parameters = lengths.Select((length, index) =>
                CreateParameter(
                    Values(length, index * 11 + 2, 0.4f),
                    [length],
                    $"weight.{index}"))
                .ToArray();
            var optimizer = new AdamW(parameters, Options());
            try
            {
                var expected = new (Bfp8EncodedStorage Data,
                    Bfp8EncodedStorage First,
                    Bfp8EncodedStorage Second)[parameters.Length];
                foreach ((Parameter parameter, int index)
                    in parameters.Select((parameter, index) =>
                        (parameter, index)))
                {
                    Bfp8EncodedStorage initial = Read(
                        parameter.T.EnsureCudaBfp8Buffer(0));
                    PublishGradient(
                        parameter,
                        Values(parameter.T.Numel, index * 13 + 7, 0.05f),
                        0);
                    Assert.True(parameter.T.TryGetCudaBfp8GradientBuffer(
                        0, out CudaBfp8BufferView gradient));
                    expected[index] = ReferenceStep(
                        initial,
                        Read(gradient),
                        Zero(parameter.T.Numel),
                        Zero(parameter.T.Numel),
                        Options(),
                        step: 1);
                }
                NativeCudaAllocationTelemetry before =
                    NativeCudaRuntime.AllocationTelemetry;

                optimizer.step();

                NativeCudaAllocationTelemetry allocated =
                    NativeCudaRuntime.AllocationTelemetry - before;
                long expectedStateBytes = lengths.Sum(
                    length => 2L * (length + sizeof(float)));
                long expectedPlanBytes = lengths.Length * (
                    System.Runtime.InteropServices.Marshal.SizeOf<
                        CudaOptimizerNative.AdamWBfp8TensorDescriptor>()
                    + 6L * sizeof(float));
                long expectedBytes = expectedStateBytes
                    + expectedPlanBytes
                    + sizeof(int);
                Assert.Equal(4L * lengths.Length + 3, allocated.AllocationCount);
                Assert.Equal(expectedBytes, allocated.AllocationBytes);
                for (int index = 0; index < parameters.Length; index++)
                {
                    AssertEncodedWithinScaleUlps(
                        expected[index].Data,
                        Read(parameters[index].T.EnsureCudaBfp8Buffer(0)));
                    var moments = optimizer.GetCudaBfp8Moments(index, 0);
                    AssertEncodedWithinScaleUlps(
                        expected[index].First, Read(moments.First));
                    AssertEncodedWithinScaleUlps(
                        expected[index].Second, Read(moments.Second));
                }

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
    public void BinaryCheckpointResumeMatchesUninterruptedNextUpdate()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            Parameter source = CreateParameter(
                Values(129, 2, 0.75f), [3, 43], "weight");
            var uninterrupted = new AdamW([source], Options());
            AdamW? resumed = null;
            Parameter? restored = null;
            try
            {
                PublishGradient(source, Values(129, 19, 0.13f), 0);
                uninterrupted.step();
                float[] restoredData = source.T.Data.ToArray();
                using var checkpoint = new MemoryStream();
                OptimizerStateStream.SaveStateBinary(
                    uninterrupted, checkpoint);

                restored = CreateParameter(
                    restoredData, [3, 43], "weight");
                resumed = new AdamW([restored], Options());
                checkpoint.Position = 0;
                OptimizerStateStream.LoadStateBinary(resumed, checkpoint);

                float[] nextGradient = Values(129, 47, 0.07f);
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
                AssertEncoded(Read(expected.First), Read(actual.First));
                AssertEncoded(Read(expected.Second), Read(actual.Second));
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

    private static AdamWOptions Options() => new()
    {
        LearningRate = 0.003f,
        Beta1 = 0.8f,
        Beta2 = 0.91f,
        Epsilon = 1e-6f,
        WeightDecay = 0.02f,
        Decay1D = true,
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

    private static void PublishGradient(
        Parameter parameter,
        float[] values,
        int device)
    {
        parameter.T.SetCudaGradient(values, device);
        parameter.T.PublishCudaBfp8Gradient(device);
    }

    private static (Bfp8EncodedStorage Data,
        Bfp8EncodedStorage First,
        Bfp8EncodedStorage Second) ReferenceStep(
        Bfp8EncodedStorage data,
        Bfp8EncodedStorage gradient,
        Bfp8EncodedStorage first,
        Bfp8EncodedStorage second,
        AdamWOptions options,
        int step)
    {
        float[] dataValues = Decode(data);
        float[] gradientValues = Decode(gradient);
        float[] firstValues = Decode(first);
        float[] secondValues = Decode(second);
        float bc1 = 1f - MathF.Pow(options.Beta1, step);
        float bc2 = 1f - MathF.Pow(options.Beta2, step);
        float sqrtBc2 = MathF.Sqrt(bc2);
        float updateScale = options.LearningRate * sqrtBc2 / bc1;
        float scaledEpsilon = options.Epsilon * sqrtBc2;
        for (int index = 0; index < dataValues.Length; index++)
        {
            float g = gradientValues[index];
            firstValues[index] = options.Beta1 * firstValues[index]
                + (1f - options.Beta1) * g;
            secondValues[index] = options.Beta2 * secondValues[index]
                + (1f - options.Beta2) * g * g;
        }
        Bfp8EncodedStorage encodedFirst = Encode(firstValues);
        Bfp8EncodedStorage encodedSecond = Encode(secondValues);
        firstValues = Decode(encodedFirst);
        secondValues = Decode(encodedSecond);
        float varianceFloor = 0.5f * encodedSecond.Scales.Span[0];
        for (int index = 0; index < dataValues.Length; index++)
        {
            dataValues[index] *= 1f
                - options.LearningRate * options.WeightDecay;
            dataValues[index] -= updateScale * firstValues[index]
                / (MathF.Sqrt(MathF.Max(
                    secondValues[index], varianceFloor)) + scaledEpsilon);
        }
        return (Encode(dataValues), encodedFirst, encodedSecond);
    }

    private static Bfp8EncodedStorage Zero(int length)
        => Encode(new float[length]);

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

    private static void AssertEncodedWithinScaleUlps(
        Bfp8EncodedStorage expected,
        Bfp8EncodedStorage actual)
    {
        Assert.Equal(expected.Descriptor, actual.Descriptor);
        Assert.Equal(expected.Payload.ToArray(), actual.Payload.ToArray());
        Assert.Equal(expected.Scales.Length, actual.Scales.Length);
        for (int index = 0; index < expected.Scales.Length; index++)
        {
            float expectedScale = expected.Scales.Span[index];
            float actualScale = actual.Scales.Span[index];
            Assert.True(expectedScale > 0f && actualScale > 0f);
            int ulps = Math.Abs(
                BitConverter.SingleToInt32Bits(expectedScale)
                - BitConverter.SingleToInt32Bits(actualScale));
            Assert.True(
                ulps <= 1,
                $"Scale {index} differs by {ulps} ULPs: "
                + $"expected {expectedScale:R}, actual {actualScale:R}.");
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
