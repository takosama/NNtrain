using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaBfp8GradientClipNormTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OneGpuGraphPublicationCachesQuantizedNormAndClipsScaleOnly(
        bool shouldClip)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            float[] firstSource = Values(515, 17, 2.25f);
            float[] secondSource = Values(257, 61, 0.75f);
            Parameter first = CreateParameter(firstSource.Length, [0]);
            Parameter second = CreateParameter(secondSource.Length, [0]);
            try
            {
                first.T.SetCudaGradient(firstSource, 0);
                second.T.SetCudaGradient(secondSource, 0);
                first.T.PrepareCudaBfp8GradientReplica(0);
                second.T.PrepareCudaBfp8GradientReplica(0);
                Bfp8EncodedStorage firstExpected = Encode(firstSource);
                Bfp8EncodedStorage secondExpected = Encode(secondSource);
                float expectedNorm = Norm(
                    Decode(firstExpected).Concat(Decode(secondExpected)));
                float maxNorm = shouldClip
                    ? expectedNorm * 0.375f
                    : expectedNorm * 2f + 1f;
                float multiplier = shouldClip
                    ? maxNorm / (expectedNorm + 1e-6f)
                    : 1f;

                NativeCudaAllocationTelemetry allocationBefore =
                    NativeCudaRuntime.AllocationTelemetry;
                NativeCudaTransferTelemetry transferBefore =
                    NativeCudaRuntime.TransferTelemetry;
                using (CudaBfp8GradientPublicationScope scope =
                    Assert.IsType<CudaBfp8GradientPublicationScope>(
                        CudaBfp8GradientPublicationScope.TryCreate(
                            [first.T, second.T])))
                {
                    Assert.True(
                        CudaBfp8GradientPublicationScope.TryPublish(
                            first.T));
                    Assert.True(
                        CudaBfp8GradientPublicationScope.TryPublish(
                            second.T));
                    scope.Complete(publish: true);

                    float actualNorm = nn.utils.clip_grad_norm_(
                        [first, second],
                        maxNorm);
                    AssertNorm(expectedNorm, actualNorm);
                }
                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - transferBefore;
                NativeCudaAllocationTelemetry allocation =
                    NativeCudaRuntime.AllocationTelemetry - allocationBefore;

                // One graph-wide finite int and one quantized squared-norm
                // double. clip_grad_norm_ consumes the cache with no D2H.
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(sizeof(int) + sizeof(double),
                    transfer.DeviceToHostBytes);
                Assert.Equal(2, allocation.AllocationCount);
                Assert.Equal(allocation.AllocationCount, allocation.FreeCount);
                Assert.Equal(allocation.AllocationBytes, allocation.FreeBytes);
                AssertGradient(
                    first.T,
                    0,
                    firstExpected.Payload.Span,
                    firstExpected.Scales.Span[0] * multiplier);
                AssertGradient(
                    second.T,
                    0,
                    secondExpected.Payload.Span,
                    secondExpected.Scales.Span[0] * multiplier);
                Assert.True(first.T.HasAuthoritativeCudaBfp8Gradient);
                Assert.True(second.T.HasAuthoritativeCudaBfp8Gradient);
            }
            finally
            {
                first.T.InvalidateCudaBuffers();
                second.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void OneGpuCacheMissReducesResidentPayloadAndFreesScratch()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda([0], () =>
        {
            float[] source = Values(257, 31, 3.125f);
            Parameter parameter = CreateParameter(source.Length, [0]);
            try
            {
                parameter.T.SetCudaGradient(source, 0);
                Bfp8EncodedStorage expected = Encode(source);
                _ = parameter.T.PublishCudaBfp8Gradient(0);
                float expectedNorm = Norm(Decode(expected));
                float maxNorm = expectedNorm * 0.5f;
                float multiplier = maxNorm / (expectedNorm + 1e-6f);
                NativeCudaTransferTelemetry transferBefore =
                    NativeCudaRuntime.TransferTelemetry;
                NativeCudaAllocationTelemetry allocationBefore =
                    NativeCudaRuntime.AllocationTelemetry;

                float actualNorm = nn.utils.clip_grad_norm_(
                    [parameter],
                    maxNorm);

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - transferBefore;
                NativeCudaAllocationTelemetry allocation =
                    NativeCudaRuntime.AllocationTelemetry - allocationBefore;
                AssertNorm(expectedNorm, actualNorm);
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(sizeof(int) + sizeof(double),
                    transfer.DeviceToHostBytes);
                Assert.Equal(2, allocation.AllocationCount);
                Assert.Equal(allocation.AllocationCount, allocation.FreeCount);
                Assert.Equal(allocation.AllocationBytes, allocation.FreeBytes);
                AssertGradient(
                    parameter.T,
                    0,
                    expected.Payload.Span,
                    expected.Scales.Span[0] * multiplier);
                Assert.True(parameter.T.HasAuthoritativeCudaBfp8Gradient);
            }
            finally
            {
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TwoGpuReducerCacheClipsEveryReplicaScaleWithoutTransfers(
        bool shouldClip)
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithCuda([0, 1], () =>
        {
            float[] first = Values(515, 7, 1.75f);
            float[] second = Values(515, 43, 0.875f);
            Parameter parameter = CreateParameter(first.Length, [0, 1]);
            using var reducer = new CudaBfp8GradientAllReducePlan(
                [parameter],
                [0, 1]);
            try
            {
                long stepId = reducer.BeginStep();
                reducer.BeginDeviceStep(stepId, 0);
                reducer.BeginDeviceStep(stepId, 1);
                parameter.T.SetCudaGradient(first, 0);
                reducer.NotifyGradientReady(parameter.T, 0, stepId);
                parameter.T.SetCudaGradient(second, 1);
                reducer.NotifyGradientReady(parameter.T, 1, stepId);
                reducer.Complete(stepId);

                Bfp8EncodedStorage expected = Reduced(first, second);
                float expectedNorm = Norm(Decode(expected));
                float maxNorm = shouldClip
                    ? expectedNorm * 0.625f
                    : expectedNorm * 2f + 1f;
                float multiplier = shouldClip
                    ? maxNorm / (expectedNorm + 1e-6f)
                    : 1f;
                NativeCudaTransferTelemetry transferBefore =
                    NativeCudaRuntime.TransferTelemetry;
                NativeCudaAllocationTelemetry allocationBefore =
                    NativeCudaRuntime.AllocationTelemetry;

                float actualNorm = nn.utils.clip_grad_norm_(
                    [parameter],
                    maxNorm);

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - transferBefore;
                NativeCudaAllocationTelemetry allocation =
                    NativeCudaRuntime.AllocationTelemetry - allocationBefore;
                AssertNorm(expectedNorm, actualNorm);
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(0, transfer.DeviceToHostBytes);
                Assert.Equal(0, allocation.AllocationCount);
                Assert.Equal(0, allocation.FreeCount);
                foreach (int device in new[] { 0, 1 })
                {
                    AssertGradient(
                        parameter.T,
                        device,
                        expected.Payload.Span,
                        expected.Scales.Span[0] * multiplier);
                }
                Assert.True(parameter.T.HasAuthoritativeCudaBfp8Gradient);
            }
            finally
            {
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    private static Parameter CreateParameter(
        int length,
        IReadOnlyList<int> devices)
    {
        var parameter = new Parameter(
            new float[length],
            [length],
            "clip.weight",
            WeightDecayPolicy.Apply);
        parameter.T.ConvertStorageInPlace(
            TensorDType.Bfp8,
            Bfp8QuantizationDescriptor.TensorWide,
            preserveFloat32Master: false);
        parameter.T.to(new TorchDevice(TensorDevice.Cuda, devices[0]));
        for (int index = 1; index < devices.Count; index++)
            parameter.T.EnsureCudaBfp8Buffer(devices[index]);
        return parameter;
    }

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index => index == length - 1
                ? scale * 3.25f
                : MathF.Sin((index + offset) * 0.097f) * scale)
            .ToArray();

    private static Bfp8EncodedStorage Encode(float[] values)
        => Bfp8QuantizationCodec.Default.Encode(
            values,
            Bfp8QuantizationDescriptor.TensorWide);

    private static Bfp8EncodedStorage Reduced(
        float[] first,
        float[] second)
    {
        float[] left = Decode(Encode(first));
        float[] right = Decode(Encode(second));
        return Encode(left.Zip(
            right,
            static (leftValue, rightValue) =>
                leftValue + rightValue).ToArray());
    }

    private static float[] Decode(Bfp8EncodedStorage encoded)
    {
        var values = new float[encoded.Count];
        Bfp8QuantizationCodec.Default.Decode(
            encoded.Payload.Span,
            encoded.Scales.Span,
            encoded.Descriptor,
            values);
        return values;
    }

    private static float Norm(IEnumerable<float> values)
        => (float)Math.Sqrt(values.Sum(value => (double)value * value));

    private static void AssertNorm(float expected, float actual)
    {
        float tolerance = MathF.Max(1e-5f, MathF.Abs(expected) * 2e-6f);
        Assert.InRange(MathF.Abs(expected - actual), 0f, tolerance);
    }

    private static void AssertGradient(
        Tensor tensor,
        int device,
        ReadOnlySpan<sbyte> expectedPayload,
        float expectedScale)
    {
        Assert.True(tensor.TryGetCudaBfp8GradientBuffer(
            device,
            out CudaBfp8BufferView actual));
        var payload = new sbyte[expectedPayload.Length];
        var scale = new float[1];
        actual.Payload.CopyToCPU(payload);
        actual.Scales.CopyToCPU(scale);
        Assert.Equal(expectedPayload.ToArray(), payload);
        float tolerance = MathF.Max(
            1e-8f,
            MathF.Abs(expectedScale) * 2e-6f);
        Assert.InRange(
            MathF.Abs(expectedScale - scale[0]),
            0f,
            tolerance);
    }

    private static void WithCuda(
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
                    PrecisionPolicy.Bfp8);
            action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }
}
