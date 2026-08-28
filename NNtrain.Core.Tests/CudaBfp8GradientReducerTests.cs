using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaBfp8GradientReducerTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(257)]
    [InlineData(515)]
    public void TwoGpuReducerSumsRequantizesAndBroadcastsWithoutPayloadHostCopy(
        int length)
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithTwoCudaDevices(() =>
        {
            Parameter parameter = CreateParameter(length);
            float[] first = Values(length, 3, 1.7f);
            float[] second = Values(length, 41, 0.9f);
            using var plan = new CudaBfp8GradientAllReducePlan(
                [parameter], [0, 1]);
            long stepId = plan.BeginStep();
            plan.BeginDeviceStep(stepId, 0);
            plan.BeginDeviceStep(stepId, 1);

            parameter.T.SetCudaGradient(first, 0);
            plan.NotifyGradientReady(parameter.T, 0, stepId);
            parameter.T.SetCudaGradient(second, 1);
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;
            plan.NotifyGradientReady(parameter.T, 1, stepId);
            plan.Complete(stepId);

            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - before;
            Assert.Equal(0, transfers.HostToDeviceBytes);
            // finite-status per device + one squared-norm scalar only.
            Assert.Equal(16, transfers.DeviceToHostBytes);
            Assert.True(parameter.T.HasAuthoritativeCudaBfp8Gradient);

            Bfp8EncodedStorage firstEncoded = Encode(first);
            Bfp8EncodedStorage secondEncoded = Encode(second);
            float[] firstDecoded = Decode(firstEncoded);
            float[] secondDecoded = Decode(secondEncoded);
            float[] summed = firstDecoded.Zip(
                secondDecoded,
                static (left, right) => left + right).ToArray();
            Bfp8EncodedStorage expected = Encode(summed);

            AssertReplica(parameter.T, 0, expected);
            AssertReplica(parameter.T, 1, expected);
            parameter.T.InvalidateCudaBuffers();
        });
    }

    [Fact]
    public void DuplicateNotificationIsRejectedAndAbortAllowsNextStep()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithTwoCudaDevices(() =>
        {
            Parameter parameter = CreateParameter(33);
            using var plan = new CudaBfp8GradientAllReducePlan(
                [parameter], [0, 1]);
            long failedStep = plan.BeginStep();
            plan.BeginDeviceStep(failedStep, 0);
            plan.BeginDeviceStep(failedStep, 1);
            parameter.T.SetCudaGradient(Values(33, 2, 0.5f), 0);
            plan.NotifyGradientReady(parameter.T, 0, failedStep);
            InvalidOperationException duplicate =
                Assert.Throws<InvalidOperationException>(() =>
                    plan.NotifyGradientReady(parameter.T, 0, failedStep));
            Assert.Contains("twice", duplicate.Message);
            plan.Abort(failedStep);

            long nextStep = plan.BeginStep();
            plan.BeginDeviceStep(nextStep, 0);
            plan.BeginDeviceStep(nextStep, 1);
            for (int device = 0; device < 2; device++)
            {
                parameter.T.SetCudaGradient(new float[33], device);
                plan.NotifyGradientReady(parameter.T, device, nextStep);
            }
            plan.Complete(nextStep);
            Assert.True(parameter.T.HasAuthoritativeCudaBfp8Gradient);
            parameter.T.InvalidateCudaBuffers();
        });
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void DeviceAccumulatorNonFiniteFailsBeforeBfp8Publication(float value)
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithTwoCudaDevices(() =>
        {
            Parameter parameter = CreateParameter(17);
            using var plan = new CudaBfp8GradientAllReducePlan(
                [parameter], [0, 1]);
            long stepId = plan.BeginStep();
            plan.BeginDeviceStep(stepId, 0);
            plan.BeginDeviceStep(stepId, 1);
            for (int device = 0; device < 2; device++)
            {
                var values = new float[17];
                if (device == 1)
                    values[16] = value;
                NativeCudaBuffer<float> gradient =
                    parameter.T.EnsureCudaGradientBuffer(device);
                gradient.CopyFromCPU(values);
                parameter.T.MarkCudaGradientMutated(device);
                plan.NotifyGradientReady(parameter.T, device, stepId);
            }

            InvalidOperationException failure =
                Assert.Throws<InvalidOperationException>(
                    () => plan.Complete(stepId));
            Assert.Contains("Non-finite", failure.Message);
            Assert.False(parameter.T.HasAuthoritativeCudaBfp8Gradient);
            parameter.T.InvalidateCudaBuffers();
        });
    }

    [Fact]
    public void ExecutionSessionCommunicationStreamsAreBorrowedAndNotDestroyed()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            using var session = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(0, 1),
                    Precision = PrecisionPolicy.Bfp8,
                },
                [
                    CudaExecutionLaneFactory.Create(0),
                    CudaExecutionLaneFactory.Create(1),
                ]);
            using IDisposable scope = session.Enter();
            Parameter parameter = CreateParameter(8);
            using (var plan = new CudaBfp8GradientAllReducePlan(
                [parameter], [0, 1]))
            {
                Assert.False(plan.OwnsCommunicationStream(0));
                Assert.False(plan.OwnsCommunicationStream(1));
            }

            // The borrowed handles remain lane-owned after plan disposal.
            foreach (int device in new[] { 0, 1 })
            {
                var lane = Assert.IsType<CudaExecutionLane>(
                    session.GetRequiredLane(ExecutionDeviceKind.Cuda, device));
                Assert.NotEqual(nint.Zero, lane.CommunicationStreamHandle);
                lane.SynchronizeCommunicationStream();
            }
            parameter.T.InvalidateCudaBuffers();
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    private static Parameter CreateParameter(int length)
    {
        var parameter = new Parameter(
            new float[length],
            [length],
            "bfp.weight",
            WeightDecayPolicy.Apply);
        parameter.T.ConvertStorageInPlace(
            TensorDType.Bfp8,
            Bfp8QuantizationDescriptor.TensorWide,
            preserveFloat32Master: false);
        parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
        parameter.T.EnsureCudaBfp8Buffer(1);
        return parameter;
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

    private static void AssertReplica(
        Tensor tensor,
        int device,
        Bfp8EncodedStorage expected)
    {
        Assert.True(tensor.TryGetCudaBfp8GradientBuffer(
            device, out CudaBfp8BufferView actual));
        var payload = new sbyte[expected.Count];
        var scale = new float[1];
        actual.Payload.CopyToCPU(payload);
        actual.Scales.CopyToCPU(scale);
        Assert.Equal(expected.Payload.ToArray(), payload);
        Assert.InRange(
            MathF.Abs(expected.Scales.Span[0] - scale[0]),
            0f,
            1e-7f);
    }

    private static void WithTwoCudaDevices(Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0, 1];
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
