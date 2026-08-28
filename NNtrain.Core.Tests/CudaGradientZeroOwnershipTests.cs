using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaGradientZeroOwnershipTests
{
    [Fact]
    public void SingleGpuWithoutReducerKeepsPerTensorPhysicalClear()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        WithCudaDevices([0], PrecisionPolicy.Float32, () =>
        {
            var parameter = new Parameter(
                new float[8],
                [8],
                "single.weight",
                WeightDecayPolicy.Apply);
            parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
            parameter.T.SetCudaGradient(
                Enumerable.Repeat(7f, 8).ToArray(),
                0);
            var optimizer = new AdamW([parameter]);

            NativeCudaMemsetTelemetry before =
                NativeCudaRuntime.MemsetTelemetry;
            optimizer.zero_grad();
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
            NativeCudaMemsetTelemetry delta =
                NativeCudaRuntime.MemsetTelemetry - before;

            Assert.Equal(1, delta.LaunchCount);
            Assert.Equal(8 * sizeof(float), delta.Bytes);
            Assert.All(parameter.T.Grad, value => Assert.Equal(0f, value));
            parameter.T.InvalidateCudaBuffers();
        });
    }

    [Fact]
    public void BFloat16ReducerZeroGradIsLogicalAndWorkerClearsOneArenaPerDevice()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithCudaDevices([0, 1], PrecisionPolicy.BFloat16, () =>
        {
            Parameter left = CreateBFloat16Parameter("left", 5);
            Parameter right = CreateBFloat16Parameter("right", 7);
            using var plan = new CudaBFloat16GradientAllReducePlan(
                [left, right],
                [0, 1]);
            var optimizer = new AdamW([left, right]);
            SeedBothDevices(left, 9f);
            SeedBothDevices(right, 11f);

            NativeCudaMemsetTelemetry beforeZero =
                NativeCudaRuntime.MemsetTelemetry;
            optimizer.zero_grad();
            NativeCudaMemsetTelemetry logicalZero =
                NativeCudaRuntime.MemsetTelemetry - beforeZero;
            Assert.Equal(0, logicalZero.LaunchCount);
            Assert.Equal(0, logicalZero.Bytes);
            Assert.All(left.T.Grad, value => Assert.Equal(0f, value));
            Assert.All(right.T.Grad, value => Assert.Equal(0f, value));

            long stepId = plan.BeginStep();
            NativeCudaMemsetTelemetry beforeWorkers =
                NativeCudaRuntime.MemsetTelemetry;
            BeginDevice(plan, stepId, 0);
            BeginDevice(plan, stepId, 1);
            NativeCudaMemsetTelemetry workerClear =
                NativeCudaRuntime.MemsetTelemetry - beforeWorkers;

            // One FP32 arena per device plus the primary squared-sum scalar.
            Assert.Equal(3, workerClear.LaunchCount);
            Assert.Equal(
                2L * (left.T.Numel + right.T.Numel) * sizeof(float)
                    + sizeof(double),
                workerClear.Bytes);
            AssertPhysicalZero(left.T, 0);
            AssertPhysicalZero(left.T, 1);
            AssertPhysicalZero(right.T, 0);
            AssertPhysicalZero(right.T, 1);

            Publish(plan, left, stepId, 0, 1f);
            Publish(plan, right, stepId, 0, 4f);
            Publish(plan, left, stepId, 1, 2f);
            Publish(plan, right, stepId, 1, 8f);
            plan.Complete(stepId);

            Assert.All(left.T.Grad, value => Assert.Equal(3f, value));
            Assert.All(right.T.Grad, value => Assert.Equal(12f, value));
        });
    }

    [Fact]
    public void Bfp8ReducerZeroGradSkipsCoordinatorAndWorkerClearsEachAccumulatorOnce()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        WithCudaDevices([0, 1], PrecisionPolicy.Bfp8, () =>
        {
            const int length = 17;
            Parameter parameter = CreateBfp8Parameter(length);
            using var plan = new CudaBfp8GradientAllReducePlan(
                [parameter],
                [0, 1]);
            var optimizer = new AdamW([parameter]);
            parameter.T.SetCudaGradient(
                Enumerable.Repeat(5f, length).ToArray(),
                0);
            parameter.T.SetCudaGradient(
                Enumerable.Repeat(6f, length).ToArray(),
                1);

            NativeCudaMemsetTelemetry beforeZero =
                NativeCudaRuntime.MemsetTelemetry;
            optimizer.zero_grad();
            NativeCudaMemsetTelemetry logicalZero =
                NativeCudaRuntime.MemsetTelemetry - beforeZero;
            Assert.Equal(0, logicalZero.LaunchCount);
            Assert.Equal(0, logicalZero.Bytes);
            Assert.All(parameter.T.Grad, value => Assert.Equal(0f, value));

            long stepId = plan.BeginStep();
            NativeCudaMemsetTelemetry beforeWorkers =
                NativeCudaRuntime.MemsetTelemetry;
            BeginDevice(plan, stepId, 0);
            BeginDevice(plan, stepId, 1);
            NativeCudaMemsetTelemetry workerClear =
                NativeCudaRuntime.MemsetTelemetry - beforeWorkers;

            // Two finite flags, one primary squared sum and one FP32
            // accumulator per parameter/device.
            Assert.Equal(5, workerClear.LaunchCount);
            Assert.Equal(
                2L * sizeof(int) + sizeof(double)
                    + 2L * length * sizeof(float),
                workerClear.Bytes);
            AssertPhysicalZero(parameter.T, 0);
            AssertPhysicalZero(parameter.T, 1);

            float[] first = Values(length, 3, 1.25f);
            float[] second = Values(length, 29, 0.75f);
            parameter.T.SetCudaGradient(first, 0);
            plan.NotifyGradientReady(parameter.T, 0, stepId);
            parameter.T.SetCudaGradient(second, 1);
            plan.NotifyGradientReady(parameter.T, 1, stepId);
            plan.Complete(stepId);

            float[] firstLocal = QuantizeRoundTrip(first);
            float[] secondLocal = QuantizeRoundTrip(second);
            float[] reduced = firstLocal.Zip(
                secondLocal,
                static (left, right) => left + right).ToArray();
            float[] expected = QuantizeRoundTrip(reduced);
            for (int index = 0; index < expected.Length; index++)
            {
                Assert.InRange(
                    MathF.Abs(expected[index] - parameter.T.Grad[index]),
                    0f,
                    2e-6f);
            }
            Assert.True(parameter.T.HasAuthoritativeCudaBfp8Gradient);
            parameter.T.InvalidateCudaBuffers();
        });
    }

    private static Parameter CreateBFloat16Parameter(string name, int length)
        => new(
            new float[length],
            [length],
            name,
            WeightDecayPolicy.Apply,
            TensorDType.BFloat16);

    private static Parameter CreateBfp8Parameter(int length)
    {
        var parameter = new Parameter(
            new float[length],
            [length],
            "bfp8.weight",
            WeightDecayPolicy.Apply);
        parameter.T.ConvertStorageInPlace(
            TensorDType.Bfp8,
            Bfp8QuantizationDescriptor.TensorWide,
            preserveFloat32Master: false);
        parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
        _ = parameter.T.EnsureCudaBfp8Buffer(1);
        return parameter;
    }

    private static void SeedBothDevices(Parameter parameter, float value)
    {
        float[] values = Enumerable.Repeat(value, parameter.T.Numel).ToArray();
        parameter.T.SetCudaGradient(values, 0);
        parameter.T.SetCudaGradient(values, 1);
    }

    private static void BeginDevice(
        CudaBFloat16GradientAllReducePlan plan,
        long stepId,
        int deviceIndex)
    {
        using IDisposable device = TensorExecutionContext.Push(
            new TorchDevice(TensorDevice.Cuda, deviceIndex));
        plan.BeginDeviceStep(stepId, deviceIndex);
    }

    private static void BeginDevice(
        CudaBfp8GradientAllReducePlan plan,
        long stepId,
        int deviceIndex)
    {
        using IDisposable device = TensorExecutionContext.Push(
            new TorchDevice(TensorDevice.Cuda, deviceIndex));
        plan.BeginDeviceStep(stepId, deviceIndex);
    }

    private static void Publish(
        CudaBFloat16GradientAllReducePlan plan,
        Parameter parameter,
        long stepId,
        int deviceIndex,
        float value)
    {
        parameter.T.SetCudaGradient(
            Enumerable.Repeat(value, parameter.T.Numel).ToArray(),
            deviceIndex);
        plan.NotifyGradientReady(parameter.T, deviceIndex, stepId);
    }

    private static void AssertPhysicalZero(Tensor tensor, int deviceIndex)
    {
        var actual = new float[tensor.Numel];
        tensor.EnsureCudaGradientBuffer(deviceIndex).CopyToCPU(actual);
        Assert.All(actual, value => Assert.Equal(0f, value));
    }

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index => MathF.Sin((index + offset) * 0.173f) * scale)
            .ToArray();

    private static float[] QuantizeRoundTrip(float[] values)
    {
        Bfp8EncodedStorage encoded = Bfp8QuantizationCodec.Default.Encode(
            values,
            Bfp8QuantizationDescriptor.TensorWide);
        var decoded = new float[values.Length];
        Bfp8QuantizationCodec.Default.Decode(
            encoded.Payload.Span,
            encoded.Scales.Span,
            encoded.Descriptor,
            decoded);
        return decoded;
    }

    private static void WithCudaDevices(
        int[] devices,
        PrecisionPolicy precision,
        Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = devices;
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using var session = new ExecutionSession(
                new ExecutionOptions
                {
                    Device = ExecutionDeviceKind.Cuda,
                    CudaDevices = new DeviceSet(devices),
                    Precision = precision,
                },
                devices.Select(device =>
                    (IExecutionLane)CudaExecutionLaneFactory.Create(device)));
            using IDisposable execution = session.Enter();
            action();
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }
}
