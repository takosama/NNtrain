using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

[Collection(TensorSimdCollection.Name)]
public sealed class CudaOptimizerResidencyTests
{
    [Theory]
    [InlineData(true, false, false, 1)]
    [InlineData(false, true, false, 1)]
    [InlineData(true, false, true, 1)]
    [InlineData(false, true, true, 1)]
    [InlineData(true, false, false, 2)]
    [InlineData(false, true, true, 2)]
    public void AsymmetricAdamWMomentsStayResidentWithoutHotTransfers(
        bool firstMomentBFloat16,
        bool secondMomentBFloat16,
        bool mix16,
        int deviceCount)
    {
        if (Tensor.CudaDeviceCount < deviceCount)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        Parameter? gpuParameter = null;
        AdamW? gpuOptimizer = null;
        try
        {
            int[] devices = Enumerable.Range(0, deviceCount).ToArray();
            PrecisionPolicy policy = mix16
                ? PrecisionPolicy.Mix16_32
                : PrecisionPolicy.Float32;
            AdamWOptions options = new()
            {
                LearningRate = 3e-4f,
                Beta1 = 0.9f,
                Beta2 = 0.95f,
                Epsilon = 1e-8f,
                WeightDecay = 0.01f,
                UseBFloat16FirstMoment = firstMomentBFloat16,
                UseBFloat16SecondMoment = secondMomentBFloat16,
            };
            const int length = 8193;
            float[] initial = Values(length, 5);
            float[] firstGradient = Values(length, 37);
            float[] secondGradient = Values(length, 83);
            float[] thirdGradient = Values(length, 131);

            Tensor.ExecutionDevice = TensorDevice.Cpu;
            float[] expectedData;
            AdamWState expectedState;
            float[] expectedRestoredData;
            AdamWState expectedRestoredState;
            using (TensorExecutionContext.PushPrecisionPolicy(policy))
            {
                Parameter cpuParameter = CreateParameter(
                    "mixed.cpu", length, [length]);
                initial.CopyTo(cpuParameter.DataBuffer, 0);
                if (mix16)
                {
                    cpuParameter.T.ConvertStorageInPlace(
                        TensorDType.BFloat16,
                        preserveFloat32Master: true);
                }
                var cpuOptimizer = new AdamW([cpuParameter], options);
                firstGradient.CopyTo(cpuParameter.T.MutableGrad);
                cpuOptimizer.step();
                cpuOptimizer.zero_grad();
                secondGradient.CopyTo(cpuParameter.T.MutableGrad);
                cpuOptimizer.step();
                expectedData = cpuParameter.DataBuffer.ToArray();
                expectedState = cpuOptimizer.CaptureState();
                cpuOptimizer.zero_grad();
                thirdGradient.CopyTo(cpuParameter.T.MutableGrad);
                cpuOptimizer.step();
                expectedRestoredData = cpuParameter.DataBuffer.ToArray();
                expectedRestoredState = cpuOptimizer.CaptureState();
            }

            Tensor.CudaDeviceIndices = devices;
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(policy);
            gpuParameter = CreateParameter(
                "mixed.gpu", length, [length]);
            initial.CopyTo(gpuParameter.DataBuffer, 0);
            if (mix16)
            {
                gpuParameter.T.ConvertStorageInPlace(
                    TensorDType.BFloat16,
                    preserveFloat32Master: true);
            }
            gpuParameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
            foreach (int deviceIndex in devices)
            {
                _ = gpuParameter.T.EnsureCudaMasterFloat32Buffer(
                    deviceIndex);
                if (mix16)
                    _ = gpuParameter.T.EnsureCudaBFloat16Buffer(deviceIndex);
            }
            gpuOptimizer = new AdamW([gpuParameter], options);
            gpuOptimizer.prepare();

            PublishGradient(gpuParameter, firstGradient, devices);
            gpuOptimizer.step();
            gpuOptimizer.zero_grad();
            PublishGradient(gpuParameter, secondGradient, devices);

            NativeCudaTransferTelemetry transferBefore =
                NativeCudaRuntime.TransferTelemetry;
            NativeCudaAllocationTelemetry allocationBefore =
                NativeCudaRuntime.AllocationTelemetry;
            DeviceTransferSnapshot guarded;
            using (DeviceTransferGuard.EnterTrainingStep(deviceCount))
            {
                gpuOptimizer.step();
                guarded = DeviceTransferGuard.CurrentSnapshot!.Value;
            }
            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - transferBefore;
            NativeCudaAllocationTelemetry allocations =
                NativeCudaRuntime.AllocationTelemetry - allocationBefore;

            Assert.Equal(0, guarded.HostToDeviceCopyCount);
            Assert.Equal(0, guarded.DeviceToHostCopyCount);
            Assert.Equal(0, transfers.HostToDeviceCopyCount);
            Assert.Equal(0, transfers.DeviceToHostCopyCount);
            Assert.Equal(0, allocations.AllocationCount);
            Assert.Equal(0, allocations.FreeCount);
            Assert.Equal(deviceCount, gpuOptimizer.CudaMultiTensorPlanBuildCount);

            foreach (int deviceIndex in devices)
            {
                AssertClose(
                    expectedData,
                    Read(gpuParameter.T.EnsureCudaMasterFloat32Buffer(
                        deviceIndex)),
                    2e-5f);
            }
            AdamWState actualState = gpuOptimizer.CaptureState();
            AssertClose(
                expectedState.ParameterStates[0].FirstMoment,
                actualState.ParameterStates[0].FirstMoment,
                2e-5f);
            AssertClose(
                expectedState.ParameterStates[0].SecondMoment,
                actualState.ParameterStates[0].SecondMoment,
                2e-5f);

            // Restoring invalidates every captured native address. Rebuild
            // residency before the guarded continuation, then require the
            // continuation itself to remain allocation/transfer free.
            gpuOptimizer.RestoreState(actualState);
            gpuOptimizer.prepare();
            PublishGradient(gpuParameter, thirdGradient, devices);
            transferBefore = NativeCudaRuntime.TransferTelemetry;
            allocationBefore = NativeCudaRuntime.AllocationTelemetry;
            using (DeviceTransferGuard.EnterTrainingStep(deviceCount))
                gpuOptimizer.step();
            transfers = NativeCudaRuntime.TransferTelemetry - transferBefore;
            allocations = NativeCudaRuntime.AllocationTelemetry
                - allocationBefore;
            Assert.Equal(0, transfers.HostToDeviceCopyCount);
            Assert.Equal(0, transfers.DeviceToHostCopyCount);
            Assert.Equal(0, allocations.AllocationCount);
            Assert.Equal(0, allocations.FreeCount);
            Assert.Equal(
                deviceCount * 2,
                gpuOptimizer.CudaMultiTensorPlanBuildCount);
            foreach (int deviceIndex in devices)
            {
                AssertClose(
                    expectedRestoredData,
                    Read(gpuParameter.T.EnsureCudaMasterFloat32Buffer(
                        deviceIndex)),
                    2e-5f);
            }
            AdamWState restoredActualState = gpuOptimizer.CaptureState();
            AssertClose(
                expectedRestoredState.ParameterStates[0].FirstMoment,
                restoredActualState.ParameterStates[0].FirstMoment,
                2e-5f);
            AssertClose(
                expectedRestoredState.ParameterStates[0].SecondMoment,
                restoredActualState.ParameterStates[0].SecondMoment,
                2e-5f);
        }
        finally
        {
            gpuOptimizer?.DisposeCudaResources();
            gpuParameter?.T.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    [Fact]
    public void TwoGpuMix16FixedNs5SteadyOptimizerHasNoHostTransfers()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        Parameter? hidden = null;
        Parameter? auxiliary = null;
        NekoMuon? nekoMuon = null;
        AdamW? adamW = null;
        try
        {
            Tensor.CudaDeviceIndices = [0, 1];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(
                    PrecisionPolicy.Mix16_32);
            hidden = CreateParameter("hidden.weight", 48 * 64, [48, 64]);
            auxiliary = CreateParameter("norm.weight", 256, [256]);
            foreach (Parameter parameter in new[] { hidden, auxiliary })
            {
                parameter.T.ConvertStorageInPlace(
                    TensorDType.BFloat16,
                    preserveFloat32Master: true);
                parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
                _ = parameter.T.EnsureCudaBFloat16Buffer(1);
                _ = parameter.T.EnsureCudaMasterFloat32Buffer(0);
                _ = parameter.T.EnsureCudaMasterFloat32Buffer(1);
            }

            nekoMuon = new NekoMuon(
                [hidden],
                new NekoMuonOptions
                {
                    LearningRate = 3e-4f,
                    WeightDecay = 0.01f,
                    MaxNewtonSchulzSteps = 5,
                    NewtonSchulzInterval = 1,
                    NewtonSchulzDepthMode =
                        NekoMuonNewtonSchulzDepthMode.Fixed,
                    NewtonSchulzDepth = 5f,
                });
            adamW = new AdamW(
                [auxiliary],
                new AdamWOptions
                {
                    LearningRate = 3e-4f,
                    Beta1 = 0.9f,
                    Beta2 = 0.95f,
                    Epsilon = 1e-8f,
                    WeightDecay = 0.01f,
                });
            var optimizer = new CompositeOptimizer(nekoMuon, adamW);
            optimizer.prepare();

            PublishGradients(hidden, auxiliary, offset: 0);
            optimizer.step();
            PublishGradients(hidden, auxiliary, offset: 17);

            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;
            DeviceTransferSnapshot guarded;
            using (DeviceTransferGuard.EnterTrainingStep(
                cudaDeviceCount: 2))
            {
                optimizer.step();
                guarded = DeviceTransferGuard.CurrentSnapshot!.Value;
            }
            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - before;

            Assert.Equal(0, guarded.HostToDeviceCopyCount);
            Assert.Equal(0, guarded.HostToDeviceBytes);
            Assert.Equal(0, guarded.DeviceToHostCopyCount);
            Assert.Equal(0, guarded.DeviceToHostBytes);
            Assert.Equal(0, transfers.HostToDeviceCopyCount);
            Assert.Equal(0, transfers.HostToDeviceBytes);
            Assert.Equal(0, transfers.DeviceToHostCopyCount);
            Assert.Equal(0, transfers.DeviceToHostBytes);
        }
        finally
        {
            nekoMuon?.DisposeCudaResources();
            adamW?.DisposeCudaResources();
            hidden?.T.InvalidateCudaBuffers();
            auxiliary?.T.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private static Parameter CreateParameter(
        string name,
        int length,
        int[] shape)
        => new(
            Values(length, 0),
            shape,
            name,
            WeightDecayPolicy.Apply);

    private static void PublishGradients(
        Parameter hidden,
        Parameter auxiliary,
        int offset)
    {
        foreach (Parameter parameter in new[] { hidden, auxiliary })
        {
            float[] gradient = Values(parameter.T.Numel, offset + 31);
            parameter.T.SetCudaGradient(gradient, 0);
            parameter.T.SetCudaGradient(gradient, 1);
            parameter.T.MarkCudaGradientsSynchronized([0, 1]);
        }
    }

    private static void PublishGradient(
        Parameter parameter,
        float[] gradient,
        IReadOnlyList<int> devices)
    {
        foreach (int deviceIndex in devices)
            parameter.T.SetCudaGradient(gradient, deviceIndex);
        parameter.T.MarkCudaGradientsSynchronized(devices);
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

    private static float[] Values(int length, int offset)
        => Enumerable.Range(0, length)
            .Select(index => MathF.Sin((index + offset) * 0.173f) * 0.025f)
            .ToArray();
}
