using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

[Collection(TensorSimdCollection.Name)]
public sealed class CudaOptimizerResidencyTests
{
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

    private static float[] Values(int length, int offset)
        => Enumerable.Range(0, length)
            .Select(index => MathF.Sin((index + offset) * 0.173f) * 0.025f)
            .ToArray();
}
