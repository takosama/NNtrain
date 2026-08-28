using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaOutputGradientSeedTests
{
    [Fact]
    public void DefaultAndCustomScalarSeedsAccumulateWithoutH2d()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            var output = new Tensor([2f], [1]);
            try
            {
                output.to(new TorchDevice(TensorDevice.Cuda, 0));
                using (DeviceTransferGuard.EnterTrainingStep(1))
                {
                    output.Backward();
                    output.Backward([0.375f]);

                    DeviceTransferSnapshot snapshot = Assert.NotNull(
                        DeviceTransferGuard.CurrentSnapshot);
                    Assert.Equal(0, snapshot.HostToDeviceCopyCount);
                    Assert.Equal(0, snapshot.HostToDeviceBytes);
                }

                Assert.Equal(1.375f, Assert.Single(output.Grad));
            }
            finally
            {
                output.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void ExistingHostScalarGradientIsFoldedIntoKernelArgument()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        var output = new Tensor([3f], [1]);
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            output.Backward([0.625f]);
            output.to(new TorchDevice(TensorDevice.Cuda, 0));

            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using (DeviceTransferGuard.EnterTrainingStep(1))
            {
                output.Backward([1.125f]);

                DeviceTransferSnapshot snapshot = Assert.NotNull(
                    DeviceTransferGuard.CurrentSnapshot);
                Assert.Equal(0, snapshot.HostToDeviceCopyCount);
                Assert.Equal(0, snapshot.HostToDeviceBytes);
            }

            Assert.Equal(1.75f, Assert.Single(output.Grad));
        }
        finally
        {
            output.InvalidateCudaBuffers();
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void ExplicitVectorSeedPublishesLocalCudaGradientAndAccumulates()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            var output = new Tensor([1f, 2f, 3f, 4f], [2, 2]);
            float[] seed = [0.25f, -0.5f, 0.75f, 1f];
            try
            {
                // The tensor may still be a lazy CPU facade; CUDA execution
                // makes the explicit root seed authoritative on device 0.
                output.Backward(seed);
                output.Backward(seed);

                Assert.Equal(
                    [0.5f, -1f, 1.5f, 2f],
                    output.Grad.ToArray());
                CudaGradientCoherenceSnapshot coherence =
                    output.GetCudaGradientCoherenceSnapshot();
                Assert.Equal(CudaGradientCoherenceKind.Local, coherence.Kind);
                Assert.Equal(0, coherence.LocalDeviceIndex);

                output.ZeroGrad();
                using (DeviceTransferGuard.EnterTrainingStep(1))
                {
                    Assert.Throws<InvalidOperationException>(
                        () => output.Backward(seed));
                }
            }
            finally
            {
                output.InvalidateCudaBuffers();
            }
        });
    }

    private static void WithCuda(Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            action();
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }
}
