using NNtrain;
using Xunit;

public sealed class TensorOwnershipTests
{
    [Fact]
    public void FacadeKeepsOneLogicalStorageReplicaAndAutogradOwner()
    {
        var tensor = new Tensor([1f, 2f, 3f, 4f], [2, 2]);
        TensorValue value = tensor.Value;
        TensorStorageOwner storage = value.Storage;
        Tensor.DeviceReplicaSet replicas = value.Replicas;
        AutogradNode autograd = value.Autograd;

        using (Tensor.DataMutation mutation = tensor.BeginDataMutation())
            mutation.Values[0] = 7f;
        tensor.ZeroGrad();

        Assert.Same(value, tensor.Value);
        Assert.Same(storage, tensor.Value.Storage);
        Assert.Same(replicas, tensor.Value.Replicas);
        Assert.Same(autograd, tensor.Value.Autograd);
        Assert.Same(autograd, tensor.Node);
        Assert.Equal(7f, tensor[0, 0]);
        Assert.Equal([2, 2], tensor.Shape);
    }

    [Fact]
    public void AutogradOwnerReleasesItsLeaseExactlyOnce()
    {
        var tracked = new TrackedResource();
        Tensor left = Tensor.Scalar(2f);
        Tensor right = Tensor.Scalar(3f);
        Tensor output = left * right;
        AutogradNode node = output.Value.Autograd;
        node.RegisterResource(tracked);

        output.BackwardAndRelease();
        output.BackwardAndRelease();

        Assert.Same(node, output.Value.Autograd);
        Assert.Equal(1, tracked.DisposeCount);
        Assert.False(node.HasLeases);
        Assert.Empty(node.Parents);
    }

    [Fact]
    public void TwoGpuReplicaCreationSharesOneOwnerAndCleanupClearsIt()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        var tensor = new Tensor(
            Enumerable.Range(0, 257).Select(index => (float)index).ToArray(),
            [257]);
        try
        {
            Tensor.CudaDeviceIndices = [0, 1];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            TensorValue owner = tensor.Value;
            nint[] pointers = new nint[2];
            using var start = new Barrier(2);

            Parallel.For(0, 2, deviceIndex =>
            {
                start.SignalAndWait();
                pointers[deviceIndex] = tensor
                    .EnsureCudaFloat32Buffer(deviceIndex)
                    .NativePtr;
                _ = tensor.EnsureCudaGradientBuffer(deviceIndex);
            });

            Assert.Same(owner, tensor.Value);
            Assert.Equal(2, owner.Replicas.DataReplicaCount);
            Assert.Equal(2, owner.Replicas.GradientReplicaCount);
            Assert.NotEqual(nint.Zero, pointers[0]);
            Assert.NotEqual(nint.Zero, pointers[1]);
            Assert.NotEqual(pointers[0], pointers[1]);
            Assert.Equal([0, 1], tensor.GetResidentCudaDeviceIndices());

            tensor.InvalidateCudaBuffers();
            tensor.InvalidateCudaBuffers();

            Assert.Equal(0, owner.Replicas.DataReplicaCount);
            Assert.Equal(0, owner.Replicas.GradientReplicaCount);
            Assert.Empty(tensor.GetResidentCudaDeviceIndices());
            Assert.Equal(TensorDevice.Cpu, tensor.Device);
        }
        finally
        {
            tensor.InvalidateCudaBuffers();
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    private sealed class TrackedResource : IDisposable
    {
        private int _disposeCount;

        internal int DisposeCount => Volatile.Read(ref _disposeCount);

        public void Dispose() => Interlocked.Increment(ref _disposeCount);
    }
}
