using NNtrain;
using NNtrain.Arc;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class ArcInferenceSessionTests
{
    [Fact]
    public void ArcDeviceSetKeepsLegacySingleDeviceAndRejectsUnlistedLane()
    {
        using var ambient = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu));
        var legacy = new ExecutionOptions
        {
            Device = ExecutionDeviceKind.Arc,
            ArcDeviceIndex = 1,
        }.Validate();
        Assert.True(legacy.IncludesArcDevice(1));
        Assert.False(legacy.IncludesArcDevice(0));

        using var session = new ExecutionSession(new ExecutionOptions
        {
            Device = ExecutionDeviceKind.Arc,
            ArcDeviceIndex = 0,
            ArcDevices = new DeviceSet(0, 1),
        });
        session.AttachLane(new FakeArcLane(0));
        session.AttachLane(new FakeArcLane(1));
        Assert.Throws<ArgumentException>(() => session.AttachLane(new FakeArcLane(2)));
        Assert.Throws<ArgumentException>(() => new ExecutionOptions
        {
            Device = ExecutionDeviceKind.Arc,
            ArcDeviceIndex = 2,
            ArcDevices = new DeviceSet(0, 1),
        }.Validate());

        using (session.Enter())
        {
            Assert.Equal(0, TensorExecutionContext.Device.Index);
            using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, 1)))
                Assert.Equal(1, TensorExecutionContext.Device.Index);
            Assert.Equal(0, TensorExecutionContext.Device.Index);
            Assert.Throws<InvalidOperationException>(() =>
                TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, 2)));
        }
    }

    [Fact]
    public void InferenceScopeOwnsBothArcLanesAndRestoresAmbientContext()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count < 2, "Two Intel Arc OpenCL GPUs are required.");
        ExecutionSession? previousSession = ExecutionSession.Current;
        TorchDevice previousDevice = TensorExecutionContext.Device;
        ArcExecutionLane[] lanes;
        using (Tensor.BeginArcInferenceExecution([0, 1]))
        {
            ExecutionSession session = Assert.IsType<ExecutionSession>(ExecutionSession.Current);
            lanes = session.Lanes.Cast<ArcExecutionLane>().OrderBy(lane => lane.DeviceIndex).ToArray();
            Assert.Equal([0, 1], lanes.Select(lane => lane.DeviceIndex).ToArray());
            Assert.Equal(0, Tensor.ArcLane.DeviceIndex);
            using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, 1)))
                Assert.Equal(1, Tensor.ArcLane.DeviceIndex);
            Assert.Equal(0, Tensor.ArcLane.DeviceIndex);
        }
        Assert.Same(previousSession, ExecutionSession.Current);
        Assert.Equal(previousDevice, TensorExecutionContext.Device);
        Assert.All(lanes, lane => Assert.Throws<ObjectDisposedException>(() => lane.Allocate(1)));
    }

    private sealed class FakeArcLane(int index) : IExecutionLane,
        IDeviceMemoryManager, IKernelCapabilitySet
    {
        public ExecutionDeviceKind DeviceKind => ExecutionDeviceKind.Arc;
        public int DeviceIndex => index;
        public IDeviceMemoryManager MemoryManager => this;
        public IKernelCapabilitySet Capabilities => this;
        public IExecutionProfiler Profiler => NullExecutionProfiler.Instance;
        public long AllocationCount => 0;
        public long AllocatedBytes => 0;
        public bool Supports(string feature) => false;
        public void Dispose() { }
    }
}
