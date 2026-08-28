using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaPinnedUploadSlotTests
{
    [Fact]
    public void LaneUploadsStayBoundedAndReleasePinnedMemoryAndEvents()
    {
        if (Tensor.CudaDeviceCount == 0)
            return;

        NativeCudaPinnedUploadTelemetry before =
            NativeCudaPinnedUploadTracker.Telemetry;
        var session = new ExecutionSession(
            new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(0),
            },
            [CudaExecutionLaneFactory.Create(0)]);
        try
        {
            using IDisposable scope = session.Enter();
            NativeCudaDevice device = NativeCudaRuntime.GetDevice(0);
            int[] adaptiveLengths =
                [32, 32, 32, 64, 64, 64, 96, 96, 96, 32, 128, 128, 128];
            for (int iteration = 0; iteration < 384; iteration++)
            {
                int length = adaptiveLengths[
                    iteration % adaptiveLengths.Length];
                int[] input = Enumerable.Repeat(iteration, length).ToArray();
                int[] target = Enumerable.Repeat(iteration + 1, length).ToArray();
                NativeCudaBuffer<int>? inputBuffer = null;
                NativeCudaBuffer<int>? targetBuffer = null;
                try
                {
                    // Keep both buffers live while their asynchronous copies
                    // are pending, exactly as embedding input and CE labels do.
                    inputBuffer = Tensor.RentCudaIntBuffer(0, input);
                    targetBuffer = Tensor.RentCudaIntBuffer(0, target);
                }
                finally
                {
                    if (targetBuffer is not null)
                        Tensor.ReturnCudaIntBuffer(device, targetBuffer);
                    if (inputBuffer is not null)
                        Tensor.ReturnCudaIntBuffer(device, inputBuffer);
                }
            }

            BoundedUploadSlotCacheTelemetry slots =
                Tensor.GetCudaIntUploadSlotTelemetry(0);
            NativeCudaPinnedUploadTelemetry active =
                NativeCudaPinnedUploadTracker.Telemetry - before;
            Assert.Equal(3, slots.ActiveLengthCount);
            Assert.Equal(6, slots.ActiveSlotCount);
            Assert.Equal(384 * 2, slots.UseCount);
            Assert.Equal(slots.CreatedSlotCount - slots.ActiveSlotCount,
                slots.DisposedSlotCount);
            Assert.Equal(slots.ActiveSlotCount, active.ActiveSlotCount);
            Assert.Equal(slots.ActiveSlotCount, active.ActiveEventCount);
            Assert.Equal(
                checked(slots.ActiveElementCapacity * sizeof(int)),
                active.ActivePinnedBytes);
            Assert.True(active.ReuseSynchronizationCount > 0);
        }
        finally
        {
            session.Dispose();
        }

        NativeCudaPinnedUploadTelemetry released =
            NativeCudaPinnedUploadTracker.Telemetry - before;
        Assert.Equal(0, released.ActiveSlotCount);
        Assert.Equal(0, released.ActiveEventCount);
        Assert.Equal(0, released.ActivePinnedBytes);
        Assert.Equal(released.CreatedSlotCount, released.DisposedSlotCount);
        Assert.Equal(released.HostAllocationCount, released.HostFreeCount);
        Assert.Equal(released.EventCreateCount, released.EventDestroyCount);
    }
}
