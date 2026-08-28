using NNtrain;
using Xunit;

public sealed class NativeCudaPinnedTransferTests
{
    [Fact]
    public void TransfersLargerThanSixteenMibUseBoundedPinnedChunks()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int elementsPerChunk = (16 * 1024 * 1024) / sizeof(float);
        const int length = elementsPerChunk + 17;
        float[] source = new float[length];
        source[0] = 1.25f;
        source[elementsPerChunk - 1] = -2.5f;
        source[elementsPerChunk] = 3.75f;
        source[^1] = -4.125f;
        float[] destination = new float[length];
        NativeCudaDevice device = NativeCudaRuntime.GetDevice(0);
        using NativeCudaBuffer<float> buffer = device.Allocate1D<float>(length);

        NativeCudaTransferTelemetry before =
            NativeCudaRuntime.TransferTelemetry;
        buffer.CopyFromCPU(source);
        buffer.CopyToCPU(destination);
        NativeCudaTransferTelemetry transfer =
            NativeCudaRuntime.TransferTelemetry - before;

        Assert.Equal(2, transfer.HostToDeviceCopyCount);
        Assert.Equal(checked((long)length * sizeof(float)),
            transfer.HostToDeviceBytes);
        Assert.Equal(2, transfer.DeviceToHostCopyCount);
        Assert.Equal(checked((long)length * sizeof(float)),
            transfer.DeviceToHostBytes);
        Assert.Equal(source, destination);
    }
}
