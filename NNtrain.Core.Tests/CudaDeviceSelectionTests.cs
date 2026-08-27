using NNtrain;
using Xunit;

public sealed class CudaDeviceSelectionTests
{
    [Fact]
    public void HostGradientPipelineKeepsNativeDeviceSelectionCacheCoherent()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        const int sourceDevice = 1;
        const int destinationDevice = 0;
        const int length = 256;

        NativeCudaDevice source =
            ForgetMemoryV2Cuda.GetAccelerator(sourceDevice);
        NativeCudaDevice destination =
            ForgetMemoryV2Cuda.GetAccelerator(destinationDevice);
        using NativeCudaBuffer<float> localSource = destination.Allocate1D(
            Enumerable.Repeat(1f, length).ToArray());
        using NativeCudaBuffer<ushort> local =
            destination.Allocate1D<ushort>(length);
        using NativeCudaBuffer<float> remoteSource = source.Allocate1D(
            Enumerable.Repeat(2f, length).ToArray());
        using NativeCudaBuffer<ushort> remote =
            source.Allocate1D<ushort>(length);
        using NativeCudaBuffer<float> reduced =
            destination.Allocate1D<float>(length);
        using NativeCudaBuffer<float> addend = source.Allocate1D(
            Enumerable.Repeat(3f, length).ToArray());
        using NativeCudaBuffer<float> result =
            source.Allocate1D<float>(length);

        nint localReady = 0;
        nint remoteReady = 0;
        nint pipeline = 0;
        try
        {
            localReady = CudaGradientBuckets.CreateReadyEvent(
                destination, destinationDevice);
            remoteReady = CudaGradientBuckets.CreateReadyEvent(
                source, sourceDevice);
            CudaGradientBuckets.Pack(
                destinationDevice, destination, localSource, local, 0, length);
            CudaGradientBuckets.RecordReady(
                destinationDevice, destination, localReady);
            CudaGradientBuckets.Pack(
                sourceDevice, source, remoteSource, remote, 0, length);
            CudaGradientBuckets.RecordReady(
                sourceDevice, source, remoteReady);
            pipeline = CudaGradientBuckets.CreateHostPipeline(
                sourceDevice, destinationDevice, length);

            // Seed both device-selection caches with sourceDevice. Before the
            // regression fix, the native host pipeline used direct
            // cudaSetDevice calls and returned on destinationDevice without
            // updating cuda_runtime_bridge's thread-local selected device.
            source.Bind();
            CudaTensorNative.Add(
                sourceDevice,
                remoteSource.NativePtr,
                addend.NativePtr,
                result.NativePtr,
                length,
                bfloat16: false);
            source.Synchronize();

            CudaGradientBuckets.HostPipelineExchange(
                destination,
                pipeline,
                local,
                remote,
                reduced,
                length,
                squaredSum: 0,
                localReady,
                remoteReady);

            // This must bind sourceDevice again on the same managed/native
            // thread. A stale native selection cache launches this kernel on
            // destinationDevice with sourceDevice pointers and poisons the
            // context with cudaErrorIllegalAddress (700).
            CudaTensorNative.Add(
                sourceDevice,
                remoteSource.NativePtr,
                addend.NativePtr,
                result.NativePtr,
                length,
                bfloat16: false);
            source.Synchronize();
            destination.Synchronize();

            var actual = new float[length];
            result.CopyToCPU(actual);
            Assert.All(actual, value => Assert.Equal(5f, value));
        }
        finally
        {
            CudaGradientBuckets.DestroyHostPipeline(destination, pipeline);
            CudaGradientBuckets.DestroyEvent(
                destination, destinationDevice, localReady);
            CudaGradientBuckets.DestroyEvent(source, sourceDevice, remoteReady);
        }
    }
}
