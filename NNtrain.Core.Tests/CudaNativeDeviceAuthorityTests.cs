using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Interop;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaNativeDeviceAuthorityTests
{
    [Fact]
    public void ExplicitDeviceKernelAbisKeepBridgeCacheAndStreamCoherent()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        const int firstDevice = 0;
        const int secondDevice = 1;
        const int length = 32;
        const int iterations = 96;

        CudaExecutionLane[] lanes =
        [
            CudaExecutionLaneFactory.Create(firstDevice),
            CudaExecutionLaneFactory.Create(secondDevice),
        ];
        using var session = new ExecutionSession(
            new ExecutionOptions
            {
                Device = ExecutionDeviceKind.Cuda,
                CudaDevices = new DeviceSet(firstDevice, secondDevice),
                Precision = PrecisionPolicy.Mix16_32,
            },
            lanes);
        using IDisposable execution = session.Enter();

        NativeCudaDevice first =
            ForgetMemoryV2Cuda.GetAccelerator(firstDevice);
        NativeCudaDevice second =
            ForgetMemoryV2Cuda.GetAccelerator(secondDevice);
        using NativeCudaBuffer<float> firstValues = first.Allocate1D(
            Enumerable.Repeat(1f, length).ToArray());
        using NativeCudaBuffer<float> logits = second.Allocate1D<float>(
            [1f, 5f, 2f, -1f]);
        using NativeCudaBuffer<int> targets = second.Allocate1D<int>([1]);
        using NativeCudaBuffer<int> correctCount = second.Allocate1D<int>(1);
        using NativeCudaBuffer<sbyte> tablePayload = second.Allocate1D<sbyte>(
            [4, -2, 1, 3]);
        using NativeCudaBuffer<float> tableScales =
            second.Allocate1D<float>([0.5f]);
        using NativeCudaBuffer<int> indices = second.Allocate1D<int>([0]);
        using NativeCudaBuffer<sbyte> outputPayload =
            second.Allocate1D<sbyte>(4);
        using NativeCudaBuffer<float> outputScales =
            second.Allocate1D<float>(1);
        using NativeCudaBuffer<float> workspace =
            second.Allocate1D<float>(1);
        using NativeCudaBuffer<float> gradientScale =
            second.Allocate1D<float>([1f]);

        for (int iteration = 0; iteration < iterations; iteration++)
        {
            // Seed the native bridge and the managed prepared-stream cache
            // with device 0's production lane. The ABI call then selects
            // device 1 directly on this same native thread. Its explicit
            // device+stream contract means it returns with device 1 current;
            // it must therefore also publish that selection to the bridge.
            first.Bind();

            int status = (iteration % 3) switch
            {
                0 => CudaNativeGateway.ClassificationCorrectFloat32(
                    secondDevice,
                    logits.NativePtr,
                    targets.NativePtr,
                    correctCount.NativePtr,
                    sampleCount: 1,
                    classCount: 4,
                    lanes[secondDevice].ComputeStreamHandle),
                1 => CudaNativeGateway.Bfp8EmbeddingForward(
                    secondDevice,
                    tablePayload.NativePtr,
                    tableScales.NativePtr,
                    tableLength: 4,
                    tableBlockSize: 4,
                    indices.NativePtr,
                    indexCount: 1,
                    width: 4,
                    outputPayload.NativePtr,
                    outputScales.NativePtr,
                    outputBlockSize: 4,
                    outputScaleCount: 1,
                    workspace.NativePtr,
                    workspaceLength: 1,
                    lanes[secondDevice].ComputeStreamHandle),
                _ => CudaNativeGateway.Bfp8GradientScale(
                    secondDevice,
                    gradientScale.NativePtr,
                    multiplier: 1f,
                    lanes[secondDevice].ComputeStreamHandle),
            };
            Assert.Equal(0, status);
            CudaNativeThreadContextSnapshot selectedByAbi =
                CudaNativeGateway.CurrentThreadContext;
            Assert.True(selectedByAbi.HasSelectedDevice);
            Assert.Equal(secondDevice, selectedByAbi.SelectedDevice);

            // The managed cache still says lane 0 is prepared. Its cached bind
            // must nevertheless consult the native bridge authority. Before
            // this fix it skipped the bind, then launched a device-0 pointer
            // on device 0's stream while physical device 1 was current.
            CudaTensorNative.Scale(
                firstDevice,
                firstValues.NativePtr,
                length,
                scale: 1f);
            lanes[firstDevice].SynchronizeComputeStream();
        }

        lanes[secondDevice].SynchronizeComputeStream();
        first.Bind();
        CudaNativeThreadContextSnapshot beforeCachedBinds =
            CudaNativeGateway.CurrentThreadContext;
        for (int iteration = 0; iteration < 100_000; iteration++)
            first.Bind();
        CudaNativeThreadContextSnapshot afterCachedBinds =
            CudaNativeGateway.CurrentThreadContext;
        Assert.Equal(
            beforeCachedBinds.Generation,
            afterCachedBinds.Generation);
        Assert.Equal(
            beforeCachedBinds.SetDeviceCallCount,
            afterCachedBinds.SetDeviceCallCount);
        Assert.Equal(
            beforeCachedBinds.UseExternalStreamCallCount,
            afterCachedBinds.UseExternalStreamCallCount);

        var actual = new float[length];
        firstValues.CopyToCPU(actual);
        Assert.All(actual, value => Assert.Equal(1f, value));
    }
}
