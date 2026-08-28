using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaBfp8CleanupTests
{
    [Fact]
    public void DataGradientPayloadScaleAndCachesFreeAcrossEveryReplica()
    {
        if (Tensor.CudaDeviceCount < 2)
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0, 1];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(
                    PrecisionPolicy.Bfp8);
            Tensor tensor = Tensor.FromBfp8(
                Enumerable.Range(0, 257)
                    .Select(index => MathF.Sin(index * 0.1f))
                    .ToArray(),
                [1, 257],
                Bfp8QuantizationDescriptor.TensorWide);
            NativeCudaAllocationTelemetry before =
                NativeCudaRuntime.AllocationTelemetry;

            for (int device = 0; device < 2; device++)
            {
                tensor.EnsureCudaBfp8Buffer(device);
                tensor.EnsureCudaBfp8BFloat16Buffer(device);
                tensor.EnsureCudaBfp8ColumnMajorPayload(1, 257, device);
                tensor.PrepareCudaBfp8GradientReplica(device);
            }
            NativeCudaAllocationTelemetry allocated =
                NativeCudaRuntime.AllocationTelemetry - before;
            Assert.Equal(12, allocated.AllocationCount);

            NativeCudaAllocationTelemetry beforeDispose =
                NativeCudaRuntime.AllocationTelemetry;
            tensor.InvalidateCudaBuffers();
            NativeCudaAllocationTelemetry freed =
                NativeCudaRuntime.AllocationTelemetry - beforeDispose;
            Assert.Equal(allocated.AllocationCount, freed.FreeCount);
            Assert.Equal(allocated.AllocationBytes, freed.FreeBytes);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }
}
