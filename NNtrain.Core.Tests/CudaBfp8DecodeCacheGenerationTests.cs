using NNtrain;
using Xunit;

public sealed class CudaBfp8DecodeCacheGenerationTests
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void OptimizerPublicationsRefreshDecodeInPlaceAndFinalInvalidationFreesIt(
        bool tensorWide)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        Tensor? tensor = null;
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Bfp8QuantizationDescriptor descriptor = tensorWide
                ? Bfp8QuantizationDescriptor.TensorWide
                : Bfp8QuantizationDescriptor.Mix8_32;
            const int length = 257;
            tensor = Tensor.FromBfp8(
                Values(length, generation: 0),
                [length],
                descriptor);
            tensor.to(new TorchDevice(TensorDevice.Cuda, 0));

            CudaBfp8BufferView encodedReplica =
                tensor.EnsureCudaBfp8Buffer(0);
            NativeCudaBuffer<ushort> decodedReplica =
                tensor.EnsureCudaBfp8BFloat16Buffer(0);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
            nint decodePointer = decodedReplica.NativePtr;
            NativeCudaAllocationTelemetry steadyAllocationBaseline =
                NativeCudaRuntime.AllocationTelemetry;
            CudaBfp8GemmTelemetrySnapshot steadyDecodeBaseline =
                CudaBfp8GemmTelemetry.Snapshot;
            ushort[]? previousBits = null;

            const int publicationCount = 12;
            for (int generation = 1;
                 generation <= publicationCount;
                 generation++)
            {
                Bfp8EncodedStorage encoded = Bfp8QuantizationCodec.Default.Encode(
                    Values(length, generation),
                    descriptor);
                encodedReplica.Payload.CopyFromCPU(encoded.Payload.Span);
                encodedReplica.Scales.CopyFromCPU(encoded.Scales.Span);

                // This is the same generation publication used after a CUDA
                // optimizer commits its resident BFP8 parameter replicas.
                tensor.MarkCudaBfp8DataReplicasSynchronized([0]);
                NativeCudaBuffer<ushort> refreshed =
                    tensor.EnsureCudaBfp8BFloat16Buffer(0);
                ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

                Assert.Equal(decodePointer, refreshed.NativePtr);
                var actualBits = new ushort[length];
                refreshed.CopyToCPU(actualBits);
                ushort[] expectedBits = ExpectedBFloat16Bits(encoded);
                Assert.Equal(expectedBits, actualBits);
                if (previousBits is not null)
                    Assert.False(previousBits.SequenceEqual(actualBits));
                previousBits = actualBits;
            }

            NativeCudaAllocationTelemetry steadyAllocations =
                NativeCudaRuntime.AllocationTelemetry
                - steadyAllocationBaseline;
            Assert.Equal(0, steadyAllocations.AllocationCount);
            Assert.Equal(0, steadyAllocations.AllocationBytes);
            Assert.Equal(0, steadyAllocations.FreeCount);
            Assert.Equal(0, steadyAllocations.FreeBytes);
            CudaBfp8GemmTelemetrySnapshot steadyDecodes =
                CudaBfp8GemmTelemetry.Snapshot - steadyDecodeBaseline;
            Assert.Equal(
                publicationCount,
                steadyDecodes.BFloat16DecodeCacheMisses);

            NativeCudaAllocationTelemetry beforeInvalidation =
                NativeCudaRuntime.AllocationTelemetry;
            tensor.InvalidateCudaBuffers();
            NativeCudaAllocationTelemetry invalidation =
                NativeCudaRuntime.AllocationTelemetry - beforeInvalidation;
            long expectedBytes = checked(
                (long)length * (sizeof(sbyte) + sizeof(ushort))
                + (long)descriptor.GetScaleCount(length) * sizeof(float));
            Assert.Equal(3, invalidation.FreeCount);
            Assert.Equal(expectedBytes, invalidation.FreeBytes);
            Assert.Throws<ObjectDisposedException>(
                () => _ = decodedReplica.NativePtr);
            tensor = null;
        }
        finally
        {
            tensor?.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private static ushort[] ExpectedBFloat16Bits(Bfp8EncodedStorage encoded)
    {
        var decoded = new float[encoded.Count];
        Bfp8QuantizationCodec.Default.Decode(
            encoded.Payload.Span,
            encoded.Scales.Span,
            encoded.Descriptor,
            decoded);
        var bits = new ushort[decoded.Length];
        TensorStorageCodec.EncodeBFloat16(decoded, bits);
        return bits;
    }

    private static float[] Values(int length, int generation)
        => Enumerable.Range(0, length)
            .Select(index =>
                MathF.Sin((index + generation * 17) * 0.113f)
                    * (0.25f + generation * 0.013f)
                + MathF.Cos((index - generation * 11) * 0.071f) * 0.09f)
            .ToArray();
}
