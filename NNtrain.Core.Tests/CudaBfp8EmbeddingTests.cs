using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaBfp8EmbeddingTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LookupDecodesOnlySelectedTailsAndBackwardIsCollisionSafe(
        bool blockScaled)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        Bfp8QuantizationDescriptor descriptor = blockScaled
            ? Bfp8QuantizationDescriptor.Mix8_32
            : Bfp8QuantizationDescriptor.TensorWide;
        WithCuda(descriptor, () =>
        {
            const int rows = 7;
            const int width = 257;
            int[] indices = [6, 1, 6];
            float[] source = Values(rows * width, 11, 1.7f);
            float[] decodedSource = Quantized(source, descriptor);
            float[] selected = Gather(decodedSource, indices, width);
            float[] expectedOutput = Quantized(selected, descriptor);
            float[] seed = Values(indices.Length * width, 37, 0.09f);
            float[] expectedGradient = GatherGradient(
                rows, width, indices, seed);
            if (!blockScaled)
            {
                expectedGradient = Quantized(
                    expectedGradient,
                    Bfp8QuantizationDescriptor.TensorWide);
            }
            Tensor table = Tensor.FromBfp8(
                source, [rows, width], descriptor, "tokens");
            table.to(new TorchDevice(TensorDevice.Cuda, 0));
            _ = table.EnsureCudaBfp8Buffer(0);
            NativeCudaTransferTelemetry transferBefore =
                NativeCudaRuntime.TransferTelemetry;
            CudaBfp8GemmTelemetrySnapshot gemmBefore =
                CudaBfp8GemmTelemetry.Snapshot;
            CudaBfp8EmbeddingTelemetrySnapshot embeddingBefore =
                CudaBfp8EmbeddingTelemetry.Snapshot;

            Tensor output = table.EmbeddingLookup(indices, indices.Length);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaTransferTelemetry transfer =
                NativeCudaRuntime.TransferTelemetry - transferBefore;
            CudaBfp8GemmTelemetrySnapshot gemm =
                CudaBfp8GemmTelemetry.Snapshot - gemmBefore;
            CudaBfp8EmbeddingTelemetrySnapshot embedding =
                CudaBfp8EmbeddingTelemetry.Snapshot - embeddingBefore;
            Assert.Equal(indices.Length * sizeof(int),
                transfer.HostToDeviceBytes);
            Assert.Equal(0, transfer.DeviceToHostBytes);
            Assert.Equal(0, gemm.BFloat16DecodeCacheMisses);
            Assert.Equal(1, embedding.LookupExecutions);
            Assert.Equal(0, embedding.LookupWithPositionsExecutions);
            Assert.Equal(selected.Length, embedding.SelectedValuesDecoded);
            Assert.Equal(
                descriptor.GetScaleCount(selected.Length) == 1
                    ? (selected.Length + 1023) / 1024
                    : 0,
                embedding.ReductionWorkspaceElements);
            Assert.Equal(descriptor, output.Bfp8Quantization);
            AssertClose(expectedOutput, output.Data, 1e-5f);

            output.BackwardAndRelease(seed);
            if (blockScaled)
                Assert.False(table.HasAuthoritativeCudaBfp8Gradient);
            else
                AssertPureGradientPublished(table);
            AssertClose(expectedGradient, table.Grad, 1e-5f);
            output.InvalidateCudaBuffers();
            table.InvalidateCudaBuffers();
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LookupWithPositionsPreservesDescriptorAndBackwardTails(
        bool blockScaled)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        Bfp8QuantizationDescriptor descriptor = blockScaled
            ? Bfp8QuantizationDescriptor.Block(96)
            : Bfp8QuantizationDescriptor.TensorWide;
        WithCuda(descriptor, () =>
        {
            const int tokenRows = 9;
            const int batch = 2;
            const int sequence = 3;
            const int width = 129;
            int[] indices = [8, 2, 8, 1, 2, 8];
            float[] tokens = Values(tokenRows * width, 5, 1.1f);
            float[] positions = Values(sequence * width, 71, 0.6f);
            float[] decodedTokens = Quantized(tokens, descriptor);
            float[] decodedPositions = Quantized(positions, descriptor);
            float[] added = GatherWithPositions(
                decodedTokens,
                decodedPositions,
                indices,
                sequence,
                width);
            float[] expectedOutput = Quantized(added, descriptor);
            float[] seed = Values(indices.Length * width, 101, 0.07f);
            float[] expectedTokenGradient = GatherGradient(
                tokenRows, width, indices, seed);
            float[] expectedPositionGradient = PositionGradient(
                batch, sequence, width, seed);
            if (!blockScaled)
            {
                expectedTokenGradient = Quantized(
                    expectedTokenGradient,
                    Bfp8QuantizationDescriptor.TensorWide);
                expectedPositionGradient = Quantized(
                    expectedPositionGradient,
                    Bfp8QuantizationDescriptor.TensorWide);
            }
            Tensor tokenTable = Tensor.FromBfp8(
                tokens, [tokenRows, width], descriptor, "tokens");
            Tensor positionTable = Tensor.FromBfp8(
                positions, [sequence, width], descriptor, "positions");
            MoveToCuda(tokenTable, positionTable);
            _ = tokenTable.EnsureCudaBfp8Buffer(0);
            _ = positionTable.EnsureCudaBfp8Buffer(0);
            NativeCudaTransferTelemetry transferBefore =
                NativeCudaRuntime.TransferTelemetry;
            CudaBfp8GemmTelemetrySnapshot gemmBefore =
                CudaBfp8GemmTelemetry.Snapshot;
            CudaBfp8EmbeddingTelemetrySnapshot embeddingBefore =
                CudaBfp8EmbeddingTelemetry.Snapshot;

            Tensor output = tokenTable.EmbeddingLookupWithPositions(
                positionTable, indices, batch, sequence);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaTransferTelemetry transfer =
                NativeCudaRuntime.TransferTelemetry - transferBefore;
            CudaBfp8GemmTelemetrySnapshot gemm =
                CudaBfp8GemmTelemetry.Snapshot - gemmBefore;
            CudaBfp8EmbeddingTelemetrySnapshot embedding =
                CudaBfp8EmbeddingTelemetry.Snapshot - embeddingBefore;
            Assert.Equal(indices.Length * sizeof(int),
                transfer.HostToDeviceBytes);
            Assert.Equal(0, transfer.DeviceToHostBytes);
            Assert.Equal(0, gemm.BFloat16DecodeCacheMisses);
            Assert.Equal(0, embedding.LookupExecutions);
            Assert.Equal(1, embedding.LookupWithPositionsExecutions);
            Assert.Equal(added.Length * 2L,
                embedding.SelectedValuesDecoded);
            Assert.Equal(descriptor, output.Bfp8Quantization);
            AssertClose(expectedOutput, output.Data, 1e-5f);

            output.BackwardAndRelease(seed);
            if (blockScaled)
            {
                Assert.False(tokenTable.HasAuthoritativeCudaBfp8Gradient);
                Assert.False(positionTable.HasAuthoritativeCudaBfp8Gradient);
            }
            else
            {
                AssertPureGradientPublished(tokenTable);
                AssertPureGradientPublished(positionTable);
            }
            AssertClose(expectedTokenGradient, tokenTable.Grad, 1e-5f);
            AssertClose(expectedPositionGradient, positionTable.Grad, 1e-5f);
            output.InvalidateCudaBuffers();
            tokenTable.InvalidateCudaBuffers();
            positionTable.InvalidateCudaBuffers();
        });
    }

    [Fact]
    public void LargeTableLookupAllocatesNoFullTableDecodeAndReleasesResources()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        Bfp8QuantizationDescriptor descriptor =
            Bfp8QuantizationDescriptor.TensorWide;
        WithCuda(descriptor, () =>
        {
            const int rows = 4096;
            const int width = 257;
            int[] indices = [4095, 3, 2001];
            Tensor table = Tensor.FromBfp8(
                Values(rows * width, 13, 0.8f),
                [rows, width],
                descriptor,
                "large.tokens");
            table.to(new TorchDevice(TensorDevice.Cuda, 0));
            _ = table.EnsureCudaBfp8Buffer(0);
            Tensor.ClearCudaFloatBufferPool(0);
            NativeCudaAllocationTelemetry before =
                NativeCudaRuntime.AllocationTelemetry;
            CudaBfp8GemmTelemetrySnapshot decodeBefore =
                CudaBfp8GemmTelemetry.Snapshot;
            Tensor output;
            using (AutogradContext.NoGrad())
            {
                output = table.EmbeddingLookup(indices, indices.Length);
            }
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaAllocationTelemetry allocated =
                NativeCudaRuntime.AllocationTelemetry - before;
            CudaBfp8GemmTelemetrySnapshot decode =
                CudaBfp8GemmTelemetry.Snapshot - decodeBefore;
            Assert.Equal(0, decode.BFloat16DecodeCacheMisses);
            Assert.True(
                allocated.AllocationBytes < table.Numel * sizeof(ushort),
                $"Embedding allocated {allocated.AllocationBytes} bytes for " +
                $"a {table.Numel}-element source table.");

            NativeCudaAllocationTelemetry beforeRelease =
                NativeCudaRuntime.AllocationTelemetry;
            output.InvalidateCudaBuffers();
            Tensor.ClearCudaFloatBufferPool(0);
            NativeCudaAllocationTelemetry released =
                NativeCudaRuntime.AllocationTelemetry - beforeRelease;
            Assert.Equal(allocated.AllocationCount, released.FreeCount);
            Assert.Equal(allocated.AllocationBytes, released.FreeBytes);
            table.InvalidateCudaBuffers();
        });
    }

    [Fact]
    public void MixedBfp8AndFloatPositionTablesRefuseFullDecodeFallback()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(Bfp8QuantizationDescriptor.Mix8_32, () =>
        {
            Tensor tokens = Tensor.FromBfp8(
                Values(12, 1, 0.5f),
                [4, 3],
                Bfp8QuantizationDescriptor.Mix8_32);
            Tensor positions = new(Values(6, 7, 0.4f), [2, 3]);
            MoveToCuda(tokens, positions);
            NotSupportedException exception = Assert.Throws<NotSupportedException>(
                () => tokens.EmbeddingLookupWithPositions(
                    positions, [0, 1], batchSize: 1, sequenceLength: 2));
            Assert.Contains("full table", exception.Message);
            tokens.InvalidateCudaBuffers();
            positions.InvalidateCudaBuffers();
        });
    }

    private static float[] Quantized(
        float[] source,
        Bfp8QuantizationDescriptor descriptor)
    {
        Bfp8EncodedStorage encoded =
            Bfp8QuantizationCodec.Default.Encode(source, descriptor);
        var decoded = new float[source.Length];
        Bfp8QuantizationCodec.Default.Decode(
            encoded.Payload.Span,
            encoded.Scales.Span,
            descriptor,
            decoded);
        return decoded;
    }

    private static void AssertPureGradientPublished(Tensor tensor)
    {
        Assert.True(tensor.HasAuthoritativeCudaBfp8Gradient);
        CudaBfp8BufferView gradient =
            tensor.EnsureCudaBfp8GradientBuffer(0);
        Assert.Equal(
            Bfp8QuantizationDescriptor.TensorWide,
            gradient.Descriptor);
        Assert.Equal(Bfp8ScaleGranularity.Tensor,
            gradient.Descriptor.Granularity);
        Assert.Equal(1, gradient.Scales.Length);
    }

    private static float[] Gather(
        float[] table,
        int[] indices,
        int width)
    {
        var output = new float[checked(indices.Length * width)];
        for (int position = 0; position < indices.Length; position++)
        {
            Array.Copy(
                table,
                indices[position] * width,
                output,
                position * width,
                width);
        }
        return output;
    }

    private static float[] GatherWithPositions(
        float[] tokens,
        float[] positions,
        int[] indices,
        int sequenceLength,
        int width)
    {
        var output = new float[checked(indices.Length * width)];
        for (int position = 0; position < indices.Length; position++)
        {
            for (int column = 0; column < width; column++)
            {
                output[position * width + column] =
                    tokens[indices[position] * width + column] +
                    positions[(position % sequenceLength) * width + column];
            }
        }
        return output;
    }

    private static float[] GatherGradient(
        int rows,
        int width,
        int[] indices,
        float[] seed)
    {
        var gradient = new float[checked(rows * width)];
        for (int position = 0; position < indices.Length; position++)
        {
            for (int column = 0; column < width; column++)
            {
                gradient[indices[position] * width + column] +=
                    seed[position * width + column];
            }
        }
        return gradient;
    }

    private static float[] PositionGradient(
        int batch,
        int sequence,
        int width,
        float[] seed)
    {
        var gradient = new float[checked(sequence * width)];
        for (int batchIndex = 0; batchIndex < batch; batchIndex++)
        {
            for (int position = 0; position < sequence; position++)
            {
                for (int column = 0; column < width; column++)
                {
                    gradient[position * width + column] += seed[
                        (batchIndex * sequence + position) * width + column];
                }
            }
        }
        return gradient;
    }

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index =>
                MathF.Sin((index + offset) * 0.137f) * scale +
                MathF.Cos((index + offset) * 0.043f) * scale * 0.31f)
            .ToArray();

    private static void MoveToCuda(params Tensor[] tensors)
    {
        foreach (Tensor tensor in tensors)
            tensor.to(new TorchDevice(TensorDevice.Cuda, 0));
    }

    private static void WithCuda(
        Bfp8QuantizationDescriptor descriptor,
        Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(
                    descriptor.Granularity == Bfp8ScaleGranularity.Tensor
                        ? PrecisionPolicy.Bfp8
                        : PrecisionPolicy.Mix8_32);
            action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private static void AssertClose(
        IReadOnlyList<float> expected,
        IReadOnlyList<float> actual,
        float tolerance)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int index = 0; index < expected.Count; index++)
        {
            Assert.InRange(
                MathF.Abs(expected[index] - actual[index]),
                0f,
                tolerance);
        }
    }
}
