using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaEmbeddingBackwardReductionTests
{
    [Theory]
    [InlineData(TensorDType.Float32, false)]
    [InlineData(TensorDType.BFloat16, false)]
    [InlineData(TensorDType.Bfp8, false)]
    [InlineData(TensorDType.Bfp8, true)]
    public void CollisionHeavyTokenAndPositionGradientsUseOwnerReduction(
        TensorDType dtype,
        bool blockScaled)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(dtype, blockScaled, () =>
        {
            const int tokenRows = 11;
            const int batch = 5;
            const int sequence = 7;
            const int width = 257;
            int[] indices = Enumerable.Range(0, batch * sequence)
                .Select(position => position % 5 == 0
                    ? 9
                    : (position * 3 + position / sequence) % 4)
                .ToArray();
            float[] seed = Values(indices.Length * width, 17, 0.031f);
            float[] referenceSeed = dtype == TensorDType.BFloat16
                ? BFloat16(seed)
                : seed;
            float[] expectedTokens = TokenGradient(
                tokenRows, width, indices, referenceSeed);
            float[] expectedPositions = PositionGradient(
                batch, sequence, width, referenceSeed);
            if (dtype == TensorDType.BFloat16)
            {
                expectedTokens = BFloat16(expectedTokens);
                expectedPositions = BFloat16(expectedPositions);
            }
            if (dtype == TensorDType.Bfp8 && !blockScaled)
            {
                expectedTokens = Quantized(
                    expectedTokens,
                    Bfp8QuantizationDescriptor.TensorWide);
                expectedPositions = Quantized(
                    expectedPositions,
                    Bfp8QuantizationDescriptor.TensorWide);
            }

            Tensor tokens = CreateTensor(
                Values(tokenRows * width, 3, 0.2f),
                [tokenRows, width],
                dtype,
                blockScaled);
            Tensor positions = CreateTensor(
                Values(sequence * width, 41, 0.1f),
                [sequence, width],
                dtype,
                blockScaled);
            tokens.to(new TorchDevice(TensorDevice.Cuda, 0));
            positions.to(new TorchDevice(TensorDevice.Cuda, 0));
            Tensor output = tokens.EmbeddingLookupWithPositions(
                positions,
                indices,
                batch,
                sequence);
            CudaEmbeddingBackwardTelemetrySnapshot before =
                CudaEmbeddingBackwardTelemetry.Snapshot;

            output.BackwardAndRelease(seed);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            CudaEmbeddingBackwardTelemetrySnapshot telemetry =
                CudaEmbeddingBackwardTelemetry.Snapshot - before;
            long gradientValues = checked((long)indices.Length * width);
            Assert.Equal(0, telemetry.ReducedLookupExecutions);
            Assert.Equal(1, telemetry.ReducedLookupWithPositionsExecutions);
            Assert.Equal(gradientValues, telemetry.GradientValuesAccumulated);
            Assert.Equal(
                checked(gradientValues * 2),
                telemetry.LegacyTableAtomicAddsAvoided);
            Assert.Equal(0, telemetry.ReducedTableAtomicAdds);
            Assert.Equal(
                checked((long)indices.Length * 2),
                telemetry.HashBookkeepingAtomicLowerBound);
            Assert.Equal(
                CudaEmbeddingBackwardDispatcher.GetWorkspaceIntCount(
                    indices.Length),
                telemetry.WorkspaceIntsRented);
            AssertClose(expectedTokens, tokens.Grad, 1e-5f);
            AssertClose(expectedPositions, positions.Grad, 1e-5f);

            output.InvalidateCudaBuffers();
            tokens.InvalidateCudaBuffers();
            positions.InvalidateCudaBuffers();
        });
    }

    [Fact]
    public void TailWidthsAndRepeatedBackwardAccumulateWithoutOutOfBounds()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(TensorDType.Float32, blockScaled: false, () =>
        {
            int[] widths = [1, 3, 4, 31, 128, 257];
            foreach (int width in widths)
            {
                const int tokenRows = 13;
                const int batch = 3;
                const int sequence = 11;
                int[] indices = Enumerable.Range(0, batch * sequence)
                    .Select(position => position % 4 == 0
                        ? 12
                        : position % 3)
                    .ToArray();
                float[] firstSeed = Values(
                    indices.Length * width, width + 3, 0.013f);
                float[] secondSeed = Values(
                    indices.Length * width, width + 71, 0.009f);
                float[] combinedSeed = firstSeed.Zip(
                        secondSeed,
                        static (left, right) => left + right)
                    .ToArray();
                float[] expectedTokens = TokenGradient(
                    tokenRows, width, indices, combinedSeed);
                float[] expectedPositions = PositionGradient(
                    batch, sequence, width, combinedSeed);

                Tensor tokens = new(
                    Values(tokenRows * width, 5, 0.2f),
                    [tokenRows, width]);
                Tensor positions = new(
                    Values(sequence * width, 29, 0.1f),
                    [sequence, width]);
                tokens.to(new TorchDevice(TensorDevice.Cuda, 0));
                positions.to(new TorchDevice(TensorDevice.Cuda, 0));
                Tensor first = tokens.EmbeddingLookupWithPositions(
                    positions, indices, batch, sequence);
                Tensor second = tokens.EmbeddingLookupWithPositions(
                    positions, indices, batch, sequence);

                first.BackwardAndRelease(firstSeed);
                second.BackwardAndRelease(secondSeed);
                ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

                AssertClose(expectedTokens, tokens.Grad, 2e-5f);
                AssertClose(expectedPositions, positions.Grad, 2e-5f);
                first.InvalidateCudaBuffers();
                second.InvalidateCudaBuffers();
                tokens.InvalidateCudaBuffers();
                positions.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void ProductionShapeTelemetryEliminatesBothGradientAtomicStreams()
    {
        const int positions = 36 * 512;
        const int width = 1024;
        CudaEmbeddingBackwardTelemetrySnapshot before =
            CudaEmbeddingBackwardTelemetry.Snapshot;

        CudaEmbeddingBackwardTelemetry.Record(
            includesPositions: true,
            positions,
            width,
            CudaEmbeddingBackwardDispatcher.GetWorkspaceIntCount(positions));

        CudaEmbeddingBackwardTelemetrySnapshot telemetry =
            CudaEmbeddingBackwardTelemetry.Snapshot - before;
        Assert.Equal(37_748_736, telemetry.LegacyTableAtomicAddsAvoided);
        Assert.Equal(0, telemetry.ReducedTableAtomicAdds);
        Assert.Equal(36_864, telemetry.HashBookkeepingAtomicLowerBound);
        Assert.Equal(167_937, telemetry.WorkspaceIntsRented);
    }

    private static Tensor CreateTensor(
        float[] values,
        int[] shape,
        TensorDType dtype,
        bool blockScaled)
        => dtype == TensorDType.Bfp8
            ? Tensor.FromBfp8(
                values,
                shape,
                blockScaled
                    ? Bfp8QuantizationDescriptor.Mix8_32
                    : Bfp8QuantizationDescriptor.TensorWide)
            : new Tensor(values, shape, dtype: dtype);

    private static float[] TokenGradient(
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

    private static float[] Quantized(
        float[] values,
        Bfp8QuantizationDescriptor descriptor)
    {
        Bfp8EncodedStorage encoded =
            Bfp8QuantizationCodec.Default.Encode(values, descriptor);
        var decoded = new float[values.Length];
        Bfp8QuantizationCodec.Default.Decode(
            encoded.Payload.Span,
            encoded.Scales.Span,
            descriptor,
            decoded);
        return decoded;
    }

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index =>
                MathF.Sin((index + offset) * 0.071f) * scale
                + MathF.Cos((index + offset) * 0.019f) * scale * 0.37f)
            .ToArray();

    private static void WithCuda(
        TensorDType dtype,
        bool blockScaled,
        Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            PrecisionPolicy policy = dtype switch
            {
                TensorDType.Bfp8 when blockScaled => PrecisionPolicy.Mix8_32,
                TensorDType.Bfp8 => PrecisionPolicy.Bfp8,
                TensorDType.BFloat16 => PrecisionPolicy.BFloat16,
                _ => PrecisionPolicy.Float32,
            };
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(policy);
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

    private static float[] BFloat16(IEnumerable<float> values)
        => values.Select(TensorStorageCodec.RoundToBFloat16).ToArray();
}
