using NNtrain;
using NNtrain.Cuda.Interop;
using Xunit;

public sealed class CudaTopKSamplingTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.BFloat16)]
    [InlineData(TensorPrecisionMode.Bfp8)]
    public void TwoStageTopKMatchesStableReferenceForOffsetTailAndTies(
        TensorPrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            const int length = 311;
            const int offset = 17;
            const int count = 257;
            const int k = 64;
            float[] source = Enumerable.Range(0, length)
                .Select(index => MathF.Sin(index * 0.173f) * 3f
                    + MathF.Cos(index * 0.071f))
                .ToArray();
            // Exact collisions straddle reduction partitions and the tail.
            source[offset + 3] = 9f;
            source[offset + 91] = 9f;
            source[offset + 256] = 9f;
            Tensor tensor = Create(source, mode);
            try
            {
                tensor.to(new TorchDevice(TensorDevice.Cuda, 0));
                float[] physical = tensor.Data.ToArray();
                if (mode == TensorPrecisionMode.Bfp8)
                {
                    physical = physical
                        .Select(TensorStorageCodec.RoundToBFloat16)
                        .ToArray();
                }
                _ = EnsureResident(tensor);
                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;

                CudaTopKCandidate[] actual = tensor
                    .ReadCudaTopK(offset, count, k, 0)
                    .Candidates;

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - before;
                int[] expected = Enumerable.Range(0, count)
                    .OrderByDescending(index => physical[offset + index])
                    .ThenBy(index => index)
                    .Take(k)
                    .ToArray();
                Assert.Equal(expected, actual.Select(item => item.Index));
                for (int index = 0; index < k; index++)
                {
                    Assert.Equal(
                        physical[offset + expected[index]],
                        actual[index].Value);
                }
                Assert.Equal(3, actual[0].Index);
                Assert.Equal(91, actual[1].Index);
                Assert.Equal(256, actual[2].Index);
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(k * 2L * sizeof(float),
                    transfer.DeviceToHostBytes);
            }
            finally
            {
                tensor.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void GreedyHandlesInfiniteNanAndEqualValuesDeterministically()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            float[] source =
            [
                float.NaN,
                -3f,
                float.PositiveInfinity,
                7f,
                float.PositiveInfinity,
                7f,
                float.NegativeInfinity,
            ];
            var tensor = new Tensor(source, [source.Length]);
            try
            {
                tensor.to(new TorchDevice(TensorDevice.Cuda, 0));
                _ = tensor.EnsureCudaFloat32Buffer(0);

                CudaTopKCandidate[] actual = tensor
                    .ReadCudaTopK(0, source.Length, source.Length, 0)
                    .Candidates;

                Assert.Equal<int>([2, 4, 3, 5, 1, 6, 0],
                    actual.Select(item => item.Index));
                Assert.True(float.IsNaN(actual[^1].Value));
            }
            finally
            {
                tensor.InvalidateCudaBuffers();
            }
        });
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(40)]
    [InlineData(64)]
    public void TransfersExactlyOneEightBytePairPerCandidate(int k)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            float[] values = Enumerable.Range(0, 11500)
                .Select(index => MathF.Sin(index * 0.013f))
                .ToArray();
            var tensor = new Tensor(
                values,
                [values.Length],
                dtype: TensorDType.BFloat16);
            try
            {
                tensor.to(new TorchDevice(TensorDevice.Cuda, 0));
                _ = tensor.EnsureCudaBFloat16Buffer(0);
                _ = tensor.ReadCudaTopK(0, values.Length, k, 0);
                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;

                CudaTopKSelection selection = tensor.ReadCudaTopK(
                    0, values.Length, k, 0);

                NativeCudaTransferTelemetry transfer =
                    NativeCudaRuntime.TransferTelemetry - before;
                Assert.Equal(k, selection.Candidates.Length);
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(8L * k, transfer.DeviceToHostBytes);
            }
            finally
            {
                tensor.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void LanguageModelSamplingReadsOnlyGreedyOrTopKCandidates()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            const int vocabulary = 11500;
            float[] values = Enumerable.Range(0, vocabulary)
                .Select(index => MathF.Sin(index * 0.017f) * 3f
                    + MathF.Cos(index * 0.003f))
                .ToArray();
            var tensor = new Tensor(
                values,
                [vocabulary],
                dtype: TensorDType.BFloat16);
            try
            {
                tensor.to(new TorchDevice(TensorDevice.Cuda, 0));
                _ = tensor.EnsureCudaBFloat16Buffer(0);
                // Prime the native gateway and reusable workspace so the
                // measured deltas contain sampling transfers only.
                _ = tensor.ReadCudaTopK(0, vocabulary, 1, 0);
                _ = tensor.ReadCudaTopK(0, vocabulary, 40, 0);
                NativeCudaAllocationTelemetry beforeAllocation =
                    NativeCudaRuntime.AllocationTelemetry;

                NativeCudaTransferTelemetry beforeGreedy =
                    NativeCudaRuntime.TransferTelemetry;
                int greedy = LanguageModel.SampleLogits(
                    tensor,
                    0,
                    vocabulary,
                    temperature: 0f,
                    topK: 40,
                    new Random(41));
                NativeCudaTransferTelemetry greedyTransfer =
                    NativeCudaRuntime.TransferTelemetry - beforeGreedy;

                int expectedGreedy = Enumerable.Range(0, vocabulary)
                    .OrderByDescending(index =>
                        TensorStorageCodec.RoundToBFloat16(values[index]))
                    .ThenBy(index => index)
                    .First();
                Assert.Equal(expectedGreedy, greedy);
                Assert.Equal(0, greedyTransfer.HostToDeviceBytes);
                Assert.Equal(8, greedyTransfer.DeviceToHostBytes);

                HashSet<int> expectedTop40 = Enumerable.Range(0, vocabulary)
                    .OrderByDescending(index =>
                        TensorStorageCodec.RoundToBFloat16(values[index]))
                    .ThenBy(index => index)
                    .Take(40)
                    .ToHashSet();
                NativeCudaTransferTelemetry beforeSample =
                    NativeCudaRuntime.TransferTelemetry;
                int sampled = LanguageModel.SampleLogits(
                    tensor,
                    0,
                    vocabulary,
                    temperature: 0.8f,
                    topK: 40,
                    new Random(42));
                NativeCudaTransferTelemetry sampleTransfer =
                    NativeCudaRuntime.TransferTelemetry - beforeSample;

                Assert.Contains(sampled, expectedTop40);
                Assert.Equal(0, sampleTransfer.HostToDeviceBytes);
                Assert.Equal(40L * 8L, sampleTransfer.DeviceToHostBytes);
                NativeCudaAllocationTelemetry allocation =
                    NativeCudaRuntime.AllocationTelemetry - beforeAllocation;
                Assert.Equal(0, allocation.AllocationCount);
                Assert.Equal(0, allocation.FreeCount);
            }
            finally
            {
                tensor.InvalidateCudaBuffers();
            }
        });
    }

    private static Tensor Create(
        float[] values,
        TensorPrecisionMode mode)
        => mode switch
        {
            TensorPrecisionMode.Bfp8 => Tensor.FromBfp8(
                values,
                [values.Length],
                Bfp8QuantizationDescriptor.TensorWide),
            _ => new Tensor(
                values,
                [values.Length],
                dtype: mode.ToStorageDType()),
        };

    private static object EnsureResident(Tensor tensor)
        => tensor.DType switch
        {
            TensorDType.BFloat16 => tensor.EnsureCudaBFloat16Buffer(0),
            TensorDType.Bfp8 => tensor.EnsureCudaBfp8Buffer(0),
            _ => tensor.EnsureCudaFloat32Buffer(0),
        };

    private static void WithCuda(Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }
}
