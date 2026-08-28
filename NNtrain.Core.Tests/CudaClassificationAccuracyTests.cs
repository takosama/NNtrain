using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaClassificationAccuracyTests
{
    [Theory]
    [InlineData(PrecisionMode.Float32)]
    [InlineData(PrecisionMode.BFloat16)]
    [InlineData(PrecisionMode.Mix16_32)]
    [InlineData(PrecisionMode.Bfp8)]
    [InlineData(PrecisionMode.Mix8_32)]
    public void ResidentArgmaxReducesTailClassesAndReadsOneInt32(
        PrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int samples = 7;
        const int classes = 259;
        PrecisionPolicy precision = PrecisionPolicy.For(mode);
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        Tensor? logits = null;
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable policy =
                TensorExecutionContext.PushPrecisionPolicy(precision);

            float[] source = CreateLogits(samples, classes);
            logits = CreateTensor(source, samples, classes, mode);
            float[] physical = logits.Data.ToArray();
            int[] argmax = Enumerable.Range(0, samples)
                .Select(sample => ArgMax(physical, sample * classes, classes))
                .ToArray();
            int[] targets = argmax
                .Select((value, sample) => sample % 2 == 0
                    ? value
                    : (value + 1) % classes)
                .ToArray();
            int expected = (samples + 1) / 2;

            logits.to(new TorchDevice(TensorDevice.Cuda, 0));
            _ = EnsureResident(logits);

            // Prime P/Invoke, the pinned scalar slot, and transient pools.
            using (CudaClassificationCorrectCountReadback warmup =
                logits.BeginCudaClassificationCorrectCount(
                    targets,
                    classes,
                    0))
            {
                Assert.Equal(expected, warmup.CompleteAndReturn());
            }

            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;
            using IDisposable guard = DeviceTransferGuard.EnterTrainingStep(
                cudaDeviceCount: 1,
                maximumDeviceToHostCopies: 1);
            using CudaClassificationCorrectCountReadback readback =
                logits.BeginCudaClassificationCorrectCount(
                    targets,
                    classes,
                    0);
            int actual = readback.CompleteAndReturn();
            DeviceTransferSnapshot guarded = Assert.NotNull(
                DeviceTransferGuard.CurrentSnapshot);
            NativeCudaTransferTelemetry transfer =
                NativeCudaRuntime.TransferTelemetry - before;

            Assert.Equal(expected, actual);
            Assert.Equal(1, guarded.HostToDeviceCopyCount);
            Assert.Equal(samples * sizeof(int), guarded.HostToDeviceBytes);
            Assert.Equal(1, guarded.DeviceToHostCopyCount);
            Assert.Equal(sizeof(int), guarded.DeviceToHostBytes);
            Assert.Equal(samples * sizeof(int), transfer.HostToDeviceBytes);
            Assert.Equal(sizeof(int), transfer.DeviceToHostBytes);
            Assert.True(
                transfer.DeviceToHostBytes <
                    (long)samples * classes * sizeof(float));
        }
        finally
        {
            logits?.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }

    [Fact]
    public void StrictArgmaxKeepsFirstTieAndClassZeroNanSemantics()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        Tensor? logits = null;
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable policy = TensorExecutionContext
                .PushPrecisionPolicy(PrecisionPolicy.Float32);
            float[] values =
            [
                float.NaN, 100f, 200f, 300f,
                2f, 7f, 7f, float.NaN,
            ];
            logits = new Tensor(values, [2, 4]);
            logits.to(new TorchDevice(TensorDevice.Cuda, 0));
            _ = logits.EnsureCudaFloat32Buffer(0);

            using CudaClassificationCorrectCountReadback readback =
                logits.BeginCudaClassificationCorrectCount(
                    [0, 1],
                    classCount: 4,
                    deviceIndex: 0);

            Assert.Equal(2, readback.CompleteAndReturn());
        }
        finally
        {
            logits?.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }

    private static Tensor CreateTensor(
        float[] source,
        int samples,
        int classes,
        PrecisionMode mode)
        => mode switch
        {
            PrecisionMode.Bfp8 => Tensor.FromBfp8(
                source,
                [samples, classes],
                Bfp8QuantizationDescriptor.TensorWide),
            PrecisionMode.Mix8_32 => Tensor.FromBfp8(
                source,
                [samples, classes],
                Bfp8QuantizationDescriptor.Mix8_32),
            PrecisionMode.BFloat16 or PrecisionMode.Mix16_32 => new Tensor(
                source,
                [samples, classes],
                dtype: TensorDType.BFloat16),
            _ => new Tensor(source, [samples, classes]),
        };

    private static object EnsureResident(Tensor tensor)
        => tensor.DType switch
        {
            TensorDType.BFloat16 => tensor.EnsureCudaBFloat16Buffer(0),
            TensorDType.Bfp8 => tensor.EnsureCudaBfp8Buffer(0),
            _ => tensor.EnsureCudaFloat32Buffer(0),
        };

    private static float[] CreateLogits(int samples, int classes)
    {
        float[] values = Enumerable.Range(0, samples * classes)
            .Select(index =>
            {
                int block = index / Bfp8QuantizationDescriptor.DefaultBlockSize;
                float amplitude = 0.25f + (block % 9) * 1.7f;
                return MathF.Sin(index * 0.071f) * amplitude;
            })
            .ToArray();

        // Cover class tails, a 256-thread boundary, and stable first-index
        // tie-breaking. Physical-storage rounding is applied before the CPU
        // reference is calculated.
        values[0 * classes + 0] = 70f;
        values[1 * classes + 258] = 90f;
        values[2 * classes + 127] = 85f;
        values[2 * classes + 128] = 85f;
        values[3 * classes + 256] = 110f;
        values[4 * classes + 17] = 95f;
        values[5 * classes + 201] = 105f;
        values[6 * classes + 77] = 100f;
        return values;
    }

    private static int ArgMax(float[] values, int offset, int count)
    {
        int result = 0;
        float best = values[offset];
        for (int index = 1; index < count; index++)
        {
            if (values[offset + index] > best)
            {
                best = values[offset + index];
                result = index;
            }
        }
        return result;
    }
}
