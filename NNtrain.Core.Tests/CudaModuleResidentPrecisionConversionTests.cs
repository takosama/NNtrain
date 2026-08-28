using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaModuleResidentPrecisionConversionTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.BFloat16)]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Bfp8)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void PrecisionConversionStaysResidentAndPreservesEveryReplica(
        TensorPrecisionMode mode)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        Parameter[] parameters = [];
        try
        {
            int[] devices = Enumerable.Range(
                    0,
                    Math.Min(2, Tensor.CudaDeviceCount))
                .ToArray();
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            Tensor.CudaDeviceIndices = devices;
            var model = new Linear(16, 8, new Random(811));
            parameters = model.Parameters().ToArray();
            Tensor[] identities = parameters
                .Select(static parameter => parameter.T)
                .ToArray();
            float[][] original = identities
                .Select(static tensor => tensor.Data.ToArray())
                .ToArray();
            var optimizer = new AdamW(
                parameters,
                new AdamWOptions
                {
                    LearningRate = 1e-3f,
                    WeightDecay = 0f,
                });

            Tensor.ExecutionDevice = TensorDevice.Cuda;
            model.to("cuda");
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;

            model.to(mode, bfp8_block_size: 32);
            NativeCudaRuntime.SynchronizeDeviceComputeStream(devices[0]);

            NativeCudaTransferTelemetry conversion =
                NativeCudaRuntime.TransferTelemetry - before;
            Assert.Equal(0, conversion.HostToDeviceCopyCount);
            Assert.Equal(0, conversion.HostToDeviceBytes);
            Assert.Equal(0, conversion.DeviceToHostCopyCount);
            Assert.Equal(0, conversion.DeviceToHostBytes);
            Assert.Equal(parameters, optimizer.Parameters);
            Assert.Equal(identities, parameters.Select(p => p.T).ToArray());
            Assert.All(
                parameters,
                parameter =>
                {
                    Assert.Equal(TensorDevice.Cuda, parameter.T.Device);
                    Assert.Equal(devices[0], parameter.T.device.Index);
                    Assert.Equal(
                        mode.ToStorageDType(),
                        parameter.T.DType);
                    Assert.Equal(
                        devices,
                        parameter.T.GetResidentCudaDeviceIndices());
                    foreach (int deviceIndex in devices)
                    {
                        AssertPhysicalReplica(parameter.T, mode, deviceIndex);
                        Assert.Equal(
                            mode is TensorPrecisionMode.Mix16_32
                                or TensorPrecisionMode.Mix8_32,
                            parameter.T.HasCudaMasterFloat32Buffer(deviceIndex));
                    }
                });

            // A post-conversion backward must continue to publish a CUDA
            // gradient in the target precision contract.
            using (TensorExecutionContext.PushPrecisionPolicy(Policy(mode)))
            {
                Tensor leaf = parameters[0].T;
                leaf.ZeroGrad();
                leaf.BackwardAndRelease(
                    Enumerable.Repeat(1f, leaf.Numel).ToArray());
                Assert.All(leaf.Grad, value => Assert.InRange(value, 0.99f, 1.01f));
            }

            nint[][] masterPointers = parameters
                .Select(parameter => devices
                    .Where(parameter.T.HasCudaMasterFloat32Buffer)
                    .Select(deviceIndex => parameter.T
                        .EnsureCudaMasterFloat32Buffer(deviceIndex)
                        .NativePtr)
                    .ToArray())
                .ToArray();
            NativeCudaTransferTelemetry beforeState =
                NativeCudaRuntime.TransferTelemetry;
            ModuleState state = model.state_dict();
            NativeCudaTransferTelemetry stateTransfers =
                NativeCudaRuntime.TransferTelemetry - beforeState;
            if (mode != TensorPrecisionMode.Float32)
                Assert.True(stateTransfers.DeviceToHostBytes > 0);
            for (int index = 0; index < parameters.Length; index++)
            {
                float tolerance = mode switch
                {
                    TensorPrecisionMode.Bfp8 => 0.12f,
                    TensorPrecisionMode.BFloat16 => 0.01f,
                    _ => 0f,
                };
                AssertValuesClose(
                    original[index], state.Parameters[index].Values, tolerance);
                nint[] afterPointers = devices
                    .Where(parameters[index].T.HasCudaMasterFloat32Buffer)
                    .Select(deviceIndex => parameters[index].T
                        .EnsureCudaMasterFloat32Buffer(deviceIndex)
                        .NativePtr)
                    .ToArray();
                Assert.Equal(masterPointers[index], afterPointers);
            }
        }
        finally
        {
            foreach (Parameter parameter in parameters)
                parameter.T.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }

    [Fact]
    public void StringAliasesKeepPrecisionConversionsOnCudaUntilExplicitCpuMove()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        Parameter[] parameters = [];
        try
        {
            int[] devices = Enumerable.Range(
                    0,
                    Math.Min(2, Tensor.CudaDeviceCount))
                .ToArray();
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            Tensor.CudaDeviceIndices = devices;
            var model = new Linear(8, 4, new Random(823));
            parameters = model.Parameters().ToArray();

            Assert.Same(model, model.to("cuda"));
            Assert.All(parameters, p => Assert.Equal(TensorDevice.Cuda, p.T.Device));
            NativeCudaTransferTelemetry beforeAuto =
                NativeCudaRuntime.TransferTelemetry;
            Assert.Same(model, model.to("auto"));
            NativeCudaTransferTelemetry autoTransfers =
                NativeCudaRuntime.TransferTelemetry - beforeAuto;
            Assert.Equal(0, autoTransfers.DeviceToHostBytes);

            (string Alias, TensorPrecisionMode Mode)[] aliases =
            [
                ("float32", TensorPrecisionMode.Float32),
                ("bfloat16", TensorPrecisionMode.BFloat16),
                ("fp16_32", TensorPrecisionMode.Mix16_32),
                ("mix16_32", TensorPrecisionMode.Mix16_32),
                ("bfp8", TensorPrecisionMode.Bfp8),
                ("mix8_32", TensorPrecisionMode.Mix8_32),
            ];
            foreach ((string alias, TensorPrecisionMode expected) in aliases)
            {
                NativeCudaTransferTelemetry before =
                    NativeCudaRuntime.TransferTelemetry;
                Assert.Same(model, model.to(alias));
                NativeCudaTransferTelemetry conversion =
                    NativeCudaRuntime.TransferTelemetry - before;
                Assert.Equal(0, conversion.HostToDeviceBytes);
                Assert.Equal(0, conversion.DeviceToHostBytes);
                Assert.Equal(expected, model.PrecisionMode);
                Assert.All(
                    parameters,
                    parameter => Assert.Equal(
                        devices,
                        parameter.T.GetResidentCudaDeviceIndices()));
            }

            NativeCudaTransferTelemetry beforeCpu =
                NativeCudaRuntime.TransferTelemetry;
            Assert.Same(model, model.to("cpu"));
            NativeCudaTransferTelemetry cpuMove =
                NativeCudaRuntime.TransferTelemetry - beforeCpu;
            Assert.True(cpuMove.DeviceToHostBytes > 0);
            Assert.All(parameters, p => Assert.Equal(TensorDevice.Cpu, p.T.Device));
        }
        finally
        {
            foreach (Parameter parameter in parameters)
                parameter.T.InvalidateCudaBuffers();
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDevices;
        }
    }

    private static void AssertPhysicalReplica(
        Tensor tensor,
        TensorPrecisionMode mode,
        int deviceIndex)
    {
        switch (mode)
        {
            case TensorPrecisionMode.Float32:
                _ = tensor.EnsureCudaFloat32Buffer(deviceIndex);
                break;
            case TensorPrecisionMode.BFloat16:
            case TensorPrecisionMode.Mix16_32:
                _ = tensor.EnsureCudaBFloat16Buffer(deviceIndex);
                break;
            case TensorPrecisionMode.Bfp8:
                Assert.Equal(
                    Bfp8QuantizationDescriptor.TensorWide,
                    tensor.EnsureCudaBfp8Buffer(deviceIndex).Descriptor);
                break;
            case TensorPrecisionMode.Mix8_32:
                CudaBfp8BufferView view =
                    tensor.EnsureCudaBfp8Buffer(deviceIndex);
                Assert.Equal(Bfp8ScaleGranularity.Block, view.Descriptor.Granularity);
                Assert.Equal(32, view.Descriptor.BlockSize);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(mode));
        }
    }

    private static PrecisionPolicy Policy(TensorPrecisionMode mode)
        => mode switch
        {
            TensorPrecisionMode.Float32 => PrecisionPolicy.Float32,
            TensorPrecisionMode.BFloat16 => PrecisionPolicy.BFloat16,
            TensorPrecisionMode.Mix16_32 => PrecisionPolicy.Mix16_32,
            TensorPrecisionMode.Bfp8 => PrecisionPolicy.Bfp8,
            TensorPrecisionMode.Mix8_32 => PrecisionPolicy.Mix8_32,
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };

    private static void AssertValuesClose(
        IReadOnlyList<float> expected,
        IReadOnlyList<float> actual,
        float tolerance)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int index = 0; index < expected.Count; index++)
        {
            Assert.InRange(
                actual[index],
                expected[index] - tolerance,
                expected[index] + tolerance);
        }
    }
}
