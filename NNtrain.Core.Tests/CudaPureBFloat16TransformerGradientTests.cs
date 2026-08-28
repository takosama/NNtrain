using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class CudaPureBFloat16TransformerGradientTests
{
    [Fact]
    public void TransformerHotBackwardPublishesEveryParentAsResidentBFloat16()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(PrecisionPolicy.BFloat16, () =>
        {
            Tensor matrix = BFloat16(Values(16 * 16, 3), [16, 16]);
            Tensor batched = BFloat16(Values(2 * 16 * 16, 7), [2, 16, 16]);
            Tensor linearInput = BFloat16(Values(4 * 16, 11), [4, 16]);
            Tensor weight = BFloat16(Values(16 * 16, 13), [16, 16]);
            Tensor bias = BFloat16(Values(16, 17), [16]);
            Tensor gamma = BFloat16(
                Enumerable.Repeat(1f, 16).ToArray(), [16]);
            Tensor beta = BFloat16(new float[16], [16]);
            Tensor branch = BFloat16(Values(4 * 16, 19), [4, 16]);

            Tensor matrixLoss = matrix.MatMul(matrix).Sum()
                + matrix.MatMulTransposedRight(matrix).Sum()
                + batched.BatchedMatMul(batched).Sum()
                + batched.BatchedMatMulTransposedRight(batched).Sum();
            Tensor linear = linearInput.LinearLastDim(
                weight,
                bias,
                applyRelu: false);
            Tensor normalized = linear.LayerNormLastDim(gamma, beta);
            Tensor dropped = normalized.Dropout(0.2f, new Random(31));
            Tensor residualDropout = normalized.AddDropout(
                normalized,
                0.15f,
                new Random(37));
            Tensor fused = linear.AddDropoutLayerNormLastDim(
                branch,
                gamma,
                beta,
                0.1f,
                new Random(41));
            Tensor loss = matrixLoss
                + dropped.Sum()
                + residualDropout.Sum()
                + fused.Sum();

            using (DeviceTransferGuard.EnterTrainingStep(101))
            {
                loss.BackwardAndRelease();
                DeviceTransferSnapshot transfer = Assert.NotNull(
                    DeviceTransferGuard.CurrentSnapshot);
                Assert.Equal(0, transfer.HostToDeviceCopyCount);
                Assert.Equal(0, transfer.HostToDeviceBytes);
                Assert.Equal(0, transfer.DeviceToHostCopyCount);
                Assert.Equal(0, transfer.DeviceToHostBytes);
            }

            foreach (Tensor tensor in new[]
            {
                matrix,
                batched,
                linearInput,
                weight,
                bias,
                gamma,
                beta,
                branch,
            })
            {
                Assert.True(
                    tensor.TryGetCudaBFloat16GradientBuffer(0, out _),
                    $"{tensor.Name} did not retain a BF16 CUDA gradient.");
            }
        });
    }

    [Fact]
    public void EmbeddingAndForgetMemoryVariantsStayResidentInPureBFloat16()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(PrecisionPolicy.BFloat16, () =>
        {
            const int width = 16;
            Tensor tokens = BFloat16(Values(23 * width, 43), [23, width]);
            Tensor positions = BFloat16(
                Values(8 * width, 47), [8, width]);
            int[] indices = Enumerable.Range(0, 16)
                .Select(static index => index % 5)
                .ToArray();
            Tensor embedded = tokens.EmbeddingLookupWithPositions(
                positions,
                indices,
                batchSize: 2,
                sequenceLength: 8);

            const int keyWidth = 16;
            const int valueWidth = 16;
            const int projectionWidth = 2 * keyWidth + 3 * valueWidth;
            Tensor projectedV2 = BFloat16(
                Values(2 * 4 * projectionWidth, 53),
                [2, 4, projectionWidth]);
            Tensor projectedV3 = BFloat16(
                Values(2 * 4 * projectionWidth, 59),
                [2, 4, projectionWidth]);
            Tensor projectedDrn = BFloat16(
                Values(2 * 4 * projectionWidth, 61),
                [2, 4, projectionWidth]);
            Tensor loss = embedded.Sum()
                + projectedV2.ForgetMemoryV2(
                    keyWidth, valueWidth, 0.2f).Sum()
                + projectedV3.ForgetMemoryV3(
                    keyWidth, valueWidth, 0.2f).Sum()
                + projectedDrn.ForgetMemoryDRN(
                    keyWidth, valueWidth, 0.2f).Sum();

            using (DeviceTransferGuard.EnterTrainingStep(102))
            {
                loss.BackwardAndRelease();
                DeviceTransferSnapshot transfer = Assert.NotNull(
                    DeviceTransferGuard.CurrentSnapshot);
                Assert.Equal(0, transfer.HostToDeviceCopyCount);
                Assert.Equal(0, transfer.DeviceToHostCopyCount);
            }

            foreach (Tensor tensor in new[]
            {
                tokens,
                positions,
                projectedV2,
                projectedV3,
                projectedDrn,
            })
            {
                Assert.True(
                    tensor.TryGetCudaBFloat16GradientBuffer(0, out _));
            }
        });
    }

    [Fact]
    public void Mix16_32KeepsTransformerGradientsFloat32()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(PrecisionPolicy.Mix16_32, () =>
        {
            Tensor input = BFloat16(Values(4 * 16, 67), [4, 16]);
            Tensor weight = BFloat16(Values(16 * 16, 71), [16, 16]);
            Tensor bias = BFloat16(Values(16, 73), [16]);
            Tensor gamma = BFloat16(
                Enumerable.Repeat(1f, 16).ToArray(), [16]);
            Tensor beta = BFloat16(new float[16], [16]);
            Tensor loss = input.LinearLastDim(weight, bias, false)
                .LayerNormLastDim(gamma, beta)
                .Sum();
            loss.BackwardAndRelease();

            foreach (Tensor tensor in new[]
            {
                input, weight, bias, gamma, beta,
            })
            {
                Assert.False(
                    tensor.TryGetCudaBFloat16GradientBuffer(0, out _));
                Assert.True(tensor.HasGradientBuffer);
            }
        });
    }

    [Fact]
    public void ClipAndAdamWConsumeAuthoritativeBFloat16AcrossSteps()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(PrecisionPolicy.BFloat16, () =>
        {
            var parameter = new Parameter(
                [0.25f, -0.5f, 0.75f, 1f],
                [2, 2],
                "bf16.clip.adam",
                WeightDecayPolicy.Apply,
                TensorDType.BFloat16);
            parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
            var optimizer = (AdamW)optim.AdamW(
                [parameter],
                lr: 0.01f,
                weight_decay: 0.01f,
                bf16_first_moment: true,
                bf16_second_moment: true);
            try
            {
                parameter.T.BackwardAndRelease([3f, 4f, 0f, 0f]);
                Assert.True(
                    parameter.T.HasAuthoritativeCudaBFloat16Gradient);
                float norm = nn.utils.clip_grad_norm_(
                    [parameter],
                    max_norm: 1f);
                Assert.InRange(norm, 4.999f, 5.001f);
                Assert.True(
                    parameter.T.HasAuthoritativeCudaBFloat16Gradient);
                optimizer.step();
                Assert.True(
                    parameter.T.HasAuthoritativeCudaBFloat16Gradient);
                Assert.False(
                    parameter.T.HasCudaMasterFloat32Buffer(0),
                    "Pure bfloat16 AdamW must not retain an FP32 weight master.");
                (NativeCudaBuffer<short> first, NativeCudaBuffer<short> second) =
                    optimizer.GetCudaBFloat16Moments(0, 0);
                Assert.Equal(parameter.T.Numel, first.Length);
                Assert.Equal(parameter.T.Numel, second.Length);

                optimizer.zero_grad();
                parameter.T.BackwardAndRelease(
                    [0.5f, -0.25f, 0.75f, -1f]);
                Assert.True(
                    parameter.T.HasAuthoritativeCudaBFloat16Gradient);
                optimizer.step();
                Assert.True(
                    parameter.T.HasAuthoritativeCudaBFloat16Gradient);
                Assert.All(
                    parameter.T.Data.ToArray(),
                    value => Assert.True(float.IsFinite(value)));
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    [Fact]
    public void NekoMuonResidentDecodeDoesNotStealBFloat16Authority()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(PrecisionPolicy.BFloat16, () =>
        {
            var parameter = new Parameter(
                Values(4 * 16, 79),
                [4, 16],
                "bf16.neko",
                WeightDecayPolicy.Exclude,
                TensorDType.BFloat16);
            parameter.T.to(new TorchDevice(TensorDevice.Cuda, 0));
            var optimizer = (NekoMuon)optim.NekoMuon(
                [parameter],
                lr: 0.005f,
                newton_schulz_steps: 2,
                newton_schulz_interval: 1,
                weight_decay: 0f);
            try
            {
                for (int step = 0; step < 2; step++)
                {
                    parameter.T.BackwardAndRelease(
                        Values(parameter.T.Numel, 83 + step));
                    Assert.True(
                        parameter.T.HasAuthoritativeCudaBFloat16Gradient);
                    optimizer.step();
                    Assert.True(
                        parameter.T.HasAuthoritativeCudaBFloat16Gradient);
                    Assert.False(
                        parameter.T.HasCudaMasterFloat32Buffer(0),
                        "Pure bfloat16 NekoMuon must not retain an FP32 weight master.");
                    (NativeCudaBuffer<ushort> fast, NativeCudaBuffer<ushort> slow) =
                        optimizer.GetCudaBFloat16Moments(0, 0);
                    Assert.Equal(parameter.T.Numel, fast.Length);
                    Assert.Equal(parameter.T.Numel, slow.Length);
                    if (step == 0)
                        optimizer.zero_grad();
                }
            }
            finally
            {
                optimizer.DisposeCudaResources();
                parameter.T.InvalidateCudaBuffers();
            }
        });
    }

    private static void WithCuda(PrecisionPolicy precision, Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndex = 0;
            using IDisposable policy =
                TensorExecutionContext.PushPrecisionPolicy(precision);
            action();
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    private static Tensor BFloat16(float[] values, int[] shape)
    {
        var tensor = new Tensor(
            values,
            shape,
            name: string.Join('x', shape),
            dtype: TensorDType.BFloat16);
        tensor.to(new TorchDevice(TensorDevice.Cuda, 0));
        return tensor;
    }

    private static float[] Values(int length, int phase)
        => Enumerable.Range(0, length)
            .Select(index => MathF.Sin((index + phase) * 0.03125f) * 0.2f)
            .ToArray();
}
