using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class CudaBfp8LayerNormTests
{
    [Theory]
    [InlineData(false, 1, 257)]
    [InlineData(true, 3, 384)]
    [InlineData(false, 2, 515)]
    public void ResidentLayerNormMatchesBf16ReferenceForScaleAndColumnTails(
        bool blockScaled,
        int rows,
        int columns)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        Bfp8QuantizationDescriptor descriptor = Descriptor(blockScaled);
        int[] inputShape = rows == 1
            ? [columns]
            : [rows, columns];
        float[] inputValues = Values(rows * columns, 17, 0.13f);
        float[] gammaValues = Enumerable.Range(0, columns)
            .Select(index => 0.82f + (index % 29) * 0.011f)
            .ToArray();
        float[] betaValues = Values(columns, 31, 0.025f);
        float[] seed = Values(rows * columns, 47, 0.037f);
        LayerNormRun expected = Bf16LayerNormReference(
            inputValues,
            gammaValues,
            betaValues,
            descriptor,
            inputShape,
            columns,
            seed);

        WithCuda(() =>
        {
            Tensor input = Bfp8Tensor(
                inputValues, inputShape, descriptor);
            Tensor gamma = Bfp8Tensor(
                gammaValues, [columns], descriptor);
            Tensor beta = Bfp8Tensor(
                betaValues, [columns], descriptor);
            MakeResident(input, gamma, beta);
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;

            Tensor output = input.LayerNormLastDim(gamma, beta);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - before;
            Assert.Equal(0, transfers.HostToDeviceBytes);
            Assert.Equal(0, transfers.DeviceToHostBytes);
            Assert.Equal(TensorDType.Bfp8, output.DType);
            Assert.Equal(descriptor, output.Bfp8Quantization);

            float[] actualOutput = output.Data.ToArray();
            output.BackwardAndRelease(seed);
            AssertClose(expected.Output, actualOutput, 4e-2f);
            AssertClose(expected.InputGradient, input.Grad, 5e-2f);
            AssertClose(expected.GammaGradient, gamma.Grad, 7e-2f);
            AssertClose(expected.BetaGradient, beta.Grad, 7e-2f);
            Release(output, input, gamma, beta);
        });
    }

    [Theory]
    [InlineData(false, 0.0, false, 257)]
    [InlineData(true, 0.25, false, 384)]
    [InlineData(false, 0.20, true, 515)]
    [InlineData(true, 0.0, true, 128)]
    [InlineData(true, 0.25, true, 512)]
    public void ResidentResidualDropoutLayerNormPreservesDropoutAndParentSemantics(
        bool blockScaled,
        double probabilityValue,
        bool sameParent,
        int columns)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int rows = 3;
        const int randomSeed = 197;
        float probability = (float)probabilityValue;
        Bfp8QuantizationDescriptor descriptor = Descriptor(blockScaled);
        float[] residualValues = Values(rows * columns, 61, 0.12f);
        float[] branchValues = sameParent
            ? residualValues
            : Values(rows * columns, 83, 0.09f);
        float[] gammaValues = Enumerable.Range(0, columns)
            .Select(index => 0.77f + (index % 31) * 0.013f)
            .ToArray();
        float[] betaValues = Values(columns, 103, 0.021f);
        float[] seed = Values(rows * columns, 127, 0.029f);
        FusedRun expected = Bf16FusedReference(
            residualValues,
            branchValues,
            gammaValues,
            betaValues,
            descriptor,
            rows,
            columns,
            probability,
            randomSeed,
            sameParent,
            seed);

        WithCuda(() =>
        {
            Tensor residual = Bfp8Tensor(
                residualValues, [rows, columns], descriptor);
            Tensor branch = sameParent
                ? residual
                : Bfp8Tensor(
                    branchValues, [rows, columns], descriptor);
            Tensor gamma = Bfp8Tensor(
                gammaValues, [columns], descriptor);
            Tensor beta = Bfp8Tensor(
                betaValues, [columns], descriptor);
            MakeResident(residual, branch, gamma, beta);
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;

            Tensor output = residual.AddDropoutLayerNormLastDim(
                branch,
                gamma,
                beta,
                probability,
                new Random(randomSeed));
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - before;
            Assert.Equal(0, transfers.HostToDeviceBytes);
            Assert.Equal(0, transfers.DeviceToHostBytes);
            Assert.Equal(TensorDType.Bfp8, output.DType);
            Assert.Equal(descriptor, output.Bfp8Quantization);

            float[] actualOutput = output.Data.ToArray();
            output.BackwardAndRelease(seed);
            AssertClose(expected.Output, actualOutput, 5e-2f);
            AssertClose(
                expected.ResidualGradient,
                residual.Grad,
                sameParent ? 8e-2f : 6e-2f);
            if (!sameParent)
            {
                AssertClose(
                    expected.BranchGradient,
                    branch.Grad,
                    7e-2f);
            }
            AssertClose(expected.GammaGradient, gamma.Grad, 9e-2f);
            AssertClose(expected.BetaGradient, beta.Grad, 9e-2f);
            Release(output, residual, gamma, beta);
            if (!sameParent)
                branch.InvalidateCudaBuffers();
        });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RepeatedNoGradLayerNormReleasesEveryTransientAllocation(
        bool fused)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int rows = 3;
        const int columns = 515;
        Bfp8QuantizationDescriptor descriptor =
            Bfp8QuantizationDescriptor.Mix8_32;

        WithCuda(() =>
        {
            Tensor input = Bfp8Tensor(
                Values(rows * columns, 151, 0.11f),
                [rows, columns],
                descriptor);
            Tensor branch = Bfp8Tensor(
                Values(rows * columns, 173, 0.08f),
                [rows, columns],
                descriptor);
            Tensor gamma = Bfp8Tensor(
                Enumerable.Range(0, columns)
                    .Select(index => 0.8f + (index % 23) * 0.015f)
                    .ToArray(),
                [columns],
                descriptor);
            Tensor beta = Bfp8Tensor(
                Values(columns, 191, 0.02f),
                [columns],
                descriptor);
            MakeResident(input, branch, gamma, beta);

            RunInference(input, branch, gamma, beta, fused);
            NativeCudaAllocationTelemetry before =
                NativeCudaRuntime.AllocationTelemetry;

            for (int iteration = 0; iteration < 20; iteration++)
                RunInference(input, branch, gamma, beta, fused);

            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();
            NativeCudaAllocationTelemetry delta =
                NativeCudaRuntime.AllocationTelemetry - before;
            Assert.Equal(delta.AllocationCount, delta.FreeCount);
            Assert.Equal(delta.AllocationBytes, delta.FreeBytes);
            Release(input, branch, gamma, beta);
        });
    }

    private static void RunInference(
        Tensor input,
        Tensor branch,
        Tensor gamma,
        Tensor beta,
        bool fused)
    {
        using IDisposable noGrad = AutogradContext.NoGrad();
        using CudaInferenceScope scope = CudaInferenceScope.Begin(
            resetPool: true,
            clearPoolOnDispose: true);
        _ = fused
            ? input.AddDropoutLayerNormLastDim(
                branch, gamma, beta, 0.2f, new Random(211))
            : input.LayerNormLastDim(gamma, beta);
    }

    private static LayerNormRun Bf16LayerNormReference(
        float[] inputValues,
        float[] gammaValues,
        float[] betaValues,
        Bfp8QuantizationDescriptor descriptor,
        int[] inputShape,
        int columns,
        float[] seed)
    {
        return WithCpu(() =>
        {
            Tensor input = Bf16FromBfp8(
                inputValues, inputShape, descriptor);
            Tensor gamma = Bf16FromBfp8(
                gammaValues, [columns], descriptor);
            Tensor beta = Bf16FromBfp8(
                betaValues, [columns], descriptor);
            Tensor output = input.LayerNormLastDim(gamma, beta);
            output.Backward(seed);
            return new LayerNormRun(
                Quantize(output.Data, descriptor),
                input.Grad.ToArray(),
                gamma.Grad.ToArray(),
                beta.Grad.ToArray());
        });
    }

    private static FusedRun Bf16FusedReference(
        float[] residualValues,
        float[] branchValues,
        float[] gammaValues,
        float[] betaValues,
        Bfp8QuantizationDescriptor descriptor,
        int rows,
        int columns,
        float probability,
        int randomSeed,
        bool sameParent,
        float[] seed)
    {
        return WithCpu(() =>
        {
            Tensor residual = Bf16FromBfp8(
                residualValues, [rows, columns], descriptor);
            Tensor branch = sameParent
                ? residual
                : Bf16FromBfp8(
                    branchValues, [rows, columns], descriptor);
            Tensor gamma = Bf16FromBfp8(
                gammaValues, [columns], descriptor);
            Tensor beta = Bf16FromBfp8(
                betaValues, [columns], descriptor);
            Tensor output = residual.AddDropoutLayerNormLastDim(
                branch,
                gamma,
                beta,
                probability,
                new Random(randomSeed));
            output.Backward(seed);
            return new FusedRun(
                Quantize(output.Data, descriptor),
                residual.Grad.ToArray(),
                sameParent ? residual.Grad.ToArray() : branch.Grad.ToArray(),
                gamma.Grad.ToArray(),
                beta.Grad.ToArray());
        });
    }

    private static Tensor Bf16FromBfp8(
        float[] values,
        int[] shape,
        Bfp8QuantizationDescriptor descriptor)
    {
        Tensor encoded = Tensor.FromBfp8(values, shape, descriptor);
        return new Tensor(
            encoded.Data.ToArray(),
            shape,
            dtype: TensorDType.BFloat16);
    }

    private static Tensor Bfp8Tensor(
        float[] values,
        int[] shape,
        Bfp8QuantizationDescriptor descriptor)
        => Tensor.FromBfp8(values, shape, descriptor);

    private static float[] Quantize(
        IReadOnlyList<float> values,
        Bfp8QuantizationDescriptor descriptor)
    {
        Bfp8EncodedStorage encoded = Bfp8QuantizationCodec.Default.Encode(
            values.ToArray(), descriptor);
        float[] decoded = new float[values.Count];
        Bfp8QuantizationCodec.Default.Decode(
            encoded.Payload.Span,
            encoded.Scales.Span,
            descriptor,
            decoded);
        return decoded;
    }

    private static Bfp8QuantizationDescriptor Descriptor(bool blockScaled)
        => blockScaled
            ? Bfp8QuantizationDescriptor.Mix8_32
            : Bfp8QuantizationDescriptor.TensorWide;

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index =>
                MathF.Sin((index + offset) * 0.113f) * scale
                + MathF.Cos((index + offset) * 0.039f) * scale * 0.37f)
            .ToArray();

    private static void MakeResident(params Tensor[] tensors)
    {
        foreach (Tensor tensor in tensors.Distinct())
        {
            tensor.to(new TorchDevice(TensorDevice.Cuda, 0));
            _ = tensor.EnsureCudaBfp8Buffer(0);
        }
    }

    private static void Release(params Tensor[] tensors)
    {
        foreach (Tensor tensor in tensors.Distinct())
            tensor.InvalidateCudaBuffers();
    }

    private static T WithCpu<T>(Func<T> action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            return action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    private static void WithCuda(Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.CudaDeviceIndex = 0;
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private sealed record LayerNormRun(
        float[] Output,
        float[] InputGradient,
        float[] GammaGradient,
        float[] BetaGradient);

    private sealed record FusedRun(
        float[] Output,
        float[] ResidualGradient,
        float[] BranchGradient,
        float[] GammaGradient,
        float[] BetaGradient);
}
