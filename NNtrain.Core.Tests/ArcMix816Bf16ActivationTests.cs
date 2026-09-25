using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMix816Bf16ActivationTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Mix8_16, false, TensorDType.Bfp8)]
    [InlineData(TensorPrecisionMode.Mix8_16, true, TensorDType.BFloat16)]
    [InlineData(TensorPrecisionMode.Mix8_32, true, TensorDType.Bfp8)]
    public void ImplicitOperationOutputsFollowOptInPolicyWithoutChangingInputs(
        TensorPrecisionMode precision, bool enabled, TensorDType expected)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");
        using var execution = Tensor.BeginArcExecution(precision: precision,
            options: new ArcExecutionOptions { Mix8_16Bf16Activations = enabled });
        var input = Tensor.FromBfp8(
            Enumerable.Range(0, 64).Select(i => MathF.Sin(i * .17f)).ToArray(),
            [2, 32], Bfp8QuantizationDescriptor.Block(32));

        Tensor first = input + input;
        Tensor second = first + input;

        Assert.Equal(TensorDType.Bfp8, input.DType);
        Assert.Equal(expected, first.DType);
        Assert.Equal(expected, second.DType);
        Assert.All(second.Data, value => Assert.True(float.IsFinite(value)));
        Tensor.ArcLane.CheckNumericStatus();
    }

    [Fact]
    public void FusedLinearConsumesBFloat16ActivationAndKeepsBfp8WeightsAndOutput()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx
            || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");
        const int rows = 4096, width = 64;
        using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_16,
            options: new ArcExecutionOptions
            {
                Mix8_16Bf16Activations = true,
                Mix8_16Int8Linear = false,
                FusedBfp8Linear = true,
            });
        Tensor Make(int[] shape, int seed) => Tensor.FromBfp8(
            Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b))
                .Select(i => MathF.Sin(i * .013f + seed) * .031f).ToArray(),
            shape, Bfp8QuantizationDescriptor.Block(32));
        Tensor packedInput = Make([rows, width], 1);
        Tensor weights = Make([width, width], 2);
        Tensor bias = Make([width], 3);

        Tensor bf16Input = packedInput + packedInput;
        Tensor output = bf16Input.LinearLastDim(weights, bias, applyRelu: false);

        Assert.Equal(TensorDType.BFloat16, bf16Input.DType);
        Assert.Equal(TensorDType.Bfp8, weights.DType);
        Assert.Equal(TensorDType.Bfp8, bias.DType);
        Assert.Equal(TensorDType.Bfp8, output.DType);
        Assert.All(output.Data, value => Assert.True(float.IsFinite(value)));
        Assert.Contains("gemm_xmx_bfp8_epilogue_16x32", Tensor.ArcLane.KernelTimings.Keys);
        Tensor.ArcLane.CheckNumericStatus();
    }
}
