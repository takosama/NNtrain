using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMix816AttentionDispatchCandidateTests
{
    private static readonly string[] CandidateKernels = [
        "attention_probabilities_native_exp_2048",
        "attention_probabilities_saved_native_exp_2048",
        "attention_mix8_16_bf16_pv_bpair_d32_2048",
        "attention_mix8_16_bf16_dpds_k32_d32_2048",
        "attention_mix8_16_bf16_dq_packed_bpair_2048",
        "attention_mix8_16_bf16_dkv_packed_slm_uint_2048",
    ];

    [Fact]
    public void FullSequenceCausalAttentionRoutesFiveCandidatesAndBoundsGradients()
    {
        RequireArc();
        // 96 MiB permits three T2048 score heads per tile. The first launch
        // spans batch 0 heads 0/1 and batch 1 head 0; the second starts at
        // first=3, exercising both cross-batch and nonzero head offsets.
        const int batch = 2, sequence = 2048, width = 64, heads = 2;
        var baseline = Run(TensorPrecisionMode.Mix8_16, batch, sequence, width, heads,
            candidates: false);
        var candidate = Run(TensorPrecisionMode.Mix8_16, batch, sequence, width, heads,
            candidates: true);
        AssertRouting(baseline.KernelMs, candidates: false);
        AssertRouting(candidate.KernelMs, candidates: true);

        // All non-native-exp candidates retain the same FP32 FMA sequence and
        // BF16 packing. BF16 unit roundoff is 1/256; a quarter of that is a
        // conservative relative RMS ceiling for native_exp propagation.
        const double relativeRmsBudget = 1.0 / 1024.0;
        const double maxElementBudgetFraction = 1.0 / 256.0;
        AssertNear(baseline.Outputs, candidate.Outputs, "attention output",
            relativeRmsBudget, maxElementBudgetFraction);
        AssertNear(baseline.Gradients, candidate.Gradients, "input gradient",
            relativeRmsBudget, maxElementBudgetFraction);
    }

    [Fact]
    public void Mix832KeepsOriginalAttentionRouteAndBits()
    {
        RequireArc();
        const int batch = 1, sequence = 2048, width = 64, heads = 2;
        var baseline = Run(TensorPrecisionMode.Mix8_32, batch, sequence, width, heads,
            candidates: false);
        var candidate = Run(TensorPrecisionMode.Mix8_32, batch, sequence, width, heads,
            candidates: true);
        foreach (string kernel in CandidateKernels)
        {
            Assert.False(baseline.KernelMs.ContainsKey(kernel), $"Baseline ran {kernel}.");
            Assert.False(candidate.KernelMs.ContainsKey(kernel), $"Fallback ran {kernel}.");
        }
        AssertBits(baseline.Outputs, candidate.Outputs, "mix8_32 output");
        AssertBits(baseline.Gradients, candidate.Gradients, "mix8_32 gradient");
    }

    private static void RequireArc()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        ArcDeviceInfo device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_intel_subgroup_local_block_io"),
            "The T2048 candidates require an SG16 XMX device with SLM block I/O.");
    }

    private static (float[] Outputs, float[] Gradients, Dictionary<string, double> KernelMs)
        Run(TensorPrecisionMode precision, int batch, int sequence, int width, int heads,
            bool candidates)
    {
        using var execution = Tensor.BeginArcExecution(precision: precision,
            options: new ArcExecutionOptions
            {
                Mix8_16Bf16QkvActivations = true,
                Mix8_16Bf16Activations = true,
                Mix8_16PackedAttentionBackward = true,
                Mix8_16AttentionRowDelta = true,
                Mix8_16FusedAttentionDpDs = true,
                Mix8_16FusedAttentionDpDsXmx = false,
                Mix8_16NativeExpAttention = candidates,
                Mix8_16DpDsK32 = candidates,
                Mix8_16DkvPackedSlm = candidates,
                Mix8_16PvPackedSlm = candidates,
                Mix8_16DqPackedSlm = candidates,
                Mix8_16PvM128 = false,
                Mix8_16DqM128 = false,
                AttentionWorkspaceMiB = 96,
            });
        float[] values = Enumerable.Range(0, batch * sequence * width * 3)
            .Select(i => MathF.Sin(i * .013f) * .1f).ToArray();
        Tensor input = new(values, [batch, sequence, width * 3]);
        input.ConvertStorageInPlace(TensorDType.Bfp8, Bfp8QuantizationDescriptor.Block(32));
        Tensor output = input.FusedMultiHeadAttention(heads, causal: true);
        float[] outputs = output.Data.ToArray();
        float[] seed = Enumerable.Range(0, batch * sequence * width)
            .Select(i => MathF.Cos(i * .017f) * .01f).ToArray();
        output.BackwardAndRelease(seed);
        Tensor.ArcLane.Synchronize();
        float[] gradients = input.Grad.ToArray();
        Assert.All(outputs, value => Assert.True(float.IsFinite(value)));
        Assert.All(gradients, value => Assert.True(float.IsFinite(value)));
        return (outputs, gradients,
            new Dictionary<string, double>(Tensor.ArcLane.KernelTimings));
    }

    private static void AssertRouting(Dictionary<string, double> timings, bool candidates)
    {
        foreach (string kernel in CandidateKernels)
            Assert.Equal(candidates, timings.ContainsKey(kernel));
        foreach (string production in new[] {
            "attention_mix8_16_bf16_pv_d32_2048",
            "attention_mix8_16_bf16_dpds_d32_2048",
            "attention_mix8_16_bf16_dq_packed_2048",
            "attention_mix8_16_bf16_dkv_packed_2048",
        })
            Assert.Equal(!candidates, timings.ContainsKey(production));
    }

    private static void AssertNear(float[] expected, float[] actual, string description,
        double relativeRmsBudget, double maxElementBudgetFraction)
    {
        Assert.Equal(expected.Length, actual.Length);
        double errorSquared = 0, signalSquared = 0, largestError = 0, largestSignal = 0;
        for (int i = 0; i < expected.Length; ++i)
        {
            Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]),
                $"{description} is non-finite at {i}.");
            double error = Math.Abs((double)actual[i] - expected[i]);
            errorSquared += error * error;
            signalSquared += (double)expected[i] * expected[i];
            largestError = Math.Max(largestError, error);
            largestSignal = Math.Max(largestSignal, Math.Abs(expected[i]));
        }
        Assert.True(signalSquared > 0 && largestSignal > 0,
            $"{description} has no nonzero reference signal.");
        double relativeRms = Math.Sqrt(errorSquared / signalSquared);
        double elementFraction = largestError / largestSignal;
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"{description}: relative RMS={relativeRms:G6} (budget {relativeRmsBudget:G6}), "
            + $"largest error/signal={elementFraction:G6} "
            + $"(budget {maxElementBudgetFraction:G6}); "
            + "budgets are fractions of BF16 unit roundoff 1/256, with no fixed absolute floor.");
        Assert.True(relativeRms <= relativeRmsBudget,
            $"{description} relative RMS {relativeRms:G6} exceeds {relativeRmsBudget:G6}.");
        Assert.True(elementFraction <= maxElementBudgetFraction,
            $"{description} largest error/signal {elementFraction:G6} exceeds "
            + $"{maxElementBudgetFraction:G6}.");
    }

    private static void AssertBits(float[] expected, float[] actual, string description)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i])
                == BitConverter.SingleToInt32Bits(actual[i]),
                $"{description} differs at {i}.");
    }
}
