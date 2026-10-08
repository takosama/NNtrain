using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35Iq2InferenceRoutingTests
{
    [Fact]
    public void DisabledInferenceXmxNeverSelectsFactoredTrainingProjection()
    {
        ArcDeviceInfo? device = ArcDevices.Enumerate().FirstOrDefault();
        Assert.SkipWhen(device is null || !device.SupportsXmx || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_intel_subgroups"), "Intel SG16 is required.");
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(
            tiedOutput: false, contextLength: 160, iq2Qkv: true, iq2Gate: true);
        var options = new Qwen35ExecutionOptions
        {
            LoraTraining = false, InferencePrefillChunkTokens = 128,
            InferenceProjectionRows = 1, InferenceBatchTextAttention = true,
            InferenceXmxPrefill = false, InferenceXmxPackedPrefill = false,
            InferenceXmxFactoredPrefill = false, InferenceXmxGgufBslmPrefill = false,
            InferenceResidentIq2Panels = false, IQ2TiledForward = false,
            CollectKernelTimings = true
        };
        using Qwen35QuantizedModel previous = Qwen35QuantizedModel.Load(file.Path, [0], options: options);
        using Qwen35QuantizedModel candidate = Qwen35QuantizedModel.Load(file.Path, [0],
            options: options with { IQ2TiledForward = true });
        int[] prefix = Enumerable.Range(0, 128).Select(i => (i * 3 + i / 7) % 4).ToArray();
        previous.PrimePromptPrefix(prefix, TestContext.Current.CancellationToken);
        candidate.PrimePromptPrefix(prefix, TestContext.Current.CancellationToken);
        // At least one entire 128-row IQ2 projection must execute, so this
        // catches accidental eligibility for the training-only tiled branch.
        Assert.Contains("q35l_iq2_s_sg16_pair", previous.KernelMilliseconds.Keys);
        Assert.Contains("q35l_iq2_s_sg16_pair", candidate.KernelMilliseconds.Keys);
        Assert.DoesNotContain(candidate.KernelMilliseconds.Keys,
            name => name.StartsWith("q35s_", StringComparison.Ordinal));
        Assert.DoesNotContain(candidate.KernelMilliseconds.Keys,
            name => name.StartsWith("q35l_prefill_xmx_", StringComparison.Ordinal));
        foreach (int token in new[] { 3, 0, 2 })
            Assert.Equal(previous.ForwardToken(token).Select(BitConverter.SingleToInt32Bits),
                candidate.ForwardToken(token).Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(previous.ResidentWeightBytes, candidate.ResidentWeightBytes);
        Assert.Equal(previous.ResidentStateBytes, candidate.ResidentStateBytes);
    }
}
