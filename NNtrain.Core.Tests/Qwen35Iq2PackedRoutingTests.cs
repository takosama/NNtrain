using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35Iq2PackedRoutingTests
{
    [Theory]
    [InlineData("single")]
    [InlineData("pair")]
    [InlineData("rows4")]
    [InlineData("bslm")]
    public void ModelPackedRoutesRetainEveryLogitAndResidentWeightByteCount(string route)
    {
        RequireDevice(requireHalf: route == "bslm");
        using TemporaryQwenGguf fixture = Fixture();
        Qwen35ExecutionOptions options = Options() with
        {
            InferencePairedProjectionTypes = route == "single" ? 0 : 1,
            InferenceProjectionRows = route == "rows4" ? 4 : 1,
            InferenceXmxGgufBslmPrefill = route == "bslm"
        };
        using Qwen35QuantizedModel raw = Qwen35QuantizedModel.Load(fixture.Path, [0], options: options);
        using Qwen35QuantizedModel packed = Qwen35QuantizedModel.Load(fixture.Path, [0],
            options: options with { InferencePackedIq2Decode = true });
        AssertSameStorage(raw, packed);
        foreach (int token in new[] { 1, 3, 0, 2, 1 })
            AssertBits(raw.ForwardToken(token), packed.ForwardToken(token));
        string decode = "q35l_iq2_s_packed82_sg16" + (route == "single" ? "" : "_pair");
        Assert.Contains(decode, packed.KernelMilliseconds.Keys);
        AssertNoPackedRoute(raw);

        raw.Reset(); packed.Reset();
        int[] history = Enumerable.Range(0, 21).Select(i => (i * 3 + i / 7) % 4).ToArray();
        Assert.Equal(raw.PrimePromptPrefix(history, TestContext.Current.CancellationToken),
            packed.PrimePromptPrefix(history, TestContext.Current.CancellationToken));
        foreach (int token in new[] { 3, 0, 2 })
            AssertBits(raw.ForwardToken(token), packed.ForwardToken(token));
        string prefill = route switch
        {
            "rows4" => "q35l_prefill_linear_iq2_s_rows4_packed82",
            "bslm" => "q35l_prefill_xmx_iq2_s_gguf_bslm_packed82",
            _ => decode
        };
        Assert.Contains(prefill, packed.KernelMilliseconds.Keys);
        Assert.DoesNotContain("q35l_iq2_s_unpack82", packed.KernelMilliseconds.Keys);
        AssertSameStorage(raw, packed);
        Assert.Equal(raw.ResidentStateBytes, packed.ResidentStateBytes);
        Assert.Equal(raw.GenerateTokenIds([1, 3, 0, 2], 5), packed.GenerateTokenIds([1, 3, 0, 2], 5));
        AssertBits(raw.ForwardToken(1), packed.ForwardToken(1));
    }

    [Fact]
    public void TrainingIgnoresPackedInferenceOptionAndRetainsRawGradients()
    {
        RequireDevice(requireHalf: false);
        // A later recurrent layer has differentiable input from earlier LoRA
        // targets, so its IQ2 matrices must execute the raw transpose route.
        using TemporaryQwenGguf fixture = Fixture(layerCount: 4);
        Qwen35ExecutionOptions options = Options() with
        {
            LoraTraining = true, TrainingBufferPoolMiB = 0, InferencePrefillChunkTokens = 0
        };
        using Qwen35QuantizedModel raw = Qwen35QuantizedModel.Load(fixture.Path, [0], options: options);
        using Qwen35QuantizedModel requestedPacked = Qwen35QuantizedModel.Load(fixture.Path, [0],
            options: options with { InferencePackedIq2Decode = true });
        AssertSameStorage(raw, requestedPacked);
        foreach (int token in new[] { 1, 3, 0 })
            AssertBits(raw.ForwardToken(token), requestedPacked.ForwardToken(token));
        var adapters = new Qwen35LoraOptions { Rank = 2, Alpha = 4, Seed = 671 };
        raw.AttachLora(adapters); requestedPacked.AttachLora(adapters);
        int[] tokens = [1, 3, 0, 2];
        Qwen35LoraStepResult expected = raw.ComputeLoraGradients(tokens, responseStartIndex: 1);
        Qwen35LoraStepResult actual = requestedPacked.ComputeLoraGradients(tokens, responseStartIndex: 1);
        Assert.True(double.IsFinite(expected.Loss) && double.IsFinite(expected.GradientNorm) && expected.GradientNorm > 0);
        Assert.Equal(BitConverter.DoubleToInt64Bits(expected.Loss), BitConverter.DoubleToInt64Bits(actual.Loss));
        Assert.Equal(BitConverter.DoubleToInt64Bits(expected.GradientNorm), BitConverter.DoubleToInt64Bits(actual.GradientNorm));
        foreach (string name in raw.LoraMatrices.Keys)
        {
            float[][] expectedGradients = raw.LoraMatrices[name].ReadGradients();
            float[][] actualGradients = requestedPacked.LoraMatrices[name].ReadGradients();
            Assert.Equal(expectedGradients.Length, actualGradients.Length);
            for (int part = 0; part < expectedGradients.Length; part++) AssertBits(expectedGradients[part], actualGradients[part]);
        }
        Assert.Contains("q35t_linear_iq2_s_rows4", requestedPacked.KernelMilliseconds.Keys);
        Assert.Contains(requestedPacked.KernelMilliseconds.Keys, kernel => kernel.StartsWith("q35t_xpose_iq2_s", StringComparison.Ordinal));
        AssertNoPackedRoute(requestedPacked);
        Assert.Equal(raw.ResidentWeightBytes, requestedPacked.ResidentWeightBytes);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NonBslmXmxRetainsRawStorageWithoutDuplicateWeightScratch(bool factored)
    {
        RequireDevice(requireHalf: true);
        using TemporaryQwenGguf fixture = Fixture();
        Qwen35ExecutionOptions options = Options() with
        {
            InferenceXmxPrefill = true, InferenceXmxFactoredPrefill = factored,
            InferenceXmxGgufBslmPrefill = false, InferenceXmxPackedPrefill = false
        };
        using Qwen35QuantizedModel raw = Qwen35QuantizedModel.Load(fixture.Path, [0], options: options);
        using Qwen35QuantizedModel requestedPacked = Qwen35QuantizedModel.Load(fixture.Path, [0],
            options: options with { InferencePackedIq2Decode = true });
        AssertSameStorage(raw, requestedPacked);
        foreach (int token in new[] { 1, 3, 0 })
            AssertBits(raw.ForwardToken(token), requestedPacked.ForwardToken(token));
        raw.Reset(); requestedPacked.Reset();
        int[] history = Enumerable.Range(0, 19).Select(i => (i * 3 + i / 7) % 4).ToArray();
        Assert.Equal(raw.PrimePromptPrefix(history, TestContext.Current.CancellationToken),
            requestedPacked.PrimePromptPrefix(history, TestContext.Current.CancellationToken));
        foreach (int token in new[] { 2, 1, 3 })
            AssertBits(raw.ForwardToken(token), requestedPacked.ForwardToken(token));
        string xmx = "q35l_prefill_xmx_iq2_s_" + (factored ? "factored" : "f16x2");
        Assert.Contains(xmx, requestedPacked.KernelMilliseconds.Keys);
        Assert.Contains("q35l_iq2_s_sg16_pair", requestedPacked.KernelMilliseconds.Keys);
        AssertNoPackedRoute(requestedPacked);
        AssertSameStorage(raw, requestedPacked);
        // ResidentWeightBytes alone cannot expose a full temporary raw copy.
        // Identical model operations with pooling disabled must have the same
        // tracked peak allocation and leave the same live/native footprint.
        Assert.Equal(raw.PeakDeviceBytes, requestedPacked.PeakDeviceBytes);
        Assert.Equal(raw.TotalNativeDeviceBytes, requestedPacked.TotalNativeDeviceBytes);
    }

    private static TemporaryQwenGguf Fixture(uint layerCount = 2)
        => Qwen35ResidentModelTests.CreateFixture(tiedOutput: false, contextLength: 64,
            iq2Qkv: true, iq2Gate: true, layerCount: layerCount);

    private static Qwen35ExecutionOptions Options() => new()
    {
        InferencePackedIq2Decode = false, InferenceResidentIq2Panels = false,
        InferencePrefillChunkTokens = 16, InferencePairedProjectionTypes = 1,
        InferenceBufferPoolMiB = 0, InferenceDeferredReleaseMiB = 0,
        CollectKernelTimings = true
    };

    private static void RequireDevice(bool requireHalf)
    {
        ArcDeviceInfo? device = ArcDevices.Enumerate().FirstOrDefault();
        Assert.SkipWhen(device is null || !device.SupportsXmx || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_intel_subgroups")
            || requireHalf && !device.Extensions.Split(' ').Contains("cl_khr_fp16"), "Intel SG16/F16 XMX is required.");
    }

    private static void AssertSameStorage(Qwen35QuantizedModel expected, Qwen35QuantizedModel actual)
    {
        Assert.Equal(expected.ResidentWeightBytes, actual.ResidentWeightBytes);
        Assert.Equal(expected.ResidentAuxiliaryWeightBytes, actual.ResidentAuxiliaryWeightBytes);
        Assert.Equal(expected.LiveDeviceBytes, actual.LiveDeviceBytes);
    }

    private static void AssertNoPackedRoute(Qwen35QuantizedModel model)
        => Assert.DoesNotContain(model.KernelMilliseconds.Keys, kernel => kernel.Contains("packed82", StringComparison.Ordinal)
            || kernel is "q35l_iq2_s_pack82" or "q35l_iq2_s_unpack82");

    private static void AssertBits(float[] expected, float[] actual)
    {
        Assert.All(expected, value => Assert.True(float.IsFinite(value)));
        Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), actual.Select(BitConverter.SingleToInt32Bits));
    }
}
