using NNtrain;
using Xunit;

public sealed class ArcGenerationCalibrationTests
{
    [Fact]
    public void ShortGenerationPaysForOnePrefillAndUsesMiddleDecodeContext()
    {
        var plan = ArcInferenceRouting.CreateCalibrationPlan(8, 200, 2048, true);
        Assert.True(plan.UseKvCache);
        Assert.Equal(199, plan.CachedDecodeTokens);
        Assert.Equal(0, plan.FullWindowTokens);
        Assert.Equal(9, plan.DecodeSampleNewTokens);
        Assert.Equal(103, plan.RepresentativePromptLength);
        Assert.Equal(100 + 199 * 2, ArcInferenceRouting.EstimateGenerationMilliseconds(plan, 100, 2, 50));
    }

    [Theory]
    [InlineData(2040, 200, 8, 191, 9)]
    [InlineData(2048, 200, 0, 199, 0)]
    [InlineData(2047, 2, 1, 0, 2)]
    [InlineData(2047, 3, 1, 1, 2)]
    [InlineData(8, 1, 0, 0, 0)]
    public void ContextBoundarySeparatesCachedAndSlidingWindowCosts(
        int prompt, int tokens, int cachedDecode, int fullWindow, int samples)
    {
        var plan = ArcInferenceRouting.CreateCalibrationPlan(prompt, tokens, 2048, true);
        Assert.Equal(cachedDecode, plan.CachedDecodeTokens);
        Assert.Equal(fullWindow, plan.FullWindowTokens);
        Assert.Equal(samples, plan.DecodeSampleNewTokens);
        Assert.Equal(tokens, 1 + plan.CachedDecodeTokens + plan.FullWindowTokens);
        Assert.True(plan.RepresentativePromptLength + Math.Max(0, samples - 1) <= 2048);
        Assert.Equal(100d + cachedDecode * 2 + fullWindow * 50,
            ArcInferenceRouting.EstimateGenerationMilliseconds(plan, 100, 2, 50));
    }

    [Theory]
    [InlineData(8, 2048, false, 108)]
    [InlineData(2049, 2048, true, 2049)]
    [InlineData(8, 8193, true, 108)]
    public void IneligibleCacheUsesFullPrefixTimeForEveryToken(int prompt, int context, bool enabled, int representative)
    {
        var plan = ArcInferenceRouting.CreateCalibrationPlan(prompt, 200, context, enabled);
        Assert.False(plan.UseKvCache);
        Assert.Equal(representative, plan.RepresentativePromptLength);
        Assert.Equal(0, plan.CachedDecodeTokens);
        Assert.Equal(0, plan.DecodeSampleNewTokens);
        Assert.Equal(400d, ArcInferenceRouting.EstimateGenerationMilliseconds(plan, 100, 2, 50));
    }

    [Fact]
    public void ZeroTokensNeedsNoEstimatedGenerationWork()
    {
        var plan = ArcInferenceRouting.CreateCalibrationPlan(8, 0, 2048, true);
        Assert.Equal(0, plan.CachedDecodeTokens);
        Assert.Equal(0, plan.FullWindowTokens);
        Assert.Equal(0, plan.DecodeSampleNewTokens);
        Assert.Equal(0d, ArcInferenceRouting.EstimateGenerationMilliseconds(plan, 100, 2, 50));
    }

    [Fact]
    public void AutoWithZeroTokensDoesNotRequireAnArcSession()
    {
        var model = new GptRinWikiJp(16, 32, 8, 2, 16, 1, new Random(1));
        using var output = new StringWriter();
        ArcInferenceRouting.ConfigureModelTokens(model, [1, 2], 0, new([0, 1], "auto"), output);
        Assert.Contains("without calibration", output.ToString());
        Assert.Contains("single GPU [0]", output.ToString());
    }

    [Fact]
    public void ForcedTensorParallelStillValidatesAtZeroTokens()
    {
        var model = new GptRinWikiJp(16, 32, 8, 2, 16, 1, new Random(1));
        using var output = new StringWriter();
        Assert.Throws<NotSupportedException>(() => ArcInferenceRouting.ConfigureModelTokens(
            model, [1, 2], 0, new([0, 1], "tensorParallel"), output));
        Assert.Throws<ArgumentException>(() => ArcInferenceRouting.ConfigureModelTokens(
            model, [1, 2], 0, new([0], "tensorParallel"), output));
    }

    [Fact]
    public void CacheBenefitCanChangeAutoDecision()
    {
        var plan = ArcInferenceRouting.CreateCalibrationPlan(8, 200, 2048, true);
        double single = ArcInferenceRouting.EstimateGenerationMilliseconds(plan, 20, 1, 50);
        double parallel = ArcInferenceRouting.EstimateGenerationMilliseconds(plan, 10, 2, 25);
        Assert.True(ArcInferenceRouting.ShouldUseTensorParallel(20, 10));
        Assert.False(ArcInferenceRouting.ShouldUseTensorParallel(single, parallel));
    }
}
