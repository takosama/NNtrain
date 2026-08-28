using Xunit;

namespace NNtrain.Benchmarks;

public sealed class PerformanceBaselineGateTests
{
    [Fact]
    public void Compare10UsesCpuOneGpuAndTwoGpuForTenStepsEach()
    {
        BaselineScenario[] scenarios =
            PerformanceBaselineCommand.CreateScenarios("compare10");

        Assert.Collection(
            scenarios,
            scenario => AssertScenario(
                scenario, BaselineDeviceKind.Cpu, [0], 1, 10, 1),
            scenario => AssertScenario(
                scenario, BaselineDeviceKind.Cuda, [0], 1, 10, 1),
            scenario => AssertScenario(
                scenario, BaselineDeviceKind.Cuda, [0, 1], 1, 10, 1));
    }

    [Fact]
    public void OfficialTwoGpuUsesFrozenTwentyByTwoHundredTenByThreeGate()
    {
        BaselineScenario scenario = Assert.Single(
            PerformanceBaselineCommand.CreateScenarios("official2gpu"));

        AssertScenario(
            scenario,
            BaselineDeviceKind.Cuda,
            [0, 1],
            warmup: 20,
            measured: 210,
            repetitions: 3);
        BaselinePerformanceGateConfiguration gate =
            Assert.IsType<BaselinePerformanceGateConfiguration>(
                scenario.PerformanceGate);
        Assert.Equal(475.48d, gate.FrozenStepP50Milliseconds, 10);
        Assert.Equal(0.80d, gate.MaximumBaselineRatio, 10);
        Assert.Equal(380.384d, gate.MaximumAllowedStepP50Milliseconds, 10);
        Assert.Equal(72, gate.RequiredBatch);
        Assert.Equal(512, gate.RequiredSequence);
        Assert.Equal(TensorPrecisionModeNames.Mix16_32, gate.RequiredPrecision);
        Assert.Equal("fixed", gate.RequiredNewtonSchulzDepthMode);
        Assert.Equal(5, gate.RequiredNewtonSchulzSteps);
    }

    [Theory]
    [InlineData(370d, 390d, 375d, true)]
    [InlineData(370d, 390d, 385d, false)]
    public void OfficialGateUsesMedianOfThreeRunP50s(
        double first,
        double second,
        double third,
        bool expectedPassed)
    {
        BaselineScenario scenario = Assert.Single(
            PerformanceBaselineCommand.CreateScenarios("official2gpu"));
        BaselineValidationResult validation =
            PerformanceBaselineGatePolicy.Evaluate(
                scenario.PerformanceGate!,
                scenario,
                CreateOfficialModel(),
                [
                    CreateRun(1, first, 210),
                    CreateRun(2, second, 210),
                    CreateRun(3, third, 210),
                ]);

        Assert.Equal(expectedPassed, validation.Passed);
        BaselineGateResult gate = Assert.Single(
            validation.Gates,
            value => value.Name == "frozen-baseline-step-p50");
        Assert.Equal(expectedPassed, gate.Passed);
        Assert.Contains("380.384 ms", gate.Required, StringComparison.Ordinal);
    }

    [Fact]
    public void GateRejectsMissingRepetitionEvenWhenObservedP50Passes()
    {
        BaselineScenario scenario = Assert.Single(
            PerformanceBaselineCommand.CreateScenarios("official2gpu"));
        BaselineValidationResult validation =
            PerformanceBaselineGatePolicy.Evaluate(
                scenario.PerformanceGate!,
                scenario,
                CreateOfficialModel(),
                [CreateRun(1, 1d, 210), CreateRun(2, 1d, 210)]);

        Assert.False(validation.Passed);
        Assert.False(validation.Gates.Single(value =>
            value.Name == "official-run-shape").Passed);
        Assert.False(validation.Gates.Single(value =>
            value.Name == "frozen-baseline-step-p50").Passed);
    }

    [Fact]
    public void OfficialContractPinsEffectiveModelAndRecordsEveryOverride()
    {
        BaselineModelConfiguration configured =
            PerformanceBaselineCommand.CreateSmokeConfiguration() with
            {
                Batch = 3,
                Sequence = 7,
                Precision = TensorPrecisionModeNames.Mix8_32,
                NewtonSchulzDepthMode = "adaptive",
                NewtonSchulzDepth = 2,
            };
        BaselineConfigurationResolution resolution =
            PerformanceBaselineCommand.ResolveEffectiveConfiguration(
                new BaselineCommandOptions(
                    "official2gpu", null, null, null, null),
                configured);

        Assert.Equal(72, resolution.Model.Batch);
        Assert.Equal(512, resolution.Model.Sequence);
        Assert.Equal(
            TensorPrecisionModeNames.Mix16_32,
            resolution.Model.Precision);
        Assert.Equal("fixed", resolution.Model.NewtonSchulzDepthMode);
        Assert.Equal(5, resolution.Model.NewtonSchulzDepth);
        Assert.Equal(6, resolution.EffectiveOverrides.Count);
        Assert.Contains(resolution.EffectiveOverrides, value =>
            value.Setting == "precisionMode"
            && value.ConfiguredValue == TensorPrecisionModeNames.Mix8_32
            && value.EffectiveValue == TensorPrecisionModeNames.Mix16_32
            && value.Changed);
        Assert.Equal(3, configured.Batch);
        Assert.Equal(TensorPrecisionModeNames.Mix8_32, configured.Precision);
    }

    [Fact]
    public void OfficialGateRejectsWrongEffectiveModelEvenWhenTimingPasses()
    {
        BaselineScenario scenario = Assert.Single(
            PerformanceBaselineCommand.CreateScenarios("official2gpu"));
        BaselineModelConfiguration wrong = CreateOfficialModel() with
        {
            Precision = TensorPrecisionModeNames.Mix8_32,
        };

        BaselineValidationResult validation =
            PerformanceBaselineGatePolicy.Evaluate(
                scenario.PerformanceGate!,
                scenario,
                wrong,
                [
                    CreateRun(1, 1d, 210),
                    CreateRun(2, 1d, 210),
                    CreateRun(3, 1d, 210),
                ]);

        Assert.False(validation.Passed);
        Assert.False(validation.Gates.Single(value =>
            value.Name == "official-effective-model-contract").Passed);
        Assert.False(validation.Gates.Single(value =>
            value.Name == "frozen-baseline-step-p50").Passed);
    }

    [Theory]
    [InlineData("float32", TensorPrecisionMode.Float32)]
    [InlineData("bfloat16", TensorPrecisionMode.BFloat16)]
    [InlineData("mix16_32", TensorPrecisionMode.Mix16_32)]
    [InlineData("fp16_32", TensorPrecisionMode.Mix16_32)]
    [InlineData("bfp8", TensorPrecisionMode.Bfp8)]
    [InlineData("mix8_32", TensorPrecisionMode.Mix8_32)]
    public void ComparisonPresetsAcceptEverySupportedPrecisionName(
        string value,
        TensorPrecisionMode expected)
    {
        BaselineCommandOptions options =
            PerformanceBaselineCommand.ParseOptions(
                ["compare10", "--precision", value]);

        Assert.Equal(expected, options.Precision);
    }

    [Fact]
    public void Mix8ComparisonAcceptsExplicitPositiveBlockSize()
    {
        BaselineCommandOptions options =
            PerformanceBaselineCommand.ParseOptions(
                [
                    "gpu2-10",
                    "--precision", "mix8_32",
                    "--bfp8-block-size", "256",
                ]);
        BaselineConfigurationResolution resolution =
            PerformanceBaselineCommand.ResolveEffectiveConfiguration(
                options,
                PerformanceBaselineCommand.CreateSmokeConfiguration());

        Assert.Equal(TensorPrecisionModeNames.Mix8_32, resolution.Model.Precision);
        Assert.Equal(256, resolution.Model.Bfp8BlockSize);
        Assert.Equal(2, resolution.EffectiveOverrides.Count);
    }

    [Theory]
    [InlineData("official2gpu", "--precision", "mix8_32")]
    [InlineData("soak2100", "--precision", "mix8_32")]
    [InlineData("compare10", "--precision", "bfp8", "--bfp8-block-size", "128")]
    [InlineData("compare10", "--precision", "float32", "--bfp8-block-size", "128")]
    [InlineData("compare10", "--bfp8-block-size", "0")]
    [InlineData("compare10", "--precision", "float16")]
    public void UnsupportedPrecisionOptionCombinationsFailExplicitly(
        params string[] args)
    {
        Assert.Throws<ArgumentException>(() =>
            PerformanceBaselineCommand.ParseOptions(args));
    }

    [Fact]
    public void BlockSizeWithoutPrecisionRequiresMix8Configuration()
    {
        BaselineCommandOptions options =
            PerformanceBaselineCommand.ParseOptions(
                ["compare10", "--bfp8-block-size", "128"]);

        Assert.Throws<ArgumentException>(() =>
            PerformanceBaselineCommand.ResolveEffectiveConfiguration(
                options,
                PerformanceBaselineCommand.CreateSmokeConfiguration()));
    }

    [Fact]
    public void SoakUsesProductionOneBasedCommitEventsAndRequiredLimits()
    {
        BaselineScenario scenario = Assert.Single(
            PerformanceBaselineCommand.CreateScenarios("soak2100"));
        BaselineSoakConfiguration soak =
            Assert.IsType<BaselineSoakConfiguration>(scenario.Soak);

        AssertScenario(
            scenario,
            BaselineDeviceKind.Cuda,
            [0, 1],
            warmup: 20,
            measured: 2100,
            repetitions: 1);
        Assert.Equal(2100, soak.TotalCommittedSteps);
        Assert.Equal(20, soak.PerformanceWarmupSteps);
        Assert.Equal(100, soak.TrendWindowSteps);
        Assert.Equal(2000, soak.GenerationStep);
        Assert.Equal(1050, soak.RestartStep);
        Assert.Equal(256L * 1024L * 1024L,
            soak.MaximumPostWarmupVramGrowthBytes);
        Assert.Equal(1.05d, soak.MaximumLastToFirstP50Ratio, 10);
    }

    private static BaselineRunResult CreateRun(
        int repetition,
        double p50,
        int count)
    {
        BaselineDistribution distribution = new(
            count,
            p50,
            p50,
            p50,
            p50,
            p50);
        return new BaselineRunResult(
            Repetition: repetition,
            StartedUtc: DateTimeOffset.UnixEpoch,
            FinishedUtc: DateTimeOffset.UnixEpoch,
            Step: distribution,
            ZeroGrad: distribution,
            ForwardBackward: distribution,
            Forward: null,
            LossPhase: null,
            Backward: null,
            ReduceWait: null,
            Transfer: null,
            Clip: distribution,
            Optimizer: distribution,
            NekoMuon: distribution,
            AdamW: distribution,
            ManagedAllocationBytes: distribution,
            NativeAllocationCount: distribution,
            NativeAllocationBytes: distribution,
            NativeFreeCount: distribution,
            NativeFreeBytes: distribution,
            HostToDeviceCopyCount: distribution,
            HostToDeviceBytes: distribution,
            DeviceToHostCopyCount: distribution,
            DeviceToHostBytes: distribution,
            DeviceMemory: [],
            FinalShardBatchSizes: [],
            TrainingGraph: null,
            Measurements: []);
    }

    private static BaselineModelConfiguration CreateOfficialModel()
        => PerformanceBaselineCommand.ResolveEffectiveConfiguration(
            new BaselineCommandOptions(
                "official2gpu", null, null, null, null),
            PerformanceBaselineCommand.CreateSmokeConfiguration()).Model;

    private static void AssertScenario(
        BaselineScenario scenario,
        BaselineDeviceKind device,
        int[] devices,
        int warmup,
        int measured,
        int repetitions)
    {
        Assert.Equal(device, scenario.Device);
        Assert.Equal(devices, scenario.DeviceIndices);
        Assert.Equal(warmup, scenario.WarmupSteps);
        Assert.Equal(measured, scenario.MeasuredSteps);
        Assert.Equal(repetitions, scenario.Repetitions);
    }
}
