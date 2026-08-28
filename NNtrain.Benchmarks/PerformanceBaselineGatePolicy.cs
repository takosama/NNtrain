namespace NNtrain.Benchmarks;

/// <summary>
/// Defines the frozen, reproducible performance acceptance policy separately
/// from benchmark execution so a measured result cannot silently change its
/// own target.
/// </summary>
internal static class PerformanceBaselineGatePolicy
{
    internal const double OfficialTwoGpuFrozenStepP50Milliseconds = 475.48d;
    internal const double OfficialTwoGpuMaximumBaselineRatio = 0.80d;
    internal const int OfficialBatch = 72;
    internal const int OfficialSequence = 512;
    internal const int OfficialNewtonSchulzSteps = 5;
    internal const string OfficialNewtonSchulzDepthMode = "fixed";
    internal const string OfficialPrecision = TensorPrecisionModeNames.Mix16_32;

    internal static BaselinePerformanceGateConfiguration OfficialTwoGpu { get; }
        = new(
            RequiredCudaDeviceCount: 2,
            RequiredWarmupSteps: 20,
            RequiredMeasuredSteps: 210,
            RequiredRepetitions: 3,
            RequiredBatch: OfficialBatch,
            RequiredSequence: OfficialSequence,
            RequiredPrecision: OfficialPrecision,
            RequiredNewtonSchulzDepthMode:
                OfficialNewtonSchulzDepthMode,
            RequiredNewtonSchulzSteps: OfficialNewtonSchulzSteps,
            FrozenStepP50Milliseconds:
                OfficialTwoGpuFrozenStepP50Milliseconds,
            MaximumBaselineRatio:
                OfficialTwoGpuMaximumBaselineRatio);

    internal static BaselineValidationResult Evaluate(
        BaselinePerformanceGateConfiguration configuration,
        BaselineScenario scenario,
        BaselineModelConfiguration model,
        IReadOnlyList<BaselineRunResult> runs)
    {
        ArgumentNullException.ThrowIfNull(configuration);
        ArgumentNullException.ThrowIfNull(scenario);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(runs);

        bool conditionsMatch =
            scenario.Device == BaselineDeviceKind.Cuda
            && scenario.DeviceIndices.Length
                == configuration.RequiredCudaDeviceCount
            && scenario.WarmupSteps == configuration.RequiredWarmupSteps
            && scenario.MeasuredSteps == configuration.RequiredMeasuredSteps
            && scenario.Repetitions == configuration.RequiredRepetitions;
        bool runShapeMatches =
            runs.Count == configuration.RequiredRepetitions
            && runs.All(run =>
                run.Step.Count == configuration.RequiredMeasuredSteps);
        bool modelContractMatches =
            model.Batch == configuration.RequiredBatch
            && model.Sequence == configuration.RequiredSequence
            && string.Equals(
                model.Precision,
                configuration.RequiredPrecision,
                StringComparison.Ordinal)
            && string.Equals(
                model.NewtonSchulzDepthMode,
                configuration.RequiredNewtonSchulzDepthMode,
                StringComparison.OrdinalIgnoreCase)
            && model.NewtonSchulzDepth
                == configuration.RequiredNewtonSchulzSteps;
        double[] runP50s = runs.Select(run => run.Step.P50).ToArray();
        double medianRunP50 = runP50s.Length == 0
            ? double.NaN
            : BaselineDistribution.From(runP50s).P50;
        double maximum = configuration.MaximumAllowedStepP50Milliseconds;
        double ratio = medianRunP50
            / configuration.FrozenStepP50Milliseconds;
        bool finiteMeasurement = double.IsFinite(medianRunP50)
            && medianRunP50 >= 0d
            && double.IsFinite(ratio);

        BaselineGateResult[] gates =
        [
            new BaselineGateResult(
                "official-run-shape",
                conditionsMatch && runShapeMatches,
                $"device={scenario.Device}, GPUs=" +
                    $"[{string.Join(',', scenario.DeviceIndices)}], " +
                    $"warmup={scenario.WarmupSteps}, " +
                    $"measured={scenario.MeasuredSteps}, " +
                    $"repetitions={scenario.Repetitions}, " +
                    $"run-counts=[{string.Join(',', runs.Select(run => run.Step.Count))}]",
                $"CUDA/{configuration.RequiredCudaDeviceCount} GPUs, " +
                    $"{configuration.RequiredWarmupSteps} warmup + " +
                    $"{configuration.RequiredMeasuredSteps} measured steps x " +
                    $"{configuration.RequiredRepetitions}"),
            new BaselineGateResult(
                "official-effective-model-contract",
                modelContractMatches,
                $"batch={model.Batch}, sequence={model.Sequence}, " +
                    $"precision={model.Precision}, optimizer=NekoMuon, " +
                    $"NS={model.NewtonSchulzDepthMode}" +
                    $"{model.NewtonSchulzDepth}",
                $"batch={configuration.RequiredBatch}, " +
                    $"sequence={configuration.RequiredSequence}, " +
                    $"precision={configuration.RequiredPrecision}, " +
                    "optimizer=NekoMuon, NS=" +
                    $"{configuration.RequiredNewtonSchulzDepthMode}" +
                    $"{configuration.RequiredNewtonSchulzSteps}"),
            new BaselineGateResult(
                "frozen-baseline-step-p50",
                conditionsMatch
                    && runShapeMatches
                    && modelContractMatches
                    && finiteMeasurement
                    && medianRunP50 <= maximum,
                finiteMeasurement
                    ? $"median-of-run-p50={medianRunP50:F3} ms, " +
                        $"ratio={ratio:F6}, run-p50s=" +
                        $"[{string.Join(',', runP50s.Select(value => value.ToString("F3")))}]"
                    : "no finite measured run p50",
                $"median of the {configuration.RequiredRepetitions} run p50s " +
                    $"<= {maximum:F3} ms " +
                    $"({configuration.MaximumBaselineRatio:P0} of frozen " +
                    $"{configuration.FrozenStepP50Milliseconds:F3} ms)"),
        ];
        return new BaselineValidationResult(
            "official two-GPU frozen performance baseline",
            gates.All(gate => gate.Passed == true),
            gates,
            Soak: null);
    }
}
