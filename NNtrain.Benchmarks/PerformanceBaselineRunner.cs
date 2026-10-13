using System.Diagnostics;
using System.Security.Cryptography;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Execution;
using NNtrain.Training.Metrics;
using NNtrain.Training.Optimization;
using NNtrain.Training.Persistence;

namespace NNtrain.Benchmarks;

internal static partial class PerformanceBaselineRunner
{
    internal static BaselineScenarioResult Run(BaselineWorkerJob job)
    {
        ArgumentNullException.ThrowIfNull(job);
        Validate(job);
        BaselineScenario scenario = job.Scenario;
        BaselineModelConfiguration configuration = job.Model;
        TensorPrecisionMode precision = TensorPrecisionModeNames.Parse(
            configuration.Precision);
        TensorDType storageDType = precision.ToStorageDType();

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDeviceIndices = Tensor.CudaDeviceIndices.ToArray();
        bool previousSimd = Tensor.SimdEnabled;
        try
        {
            Tensor.ExecutionDevice = scenario.Device == BaselineDeviceKind.Cuda
                ? TensorDevice.Cuda
                : TensorDevice.Cpu;
            Tensor.CudaDeviceIndices = scenario.DeviceIndices;
            Tensor.SimdEnabled = true;

            using ExecutionSession execution = CreateExecutionSession(
                scenario,
                precision);
            using IDisposable executionScope = execution.Enter();

            IReadOnlyList<BaselineGpu> selectedGpus = ResolveSelectedGpus(
                scenario, job.Gpus);
            if (scenario.Soak is not null)
            {
                return RunSoak(
                    job,
                    precision,
                    storageDType,
                    selectedGpus,
                    execution);
            }
            var runs = new List<BaselineRunResult>(scenario.Repetitions);
            BaselinePhaseProbe? probe = null;
            for (int repetition = 1; repetition <= scenario.Repetitions;
                repetition++)
            {
                using BaselineFixture fixture = CreateFixture(
                    configuration,
                    precision,
                    storageDType,
                    scenario);
                Console.WriteLine(
                    $"{scenario.Name}: repetition {repetition}/" +
                    $"{scenario.Repetitions}, warming up " +
                    $"{scenario.WarmupSteps} steps");
                for (int warmup = 0; warmup < scenario.WarmupSteps; warmup++)
                {
                    _ = MeasureStep(
                        fixture,
                        configuration,
                        scenario,
                        diagnosticLabel: $"warmup-{warmup + 1}");
                }
                CudaTrainingGraphTelemetry? graphBeforeMeasurement =
                    fixture.DataParallelEngine?.TrainingGraphTelemetry;

                DateTimeOffset started = DateTimeOffset.UtcNow;
                var measurements = new List<BaselineStepMeasurement>(
                    scenario.MeasuredSteps);
                int progressInterval = Math.Max(
                    1, scenario.MeasuredSteps / 20);
                for (int step = 1; step <= scenario.MeasuredSteps; step++)
                {
                    RawStep raw = MeasureStep(
                        fixture,
                        configuration,
                        scenario,
                        diagnosticLabel: $"steady-{step}");
                    measurements.Add(raw.ToMeasurement(step));
                    if (step == 1
                        || step == scenario.MeasuredSteps
                        || step % progressInterval == 0)
                    {
                        Console.WriteLine(
                            $"{scenario.Name} run {repetition}/" +
                            $"{scenario.Repetitions} step {step,3}/" +
                            $"{scenario.MeasuredSteps}: " +
                            $"{raw.TotalMilliseconds,9:F2} ms, " +
                            $"loss={raw.Loss:F6}, " +
                            $"fwd+bwd=" +
                            $"{raw.ForwardBackwardMilliseconds:F2}, " +
                            $"clip={raw.ClipMilliseconds:F2}, " +
                            $"optimizer={raw.OptimizerMilliseconds:F2}");
                    }
                }
                CudaTrainingGraphTelemetry? graphAfterMeasurement =
                    fixture.DataParallelEngine?.TrainingGraphTelemetry;
                runs.Add(CreateRun(
                    repetition,
                    started,
                    DateTimeOffset.UtcNow,
                    fixture.DataParallelEngine?.LastShardBatchSizes ?? [],
                    CreateGraphTelemetry(
                        graphBeforeMeasurement,
                        graphAfterMeasurement,
                        scenario.MeasuredSteps),
                    measurements));
                if (scenario.CollectPhaseProbe
                    && repetition == scenario.Repetitions)
                {
                    // Reuse the already-warmed final fixture. Constructing a
                    // fourth full model/optimizer/graph after the official
                    // three runs can retain enough session-generation memory
                    // to make the diagnostic eager backward fail even though
                    // every measured run itself completed. The probe remains
                    // outside all measured intervals and runs only after the
                    // final BaselineRunResult has been committed.
                    probe = CreatePhaseProbe(
                        fixture,
                        configuration,
                        scenario,
                        runs);
                }
            }
            BaselineConditions conditions = CreateConditions(
                job, precision, storageDType, selectedGpus);
            BaselineDistribution aggregateStep = BaselineDistribution.From(
                runs.SelectMany(run => run.Measurements)
                    .Select(measurement => measurement.TotalMilliseconds));
            BaselineValidationResult? validation =
                scenario.PerformanceGate is { } performanceGate
                    ? PerformanceBaselineGatePolicy.Evaluate(
                        performanceGate,
                        scenario,
                        configuration,
                        runs)
                    : null;
            return new BaselineScenarioResult(
                conditions,
                runs,
                aggregateStep,
                probe,
                CreateNotes(scenario),
                validation);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDeviceIndices;
            Tensor.SimdEnabled = previousSimd;
        }
    }
}
