using BenchmarkDotNet.Running;
using NNtrain;
using NNtrain.Benchmarks;

if (args.Length > 0
    && string.Equals(args[0], "--compare-ten-step", StringComparison.Ordinal))
{
    int steps = args.Length > 1 ? int.Parse(args[1]) : 10;
    bool cudaOnly = args.Length > 2
        && string.Equals(args[2], "cuda-only", StringComparison.Ordinal);
    TenStepCompareProfiler.Run(steps, cudaOnly);
}
else if (args.Length > 0
    && string.Equals(args[0], "--profile-adamw", StringComparison.Ordinal))
{
    string configurationPath = args.Length > 1
        ? args[1]
        : "training.wiki-jp.json";
    int? workerOverride = args.Length > 2
        ? int.Parse(args[2])
        : null;
    bool? simdOverride = args.Length > 3
        ? bool.Parse(args[3])
        : null;
    AdamWJsonProfiler.Run(
        configurationPath,
        workerOverride,
        simdOverride);
}
else if (args.Length > 0
    && string.Equals(args[0], "--profile-wiki", StringComparison.Ordinal))
{
    string configurationPath = args.Length > 1
        ? args[1]
        : "training.wiki-jp.json";
    TensorDType? dtypeOverride = args.Length > 2
        ? args[2].ToLowerInvariant() switch
        {
            "float16" or "half" => TensorDType.Float16,
            "float32" => TensorDType.Float32,
            _ => throw new ArgumentException(
                $"Unsupported profile dtype '{args[2]}'."),
        }
        : null;
    bool? nativeFloat16Override = args.Length > 3
        ? bool.Parse(args[3])
        : null;
    int? warmupStepsOverride = args.Length > 4
        ? int.Parse(args[4])
        : null;
    int? measuredStepsOverride = args.Length > 5
        ? int.Parse(args[5])
        : null;
    WikiTrainingPhaseProfiler.Run(
        configurationPath,
        dtypeOverride,
        nativeFloat16Override,
        warmupStepsOverride,
        measuredStepsOverride);
}
else
{
    BenchmarkSwitcher
        .FromAssembly(typeof(Program).Assembly)
        .Run(args);
}
