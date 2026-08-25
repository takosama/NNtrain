using BenchmarkDotNet.Running;
using NNtrain;
using NNtrain.Benchmarks;

if (args.Length > 0
    && string.Equals(args[0], "--benchmark-generation-cache", StringComparison.Ordinal))
{
    int warmup = args.Length > 1 ? int.Parse(args[1]) : 1;
    int iterations = args.Length > 2 ? int.Parse(args[2]) : 3;
    bool productionShape = args.Length > 3
        && string.Equals(args[3], "production", StringComparison.OrdinalIgnoreCase);
    GenerationKvCacheProfiler.Run(warmup, iterations, productionShape);
}
else if (args.Length > 0
    && string.Equals(args[0], "--benchmark-gpu-primitives", StringComparison.Ordinal))
{
    int warmup = args.Length > 1 ? int.Parse(args[1]) : 2;
    int iterations = args.Length > 2 ? int.Parse(args[2]) : 5;
    GpuPrimitiveProfiler.Run(warmup, iterations);
}
else if (args.Length > 0
    && string.Equals(args[0], "--profile-transformer-detail-json", StringComparison.Ordinal))
{
    string configurationPath = args.Length > 1
        ? args[1]
        : "training.transformer.json";
    int warmup = args.Length > 2 ? int.Parse(args[2]) : 2;
    int steps = args.Length > 3 ? int.Parse(args[3]) : 5;
    string? precisionMode = args.Length > 4 ? args[4] : null;
    TransformerCudaProfiler.RunDetailedFromConfiguration(
        configurationPath, warmup, steps, precisionMode);
}
else if (args.Length > 0
    && string.Equals(args[0], "--profile-transformer-json", StringComparison.Ordinal))
{
    string configurationPath = args.Length > 1
        ? args[1]
        : "training.transformer.json";
    int warmup = args.Length > 2 ? int.Parse(args[2]) : 1;
    int steps = args.Length > 3 ? int.Parse(args[3]) : 10;
    int generationEvery = args.Length > 4 ? int.Parse(args[4]) : 0;
    int generatedTokens = args.Length > 5 ? int.Parse(args[5]) : 0;
    string? precisionMode = args.Length > 6 ? args[6] : null;
    TransformerCudaProfiler.RunFromConfiguration(
        configurationPath,
        warmup,
        steps,
        generationEvery,
        generatedTokens,
        precisionMode);
}
else if (args.Length > 0
    && string.Equals(args[0], "--profile-transformer-cuda", StringComparison.Ordinal))
{
    int warmup = args.Length > 1 ? int.Parse(args[1]) : 2;
    int steps = args.Length > 2 ? int.Parse(args[2]) : 5;
    int batch = args.Length > 3 ? int.Parse(args[3]) : 8;
    int sequence = args.Length > 4 ? int.Parse(args[4]) : 128;
    int deviceCount = args.Length > 5 ? int.Parse(args[5]) : 2;
    bool useNekoMuon = args.Length > 6
        && string.Equals(args[6], "nekomuon", StringComparison.OrdinalIgnoreCase);
    string? precisionMode = args.Length > 7 ? args[7] : null;
    TransformerCudaProfiler.Run(
        warmup,
        steps,
        batch,
        sequence,
        deviceCount,
        useNekoMuon,
        precisionMode);
}
else if (args.Length > 0
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
        : "training.forgetmemoryv2-wiki-jp.json";
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
        : "training.forgetmemoryv2-wiki-jp.json";
    TensorPrecisionMode? precisionModeOverride = args.Length > 2
        ? TensorPrecisionModeNames.Parse(args[2])
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
        precisionModeOverride,
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
