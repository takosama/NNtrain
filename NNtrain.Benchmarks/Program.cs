using BenchmarkDotNet.Running;
using NNtrain;
using NNtrain.Benchmarks;

if (args.Length == 2 && args[0] == "--probe-arc-bfp8-epilogue")
{
    ArcBfp8EpilogueProbe.Run(args[1]);
}
else if (args.Length == 1 && args[0] == "--probe-arc-capabilities")
{
    Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(NNtrain.Arc.ArcDevices.Enumerate(),
        new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
}
else if (args.Length == 2 && args[0] == "--probe-arc-pow2-pack")
{
    ArcPow2PackProbe.Run(args[1]);
}
else if (args.Length is >= 2 and <= 4 && args[0] == "--probe-arc-flash")
{
    ArcFlashAttentionProbe.Run(args[1], args.Length >= 3 ? int.Parse(args[2]) : null,
        args.Length >= 4 ? int.Parse(args[3]) : null);
}
else if (args.Length == 3 && args[0] == "--probe-arc-qk-bulk")
{
    if (args[2] is not ("on" or "off")) throw new ArgumentException("Bulk Q/K mode must be on or off.");
    ArcFlashAttentionProbe.Run(args[1], 64, 0, args[2] == "on");
}
else if (args.Length > 0 && args[0] == "--probe-arc-transformer")
{
    ArcTransformerProbe.Run(args[1..]);
}
else if (args.Length == 2 && args[0] == "--probe-arc-attention-fp32-tiles")
{
    ArcAttentionFp32TileProbe.Run(args[1]);
}
else if (args.Length == 2 && args[0] == "--probe-arc-attention-fp32-special")
{
    ArcAttentionFp32TileProbe.Run(args[1], specializationOnly: true);
}
else if (args.Length == 2 && args[0] == "--probe-arc-storage-pack-linear")
{
    ArcXmxStoragePackLinearProbe.Run(args[1]);
}
else if (args.Length == 2 && args[0] == "--probe-arc-serial-gemm")
{
    ArcSerialGemmProbe.Run(args[1]);
}
else if (args.Length == 2 && args[0] == "--probe-arc-bfp8-coalesced")
{
    ArcBfp8CoalescedProbe.Run(args[1]);
}
else if (args.Length == 2 && args[0] == "--probe-arc-dkv-special")
{
    ArcDkvSpecialProbe.Run(args[1]);
}
else if (args.Length == 2 && args[0] == "--probe-arc-dkv-split")
{
    ArcDkvSpecialProbe.Run(args[1], splitOnly: true);
}
else if (args.Length == 2 && args[0] == "--probe-arc-dkv-register-query")
{
    ArcDkvSpecialProbe.Run(args[1], registerQueryOnly: true);
}
else if (args.Length == 2 && args[0] == "--probe-arc-dkv-narrow")
{
    ArcDkvSpecialProbe.Run(args[1], narrowOnly: true);
}
else if (args.Length == 2 && args[0] == "--probe-arc-attention-row-subgroup")
{
    ArcAttentionRowSubgroupProbe.Run(args[1]);
}
else if (args.Length == 2 && args[0] == "--probe-arc-attention-row-register")
{
    ArcAttentionRowRegisterProbe.Run(args[1]);
}
else if (args.Length == 2 && args[0] == "--probe-arc-norm-coalesced")
{
    ArcNormCoalescedProbe.Run(args[1]);
}
else if (args.Length == 2 && args[0] == "--probe-arc-qk-direct")
{
    ArcAttentionQkDirectProbe.Run(args[1]);
}
else if (args.Length == 2 && args[0] == "--probe-arc-dp-ds")
{
    ArcAttentionDpDsProbe.Run(args[1]);
}
else if (args.Length == 2 && args[0] == "--probe-arc-gemm")
{
    ArcGemmProbe.Run(args[1]);
}
else if (args.Length == 2 && args[0] == "--probe-arc-xmx-tune")
{
    ArcXmxTuningProbe.Run(args[1]);
}
else if (args.Length == 2 && args[0] == "--probe-arc-xmx-next")
{
    ArcXmxNextProbe.Run(args[1]);
}
else if (args.Length == 2 && args[0] == "--probe-arc-xmx-pack-tune")
{
    ArcXmxPackTuneProbe.Run(args[1]);
}
else if (args.Length == 2 && args[0] == "--probe-arc-xmx-direct-tune")
{
    ArcXmxDirectTuneProbe.Run(args[1]);
}
else if (args.Length == 2 && args[0] == "--probe-arc-xmx-expanded")
{
    ArcXmxDirectTuneProbe.Run(args[1], expanded: true);
}
else if (args.Length == 2 && args[0] == "--probe-arc-backward")
{
    ArcBackwardGemmProbe.Run(args[1]);
}
else if (args.Length > 0 && args[0] == "--probe-dpo-resume")
{
    DpoProbe.Resume(args[1], args[2]);
}
else if (args.Length > 0 && args[0] == "--probe-dpo-pipeline")
{
    DpoPipelineProbe.Run(args[1], args[2], int.Parse(args[3]), args.Length > 4 ? int.Parse(args[4]) : 2,
        args.Length > 5 ? int.Parse(args[5]) : 128, args.Length > 6 ? int.Parse(args[6]) : 5);
}
else if (args.Length > 0 && args[0] == "--probe-dpo")
{
    DpoProbe.Run(args[1], args[2], int.Parse(args[3]), int.Parse(args[4]), args.Length > 5 && args[5] == "legacy",
        args.Length > 6 ? args[6] : "mix16_32", args.Length > 7 ? int.Parse(args[7]) : 32);
}
else if (args.Length > 0 && args[0] == "--probe-drn-chunk")
{
    if (args.Length is < 2 or > 4 || args.Length >= 3 && args[2] is not ("parallel" or "slow"))
        throw new ArgumentException("Expected new output path and optional parallel/slow.");
    DrnChunkProbe.Run(args[1], args.Length >= 3, args.Length >= 3 && args[2] == "slow", args.Length == 4 ? int.Parse(args[3]) : 8);
}
else if (args.Length > 0 && args[0] == "--probe-linear-bias")
{
    if (args.Length != 2) throw new ArgumentException("Expected new output path.");
    DrnBackwardKernelProbe.RunBias(args[1]);
}
else if (args.Length > 0 && args[0] == "--probe-drn-backward-kernel")
{
    if (args.Length is < 2 or > 3) throw new ArgumentException("Expected new output path and optional reference path.");
    DrnBackwardKernelProbe.Run(args[1], args.Length == 3 ? args[2] : null);
}
else if (args.Length > 0 && args[0] == "--probe-drn-forward-kernel")
{
    if (args.Length != 2) throw new ArgumentException("Expected new output path.");
    DrnForwardKernelProbe.Run(args[1]);
}
else if (args.Length > 0 && args[0] == "--probe-drn-precision")
{
    if (args.Length is < 6 or > 7) throw new ArgumentException("Expected config, new output, precision, batch (5/12), examples, optional sequence (32/512).");
    DrnPrecisionProbe.Run(args[1], args[2], args[3], int.Parse(args[4]), int.Parse(args[5]),
        args.Length == 7 ? int.Parse(args[6]) : 32);
}
else if (args.Length > 0
    && string.Equals(args[0], "--diagnose-drn-depth", StringComparison.Ordinal))
{
    string output = args.Length > 1
        ? args[1]
        : "benchmark-results/drn-depth-diagnostic.json";
    int steps = args.Length > 2 ? int.Parse(args[2]) : 200;
    DrnDepthConvergenceProfiler.Run(output, steps);
}
else if (args.Length > 0
    && string.Equals(args[0], "--performance-baseline", StringComparison.Ordinal))
{
    Environment.ExitCode = PerformanceBaselineCommand.Run(args[1..]);
}
else if (args.Length > 0
    && string.Equals(
        args[0], "--performance-baseline-worker", StringComparison.Ordinal))
{
    if (args.Length != 3)
    {
        throw new ArgumentException(
            "Baseline worker requires job and result JSON paths.");
    }
    Environment.ExitCode = PerformanceBaselineCommand.RunWorker(args[1], args[2]);
}
else if (args.Length > 0
    && string.Equals(args[0], "--profile-drn-real-json", StringComparison.Ordinal))
{
    if (args.Length is < 4 or > 5)
        throw new ArgumentException("Expected config, updates, new result JSON, and optional LR multiplier.");
    ForgetMemoryDrnCudaProfiler.Run(args[1], 0, int.Parse(args[2]), resultPath: args[3],
        realData: true, learningRateMultiplier: args.Length == 5
            ? float.Parse(args[4], System.Globalization.CultureInfo.InvariantCulture) : 1f);
}
else if (args.Length > 0
    && string.Equals(args[0], "--profile-transformer-convergence-json", StringComparison.Ordinal))
{
    string configurationPath = args.Length > 1
        ? args[1]
        : "training.transformer.json";
    int steps = args.Length > 2 ? int.Parse(args[2]) : 3;
    float matrixLearningRate = args.Length > 3
        ? float.Parse(args[3], System.Globalization.CultureInfo.InvariantCulture)
        : 0.001f;
    float auxiliaryLearningRate = args.Length > 4
        ? float.Parse(args[4], System.Globalization.CultureInfo.InvariantCulture)
        : 0.001f;
    string schedule = args.Length > 5 ? args[5] : "pure-cosine";
    bool forceFullNewtonSchulz = args.Length > 6
        && string.Equals(args[6], "full-ns", StringComparison.Ordinal);
    TransformerConvergenceProfiler.Run(
        configurationPath,
        steps,
        matrixLearningRate,
        auxiliaryLearningRate,
        schedule,
        forceFullNewtonSchulz);
}
else if (args.Length > 0
    && string.Equals(args[0], "--benchmark-generation-cache", StringComparison.Ordinal))
{
    int warmup = args.Length > 1 ? int.Parse(args[1]) : 1;
    int iterations = args.Length > 2 ? int.Parse(args[2]) : 3;
    bool productionShape = args.Length > 3
        && string.Equals(args[3], "production", StringComparison.OrdinalIgnoreCase);
    GenerationKvCacheProfiler.Run(warmup, iterations, productionShape);
}
else if (args.Length > 0
    && string.Equals(args[0], "--benchmark-cuda-topk", StringComparison.Ordinal))
{
    int warmup = args.Length > 1 ? int.Parse(args[1]) : 5;
    int iterations = args.Length > 2 ? int.Parse(args[2]) : 20;
    int vocabulary = args.Length > 3 ? int.Parse(args[3]) : 11500;
    CudaTopKProfiler.Run(warmup, iterations, vocabulary);
}
else if (args.Length > 0
    && string.Equals(args[0], "--benchmark-cuda-graph", StringComparison.Ordinal))
{
    int warmup = args.Length > 1 ? int.Parse(args[1]) : 10;
    int iterations = args.Length > 2 ? int.Parse(args[2]) : 100;
    int operations = args.Length > 3 ? int.Parse(args[3]) : 16;
    int length = args.Length > 4 ? int.Parse(args[4]) : 4096;
    CudaGraphProfiler.Run(warmup, iterations, operations, length);
}
else if (args.Length > 0
    && string.Equals(args[0], "--benchmark-gpu-primitives", StringComparison.Ordinal))
{
    int warmup = args.Length > 1 ? int.Parse(args[1]) : 2;
    int iterations = args.Length > 2 ? int.Parse(args[2]) : 5;
    GpuPrimitiveProfiler.Run(warmup, iterations);
}
else if (args.Length > 0
    && string.Equals(args[0], "--benchmark-cuda-public-ops", StringComparison.Ordinal))
{
    int warmup = args.Length > 1 ? int.Parse(args[1]) : 3;
    int iterations = args.Length > 2 ? int.Parse(args[2]) : 20;
    int length = args.Length > 3 ? int.Parse(args[3]) : 262144;
    CudaPublicOpsProfiler.Run(warmup, iterations, length);
}
else if (args.Length > 0
    && string.Equals(args[0], "--benchmark-bfp8-gemm", StringComparison.Ordinal))
{
    int warmup = args.Length > 1 ? int.Parse(args[1]) : 3;
    int iterations = args.Length > 2 ? int.Parse(args[2]) : 10;
    int m = args.Length > 3 ? int.Parse(args[3]) : 256;
    int k = args.Length > 4 ? int.Parse(args[4]) : 512;
    int n = args.Length > 5 ? int.Parse(args[5]) : 256;
    Bfp8CudaGemmProfiler.Run(warmup, iterations, m, k, n);
}
else if (args.Length > 0
    && string.Equals(args[0], "--benchmark-bfp8-codec", StringComparison.Ordinal))
{
    int warmup = args.Length > 1 ? int.Parse(args[1]) : 2;
    int iterations = args.Length > 2 ? int.Parse(args[2]) : 10;
    int length = args.Length > 3 ? int.Parse(args[3]) : 41_287_680;
    int blockSize = args.Length > 4 ? int.Parse(args[4]) : 32;
    string? outputPath = args.Length > 5 ? args[5] : null;
    Bfp8CudaCodecProfiler.Run(
        warmup, iterations, length, blockSize, outputPath);
}
else if (args.Length > 0
    && string.Equals(
        args[0], "--profile-pure-bfp8-stability", StringComparison.Ordinal))
{
    int steps = args.Length > 1 ? int.Parse(args[1]) : 12;
    PureBfp8StabilityProfiler.Run(steps);
}
else if (args.Length > 0
    && string.Equals(
        args[0], "--benchmark-nekomuon-fixed-ns5", StringComparison.Ordinal))
{
    int parameterCount = args.Length > 1 ? int.Parse(args[1]) : 8;
    int rows = args.Length > 2 ? int.Parse(args[2]) : 48;
    int columns = args.Length > 3 ? int.Parse(args[3]) : 64;
    int warmup = args.Length > 4 ? int.Parse(args[4]) : 2;
    int iterations = args.Length > 5 ? int.Parse(args[5]) : 10;
    NekoMuonFixedNs5Profiler.Run(
        parameterCount, rows, columns, warmup, iterations);
}
else if (args.Length > 0
    && string.Equals(
        args[0], "--benchmark-optimizer-precision", StringComparison.Ordinal))
{
    string optimizer = args.Length > 1 ? args[1] : "all";
    string precision = args.Length > 2 ? args[2] : "all";
    int deviceCount = args.Length > 3 ? int.Parse(args[3]) : 1;
    int warmup = args.Length > 4 ? int.Parse(args[4]) : 3;
    int iterations = args.Length > 5 ? int.Parse(args[5]) : 10;
    int parameterCount = args.Length > 6 ? int.Parse(args[6]) : 16;
    int rows = args.Length > 7 ? int.Parse(args[7]) : 512;
    int columns = args.Length > 8 ? int.Parse(args[8]) : 512;
    OptimizerPrecisionProfiler.Run(
        optimizer,
        precision,
        deviceCount,
        warmup,
        iterations,
        parameterCount,
        rows,
        columns);
}
else if (args.Length > 0
    && string.Equals(
        args[0], "--benchmark-embedding-backward", StringComparison.Ordinal))
{
    int warmup = args.Length > 1 ? int.Parse(args[1]) : 3;
    int iterations = args.Length > 2 ? int.Parse(args[2]) : 10;
    EmbeddingBackwardProfiler.Run(warmup, iterations);
}
else if (args.Length > 0
    && string.Equals(
        args[0], "--benchmark-cross-entropy", StringComparison.Ordinal))
{
    int warmup = args.Length > 1 ? int.Parse(args[1]) : 3;
    int iterations = args.Length > 2 ? int.Parse(args[2]) : 10;
    CrossEntropyProfiler.Run(warmup, iterations);
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
    && string.Equals(args[0], "--profile-drn-json", StringComparison.Ordinal))
{
    string configurationPath = args.Length > 1 ? args[1] : "training.forgetmemorydrn-wiki-jp.json";
    int warmup = args.Length > 2 ? int.Parse(args[2]) : 10;
    int steps = args.Length > 3 ? int.Parse(args[3]) : 20;
    bool detail = args.Length > 4 && string.Equals(args[4], "detail", StringComparison.OrdinalIgnoreCase);
    string? resultPath = args.Length > 5 ? args[5] : null;
    ForgetMemoryDrnCudaProfiler.Run(configurationPath, warmup, steps, detail, resultPath);
}
else if (args.Length > 0 && args[0] == "--verify-resume-cursor")
{
    WikiLanguageModelCommand.VerifyResumeDocumentCursor(
        args.Length > 1 ? args[1] : "training.forgetmemorydrn-wiki-jp.json", Console.Out);
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
    float? learningRate = args.Length > 7
        ? float.Parse(args[7], System.Globalization.CultureInfo.InvariantCulture)
        : null;
    float? auxiliaryLearningRate = args.Length > 8
        ? float.Parse(args[8], System.Globalization.CultureInfo.InvariantCulture)
        : null;
    TransformerCudaProfiler.RunFromConfiguration(
        configurationPath,
        warmup,
        steps,
        generationEvery,
        generatedTokens,
        precisionMode,
        learningRate,
        auxiliaryLearningRate);
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
