# Performance benchmarking

Phase 8 replaces isolated elapsed-time comments with reproducible
BenchmarkDotNet results. A useful result must identify the machine, runtime,
build configuration, workload, and memory behavior together.

The first captured result is documented in
`performance-baseline-2026-08-01.md`.

## Baseline workload

`TrainingBenchmarks.TrainAndEvaluateOneEpoch` measures the complete Core
learning path: Trainer, Transformer forward, autograd backward, and AdamW. Its
dataset is deterministic and in memory so disk state does not distort the Core
baseline. It uses MNIST-compatible dimensions with these fixed settings:

| Setting | Value |
| --- | ---: |
| Input shape | 28 x 28 |
| Classes | 10 |
| Attention heads | 1 |
| Hidden size | 4 |
| Transformer layers | 1 |
| Training samples available | 32 |
| Training steps per measured epoch | 192 |
| Evaluation samples | 32 |
| Learning rate | 0.001 |
| Model seed | 3 |
| Trainer seed | 7 |
| Initialization scale | 0.01 |

Shape and model-size settings are BenchmarkDotNet parameters, so they appear
beside every result rather than living in an untraceable comment.

## Debug and Release comparison

Run both configurations in one benchmark session:

```text
dotnet run --configuration Release --project NNtrain.Benchmarks -- \
  --filter "*TrainingBenchmarks*"
```

The host is built in Release so benchmark orchestration is stable. The two
BenchmarkDotNet jobs build the measured dependency graph separately with Debug
and Release configurations. Do not run under a debugger, and close unrelated
CPU- or memory-intensive applications before collecting a baseline.

For a quick discovery/build check that does not constitute a performance
baseline:

```text
dotnet run --configuration Release --project NNtrain.Benchmarks -- --list flat
```

## Required evidence

Keep the generated Markdown and full JSON reports from
`BenchmarkDotNet.Artifacts/results`. A performance PR must report:

- repository revision or patch identity;
- BenchmarkDotNet environment summary, including OS, CPU, logical cores,
  .NET SDK/runtime, JIT, and GC mode;
- all workload parameter columns and the Debug/Release job;
- mean, error, standard deviation, and ratio;
- allocated bytes per operation and Gen0/Gen1/Gen2 collections;
- the existing unit and integration test results;
- confirmation that numerical results remain inside the established test
  tolerances.

Debug results expose the development-build cost, while Release is the primary
optimization baseline. Results from different machines must not be compared as
if they were a controlled before/after measurement.

## Change isolation

A benchmark-only PR records the current implementation and must not change a
numeric algorithm. Later allocation, buffer-reuse, loop-order, Span, SIMD, and
autograd-collection changes each receive a separate before/after benchmark and
regression-test run so they can be reviewed or reverted independently.
