# Training performance baseline — 2026-08-01

This is the first measurement-backed baseline for phase 8. It replaces
standalone elapsed-time comments that do not identify the build, environment,
or workload. The full machine-readable and generated reports remain under
`BenchmarkDotNet.Artifacts/results`.

## Environment

| Item | Value |
| --- | --- |
| BenchmarkDotNet | 0.15.8 |
| OS | Windows 11 25H2, build 10.0.26200.8655 |
| CPU | AMD Ryzen 7 5700X 3.40 GHz |
| Cores | 8 physical, 16 logical |
| .NET SDK | 10.0.301 |
| Runtime | .NET 10.0.9, X64 RyuJIT x86-64-v3 |
| GC | Concurrent Workstation |
| Launches / warmups / measurements | 1 / 5 / 10 |

The workspace was not a Git repository when this baseline was collected, so a
commit identifier is unavailable. The dated source state and generated full
JSON report identify this initial comparison.

## Workload

- one epoch containing 192 training steps and 32 evaluation samples;
- deterministic in-memory MNIST-shaped inputs (`28 x 28`, 10 classes);
- one attention head, hidden size 4, and one Transformer layer;
- learning rate `0.001`, model seed 3, Trainer seed 7;
- model construction occurs in `IterationSetup` and is excluded from the
  measured operation;
- the operation includes sample conversion, forward, loss, backward, AdamW
  update, evaluation, and metrics aggregation.

## Results

| Build | Mean | Error | StdDev | Time ratio | Gen0 collections/op | Allocated/op |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Debug | 412.5 ms | 3.16 ms | 1.88 ms | 1.00 | 2 | 45.57 MB |
| Release | 104.1 ms | 1.13 ms | 0.59 ms | 0.25 | 2 | 45.54 MB |

Release is approximately four times faster for this workload. Managed
allocation is effectively unchanged between configurations, showing that the
initial allocation target is the common learning path rather than a
configuration-specific behavior.

These numbers are valid only as a same-machine baseline. Future optimization
PRs must rerun both jobs on the same environment and include the complete
report, numerical regression tests, and an explanation of any workload change.
