# Arc B580: XMX operand layout, tile selection and attention tuning

This continues [the detailed profiling pass](arc-profile-tuning-2026-09-21.md).
The selected kernels are enabled in the normal CLI without changing the JSON,
model, optimizer, checkpoint or dataset. CPU and CUDA implementations are unchanged.

## Scope and conditions

- Intel Arc B580, OpenCL driver 32.0.101.9030, Windows x64, .NET 10 Release.
- `training.transformer.json`: B16, accumulation4, T1024, L32, width512,
  heads16, hidden1536, vocabulary11500, 90,507,500 parameters, mix8_32/block32,
  tied embeddings, dropout0.1, Muon NS5 LR0.001 plus AdamW LR0.0003, WD0.01.
- Seed1234, fresh model, fixed synthetic token/target batches, fixed LR.
  Dataset/tokenizer I/O, saving and generation are not in the measured update.
  An update contains four microbatches (65,536 tokens).
- GPU tests and benchmarks ran serially, with no competing NNtrain.Cli process.
- Final A/B uses the same binary and config hashes, 3 warmup + 10 measured
  updates per route, synchronized phase wall times and OpenCL event profiling.
  The profiler's GPU durations and host wait durations overlap: do not add
  them together as independent Amdahl costs.

## Adopted implementation

1. **DPAS-friendly shared-memory B layout.** Adjacent BF16 values along K are
   packed once, into panels matching the matrix-MAD operand layout. Subgroup
   `intel_sub_group_block_read8` replaces repeated scalar SLM reads/packing.
   A uses a padded SLM tile. No host packing or extra global scratch is added.
2. **Larger output tiles where they pay.** 256x128x32 tiles reduce repeated
   operand loading for large output grids. 128x64x32 tiles keep small/long-K
   products sufficiently parallel (notably loss dX and Newton-Schulz).
   Auto uses wide when M>=256, N>=128, M*N>=524,288; narrow when M>=128,N>=64;
   tiny shapes keep the legacy kernel. Non-SG16 devices retain the legacy
   selection; unsupported XMX still follows the existing FP32 CUDA-independent
   Arc fallback. This is a measured heuristic, not a universal peak-performance
   autotuner.
3. **Compile-time NN/NT/TN/TT addressing.** The four transpose combinations
   have dedicated kernels, retaining all gate, bias, ReLU and accumulate cases.
   Existing parallel split-K and deterministic reduction stay unchanged.
4. **Attention dQ/dK/dV loop unrolling.** Constant-width FP32 loops with tail
   predicates preserve the original FMA sequence. No BF16 substitution in
   backward attention, no approximate math flags.
5. **QK attention uses the same packed B panels.** Strided head/batch addressing,
   causal masking and BF16 rounding remain unchanged.
6. **Driver resource telemetry.** `ArcExecutionLane.GetKernelResources` reports
   maximum workgroup size, local/private bytes and Intel spill bytes. Unsupported
   queries return null, never a misleading zero. Selected GEMMs report no spills
   on this driver; wide SLM is25,088 bytes and narrow SLM is12,544 bytes/workgroup.

The implementation uses Intel XMX/matrix-MAD and local block-I/O intrinsics,
not hand-written assembly. CUDA BF16 GEMM and FlashAttention staging code were
used as design references; CUDA PTX/cp.async is not portable to Arc. No Intel
assembler/disassembler is installed, so generated ISA was not disassembled.
Precision remains BF16 matrix operands with FP32 accumulation/destinations.
Weights/activations/gradients do not cross PCIe to implement these changes.

## Measured experiments, including rejections

All microbenchmarks use resident inputs, 3 warmup+10 dispatches per shape,
GPU event and wall timings, and CPU-checked output samples outside timing.
Full-array correctness tests separately exercise tails and arbitrary finite
FP32 inputs rounded to BF16. Retired experimental variants are not shipped;
their benchmark JSONs remain as evidence.

| Trial | Observation / decision |
|---|---|
| Packed pairs and128x128 tile | Main forward4.17->2.63 ms; proceed |
| 256x128 tile | Main forward2.51 ms; proceed |
| Subgroup block I/O | Main forward1.99 ms, dW3.00 ms; adopt |
| Transpose specialization | Main forward1.87 ms, dW2.92 ms; adopt |
| Wider128x256 tile | 9,536 spill bytes reported; slower; reject |
| Larger K tiles64/128 | More SLM without consistent speedup; reject |
| Large GRF compiler option | Removed spills in the widest trial, but slower overall; reject |
| A-side panel/block I/O | Little/no improvement over B-only layout; reject |
| Private-array lookahead | Correct, but slower (wide forward~2.34 ms); reject |
| Wide tile on every shape | Loss dX2.52->2.95 ms regression; use narrow there |
| Attention loop unroll | Small-probe backward tile31.73->28.97 ms; adopt |
| Four columns per attention work item | Backward tile44.10 ms; reject |

Primary microbenchmark artifacts under `benchmark-results/`:

- `arc-xmx-tiles-v1-20260921.json`, `arc-xmx-tiles-v2-20260921.json`
- `arc-xmx-tiles-large-grf-20260921.json`
- `arc-xmx-block-io-20260921.json`, `arc-xmx-panel-special-20260921.json`
- `arc-xmx-pipeline-20260921.json`, `arc-xmx-final-micro-20260921.json`

Small-model screening uses production B16/T1024/width/heads/hidden/vocabulary,
but L2 and accumulation1. Successive builds, not full-model speed claims:

| Route | p50 ms/update |
|---|---:|
| Shape-selected XMX only | 344.31 |
| + Attention loop unroll | 340.99 |
| + QK panels | 339.67 |
| Four-column attention trial (rejected) | 354.85 |

`arc-xmx-auto-full-trial-20260921.json` (1warm+3measure, full shape) measured
11,931.90ms/update before the additional attention changes, versus15,643.06ms
in `arc-xmx2-baseline-full-20260921.json`. Final evidence follows below.

## Final full-shape comparison

Both routes used identical Core/Arc/probe binary SHA256 hashes and the same
configuration SHA256. Results are not estimated from individual kernels.

| Metric | Previous route | Selected default |
|---|---:|---:|
| p50 ms/update | 15,628.36 | **11,657.08** |
| Mean ms/update | 15,630.55 | 11,659.62 |
| Tokens/s | 4,193.40 | **5,621.99** |
| Mean forward ms | 4,185.17 | 3,077.48 |
| Mean backward ms | 11,033.96 | 8,269.11 |
| Mean optimizer ms | 384.91 | 286.64 |
| Mean clip ms | 20.23 | 20.12 |
| Peak backend allocation MiB | 5,386.63 | 5,386.63 |
| End-update active MiB | 1,478.14 | 1,478.14 |
| End-update cached MiB | 497.99 | 497.99 |
| Native allocations/update | 2,928 | 2,928 |
| Kernel launches/update | 61,663 | 61,663 |
| Synchronizations/update | 688 | 688 |
| H2D bytes/update | 524,288 | 524,288 |
| D2H bytes/update | 2,588 | 2,588 |
| D2H scalar copies/update | 261 | 261 |

**25.41% shorter updates / 1.3407x throughput.** The new speedup is kernel-side,
not an increase in batch size, extra host processing or removal of validation.
The memory metric includes active, cached and retired native allocations; it
is **not driver-reported total dedicated VRAM**. End-update retired bytes are
zero and active/cached allocations are stable over the measured window.

Mean device event groups illustrate where the saving came from:

| Kernel/shape | Before ms/update | After ms/update |
|---|---:|---:|
| dW M1536/N512/K16384 | 1,627.14 | 732.31 |
| dX M16384/N512/K1536 | 1,259.95 | 658.15 |
| Forward M16384/N1536/K512 | 1,065.74 | 484.58 |
| Attention dQ/dK/dV | 2,033.13 | 1,854.41 |
| Forward QK | 324.02 | 283.14 |
| Backward QK recomputation | 323.76 | 282.00 |

All measured losses/norms/gradients are finite. Maximum absolute loss difference
between corresponding measured updates is **0.0000635**, maximum norm difference
0.00031577; final losses are 8.560418 versus 8.560478. Full training trajectories
are not bitwise identical, despite exact matches in the bounded kernel tests.
These synthetic trajectories do **not** establish equal real-corpus convergence.

Amdahl after the changes: backward 70.92%, forward 26.39%, optimizer 2.46%.
Even halving optimizer cost would yield only 1.0124x overall. Halving backward
would yield 1.5494x; attention backward (1,854 ms/update), remaining GEMMs and
attention derivative traffic remain the next substantial targets. The reported
kernel event times overlap host waits and cannot be added to the phase totals.

Artifacts:

- [Final baseline](../benchmark-results/arc-xmx-final-baseline-full-20260921.json)
- [Final optimized](../benchmark-results/arc-xmx-final-optimized-full-20260921.json)
- [Retained GEMM microbenchmarks](../benchmark-results/arc-xmx-final-micro-20260921.json)

## Validation and reproduction

- Release solution build: 0 warnings/errors.
- Selected Core tests:154 passed,0 failed,0 skipped (CUDA-named tests excluded).
- Selected Integration tests:202 passed,0 failed,0 skipped.
- GEMM tests check exact full-array equality against the old route for all
  transpose directions, BF16 rounding, tails, gates, bias/ReLU, split-K and
  accumulation, plus no added host transfers and automatic selection bounds.
- Attention tests compare complete outputs and repeated accumulated gradients
  exactly, including causal/noncausal and head widths16/31/32/65.
- Long-K, varying-exponent GEMMs also pass the existing Arc mixed-matrix
  reference tolerance (absolute 1e-5), including repeated accumulation.
- No existing test tolerances were widened. This is356 targeted tests, not a
  claim of all-solution or unavailable CUDA hardware coverage.
- Test records: `benchmark-results/test-results/arc-xmx-core-final-20260921.trx`
  and `arc-xmx-integration-20260921.trx`.

```powershell
dotnet run --project .\NNtrain.Benchmarks -c Release --no-build -- `
  --probe-arc-transformer .\training.transformer.json .\benchmark-results\new-baseline.json `
  --warmup 3 --steps 10 --profile --arc-mode pre-xmx

dotnet run --project .\NNtrain.Benchmarks -c Release --no-build -- `
  --probe-arc-transformer .\training.transformer.json .\benchmark-results\new-optimized.json `
  --warmup 3 --steps 10 --profile

dotnet run --project .\NNtrain.Benchmarks -c Release --no-build -- `
  --probe-arc-xmx-tune .\benchmark-results\new-gemm-comparison.json
```

Output paths must be new. `--xmx-mode legacy|narrow|wide|auto` and
`--attention-mode legacy|unrolled|panel|optimized` provide independent A/B
overrides without editing production configuration. Normal training uses
Auto+optimized. `--arc-mode pre-xmx` freezes the immediate previous route.

Long real-corpus convergence and a long-duration memory soak are not established
by these bounded runs. The former2xRTX3070Ti hardware is unavailable here, so
CUDA parity and closeness to hardware peak are **not verified**.

Official implementation references:

- [Intel subgroup matrix MAD extension](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html)
- [Intel subgroup local block I/O](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_local_block_io.html)
- [Intel kernel spill/resource query](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_required_subgroup_size.html)
- [Intel XMX programming guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/programming-intel-xmx-using-sycl-joint-matrix.html)
