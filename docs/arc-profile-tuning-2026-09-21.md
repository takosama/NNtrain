# Arc B580: detailed profiling and repeated optimization

## Scope and measurement

This pass continues [the earlier backward tuning](arc-backward-tuning-2026-09-21.md).
It is followed by [the XMX operand-layout pass](arc-xmx-layout-tuning-2026-09-21.md);
use `--arc-mode pre-xmx` to reproduce the selected default measured in this report.
The production CLI uses the selected defaults without a JSON change. No
checkpoint, corpus, tokenizer, learning rate, batch or model setting was edited.
This is Transformer optimization, not a claim of DRN support or CUDA parity.

`ArcDetailedProfiler` records per-phase/per-shape GPU event duration, host
submission duration, native allocation/pool-hit size counts, H2D/D2H bytes,
synchronization reason and deferred-release bytes. Enable it in the bounded
Transformer benchmark with `--profile`; it is off by default in training.
Labels are aggregated and bounded, not one retained record per launch. The
benchmark prints the ten largest device-kernel groups and saves all counters.
Device transfers performed through staged host arguments and allocation-time
uploads are included, not just explicit `Read`/`Write` calls.

GPU event durations and host synchronization durations **overlap**. Queue delay
also overlaps across pending commands. They must not be added together as
serial Amdahl costs. The disjoint synchronized forward/backward/clip/optimizer
wall times are used for Amdahl fractions. The explicit finite-gradient audit
is outside timed updates and transfer deltas.

Conditions: Intel Arc B580, OpenCL driver 32.0.101.9030, .NET 10 Release,
`training.transformer.json`, B16/T1024/L32, accumulation4 (effective batch64),
width512, heads16, hidden1536, vocabulary11500, 90,507,500 parameters,
mix8_32/block32, tied embedding, dropout0.1, Muon fixed NS5 LR0.001 plus AdamW
LR0.0003, weight decay0.01, seed1234. Tokens/targets are fixed synthetic data,
the model is freshly seeded, and LR is fixed. Corpus reading, checkpointing
and sample generation are excluded. No competing NNtrain.Cli process was
running during the reported selected measurements.

## What was actually slow

Initial 1-warmup/3-measurement profile: **18,145.24 ms/update**. In its last
measured update, GPU kernels accounted for 16,242.92 ms, native allocations
for 479.30 ms, with **3,992 native allocations and 1,600 synchronizations**.
The CPU wait time is mostly waiting for those same GPU kernels, not another
independent 16 seconds of work.

The dominant kernel groups included approximately 2,170 ms of narrow attention
backward, 2,028 ms of the largest dW shape, and 1,875 ms of the largest dX shape.
Optimizer time was only about 2% of the update. Doubling that small phase would
barely change total performance; backward was the appropriate first target.

### Adopted changes

1. **Mixed backward operand policy corrected.** Arc Linear and loss-head
   backward were using full FP32 GEMMs even when both stored operands and the
   active central precision policy specified a mixed mode. The existing CUDA
   implementation in `CudaBfp8Gemm.LinearBackward` encodes the upstream gradient
   as BF16 for GEMM. Arc now follows that same contract: BF16 operand rounding,
   XMX GEMM, FP32 destinations and gradient accumulation. Float32 mode is not
   changed. ReLU masking and bias reduction consume the same encoded gradient
   as CUDA. The upstream gradient itself is not overwritten. The staged Arc
   reference follows the same policy, and independent CPU-formula tests check
   it. This is an intentional numerical-policy correction, not a claim of
   bitwise equivalence with the erroneous all-FP32 mixed backward.
2. **Parallel XMX split-K for dW.** Independent K slices execute in one 3D
   dispatch, followed by deterministic FP32 reduction. Scratch is bounded to
   64 MiB; other shapes keep the existing Arc device path. Tail sizes,
   existing gradient accumulation and FP32 destination storage are tested.
3. **Bounded deferred native frees.** Previously an uncached temporary could
   force an immediate queue-wide fence. Up to 128 MiB of retired buffers now
   remain owned and accounted until a known queue-complete boundary. They are
   never freed while kernels still use them. Normal completion, failed
   dispatch and lane disposal drain the resources.
4. **Narrow attention backward tiles.** dQ/dK/dV use a 32-row rather than
   64-row FP32 tile. Forward keeps the previous tile. This increases independent
   work and reduces local-memory/register pressure without rounding operands
   differently. Causal/noncausal cases, tail dimensions and two accumulated
   backwards match the old output and gradient arrays exactly in tests.
5. **Bounded buffer reuse.** Raising the session-owned exact-size pool from
   256 to 512 MiB reduces repeated allocation and release fences. It is a cap,
   not an upfront reservation or a static permanent cache. The separate
   retired-buffer budget is still included in the peak allocation counter.

All new scratch, reductions, gradient encodings and reuse stay on the device.
There is no new weight/activation/gradient D2H or optimizer-state H2D per step.

## Iteration evidence

Full production shape, 1 warmup + 3 measured updates in the exploratory runs:

| Variant | p50 ms/update | Peak backend MiB | Native allocations in last update |
|---|---:|---:|---:|
| Initial profile | 18,145.24 | 5,162.63 | 3,992 |
| Deferred frees only | 17,764.02 | 5,226.63 | 3,992 |
| Deferred + mixed XMX + parallel dW | 16,262.07 | 5,258.63 | 3,913 |
| Above + compact attention backward | 16,148.30 | 5,258.63 | 3,913 |
| Mixed XMX + deferred + 512 MiB pool, old attention tile | 15,760.33 | 5,386.63 | 2,928 |

These exploratory runs used successive builds. Final same-binary comparison
is recorded separately below. Allocator peak includes active, cached and
retired device allocations; it is **not driver-reported dedicated VRAM**.

Artifacts, under `benchmark-results/`:

- `arc-profile-baseline-full-20260921.json`
- `arc-profile-deferred-full-20260921.json`
- `arc-profile-mixed-full-20260921.json`
- `arc-compact-backward-full-20260921.json`
- `arc-pool512-full-20260921.json`

### Trials not adopted

- Increasing attention scratch from 64 to 128/256 MiB: the B16/T1024/L2,
  accumulation1 probe regressed from 507.07 to 512.29/540.86 ms. Default stays64.
- Preserving arbitrary FP32 backward operands through three BF16 XMX products:
  added packing/products outweighed the benefit. Both on-tile and prepacked
  versions were slower. This also proved unnecessary after checking CUDA's
  actual mixed-precision operand contract.
- Larger FP32 GEMM tiles: some individual shapes improved, others regressed;
  the mixed-XMX policy correction superseded that candidate.
- Packed backward XMX input trial: 515.43 versus 477.84 ms for the unpacked
  mixed split-K probe. No adoption.
- XMX attention backward, including a narrow-tile version: 494.86/491.45 versus
  477.84 ms. Head width32 did not benefit; FP32 attention arithmetic is retained.
- Compact tiles for both forward and backward: only 476.52 ms. Restricting them
  to backward improves to 475.95 ms; forward keeps its old route.
- Increasing queue depth128 to512: 466.32 versus 467.39 ms after pooling and
  compact backward, only 0.23%. Insufficient benefit to justify retaining more
  pending commands/events, so the experimental option was removed.

Unless noted otherwise, the small probes use 3 warmup + 10 measurements, with
production width/head/hidden/vocabulary/context and only layers/accumulation
reduced to2/1. They are screening measurements, not full-shape speed claims.
Only this pass's rejected experimental source files were removed; their result
JSONs remain. `arc-compensated-gemm-v2-20260921.json` was contaminated by a
concurrent user training run and is excluded; its isolated rerun also regressed.

## Final same-binary comparison

Both routes used **3 warmup + 10 measured updates**, detailed profiling enabled,
identical configuration and binary SHA256 hashes, with no dimension overrides.
Each optimizer update contains four microbatches (65,536 tokens total).

| Metric | Starting route | Selected default |
|---|---:|---:|
| p50 ms/update | 18,148.12 | **15,632.21** |
| Tokens/s | 3,611.17 | **4,192.37** |
| Mean forward ms | 4,228.39 | 4,187.84 |
| Mean backward ms | 13,501.50 | **11,034.46** |
| Mean optimizer ms | 384.59 | 383.83 |
| Mean clip ms | 20.84 | 20.71 |
| Peak backend allocation MiB | 5,162.63 | 5,386.63 |
| Native allocations/update | 3,992 | 2,928 |
| Queue synchronizations/update | 1,600 | 688 |
| H2D bytes/update | 524,288 | 524,288 |
| D2H bytes/update | 2,588 | 2,588 |

**13.86% less time, 1.161x throughput.** The extra peak backend memory is
224 MiB. End-of-update active allocation is 1,478.14 MiB in all ten optimized
updates, cached allocation497.99 MiB, retired allocation0. Peak does not grow.
Measured times range15,623.03–15,645.70 ms, with no late-step slowdown in this
short run. This does not replace a long-run memory soak or driver VRAM sampling.

The H2D traffic consists of8 token/target copies per update. D2H consists of
4 loss scalars, one12-byte clipping result, and128 sets of Muon statistics
(16-byte confidence statistics plus a4-byte norm), **261 copies total**.
Those optimizer scalar synchronizations remain; they are not hidden in a claim
of zero transfer. No parameter, activation or gradient arrays cross to the host
inside measured updates. Startup, explicit inspection and checkpoint transfers
are outside these step counters.

Loss and all gradients remain finite. The final synthetic loss is8.560406
versus8.560417 in the starting route; these similar values are a smoke check,
not evidence of real-corpus convergence. Mixed backward intentionally changes
operand rounding to the documented policy, and existing embedding atomics can
also produce small run-to-run differences.

### Amdahl interpretation and remaining bottlenecks

Backward initially occupied74.4% of wall time. Its measured speedup is
13,501.50/11,034.46 =1.224x. Amdahl predicts roughly1.158x overall from that
phase alone; the measured total1.161x is consistent with that improvement
plus the smaller forward/allocation changes.

Backward still occupies70.6% of the optimized update. Doubling it alone would
yield approximately1.545x overall; doubling the2.46% optimizer phase gives
only about1.012x. Last-update GPU kernel time is14,894.64 ms of15,630.95 ms
wall time, so eliminating host overhead alone cannot close a large remaining
performance gap. The next substantial gains must come from device arithmetic
and device-memory access, not just smaller host copies.

Largest remaining mean device groups:

| Group | ms/update |
|---|---:|
| Attention dQ/dK/dV, M1024/N32/K1024 | 2,031.35 |
| XMX dW, M1536/N512/K16384 | 1,627.21 |
| XMX dX, M16384/N512/K1536 | 1,255.43 |
| Forward XMX, M16384/N1536/K512 | 1,065.57 |
| XMX dW, M512/N1536/K16384 | 785.15 |

Artifacts:

- [`arc-profile-final-baseline-20260921.json`](../benchmark-results/arc-profile-final-baseline-20260921.json)
- [`arc-profile-final-optimized-20260921.json`](../benchmark-results/arc-profile-final-optimized-20260921.json)

## Validation and reproduction

- Release solution build: 0 warnings, 0 errors.
- Core targeted tests: **135 passed, 0 failed, 0 skipped**.
- Integration targeted tests: **138 passed, 0 failed, 0 skipped**.
- Coverage includes Arc kernels/residency, NekoMuon numerical/state behavior,
  CPU AdamW storage transitions, Tensor characterization, Arc training, Wiki
  CLI and checkpoint/configuration tests. New tests cover BF16 mixed operands
  with FP32 gradients, XMX split-K tails/accumulation, exact compact-attention
  equivalence, bounded deferred lifetimes and profiler transfer accounting.
- Existing test tolerances were not widened. This is **273 targeted tests**,
  not a claim that the complete solution or unavailable CUDA hardware passed.

```powershell
dotnet run --project .\NNtrain.Benchmarks -c Release --no-build -- `
  --probe-arc-transformer .\training.transformer.json .\benchmark-results\arc-new-profile.json `
  --warmup 3 --steps 10 --profile
```

Use `--arc-mode pre-profile` for the starting route, `deferred` for only deferred
frees, `mixed` for mixed XMX + parallel dW + deferred frees, `compact` to add
the small backward tile, or `pool512` for mixed+deferred+larger pool without
compact attention. `pooled` freezes this pass's selected changes; `optimized`
now also includes the subsequent XMX layout pass. The result path must not
exist; the probe refuses to overwrite it.

Full real-corpus convergence, long-run memory stability and a current CUDA
comparison remain unverified. The current host does not have the earlier pair
of RTX3070Ti GPUs available for a like-for-like CUDA run. Synthetic loss and
finite gradients alone cannot establish convergence or CUDA performance parity.
