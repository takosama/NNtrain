# Arc B580 T2048 attention derivative follow-up (2026-09-23)

## Measured workload and diagnosis

- Intel Arc B580, driver 32.0.101.9030, Windows x64, .NET 10 Release.
- `training.transformer.json` SHA-256 `9B298E978EB748C5218CB87A8A9AF31BF13219519C3B28B5A39826F03C312CCD`: batch 16 × accumulation 8, sequence 2048, width 512, 16 heads/D32, 32 layers, `mix8_32` block 32, 262,144 tokens/optimizer update.
- Fixed-seed synthetic training benchmark, including forward, backward, clipping and optimizer, but excluding dataset/tokenizer/checkpoint I/O. Within each A/B series, the same executable and Core/Arc DLL hashes were used for every run.
- Before this change, the exclusive timeline attributed about 24.61/37.9 seconds per update to attention (64.9%). Derivative was about 4.88 seconds. The GPU command-idle interval was only about 0.57 seconds (1.5%). Thus this work targeted a GPU kernel, not host transfer.
- The dedicated derivative diagnostic (`benchmark-results/arc-attn-deriv-diagnostic-20260923.json`) measured production, streaming-only and reduction-only controls under one resident-input setup. The streaming control was about as expensive as production while reduction-only was much shorter; this motivates limiting rereads. It is **not** a hardware-bandwidth counter or an additive time decomposition. Kernel-resource queries reported 256 bytes of SLM and no spill for the accepted candidate. VTune/GPA hardware counters were unavailable on this host, so no measured DRAM-bandwidth percentage is claimed.

## Accepted prefix4 change

`attention_derivatives_prefix4_2048` caches the first four P/dP pairs of each modulo-64 lane stream for reuse after the row reduction. It keeps the original FP32 FMA order, reduction tree, causal mask and D32 scale. Dispatch is restricted to T2048/D32 and SG16/XMX-capable Arc. Other shapes and explicit subgroup-reduction sessions retain their established route. The A/B flag is `--deriv-prefix4-2048 on|off` in the transformer benchmark.

The full-update OFF→ON→ON→OFF sequence used **2 warmups + 4 measured optimizer updates per run**, with no competing GPU process. Run p50 values:

| Run | Kernel | ms/update | tokens/s |
| --- | --- | ---: | ---: |
| OFF1 | Original | 37,901.232 | 6,916.503 |
| ON1 | Prefix4 | 37,794.193 | 6,936.092 |
| ON2 | Prefix4 | 37,781.311 | 6,938.457 |
| OFF2 | Original | 37,906.436 | 6,915.554 |

The mean of the two ON run medians is **116.08 ms/update (0.306%) faster** than the mean of the two OFF medians, or about **+21.25 tokens/s**. All eight measured ON updates were faster than all eight OFF updates. Peak native allocation was 9,063.03 MiB for both. Per update H2D was 2,097,152 bytes, D2H 2,604 bytes, and native allocation/release about 27.17 GiB; these did not change. A separate short profiled A/B/A localized roughly 102 ms of GPU-kernel reduction to the derivative stage, consistent with the wall-clock difference. These are synthetic-training measurements, not a FineWeb throughput claim.

The ABBA artifacts are `benchmark-results/arc-attn-deriv-prefix4-abba-{a1,b1,b2,a2}-20260923.json`. A resident two-head microprobe measured derivative GPU p50 0.1478 → 0.1411 ms, with zero bit mismatches.

## Rejected candidates

The same resident-input probe (`benchmark-results/arc-attn-deriv-sg-dkv-unroll-20260923.json`, 3 warmups + 10 measurements per kernel) gave zero bit mismatches for each candidate, but:

| Candidate | Reference → candidate GPU p50 | Decision |
| --- | ---: | --- |
| Prefix4 plus SG16 reduction | 0.1411 → 0.1445 ms | Slower; keep benchmark-only |
| dK/dV inner unroll 4 | 0.1657 → 0.1701 ms | Slower |
| dK/dV inner unroll 8 | 0.1657 → 0.1660 ms | No gain |
| dK/dV inner unroll 16 | 0.1657 → 0.1650 ms | Sub-microsecond margin; wall p50 0.1882 → 0.1894 ms, not accepted |

## Prefix8 adoption and final bounded candidate

A subsequent exact-output prefix8 candidate measured 0.1398 ms against prefix4's 0.1410 ms in `benchmark-results/arc-attn-prefix8-20260923.json`. A preliminary same-binary A/B/A (1 warmup + 2 measured updates per run) measured prefix4 37,777.46 → prefix8 37,688.64 → prefix4 37,778.56 ms/update, with all candidate samples faster than all control samples and no peak-memory/transfer change.

The extended OFF→ON→ON→OFF (2 warmups + 4 measured updates per run) confirmed this: prefix4 run medians 37,781.699 and 37,797.403 ms/update, prefix8 medians 37,692.320 and 37,692.783. Means of run medians: **37,789.551 → 37,692.552 ms/update**, **97.00 ms/update (0.257%) faster**, approximately **+17.85 tokens/s**. All eight prefix8 measurements were faster than all eight prefix4 measurements. Peak native 9,063.03 MiB, H2D 2,097,152 B/update, D2H 2,604 B/update, and native churn about 27.17 GiB/update were unchanged. Config and all three binary hashes matched across the four runs. Artifacts: `benchmark-results/arc-attn-prefix8-abba-{a1,b1,b2,a2}-20260923.json`. Prefix8 is accepted as the default T2048/D32 derivative path, with `--deriv-prefix8-2048 off` restoring prefix4 for A/B.

A further prefix16 probe is bitwise exact and showed derivative GPU p50 0.1377 ms against prefix8's 0.1397 ms in `benchmark-results/arc-attn-prefix16-20260923.json`, with no reported spill and the same 256-byte SLM footprint. The public-path six-case comparison (including production B16 shape and two backward accumulations) also passes bitwise. The preliminary whole-update A/B/A (1 warmup + 2 measurements per run) showed prefix8 37,681.96 → prefix16 37,621.94 → prefix8 37,691.36 ms/update, all samples separated, no peak-memory/transfer change. This is only about 0.17% of a full update. The extended ABBA was stopped before completion because its GPU time was disproportionate to the possible gain; it produced no result artifact. Prefix16 remains disabled and is **not** an accepted optimization.

The dK/dV and SG16-reduction candidates remain outside production dispatch. Previously tried full-row caches, larger attention workspaces and fused attention kernels were also slower; see `docs/arc-attention-t2048-b580-2026-09-23.md`.

## Correctness and remaining limit

Dedicated Arc tests pass bitwise comparisons for the derivative candidates across dense and causal modes, masked NaNs and head offsets; the dK/dV variants match across a batch/head boundary and two accumulations. The public attention path passes forward and two accumulated backward comparisons, including the production B16/T2048/W512/H16 `mix8_32` shape, with no implicit D2H during the operation. The later prefix8 comparison repeats all six public-path cases and passes as well.

The Release solution build completed with 0 warnings and 0 errors. The suite excluding the existing `ArcAttentionExperimentalAcceptance` category passed: Core 1,706 passed / 109 skipped; Integration 427 passed / 6 skipped; Benchmarks 46 passed / 0 skipped. Most skips are hardware-conditional CUDA tests on this Arc-only run; existing known-issue skips were retained. The separate six-case public-attention shape/precision comparison passed without skips.

Even entirely removing derivative would save at most about 4.88 seconds of this 37.9-second update, so a multi-fold step improvement requires redesigning more than this one kernel. The next high-impact measured slices are dK/dV and dQ; any rewrite needs an exactness gate and whole-update A/B before default activation.

## Compute-budget decision

For subsequent candidates, optimize measured end-to-end tokens/s gained per GPU-hour spent validating, with a minimum projected gain of 0.5 s/update (about 1.3% here) before another full-update A/B. A resident-input microprobe and bitwise correctness check may reject candidates cheaply. The previous 0.17% prefix16 pilot does not pass this gate, so no further full-update runs are justified for it. This threshold is a prospective decision rule, not a measured speedup.
