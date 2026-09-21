# Arc B580: XMX, backward tiling and bounded command batching

## Scope and outcome

Transformer Arc backend optimization; model, data, optimizer, precision policy,
batch size and accumulation are unchanged. No dataset/tokenizer/checkpoint was
overwritten. No commit or push is part of this tuning run.

Hardware: Intel Arc B580, OpenCL driver 32.0.101.9030. The real device reports
minimum subgroup size 16 and `cl_intel_subgroup_matrix_multiply_accumulate`.
The only NVIDIA adapter currently installed is a GT 710, not the previous
RTX 3070 Ti pair. **CUDA performance parity is not established.**

## Adopted implementation

1. Attention is head-tiled batched GEMM: QK, softmax, PV, and owner-computed
   dQ/dK/dV without floating-point atomics. P and dS are recomputed in backward,
   not saved per layer. Two scratch buffers are limited to 32 MiB each; shapes
   exceeding that per-head budget keep the streaming fallback.
2. BF16-operand GEMM and QK use Intel XMX, FP32 accumulation. A 128x64 output
   tile reuses 32-deep cooperative local-memory tiles. Tail, transpose, bias,
   gate, ReLU and ordered split-K are supported. Device capability and minimum
   subgroup size gate compilation/dispatch; the subgroup-8 variant is implemented
   but has **not** been exercised on this subgroup-16 B580.
3. FP32 operands/gradients are **not** rounded to select XMX. FP32 GEMM uses
   64x64 tiles, four-by-four output registers per thread, ordered FP32 FMA.
   Narrow attention products use 64x32x32 tiles, preserving ordered arithmetic.
4. Bias and LayerNorm parameter gradients use 32-column/256-row blocks followed
   by a deterministic second reduction. No float atomics or host partial sums.
5. In-order commands are submitted in bounded groups (maximum 128 profiling
   events), not fenced twice per kernel. Pool reuse is ordered on the same
   queue; eviction, failure, readback and disposal are fence boundaries.
   GPU event timing is drained after completion. Benchmarks explicitly finish
   each measured phase, so timings do not mistake launch latency for GPU work.
6. Loss-head chunks grow from 128 to 512 rows, also capped at 32 MiB per
   logits/dLogits buffer. This amortizes vocabulary-weight gradient updates.
   Full token-by-vocabulary logits are still never retained for the model.

XMX uses the [Intel OpenCL matrix extension](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html).
The first naive 8-row XMX implementation regressed the small benchmark to
199.78 ms and was replaced; it is not the final kernel. The 32-row cooperative
version still lost on small optimizer matrices. The final 128-row reuse tile
improved measured matrix shapes, including 16384x1536x512 from 10.7634 ms
(original FP32 tile) to 4.1814 ms (XMX), 3 warmup + 10 samples.
See `benchmark-results/arc-gemm-xmx-v2-20260921.json`,
`arc-gemm-wide-20260921.json`, and `arc-gemm-xmx-v3-20260921.json` for iterations.

## Controlled final A/B: same binary and configuration hashes

B4/T512/L2, accumulation1, width512, heads16, hidden1536, vocab11500,
dropout0.1, tied embedding, mix8_32/block32, Muon NS5 LR0.001 + AdamW
LR0.0003, weight decay0.01. Fresh seed1234, identical synthetic full-length
tokens, fixed LR, no corpus/checkpoint/scheduler/sampling. 3 warmup + 10 updates.

| Metric | Residency-only route | Tuned default |
|---|---:|---:|
| p50 ms/update | 187.73 | 89.85 |
| Tokens/s | 10,909 | 22,795 |
| H2D bytes/update | 16,384 | 16,384 |
| D2H bytes/update | 176 | 176 |

**2.09x faster**, with no new host tensor traffic. The maximum difference in
the ten logged losses is below 0.0001, not bitwise equality. XMX accumulation
and block reductions can change rounding; existing correctness tolerances were
not relaxed. This short synthetic run is not a convergence study.

Artifacts: `benchmark-results/arc-final-resident-b4-s512-20260921.json`
and `benchmark-results/arc-final-tuned-b4-s512-20260921.json`.

## Full production JSON shape

B16/T1024/L32, accumulation4 (effective batch64), 90,507,500 parameters;
the remaining conditions are as above. All dimensions come from the unchanged
`training.transformer.json`. Each update processes 65,536 token positions.

| Revision | Warmup / measured | p50 ms/update |
|---|---:|---:|
| Prior resident streaming attention | 1 / 3 | 72,691.16 |
| Batched attention, first cooperative XMX, block reductions | 1 / 3 | 30,314.28 |
| Wider GEMM and narrow attention tiles | 1 / 3 | 25,171.08 |
| Final command batching and larger loss chunks | 3 / 10 | **20,753.20** |

Final throughput: **3,157.9 tokens/s**. Relative to the earlier full-shape
resident measurement: **3.50x**, 71.45% less update time. These full-shape
iterations used different binaries/time windows and warmup counts; the
controlled same-binary A/B is the smaller-shape pair above.

Final phase means: forward4,714.45 ms, backward15,626.33 ms, optimizer381.89 ms,
clip20.06 ms, zero6.92 ms. Backward still accounts for75.3%; Amdahl predicts
1.604x total improvement if that entire phase is halved, not 2x overall.
The largest remaining kernel groups are FP32 GEMM (~7.73s/update), narrow
attention (~2.97s), and XMX GEMM (~2.46s). Whole-phase CUDA parity and a
high-performance fused XMX FlashAttention implementation remain future work.

Every final measured update:

- H2D **524,288 bytes**: token/target IDs only.
- D2H **2,588 bytes**: four loss scalars, clip/numeric status and Muon matrix
  confidence/normalization statistics. These statistics remain per matrix;
  they are not claimed to be a single constant-count metric transfer.
- No weight/activation/gradient/master/moment-array readbacks inside the step.
- Peak native allocations including pool **5,138.626 MiB**, retained after
  update **1,478.139 MiB**; both stable across all ten updates. Peak is only
  3.815 MiB above the prior full-shape resident result. These are **backend
  counters, not driver-reported dedicated VRAM**.
- All losses and gradients finite; measured loss9.171261 → 8.560395.
  No OOM. No claim of a long real-data convergence or memory soak.

Final artifact: `benchmark-results/arc-tuned-full-20260921.json`.
Intermediate artifacts: `arc-xmx-reduced-full-20260921.json`,
`arc-wide-full-before-batching-20260921.json`.

## Validation and reproduction

Release solution build: **0 warnings, 0 errors**. Related Core tests:
**113 passed**, Integration tests: **138 passed**; zero failures/skips.
Includes real Arc execution, float32/mix16_32/mix8_32 forward/gradient and
optimizer comparisons, XMX transpose/tails/split-K/gates, exact wide-FP32
equivalence, multiple attention head tiles and causal/noncausal odd dimensions,
parallel normalization gradients, queued buffer reuse/eviction/failure, transfer
budgets, state transitions, checkpoint/resume/metrics/generation fixtures.

```powershell
dotnet build .\NNtrain.slnx -c Release --no-restore
dotnet run --project .\NNtrain.Benchmarks -c Release --no-build -- `
  --probe-arc-transformer .\training.transformer.json .\benchmark-results\arc-new-full.json `
  --warmup 3 --steps 10
```

Result paths must be new. Use `--batch 4 --sequence 512 --layers 2
--accumulation 1` for the small comparison, `--arc-mode resident` for the
frozen residency-only path, or default `optimized` for the adopted path.
`--probe-arc-gemm NEW.json` runs the bounded matrix microbenchmark.
The normal CLI command remains unchanged; no environment switches are needed.
