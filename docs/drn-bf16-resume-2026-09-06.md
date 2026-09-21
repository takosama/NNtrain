# DRN mixed-BF16 resume and speed — 2026-09-06

## Changes

- Explicit `precisionMode` now permits training-state migration between
  `float32`, `mix16_32`, and `mix8_32`. Omission still inherits the checkpoint,
  including legacy F16 storage. Saved master weights and FP32 optimizer state
  are restored through the existing bounded streaming readers. Scheduler,
  step, RNG and data cursor restoration are unchanged. Migration is logged.
- Pure BF16/BFP8 optimizer-state migration is explicitly rejected, not silently
  reset. The user's experiment uses `mix16_32`: BF16 operands/storage and FP32
  master weights, gradients and optimizer state.
- Connected mixed-BF16 DRN to the existing exact checked chunk forward and
  fused replay/backward kernels. Previously these were only connected to BFP8.
  No BF16-to-BFP8 roundtrip is introduced. Checkpoints replace full FP32
  recurrent history. Gradients remain FP32. Pure-BF16 and recurrent inference
  continue using their original numerical paths.
- Per-layer boundary limit increased from 4 to 8 MiB; model/replica boundary
  retention remains capped at 256 MiB. Unsupported shapes retain old kernels.
- Production JSON changes only microbatch 42 → 14 and accumulation 1 → 3;
  effective batch remains 42. Precision was already `mix16_32`. LR, tokenizer,
  corpus, model dimensions and checkpoint settings are not changed.

## Frozen A/B benchmark

2 × RTX 3070 Ti, Release, DRN 48 layers, width 256, hidden 768, K/V 32/32,
context 1024, vocabulary 11500, dropout 0.1, seed 1234, `mix16_32`.
Microbatch 14 (7/7 shards), accumulation 3, effective batch 42.
Muon LR .001, momentum .95, fixed NS5 every update; AdamW LR .0003,
beta2 .95, weight decay .01, gradient clip 1.

Synthetic fixed-seed full-length tokens, fixed LR. No FineWeb/tokenizer reads,
checkpoint writes, generation or LR schedule. A1 → B1 → B2 → A2, separate
processes, each 10 warmup + 20 measured optimizer updates. Five additional
synchronized diagnostic updates per run are excluded from wall p50.

| Metric | Original BF16 | Chunk BF16 |
|---|---:|---:|
| Pooled 40-update p50 | 867.040 ms | 607.167 ms |
| Mean | 869.474 ms | 608.993 ms |
| GPU0 memory after diagnostic steps | 5916.5 MiB | 4678.5 MiB |
| GPU0 allocator bytes | 4414.32 MiB | 3159.60 MiB |
| Measured allocation/free calls | 0/0 | 0/0 |
| Graph capture/replay/fallback per measured run | 0/60/0 | 0/60/0 |

**29.97% shorter updates, 1238 MiB lower observed GPU memory.**
Individual p50: A1 872.300, A2 864.244, B1 605.882, B2 609.212 ms.

Extra-synchronized phase means (not an additive decomposition of normal graph
timing): forward/backward/reduce 792.37 → 531.64 ms; zero 13.04 → 15.05 ms;
clip 19.10 → 20.66 ms; Muon 41.38 → 42.04 ms; AdamW 1.99 → 2.24 ms.
The dominant measured improvement is in forward/backward, not optimizer work.

Exploratory batch16/accum1, 3 warmup + 10 measured: 409.30 → 324.11 ms,
6198.5 → 4770.5 MiB. Do not compare its update timing to the batch42 experiment.

Batch42/accum1 OOMs with both implementations. The original fails in forward;
the candidate reaches backward but cannot allocate a 989,184,000-byte buffer
for the vocabulary output path. Microbatch14/accum3 is the tested workaround,
not a claim that the large-batch memory bottleneck has been eliminated.

Native ABI 1.34 unchanged, DLL SHA256:
`AA0900F52D7CD947010277119FEFCC31084A1F42F8EDA13F856FECB11285ABF9`.
Raw conditions and samples: `benchmarks/drn-bf16-resume-2026-09-06/`.

## Verification and limitations

- CLI Release build: 0 warnings/errors.
- Checkpoint/resume/epoch integration subset: 47 passed, 0 skipped/failed.
  Includes AdamW and Muon mix8 → mixed-BF16/FP32 migration: exact restored
  master values and optimizer state, scheduler/global step, next update and
  re-save metadata. Unchanged-mode resume remains bitwise tested.
- ForgetMemory/DRN/ABI CUDA/Core subset: 114 passed, 4 skipped, 0 failed.
  This is a subset, not a full solution or long production soak.
- New resident BF16 tests cover B2/T65/K16/V32, B2/T513/K32/V32,
  B21/T1024/K32/V32: forward outputs bitwise equal to the original, gradient
  absolute error <= existing 6e-5 tolerance. Native chunk tests also cover
  exact fallback and nonzero terminal state gradients.
- Repeated optimizer updates are not bitwise identical due to changed FP32
  accumulation order and subsequent BF16 rounding; no tolerance was widened.
- The user's actual checkpoint path is absent in this workspace. We did not
  delete it, restore that run, or prove that loss will fall below 4. Synthetic
  throughput results do not establish convergence. Real-data migration and
  long-run convergence remain to be tested with the user's checkpoint.
- Do not compare different tokenizers, token budgets or effective batches to
  conclude that 8-bit precision caused the loss plateau. Starting from the
  same checkpoint with the same held-out tokens gives a meaningful test.
