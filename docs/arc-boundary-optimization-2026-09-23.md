# Arc B580 Transformer operator-boundary optimization, 2026-09-23

## Conditions and scope

- Windows x64 / .NET 10 Release / Intel Arc B580 driver `32.0.101.9030`.
- `training.transformer.json` SHA-256 `9B298E978EB748C5218CB87A8A9AF31BF13219519C3B28B5A39826F03C312CCD` (not edited for this work).
- Transformer batch 16, gradient accumulation 8, sequence 2048, width 512, 16 heads, hidden 1536, 32 layers, vocabulary 11500, `mix8_32` block 32, Muon NS5 + AdamW; 262,144 synthetic tokens/update.
- Fresh fixed-seed model and tokens. Each final number is synchronous update wall-clock p50 from two warmup plus four measured updates, with profiling enabled. It does **not** include FineWeb/tokenizer, checkpoint, HTML, or generation I/O. GPU kernel/event times are not additive wall-clock slices.

## Adopted paths

| Incremental path | p50 ms/update | tokens/s | native ownership peak MiB | H2D/D2H bytes per update |
| --- | ---: | ---: | ---: | ---: |
| Frozen audit baseline, before the separately reported attention work | 38,989.10 | 6,723.5 | 9,147.8 | 2,097,152 / 2,604 |
| Prior T2048 attention replay, boundary starting point | 38,551.76 | 6,799.8 | 9,147.8 | 2,097,152 / 2,604 |
| Fused BFP8 QKV decode + Q/K BF16 panel publication | 38,156.02 | 6,870.3 | 9,147.8 | 2,097,152 / 2,604 |
| Above + ReLU backward panel/bias fusion | 37,988.02 | 6,900.7 | 9,051.8 | 2,097,152 / 2,604 |
| Above + post-bias BF16 logits rounding in GEMM epilogue | 37,943.91 | 6,908.7 | 9,051.8 | 2,097,152 / 2,604 |
| Above + dual BF16 dLogits panel pack for loss head | **37,914.20** | **6,914.1** | **9,063.0** | **2,097,152 / 2,604** |

From the boundary starting point, the selected paths save **637.56 ms/update (1.65%)** and add **114.3 tokens/s (1.68%)**. Relative to the original audit baseline, including the previous attention change, the reduction is 1,074.90 ms/update (2.76%). This is not a 2.5×/CUDA-parity result.

The QKV fusion removes a decode/pack pass, **not** the 192 MiB FP32 decoded QKV buffer: V, dP, dQ, and dK/dV still consume it. ReLU fusion avoids the 96 MiB row-major BF16 intermediate while preserving gate, BF16 rounding, and bias reduction order. The dual pack affects loss-head dLogits only; its normal and transposed panels retain their original layouts and FP32 dLogits remains resident. The logits fusion moves the required BF16 rounding after bias into the final GEMM store; it does not eliminate that rounding.

Final peak native ownership is **84.8 MiB below** the audit/boundary starting point. Native allocations are 2,428/update, versus 2,484/update at the starting point; this is native ownership accounting, not Task Manager's dedicated VRAM figure. H2D is only token+target input for the eight microbatches; no weight/activation/gradient D2H was introduced.

The profile supports the wall-time change without adding unrelated kernel groups: the old path spent about 468 ms/update in the two Q/K pack kernels; the new combined decode+pack kernels take about 622 ms/update but replace the expensive QKV share of `decode_bfp8`, leaving only about 15 ms/update of that generic decode. The original ReLU encode/normal-pack/transpose-pack/bias group cost about 935 ms/update; merged panel+bias and transpose packs cost about 774 ms/update. The standalone loss-logit `round_bf16_values` group (about 92 ms/update) disappears. The dual loss-gradient pack costs about 77 ms/update in place of about 116 ms/update of the two removed panel-pack portions. These GPU event sums explain direction, not an exclusive decomposition of update wall time.

## Rejected or retained opt-in

| Candidate | Controlled result | Decision |
| --- | --- | --- |
| Simple ReLU gate-to-panel fusion without bias fusion | 38,565.62 → 38,621.53 ms/update (1 warmup + 2 measured) | Slower because bias independently re-decodes/rounds; retain OFF. |
| Residual/LayerNorm fusion with 8 KiB SLM row staging | QKV-only 38,150.53 → QKV+norm 38,350.90 ms/update (both 1+2) | Peak memory falls about 64 MiB, but row staging/barrier and parameter partial work outweigh it; retain OFF. |
| 6 GiB vs 4 GiB Arc buffer pool | 38,551.76 → 38,448.34 ms/update using different 2+4 vs 1+2 samples | Native allocation count falls substantially, but the ~0.27% speed delta is not established on a matched final run; keep the 4 GiB default. |

Candidate switches remain available in `ArcExecutionOptions` and the `--probe-arc-transformer` harness for further A/B without changing the production defaults. The enabled paths are guarded by their applicable packed/XMX/shape contracts; unsupported cases keep their established fallback.

## Verification

- Release solution build: zero warnings/errors.
- Focused GPU tests: QKV 3/3, ReLU old/new 25/25, residual norm 5/5, dual gradient 5/5, logits rounding 5/5. They include bitwise panel/output comparisons, FP32/BF16/BFP8 cases as applicable, negative zero/NaN/Inf, tail dimensions, and accumulated gradients.
- The original ReLU kernel-name assertion was pinned to its legacy path. Its numerical comparison remains unchanged; the separate fused-path tests verify the new dispatch and bitwise behavior.
- Final Release solution test excluding the six already-known `ArcAttentionExperimentalAcceptance` tests: **2,176 passed, 115 skipped, zero failed** (Core 1,703/109, Integration 427/6, Benchmarks 46/0). Four of those six small-shape FlashAttention experimental acceptance cases fail independently of these T2048/packed-boundary changes; they were not used as a success signal.

Artifacts: `benchmark-results/arc-attn-final-t2048-20260923.json`, `arc-boundary-qkv-final-t2048-20260923.json`, `arc-boundary-qkv-relu-final-t2048-20260923.json`, `arc-boundary-qkv-relu-round-final-t2048-20260923.json`, `arc-boundary-qkv-relu-round-dual-t2048-20260923.json`; all paths are relative to `benchmark-results/` except the first prefix shown for clarity.
