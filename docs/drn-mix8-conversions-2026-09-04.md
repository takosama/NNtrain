# DRN mix8_32: redundant conversion audit (2026-09-04)

## Implemented

1. **DRN training loss head:** consume GEMM's BF16 logits directly in cross entropy instead of encoding vocabulary-sized logits to BFP8 and decoding them again. Reuse the existing BF16 loss-head backward path, including its BF16 gradient buffer, rather than materializing a full FP32 logits gradient and converting it back. Parameters remain block-BFP8; FP32 parameter gradients, master/update state and optimizer statistics are unchanged. Ordinary `Forward` still returns BFP8 logits. Other model architectures, CPU and other precision modes retain their existing paths.
2. **LayerNorm affine parameters:** reuse the existing versioned BF16 leaf cache for gamma/beta instead of temporary decode buffers in forward and another gamma decode in backward. Parameter publication invalidates the decode; graph capture records its refresh. Non-leaf operands still use temporary leases. Existing direct fused BFP8 LayerNorm kernels are unchanged.

The loss-head change removes an extra quantization error; it intentionally does not reproduce the old BFP8-logits training trace bit-for-bit. LayerNorm cached/uncached outputs and gradients are bit-identical in the regression tests. This is not a claim of improved convergence on the real corpus.

## Conversions deliberately retained

- BFP8 to BF16 GEMM operands: required by the current block-scaled Tensor Core implementation. Updated weight decodes use the generation-aware cache.
- BF16 to BFP8 persistent activations/weights: these implement the requested storage precision, not a gratuitous round trip.
- DRN recurrent state and reductions, parameter gradients and optimizer statistics: FP32 is retained for numerical stability. The recurrence's decoded projections are retained/reused for backward/rematerialization.
- Some backward activation decodes: caching all expanded activations would trade away the VRAM benefit. Existing direct residual/dropout and private FFN BF16-gradient paths already avoid several full-buffer conversions.

This audit removes two verified redundancies; it does not claim that every remaining kernel cast has been eliminated.

## A/B measurement

The user's production training was still running on both GPUs. Only a small GPU0 workload was run alongside it; production-size, two-GPU throughput and convergence tests were not attempted. The production JSON was not changed by this conversion work (it was observed to have been changed by the user to mix16_32/Muon).

Conditions: RTX 3070 Ti, Release .NET 10, native ABI 1.32; DRN mix8_32, block 128, batch 1, sequence 128, vocabulary 11,500, width 128, hidden 384, 2 layers, K/V 16/16, dropout 0.1, seed 1234. NekoMuon fixed NS5 every step, LR 0.003, AdamW LR 0.001, weight decay 0.01, clip 1. Synthetic fixed tokens; no checkpoint/corpus I/O or LR schedule. Each fresh process: 10 warmup + 30 measured steps; order baseline/candidate/candidate/baseline. Additional synchronized diagnostic steps and one eager F/B pass are excluded from throughput results.

Both variants use the same candidate managed binary and native DLL. Baseline disables the two new paths; candidate enables them. Native SHA256: `A61C5BD17581DFB36BFE07CB143EC9ACB8189608FBA849CB531FFE91D273F22D` (native code not changed for this task).

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Run 1 p50 / mean, ms | 19.6188 / 18.0400 | 20.2458 / 17.9629 |
| Run 2 p50 / mean, ms | 20.6818 / 17.6880 | 19.7444 / 17.9701 |
| Pooled 60-step p50 / mean, ms | 20.1828 / 17.8640 | 19.7444 / 17.9665 |
| Allocator-owned memory after measurement, MiB | 101.8477 | 94.7868 |
| Standalone BF16/BFP8 codec launches, eager F/B | 81 | 73 |
| Elements processed by those codecs, eager F/B | 6,947,340 | 2,530,700 |

**Verified:** counted conversion workload -63.57%; allocator-owned memory -7.06 MiB (-6.93%) for this small workload. Per measured step, both variants had zero native allocations/frees; each run had 30 graph replays, zero new captures/fallbacks. Transfers were token/target H2D (1,024 bytes) and scalar D2H (20 bytes) per step.

**Not established:** throughput improvement. The pooled p50 improved 2.17% but mean worsened 0.57%, with concurrent training and substantial scheduling variation. Treat speed as inconclusive/roughly unchanged, not a proven percentage speedup. The memory column is allocator ownership including graph-pinned buffers, not total process/physical VRAM. Codec telemetry counts explicit managed encode/decode calls in the separate eager pass, not register conversions or replayed graph instructions. The graph loss/gradient buffers also change precision/lifetime; therefore the allocator savings are not solely gamma caching.

Raw results and frozen config: `docs/benchmarks/drn-conversions-2026-09-04/{micro,baseline1,candidate1,candidate2,baseline2}.json`.

Reproduction (use a new output path, and run without competing training for reliable timings):

```powershell
$env:CUDA_MODULE_LOADING = 'LAZY'
# 1 = baseline; 0 = candidate. Set both together.
$env:NNTRAIN_DISABLE_DIRECT_DRN_MIX8_LOSS_HEAD = '0'
$env:NNTRAIN_DISABLE_BFP8_LAYERNORM_PARAMETER_CACHE = '0'
dotnet .\benchmark-results\drn-conversions-2026-09-04\candidate-bin\NNtrain.Benchmarks.dll `
  --profile-drn-json .\docs\benchmarks\drn-conversions-2026-09-04\micro.json `
  10 30 detail .\docs\benchmarks\drn-conversions-2026-09-04\candidate-next.json
Remove-Item Env:NNTRAIN_DISABLE_DIRECT_DRN_MIX8_LOSS_HEAD
Remove-Item Env:NNTRAIN_DISABLE_BFP8_LAYERNORM_PARAMETER_CACHE
```

## Validation

- Release builds: zero warnings/errors.
- CPU-only policy/model/generation regression selection: 23 passed (`drn-conversions-cpu-only.trx`).
- Final small CUDA selection: 28 passed, zero skipped/failed (`drn-conversions-final-small.trx`). This includes LayerNorm and decode-cache existing tests plus new regression coverage:
  - Direct DRN loss logits are BF16; weights and ordinary Forward remain BFP8; loss agrees with a host log-sum-exp reference and all parameter gradients are finite. Block 32/128 and vocabulary tails covered.
  - Cached/uncached LayerNorm values and all gradients exactly agree before and after changing the GPU parameter generation; widths 127/512 and block 32/128 covered.
  - Five optimizer updates through eager vs CUDA Graph execution: loss and parameter agreement within 1e-5, at least four replays and no fallback. Checks that updated LayerNorm coefficients do not become stale on replay.
- Full-suite, full-size two-GPU, real-data convergence and long-duration tests were not rerun during the user's training.

The new paths are enabled by default on the next build/run. No changes are injected into the already-running process.
