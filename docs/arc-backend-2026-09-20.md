# Intel Arc B580 backend — implementation and measurement record

## Scope and numeric contract

- Windows x64 / .NET 10 / Intel Arc OpenCL, independent `TensorDevice.Arc` and `ExecutionDeviceKind.Arc`.
- Transformer only; single Arc. `float32`, `mix16_32`, `mix8_32`; Muon, NekoMuon and AdamW.
- The default route now keeps packed storage, FP32 gradients/masters and optimizer state resident on Arc. Host mirrors are refreshed on explicit inspection, checkpoint and session closure. OpenCL FP32 arithmetic with BF16 matrix-operand rounding for the mixed contracts; this does not claim XMX or native INT8 matrix instructions.
- Explicit `ExecutionSession` lifetime owns OpenCL context, queue, compiled kernels and buffers. A bounded 256 MiB per-session pool replaces per-operation native allocation. Unsupported operators fail instead of choosing CPU arithmetic.
- Public CPU/CUDA routes remain available. No changes to tokenizer, dataset, checkpoint filenames/formats, objective, learning rates, effective batch or model architecture are required by the backend.
- Current `training.transformer.json` selects Arc 0. Batch was 32 when initial measurements began, and was externally changed to 16 during work; that change was preserved. Each probe records the effective shape and configuration SHA256.

This document records the September 20 implementation and measurements. The
subsequent default route is described in [September 21 tuning](arc-tuning-2026-09-21.md).
Use `--arc-mode resident` to reproduce the residency-only route, not today's
`optimized` default. Benchmark JSON files retain the actual options/binary hashes.

## Adopted changes (September 20 snapshot)

1. Portable 32×32-output tiled GEMM (16×16 workgroup, 2×2 registers per work item, local staging). Transpose/tails, bias, ReLU gate and ordered split-K supported. NS uses the same matrix kernel.
2. Muon moments, confidence reduction, normalization, all NS iterations and weight update share device buffers. Moments and masters now remain resident between steps as well. Confidence uses block reduction instead of one serial work item.
3. Session-local allocation reuse; device/stream ownership is not static. Exceptions finish the queue before recycling buffers; session disposal releases the entire pool.
4. Attention keeps only O(BHT) row statistics, reconstructs probabilities in workgroup-local memory, and uses query/key ownership in backward. No global O(BHT²) P/dS buffers or float-atomic DKV. Backward uses unquantized softmax statistics, not quantized output values.
5. Loss head processes 128 rows at a time and recomputes logits in backward. It avoids full N×vocabulary logits/gradients; mixed modes still round head logits to BF16. Row-parallel softmax/cross-entropy reduction.
6. Packed BFP8/BF16 uploads and device decoding for matrix operands, reducing expanded FP32 staging. FP32 gradients and optimizer statistics are not quantized.
7. LayerNorm residual/dropout intermediates remain device-local within the operator; backward reconstructs the same unquantized input using the saved dropout seed. No saved full FP32 residual array per layer.
8. Intermediate gradients are allocated lazily during Arc backward and released after their last use by `BackwardAndRelease`, instead of reserving every FP32 gradient during forward.
9. Device-authoritative packed activations, weights, FP32 gradients/master and Muon/AdamW state. Forward, backward, accumulation and clipping do not read these arrays back. Token/target indices are uploaded once per microbatch and reused in backward. Saved attention/LayerNorm statistics stay on-device.
10. Device BFP8/BF16 publication replaces host quantization round trips. OpenCL correctly-rounded divide/sqrt is enabled: without it, a quantization-boundary gradient regression was detected, and fixed without increasing test tolerances. BFP8 invalid-value status is accumulated on device and checked at scalar loss/clip boundaries.
11. Graph release discards intermediates without readback; inference frames release per-token temporary buffers. Host mutation, precision/device changes, state serialization and session closure explicitly transfer authority. Persistent buffers remain allocated by design; tests check stable retained bytes and zero bytes after lane disposal rather than incorrectly requiring zero VRAM after every step.

## Measurement method

`--probe-arc-transformer` builds the model/optimizer through the production factories with synthetic seeded tokens. No corpus/tokenizer/checkpoint/metrics are modified. Fixed LR, no scheduler or sampling. p50 is over measured updates after the stated warmup. Each JSON records the seed, precision, optimizer, shape, phases, byte counters and finite-gradient checks. Later reports additionally record binary hashes and all Arc options.

GPU event time is device execution time. Transfer wall time can include the wait for preceding GPU work; allocation wall time includes first upload on an allocation miss. They must **not** be added to phase wall times. Amdahl shares use only the disjoint zero/forward/backward/clip/optimizer wall phases.

## Iteration log

All runs below use width 512, heads 16, hidden 1536, vocabulary 11500, dropout .1, mix8_32 block32, Muon NS5 plus AdamW, accumulation 1. Short runs establish direction, not a convergence claim.

| Shape | Revision | Warmup / measured | p50 ms/update | Artifact |
|---|---|---:|---:|---|
| B1/T128/L2 | initial reference | 1 / 3 | 1838.06 | `benchmark-results/arc-baseline-20260920.json` |
| B1/T128/L2 | pooled allocation + resident NS + confidence reduction | 1 / 3 | 704.07 | `arc-muon-pool-20260920.json` |
| B1/T128/L2 | tiled linear + streaming attention | 1 / 3 | 677.00 | `arc-tiled-streaming-20260920.json` |
| B1/T128/L2 | whole Muon parameter update device-local | 1 / 5 | 584.49 | `arc-muon-fused-20260920.json` |
| B4/T512/L2 | reference | 1 / 3 | 3810.29 | `arc-reference-b4-s512-20260920.json` |
| B4/T512/L2 | tiled + streaming + whole Muon | 1 / 3 | 1687.77 | `arc-optimized-b4-s512-20260920.json` |
| B4/T512/L2 | chunked loss head | 1 / 3 | 1412.65 | `arc-chunked-head-b4-s512-20260920.json` |

The initial B1 optimizer share was 73.2%; making that phase 2× faster permits 1.577× overall by Amdahl. After removing those costs the B4 backward phase became the largest (about 47%), motivating attention/loss/normalization memory changes.

Later wall times shifted substantially even for unchanged optimizer kernels. The adjacent packed/non-packed runs were 2674.48 / 2919.23 ms (`arc-packed-reduction-b4-s512-20260920.json`, `arc-reduction-no-packed-b4-s512-20260920.json`); do not compare these absolute times to earlier runs as a controlled claim. The final same-window comparison and full-shape checks are recorded separately below.

## Final resident-versus-staged A/B

The same compiled binaries were used for both runs below. Intel Arc B580, driver 32.0.101.9030; B4/T512/L2, width512/heads16/hidden1536/vocab11500, dropout .1, tied embedding, mix8_32 block32, Muon NS5 LR .001 + AdamW LR .0003, weight decay .01, accumulation1. Fresh seed1234 and identical synthetic tokens; 3 warmup + 10 measured updates. `staged` retains all previous tiled/streaming/chunked optimizations but disables residency.

| Metric | Prior optimized `staged` | Device-resident default |
|---|---:|---:|
| p50 ms/update | 776.9777 | 189.20905 |
| tokens/s | 2635.9 | 10824.0 |
| mean forward ms | 189.77 | 34.79 |
| mean backward ms | 328.48 | 108.41 |
| mean clip ms | 59.44 | 3.82 |
| mean optimizer ms | 197.34 | 41.82 |
| H2D bytes/update | 706,159,316 | **16,384** |
| D2H bytes/update | 479,186,728 | **176** |
| peak backend native allocation MiB | 179.05 | 296.59 |

Speedup is **4.106×**, step time down **75.65%**. All ten serialized loss and gradient-norm values match exactly between this pair; this is not a convergence guarantee. Retaining data increases device memory in exchange for removing PCIe traffic and managed allocations.

The 16,384 H2D bytes are exactly 2,048 input IDs plus 2,048 targets, four bytes each. The 176 D2H bytes are loss4 + clipping/numeric status12 + eight matrices × (confidence16 + NS norm4). **No weight, activation, gradient, master or optimizer-moment arrays cross PCIe during these steady-state steps.** Generation may explicitly download the final logits for CPU sampling; checkpoint/state inspection/session teardown have necessary readbacks and are not counted as training steps.

Artifacts:

- `benchmark-results/arc-residency-final-staged-b4-s512-20260920.json`
- `benchmark-results/arc-residency-final-device-b4-s512-20260920.json`

For reproduction, use the README probe command with a new output path and `--arc-mode staged` or default `optimized`. Binary/config hashes and options are embedded in each result. No real dataset/tokenizer/checkpoint was modified by a probe.

## Full JSON shape and Amdahl result

Current shape: B16/T1024/L32, accumulation4 (effective batch64), width512, heads16, hidden1536, vocab11500, 90,507,500 parameters, same precision/optimizers above. Synthetic full-length batches; no shape override.

`benchmark-results/arc-residency-full-20260920.json`: 1 warmup + 3 measured updates, before the final scalar invalid-value guard:

- p50 **72,691.16 ms/update**, **901.6 tokens/s**.
- Every measured update: H2D **524,288 bytes** (exactly token+target IDs), D2H **2,584 bytes** (the final guard adds 4 bytes).
- Native allocated peak **5,134.812 MiB** including the pool; post-step retained **1,478.139 MiB**. Both unchanged over these three updates. These are backend counters, **not** driver-reported total dedicated VRAM.
- Managed live bytes approximately **1,139–1,142 MiB**; no OOM, finite losses/norms, loss 9.312821 → 9.241086 → 9.171274.
- Average backward is approximately **56.44 s (77.7%)**. Amdahl: halving backward gives at most **1.635×** overall. Optimizer is only about **0.65 s (<1%)**; optimizing it further is not the highest-impact next step.
- Device kernel event durations for the first measured update: attention dQ20.57s, dKV17.14s, tiled GEMM15.39s, attention forward9.12s, bias reduction3.72s, norm parameter reduction1.50s. These identify GPU attention/backward tiling/coalescing and bias/normalization reduction as the next work, not host transfers.

The earlier full-shape host-staged smoke was 303.98s, but it was a single cold update from an earlier binary/time window. It is **not** the controlled speedup claim; only the paired smaller-shape A/B above is used for that claim.

## Validation

- Full solution Release build: **0 warnings / 0 errors**.
- Related Core tests: **95 passed**, Integration tests: **138 passed**, no skipped tests in those runs. GPU availability is checked explicitly; real Arc executed here.
- Forward/gradient comparisons across float32, mix16_32 and mix8_32; tail shapes, dropout, tied weights and ignored targets. AdamW/Muon/NekoMuon numerical tests, state round trip and fixed/fractional NS depths.
- Exact steady-state transfer-budget tests for all three precisions, accumulation, stable retained memory, host mutation/zeroing/device and precision transitions, failed-backward cleanup, non-finite quantization rejection, session disposal.
- Temporary FineWeb fixture through actual CLI training, checkpoint/auto-resume, HTML/metrics continuation and generation. Real user data/checkpoints were not used or overwritten.
- No claim of a 210/2100-step soak, full repository/CUDA test completion or long-run convergence. Tests and bounded benchmarks above are the verified scope.

## Known remaining work

Default Transformer training no longer stages tensor arrays through the host between operators/steps. Remaining performance work is GPU attention memory coalescing/tiling, bias and norm parameter reductions, asynchronous command batching and XMX/library GEMM. Packed values are decoded into bounded FP32 **device** scratch for current kernels; that does not involve PCIe, but still costs device bandwidth. Muon adaptive-control statistics are currently read back per matrix (20 bytes), not batched into one transfer. Neither native BF16/INT8 matrix throughput nor long-run convergence equivalence has been claimed. Tiny and tail-shape numeric tests, checkpoint/resume and bounded performance probes cannot substitute for a long training soak.
