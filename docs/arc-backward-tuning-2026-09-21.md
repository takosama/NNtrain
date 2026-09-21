# Arc backward tuning: causal bounds, shape routing, parallel dW

This continues [the first XMX tuning pass](arc-tuning-2026-09-21.md).
No precision, optimizer, model, batch, tokenizer, dataset or checkpoint setting
was changed. The normal CLI selects the improvements by default.

## Implemented changes

- Causal attention skips GEMM tiles strictly above the diagonal. PV and dQ
  reduce only through the largest query in their output tile; dK/dV begin at
  the smallest key in their tile. Softmax and derivative kernels initialize
  only the diagonal region that downstream tiles can read. The noncausal path
  still computes the complete matrix. No host decision or tensor transfer.
- FP32 64x64x32 GEMM uses half as many local-memory synchronization points as
  64x64x16. Numerical operation order is preserved. Microbenchmarks showed
  regressions for some short-output/long-reduction shapes, so the deeper tile
  is selected only for m>=1024 and (non-transposed A or k<=2048).
- Long weight-gradient reductions are split into independent 2048-element K
  ranges in one 3D dispatch. A second GPU kernel adds the partials in fixed
  order, without floating-point atomics. FP32 input, output and accumulation
  are retained. Scratch is capped at 64 MiB; incompatible/larger shapes keep
  the old sequential path. This changes reduction association, not precision.

The split path is restricted to transposed-A, non-transposed-B accumulating
gradient shapes with no bias/ReLU epilogue, k>=4096, and m,n>=64. ReLU operand
gates remain supported. Existing tails and accumulation behavior are retained.

## Matrix experiments

FP32 operands; 3 warmup + 10 measurements; dispatch plus synchronization wall
time, excluding allocations/uploads. Intel Arc B580, driver32.0.101.9030.

| M x N x K | Previous tile ms | Selected path ms |
|---|---:|---:|
| 16384 x 512 x 1536, dX | 7.1104 | 5.9992 (deeper tile) |
| 16384 x 1536 x 512, dX | 6.6565 | 5.8829 (deeper tile) |
| 1536 x 512 x 16384, dW | 9.1543 | 7.2583 (parallel K) |
| 512 x 1536 x 16384, dW | 9.0968 | 7.3069 (parallel K) |
| 512 x 512 x 16384, dW | 3.0008 | 2.6017 (parallel K) |
| 11500 x 512 x 512, loss-head dW | 1.7536 | 1.4933 (deeper tile) |

Artifact: `benchmark-results/arc-backward-split-v1-20260921.json`.
The initial all-shapes deeper-tile experiment is recorded separately in
`arc-backward-deep-v1-20260921.json`; its slower routes were not adopted.

### Rejected packed-read experiment

Direct BF16/BFP8 reads inside backward GEMM removed expanded FP32 operand
scratch, but decoding overhead offset the traffic saving. B16/T1024/L2,
accumulation1, mix8_32: 513.28 ms for unpacked versus 533.50 ms for packed.
Replacing power-of-two scale-index division with a shift recovered only to
513.66 ms. No speed benefit was demonstrated, so the experimental packed-GEMM
code was removed. Its result JSONs remain as evidence:
`arc-round2-split-b16-l2-20260921.json`, `arc-packed-matrix-b16-l2-20260921.json`,
`arc-packed-matrix-shift-b16-l2-20260921.json`.

## Full-shape results

| Route | Warmup / measured updates | Median ms/update | Tokens/s |
|---|---:|---:|---:|
| Previous tuned path, all three changes disabled | 1 / 3 | 20,769.49 | 3,155.40 |
| New default | 3 / 10 | 18,109.11 | 3,618.95 |

Time per optimizer update decreased **12.8%**, or **1.147x throughput**.
Each update includes four microbatches. Both runs have identical configuration
and binary hashes in their result JSONs. Warmup/sample counts differ as shown;
they are not a same-update numerical comparison. Numerical equivalence is
checked separately by the regression tests below.

The new ten measured updates range from 18,076.31 to 18,137.67 ms. Loss is
finite throughout (9.171305 to 8.560438); this short synthetic run does not
demonstrate real-corpus convergence.

| Mean wall-time phase | ms/update | Share |
|---|---:|---:|
| Backward | 13,479.21 | 74.43% |
| Forward | 4,222.26 | 23.32% |
| Optimizer | 381.95 | 2.11% |
| Clip | 19.59 | 0.11% |
| Zero grad | 6.19 | 0.03% |

Backward remains the next optimization priority: doubling it alone would
improve total throughput by approximately 1.593x; doubling the optimizer
alone would improve total throughput by only 1.011x.

All ten updates transfer exactly **524,288 H2D bytes** (tokens/targets) and
**2,588 D2H bytes** (scalar metrics and optimizer statistics), unchanged from
the previous path. No new weight/activation/gradient host transfer was added.
The benchmark's explicit finite-gradient audit is outside these step deltas.

Peak backend allocation increases by only **24 MiB**, from 5,138.63 to
**5,162.63 MiB**, due to the parallel weight-gradient workspace. End-of-update
retained allocation stays at **1,478.14 MiB** in every measured update;
neither peak nor retained allocation grows during these ten updates. These
are allocator counters, **not driver-reported dedicated VRAM**. A long-run
memory soak and a current CUDA comparison were not performed.

Artifacts:

- `benchmark-results/arc-round2-previous-full-20260921.json`
- `benchmark-results/arc-round2-final-full-20260921.json`

## Validation

- Release solution build: **0 warnings, 0 errors**.
- Core targeted suite: **121 passed, 0 failed, 0 skipped**. Includes Arc
  kernels/residency, NekoMuon numerical/state tests, CPU AdamW storage
  transitions, and Tensor characterization.
- Integration targeted suite: **138 passed, 0 failed, 0 skipped**. Includes
  Arc training, Wiki CLI, dtype/checkpoint and profile/configuration tests.
- New tests compare causal-bounded attention against the previous path
  exactly for non-tile-aligned sequence/head dimensions; compare deeper FP32
  GEMM exactly with the old route; and compare three accumulated split-K
  updates (including ReLU gates and tails) within 1e-5 with no extra H2D/D2H.
  Existing test tolerances were not widened.

This is **259 targeted tests**, not a claim that the entire solution test
suite or unavailable CUDA hardware was validated. A final rebuild and test
run also cover the benchmark-only historical-mode switch/help updates made
after the full-shape measurements; the default optimized math is unchanged.

## Reproduction

The full-shape probe uses the unchanged `training.transformer.json`: Arc0,
B16/T1024/L32, width512/heads16/hidden1536/vocab11500, accumulation4 (effective
batch64), 90,507,500 parameters, mix8_32/block32, tied embedding, dropout0.1,
Muon NS5 LR0.001 + AdamW LR0.0003, weight decay0.01, seed1234.
Tokens are synthetic and fixed, LR is fixed, and corpus/checkpoint/sampling
I/O is excluded. These measurements do not establish convergence or CUDA parity.

```powershell
dotnet run --project .\NNtrain.Benchmarks -c Release --no-build -- `
  --probe-arc-transformer .\training.transformer.json .\benchmark-results\arc-new.json `
  --warmup 3 --steps 10
```

Use `--arc-mode previous` for the prior tuned path (all three new changes off).
Result paths must not already exist. For the matrix benchmark, use
`--probe-arc-backward NEW.json`.
