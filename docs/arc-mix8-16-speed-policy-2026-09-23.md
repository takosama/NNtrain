# Arc `mix8_16` speed policy and measured candidates (2026-09-23)

## Contract and selection rule

The requested `mix8_16` policy prefers 8-bit weight arithmetic. A 16-bit
weight route is permitted when it is faster on the target shape. Speed of
non-weight work takes priority over preserving a particular arithmetic order.
Weights and fused Linear outputs use block-scaled BFP8 storage; the selected
Arc route publishes eligible non-weight activations as physical BF16 because
that reduced total update time. Gradients, master weights, and optimizer moments are
retained as BF16 values. FP32/INT32 temporary accumulation remains permitted.
The policy is expressed in `PrecisionPolicy.Mix8_16`; Arc implementation
choices are guarded by `ArcExecutionOptions` and measured with the synthetic
Transformer probe. BFP8 storage is signed INT8 payload plus one FP32 scale per
block, rather than native FP8 floating-point arithmetic.

Except for the explicitly marked batch comparison, full-shape results use the current `training.transformer.json` shape:
two Arc B580 GPUs, batch 16, accumulation 8, sequence 2048, 32 layers, width
512, 16 heads of width 32, and 262,144 fixed synthetic tokens per update.
The probe excludes corpus, tokenizer, and checkpoint I/O. Each comparison had
one warmup and two measured updates unless marked final; the final comparison
uses four measured updates per route.

## Native INT8 weight projection

The opt-in `--mix8-16-int8-linear on` candidate packs block32 BFP8 input and
weight bytes directly for SG16 `intel_sub_group_i8_i8_matrix_mad_k32`. It
applies the two block scales to each K32 INT32 partial dot before FP32
accumulation, then publishes bias/ReLU output as block32 BFP8. It covers the
forward Linear only; dX and dW continue through the BF16 operand route. The
candidate is restricted to compatible aligned shapes and has its own OpenCL
compile guard. The [INT8 implementation note](arc-int8-bfp8-linear-2026-09-23.md)
records the panel format, guards, and focused tests.

| Same binary/config route | Median ms/update | Tokens/s | GPU 0 backend peak |
| --- | ---: | ---: | ---: |
| BF16 XMX, INT8 off | 18,017.92 | 14,549.1 | 4,555.9 MiB |
| Native INT8 XMX, INT8 on | 18,532.11 | 14,145.4 | 4,568.6 MiB |

The INT8 candidate was 2.85% slower on this full shape, so the default is
BF16. Focused SG16 tests passed 4/4: output relative RMS against BF16 XMX was
0.13–0.15%; dX/dW were equal without ReLU, and ReLU-case dX/dW differed by
0.23%/1.18% relative RMS. The final full-shape losses were 9.313025 (BF16)
and 9.313011 (INT8), with finite gradients. The two probe artifacts have
matching Core, Arc, benchmark binary hashes and configuration hash:
[off](../benchmark-results/arc-mix8-16-int8-linear-off-20260923.json),
[on](../benchmark-results/arc-mix8-16-int8-linear-on-20260923.json).

## Row-delta attention derivative fusion

For causal T2048/D32 training, the selected row-delta route saves the **raw
pre-publication** attention output as BF16 during forward. In backward it forms
one FP32 row value `delta = dY · O_raw_BF16`. The identity
`sum_key(P * dP) = dY · O` then allows
`dS = P * (dP - delta) / sqrt(32)` without a separate T-wide derivative
reduction per query. This saves a BF16 activation only when autograd records
the forward; the no-grad focused test checks that inference does not retain it.
The fused FP32 tile computes `dP = dY × Vᵀ` and writes the paired BF16 `P/dS`
into the existing score workspace, avoiding a separate FP32 dP/dS matrix and
derivative launch. The optional XMX variant rounds dY to BF16 for its two K16
products and retains the same packed publication boundary.

The focused test checks bitwise equal forward output, accumulated-gradient
relative RMS below 0.2% for row-delta versus the previous derivative path, and
bitwise equality between the separate packed row-delta result and fused FP32
`dP/dS` result. It bounds the XMX fused `dS` difference against fused FP32 at
3% relative RMS. In the recorded GPU tests, accumulated-gradient relative RMS
was 0.00849% after one backward and 0.01138% after two, with exactly equal
forward outputs. Live allocation after each released backward was the same
1,228,804 bytes for the previous and fused routes with BFP8 publication, or
1,228,800 bytes with BF16 publication. Both publication modes had the same
measured gradient error. XMX fused dS relative RMS
against the FP32 fused tile was 0.11145%.

| Same binary/config row-delta route | Median ms/update | Tokens/s | Final loss | Final gradient norm |
| --- | ---: | ---: | ---: | ---: |
| Fused FP32 `dP/dS` control | 15,842.04 | 16,547.4 | 9.313040 | 0.073816404 |
| Fused XMX `dP/dS` | 15,956.98 | 16,428.2 | 9.313037 | 0.073879350 |

Only `Mix8_16FusedAttentionDpDsXmx` changed between these two artifacts; the
configuration and Core/Arc/benchmark binary hashes match. XMX is 0.72%
slower in this short full-shape comparison, so the fused FP32 implementation
remains the measured choice. Both runs retained finite gradients and reported
5,065.9 MiB GPU 0 peak backend allocation. These two runs already have
row-delta and fusion enabled, so their throughput does **not** measure the gain
from enabling row-delta itself. Artifacts:
[FP32 control](../benchmark-results/arc-mix8-16-rowdelta-current-control-20260923.json),
[XMX](../benchmark-results/arc-mix8-16-rowdelta-fused-xmx-20260923.json).

## Other XMX attention results

An earlier same-binary full-shape comparison of the opt-in BF16 XMX attention
products measured **14,550.0 → 8,648.4 tok/s** (18,016.73 → 30,311.24
ms/update). Its candidate was markedly slower, so that option remains off.
Those artifacts describe the implementation at their recorded binary hashes;
subsequent packing experiments are separate candidates:
[control](../benchmark-results/arc-mix8-16-speed-contract-control-20260923.json),
[XMX](../benchmark-results/arc-mix8-16-speed-contract-xmx-20260923.json).

The coalesced transpose-pack and SLM variants remain experimental. Their
one-layer diagnostic runs were started before the previous process had exited,
so they are not used to select a default or establish a performance claim.

## Activation publication and batch shape

With the FP32 fused row-delta route, publishing eligible non-weight outputs as
BF16 measured 16,606.4 tok/s (15,785.72 ms/update), versus 16,547.4 tok/s with
the same binary and BFP8 publication. The pilot gain was only 0.36%; backend
peak allocation increased by 661 MiB per GPU, to 5,726.9/5,194.7 MiB.
The fused Linear epilogues still publish BFP8. Artifact:
[BF16 activation pilot](../benchmark-results/arc-mix8-16-rowdelta-bf16-activations-20260923.json).

The four-update repeat measured 16,613.5 tok/s with BF16 publication and
16,570.1 tok/s without it (+0.26%). Both comparisons had the same direction;
the gain is small. In keeping with the requested speed priority, BF16
publication is enabled by default along with row-delta and FP32 dP/dS fusion.
The default INT8 Linear, XMX dP/dS, and whole-attention XMX switches stay off.

Batch 32 with accumulation 4 preserves 262,144 tokens per update. With
row-delta and FP32 fusion it measured 15,518.2 tok/s (16,892.64 ms/update),
with 7,618.2/7,075.1 MiB peak backend allocation. It is slower than batch 16
with accumulation 8, so the production batch configuration is unchanged.
Artifact: [batch 32](../benchmark-results/arc-mix8-16-rowdelta-b32-a4-20260923.json).

## Final same-binary comparison

All three runs below used one warmup plus four measured updates, the same
configuration hash, and identical Core/Arc/benchmark binary hashes. GPU runs
were serialized. The measured BF16-publication option was subsequently made
the default; the selected runtime options match that measurement.

| Route | Median ms/update | Tokens/s | GPU 0 / 1 backend peak MiB |
| --- | ---: | ---: | ---: |
| Previous defaults, row-delta/BF16-publication off | 18,010.35 | 14,555.2 | 4,555.9 / 4,021.7 |
| Row-delta + FP32 fused dP/dS | 15,820.34 | 16,570.1 | 5,065.9 / 4,533.7 |
| Selected: above + BF16 activation publication | 15,779.02 | 16,613.5 | 5,726.9 / 5,194.7 |

The combined throughput gain is **14.14%**, with unchanged 262,144 tokens per
optimizer update. Final losses were 9.241410 / 9.241384 / 9.241210 and all
gradient scans were finite. The selected planner estimates 4,689,494,016 bytes
of saved activations within its 7,471,504,588-byte budget, with no activation
recomputation. These are backend allocation counters, not driver-wide VRAM.

Artifacts: [baseline](../benchmark-results/arc-mix8-16-speed-policy-final-baseline-20260923.json),
[row-delta](../benchmark-results/arc-mix8-16-speed-policy-final-default-20260923.json),
[selected BF16 activations](../benchmark-results/arc-mix8-16-speed-policy-final-bf16-activations-20260923.json).

Reproduce with a new output path:

```powershell
dotnet run -c Release --project NNtrain.Benchmarks -- --probe-arc-transformer training.transformer.json benchmark-results/new-default-run.json --warmup 1 --steps 4
```

For the previous behavior, append `--mix8-16-row-delta off --mix8-16-fused-dpds off --mix8-16-bf16-activations off`.

## Verification

The final Release solution build succeeded with zero warnings/errors. With
BF16 activation publication enabled by default, the focused Core suite passed
74/74 tests, covering the numeric policy, all new attention and INT8/BF16
activation candidates, fused attention with either activation publication
format, two-GPU reduction/training, and memory planning. The two BF16
checkpoint/master/optimizer save/resume integration cases also passed.
No tests were skipped.

Artifacts: [selected-default Core suite](../benchmark-results/test-results/arc-mix8-16-speed-policy-selected-core-20260923.trx),
[selected-default checkpoint integration](../benchmark-results/test-results/arc-mix8-16-speed-policy-selected-checkpoint-20260923.trx).
These checks and short synthetic runs do not measure long-run model convergence.
