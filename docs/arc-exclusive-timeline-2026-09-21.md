# Arc B580: exclusive wall-time profile and measured changes (2026-09-21)

## Scope and frozen conditions

Windows x64, .NET 10 Release, Intel Arc B580, driver 32.0.101.9030, one Arc lane.
`training.transformer.json` was not changed (SHA256
`EC42E94F3A9BA43DBDB90E5A44368798C422F76AD5330BC936F701261E7D64C5`).
Batch 64, accumulation 4 (effective 256), sequence 1024, width 512, 16 heads,
head width 32, hidden 1536, 32 layers, vocabulary 11500, tied embeddings,
dropout 0.1, `mix8_32`, block 32, seed 1234. 90,507,500 parameters.
Muon fixed NS5 + AdamW, LR 0.001 / 0.0003, weight decay 0.01, clip norm 1.
Automatic memory plan: 11 checkpointed prefix blocks, FFN recomputation enabled.

All full-size results use one warmup update and four measured updates, unless
explicitly identified otherwise. Each update includes four forward/backward
microbatches, one clip, and one optimizer update. Tokens are synthetic,
fixed-seed and full length; the model is freshly initialized. These are not
FineWeb/resume throughput measurements. No corpus, checkpoint, tokenizer or
loss-history files are changed. The benchmark uses the CLI model/optimizer
factories and does not override the mathematical shape.

## What the timeline measures

`ArcTimeline` records host scopes and OpenCL event QUEUED/SUBMIT/START/END for
kernels and H2D/D2H/D2D transfers. Device/host clocks are correlated before and
after each update using `clGetDeviceAndHostTimer`; each endpoint selects the
tightest of nine host timestamp brackets. Raw calibration anchors are included
in the final trace. Typical endpoint half-bracket is 0.029 ms. The linear clock
mapping is an estimate within this uncertainty, not a claim of nanosecond
hardware-counter accuracy.

The exclusive partition gives GPU execution first priority. Only the complement
of GPU execution is assigned to the currently deepest host scope. Consequently
`Partition.WallCategories` adds up to 100% of measured wall time. A separate
`HostExclusive` view includes waits overlapping GPU work: **never add it to the
GPU columns**. Parallel host worker tracks are exported separately; the latest
active nested scope is used for wall-time attribution, not summed CPU time.
"GPU-idle" means no measured command is executing on this lane, not a
system-wide hardware-idle counter. "Idle with queued commands" is another view
of that same idle time, not an extra category.

Full phase fences, console progress and event queries are inside measured wall
time. Trace serialization and the finite-gradient validation scan are outside.
The final version also snapshots managed allocation/GC counts before export.
There is no full tensor readback added by the profiler. `CL_MEM_COPY_HOST_PTR`
is replaced by an explicit upload so a native allocation cannot hide a transfer.
The benchmark fails its timeline quality gate for missing events, opaque uploads,
non-unit coverage or overlapping events on this in-order queue.

Traces use gzipped Chrome/Perfetto JSON. Preserve the JSON results and matching
`.step-N.trace.json.gz` files together. They contain raw command timestamps,
operation labels, host tracks and exact per-update partitions.

## Original bottleneck

`benchmark-results/arc-wall-timeline-baseline-20260921.json`:
p50 **31,918.05 ms/update**, mean **31,914.30 ms/update**,
four-update total **127,657.22 ms**. Four samples: 31,915.53 / 31,900.47 /
31,920.66 / 31,920.56 ms. All four traces: 362,422 events, no missing events,
no opaque transfers, no GPU overlap, 100% coverage. There were approximately
0.02 ms/update of host scope-boundary ticks; the final recorder explicitly
labels these benchmark timestamp/bookkeeping boundaries.

Mean disjoint contributions (rounded):

| Contribution | ms/update | Wall share |
|---|---:|---:|
| Attention GPU compute | 11,703.78 | 36.67% |
| GEMM GPU compute | 8,901.85 | 27.89% |
| GPU pack/decode/publication | 5,908.33 | 18.51% |
| GPU normalization/reduction | 2,453.03 | 7.69% |
| GPU other kernels | 702.84 | 2.20% |
| GPU loss | 186.05 | 0.58% |
| GPU D2D copy | 83.57 | 0.26% |
| GPU H2D + D2H | 0.98 | 0.003% |
| GPU idle during event collection | 1,500.33 | 4.70% |
| GPU idle during queue submission | 266.30 | 0.83% |
| GPU idle during full queue wait | 129.67 | 0.41% |
| GPU idle during native allocation | 13.53 | 0.04% |
| Remaining host/allocator/free/bookkeeping idle | 64.04 | 0.20% |

The baseline JSON's initial category classifier treated some *packed compute*
as conversion. The table reclassifies `norm_packed_*` and `gradient_rows*` as
normalization/reduction, `linear_relu_grad_packed` as compute, and `round_bf16*`
as conversion, using the unchanged operation timings. The final recorder uses
this corrected classifier. Rounded table entries are not the exact sum; the
machine-readable partitions preserve all intervals.

The largest individual kernel was
`backward/attention_dkv_d32_k32_q32_causal`: approximately **3,695.88 ms/update**
(11.58%). Even an infinitely fast replacement of this one kernel could only
give approximately **1.13x** total speedup. Attention as a whole, GEMM and
format conversion must all be considered for a much larger speedup.

Native allocation was approximately 149 ms in the overlapping host view but
only 13.53 ms exposed as GPU idle. Cumulative requested/allocated bytes are not
simultaneously live VRAM and do not mean that weights are copied over PCIe on
every update. The oscillating memory graph by itself did not establish the
performance bottleneck.

## Adopted changes and A/B results

| Cumulative change | Full update p50 | Artifact suffix |
|---|---:|---|
| Original path, timeline | 31,918.05 ms | `baseline-20260921.json` |
| Pipeline event collection | 30,208.79 ms | `pipeline-20260921.json` |
| + power-of-two block address | 30,105.72 ms | `pow2-20260921.json` |
| + phase-local bulk Q/K panels, provisional | 29,698.46 ms | `bulk-20260921.json` |

Artifact prefix: `benchmark-results/arc-wall-timeline-`.
The provisional bulk run exposed four allocation-internal uploads/update.
It is not the final clean profile: the final run uses explicit uploads.

1. **Event collection:** previously every 128 launches ended with `clFinish`,
   draining GPU work before querying/releasing events. Now flush the queue,
   wait for the older 64 commands, and collect their events while the newer
   half can execute. Retired memory and copy-event owners still require the
   original full-completion fence. Exposed event-collection idle fell from
   1,500.33 to 138.41 ms/update. This also applies without timeline recording;
   the runtime already collected per-kernel timing events.
2. **BFP8 block indexing:** for power-of-two block sizes, replace exact integer
   division in packing with the equivalent shift. Other sizes keep division.
   No format, scales or RNE changes. 48 full-panel cases were bit-exact, including
   gates, offsets, exceptional scales and tails. Relevant pack microbenchmarks
   improved 10–20%, but the **full model improved only about 103 ms**.
3. **Bulk Q/K panels:** pack Q/K once for a forward or backward phase and use
   head offsets for its tiles. Keep the original BF16 operands and summation;
   release at the end of that phase, not a persistent cross-layer cache.
   Only the verified mixed-precision D32/SG16 path is eligible, with a 256 MiB
   temporary cap and the old path for other shapes/devices. Matched attention
   microbenchmarks: 96.459 -> 93.970 ms, +126.5 MiB temporary peak. Full model
   removed 54,488 GPU events/update and provisional peak owner-accounted VRAM
   rose about 63.75 MiB, not gigabytes.

All three switches are enabled by default for the optimized Arc runtime;
historical/reference benchmark modes keep them off. Experimental Flash and DKV
variants are not enabled.

## Rejected candidates

The largest single bottleneck, DKV backward, was tested with register key reuse,
workgroup row counts 8/4/2, XOR-swizzled local memory, and unroll counts 1/4/8/16.
The production causal workload is four heads per launch, T1024, D32. Baseline
GPU time was 0.1091 ms; the unroll variants were 0.1580 / 0.1147 / 0.1149 /
0.1116 ms. Register/swizzle alternatives were also slower (register-row-2
spilled). All tested outputs were bit-exact, but slower variants were rejected.
Some dense/noncausal variants won a microbenchmark; this does not justify
enabling them for the measured causal model.

Evidence: `arc-dkv-register-tile-20260921.json`,
`arc-dkv-swizzle-tile-20260921.json`, `arc-dkv-unroll-20260921.json`.
Earlier experimental Flash/XMX attention was slower and two mixed-mode model
acceptance tests failed. Those paths remain disabled; their failure is not
masked by relaxing tolerances or presenting raw-kernel passes as model passes.

## FP8 decision

The installed B580 exposes subgroup-16 XMX, BF16 conversion and TF32 matrix
extensions, but no FP8 matrix API. The [Intel B580 architecture deck](https://download.intel.com/newsroom/2024/client-computing/Intel-Arc-B580-B570-Media-Deck.pdf)
lists INT2/4/8, FP16, BF16 and TF32; the actual
[Intel OpenCL matrix extension](https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_subgroup_matrix_multiply_accumulate.html)
defines integer, BF16 and FP16 matrix operations, not E4M3/E5M2.

Current BFP8 is signed-byte payload plus FP32 block scale, decoded to the
existing BF16 XMX operand. It is not an E4M3/E5M2 float. For example, payload 31
at scale 1 is exact in the current path, but not exactly representable in either
of those FP8 formats. Substituting them changes quantization semantics. Software
FP8 would add conversion without an exposed native FP8 matrix instruction here.
It is therefore **not adopted**. This is a statement about this B580/backend,
not all Intel GPUs or all future drivers.

Read-only capability command:

```powershell
dotnet run --project .\NNtrain.Benchmarks -c Release --no-build -- --probe-arc-capabilities
```

## Reproduce

Each output must be a new filename. Run GPU jobs serially with training stopped.

```powershell
dotnet build .\NNtrain.Benchmarks -c Release
dotnet run --project .\NNtrain.Benchmarks -c Release --no-build -- `
  --probe-arc-transformer .\training.transformer.json .\benchmark-results\arc-wall-repeat.json `
  --warmup 1 --steps 4 --timeline
```

To compare old scheduling/packing using the same binary, add
`--pipeline-events off --pow2-pack off --bulk-qk off`. To measure without the
timeline or detailed profiler, omit `--timeline` (and do not pass `--profile`).
The phase fences and standard timing events remain part of that harness.

## Final clean profile

`benchmark-results/arc-wall-timeline-final-20260921.json` uses the new defaults
without optimization flags. p50 **29,705.47 ms/update**, mean **29,706.33 ms**,
four-update total **118,825.31 ms**. Samples: 29,700.77 / 29,702.78 / 29,713.61 /
29,708.16 ms. Reduction versus the frozen baseline: **6.93%**, speedup **1.0745x**.
This is not the previously requested 2.5x; that target remains unmet.

Without timeline/detailed profiling, the same binary and defaults measured
**29,676.34 ms/update** p50 (8,833.4 synthetic tokens/s), in
`benchmark-results/arc-wall-unprofiled-final-20260921.json`. The two final runs
differ by 29.13 ms / 0.098% at p50. This is an observed difference including
run-to-run noise, not an exact subtraction-based measurement of profiler cost.

All four final traces have 307,938 events, 100% exclusive coverage, zero missing
events, zero opaque uploads, zero overlap, and zero unassigned host time.
Their trace-window sum and partition sum are both 118,825.2312 ms (difference
below 3e-11 ms from floating-point summation). The outer benchmark stopwatch
includes an additional 0.0825 ms across all four updates for opening/closing
that trace window; it is not a hidden kernel/transfer interval. The percentages
refer to the explicitly timestamped trace window.
Endpoint half-brackets: 0.0284–0.0286 ms; fitted clock drift: 33.8–37.2 ppm.
Final mean event-collection idle is 129.85 ms (from 1,500.33 ms).
Native-allocation exposed idle is 4.44 ms. Remaining GPU time is dominated by
attention 11,746.34 ms, GEMM 8,815.13 ms and pack/decode 5,348.41 ms.
The original DKV kernel remains the largest operation at 3,693.62 ms/update;
this work does **not** claim to have accelerated it.

H2D remains **2,097,152 bytes/update** (the four input/target microbatches), and
D2H **2,588 bytes/update** (existing scalar diagnostics). No weight, activation
or gradient readbacks were added. Peak OpenCL owner-accounted allocation is
**10,591.49 MiB**, versus baseline **10,527.74 MiB** (+63.75 MiB). Active buffers
at update boundaries remain 1,478.14 MiB and idle cache 4,066.94 MiB across all
four samples. This is backend accounting, not driver-reported dedicated VRAM.

All full-size updates had finite loss and gradient norms. Largest difference
from the earlier baseline run was 0.000028 loss and 0.000212171 gradient norm.
This is **not** a bitwise full-model equivalence claim: embedding gradient
scatter uses floating-point atomic accumulation, and quantized optimizer
updates can amplify run-to-run ordering differences. Isolated Q/K and packing
comparisons are bit-exact; the small-model forward/gradient/master-weight/update
tests retain their existing tolerance. Long-run convergence is not established
by a five-update benchmark.

## Validation and limits

- Full solution Release build: zero warnings, zero errors.
- Core Arc regression: **416 distinct cases passed**, combining
  `arc-wall-final-regression-20260921.trx` with the final results in
  `arc-wall-dispatch-verified-20260921.trx`. The first run had 406 passes and 10
  fixture/dispatch-expectation failures. The 31-case rerun passed after freezing
  the old Q/K test to its old dispatch, checking the new offset dispatch in the
  bulk test, and populating deferred-release test temporaries on-device so the
  new explicit blocking upload cannot bypass the retirement-budget condition.
  No numerical tolerance was widened and no failing numerical assertion was
  deleted. The new upload-pool-miss test confirms an actual H2D event, exact
  payload values and zero opaque copies.
- Arc CLI integration: **19 passed**, including FineWeb tokenization/training,
  checkpoint/resume/generation and persistent HTML in an isolated small run:
  `arc-wall-integration-20260921.trx`.
- Benchmark tests: **35 passed**, `arc-wall-benchtests-20260921.trx`.
- GPU work was run serially. CUDA hardware comparison was not rerun; CUDA is
  unavailable in the current environment. No 210/2100-step soak or full-size
  corpus convergence experiment was performed in this pass.
- The six `ArcAttentionExperimentalAcceptance` cases were explicitly excluded
  from this regression run. Earlier mixed-model failures in disabled Flash/XMX
  candidates remain unresolved; this report does not claim a green unfiltered
  test suite or recommend enabling those experimental options.
- Production JSON, datasets, checkpoints and loss history were left unchanged.
  No commit or push was performed.
