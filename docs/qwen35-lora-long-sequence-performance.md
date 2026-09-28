# Qwen3.5 long-sequence LoRA training performance

Measured on two Intel Arc B580 GPUs using the 27B IQ2_M GGUF and one exact
ChatML training example. The example has 998 tokens, including 976 supervised
response tokens. Each measurement performs one complete optimizer update from
the same freshly initialized adapter. Model loading, tokenization, diagnostic
snapshots, and checkpoint saving are outside the measured update. No training
data was shortened or skipped.

The original update took **100.378 s**. The final default, exact-IQ2 path took
**89.149 s**, an **11.2% reduction in update time** (1.126x throughput).
The opt-in FP16 XMX path with a bounded 1 GiB per-Arc GPU projection cache took
**55.813 s**, a **44.4% reduction** (1.798x throughput) from the original.
The target of one tenth the original time has **not** been reached. These
numbers describe this example, not the mean of a full epoch. They exclude model
load, tokenization and checkpoint saving.

## Selected changes

- Long-sequence layer checkpoints stay on their owning GPU when the projected
  allocation fits. Only the model-parallel boundary crosses host memory. The
  host checkpoint path remains as a bounded-memory fallback.
- DeltaNet reverse recurrence and its Q/K gradients run per 64-token chunk,
  avoiding a kernel launch for each token. The initial checkpoint-building
  forward no longer saves DeltaNet boundary states that it never reuses.
- The Q5_K vocabulary head reuses each decoded GGUF block across eight
  supervised rows in forward and backward. Its encoded weights remain on GPU.
- Backward scratch chunks align to the active row tile, avoiding redundant
  full-weight traversals at chunk boundaries.
- The optional FP16 XMX IQ2_S forward path decodes packed GGUF blocks into
  128-row by 64-column workgroup tiles. Only activations and the decoded tile
  are rounded to FP16; the base weights stay quantized in Arc VRAM.
- The optional GPU cache retains selected frozen IQ2_S base outputs before
  LoRA addition, then reuses them during backward recomputation. It is capped
  at 1 GiB per Arc and also constrained by the 90% device memory budget.

The loss for the final default path matches the original exactly:
`2.2414902405012924`; its gradient norm differs by about `1.2e-8` after the
DeltaNet gate simplification. The packed weights stay in
their original GGUF form; resident encoded weights are 5,035,468,800 bytes on
Arc 0 and 5,116,405,760 bytes on Arc 1.

## One-update comparisons

| Path | Seconds | Result |
| --- | ---: | --- |
| Original | 100.378 | Baseline |
| GPU checkpoints + chunked DeltaNet | 97.122 | Selected |
| Plus Q5_K forward, 8 rows | 95.199 | Selected |
| Plus aligned backward chunks and Q5_K backward, 8 rows | 91.526 | Selected |
| Rebuilt earlier exact defaults | 91.502 | Selected |
| Final exact defaults, same binary as fastest path | 89.149 | Selected |
| Final exact plus 1 GiB GPU cache per Arc | 80.509 | Opt-in, exact numerics |
| Final FP16 XMX, no cache | 57.326 | Opt-in |
| Final FP16 XMX, 1 GiB GPU cache per Arc | 55.813 | Opt-in |
| Exact IQ2_S host cache, 4 GiB prioritized | 83.669 | Opt-in, separate build |
| Exact IQ2_S host cache, 13 GiB | 79.066 | Opt-in, separate build |
| FP16 XMX plus 13 GiB host cache | 71.835 | Rejected; slower than FP16 alone |
| IQ2_S forward, 8 rows | 106.905 | Rejected |
| IQ2_S compensated XMX forward, 32 rows | 121.777 | Rejected |
| IQ2_S compensated XMX backward, 32 rows | 120.624 | Rejected |
| Larger Q5_K backward output splits (2,048/4,096) | 91.647/91.648 | Rejected |

The rejected compensated IQ2_S XMX code and larger Q5_K split options were removed.
These are separate runs with GPU and driver caches present; small timing
differences should not be overinterpreted.

With kernel timing enabled on the selected path, the main costs were:

| Kernel | GPU time across both devices |
| --- | ---: |
| IQ2_S forward projection, four rows | 40.82 s |
| IQ2_S backward projection, eight rows | 13.15 s |
| Q4_K forward projection, four rows | 8.23 s |
| Q5_K head forward, eight rows | 1.86 s |
| Q5_K head backward, eight rows | 1.23 s |

The total recorded kernel time was 91.07 s, versus 91.71 s wall time in that
earlier profiled update. With the final FP16 tile and 1 GiB per-Arc GPU cache,
the profiled update took 55.862 s. Its largest kernels were:

| Kernel | GPU time across both devices |
| --- | ---: |
| IQ2_S backward projection, eight rows | 13.15 s |
| Q4_K forward projection, four rows | 8.24 s |
| IQ2_S FP16 XMX forward projection, 128×64 tile | 7.23 s |
| DeltaNet recurrence | 3.81 s |
| DeltaNet reverse chunk | 3.49 s |
| IQ3_S forward projection, four rows | 3.46 s |

IQ2_S backward projection is the largest remaining cost. A previous 32-row
compensated XMX transpose experiment was slower, so another implementation
requires its own kernel and full-step evidence before adoption.

## Precision and memory checks

The FP16 path has a slightly different training trajectory. From the same
initialized adapter, two consecutive updates on records 16 and 17 had exact
losses `2.24149024`, `2.13277134` and FP16 losses `2.24150947`,
`2.13280109`. Their combined update times were 161.851 s and 118.458 s
respectively with the prior FP16 tile. This checks two updates, not full-epoch
quality. `iq2ForwardPrecision: "exact"` remains the default; `"fp16"` has a
separate checkpoint identity so it cannot silently resume an exact run.
In a synthetic small-activation check, FP16 relative RMS error grew from about
0.03% at an input scale of `1e-2` to about 3% at `1e-6` because of subnormal
rounding. This is another reason to keep it opt-in until longer training
quality is measured.

The 1 GiB per-Arc GPU cache captured and reused 146 outputs. On the exact path
it reduced the update from 89.149 to 80.509 s with identical loss and norm.
Peak additional
cache was 1,073,569,600 bytes per Arc, with total peak device allocations of
10.194 and 10.530 GB against 12.453 GB physical VRAM each. Host-device
transfer bytes, loss and gradient norm exactly matched the corresponding FP16
no-cache run. The host cache instead moves FP32 outputs through main memory;
its 13 GiB setting slowed the FP16 path, so it is not recommended with FP16.

## Reproduce

The benchmark accepts the JSONL prompt verbatim with `--prompt-mode direct`,
matching an existing ChatML training file rather than wrapping it in another
user/assistant template. For this dataset, record 16 is the longest example.

```powershell
dotnet build NNtrain.Benchmarks --configuration Release
dotnet .\NNtrain.Benchmarks\bin\Release\net10.0\NNtrain.Benchmarks.dll `
  --qwen35-lora-training-probe --model <base.gguf> --data <chatml.jsonl> `
  --examples 16 --prompt-mode direct --devices 0,1 `
  --options '{"CollectKernelTimings":false}' --output <new-report.json>
```

The report records token IDs, data/model/binary hashes, exact options, loss,
gradient norm, timing, memory counters, and GPU transfer/launch counters.
The benchmark never overwrites an existing report.

For the fastest measured path, replace the options JSON above with:

```json
{"CollectKernelTimings":false,"TrainingIQ2Fp16XmxForward":true,"TrainingIQ2GpuProjectionCacheMiB":1024}
```

For the exact path with GPU caching, omit `TrainingIQ2Fp16XmxForward`.
In the ordinary `qwen-lora` training JSON, the equivalent settings are
`"iq2ForwardPrecision": "fp16"` and `"iq2GpuProjectionCacheMiB": 1024`.
