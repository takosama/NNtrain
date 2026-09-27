# IQ2_M generation speed improvement

Measured on 2026-09-27 with the same
`Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf` and Intel Arc B580 GPUs.

## Result

| Workload | Original | Optimized |
| --- | ---: | ---: |
| Two GPUs, raw 8-token prompt, 24 output tokens, mean of 2 runs | 4.405 token/s | 14.879 token/s |
| One GPU, same raw prompt, 256 output tokens, 2 runs | not measured | 14.717 / 14.718 token/s |
| Normal streaming CLI, one GPU, manual chat prompt, 64 output tokens | 4.40 token/s | 14.80 token/s |

The matched two-GPU comparison improves decode throughput **3.38x**.
The requested **10 token/s** target is exceeded with one or two GPUs in these
tests. Load time is excluded from decode timing. The benchmark disables EOS
stopping; decode spans the first-to-last output callback. OS/driver caches were
not flushed. The normal CLI uses its usual streaming text output and EOS behavior.

## Changes

- IQ2_S and IQ3_S SG16 kernels decode eight weights together. One table lookup
  supplies a packed octet; byte reinterpretation replaces repeated 64-bit shifts.
  Eight independent FP32 accumulators shorten dependency chains.
- Q5_K reuses each high-bit byte across all eight quantization groups and each
  low byte across both nibbles, preserving its original lane accumulation order.
- Original quantized bytes stay in VRAM. No dense weight expansion, requantization,
  weight changes, CPU projection or per-token weight upload was introduced.
- Existing cooperative/reference kernels, GPU KV/DeltaNet state, greedy selection
  and streaming text remain available.

Second-run accumulated kernel times for the matched short workload:

| Kernel | Original | Optimized |
| --- | ---: | ---: |
| IQ2_S | 5111 ms | 1238 ms |
| Q5_K | 716 ms | 104 ms |
| IQ3_S | 357 ms | 82 ms |

These are profiler event sums, not independent wall-clock intervals.

## Correctness and memory

Release build: zero warnings/errors. **56 tests passed**, no failures/skips:

```powershell
dotnet build NNtrain.slnx -c Release --no-restore
dotnet test NNtrain.Core.Tests -c Release --no-build --no-restore --filter "FullyQualifiedName~Qwen35GpuIqTests|FullyQualifiedName~Qwen35GgufTests|FullyQualifiedName~Qwen35GpuLinearTests|FullyQualifiedName~Qwen35ResidentModelTests"
```

The numeric tests compare against independent native ggml decoded values,
including all IQ tables, signs, half subnormals, matrix widths and tail guards.
An initial draft used the OpenCL reserved name `half` for a loop variable; it
was corrected to `halfIndex` before the accepted tests and measurements.

Both optimized 24-token sequences match the original baseline exactly. Both
optimized 256-token sequences match each other, and their first 24 tokens match
the baseline. The final 64-token chat CLI text also matches the original text.
These checks do not promise bitwise logits or identical arbitrary continuations;
the IQ kernels' FP32 accumulation order changed.

Single-GPU encoded weights remain **10,151,874,560 bytes**. After both 256-token
runs, state bytes are 224,002,048, live allocation is 10,387,451,904 and peak is
10,456,647,308. These values are stable on reuse, including KV growth to 512 slots.
Each run uploads only 1,052 bytes of input IDs and downloads 2,048 bytes of
selected token/status results. Two-GPU encoded weights and transfer counters also
match the baseline. Backend counters include pools and are not driver-reported VRAM.

## Reproduce

From the repository root (choose a new report filename):

```powershell
dotnet run -c Release --no-build --project NNtrain.Benchmarks -- --qwen35-generation-probe --model ".\models\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf" --output iq-single-256.json --tokens 256 --runs 2 --devices 0 --kernel auto
```

The normal `qwen-gguf` command needs no extra option: `Auto` selects the optimized
SG16 kernels on supported Intel devices. The 10-token/s target is validated for
this tested file and workload; long contexts and other devices remain unmeasured.

[Measurement data](qwen35-iq-performance.json) records IDs, token timings,
memory, model/binary hashes and final CLI verification. The earlier
[format-support report](qwen35-iq-validation.md) records the 4.4-token/s baseline.
