# Qwen3.5 GPU generation performance

## Configuration

Measured on two Intel Arc B580 12 GiB GPUs with the local
`Qwen.Qwen3.8-27B.f16.gguf.Q4_K_M.gguf` file (its architecture is `qwen35`).
All comparison runs use raw prompt `国会議事堂への行き方を教えて` (8 tokens),
24 greedy output tokens, two runs per loaded model, and no early EOS stop.
Each run resets sequence state. Tokenization, OpenCL compilation, weight loading,
console output and report export are outside generation timing. OS file caches
and driver caches were not flushed.

Decode speed is 23 divided by the time between the first and last output-token
callbacks. First-token time includes reset, prompt processing, the output head
and argmax. It measures token computation; the CLI prints the completed text
after generation. The comparison baseline already has GPU KV and DeltaNet state.

## Local comparison (2026-09-27)

Arithmetic means of two runs:

| Setting | First token | Decode | Total generation |
| --- | ---: | ---: | ---: |
| Original GPU reference, queue 128 | 27.327 s | 0.259 token/s | 116.150 s |
| Cooperative Q4/Q6 kernels | 3.727 s | 1.937 token/s | 15.600 s |
| Intel SG16 kernels | 0.727 s | 9.773 token/s | 3.081 s |
| SG16 + fused DeltaNet, queue 128 | 0.711 s | 10.141 token/s | 2.980 s |
| Selected: SG16 + fused DeltaNet, queue 512 | 0.602 s | 10.074 token/s | 2.885 s |

The selected configuration improves decode throughput **38.91 times** over the
GPU reference for this short workload. Queue 512 trades a small decode difference
for faster prompt processing. These are local observations from two samples,
not a universal maximum throughput. Startup/loading time is reported separately
and includes OpenCL compilation; the table starts after the model is loaded.

## Changes

- Coalesced Q4_K/Q6_K matrix-vector kernels with Intel SG16 reductions. `Auto`
  selects this path on supported Arc devices and a portable work-group reduction
  otherwise. Original kernels remain selectable as `Reference` for comparisons.
- Fused width-128 DeltaNet recurrent update and gated RMSNorm. Other head widths
  retain the generic implementation.
- A 512-event queue limit reduces waits during prompt processing.
- Full-attention KV, convolution history and recurrent state stay on the GPU.
  KV grows by device-to-device copy and is reused after sequence reset.
- Encoded weights remain unchanged in VRAM. There is no full-weight Float32
  expansion or per-token weight upload.
- CLI timing output separates loading, first-token generation and decode speed.
- Qwen3.5 lanes compile only their five required OpenCL resources. General Arc
  sessions retain the complete kernel set. The measured model load fell from
  74.41 to 12.64 seconds after this change (warm OS caches, compilation included).

Four independent accumulators and larger SG16 work-groups were measured and
removed because neither improved this workload. A 1024-event queue shortened
the first-token time further but reduced decode throughput; 512 is the default.
FP32 reduction order changes, so bitwise logit equality is not promised. The
comparison's generated 24-token sequences match the reference in every run.

Detailed candidate results, per-kernel times, memory counters and output IDs:
[comparison data](qwen35-performance-comparison.json).

## Final integrated verification

With `Auto`, fused DeltaNet, queue 512 and the restricted inference kernel
program, the final 24-token runs measured **9.892 and 10.213 token/s**.
Model loading took **12.639 seconds**. Two 256-token runs measured **10.104 and
10.138 token/s**, with first-token times of 0.615 and 0.599 seconds.
Both complete 256-token sequences matched each other and the earlier full-program
implementation; all 24-token prefixes matched the original reference baseline.

The 256-token check crosses KV capacity growth from 16 to 512 slots. Encoded
resident weights stayed at 8,099,020,800 and 8,426,803,200 bytes across the two
devices. After both runs, state bytes were 112,001,024 per device, live bytes
were 8,217,295,872 / 8,545,098,752 and peak bytes were
8,253,638,468 / 8,582,434,632. These counters remained stable on the second run.
Each run downloaded 2,048 bytes of final token results from the output device;
the other 5,386,240 downloaded bytes were hidden-vector transport between GPU
contexts. The model does not download vocabulary logits during greedy generation.

[Final validation data](qwen35-performance-validation.json) records both probes,
model and binary hashes, token IDs, timings, device details and memory counters.

## Reproduce

Run from this branch's repository root:

```powershell
dotnet build NNtrain.slnx -c Release --no-restore
dotnet test NNtrain.Core.Tests -c Release --no-build --no-restore --filter "FullyQualifiedName~Qwen|FullyQualifiedName~Gguf|FullyQualifiedName~TensorFloat16Operation"
dotnet run -c Release --no-build --project NNtrain.Cli -- qwen-gguf --model "C:\Users\takos\Downloads\Qwen.Qwen3.8-27B.f16.gguf.Q4_K_M.gguf" --prompt "国会議事堂への行き方を教えて" --max-new-tokens 24 --devices 0,1
```

The benchmark disables early EOS stopping to measure exactly the requested
number of output tokens. It writes the model/binary hashes, driver information,
all output IDs, token times, kernel durations and GPU allocation/transfer counters.
Choose a new output filename each time:

```powershell
dotnet run -c Release --no-build --project NNtrain.Benchmarks -- --qwen35-generation-probe --model "C:\Users\takos\Downloads\Qwen.Qwen3.8-27B.f16.gguf.Q4_K_M.gguf" --output generation-auto.json --tokens 24 --runs 2 --kernel auto
dotnet run -c Release --no-build --project NNtrain.Benchmarks -- --qwen35-generation-probe --model "C:\Users\takos\Downloads\Qwen.Qwen3.8-27B.f16.gguf.Q4_K_M.gguf" --output generation-reference.json --tokens 24 --runs 2 --kernel reference --fused-delta off --queue 128
```

The same probe accepts `--kernel cooperative|subgroup`, `--fused-delta on|off`,
`--queue 16..4096`, `--prompt`, and `--devices`. Kernel event durations are summed
across devices and must not be added to wall-clock durations. Allocation counters
describe backend buffers, including pools, rather than driver-reported VRAM.

This is a local throughput and state-reuse check. It does not establish full-logit
parity with llama.cpp, chat-template quality or maximum-context performance.

## Regression verification

Release solution build: zero warnings/errors. Before restricting the compiled
kernel set, the focused Qwen/GGUF and Float16
operation-inventory command above passed **133 tests**, with zero failures or
skips. This includes CPU-decoded Q4_K/Q6_K projection comparisons, fused versus
reference recurrent-state comparisons, and complete-model reference versus
optimized logits through KV growth. Shared and separate output heads, streaming
token callbacks, exact transfer counts, reset/reuse and two-GPU placement are
covered. The suite took 24 minutes 49 seconds on this machine; GPU context and
OpenCL program setup are repeated by the tests. After restricting kernel setup,
the same Qwen3.5 GPU numerical and resident-model tests passed **31/31** in
49 seconds, with zero skips. Their test lanes also use the restricted resource
set, including the original projection kernels used as numerical references.

```powershell
dotnet test NNtrain.Core.Tests -c Release --no-build --no-restore --filter "FullyQualifiedName~Qwen35Gpu|FullyQualifiedName~Qwen35ResidentModelTests"
```
