# Qwen3.5 Attention / MLP LoRA validation

Measured 2026-09-27 on two Intel Arc B580 12 GiB GPUs, .NET 10 Release.

## Implemented

- Frozen encoded Q4_K/Q5_K/Q6_K/IQ2_S/IQ3_S matrices stay in VRAM; transpose products decode directly from those bytes for input gradients.
- FP32 rank/alpha configurable LoRA on Attention, Gated DeltaNet projections and MLP, with selectable layers/targets and optional vocabulary head.
- Full causal GQA and Gated DeltaNet backpropagation through time, including causal convolution, partial RoPE, Q/K normalization, gates and residuals. Large recurrent state tapes are recomputed for one layer during its backward pass.
- Response-only shifted cross entropy including EOS, global gradient clipping and AdamW update only adapter parameters.
- Atomic adapter/optimizer save, base SHA-256 and dataset/config identity checks, checksum, deterministic resume cursor. Inference accepts `--adapter` and keeps streaming/KV state.

## Verification

Final normal-checkout Release build: **zero warnings/errors**.

```powershell
dotnet build NNtrain.slnx -c Release --no-restore --verbosity quiet
dotnet test NNtrain.Core.Tests -c Release --no-build --no-restore --filter "FullyQualifiedName~Qwen35"
dotnet test NNtrain.IntegrationTests -c Release --no-build --no-restore --filter "FullyQualifiedName~Qwen"
```

**111 core + 21 integration tests passed**, no failures/skips. This includes independent finite differences for sequence Attention/DeltaNet and sampled full-model adapter A/B gradients, decoded-reference products for all five quantizations, tied heads, masked loss, frozen-base recovery, optimizer resume equality, corrupted/wrong-base/wrong-data checkpoint rejection, CLI train/save/resume/generate and existing inference/streaming regressions.

## Actual 27B IQ2_M run

`Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf`, SHA-256
`676DB8CCD38035117E705858FD591964005CB0A921B39255B0691D8339A8DFE2`.

All **64 layers / 496 adapters**, rank 8, alpha 16, learning rate 0.0001,
58,363,904 trainable parameters. No output-head adapter. One repeated smoke
example: prompt `こんにちは`, response `こんにちは。`, configured chat prefixes,
12 tokens including EOS (11 forwarded positions, 3 supervised targets).

| Update | Loss before update | Gradient norm before clipping | Update seconds |
| ---: | ---: | ---: | ---: |
| 1 | 7.726788 | 17.4056 | 6.913 |
| 2 | 3.937487 | 17.0025 | 6.517 |
| 3 (resumed) | 1.981630 | 8.81073 | 6.537 |
| 4 (normal checkout, resumed) | 0.357264 | 5.95677 | 6.554 |

Every adapter B matrix is finite and nonzero in the saved step-4 checkpoint.
The checkpoint is **700,397,479 bytes**, including Adam state. Update timings
exclude model load, base hashing and checkpoint writes. Steps 1–3 used the
publication checkout; step 4 and final validation used the normal checkout.

Encoded weight bytes remain **5,035,468,800 / 5,116,405,760**. GPU allocation
peaks were **5,968,829,124 / 6,072,744,816 bytes** (about 5.56 / 5.66 GiB).
These are backend allocation counters, not driver VRAM readings.

Reloading the adapter and supplying the exact training prefix generated
`こんにちは。` followed by EOS. The three-token output reported 3.66 token/s;
this is too short to characterize sustained adapter inference speed. An initial
manual prompt omitted the trailing assistant newline and stopped immediately;
the documented generation command includes that newline.

Without an adapter, the same earlier 64-token chat probe still produced identical
text at **14.76 token/s** (previous 14.80). No-adapter decoding retains the
optimized inference path.

## Reproduce

Use the normal repository directory. The supplied configuration defaults to all
Attention/MLP targets on both GPUs. Replace the example JSONL with training data.

```powershell
dotnet .\NNtrain.Cli\bin\Release\net10.0\NNtrain.Cli.dll qwen-lora --model ".\models\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf" --config .\qwen-lora.example.json
# Increase maxSteps (total desired updates), then resume:
dotnet .\NNtrain.Cli\bin\Release\net10.0\NNtrain.Cli.dll qwen-lora --model ".\models\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf" --config .\qwen-lora.example.json --resume
```

See [usage/configuration](qwen35-lora.md) for generation with `--adapter` and
[raw validation data](qwen35-lora-validation.json) for logs, hashes and per-matrix
norms. The sample is only an execution check, not evidence of useful model quality.
Long training contexts, larger corpora and sustained adapter generation speed
remain unmeasured. This is an NNtrain checkpoint, not PEFT or merged GGUF export.
