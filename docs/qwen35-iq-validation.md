# Qwen3.5 mixed IQ2_M GGUF validation

## Failure and supported storage

The local `Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf` failed at
`output.weight` with GGML type 13. Its 851 tensors actually contain:

| GGML type | Format | Tensor count |
| --- | --- | ---: |
| 0 | F32 auxiliaries | 353 |
| 12 | Q4_K | 65 |
| 13 | Q5_K output head | 1 |
| 21 | IQ3_S | 24 |
| 22 | IQ2_S | 408 |

The loader now accepts Q4_K, Q5_K, Q6_K, IQ2_S and IQ3_S for Qwen3.5 matrices,
including embedding and tied output matrices. Type-specific byte counts are used
for VRAM planning and uploads. Q5_K blocks use 176 bytes, IQ2_S 82 and IQ3_S 110,
each representing 256 values. Full Float32 weight expansion is not performed.
Cooperative, Intel SG16 and reference projection paths all decode in registers;
selected embedding rows also decode on the GPU. KV/state reuse and streaming
text remain enabled. Other IQ encodings are still rejected before payload loading.

Format layouts and lookup tables follow pinned
[llama.cpp definitions](https://github.com/ggml-org/llama.cpp/blob/95887577ab5fead779581a7030a83c7752ff3234/ggml/src/ggml-common.h)
and [decoding routines](https://github.com/ggml-org/llama.cpp/blob/95887577ab5fead779581a7030a83c7752ff3234/ggml/src/ggml-quants.c).
The MIT license and provenance are retained in `THIRD_PARTY_NOTICES.md`.

## Verification on 2026-09-27

- Release solution build succeeded with zero warnings/errors. The first build
  attempt could not replace DLLs while a user generation process held them open;
  rebuilding after it exited succeeded. No compiler fix was needed for that lock.
- 42 quantization/directory tests passed, followed by 14 existing Q4_K/Q6_K and
  resident-model regressions; no failures or skips.
- GPU embedding and three projection variants were compared against 139 blocks
  decoded by an independent native `ggml-base.dll`, including real model blocks,
  every IQ2/IQ3 table entry, sign/scale extremes, FP16 subnormals, odd output tails,
  multiple rows and real 5120/17408 input widths. Guard values stayed intact and
  projection dispatch performed no allocation/upload/download of decoded weights.
- The normal local checkout loaded all 851 tensors and streamed 64 tokens from
  the user's manual chat prompt on one automatically selected Arc B580. Encoded
  weights occupy 10,151,874,560 bytes (9.45 GiB). Load 9.625 s, first token 3.761 s,
  decode **4.40 token/s**. The original type-13 error is resolved.
- Two Arc B580 devices also generated 24 tokens twice, with identical token IDs:
  **4.397 / 4.414 token/s**. Load 16.716 s; first-token times 12.067 / 1.256 s.
  Startup/JIT and cache effects were not normalized. This is a different model
  file from the previously measured Q4_K_M checkpoint.
- Encoded weights remain 5,035,468,800 / 5,116,405,760 bytes on GPUs 0/1.
  After both runs, state bytes were 80,543,744 each and live bytes were
  5,122,286,592 / 5,203,244,032. These values remained stable on reset/reuse.
- Per 24-token run (8-token prompt), H2D was 124 / 634,880 bytes and D2H was
  634,880 / 192 bytes: input token IDs, inter-device hidden vectors and selected
  output token/status values. Quantized weights were not uploaded again.

[Recorded results](qwen35-iq-validation.json) include model/binary hashes and
the complete two-GPU probe. Independent numeric fixtures, provenance and their
generator are under `NNtrain.Core.Tests/Fixtures/IqQuantReference`.

## Commands

```powershell
dotnet build NNtrain.slnx -c Release --no-restore
dotnet test NNtrain.Core.Tests -c Release --no-restore --filter "FullyQualifiedName~Qwen35GpuIqTests|FullyQualifiedName~Qwen35GgufTests"
dotnet test NNtrain.Core.Tests -c Release --no-build --no-restore --filter "FullyQualifiedName~Qwen35GpuLinearTests|FullyQualifiedName~Qwen35ResidentModelTests"

$prompt = @'
<|im_start|>user
/think
国会議事堂への行き方を教えて
<|im_end|>
<|im_start|>assistant
'@
dotnet .\NNtrain.Cli\bin\Release\net10.0\NNtrain.Cli.dll qwen-gguf --model ".\models\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf" --prompt $prompt --max-new-tokens 64
```

Omit `--devices` for automatic selection, or use `--devices 0,1` for two GPUs.
This file fits on one B580 for the tested context. Longer contexts still require
additional KV memory and are bounded by the existing device-budget checks.
The changes do not add IQ formats to the separate Qwen2 inference path.
Full-model logits parity with llama.cpp and long-context validation of this
checkpoint remain unverified; local numeric checks establish the new block layouts.
