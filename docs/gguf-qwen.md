# Qwen2 GGUF import and Arc generation

PR #11 adds Qwen2/Qwen2.5 GGUF inspection, decoding, and Intel Arc inference.
The existing Transformer training and generation commands retain their own
configuration and precision settings.

## Generate

```powershell
dotnet run --project NNtrain.Cli -c Release -- qwen-gguf --model C:\models\qwen.gguf --prompt "国会議事堂への行き方を教えて" --device 0 --max-new-tokens 1000
```

The command uses greedy sampling and a single Arc GPU. Large projection matrices
remain in their original Q4_K/Q6_K representation in host memory and VRAM;
Q4_K/Q6_K token embeddings also remain encoded, with only selected rows decoded
on the GPU into Float32 activations. F32/F16/BF16 embeddings use the dense path.
When `output.weight` is absent, a Q4_K/Q6_K token embedding supplies the output
head. This tied path currently holds separate encoded buffers for embedding and
output; it does not share their storage. This command does not use `generate.json`
or its two-GPU selection.

Output streams by default: complete UTF-8 characters are written and flushed
after each sampled Qwen2 token. `--no-stream` buffers the generated text instead.
Mixed fullwidth digits such as `--max-new-tokens 1０００` are accepted. Qwen2 timing
on stderr separates loading, the first token, and subsequent decode throughput.
The maximum is an upper bound: EOS or the model context limit can stop earlier.
The supplied prompt is used directly; no chat template is inserted.

Qwen2 generation prefills once, retains rotary K/V per layer, then processes one
new token at a time. Only the last hidden row enters the vocabulary head and only
one vocabulary row is downloaded. Caches and temporary activations are released
on completion, EOS, or callback failure. Q4_K/Q6_K cooperative kernels and a
parallel inference RMSNorm replace the serial paths on supported devices.
Reference switches in `ArcExecutionOptions` allow numerical/performance A/B.

## Supported scope

- GGUF versions 2 and 3 with `general.architecture = qwen2`.
- Split GGUFs are rejected with a merge command before missing tensors can cause
  a misleading error: `llama-gguf-split --merge <first-shard.gguf> <merged.gguf>`.
- Dense import decodes F32, F16, BF16, Q4_K and Q6_K tensors.
- Native quantized inference requires Q4_K/Q6_K projection weights with complete
  256-element blocks per row. Tensor dimensions are checked before payload reads.
- RMSNorm, split-half RoPE, grouped-query attention and SwiGLU are implemented.
- Prompt prefill and uncached GQA support at most 4096 tokens per forward call.
  Cached decode can continue beyond 4096, up to the GGUF context limit, using
  device scratch for longer attention rows. No-grad inference avoids the
  quadratic saved-probability buffer.
- The native matrix path still requires Q4_K/Q6_K rows divisible by 256;
  checkpoints with other projection formats (including some 896-wide models)
  remain unsupported and produce explicit errors.
- Tokenization recognizes CONTROL and USER_DEFINED entries, normalizes ordinary
  text to NFC, and classifies supplementary Unicode letters by scalar value.
- Quantized Qwen2 training is unsupported. For Qwen3.5 Attention/MLP LoRA,
  see [the LoRA guide](qwen35-lora.md).

## Integration corrections

The merge includes Qwen kernel resource registration and OpenCL identifier fixes, tokenizer split-regex
repair, per-token temporary GPU buffer release, correct FP16 subnormal decoding
in quantized kernels, tensor-operation inventory entries, and failed-reader file
cleanup. Q4_K CPU decoding reuses bounded stack scratch across blocks.

Regression tests include synthetic vocabularies/models, an official gguf-py
fixture with independently decoded values, and Arc comparisons of cached versus
full-prefix logits. Real-model timings and generation comparisons are recorded
in [the optimization report](benchmarks/qwen-gguf-20260925.md). These checks do
not establish full pretrained-checkpoint logit parity with Hugging Face.

### Merge validation (2026-09-25)

The integrated source at `692a356` was tested on the local Arc B580 system:

| Check | Result |
| --- | --- |
| Release solution build (`--no-restore`) | 0 warnings, 0 errors |
| Qwen/GGUF and Float16 operation inventory | 48 passed |
| Existing Arc generation, KV cache, tensor-parallel and precision checks | 94 passed |
| Generation commands, configuration, training/resume and dual-GPU integration | 48 passed |

All 190 selected tests passed with no skips. The integration run includes both
single-GPU and two-GPU training and generation; the earlier Arc optimization
benchmark records are preserved separately under `benchmark-results/`.

### Resident embedding validation (2026-09-26)

Validated on Intel Arc B580 after rebasing the resident embedding changes onto
`5d45ecc`:

- `dotnet build NNtrain.slnx -c Release`: 0 warnings, 0 errors.
- `dotnet test NNtrain.Core.Tests -c Release --no-restore --filter "FullyQualifiedName~Qwen|FullyQualifiedName~Gguf|FullyQualifiedName~TensorFloat16Operation"`:
  53 passed, 0 failed, 0 skipped. Includes Q4_K/Q6_K selected-row CPU parity,
  encoded-weight upload reuse, tied-head directory validation and generation lifetimes.
- Real Qwen2.5-3B-Instruct Q4_K_M, raw prompt `こんにちは`, one greedy token:
  input ID `89015`, output `こんにちは、` (exit 0).
- Synthetic Q4_K tied-embedding GGUF: output `こんにちはA` (exit 0).

The real checkpoint has a separate Q6_K output head. Tied-head coverage uses a
synthetic fixture. This check does not establish full-logit parity or long-form
generation quality. No LoRA, KV cache or performance changes are included.

## Qwen3.5 dense text models

`qwen-gguf` also recognizes `general.architecture = qwen35`. Filenames are not
used to choose the architecture; for example, the local file named
`Qwen.Qwen3.8-27B.f16.gguf.Q4_K_M.gguf` contains the Qwen3.5 dense 27B layout.

```powershell
dotnet run --configuration Release --project NNtrain.Cli -- qwen-gguf --model "C:\models\Qwen.Qwen3.8-27B.f16.gguf.Q4_K_M.gguf" --prompt "こんにちは" --max-new-tokens 16 --devices 0,1
```

When no device option is supplied, Qwen3.5 selects enough available Arc devices
for its encoded weights and a workspace reserve. `--device 0` forces one GPU;
`--devices 0,1` selects two explicitly. The 27B Q4_K_M file requires more than one
12 GiB B580. Layers are assigned to fixed, contiguous groups across the selected
GPUs. The loader checks each group's capacity and each allocation size before
uploading. It does not expand the large weight matrices to Float32.

### Execution and supported scope

- Q4_K/Q5_K/Q6_K/IQ2_S/IQ3_S embedding and projection matrices stay encoded
  in Arc VRAM. Encoded
  host payloads are discarded after each blocking upload. Loading reports both
  resident weight bytes, GPU state bytes and actual live device allocations.
- RMSNorm, gated full attention, partial RoPE, softmax, Gated DeltaNet,
  residual updates, SwiGLU and greedy token selection all run on Arc. Layer
  activations stay on device. CPU work is tokenization, command scheduling and
  copying the hidden vector between separate GPU contexts at layer boundaries.
- The text path supports explicit Q/K head widths, Q/K normalization, gated Q
  projections, partial RoPE, causal convolution and persistent DeltaNet state.
  Full-attention KV and DeltaNet state remain in GPU memory for a sequence.
  KV capacity starts at 16 tokens (or the model's smaller context) and grows
  with device-to-device copies. Growth checks the live VRAM budget before
  replacing buffers; the advertised full context is not preallocated.
- Tokenization uses the `qwen35` BPE split, including combining marks. Prompts
  are raw text; the embedded chat template is not applied automatically.
- Dense text `qwen35` only: no MoE, vision input, MTP or scaled RoPE.
  Unsupported tensor directories are rejected before payload loading.
- `GenerateTokenIds` resets GPU sequence state and downloads only an 8-byte
  token/status result for each generated token. `ForwardToken` advances state
  and optionally downloads full logits for inspection. `Reset` zeroes recurrent
  state on the GPU and reuses KV capacity without host transfers. After a failed
  compute step, `Reset` is required before continuing. Instances are not thread safe.

The math and GGUF head ordering follow the
[llama.cpp Qwen3.5 implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp),
[conversion code](https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen.py)
and [tokenizer](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-vocab.cpp).

### Initial mixed CPU/GPU baseline (2026-09-27)

The local GGUF has 851 tensors, 64 layers, width 5120, 24 query heads / 4 KV heads,
head width 256 and vocabulary 248320. Its file size is 16,547,400,064 bytes.
On two Arc B580 devices, quantized residency was 8,099,020,800 bytes on device 0
and 8,426,803,200 bytes on device 1 (7.54 / 7.85 GiB).

With raw prompt `こんにちは`, NNtrain encoded token `85951` and greedily generated
`5205,150517` (`、ゆ`). An independent CPU llama.cpp run against the same GGUF
produced the same input and both output IDs. This short check does not establish
long-context parity, chat quality or performance.

Validation commands:

```powershell
dotnet build NNtrain.slnx -c Release --no-restore
dotnet test NNtrain.Core.Tests -c Release --no-restore --filter "FullyQualifiedName~Qwen|FullyQualifiedName~Gguf|FullyQualifiedName~TensorFloat16Operation"
```

The Release build passed with zero warnings/errors, and all 106 selected tests
passed with no skips. Tests cover tensor directory validation, GDN numeric
references, partial RoPE, tokenizer splits, state reset, EOS/context behavior,
separate/tied heads and single-GPU versus two-GPU logits. The existing Qwen2.5
3B smoke test also still produced `こんにちは、`.

[Recorded comparison data](qwen35-local-validation.json) includes the top-10
log probabilities and transfer counts for both real-model tokens. Initial upload
bytes equal resident encoded-weight bytes exactly. Each subsequent token uploaded
14,590,980 / 15,604,736 bytes of inputs and zero biases, rather than the 16.5 GB
of weights. The greedy IDs match the CPU reference, but probabilities differ
(first token: NNtrain -0.977396, llama.cpp CPU -0.938354). Full-logit equivalence
is not claimed.


### GPU computation verification (2026-09-27)

The same 27B GGUF was verified after moving all numerical inference operations
and persistent sequence state to Arc. The CPU math remains only as test reference.

- Actual live allocations after loading: 8,184,790,016 / 8,512,592,896 bytes on
  GPUs 0/1, including quantized weights, auxiliary Float32 weights, zero biases
  and 79,495,168 bytes of GPU sequence state per device.
- Input `85951` (`こんにちは`) still greedily generates `5205,150517` (`、ゆ`).
  The two steps' top-10 IDs and ordering match the initial Float32 baseline;
  maximum absolute top-10 logit differences are 0.000036 / 0.000015.
- Per token, upload is 4 bytes to GPU0 (input token) and 20,480 bytes to GPU1
  (the hidden vector at the layer split). The corresponding GPU0 readback is
  20,480 bytes. GPU greedy output reads only 8 bytes from GPU1 per generated
  token. Layer operations do not download activations, KV or recurrent state.
- Four real-model forward steps, including reset and greedy generation, leave
  the live allocation count unchanged. Weights are not reuploaded.
- `ForwardToken` explicitly returning logits reads 993,280 bytes from GPU1;
  normal `GenerateTokenIds` uses GPU argmax instead.

See [GPU validation data](qwen35-gpu-validation.json). Independent CPU-reference
kernel tests cover multi-token DeltaNet, head tiling, nonzero convolution history,
partial RoPE, full attention beyond 4096 tokens, stable softmax and non-finite
argmax handling. End-to-end tests cover GPU cache growth, exact transfer counts,
reset, failure recovery and single/two-GPU parity including tied heads.

This retains the initial limitation: full-logit parity against llama.cpp's CPU
backend and long-context quality are not established. The GPU migration is
compared to NNtrain's own Float32 baseline, independently of that CPU backend's
Q8_K activation arithmetic.
GPU migration regression run: the same focused command now passes **124 tests**
with zero failures/skips; the Release solution build has zero warnings/errors.

### Generation performance

The default Qwen3.5 route now uses optimized quantized projections, fused
DeltaNet operations and GPU sequence-cache reuse. The measured two-B580 27B
decode rate improved from about 0.26 to 10 tokens/second. Qwen2's separate route
is unchanged by these Qwen3.5 optimizations. See the
[performance report and reproduction commands](qwen35-performance.md) for
measurement conditions, numerical checks and the candidate comparisons.

### Qwen3.5 streaming output

Qwen3.5 generation streams text by default, flushing each completed UTF-8 chunk
as tokens arrive. Japanese characters and emoji split over multiple tokens are
buffered until complete. The final text is not printed a second time. Add
`--no-stream` to print the result after generation and include generated token IDs.
Timing is written to stderr. The prompt is still passed exactly as supplied;
chat templates are not inserted automatically.

### Mixed IQ2_M checkpoint

Qwen3.5 also accepts the Q5_K, IQ2_S and IQ3_S tensors used by the tested
`Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf`. See the
[format support and validation report](qwen35-iq-validation.md).

### Qwen3.5 LoRA training

`qwen-lora` trains Attention, Gated DeltaNet and MLP adapters on the frozen
quantized GPU base. See [configuration, data and resume commands](qwen35-lora.md)
and [the 27B training validation](qwen35-lora-validation.md).
