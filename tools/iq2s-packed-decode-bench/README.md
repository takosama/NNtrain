# IQ2_S storage and decode benchmark

This measures complete real GGUF projections, rather than an isolated 8×8 tile.
The production baseline is `q35l_iq2_s_sg16_pair`. All weights remain IQ2_S;
there is no requantization, FP16 rounding, or persistent decoded weight copy.

## Same-size storage

For `B = (inputWidth / 256) * outputWidth` blocks, one allocation contains:

| Offset | Size | Contents |
| --- | --- | --- |
| 0 | 64B bytes | 32 ushort words per block: low grid index plus 8 sign bits |
| 64B | 16B bytes | Two high grid-index pairs and their shared scale nibble |
| 80B | 2B bytes | Unmodified FP16 block coefficient bits |

Total size is exactly **82 bytes per 256 weights**, identical to GGUF IQ2_S.
The conversion is reversible for every input byte, including exceptional half
encodings. Production performs CPU packing once before the blocking upload,
so the GPU never holds both layouts. Decode reads aligned ushort codes and
shared metadata bytes while retaining the original FP32 accumulation order.

`InferencePackedIq2Decode` enables the route for eligible inference `blk.*`
IQ2_S matrices; the GUI enables it by default. Other tensor types, embeddings,
output heads, training, unsupported devices, experimental alternative decoders,
and non-BSLM XMX configurations retain their existing storage. Native packed
kernels cover single/pair decode, fused LoRA, row tiles, and BSLM/K32r prefill.

## Run

```powershell
dotnet build tools/iq2s-packed-decode-bench/Iq2sPackedDecodeBench.csproj -c Release
dotnet tools/iq2s-packed-decode-bench/bin/Release/net10.0/NNtrain.Iq2sPackedDecodeBench.dll --model models/Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf --output benchmark-results/packed-micro --device 0 --samples 9 --warmup 3 --iterations 16 --profile 0
```

Use a fresh output directory. This checks independent CPU packing against all
GPU packed bytes, restores every original GGUF byte, and compares every output
bit before and after timing. It also checks CPU reference points and output
guards. Measurements alternate route order, exclude upload/packing/validation,
and report wall-time median, mean, population SD, min/max and raw samples.
Packing latency and temporary preparation memory are reported separately.

The benchmark also retains a rejected shared signed-grid lookup candidate.
That candidate uses the original GGUF layout and a shared 128 KiB table; it is
never selected by the production model. Use `--variants packed82-pair` to omit
it. Profiled diagnostics require a separate output and `--profile 1`; do not
mix their timings with unprofiled throughput.

For a full model comparison, set only `InferencePackedIq2Decode` false/true in
two otherwise identical `Qwen35GenerationProbe` option files. Compare generated
IDs and full vocabulary logit files with `summarize_decode.py`, allowing that
single option change. For long prefill comparisons:

```powershell
dotnet build tools/iq2s-production-bench/Iq2sProductionBench.csproj -c Release
dotnet tools/iq2s-production-bench/bin/Release/net10.0/NNtrain.Iq2sProductionBench.dll --model-forward --model-packed-iq2 --model-no-profile --model models/Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf --output benchmark-results/packed-prefill --devices 0,1 --rows 4096,8192 --samples 3 --warmup 1
```

This keeps the existing row-major BSLM, batched attention and split output head
enabled on both sides. It requires bit-identical full vocabulary logits and
includes reset, prompt prefill and the final token's logits download. Model
load and tokenization are outside prefill timing. Run GPU benchmarks serially.

The saved release comparison was built from a clean export of the Git index.
To repeat it in a fresh checkout, first run
`benchmark-results/iq2s-packing-20261008/prepare_staged_build.py`, then run
`benchmark-results/iq2s-packing-20261008/run_final_validation.ps1` with a fresh
`-OutputDir`. The validation script keeps its output separate from the saved
measurements.
