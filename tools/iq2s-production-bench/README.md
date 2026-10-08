# NNtrain IQ2_S projection comparison

This benchmark compares the actual NNtrain projection operands: **FP32 activations × IQ2_S weights**, with FP32 bias and output. Activations are never quantized to IQ2_S. It references NNtrain.Arc/Core and calls the same production kernel wrappers. The default mode measures projections; `--model-forward` separately loads the complete transformer.

## Build

```powershell
dotnet build tools/iq2s-production-bench/Iq2sProductionBench.csproj -c Release
```

## Correctness smoke and range fallback

```powershell
dotnet tools/iq2s-production-bench/bin/Release/net10.0/NNtrain.Iq2sProductionBench.dll --rows=130 --input=512 --output-width=65 --tiles=8x16,16x32,32x32 --samples=1 --warmup=0 --output=benchmark-results/iq2-production-smoke
dotnet tools/iq2s-production-bench/bin/Release/net10.0/NNtrain.Iq2sProductionBench.dll --rows=130 --input=512 --output-width=65 --tiles=16x32 --fallback --samples=1 --warmup=0 --output=benchmark-results/iq2-production-fallback
```

The output directory must not contain an existing `results.json`. Failed runs are saved with `complete: false` and the exception; successful completion alone sets it to `true`.

## Full real weight matrices

```powershell
dotnet tools/iq2s-production-bench/bin/Release/net10.0/NNtrain.Iq2sProductionBench.dll --model=models/Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf --tensors=attn_gate,ffn_gate,ffn_down --rows=256,512,1024,2048,4096,8192 --tiles=16x32 --bslm-k=32r --samples=9 --warmup=3 --output=benchmark-results/iq2-production-real
```

`--tensors` accepts exact tensor names or suffixes such as `attn_gate`, `ffn_gate`, `ffn_down`. A suffix selects the first complete two-dimensional IQ2_S tensor with that suffix. It reads the entire selected tensor; input/output widths come from the GGUF directory. Synthetic input weights use `--input` (multiple of 256) and `--output-width`.

## Routes and measurement

- `--modes=inference,training,tiled` selects measured forward routes.
- Original inference: input high/residual FP16 packing, range flag zero, then existing `q35l_prefill_xmx_iq2_s_gguf_bslm`.
- `--bslm-k=32r` adds the inference candidate with the same arithmetic order and row-major high/residual input storage. This route must be bitwise equal to the original BSLM before and after timing. Other K/decode-layout candidates are experiments; their names are recorded independently.
- Original training: existing FP32 `q35t_linear_iq2_s_rows4`.
- Tiled: `ArcExecutionLane.Iq2TiledForward`, including required packing and temporary buffers each call. `--tiles` controls M×N candidates; `--panel-columns=0` uses the wrapper's bounded default.
- `--row-major` adds the row-major activation layout for the tiled learning candidate. This is distinct from `--bslm-k=32r`, which preserves the old inference BSLM arithmetic.
- `--device=0` selects one Arc GPU. Never run another GPU benchmark concurrently.
- Default 3 warmup rounds and 9 measurement rounds. Routes run round-robin with order reversed on odd rounds.
- `gpuMilliseconds` sums OpenCL kernel event durations. `wallMilliseconds` measures allocation/pool requests, host dispatch, and synchronization. Initial uploads, disk reads, CPU checks, readback, compilation, and hashing are outside both metrics.
- Raw samples contain per-kernel times, logical allocations, pool hits, and launch counts. Reported standard deviation uses the population formula.

## Validation and provenance

Every route is checked before and after timing. Entire output is compared against original inference; CPU double dot products independently decode sampled IQ2_S weight columns and check 1,024 points including corners. `--cpu-points` changes the sample count. Nonzero bias, non-finite output detection, NaN poison, and two output guards are included.

The CPU decoder uses the official IQ2_S table copied from NNtrain's kernel source. CPU error tolerance is `max(1e-7, 2e-6 × sum(abs(products), abs(bias)))`; full output relative L2 limit is `5e-5`. Fallback mode injects ±100000 activations, validates against both the original inference and independent CPU references, and additionally records comparison with the original scalar FP32 kernel. Original SG16 and new scalar fallback reduction orders differ, so bit equality is reported without being required. High/residual FP16 preserves the existing inference approximation policy.

Results record tensor SHA-256, activation SHA-256, model metadata, assembly hashes, source hashes, device and driver. Measured assemblies and lookup table must remain unchanged through completion. `--hash-model` additionally hashes the entire GGUF after timing. A result is a projection benchmark, not an end-to-end model or training speedup.

## Input-gradient transpose

Add `--transpose` to compare `dX = dY_FP32 × W_IQ2S` through original `q35t_xpose_iq2_s_vec8_rows8` + `q35t_xpose_reduce` and `Iq2TiledBackward`. This uses the same original weight shape and samples unquantized FP32 upstream gradients. Original output is zeroed per operation; the candidate overwrites it (`addToOutput: false`). `--tiles` selects candidates; `--modes` is not used in transpose mode.

`--scratch-mib=256` bounds both original split-partial row chunks and candidate transient workspace, matching the model's default transpose cap (CLI default is 128). Device-to-device copies and all temporary buffers are included in host wall timing. The GPU metric explicitly sums **kernel** event durations, excluding those copies. Transpose JSON identifies this distinction. `--output-width` must be divisible by 16 for this candidate; synthetic edge tests can use 80.

```powershell
dotnet tools/iq2s-production-bench/bin/Release/net10.0/NNtrain.Iq2sProductionBench.dll --transpose --rows=130 --input=512 --output-width=80 --tiles=8x16,16x32,32x32 --samples=1 --warmup=0 --output=benchmark-results/iq2-production-transpose-smoke
```

## Complete model comparison and production selection

```powershell
dotnet tools/iq2s-production-bench/bin/Release/net10.0/NNtrain.Iq2sProductionBench.dll --model-forward --model=models/Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf --rows=256,1024,8192 --devices=0,1 --samples=1 --warmup=0 --prefill-chunk=1024 --output=benchmark-results/iq2-model-profile
dotnet tools/iq2s-production-bench/bin/Release/net10.0/NNtrain.Iq2sProductionBench.dll --model-forward --model-no-profile --model=models/Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf --rows=256,512,1024,2048,4096,8192 --devices=0,1 --samples=3 --warmup=1 --prefill-chunk=1024 --output=benchmark-results/iq2-model-real
```

The complete-model mode uses the GUI inference settings, resetting prompt state before each measurement. It times prefill plus the next prompt token's full logits, including checkpoint capture and final readback, excluding model loading. Original and candidate sessions are sequential, so this is not an interleaved A/B test. The candidate enables **IQ2 row-major BSLM and batched text attention**. It checks full-vocabulary logit relative L2, top-1 agreement and repeat stability. The default profiling mode also checks actual candidate-kernel execution. Results are saved as `model-results.json`.

Use `--model-no-profile` for primary host latency measurements without per-kernel timing overhead. GPU timings, GPU speedup and kernel-coverage fields are then null, and the report requires separate profiled evidence from the same binary, model, device configuration and options. Both modes keep all numerical checks. Per-sample Reset, PrimePrefix and LastToken host intervals sum to wall time without extra synchronization; GC collection/allocation deltas are recorded for diagnosis. This measures base-model prefill plus one token, not GUI TTFT or steady-state generated tokens/sec.

The examples use one GPU for projection microbenchmarks and two for the full model. The complete 27B model plus a growing 8192-token KV cache can exceed one B580's memory budget. Compare each baseline and candidate under its own matching device configuration; do not combine one-GPU operator timings with two-GPU model timings as if they shared the same scope. Failed runs are excluded from reports.

`Qwen35ExecutionOptions.IQ2TiledForward` and `IQ2TiledBackward` enable the candidate production dispatch; setting them to `false` restores the original IQ2 routes. In inference the selected change is row-major packing plus `q35l_prefill_xmx_iq2_s_gguf_bslm_k32r` at **M >= 512**, preserving the old BSLM arithmetic. Smaller IQ2 batches retain the previous path. The complete-model candidate also enables `InferenceBatchTextAttention`, so its 256-token case still changes attention even though IQ2 stays original. M means rows in an actual projection, not total prompt length: a 1024-token chunk bounds projection M for longer prompts.

FP32 training forward uses the new scaled 16×32 route from 128 rows, except when an explicit experimental BF16/FP16 training option is selected. Backward requires 128+ rows, N divisible by 16, sufficient GPU memory and a scratch cap of at least 128 MiB. It chunks rows within that cap. Embedding lookup, resident IQ2 panels and other quantization formats retain their existing paths.

The learning route normalizes input rows by exact power-of-two scales before FP16 high/residual splitting; transpose also normalizes decoded weight columns. This retains tiny normal gradients, but is not bit-identical to the old FP32 reduction order. Unsupported ranges fall back to raw FP32 device arithmetic. The selected inference BSLM route does not use this normalization or change the old arithmetic order.

For the K-major learning candidate, activation high/residual panels above 384 MiB use 2,048-row device chunks to avoid the large strided working set. The row-major candidate avoids that default split. Packing and any input/output device copies are included in wall time; kernel event totals exclude copies. `rowChunkRows` can force smaller multiples of eight for numerical testing.

## Validated Japanese report

```powershell
python tools/iq2s-production-bench/summarize.py benchmark-results/iq2-production-real/results.json benchmark-results/iq2-production-transpose/results.json --model-results benchmark-results/iq2-model-real/model-results.json --model-profile benchmark-results/iq2-model-profile/model-results.json --test-trx benchmark-results/iq2s-production-20261008/tests/iq2-and-text-attention.trx --expect-rows 256,512,1024,2048,4096,8192 --expect-directions forward,transpose --expect-tensors attn_gate,ffn_gate,ffn_down --production-policy rowmajor-bslm --output benchmark-results/iq2-report
```

The stdlib-only report generator checks completion, finite outputs, CPU/full-output numerical gates, before/after bit equality for inference BSLM, raw sample statistics, kernel timing sums, model logit files and common implementation hashes. Default operator gates require 9 samples and 3 warmups; the primary full-model run requires 3 samples and 1 warmup. A separate profile audit may use 1 sample. All final inputs must come from the same frozen binaries and source; do not mix provisional builds. Multiple candidates are labelled with the best measured choice per baseline, without implicitly claiming that candidate is the production default. Explicit `rowmajor-bslm` policy requires k32r for inference and 16×32 for learning. Outputs are `report.ja.md` and `summary.json`.

The main real-weight command intentionally omits `--row-major`: production learning forward currently uses the K-major 16×32 path with bounded row chunks. The optional row-major learning candidate is a separate experiment; it must not be substituted for the integrated route in a production-speed claim.

## Generated-token throughput report

`summarize_decode.py` reads the schema-2 JSON produced by `NNtrain.Benchmarks`' real `Qwen35GenerationProbe`. It does not run the model. Use successful baseline/candidate generation artifacts from the same frozen binary, with 64 generated tokens, 4 runs, and 16 optional full-vocabulary logit snapshots per artifact:

```powershell
python tools/iq2s-production-bench/summarize_decode.py benchmark-results/iq2-decode/original.json --candidate selected=benchmark-results/iq2-decode/selected.json --require-logits --output benchmark-results/iq2-decode/report
```

The default requires at least 4 runs and excludes run 0 from performance statistics. It independently recomputes `(N−1)×1000 / (lastTokenMs−firstTokenMs)` for every run, retaining the excluded run for numerical/ID checks. It reports median/mean/population standard deviation of run throughput and median/p95 of all included token intervals. Model loading, prompt processing and the first output token are excluded from decode throughput. The JSON retains all original evidence, with source file hashes.

All runs must emit identical full generated-ID sequences, and every candidate must match the baseline's sequence. `--require-logits` also requires each adjacent `<probe.json>.logits.f32` file, validates its SHA/size/finite values and continuation, and compares every byte. Model SHA, shape, GPU devices, runtime, prompt, sampling, binaries and profiling flags must match. Only the planned residual-RMS, cached-Delta, resident-IQ2 and GGUF-BSLM option changes are allowed by default. An intentional extra switch must be declared using `--allow-option-change OptionName`; all changes appear in the report. Slower experimental candidates may be reported without selecting them as defaults.

`--delta-trx PATH` optionally adds the separate `OptionalSingleTokenBenchmark` experiment from a passing TRX. Its raw statistics are rechecked and it is labelled as a one-layer microbenchmark. Outputs are `report.decode.ja.md` and `decode-summary.json`; raw probe and logit artifacts remain unchanged.
