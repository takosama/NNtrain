# Arc Transformer generation tuning — 2026-09-24

## Result and scope

On the local Arc B580 hardware, the current 91,031,788-parameter Transformer
improved from **18.52 to 122.47 generated tokens/s (6.61×)**. Mean generation time
for 200 tokens fell from **10.798 to 1.633 seconds (84.88% less)**.

These are **fixed-seed synthetic weights**, matching `training.transformer.json`:
32 layers, width 512, 16 heads, FFN 1536, vocabulary 11500, tied embeddings,
context 2048, `mix8_16`, BFP8 block 32. The prompt contains eight fixed synthetic
token IDs. Final runs use temperature 1 and topK 40, like `generate.json`, with
exactly 200 generated tokens and no EOS stop. The configured
`checkpoints/training.transformer.tied.model.best.safetensors` was absent; these
measurements do not establish trained-model text quality.

All final comparisons use the same binaries, configuration, seed and sampling
settings. GPU work was serialized. Each process has one full warmup followed by
three measured generations. The baseline ran before and after the candidate
(A/B/A); its six samples are pooled. Throughput is total generated tokens divided
by total measured wall time. Model creation, loading, route calibration and
warmup are excluded. A2 throughput drifted −0.56% from A1.

| Route | Samples | Mean 200-token time | Tokens/s | Median first token |
|---|---:|---:|---:|---:|
| Old single GPU, A1+A2 | 6 | 10797.71 ms | 18.52 | 36.90 ms |
| Optimized single GPU | 3 | 1633.00 ms | 122.47 | 19.37 ms |
| Optimized forced 2GPU | 3 | 4664.00 ms | 42.88 | 45.55 ms |
| Auto, selected single GPU | 3 | 1639.23 ms | 122.01 | 19.37 ms |

`auto` estimated 2357.9 ms for single GPU and 4748.1 ms for tensor parallelism,
and correctly selected single GPU. Estimates are based on prompt prefill and
eight incremental decode steps near the middle of the requested continuation.
They are route-selection estimates, not substitutes for the full measurements.
Calibration requires its own startup work. Forced `tensorParallel` continues to
execute both lanes regardless of its speed.

## Changes

1. **Persistent per-layer K/V:** prefill once, then append and attend to one
   token on each owning GPU. The cache is FP32 scratch; parameter storage stays
   packed. The cache capacity is limited to the positions needed by the call.
   Per-token frames and the outer `finally` release intermediate values and
   caches, including on a streaming callback exception.
2. **One streaming call:** `--generate-config` keeps the Transformer generation
   call alive while delivering token callbacks. It previously restarted
   generation once per token. `--config ... --generate` shares the same model
   implementation and Arc routing.
3. **Packed one-token GEMV:** read physical BFP8/BF16/FP32 weight payloads
   directly. Low-precision matrix operands retain their BF16 operand rounding;
   the subgroup calculation accumulates in FP32. This avoids decoding every
   weight matrix into a full FP32 scratch buffer for each token.
4. **Fused operands and publication:** decode the small input and bias inside
   GEMV, and publish BF16 output directly when the existing dtype policy allows
   it. Tensor-parallel partial projections remain FP32 until their sum and one
   bias/publication boundary.
5. **Packed embedding lookup:** read only the selected token and position rows,
   add them before one publication, and honor the absolute position offset.
   The reference fallback also adds before rounding.
6. **Small-row normalization:** enable the existing subgroup normalization for
   no-grad inputs below 128 rows. The earlier dispatch used the serial kernel
   for each decode token. Gradient-recording dispatch remains unchanged.
7. **Cache-aware auto selection:** separately measure prefill, incremental
   decode, and any full-window suffix. The two routes can have different cache
   eligibility. The 5% minimum estimated-time improvement is retained.

The five `Inference*` optimization switches in `ArcExecutionOptions` are enabled
by default and disabled in `Reference`; each can be disabled for A/B checks.
Neither `training.transformer.json` nor `generate.json` was rewritten in this
generation optimization.

## Profile evidence and rejected assumptions

Exploratory profiles used greedy sampling and therefore are separate from the
final topK comparison. They identified the following progression:

| Candidate | Generated tokens/s |
|---|---:|
| Original full-prefix single GPU | 18.65 |
| K/V cache | 42.06 |
| + packed weight GEMV | 70.58 |
| + packed embeddings | 71.11 |
| + small-row normalization | 71.79 |
| + fused GEMV operands/publication | 110.21 |

Normalization GPU time fell from about 974 to 145 ms per 200 tokens, but initially
wall time barely changed: submission of many small kernels became the limit.
Fusing GEMV removed input/bias decode and output-publication launches. With fused
GEMV, disabling small-row normalization reduced throughput to 91.97 tokens/s;
enabling it provided a further 19.84% increase in that experiment.

In the final single-GPU run, launches fell from **224760 to 98640** per generation.
The separate final profile attributes about 614.73 ms to fused GEMV, 198.12 ms to
incremental attention, and 145.20 ms to normalization. Event durations overlap
host wall time and must not be added to wall time.

Two GPUs still require host-mediated activation and FP32 partial-result
transfers at every layer. In each optimized forced-TP generation, GPU 0 transfers
27.13 MB H2D / 18.17 MB D2H and GPU 1 transfers 13.57 MB H2D / 27.13 MB D2H.
Both execute kernels. This synchronization cost makes the single-GPU route
faster for the measured workload.

## Memory

The backend's session-cumulative peak native buffer allocation fell from
**4269.40 MiB to 130.77 MiB** for single-GPU generation. These are backend buffer
counters, **not driver-reported dedicated VRAM**. Forced TP peaks were
70.63 / 62.71 MiB. Auto's peaks include both-route calibration and are
266.46 / 88.62 MiB.

All measured repetitions have zero change in live allocated bytes after the
generation call. Integration tests additionally verify both lanes have zero
allocated and cached bytes after session disposal.

## Numerical behavior and tests

GPU regression tests cover Float32, mix16_32, mix8_32 and mix8_16; packed operand
tails, bias/ReLU, FP32 TP partials, prompt prefill, incremental attention, token
callbacks, seeded sampling, stop tokens, zero tokens, context overflow, callback
failure, and repeat/session cleanup. No-grad optimizations are also checked not
to change the gradient-recording dispatch.

The two-layer untied-model tests teacher-force different tokens and compare
every final logit at five steps, for each precision on one and two GPUs. Maximum
observed relative RMS was 1.97301e-7 for Float32, 4.88283e-8 for mix8_32, and zero
for mix16_32/mix8_16 in this fixture. Thresholds are 1e-4 / 5e-3 / 2e-2 for
Float32 / BF16 / BFP8. These are bounded regression cases, not a trained-model
quality evaluation.

Greedy exploratory 200-token outputs matched. Each final topK route reproduces
its own continuation across all measured runs. The old/new topK continuations
**differ at 197 of 200 positions, starting at the first generated token**;
bitwise or seeded-sampling identity across implementations is not claimed.
Full-shape teacher-forced diagnostics fix the baseline token history and compare
the same 91M model at continuation steps 0, 1, 2, 7, 31, 99 and 199:

- Published logits: relative RMS **0.616%–0.992%**, maximum absolute difference
  0.015625, argmax identical at all seven points.
- Top40 membership: 38–40 of 40 IDs match; total variation of the temperature-1
  top40 distribution is 0.000342–0.030186.
- At step 0 the top40 sets are identical, but IDs 4063 and 1148 exchange ranks
  3 and 4. The same RNG draw therefore selects a different ID from the sorted
  cumulative distribution, after which autoregressive input histories diverge.
- Before the final head publication, relative RMS is 0.568%–0.961%, so the
  difference includes earlier reduction/low-precision rounding, not only the
  final logits codec. Argmax also matches at every point before that rounding.

See `benchmark-results/arc-generation-full-logits-r2-20260924.json` for both
logit arrays, ordered top40 values, route options and hashes. These measurements
bound the observed numerical change on this synthetic model; trained checkpoint
quality remains unverified.

Validation artifacts:

- `NNtrain.Core.Tests/TestResults/arc-generation-selected-core-20260924.trx`:
  103 passed, zero failed/skipped.
- `NNtrain.IntegrationTests/TestResults/arc-generation-selected-integration-20260924.trx`:
  22 passed, zero failed/skipped; both real CLI entry points, single/forced TP,
  using temporary small checkpoints and checking artifact preservation.
- `NNtrain.Core.Tests/TestResults/arc-generation-logits-and-cpu-20260924.trx`:
  stricter logit thresholds plus existing GPT CPU tests, 13 passed.
- Release solution build: zero warnings/errors; `git diff --check` clean.

## Boundaries

The cache supports model contexts up to 8192 tokens. Once the window slides,
absolute positional embeddings change for retained tokens, so the remaining
suffix uses full-window evaluation. BFP8 tensor-wide or token-crossing blocks
also use that fallback; eligibility checks model/FFN widths, including TP shard
widths. The current width512/FFN1536/block32 configuration is eligible.

## Reproduce

```powershell
dotnet build NNtrain.slnx -c Release --no-restore
dotnet NNtrain.Benchmarks/bin/Release/net10.0/NNtrain.Benchmarks.dll `
  --arc-generation-probe training.transformer.json NEW-result.json `
  --mode single --tokens 200 --warmup 1 --runs 3 --top-k 40 --temperature 1
```

Baseline switches: `--kv-cache off --gemv off --fused-gemv off
--packed-embedding off --small-row-norm off`. Add `--profile` for the separate
profile, or use `--mode tensorParallel` / `--mode auto`. `--safetensors PATH`
loads an existing checkpoint; outputs must always be new paths.

Raw final artifacts are `benchmark-results/arc-generation-final-{a1,b,a2,tp,auto,profile}-20260924.json`.
`benchmark-results/summarize-arc-generation-20260924.ps1` validates their hashes,
configuration, flags, sampling and run counts and writes the
`arc-generation-final-summary-20260924.json` aggregate.

The teacher-forcing diagnostic was added after the timing runs. Its benchmark
assembly hash differs; Core, Arc and CLI binaries remain the measured versions.

```powershell
dotnet NNtrain.Benchmarks/bin/Release/net10.0/NNtrain.Benchmarks.dll `
  --arc-generation-logit-probe training.transformer.json `
  benchmark-results/arc-generation-final-a1-20260924.json `
  NEW-logits.json --include-logits
```

Measured binaries:

- Core: `4CF77D5FEC19C84A91CDD374EC65A5BACE164194F24781A2086F099607C14D41`
- Arc: `64B9967359071A70CD045022D90015E35B3CD5CFF49264C39E8C99A40015AA03`
- CLI: `6C26B9F70ADE0D6A459EB9199DF7591716B4D24B1779C9DFA34B1857E3418EE0`
- Benchmarks: `6714BE66E9444EBAB92564EECEF28856D5B7FEBE8E9E9146158EC56F21FEA500`
- Training configuration: `E876C55B7BFBE4F0BA5054D0A86070E525C776257E8C260A976A8F4305C1A50C`
