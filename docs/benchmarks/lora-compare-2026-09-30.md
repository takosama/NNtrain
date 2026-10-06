# NNtrain exact / FP16 XMX LoRA comparison — running

Started 2026-09-30 20:50:13 UTC. This is an interim report, not a completed full-epoch quality result.

## Measured calibration

Same freshly initialized all-layer rank-8 adapter, alpha 16, seed 1, LR 0.0001, clip 1, no output-head adaptation. Same QA80 record 1, direct ChatML, 575 tokens including EOS, one complete update. Both use 1 GiB IQ2 projection GPU cache per Arc, no host cache, default 2 GiB training buffer pool. Same model/data/assembly SHA256 values verified in both reports. Only IQ2_S forward precision differs.

| Measurement | exact | FP16 XMX |
|---|---:|---:|
| Update seconds | 44.021015 | 30.7514295 |
| Loss | 2.41714384066918 | 2.41718726899254 |
| Gradient norm | 0.756830919898971 | 0.756881866797603 |
| Arc 0 peak allocated bytes | 8,739,748,536 | 8,772,663,992 |
| Arc 1 peak allocated bytes | 9,744,929,760 | 9,750,807,520 |

Speed ratio: **1.4315x**; update duration reduction approximately **30.14%**. These are single-update measurements, not epoch averages. Loss difference is approximately 0.00004343. Backend peak allocations are not driver VRAM measurements and include cached/retired allocations. Model loading, tokenization, report generation, hashing and checkpoint saving are outside probe update timings.

## Environment and preservation

Current HEAD: `c1878e5` (`Support GGUF LoRA and stabilize Qwen3.5 training and GUI streams`). It was checked rather than assumed from historical information. Windows, .NET SDK 10.0.400 / Release, Ryzen 5700X, RAM 64 GB, two Intel Arc B580 12 GB devices, GT710 display adapter. Arc driver 32.0.101.9030. The full observed environment is in `benchmark-results/lora-compare-20260930/environment.json`.

Preflight found GUI PID 21104 and server PID 31072; `/internal/state` reported `is_loaded=false`. No active CLI training was found. Driver counters reported zero dedicated memory on both Arc LUIDs; the display adapter had approximately 800 MB. No existing processes were stopped or models unloaded.

CPU optimization changes in `Tensor.Qwen.cs`, `TensorStorage.cs`, and new `Tensor.QwenCpu.cs` were preserved. The tracked patch and untracked source copy were saved, and before/after SHA256 lists matched. There are many pre-existing untracked benchmark artifacts. No reset, stash, checkout, commit, push, existing checkpoint overwrite or training-data edit was performed. The original QA80 config, dataset and adapter are separate from this run; their hashes are recorded.

## Full comparison now scheduled in one local runner

Runner PID: **29560**. Started hidden and sequentially; do not start a second comparison. Its current authoritative status is `benchmark-results/lora-compare-20260930/status.json`. Progress appears in `runner.stdout.log` and per-stage logs. A named mutex prevents duplicate instances of this runner; read-only guards before each stage detect a loaded GUI model or another NNtrain CLI/probe process and stop the runner before launching the next stage. These guards do not prevent the user from starting other workloads during a stage.

Frozen binaries are in `benchmark-results/lora-compare-20260930/frozen-bin/`; file hashes are in `frozen-binary-hashes.json`. Both arms use this immutable-by-convention copy; do not edit it during the comparison.

Training source: existing `data/rintya/rintya_qa_80.chatml.jsonl`, 80 records. Mean length 1068.2 tokens including EOS, maximum 3838 (record 77). Context configuration is 3840, preserving the existing QA80 setting; inputs are neither shortened nor skipped. Two complete epochs = **160 updates per arm**. The seed-1 production `epoch-shuffle-splitmix64-v1` order is recorded for both epochs in `inventory.json`. CLI `example=N/80` logs and the final summarizer verify identical orders. All layers, targets, rank, seed, LR, clip, context, data and cache budgets match. Separate fresh adapters are used; FP16 never resumes the exact checkpoint. Each arm saves optimizer-containing `.bin` checkpoints every ten updates and keeps an independent `.bin` snapshot at each epoch. A GGUF-only adapter is not used for resume.

Stages: fresh baseline quality; exact and FP16 calibration of unshortened record 77 (3838 tokens); exact epoch 1 + quality, exact epoch 2 + quality; FP16 epoch 1 + quality, FP16 epoch 2 + quality; order/step-count verification and summary. Epoch 2 resumes the corresponding epoch-1 optimizer state.

A token-proportional estimate from the short calibration is approximately 3.6 hours exact plus 2.5 hours FP16 for updates alone. Allow **6–8 hours total or longer** for long-sequence scaling, startup, checkpoints and quality evaluation; this is an estimate, not a completion promise. The long calibration will refine it. The runner continues on the PC beyond this interaction.

## Quality measurements and remaining limits

Fixed heldout: four existing records from `data/rintya/rincha_qa_300.chatml.jsonl`, selected by ascending line SHA256 after excluding exact training prompts and prompt/response duplicates. Records **115, 100, 171, 221** are frozen for baseline and every epoch. Their token counts are 64, 68, 59, 58. Both trained adapters are evaluated using the **same exact loss path**, with token-weighted mean and per-record values. The evaluator verifies that optimizer step does not change. This small, short heldout panel cannot establish broad or long-response generalization; semantic overlap between source datasets has not been ruled out.

Each quality stage also performs greedy generation with an independent heldout prompt and a fixed Japanese long-explanation prompt, a **4096-new-token cap**, the same inference implementation and EOS handling. Token IDs, decoded text, duration, EOS/cap completion are saved for review. Reaching EOS or the cap does not itself establish coherent or accurate output. Semantic quality review remains required after output is available. Training loss alone will not decide quality.

The fresh baseline quality stage completed: token-weighted heldout loss **3.7859551963457**, optimizer step 0 before and after evaluation. The two generations produced **1137 / 1263 new tokens**, both reached EOS, with generation durations 69.847 / 78.784 seconds. They do not prove 4096-token completion. The baseline model produced a structured long explanation with planning text; detailed factual/semantic review and trained-adapter comparison remain pending. A measurement-only helper correction removed prompt tokens from the generated-token count and decoded response; the original raw baseline report and helper assembly are preserved. Only `QualityProbe.dll` was updated for this correction; all model/training assemblies remain fixed. `quality-probe-v2-hash.json` records the corrected evaluator hash.

**Full epochs, final heldout quality, and generated-text quality are pending.** The real long calibration covers 3838 tokens; dedicated **4096-token training** and **255/256/257 vocabulary-head row-boundary checks** are still pending. No result from these unperformed checks is claimed.

The additional requested 4096-token LoRA speed optimization begins **after this comparison is completed and reviewed**. No baseline implementation changes were made for that later task. Profile the 4096 update, identify the measured bottleneck, make isolated changes while preserving the CPU patch, and compare update wall time, peak allocations, loss, norm and fixed quality against the frozen baseline. Historical evidence points to IQ2_S backward projection, but its current 4096 bottleneck has not yet been measured.

## Reproduction / follow-up

All scripts, JSON configurations, raw calibration reports and optimizer checkpoints are under `benchmark-results/lora-compare-20260930/`. `run-comparison.ps1` records exact commands and refuses existing stage logs/reports. It is already running; running it again is not a resume operation.

Read current state:

```powershell
Get-Content .\benchmark-results\lora-compare-20260930\status.json
Get-Content .\benchmark-results\lora-compare-20260930\runner.stdout.log -Tail 20
```

Single calibration command (use a new report/checkpoint output path):

```powershell
dotnet .\benchmark-results\lora-compare-20260930\frozen-bin\NNtrain.Benchmarks.dll --qwen35-lora-training-probe --model .\models\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf --data .\data\rintya\rintya_qa_80.chatml.jsonl --examples 1 --prompt-mode direct --devices 0,1 --options .\benchmark-results\lora-compare-20260930\exact-options.json --output <new-report.json> --checkpoint <new-adapter.bin>
```

Use `fp16-options.json` for the other arm. Full training uses `frozen-bin/NNtrain.Cli.dll lora --model <same-model> --config <arm-epochN.json>`; only epoch 2 adds `--resume`. Quality uses the provided `QualityProbe.dll evaluate <model> <train-data> <heldout-data> <new-report> <adapter.bin-or-fresh>`.

On success, `full-comparison-summary.json` contains all 320 updates, common ordering, total update seconds, speed ratio and epoch quality artifacts. On failure or interference, `status.json` says `blocked` and `failure.txt` records the reason; later stages are not started. Inspect before resuming; do not silently reuse stale checkpoint state or start competing work. Update this report with measured full-epoch results and reviewed generations before announcing comparison completion or beginning 4096 optimization.
