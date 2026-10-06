# 4096-token LoRA optimization — 2026-09-30 UTC

## Later user-authorized default change (2026-10-01 UTC)

Current source defaults now enable both `hostCheckpointGpuGradients` and `packedAttentionScores` in CLI configuration, and `TrainingHostCheckpointGpuGradients` / `TrainingPackedAttentionScores` in the Core API. Omitted keys enable the feature on the next configuration load with the updated code. Existing saved explicit `false` takes precedence and survives serialization/reload; existing configuration files were not rewritten. Set both keys explicitly false to disable. IQ2 precision remains exact and transpose remains8 by default. Running learning/GUI processes were not restarted; GPU benchmarking was not run for this follow-up. Long-term numerical/heldout quality remains unverified.

Release build of IntegrationTests (including CLI/Core dependencies) passed with zero errors/warnings. Configuration roundtrip/default/explicit-off and resume-identity regressions: **7/7 passed**, no failed/skipped. Test evidence: `benchmark-results/lora-speed-4096-20260930/test-results/default-on-configuration.trx`. Existing CPU edits remain hash-identical.

The benchmark narrative below is the historical pre-default-change snapshot. Frozen binaries and original benchmark option files were preserved. New explicit control/combined x exact/fp16 option files for future comparisons are in `benchmark-results/lora-speed-4096-20260930/default-on-followup/`; every file specifies both feature booleans, forward precision, transpose8 and cache/pool settings. Use explicit booleans rather than depending on defaults when comparing tomorrow. None of these future comparisons has been started or scheduled.

This follow-up changes only the two existing initializer/comment locations and adds `NNtrain.IntegrationTests/QwenLoraOptimizationDefaultsTests.cs`; no parser change is included in this default-change stage.

Measured outcome: combined opt-in GPU-gradient retention and packed causal attention reduced mean update time from **301.30686285 to 294.67669685 seconds** (**2.20047% shorter**, **1.02250x throughput**, 13.59080 to 13.89659 input tokens/s), with two fresh-start samples per arm. This is a synthetic fixed-size fixture under existing FP16/XMX forward, not all-epoch or natural-quality validation.

## Environment and fixed conditions

Ryzen5700X, 64GiB RAM, ArcB58012GB x2, GT710 display; Arc driver32.0.101.9030, .NET SDK10.0.400/runtime10.0.11, Windows26200. HEAD c1878e5 plus preserved pre-existing CPU AVX edits and the separately recorded LoRA changes. Frozen binary hashes identify actual builds.

Model Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf, SHA256676db8ccd38035117e705858fd591964005cb0a921b39255b0691d8339a8dfe2; QA80 data SHA256cc1cad3d2f69c5de5667d88655b3dfd5582ae2b466c8960030bd95a05db5f93c. Source record77 response tokens cycle to4096 total tokens including EOS:4095 input rows,24-token prompt,4072 supervised tokens. Original dataset unchanged. Same full token IDs, seed1/rank8/alpha16/LR0.0001/clip1/weight-decay0, all layers, same12 targets, output adapter excluded, devices0/1, no resume/one fresh update per arm, direct prompt. Single-example fixture has no shuffle difference. GPU projection-cache budget1024MiB captured/reused0 in host fallback; buffer pool2048MiB; transpose8 except explicit rejected trials.

## All trials

| Trial | Seconds/update | Input tokens/s | Backend allocation peaks, device0/1 (GB) | Decision |
|---|---:|---:|---|---|
| Initial profile attempt | — | — | — | Startup failed before GPU; fixed internal diagnostic access; logs preserved |
| Baseline profile v2 |299.621364|13.667250|12.716856/12.797814|Profile evidence only; timing collection enabled |
| Baseline |301.009778|13.604209|12.716856/12.797814|Control, timing collection disabled |
| GPU-gradient only |295.8829068|13.839934|12.716856/12.797814|Keep opt-in; one pair,1.70% shorter |
| IQ2 transpose16 |431.2128922|9.496469|12.716856/12.797814|Reject,43.26% longer vs initial baseline |
| Packed attention only |298.6892867|13.709899|11.206865/11.205661|Keep opt-in for allocation reduction;0.77% speed difference inconclusive |
| Baseline repeat, OS sampler |301.6039477|13.577408|12.716856/12.797814|Control;0.1974% longer than first |
| Combined |294.807944|13.890399|11.206865/11.205661|2.2533% shorter vs sampled control |
| IQ2 transpose4 |326.4000518|12.545954|12.716856/12.797814|Reject,8.2214% longer vs sampled control |
| Combined repeat, OS sampler |294.5454497|13.902778|11.206865/11.205661|Modest benefit reproduced; retain opt-in |

All completed4096 arms report finite loss1.285513789891044 and gradient norm0.2326987973948727, matching each other. This does not establish long optimizer-trajectory or heldout quality.

## Hypotheses, changes and findings

Profile kernel sums across both GPUs: IQ2 transpose53.46s, FP16 IQ2 forward34.29s, Q4 forward33.90s, attention scores25.96s. These sums and transfer/allocation counters are not independently additive wall-time attribution.

GPU-gradient retention avoids repeatedly downloading/reuploading layer gradients in host-checkpoint fallback, using on-device FP32 copies and host staging at device boundaries. H2D and D2H each decreased5,283,532,800 bytes/update in its isolated trial, with unchanged peak allocations and identical reported loss/norm. Small one-pair1.70% timing benefit was provisional.

Packed attention uses seven separate FP32 kernels and triangular time/head storage for causal score/probability entries. Dense kernels remain intact. Backend peaks decrease by1,509,991,392 /1,592,152,704bytes. Speed alone changed only0.77% in one pair.

The combined path reproduced a modest2.20% mean gain. Isolated row16/row4 tuning worsened speed, consistent with register-pressure versus decode-reuse tradeoffs; this is a plausible explanation, not a measured register-spill attribution. Both were rejected. Default8 remains.

## Allocation accounting versus Windows GPU memory

Windows per-process dedicated/shared counters were sampled every~10seconds. LUIDdf96/e427 are retained as counter identities and were not independently mapped to Arc indices. Baseline repeat dedicated maxima10.217447/11.509821GB; packed-only10.334532/11.429626GB; combined first9.999933/11.470762GB. Shared maxima0.486666GB per LUID baseline,0.430043GB packed,0.427946GB combined. These are observations with different phase coverage, not guaranteed instantaneous physical peaks. Original baseline/gradient/row16 lacked OS sampling; packed-only sampling began after loading. The backend1.5GB reduction is not established as a1.5GB real-use VRAM reduction. Real-use observations are broadly similar.

## Validation and process/source preservation

Boundary batch completed at22:38:18UTC;6/6 fresh cases. Supervised rows255/256/257 correspond to total tokens279/280/281. Shapes and finite loss/norm verified. Baseline and combined match exactly for each pair:
| Supervised rows | Loss | Gradient norm |
|---|---:|---:|
|255|1.8872672322128494|1.0676777686089434|
|256|1.892988596111536|1.061434442283864|
|257|1.887259925244365|1.064418619724619|

Attention/checkpoint regressions **27/27 passed**, including independent reference/finite-difference gradients, sequence1/129 and future-gradient isolation. Transpose vector bit/tail/guard regressions **24/24 passed**. Zero failed/skipped in both groups. Final Release builds of SpeedProbe (including CLI/Core/Arc dependencies) and Core.Tests completed with **zero errors/warnings** after reconnection. git diff --check passed. Pre-existing CPU source hashes for Tensor.Qwen.cs, TensorStorage.cs and Tensor.QwenCpu.cs match the before snapshots; detailed final records: cpu-preservation-final.json and boundary-sanity-result.json.

Read-only process inspection after reconnection found no active own SpeedProbe/QualityProbe/runner/memory-monitor/comparison processes; PID41424 was gone and status/log confirmed normal completion. No shutdown/kill was needed after reconnection. Existing user training was not stopped. Wi-Fi interrupted observation, not these computations. Long numerical/heldout quality remains deferred.

The final UTF8 read of the preserved generation-helper source found zero replacement characters and the expected Japanese 約 codepoint32004 preceding2000; console display mojibake was not treated as a source corruption. Verify identical prompts for future quality work anyway.

## Enable/disable (current defaults are on)

In an existing CLI qwen-lora configuration, enable:
```json
{
  "hostCheckpointGpuGradients": true,
  "packedAttentionScores": true
}
```
These are a fragment to merge into an existing valid configuration, not a complete training config. Disable by explicitly setting both false; omission now enables them. Saved explicit false values take precedence over defaults. Default IQ2 forward remains "exact"; default transpose remains8. To match today's measured precision separately set "iq2ForwardPrecision":"fp16"; this has not been validated for long-term quality today. Core API equivalents: TrainingHostCheckpointGpuGradients /TrainingPackedAttentionScores. Probe options use those exact Core property names.

The existing GUI has no new switch added; use CLI configuration for explicit opt-in. No user training configuration was rewritten and no production training was started. CLI contextLength4096 is a validation cap; it does not cycle/pad natural examples into this synthetic fixture.

## Local reproduction and tomorrow's comparison

Artifacts: benchmark-results/lora-speed-4096-20260930/. Frozen baseline-bin/, gpu-gradient-bin/, packed-attention-bin/, hash manifests, JSON reports, logs, per-run optimizer-containing .bin checkpoints, test-results/, source snapshot, patches and CPU preservation checks are local.

Inspect existing PID/status and GUI/other training before each run. Use new unused output names. From repository root, sequentially:
```powershell
pwsh -NoProfile -File benchmark-results/lora-speed-4096-20260930/run-one.ps1 -RunName NEW-baseline4096 -Options baseline-fp16-options.json -ProbeDir baseline-bin
pwsh -NoProfile -File benchmark-results/lora-speed-4096-20260930/run-one.ps1 -RunName NEW-combined4096 -Options combined-fp16-options.json -ProbeDir packed-attention-bin
```
run-one.ps1 has a today-only22:53UTC start guard. Future explicitly authorized work must deliberately revise that guard and inspect existing outputs, not restart it blindly. The boundary runner similarly has a22:48UTC start reserve.

Detailed exact/FP16 and heldout/long-generation comparison is deferred to2026-10-02JST. It is not scheduled or started here. Use the same frozen optimized binary for both precision arms and copied option files differing only in TrainingIQ2Fp16XmxForward=false/true, with the two new flags matched between arms; use the same base/model/data/seed/tokens/adapter state/layers/context/cache and explicit order. For before/after implementation comparisons also retain the original frozen baseline. Fresh-start comparisons and resumed trajectories are separate experiments: both resumed arms need the same optimizer-containing .bin and example sequence; GGUF adapters alone cannot resume optimizer state. Repeated probe example IDs continue training, not reset-state repetitions.

The preserved QualityProbe v2 helper and baseline panel in benchmark-results/lora-compare-20260930/ provide four fixed heldout records115/100/171/221 (source SHA selection excludes identical prompt/pair, not semantic overlap), common exact evaluation, and generation with prompt IDs excluded from output counts. Reuse the frozen identities and verify actual UTF8 generation prompts before running. Check evaluation leaves optimizer step unchanged. Long generation must explicitly report generated count/EOS/limit/error, inspect long-response quality and completion, and compare both arms with the same greedy prompt/settings. Today's old baseline generations ended at1137/1263 new tokens; they are not a4096-token completion claim. No new quality results today.

Probe snapshots use SaveLora without the production CLI training identity. They are intended for the probe's --resume and common evaluation, with the JSON fixture identity checked explicitly. Production CLI --resume validates its dataset/configuration identity; do not treat a synthetic probe snapshot as a natural-data CLI resume checkpoint. Use paired production checkpoints with the same valid training identity when evaluating that trajectory.

The old full QA80-two-epoch runner was superseded by the latest2h/speed-first instruction. Only its verified own CLI child was stopped; logs and calibration checkpoints preserved, three updates completed before stop, no ten-step checkpoint produced. Do not restart its stale full plan or call all epochs completed. Select tomorrow's duration/sample plan explicitly.

## Limits and changed files

Single synthetic source fixture, two baseline/combined samples, no statistical confidence interval, no OS/driver cache flush, no thermal attribution, load/tokenization/diagnostics/JSON/checkpoint saving excluded from update timing. Minimal loss/norm/regression checks do not establish optimizer trajectory, heldout quality or long generation quality. FP16 speed under this fixture cannot be generalized to exact precision or natural mixed-length epochs. No external upload/Library save, commit/push/publication, model/data/checkpoint overwrite, network setup or stopping another user's training.

Changed implementation files:
- NNtrain.Core/Modules/Qwen35ExecutionOptions.cs
- NNtrain.Core/Modules/Qwen35QuantizedModel.Lora.cs
- NNtrain.Core/Modules/Qwen35TrainingAttention.cs
- NNtrain.Arc/Kernels/qwen35_train_attention_packed.cl (new)
- NNtrain.Arc/ArcExecutionLane.cs
- NNtrain.Arc/NNtrain.Arc.csproj
- NNtrain.Cli/Configuration/QwenLoraTrainingConfiguration.cs
- NNtrain.Cli/QwenLoraCommand.cs
- NNtrain.Core.Tests/Qwen35LoraCheckpointParityTests.cs
- NNtrain.Core.Tests/Qwen35TrainingAttentionTests.cs

Benchmark harness/log/report files reside under the two dedicated benchmark-results directories and docs/benchmarks/. Pre-existing Tensor.Qwen.cs, TensorStorage.cs and Tensor.QwenCpu.cs edits are outside the implementation change and must remain preserved.
