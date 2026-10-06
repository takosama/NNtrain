# 4096 / 8192 LoRA speed settings enabled for CLI

User authorized enabling the adopted candidates from
`lora-speed-4096-8192-2026-10-01.md`. A read-only process inspection found no
NNtrain training/probe process. No learning or GPU measurement was started.

## Effective selection

The standard `lora --model <GGUF> --config <JSON>` / `qwen-lora` CLI path now
resolves omitted settings after tokenization and before model load:

| New run condition | fusedAttentionRows | hostCheckpointBufferHandoff | streamedAttentionTileRows | iq2RollingTranspose |
|---|---:|---:|---:|---:|
| contextLength=4096 and every example has exactly 4096 total tokens including EOS | true | false | 0 | false |
| contextLength=8192 and every example has exactly 8192 total tokens including EOS | true | true | 512 | true |
| Other context lengths, shorter/mixed examples, or empty input | false | false | 0 | false |
| Resume with these values omitted | false | false | 0 | false |

These are defaults only for omitted/null fields. An explicit true/false or tile
value always wins and survives JSON save/load. They are not silently replaced
by this table, including on resume. `useMeasuredLengthDefaults` defaults to true;
set it false to disable automatic selection. Existing explicit flags remain
effective even if that switch is false. Packed attention off disables automatic
row fusion; disabled row fusion prevents the automatic 8192 combination.
GPU gradient retention off or explicit forward-copy handoff disables the
automatic 8192 handoff/tile/rolling combination.

`fusedAttentionOutput`, forward-copy handoff and the rejected online attention
are not enabled. Existing Core execution-option defaults stay unchanged; the
new policy belongs to CLI configuration resolution because Core has no declared
context/data shape at option construction time.

The 4096 direct handoff remains false by default because it slowed the measured
case by 12.59%. The 8192 combination corresponds to the measured 7.68% one-update
time reduction. The 4096 row-fusion confirmation was 3.60% shorter in the same
binary. No performance percentages are guaranteed for other datasets/settings.

The existing IQ2 precision setting remains exact unless JSON explicitly says
`"iq2ForwardPrecision":"fp16"`; the speed evidence used FP16/XMX. Cache size,
pool size, devices, rank, seed, model/data and existing saved configs were not
rewritten. Thus activation with exact or different cache/data conditions is not
a verification of the reported speed or long-term numerical equivalence.

For deliberate explicit selection, use the following fields in the existing
configuration (do not start or change a resumed run merely to select them):

```json
// 4096 measured candidate
"fusedAttentionRows": true,
"hostCheckpointBufferHandoff": false,
"streamedAttentionTileRows": 0,
"iq2RollingTranspose": false
```

```json
// 8192 measured candidate
"fusedAttentionRows": true,
"hostCheckpointBufferHandoff": true,
"streamedAttentionTileRows": 512,
"iq2RollingTranspose": true
```

Save the explicitly selected effective values when resuming a newly optimized
run: omitted values on `--resume` retain the legacy off behavior. This policy
protects old saved runs and does not infer numerical-path choices from a
checkpoint. JSON serialization preserves null/unset rather than freezing an
automatic choice into every configuration. The CLI prints all four resolved
values before loading the model so selection is visible.

## Ordinary use and validation

The inspected `NNtrain.Gui` is inference-only; no Qwen LoRA training setting UI
was found. This change applies to the actual CLI training/save-load path, not
inference or raw Core callers. Existing `traning.rintya-qa80-lora-2epoch.json`
has contextLength=3840 and remains untouched/outside automatic selection.
Existing JSONs with explicit false remain off by design.

Changed files:

- `NNtrain.Cli/Configuration/QwenLoraTrainingConfiguration.cs`
- `NNtrain.Cli/QwenLoraCommand.cs`
- `NNtrain.IntegrationTests/QwenLoraMeasuredLengthDefaultsTests.cs` (new)

19 CPU tests passed (12 new cases plus 7 existing default/resume identity cases).
They cover exact lengths, neighboring lengths, shorter/mixed/empty input,
resume, global disable, explicit false/zero, explicit enabled saved configs,
packed-attention dependency and unchanged exact precision/transpose8.
CLI Release build succeeded with zero errors/warnings. Protected preexisting CPU
source hashes match. No push, upload, learning restart or GPU test was performed.

Evidence: `benchmark-results/measured-length-defaults-20261001/`.
Long-term quality, heldout quality and epoch throughput remain unverified; no
new claim is added to the source benchmark report.
