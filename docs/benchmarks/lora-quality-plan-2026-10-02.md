# LoRA numerical/quality comparison: preflight and time estimate

Read-only process inspection on 2026-10-02 found no NNtrain learning/probe or
active old comparison runner. Old PID29560/stopper2836 do not exist. Archived
status is blocked after the earlier intentional switch to speed work; only three
exact CLI updates were logged. Full epochs and trained-arm quality were not done.
The old runner has not been restarted. No new GPU work or QA142 training started.

HEAD remains c1878e5, but current code is dirty and has later improvements. It is
not equivalent to the archived baseline merely because HEAD matches. A current
tracked patch, 38 source hashes and 19 untracked-source copies are saved under
`benchmark-results/lora-quality-plan-20261002/`. All three protected CPU hashes
still match. Existing frozen binaries/checkpoints/logs are unchanged.

## Time evidence

Historical same-binary exact vs FP16 one-update records:

| Total tokens including EOS | exact seconds | FP16 seconds |
|---:|---:|---:|
| 575 | 44.021015 | 30.7514295 |
| 3838 | 400.111940 | 275.825305 |

Saved synthetic4096 FP16 controls were about301seconds; older packed/gradient
combination about295seconds; later row-fused confirmation about281seconds.
There is no equivalent current-code exact4096 measurement yet. Approximate
exact4096 planning budget420–460seconds/update comes from the3838 result and
is an estimate, not measured speed. Nonlinear memory/thermal behavior may change
it. Historical fresh quality took about165seconds including two EOS-terminated
generations (1137tokens/69.85seconds,1263tokens/78.78seconds) and four tiny
heldout losses. It did not generate4096tokens. At that observed ~16tokens/s,
a single4096-new-token-capped generation can take about4.3minutes per arm.

Two whole80-record epochs in both precisions require roughly6–8hours or more
with quality and overhead. That plan is excluded from the120-minute window.

## Recommended reduced scope, estimated90–110minutes

1. Build and freeze one helper plus one current Core/Arc/Runtime/CLI binary set
   for all conditions. CPU-validate fixture/options/vector-metric comparator.
2. Four fresh arms: control-exact, control-fp16, latest4096-exact,
   latest4096-fp16. Each performs two optimizer updates of the same synthetic
   record77-derived4096-token fixture, same seed1/order77,77, rank8,alpha16,
   LR0.0001,clip1, all12targets/no output adapter, devices0/1, GPU cache1024MiB,
   pool2048MiB, transpose8, kernel timing off. No per-arm resume from another
   precision or old checkpoint.
3. Control disables GPU gradient retention/packed scores/row fusion. Latest4096
   enables these three. Direct/copy/async handoff, streamed tiles, hybrid cache,
   output fusion and rolling transpose are explicitly off. Separate saved-sep30
   options enable only the older first two optimizations if that narrower
   attribution is selected. Do not claim an individual optimization's effect
   from the combined latest4096 comparison.
4. Save finite raw gradient vectors outside update timing and compare max/mean
   absolute difference, relative L2 and cosine (with explicit zero-norm rules),
   plus loss/norm and adapter states. Existing internal ReadGradients supports
   this without modifying production Core. The old probe only recorded a norm:
   vector capture/comparison instrumentation still needs preparation. Second
   update matters because initialized B=0 makes initial A gradients trivially0.
5. Evaluate every arm using one explicitly fixed common exact loss path and the
   archived heldout records115/100/171/221; verify optimizer step is unchanged.
   Greedy-generate one identical fixed long Japanese prompt per arm with4096
   new-token cap/EOS, save actual token IDs/text, completion cause and duration,
   then review coherence, repetition, instruction adherence and output endings.

Budget: updates45–55minutes; evaluation/generation roughly15–25minutes (longer
if throughput deteriorates); helper preparation, saving/vector comparisons and
review/report roughly25–30minutes. These sum to an approximate90–110minutes,
not a completion guarantee. Before each stage check remaining time and avoid
starting work that cannot fit. If calibration is slower, reduce equally to one
update per arm and explicitly mark initial-A-gradient limitation, or reduce
generation prompt count/cap equally. Never silently claim a full epoch or4096
generated-token completion when EOS occurs early. Stop launching new stages
before120minutes and preserve partial results rather than extending silently.

All quality limits remain: four heldout examples are only58–68tokens, semantic
overlap is not excluded, two synthetic updates do not establish convergence,
full-epoch quality, QA142 quality or generalization. Full4096-token generation
is only established if actually reached, and reaching the cap alone is not
semantic quality. Prior255/256/257 boundary tests are preservation evidence,
not an exact-vs-FP16 long-run quality result.

## Frozen CPU artifacts and current limits

`fixture4096.json` preserves4096token IDs, response start24,4072supervised tokens,
4095input tokens and int32LE SHA256. QA80 and heldout file hashes still match
their archived values. Model SHA is archived expected identity; fresh runtime
verification remains mandatory before measuring. Six explicit option files
cover control/saved-sep30/latest4096 × exact/fp16. `heldout-panel.json`, historical
measurements, source snapshot and status are in the same new directory.

No current GPU comparison, full gradient delta, trained heldout result or new
generation-quality result is claimed. The estimate is the requested first
report; long stages have not begun.
