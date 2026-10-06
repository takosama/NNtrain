# 4096-token LoRA: four-arm numerical and quality comparison

All requested reduced comparisons completed. No symmetric scope reduction was
needed: four fresh arms, two updates per arm, four fixed heldout examples and one
4096-new-token-capped greedy generation per arm. The existing detached runner
started at **07:23:26 UTC** and completed all nine stages at **08:26:41 UTC** on
2026-10-02, before the **09:06 UTC** deadline. No comparison GPU process remains.
QA142 production training was not started. No existing user learning was stopped,
and no files were uploaded or published.

Artifacts: `benchmark-results/lora-quality-20261002-0718/`. Full settings,
reproduction commands and measurement definitions are in that folder's README.
This report supersedes the preflight-only quality plan for the completed reduced
experiment; it does not claim full epochs or long-term quality verification.

## Conditions and frozen identity

Ryzen7 5700X (8cores/16threads), nominal64GB RAM (OS reports68640645120bytes),
two Intel Arc B58012GB GPUs, devices0/1, driver32.0.101.9030; GT710 is the display
device. Windows10.0.26200, .NET SDK10.0.400/net10.0. Detailed environment evidence
is saved locally. The CIM AdapterRAM field is32-bit and is not used to determine
the B580 capacity.

HEAD is c1878e5b98710e5bb413fff47215bd208380898f, but the source is dirty and has
later changes. HEAD alone does not identify this build. One helper/Core/Arc/Runtime
binary set was frozen and rehashed before every stage. Production code was not
edited for this comparison. The helper build passed with zero warnings/errors;
fixture/options and identical/opposite/zero metric self-tests passed.

| Identity | SHA256 |
|---|---|
| Base `Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf` (10173451200bytes) | `676DB8CCD38035117E705858FD591964005CB0A921B39255B0691D8339A8DFE2` |
| Frozen Core.dll | `C33765F94F3D5E1E64F24342E602C85D3F67394A983C016EC65C2B87E5A9E8A3` |
| Frozen measurement helper.dll | `625D4D64C41124AD45ABEA3B2C83B760848A89B2B53AE5B24F97601811A46CC2` |
| Original QA80 ChatML JSONL | `CC1CAD3D2F69C5DE5667D88655B3DFD5582AE2B466C8960030BD95A05DB5F93C` |
| Heldout QA300 ChatML JSONL | `B221C325C5BC3BABDBED5DE0A4CA7C43CBF55D828D5580FE5B7EC0DC12720740` |
| Fixed int32LE token IDs | `C65762F6124C85B9D12110DEC31BE3D696B9A710C5F235B4A27AA657149BBBE2` |

All arms use the saved synthetic cyclic4096-token fixture derived from example
77/80, order77,77, response start24,4095input tokens and4072supervised tokens per
update. Original data is unchanged. Rank8,alpha16,seed1,LR0.0001,weight decay0,
clip1, all12projection targets, all layers, no output adapter. Each arm starts
fresh and continues its own optimizer for update2; no cross-arm checkpoint resume.
GPU IQ2 cache1024MiB, CPU cache0, pool2048MiB, transpose8, kernel timing off match.

Control explicitly disables GPU gradient retention, packed attention scores and
attention row fusion. Latest4096 explicitly enables these three. All handoff,
copy/async, streamed tiles, hybrid checkpoint, output fusion and rolling-transpose
features are explicitly off. Within each bundle exact/FP16 differs only in
`TrainingIQ2Fp16XmxForward`. `option-diffs.json` records these differences.
FP16 here means IQ2 forward projection using FP16/XMX; LoRA parameters,
gradients and Adam state remain FP32. It is not an all-FP16 optimizer comparison.
The experiment measures the three-feature bundle, not each feature's contribution.

## Training speed and numerical results

Only TrainLora, including its completion synchronization, is timed. Loading,
gradient/state downloads, hashing and checkpoint writes are outside update time.
Input throughput counts4095tokens, not the4096-token fixture's EOS-inclusive size.

| Arm | Update1 seconds | Update2 seconds | Mean seconds | Aggregate input tokens/s | Loss1 | Loss2 |
|---|---:|---:|---:|---:|---:|---:|
| control-exact |430.131358|433.858407|431.994883|9.4793|1.285526978|1.217759760|
| control-fp16 |303.310407|301.438240|302.374324|13.5428|1.285513790|1.217897811|
| latest4096-exact |416.050689|416.393507|416.222098|9.8385|1.285526978|1.217759760|
| latest4096-fp16 |282.331462|281.885559|282.108510|14.5157|1.285513790|1.217897811|

The bundle reduced mean time **3.651% for exact** and **6.702% for FP16**.
FP16 was **1.429x** the exact speed with optimizations off (30.005% less time),
and **1.475x** with them on (32.222% less time). These are two sequential updates,
not confidence intervals or a full-epoch performance estimate. Kernel/OS caches
and clocks were not reset or controlled. Arm order was control-exact,
control-fp16, latest4096-exact, latest4096-fp16.

Within a precision, optimizations on/off had **zero numerical difference in all
saved gradients and all A/B/Adam m/v state values, at both updates**. The optimizer
checkpoints also matched byte-for-byte. This equality is limited to this input,
seed, binary and two-update trajectory.

| Precision | Update1 checkpoint SHA256 (off=on) | Update2 checkpoint SHA256 (off=on) |
|---|---|---|
| exact |`074F430FA16412490A197AEF9624E57E0FE91D21524451942335E1A191A90E25`|`8525981C20792D844228F41F80EA8C79AF4DDB3AE1D53CF575D24D4FCC63B959`|
| FP16 |`0B986473E08883D4CE919D535B467EEBDC9DEDAA2408FFA16EA6063014113BBC`|`087F14A4616A5FB957D13DEC27211E3B710A5EE625CC4D5CF8E52C48ED4F4A55`|

Exact versus FP16 is **not numerically identical**. Per-update gradient norms are
0.2317242304/0.2192044179 (exact) and0.2326987974/0.2182833842 (FP16), a
+0.421%/-0.420% change. Loss differences FP16-minus-exact are
-0.0000131879 and+0.0001380514. Norms alone understate vector differences:

| Exact-reference comparison | Update | Max absolute difference | Mean absolute difference | Relative L2 | Cosine |
|---|---:|---:|---:|---:|---:|
| All gradients |1|0.000456025|2.66534e-7|3.5588%|0.999378209|
| All gradients |2|0.001474368|6.56849e-7|8.9744%|0.995964911|
| A gradients |1|0|0|0 (zero vectors)|undefined|
| A gradients |2|0.000216657|4.82668e-8|1.7050%|0.999867355|
| B gradients |2|0.001474368|1.18554e-6|9.5718%|0.995408956|
| All A/B/m/v state |1|0.000199974|2.53279e-7|0.2406%|0.999997106|
| All A/B/m/v state |2|0.000399625|9.95795e-7|0.3361%|0.999994350|
| A parameters |2|0.000148721|1.66315e-6|0.1506%|0.999998866|
| B parameters |1|0.000199974|1.37011e-6|15.7040%|0.987669296|
| B parameters |2|0.000399625|4.00707e-6|10.4804%|0.994520885|

Global state metrics include unchanged/larger A parameters, so they must not be
used to hide the B differences. Initial B=0 makes first-update A gradients zero;
their initial equality is trivial. Update2 compares diverged optimizer
trajectories, not a precision substitution at exactly the same updated parameters.
Every captured gradient/state value and train/heldout loss was finite; shapes and
ordered layouts matched (58363904gradient values,175091712state values each).
Component norm/count aggregation passed, and independently streamed gradient
norms matched the model's reduction to a maximum relative difference2.365e-9.
Full m/v component statistics are in `component-comparison.json`.

## Memory

| Arm | Backend peak Arc0/1 GiB | Windows dedicated peak LUID e747/eb94 GiB |
|---|---|---|
| control-exact |11.8435 /11.9189|9.6350 /10.4627|
| control-fp16 |11.8435 /11.9189|10.7761 /10.7397|
| latest4096-exact |10.4354 /10.4361|9.6190 /10.7386|
| latest4096-fp16 |10.4372 /10.4361|9.6297 /10.3948|

Backend allocation peaks fell about1.41/1.48GiB. Windows process peaks varied:
the exact run's second LUID increased, while both FP16 LUID peaks decreased in
this run. **A general/reproducible physical VRAM reduction is not established.**
Counters are sampled approximately every10seconds and include loading/diagnostics;
short peaks may be missed. Windows LUID order has not been mapped to Arc index
order. Backend allocation accounting is not driver residency. All training runs
had about0.340GiB sampled shared memory per B580 LUID.

By the second update's counter snapshot, the bundle reduced cumulative upload
and download counters by5199667200bytes on Arc0 and5367398400bytes on Arc1, in
each direction. Loading/intermediate diagnostic traffic is included in these
counters. This supports a traffic/capacity-saving explanation, but does not
isolate which of the three features caused the measured wall-time difference.

## Heldout and generation

All arms were evaluated by the **same explicitly configured exact loss path**.
Records115/100/171/221 have64/68/59/58tokens and147supervised tokens total. Their
hashes/token lengths were checked. Evaluation left optimizer step2 unchanged.

| Record | exact loss (off=on) | FP16-trained adapter loss (off=on) |
|---:|---:|---:|
|115|3.379555273|3.379349049|
|100|3.760246350|3.760062494|
|171|3.963237322|3.963699814|
|221|3.970461421|3.970418081|
| token-weighted |**3.744065965**|**3.744043719**|

Weighted FP16-minus-exact difference is-0.0000222457. This is a small observation
on four tiny examples, not proof of quality equivalence or superiority. Exact
prompt/pair training overlap was excluded during selection; semantic overlap was
not excluded. No same-current-binary fresh-adapter quality baseline was measured,
so historical fresh results are not used to claim training improved heldout.

Each arm greedily generated the identical Japanese prompt requesting about2000
characters on urban vegetable gardening with headings, three examples and
cautions. All four generated **identical token IDs and decoded text**, with
**2251new tokens and natural EOS**, below the4096cap and context limit.

| Arm | Generation seconds | New tokens | End |
|---|---:|---:|---|
| control-exact |158.090359|2251|EOS|
| control-fp16 |158.122090|2251|EOS|
| latest4096-exact |158.140554|2251|EOS|
| latest4096-fp16 |158.141863|2251|EOS|

The article stayed on topic, included three concrete examples, six headings,
cautions and a conclusion, with no runaway repetition or abrupt truncation
observed. However the final article is **1368characters** including Markdown and
excluding the reasoning/EOS/outer whitespace: **31.6% shorter than2000**. Raw
output contains an English `<think>` block and a Japanese draft repeated in the
final article; this is not included in its length and is not claimed to be an
inference-engine defect. The unspecified Tokyo ward's500-yen allotment and
roof/watering claims are unsourced/unverified. Structural completion does not
establish factual correctness. `generation-review.json` records this review.
The common decoded-text file SHA256 is
`9714AD485FEB374081024C551D76B17F168E752710ADB26C8BB6751E9605B0E3`.

## Reproduction, preserved work and limits

Use the frozen binary and explicit four option files, fixture and panel in a
**new** output directory; follow its README commands. `step-N.bin` contains the
optimizer; `grad-N.bin`/`state-N.bin` are diagnostic vectors, not resume files.
GGUF adapters alone cannot resume optimizer state. Source patch/hash snapshots,
all eight optimizer checkpoints, raw vectors/layouts, full generation IDs/text,
stage stdout/stderr, Windows samples and summaries remain on this PC.

Protected CPU files retained their original SHA256 values:
Tensor.Qwen.cs `7A3062A79358EEC6E5F112281C7FAE3F5709393E8DD3904DEA8D5B98B799F4FC`,
TensorStorage.cs `1A4A80B575986108CB34B46552FA05C17008D9475D34BF51FCD25B89BE47217B`,
Tensor.QwenCpu.cs `F0392CE982AC52B7F036A165AB3B26F5C8C1E507D417B230CB9E4F858A7F5A9A`.
Existing data/checkpoints and other dirty edits were preserved. This task adds
measurement/report artifacts; it makes no new production optimization change.
The previous user-authorized defaults are unchanged: GPU gradient retention and
packed scores on, IQ2 precision exact, transpose8. CLI measured-length policy may
enable row fusion on eligible fresh homogeneous4096 examples; these explicit
comparison options do not depend on that policy or saved defaults.

**Not verified:** full epochs/convergence, representative mixed-data or QA142
quality, other seeds/shuffle/context/rank/layers,4096-generated-token completion,
long-run FP16 equivalence, broad factual accuracy, statistically reliable speed
or driver-memory benefit. Four conditions reused one input and one greedy prompt;
there is only one unique generated article. FP16 remains a numerical approximation
despite this prompt's token equality. The bundle's exact preservation is limited
to the measured two-update fixture. No additional learning was launched.

Final integrity checks passed: all17frozen files and388observed Core/Arc/Runtime
source files retained their hashes; original model/data hashes and all three
protected CPU hashes matched. All eight train/quality reports were completed;
all ten runner/stage stderr logs were empty. `preservation-final.json`,
`arm-completion-check.json`, `completion-check.json`, `component-sanity.json`
and `gradient-norm-sanity.json` preserve these checks.
