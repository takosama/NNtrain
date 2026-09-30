# Qwen3.5 GGUF LoRA

`qwen-lora` trains NNtrain adapters on a frozen Qwen3.5 GGUF base. The encoded
quantized matrices stay resident on the selected Arc GPUs. The original GGUF is
not modified. The loader uses the GGUF architecture and tensor types, including
the supported mixtures in IQ2_M files; the filename is not an architecture hint.

The default targets cover Attention, Gated DeltaNet projections and the MLP in
every layer. `includeOutput: true` also trains an adapter on the vocabulary head.
Embedding lookup, normalization weights, convolution weights and the encoded
base matrices are frozen. This checkpoint format belongs to NNtrain; it is not
a PEFT adapter or a merged GGUF.

## Train

From the repository directory, replace the model path with the actual local file:

```powershell
dotnet run --configuration Release --project NNtrain.Cli -- lora --model "C:\models\qwen35-27b-iq2_m.gguf" --config qwen-lora.example.json
```

The supplied example selects Arc devices `0,1`, rank 8, alpha 16, context 64 and
10 optimizer steps. Paths in the configuration are relative to that configuration
file. Change `dataPath` and `adapterPath` for your dataset and output. The example
dataset is a smoke-test sample, not a useful fine-tuning corpus.

The existing `lora` command routes `.gguf` models to this Qwen training path.
`qwen-lora` remains an equivalent command. JSON DRN models keep their existing
`lora` behavior. GGUF generation uses `qwen-gguf --adapter`, as described below.

JSONL records contain `prompt` and `response` strings:

```json
{"prompt":"日本の首都はどこですか。","response":"日本の首都は東京です。"}
```

The command encodes `promptPrefix + prompt + responsePrefix`, then the response,
and appends the GGUF EOS token. Cross entropy is shifted by one token and
supervises **the response and EOS only**, including the first response token.
No BOS is inserted automatically. The example prefixes are Qwen chat markers;
the CLI does not infer or execute a GGUF chat template.

Every example is tokenized and checked before allocating the model on the GPU.
`contextLength` counts the full prompt, response and appended EOS. Overlong
examples fail with their JSONL line number; neither prompts nor responses are
silently truncated. Keep context short for the first run, because training
retains activations and gradients in addition to inference state.

One example produces one optimizer update. New runs shuffle the full dataset
independently for each epoch with a repeatable `seed`, cycling until the total
`maxSteps` limit is reached. The `example=N/count` training log field records
the selected one-based JSONL record for each update; monitoring and audit tools
should use that field rather than assume `step % count` is the record index.
There is no batch accumulation or automatic training beyond that limit. `layers: null` means
all layers; a list such as `[0,1]` selects explicit zero-based layer indices.
Targets are projection suffixes without `blk.N.` or `.weight`; a layer uses the
listed projections that exist in its architecture.

For long IQ2_M examples, the training configuration can select the measured
Arc XMX forward path or a bounded cache of exact IQ2_S base outputs:

```json
{
  "iq2ForwardPrecision": "fp16",
  "iq2ProjectionCacheMiB": 0,
  "iq2ProjectionCachePrioritize": false,
  "iq2GpuProjectionCacheMiB": 1024
}
```

`exact` is the default forward precision and retains the previous numerical
trajectory. `fp16` requires FP16 and XMX support on every selected Arc device;
it keeps GGUF weights quantized in VRAM but rounds inputs and decoded weights
for the frozen IQ2_S forward product. Its small numerical differences make it
a separate resume contract from `exact`. A nonzero cache size stores exact
FP32 base projection outputs in host RAM for reuse in the backward replay.
Prioritization favors projections with wider inputs when the cache is too
small to hold them all. These cache settings do not change the update
contract. The GPU cache instead keeps up to the specified MiB **per Arc** in
VRAM and has no host round trip. It automatically limits its effective size
to preserve the model's VRAM budget. Choose either host or GPU cache, or zero
for neither. Benchmark the options with your own sequence lengths and memory;
the host cache can slow down the FP16 path because of GPU transfers.
The 27B two-Arc measurements and precision checks are in
[qwen35-lora-long-sequence-performance.md](qwen35-lora-long-sequence-performance.md).

## Live HTML loss graph

Training writes the same self-refreshing HTML loss graph used by the regular
transformer trainer. It opens in the default browser **before the first update**
after model/adapter validation. A resumed graph opens with its saved history.
The horizontal axis is dataset epochs: one update out of 512 examples is
`1/512` epoch; 1,536 updates finish epoch 3.

By default, the HTML path is the configuration filename with `.html` replacing
its extension. For example, `traning.transformer.json` produces
`traning.transformer.html`, and `train.json` produces `train.html`. Configure:

```json
{
  "lossGraphPath": null,
  "showLossGraph": true,
  "openLossGraph": true,
  "lossGraphEverySteps": 1
}
```

`lossGraphPath` can specify another `.html` or `.htm` path, relative to the
configuration. `openLossGraph: false` keeps writing HTML without opening a
browser. `showLossGraph: false` suppresses HTML and browser opening for a headless
run. Browser-launch errors are warnings and do not stop training.

Every update's loss is durably appended to the adjacent `.metrics.jsonl` file.
The HTML is replaced atomically every `lossGraphEverySteps` updates (and the
first update), then flushed on exit. The browser refreshes once per second.
On resume, journal entries beyond the saved checkpoint step are removed before
continuing. New runs archive any earlier HTML and journal with a `.previous-...`
suffix; keep the journal with its HTML when moving a run. Resuming an older
checkpoint with no reporting files starts its graph at the next update; losses
from earlier updates cannot be reconstructed. Reporting settings do not alter
the checkpoint's numerical training contract.

## Save and resume

The adapter is saved every `saveEverySteps` updates and at the end. Binary `.bin`
checkpoints contain FP32 adapter values, optimizer state, step, the base GGUF
SHA-256, training contract and a file checksum. Saves atomically replace the checkpoint; the GGUF, dataset,
configuration and adapter paths must be distinct. A fresh run refuses to
overwrite an existing adapter.

To continue, raise `maxSteps` to the desired **total** step count and run:

```powershell
dotnet run --configuration Release --project NNtrain.Cli -- qwen-lora --model "C:\models\qwen35-27b-iq2_m.gguf" --config qwen-lora.example.json --resume
```

New runs shuffle the complete dataset once per epoch using `seed`. The permutation
is deterministic, so a saved step resumes at the same example. Set
`shuffleExamples: false` to retain JSONL order. Older checkpoints with no
`shuffleExamples` setting automatically retain their original JSONL order when
resumed; explicitly requesting a different order is rejected by the training
identity check.

Resume verifies the base GGUF SHA-256, exact dataset bytes and numerical training
contract. Changing learning
rate, targets, rank, seed, context, formatting or dataset requires a new adapter
run. Increasing `maxSteps`, changing the save interval or device placement, or
moving unchanged data/checkpoint files does not change that contract. A mismatch
fails without replacing the saved checkpoint.

## Generate with the adapter

Use the same formatted prompt as training. This example uses PowerShell newline
escapes:

```powershell
$prompt = "<|im_start|>user`n日本の首都はどこですか。<|im_end|>`n<|im_start|>assistant`n"
dotnet run --configuration Release --project NNtrain.Cli -- qwen-gguf --model "C:\models\qwen35-27b-iq2_m.gguf" --adapter checkpoints/qwen35.adapter.bin --devices 0,1 --prompt $prompt --max-new-tokens 64
```

`--adapter` is supported for the `qwen35` architecture. Loading an adapter restores
its saved options and validates the base identity. Generation without `--adapter`
uses the unchanged base model. `--no-stream` keeps the existing complete-output
mode and token-ID reporting.

With `--adapter --max-new-tokens 0`, the model and adapter are still loaded and
validated, then the prompt is returned without generating additional tokens.

## Export an inference GGUF adapter

The dependency-free Python exporter writes a GGUF v3 LoRA adapter for use with
the same base GGUF in a compatible llama.cpp runtime:

```powershell
python tools/export_qwen35_lora.py --checkpoint checkpoints/qwen35.adapter.bin --model "C:\models\qwen35-27b-iq2_m.gguf" --output "checkpoints/lora_tuku_qwen35-27b-iq2_m.gguf" --expected-step 1536
```

`--expected-step` is optional; when supplied, export refuses a checkpoint from
any other update count. The exporter checks the checkpoint SHA-256 trailer,
full base-file SHA-256, architecture, selected target directory, tensor shapes
and finite A/B values. It streams the original FP32 A/B bytes in bounded chunks;
Adam state is omitted. Input files are preserved and the output is published
atomically. Existing output files require explicit `--force`.

The resulting file contains only the adapter, with metadata
`general.type=adapter`, `general.architecture=qwen35`, `adapter.type=lora` and
`adapter.lora.alpha`. The base tensor suffixes are `.lora_a` and `.lora_b`.
GGUF dimensions are `[input,rank]` and `[rank,output]`; the scale is `alpha/rank`.
The base name/hash, completed step and training identity are included as custom
`nntrain.*` metadata. It is not a standalone merged model. Keep the original
`.bin` for NNtrain generation and resuming training; NNtrain's `--adapter` loader
currently accepts its own checkpoint format, not the exported GGUF. Exporting
an output-head adapter from a model with tied embeddings is rejected.

The layout follows llama.cpp's
[LoRA converter](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py)
and [adapter loader](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-adapter.cpp).
Its [Qwen3.5 graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp)
routes DeltaNet QKV, gate, alpha, beta, output and full-attention output through
LoRA; the shared [QKV/FFN graph helpers](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp)
also apply the remaining Attention/MLP adapters. These source checks establish
target coverage, not numerical equivalence between runtimes.

Validation on 2026-09-27: eight CPU fixture tests passed, including non-square
matrix orientation/scaling, corruption, wrong base, wrong step, shape mismatch,
non-finite tensors, truncation and atomic output preservation. Export of the
27B step-4 smoke checkpoint produced 992 F32 tensors (496 pairs), 233,524,416
bytes, rank 8 and alpha 16. The official llama.cpp `gguf-py` reader loaded the
result; every tensor shape and every A/B byte matched the checkpoint. Full
llama.cpp generation with the exported adapter remains unverified.

```powershell
python -m unittest discover -s tools/tests -p test_export_qwen35_lora.py -v
```

`tools/verify_qwen35_lora.py` additionally compares all metadata, shapes and A/B
bytes using the official reader. This optional verification step needs NumPy and
llama.cpp's `gguf-py` package; export itself has no external dependencies:

```powershell
python tools/verify_qwen35_lora.py --checkpoint checkpoints/qwen35.adapter.bin --adapter checkpoints/lora_tuku_qwen35-27b-iq2_m.gguf --expected-step 1536 --gguf-package C:\tools\llama.cpp\gguf-py
```

## Local validation

See [the 27B validation report](qwen35-lora-validation.md) for test results, actual training loss, memory, timing and limits.
