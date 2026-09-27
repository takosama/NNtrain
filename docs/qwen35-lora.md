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
dotnet run --configuration Release --project NNtrain.Cli -- qwen-lora --model "C:\models\qwen35-27b-iq2_m.gguf" --config qwen-lora.example.json
```

The supplied example selects Arc devices `0,1`, rank 8, alpha 16, context 64 and
10 optimizer steps. Paths in the configuration are relative to that configuration
file. Change `dataPath` and `adapterPath` for your dataset and output. The example
dataset is a smoke-test sample, not a useful fine-tuning corpus.

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

One example produces one optimizer update. Examples run in file order, cycling
through the file until the total `maxSteps` limit is reached. There is no shuffle,
batch accumulation or automatic training beyond that limit. `layers: null` means
all layers; a list such as `[0,1]` selects explicit zero-based layer indices.
Targets are projection suffixes without `blk.N.` or `.weight`; a layer uses the
listed projections that exist in its architecture.

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

Resume verifies the base GGUF SHA-256, exact dataset bytes and numerical training
contract. The next example index is `savedStep % exampleCount`. Changing learning
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

## Local validation

See [the 27B validation report](qwen35-lora-validation.md) for test results, actual training loss, memory, timing and limits.
