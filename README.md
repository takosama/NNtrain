# NNtrain

NNtrain is a small neural-network training implementation for studying Tensor
operations, reverse-mode automatic differentiation, Transformer modules,
optimizers, dataset boundaries, and training orchestration in C#/.NET 10.

## Projects

- `NNtrain.Core`: Tensor, autograd, modules, optimizers, and Trainer
- `NNtrain.Data`: dataset implementations such as MNIST IDX
- `NNtrain.Cli`: JSON configuration and the command-line application
- `NNtrain.Core.Tests`: Core unit and characterization tests
- `NNtrain.IntegrationTests`: dataset, configuration, and learning-flow tests

## Build and test

```powershell
dotnet build NNtrain.slnx --configuration Release
dotnet test NNtrain.slnx --configuration Release --no-build
```

## Run training

The default `training.example.json` is configured for the Japanese Wikipedia
GPT task. Place the Parquet shards under `data/wiki` and run:

```powershell
dotnet run --configuration Release --project NNtrain.Cli -- `
  --config training.example.json
```

With no arguments the CLI selects `training.wiki-jp.json` when it is present,
then falls back to `training.example.json`. The selected absolute path and the
effective batch/model settings are printed before training. The GPT run
trains or loads the BPE tokenizer, reads bounded Wikipedia data, trains the
causal language model, writes the loss graph and checkpoint, and generates a
sample continuation. Image-classification examples remain available in the
separate CIFAR-100 configuration.
The CIFAR-100 configuration normalizes RGB channels using the training-set
statistics. Each 32x32 RGB image is emitted directly as 64 row-major 4x4 patch
tokens with 48 channel-first features per token. Its training augmentation uses
a four-pixel random crop and random horizontal flip; vertical flipping is
available but disabled by default.
GainShareAdamW is the default optimizer. It groups parameters at the configured
module depth, measures each block's gradient/update alignment through an EMA,
and redistributes the AdamW update norm between blocks while preserving the
global squared update norm. The default profile uses learning rate `3e-4`,
weight decay `5e-4`, rho `0.95`, gamma `1.0`, and scales `0.5` to `2.0`.
Set `optimizer` to `nekomuon`, `lion`, or `adamw` to retain the other update
rules. NekoMuon continues to use auxiliary AdamW for non-hidden parameters.
The CIFAR-100 profile applies five warmup epochs followed by cosine learning-
rate decay, residual dropout `0.1`, and early stopping after 15 epochs without
an evaluation-loss improvement. At exit, the best model weights are restored
and saved beside the configuration as `*.best-model.json`.

## Train Japanese Wikipedia GPT

`training.wiki-jp.json` reads the sharded Parquet files under `data/wiki`,
trains a reversible UTF-8 byte-level BPE tokenizer when one does not already
exist, streams the corpus, and trains the decoder-only
`GptRinWikiJp` model:

```powershell
dotnet run --configuration Release --project NNtrain.Cli -- `
  --config training.wiki-jp.json
```

The default GPT profile uses a 4096-token BPE vocabulary, reads every Parquet
document (`maxTrainingDocuments: 0`), and uses up to 4096 tokens from each
document. `maxTrainingTokens: 0` selects the streaming path, so the complete
tokenized corpus is not retained in memory. The tokenizer and best checkpoint
paths are controlled by `tokenizerPath` and `checkpointPath`.
The GPT profile uses `optimizer: "nekomuon"`: NekoMuon updates Transformer
matrix weights, while an auxiliary AdamW updates embeddings, normalization
parameters, biases, and the language-model output head. Their learning rates
are controlled independently by `learningRate` and `auxiliaryLearningRate`.
With `useSimd: true`, GPT training uses hardware-accelerated Vector256 kernels
for fused token/position embeddings, linear algebra, normalization, softmax
reductions, cross-entropy, and NekoMuon/AdamW updates. Startup output reports
whether Vector256 acceleration is available on the current CPU.
Wide projection linear layers cache a blocked transpose of unchanged weight
matrices so the output dimension can be processed as contiguous SIMD vectors;
square and large-input matrices keep the cache-friendly dot-product kernel. Softmax,
attention, and cross-entropy use a vectorized polynomial exponential, with a
Vector128 fallback for narrow attention heads. Large cross-entropy batches
recompute probabilities during backward
instead of retaining a vocabulary-sized probability buffer. Dropout fills its
mask in bulk, and NekoMuon reuses per-parameter workspaces to reduce allocation
and garbage-collection overhead.
The same kernels use `Parallel.For` for independent rows, attention heads,
embedding-gradient groups, loss rows, and optimizer parameter groups. Set
`maxDegreeOfParallelism` to `0` to use the runtime-selected worker count, or to
a positive number to cap the number of worker threads.
When `showLossGraph` is enabled, the graph is refreshed every
`graphUpdateSteps` optimizer steps (100 by default) and at every epoch end. Its
horizontal axis is epoch progress; validation loss is added at epoch boundaries.
Every `datasetSampleEverySteps` steps (1000 by default), a random Wikipedia
article from the retained sample pool is split in half. The end of the first
half is used as the prompt, and the dataset continuation and model continuation
are printed together for comparison. The final post-training sample uses the
same dataset-continuation flow rather than a fixed prompt.

After training, load the saved tokenizer and checkpoint and generate from a
prompt without retraining:

```powershell
dotnet run --configuration Release --project NNtrain.Cli -- `
  --config training.wiki-jp.json --generate "日本の歴史は"
```
