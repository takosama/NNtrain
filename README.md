# NNtrain

NNtrain is a small neural-network training implementation for studying Tensor
operations, reverse-mode automatic differentiation, Transformer modules,
optimizers, dataset boundaries, and training orchestration in C#/.NET 10.

## PyTorch-style API

User-facing training code follows PyTorch's vocabulary while remaining native
C#. The existing PascalCase API is kept for source compatibility, and both
surfaces call the same Tensor, autograd, SIMD, module and optimizer kernels.

```csharp
using NNtrain;

torch.manual_seed(1234);

IImageClassificationDataset trainSet = datasets.mnist(
    images: "data/train-images.idx3-ubyte",
    labels: "data/train-labels.idx1-ubyte");
IImageClassificationDataset testSet = datasets.mnist(
    images: "data/t10k-images.idx3-ubyte",
    labels: "data/t10k-labels.idx1-ubyte");

var trainLoader = torch.utils.data.DataLoader(
    trainSet,
    batch_size: 64,
    shuffle: true,
    training: true,
    generator: torch.generator());
var testLoader = torch.utils.data.DataLoader(testSet, batch_size: 64);

TransformerClassifier model = nn.transformer_classifier(
    seq_len: trainSet.Rows,
    d_model: trainSet.Columns,
    num_heads: 4,
    dim_feedforward: 256,
    num_layers: 2,
    num_classes: trainSet.ClassCount,
    dropout: 0.1f,
    generator: torch.generator());

IOptimizer optimizer = optim.AdamW(
    model.parameters(),
    lr: 3e-4f,
    weight_decay: 5e-4f);
ILRScheduler scheduler = lr_scheduler.LinearWarmupCosineAnnealingLR(
    optimizer,
    total_epochs: 20,
    warmup_epochs: 2,
    min_lr_ratio: 0.01f);

for (int epoch = 0; epoch < 20; epoch++)
{
    model.train();
    scheduler.step();
    foreach (DataBatch batch in trainLoader)
    {
        optimizer.zero_grad();
        Tensor logits = model.forward(batch.input);
        Tensor loss = nn.functional.cross_entropy(logits, batch.target);
        loss.backward();
        optimizer.step();
        Console.WriteLine(loss.item());
    }

    model.eval();
    using (torch.no_grad())
    {
        foreach (DataBatch batch in testLoader)
            _ = model.forward(batch.input);
    }
}

torch.save(model.state_dict(), "model.json");
model.load_state_dict(torch.load<ModuleState>("model.json"));
torch.save(optimizer.state_dict(), "optimizer.json");
optimizer.load_state_dict(
    torch.load<OptimizerStateDictionary>("optimizer.json"));

safetensors.torch.save_file(
    model.state_dict(),
    "model.safetensors");
model.load_state_dict(
    safetensors.torch.load_file("model.safetensors"));
```

Save and resume the model, optimizer, and scheduler as one training
checkpoint:

```csharp
const string checkpointPath = "training.checkpoint.json";
int firstEpoch = 1;

if (File.Exists(checkpointPath))
{
    TrainingCheckpoint checkpoint =
        torch.load<TrainingCheckpoint>(checkpointPath);
    model.load_state_dict(checkpoint.Model);
    optimizer.load_state_dict(checkpoint.Optimizer);
    scheduler.load_state_dict(checkpoint.Scheduler);
    firstEpoch = checkpoint.Epoch + 1;
}

for (int epoch = firstEpoch; epoch <= 20; epoch++)
{
    model.train();
    // forward, backward, optimizer.step(), and evaluation
    scheduler.step();
    torch.save(
        new TrainingCheckpoint(
            epoch,
            model.state_dict(),
            optimizer.state_dict(),
            scheduler.state_dict()),
        checkpointPath);
}
```

Text training uses the same style:

```csharp
BpeTokenizer tokenizer = tokenizers.train_bpe(
    documents,
    vocab_size: 4096);
int[] tokenIds = tokenizer.encode(text, add_bos: true, add_eos: true);
string restored = tokenizer.decode(tokenIds);

IAsyncEnumerable<string> wikipedia = datasets.wikipedia(
    root: "data/wiki",
    text_column: "text");
```

The CLI entry point is `Program.main()`. Its classification path constructs
`datasets`, `DataLoader`, `nn`, `optim`, and `lr_scheduler` objects in that
order before entering the training loop. Wikipedia training uses the same
`torch`, `nn`, `optim`, scheduler, dataset, and tokenizer facades.

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

Every 0.1 epoch and every completed epoch updates a resumable checkpoint
containing
the current model, optimizer, scheduler, epoch, and task-specific training
state. The model weights are also written as standard F32 SafeTensors beside
the JSON training state. Classification uses optimizer-update progress;
finite Wikipedia training uses batch progress; streaming Wikipedia training
uses processed-document progress. Continue from it by increasing `epochs` and
running:

```powershell
dotnet run --configuration Release --project NNtrain.Cli -- `
  --config training.example.json --resume
```

The same behavior can be enabled with `"resumeFromCheckpoint": true` in the
JSON. Classification defaults to `<config>.checkpoint.json`; Wikipedia uses
`checkpointPath`. The separate `*.best-model.json` classification artifact
and `*.best.safetensors` artifacts remain the weights used for evaluation or
generation.

With no arguments the CLI selects `training.wiki-jp.json` when it is present,
then falls back to `training.example.json`. The selected absolute path and the
effective batch/model settings are printed before training. The GPT run
trains or loads the BPE tokenizer, reads bounded Wikipedia data, trains the
causal language model, writes the loss graph and checkpoint, and generates a
sample continuation. Image-classification examples remain available in the
separate CIFAR-100 configuration.
`TransformerClassifier` is used only by the image-classification command. The
default Wikipedia configurations select `modelArchitecture:
"forgetmemoryv2"`, which constructs the custom `FrogetMemoryV2Gpt`; startup
prints the concrete model type so this selection is visible before training.
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

To train the attention-free Hyena variant with the same Wikipedia pipeline:

```powershell
dotnet run --configuration Release --project NNtrain.Cli -- `
  --config training.hyena-wiki-jp.json
```

To train ForgetScanGPT with its content-dependent associative memory scan:

```powershell
dotnet run --configuration Release --project NNtrain.Cli -- `
  --config training.forgetscan-wiki-jp.json
```

To run the conceptual ForgetMemoryV2 matrix-memory model:

```powershell
dotnet run --configuration Release --project NNtrain.Cli -- `
  --config training.forgetmemoryv2-wiki-jp.json
```

ForgetMemoryV2 is the default model. Set `modelArchitecture` to
`"forgetmemoryv2"`, `"forgetscan"`, `"hyena"`, or `"transformer"` when an
explicit architecture is required. `forgetMemoryKeyWidth` and
`forgetMemoryValueWidth` control the associative matrix shape. The retention
floor increases from `forgetMemoryRetentionMinimum` in the shallow layer to
`forgetMemoryRetentionMaximum` in the deepest layer.
`hyenaFilterWidth` controls the hidden width of the implicit long-filter MLP.
`hyenaConvolutionAlgorithm` accepts `"auto"`, `"direct"`, or `"fft"`.
The automatic mode keeps short sequences on the direct SIMD kernel and uses
the zero-padded SIMD FFT kernel from 1024 tokens during training and 2048
tokens during inference.
`HyenaGpt` follows the order-2 operator from the
[Hyena Hierarchy paper](https://arxiv.org/abs/2302.10866) and its
[official standalone implementation](https://github.com/HazyResearch/safari/blob/main/standalone_hyena.py):
a 3-way input projection, causal depthwise short filter, two data-controlled
gates, an implicit sinusoidal and exponentially modulated long filter, and an
output projection. The CPU implementation automatically selects between the
direct SIMD convolution and the zero-padded SIMD FFT convolution.

`ForgetScanGpt` projects each normalized token into forget, input, and value
gates, then evaluates `m[t] = f[t] * m[t-1] + i[t] * v[t]` with an associative
affine scan. On CPU, independent state channels are scheduled as cache-aligned
tiles across worker threads and evaluated in one `O(Ld)` pass; AVX2/FMA kernels
apply the gates and recurrence in each tile. Training saves gate values for a
SIMD reverse scan, while inference omits those buffers. Both paths remain
causal without storing attention keys or values.

`FrogetMemoryV2Layer` packs q, k, v, retention-gate, and beta projections into
one Tensor and evaluates a differentiable matrix memory recurrence:
`g = lambda + (1-lambda)sigmoid(gate)`,
`write = (1-g)sigmoid(beta)`,
`M[t] = g*M[t-1] + write*(v-M[t-1]k)k^T`, and `r[t] = M[t]q`.
The positive delta term moves the current recall toward v; using a negative
term would increase the prediction error. Time remains a direct causal
recurrence, while AVX2/FMA handles state dot products, state updates, recall,
and all major backward vectors. Independent batches use `Parallel.For`.

On a Ryzen 7 5700X, the reproducible two-layer training benchmark
(`batch=2`, `width=64`, `hidden=128`, key/value width 32) measured:

| Sequence | Attention | FrogetMemoryV2 SIMD |
|---:|---:|---:|
| 64 | 2.184 ms | 2.453 ms |
| 128 | 4.908 ms | 4.805 ms |
| 256 | 11.198 ms | 9.218 ms |

Run it with `--filter *FrogetMemoryV2AttentionBenchmarks*`. These compare the
same GPT macro dimensions; parameter counts are close but not exactly equal.

The reproducible ForgetScan microbenchmarks can be run with:

```powershell
dotnet run -c Release --project NNtrain.Benchmarks -- --filter *ForgetScan*
```

To profile one complete training step with the dimensions and optimizer read
directly from a training JSON file, run:

```powershell
dotnet run -c Release --project NNtrain.Benchmarks -- --profile-wiki training.wiki-jp.json
```

To isolate AdamW with the exact model shape and optimizer settings from the
same JSON, run:

```powershell
dotnet run -c Release --project NNtrain.Benchmarks -- --profile-adamw training.wiki-jp.json
dotnet run -c Release --project NNtrain.Benchmarks -- --filter *AdamWJsonBenchmarks*
```

`adamWUseBFloat16FirstMoment` and
`adamWUseBFloat16SecondMoment` independently store AdamW moment buffers as
bfloat16 between steps. The update is expanded to float32 in the AVX2/FMA
kernel and checkpoints are still serialized as float32, so saved-state format
and restore behavior remain portable. These options are off by default because
they trade moment precision for lower memory traffic. The Wikipedia JSON turns
both on: for its 10,551,680 parameter elements this reduces moment storage from
84,413,440 to 42,206,720 bytes.

The profiler reports forward, loss, backward, NekoMuon, and AdamW wall time,
plus allocation/GC counts and a separate summed worker-CPU breakdown inside
NekoMuon. With the current 17.2M-parameter `training.wiki-jp.json` profile on a
Ryzen 7 5700X and `nekoMuonNewtonSchulzInterval: 5`, a ten-step run averaged
589.89 ms per training step and 147.78 ms in NekoMuon. Non-refresh optimizer
steps took about 36--40 ms, while the fifth-step orthogonalization took about
585--598 ms. The profiler measures complete cadence cycles and reports means,
because a median would hide the periodic fifth-step cost.

The default GPT profile uses a 4096-token BPE vocabulary, reads every Parquet
document (`maxTrainingDocuments: 0`), and uses the JSON-configured token limit
from each
document. `maxTrainingTokens: 0` selects the streaming path, so the complete
tokenized corpus is not retained in memory. The tokenizer and best checkpoint
paths are controlled by `tokenizerPath` and `checkpointPath`.
The byte-level BPE vocabulary reserves `<pad>=0`, `<bos>=1`, `<eos>=2`, and
`<unk>=3`. Training pads incomplete final sequences with token id 0 and writes
`-1` to the corresponding targets. `CrossEntropyWithLogits` enables
`ignoreIndex=-1` by default, excludes those rows from both gradients and the
mean-loss denominator, and continues to train BOS/EOS normally.
The optimizer is selected by the JSON. With `optimizer: "adamw"`, AdamW updates
every model parameter; with `optimizer: "nekomuon"`, NekoMuon updates matrix
weights while an auxiliary AdamW updates embeddings, normalization parameters,
biases, and the language-model output head. Their learning rates are controlled
independently by `learningRate` and `auxiliaryLearningRate`.
`nekoMuonNewtonSchulzInterval` defaults to 5: moments and weights advance on
every step using the normalized current momentum, while the expensive
Newton--Schulz orthogonalization runs only every fifth step. Set it to 1 for
orthogonalization on every optimizer step. This cadence reduction is an
intentional throughput variant; the original Muon implementation instead runs
five Newton--Schulz iterations on every optimizer step.
`warmupPercent` defaults to 20: both optimizer groups linearly warm up over the
first 20% of total training progress, then follow cosine decay for the
remaining 80%. Finite-token training uses exact optimizer-step progress;
streaming all-data training uses epoch plus processed-document progress.
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
instead of retaining a vocabulary-sized probability buffer. ForgetScan fuses
residual addition with Dropout; its counter-based SIMD mask is regenerated in
backward instead of retaining an activation-sized mask. NekoMuon computes both
symmetric Gram products with a four-by-two blocked AVX2/FMA kernel, updates
eight polynomial output rows from each source-vector load, and reuses
per-parameter workspaces to reduce allocation and garbage-collection overhead.
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
