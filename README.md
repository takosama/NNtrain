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

Copy `training.example.json`, update its MNIST file paths, and run:

```powershell
dotnet run --configuration Release --project NNtrain.Cli -- `
  --config training.example.json
```

Training prints the loss after every batch and aggregate training/evaluation
metrics after every epoch. Every training sample is shuffled and processed once
per epoch; `batchSize` controls how many sample gradients are averaged before
each optimizer update. Batch forward passes and evaluation use all logical CPU
cores. Backward accumulation remains ordered to avoid races in shared parameter
gradient buffers. Training cross entropy uses PyTorch-style label smoothing
with a default value of `0.1`; evaluation always reports unsmoothed cross
entropy. Set `labelSmoothing` to `0` to recover one-hot training targets.
When `showLossGraph` is enabled, the CLI opens an HTML graph beside the selected
configuration file. Training and evaluation loss are plotted as connected
epoch points, and the browser refreshes the graph automatically.
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
