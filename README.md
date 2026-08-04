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
gradient buffers.
