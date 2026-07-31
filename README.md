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
- `NNtrain.Benchmarks`: BenchmarkDotNet performance baselines

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

## Run benchmarks

```powershell
dotnet run --configuration Release --project NNtrain.Benchmarks -- `
  --filter "*TrainingBenchmarks*"
```

The Debug/Release baseline and measurement requirements are documented in
[`docs/performance-benchmarking.md`](docs/performance-benchmarking.md).
