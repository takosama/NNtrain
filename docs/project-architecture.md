# Project architecture

Phase 7 separates the runtime, data adapters, application entry point, and
tests into five projects. The projects keep the existing `NNtrain` namespace;
the assembly boundary, rather than the namespace, defines ownership.

## Projects

| Project | Responsibility |
| --- | --- |
| `NNtrain.Core` | Tensor operations, autograd, modules, optimizers, training contracts, and `Trainer` |
| `NNtrain.Data` | Concrete dataset adapters such as the MNIST IDX reader |
| `NNtrain.Cli` | JSON configuration, object composition, console reporting, and process exit codes |
| `NNtrain.Core.Tests` | Pure Core unit and characterization tests |
| `NNtrain.IntegrationTests` | Dataset parsing, CLI configuration, and end-to-end training-flow tests |

`IImageClassificationDataset` belongs to Core because `Trainer` consumes the
contract. `Mnist` belongs to Data because it implements that contract using the
file system and IDX format. This keeps the training engine independent of the
concrete dataset package.

## Allowed dependencies

```text
NNtrain.Cli ───────→ NNtrain.Data ───────→ NNtrain.Core
      └─────────────────────────────────→ NNtrain.Core

NNtrain.Core.Tests ─────────────────────→ NNtrain.Core
NNtrain.IntegrationTests ───────────────→ Core, Data, Cli
```

`NNtrain.Core` has no project references. `NNtrain.Data` references only Core.
`NNtrain.Cli` references Core and Data. Test projects reference only the
projects they exercise. Core must never add a reference to Data or Cli.

## Public boundary

Only cross-assembly contracts and composition types are public. These include
`Tensor`, `Module`, `Parameter`, `TransformerClassifier`, the optimizer and
dataset interfaces, `AdamW`, `GainShareAdamW`, `Lion`, and `Trainer` with their
options/results. `NekoMuon` is also public; every stateful optimizer exposes
versioned state records.
Internal layer implementations and autograd graph details remain internal and
are made visible only to `NNtrain.Core.Tests`. Integration tests use Core's
public API;
only the CLI grants them access to its process-level test seam. Production
projects do not use friend-assembly access to cross a boundary.

## Commands

Build and test the complete solution:

```text
dotnet build NNtrain.slnx --configuration Release
dotnet test NNtrain.slnx --configuration Release --no-build
```

Run the application with a JSON configuration:

```text
dotnet run --project NNtrain.Cli -- --config training.example.json
```
