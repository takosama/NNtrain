# Project architecture

NNtrain separates numerical runtime, datasets, CLI composition, tests, and
benchmarks at the assembly boundary. Public library usage follows the
PyTorch-style `torch`, `nn`, `optim`, `lr_scheduler`, `datasets`,
`tokenizers`, and `safetensors` facades.

## Project dependencies

```text
NNtrain.Cli ───────→ NNtrain.Data ───────→ NNtrain.Core
      └─────────────────────────────────→ NNtrain.Core

NNtrain.Core.Tests ─────────────────────→ NNtrain.Core
NNtrain.IntegrationTests ───────────────→ Core, Data, Cli
NNtrain.Benchmarks ─────────────────────→ Core, Data, Cli
```

Core never references Data or Cli. Concrete file-format datasets remain in
Data. Configuration, checkpoint placement, progress reporting, run markers,
and process exit codes remain in Cli.

## Core layers

```text
PyTorch-style facade and Module API
                ↓
Tensor operations and autograd
                ↓
internal backend registry and execution context
                ↓
CPU scalar/SIMD or CUDA kernels
```

`TorchDevice` identifies `cpu` or an indexed `cuda:N` adapter. Execution
defaults are stored in an async-local `TensorExecutionContext`, while a Tensor
moved with `tensor.to(torch.device("cuda:N"))` retains its adapter identity.
CUDA data-parallel workers temporarily select their own adapter; shared
parameters therefore resolve buffers from the worker context during a kernel
and from Tensor identity during host synchronization.

CPU and CUDA implementations stay in `NNtrain.Core` but are reached through
the internal backend boundary. ILGPU types and resident-buffer caches must not
leak into the public Tensor or Module contracts.

## Model and optimizer contracts

`Module` owns parameter registration, mode, dtype conversion, and state
dictionaries. `LanguageModel` extends Module only with token forward and
generation behavior; it does not duplicate Module lifecycle methods.

`IOptimizer` uses the canonical PyTorch-style lifecycle:

```csharp
optimizer.zero_grad();
optimizer.step();
OptimizerStateDictionary state = optimizer.state_dict();
optimizer.load_state_dict(state);
```

Every optimizer owns its serialization. Checkpoint code does not switch on
concrete optimizer types.

## CLI training lifecycle

Configuration schema version 2 has an explicit `task.type` and common
`data`, `model`, `training`, `runtime`, `optimization`, `checkpoint`, and
`reporting` sections. The loader normalizes those sections into task-specific
validated records before composition.

`TrainingRunner` owns deterministic epoch/resume iteration, progress-unit
rounding, shuffling, and fractional checkpoint boundaries. Classification,
finite-corpus language modelling, and streaming language modelling retain
their task-specific batch and numerical code while sharing those lifecycle
rules.

JSON checkpoints are written atomically and SafeTensors sidecars carry model
weights. Legacy checkpoint versions remain readable; the next save always
writes the current format.

## Verification

```powershell
dotnet build NNtrain.slnx --configuration Release
dotnet test NNtrain.slnx --configuration Release --no-build
```

CPU, one-GPU, and two-GPU characterization tests protect forward results,
gradients, optimizer updates, device-buffer ownership, and checkpoint resume.
