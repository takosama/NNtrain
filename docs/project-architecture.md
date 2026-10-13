# Project architecture

NNtrain separates numerical runtime, GPU backends, model execution, datasets,
training services, and application composition at the assembly boundary. Public
library usage follows the PyTorch-style `torch`, `nn`, `optim`, `lr_scheduler`,
`datasets`, `tokenizers`, and `safetensors` facades.

## Project dependencies

The project files declare these first-party dependencies. `Directory.Build.targets`
checks the declared layer boundaries before reference resolution.

| Project | First-party dependencies | Responsibility |
| --- | --- | --- |
| Runtime | None | Execution contracts, precision, sessions, transfer budgets |
| Arc / Cuda | Runtime | Device resources, transfers, kernel dispatch, native interop |
| Core | Runtime, Cuda, Arc | Tensor, autograd, models, optimizers, ASR, serialization |
| Data | Core | Dataset formats and shared Parquet text streaming |
| Training | Core, Runtime | Training sessions, optimizer groups, metrics, checkpoint transactions |
| Gemma | Core | Local Gemma execution, formatting, native backend, sampling |
| Cli | Core, Data, Training, Gemma | Arguments, application composition, reporting, persistence placement |
| Gui | Core, Gemma | WPF presentation, local API host/client, inference session orchestration |
| Benchmarks | Core, Cli, Cuda, Data, Runtime, Training | Measurement, validation, fixtures, performance evidence |

Core never references Data, Cli, Gui, or Training. Runtime remains independent of
concrete GPU backends. Application-specific configuration and process exit codes
remain in Cli or Gui.

## Core layers

The public facades and Module API compose Tensor operations and autograd, which
select CPU scalar/SIMD, CUDA, or Arc execution through the backend boundary.
Native handles and resident-buffer caches remain implementation details.

Tensor device responsibilities separate replica ownership, buffer acquisition,
gradient coherence, transfers, and host synchronization while retaining one
Tensor identity. `TorchDevice` identifies the backend and adapter. Execution
sessions provide ambient policies without changing the identity of a tensor
explicitly moved to a device.

The Qwen35 model keeps shared ownership and forward execution in its main file;
loading, generation/prefix caching, layer state, and matrix storage/dispatch have
cohesive partial implementations. Stateless logit sampling is a separate helper.
Partial declarations preserve existing member and nested-type identities.

## Model and optimizer contracts

`Module` owns parameter registration, mode, dtype conversion, and state
dictionaries. `LanguageModel` extends Module with token forward and generation
behavior without duplicating the Module lifecycle.

`IOptimizer` uses the canonical PyTorch-style lifecycle:

```csharp
optimizer.zero_grad();
optimizer.step();
OptimizerStateDictionary state = optimizer.state_dict();
optimizer.load_state_dict(state);
```

Every optimizer owns its serialization. AdamW and NekoMuon separate checkpoint,
CUDA/state lifetime, diagnostics, and numerical responsibilities. Shared
parameter validation preserves reference identity and enumeration order.
Checkpoint code does not switch on concrete optimizer types.

SafeTensors state and streaming model writers share header construction and dtype
codecs. Readers preserve their existing error and allocation contracts; format
metadata, parameter ordering, encoded bytes, and bounded staging remain stable.

## CLI training lifecycle

`Program` routes application entrypoints. `TrainingCommandArguments` preserves the
strict top-level grammar; `ClassificationTrainingCommand` owns classification
execution. Dataset composition, cursor/step execution, metric counting, console
reporting, checkpoint placement, and resume policy have separate owners. Existing
internal entrypoints remain forwarding compatibility seams.

Configuration schema version 2 has an explicit `task.type` and common `data`,
`model`, `training`, `runtime`, `optimization`, `checkpoint`, and `reporting`
sections. The loader normalizes these into task-specific validated records.

`TrainingRunner` owns deterministic epoch/resume iteration, progress-unit
rounding, shuffling, and fractional checkpoint boundaries. Classification,
finite-corpus language modelling, and streaming language modelling keep their
numerical batches while sharing those lifecycle rules. Wikipedia and FineWeb
facades share the bounded Parquet reader and retain their public entrypoints.

`CheckpointRepository` owns transaction creation. `CheckpointWriteTransaction`
owns artifact staging, rollback, and manifest-last publication; fault contracts
are separate. Legacy checkpoint versions remain readable, and the next save
writes the current format.

## GUI and local API

MainWindow partials bind models, chat, audio, images, and server state to the
existing WPF controls. `LocalInferenceClient` owns authenticated HTTP/JSON/SSE;
`LocalInferenceServerProcess` owns the launched child and session token.
`ChatGenerationSession` owns thinking-only retry and interrupted partial text.
Audio model loading and per-utterance recognition are separate from visual state.

`LocalOpenAiServer` composes lifecycle, management routes, chat routes, and model
lifetime. Access policy, request validation, response framing, and inference
contracts are independent helpers. `InferenceSession` retains model ownership
while loading, generation/history priming, and vision responsibilities are
separate partials. Shared prompt formatting keeps text and multimodal framing
consistent.

The production GUI remains Windows/WPF. IntegrationTests link only the independent
GUI helpers, so protocol, authentication, cancellation, prompt, and audio
contracts can run without a window or GPU. Windows GUI checks retain their
assertions and access the new service owners through one shared test helper.

## Verification

```powershell
dotnet build NNtrain.slnx -c Debug -p:EnableWindowsTargeting=true
dotnet test NNtrain.Core.Tests/NNtrain.Core.Tests.csproj -c Debug
dotnet test NNtrain.IntegrationTests/NNtrain.IntegrationTests.csproj -c Debug
dotnet test NNtrain.Benchmarks.Tests/NNtrain.Benchmarks.Tests.csproj -c Debug
```

WPF GUI tests require Windows. GPU and real-model cases require their declared
hardware, weights, and opt-in conditions. Compare unavailable-hardware failures
and skips with the unchanged baseline instead of weakening assertions.

Benchmark source fingerprints include all relevant model, GUI, and Arc partials.
Measurement, fixtures, soak/checkpoint validation, and result aggregation are
separate responsibilities; measured algorithms and gate thresholds are preserved.
