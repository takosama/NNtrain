# Reproducible training performance validation

Run the harness through the benchmark executable:

```powershell
dotnet run --project .\NNtrain.Benchmarks\NNtrain.Benchmarks.csproj -c Release --no-build -- --performance-baseline compare10 --config .\training.transformer.json
dotnet run --project .\NNtrain.Benchmarks\NNtrain.Benchmarks.csproj -c Release --no-build -- --performance-baseline official2gpu --config .\training.transformer.json
dotnet run --project .\NNtrain.Benchmarks\NNtrain.Benchmarks.csproj -c Release --no-build -- --performance-baseline soak2100 --config .\training.transformer.json
dotnet run --project .\NNtrain.Benchmarks\NNtrain.Benchmarks.csproj -c Release --no-build -- --performance-baseline compare10 --config .\training.transformer.json --precision bfp8 --output .\benchmark-results\performance-baseline-compare10-bfp8.json
dotnet run --project .\NNtrain.Benchmarks\NNtrain.Benchmarks.csproj -c Release --no-build -- --performance-baseline compare10 --config .\training.transformer.json --precision mix8_32 --bfp8-block-size 128 --output .\benchmark-results\performance-baseline-compare10-mix8-32.json
```

`compare10` retains the CPU/one-GPU/two-GPU ten-step comparison.
`official2gpu` retains three repetitions of 20 warmup plus 210 measured
two-GPU steps. Its acceptance statistic is the median of the three per-run
step-p50 values. The frozen p50 is 475.480 ms and the required maximum is
80%, or 380.384 ms. These values are embedded in the worker job, repeated in
the JSON conditions, and enforced as a required gate. The preset also pins the
effective batch to 72, sequence to 512, precision to `mix16_32`, and optimizer
contract to NekoMuon fixed NS5. These effective values and their configured
input values are written to the console and JSON; the input configuration is
never changed. A slower or contract-invalid result returns exit code 3 without
discarding the JSON. `soak2100` commits exactly 2100
two-GPU steps and excludes its
first 20 commits from the performance trend. It checks the first/last 100-step
p50 ratio, post-warmup VRAM growth of at most 256 MiB per GPU, positive shards
on both GPUs, a generation event immediately after committed global step 2000,
and the absence of OOM, CUDA 600/700, illegal-access, or other runtime errors.
Unavailable VRAM telemetry fails the required VRAM gate.

After committed global step 1050, the soak synchronizes both GPUs and publishes
an actual Wiki format-v8 streaming checkpoint (current/best SafeTensor model
artifacts, one binary artifact per optimizer leaf, then the manifest). It
records SHA-256 and byte size for every artifact plus total bytes and save time.
The old fixture, model, optimizer, and data-parallel engine are then completely
disposed before a fresh fixture is constructed and the checkpoint is loaded.
No two model fixtures coexist in VRAM.

The restart gate validates the global step and mid-epoch cursor, training RNG,
zero-warmup cosine scheduler, adaptive-shard state, model and optimizer steps,
precision, mix8_32 block size, and parameter residency on both GPUs. It also
records load time and verifies artifact-first/manifest-last publication. The
same full restart mechanism is used by `soak-smoke` with a tiny model. The soak
continues to gate the production LossGraph append-only sidecar and rendered
HTML across the restart, and still runs generation after committed step 2000.
Successful temporary artifacts are removed; failed runs retain the diagnostic
directory reported in the result JSON.

The input configuration is read-only. The comparison presets `compare10`,
`cpu10`, `gpu1-10`, and `gpu2-10` accept `--precision` with `float32`,
`bfloat16`, `mix16_32` (`fp16_32` alias), `bfp8`, or `mix8_32`.
`--bfp8-block-size` is accepted only when the effective precision is
`mix8_32`; pure `bfp8` remains tensor-scale. Generation cadence and interruption
points are worker inputs, so `training.transformer.json` is never rewritten.
All conditions, forward, backward, all-reduce wait, clipping, optimizer,
transfer, managed/native allocation, shard, CUDA Graph, and VRAM telemetry,
plus gate outcomes, are written to the result JSON. A failed required gate
returns exit code 3 after publishing the JSON. Use `soak-smoke` to exercise the same harness
mechanics with the tiny built-in model without running 2100 steps. `cpu-smoke`
and `gpu-smoke` exercise the ordinary runner and its synchronized diagnostic
phase/transfer probe with the same tiny shape.

`soak-failure-smoke` is the transaction diagnostic. It intentionally throws
after publishing the first checkpoint artifact, must fail its validation gate,
and records `artifactsRetainedAfterFailure=true`, the retained directory, and
the partial artifact's size and SHA-256 in the same v2 result schema.
