# Muon CUDA training and Amdahl optimization — 2026-09-02

## Production conditions

- Windows x64, .NET 10, 2 x NVIDIA GeForce RTX 3070 Ti
- `training.transformer.json`
- microbatch 32, gradient accumulation 4, effective batch 128
- sequence 1024, width 512, 16 heads, hidden 1536, 32 layers
- vocabulary 11,500, `mix8_32`, BFP8 block size 32
- Muon for matrix parameters: momentum 0.95, Nesterov, fixed Newton–Schulz depth 5
- AdamW for auxiliary parameters; configured learning rates 0.01 / 0.003

Reproduction command:

```powershell
dotnet run --project .\NNtrain.Benchmarks\NNtrain.Benchmarks.csproj -c Release --no-build -- --profile-transformer-json .\training.transformer.json 5 10 0 0 mix8_32
```

## End-to-end same-binary A/B result

The disabled run used the final DLL with all three new paths disabled through
their diagnostic environment switches. The enabled run used the same DLL and
the same benchmark shape.

| Final DLL | Step p50 | Step mean | Step p95 | Throughput |
|---|---:|---:|---:|---:|
| New paths disabled | 3,882.47 ms | 3,883.47 ms | 3,909.81 ms | 33,751 tokens/s |
| New paths enabled | 3,564.01 ms | 3,561.53 ms | 3,579.55 ms | 36,802 tokens/s |

The enabled path reduces p50 by 8.20%, mean step time by 8.29%, and raises
throughput by 9.04%. All ten measured updates used CUDA Graph replay, had zero
graph fallbacks, zero CUDA malloc/free calls, and an even 16/16 shard split.
Compiled-graph pinned storage decreased by exactly 32 MiB, from 8,311,148,568
to 8,277,594,136 bytes.

Observed VRAM was 7,294.5 MiB on GPU 0 and 7,955.0 MiB on GPU 1. The second GPU
also hosted desktop graphics allocations. Batch/scalar traffic over ten
updates was 10,485,760 bytes H2D and 320 bytes D2H. The remaining transfer
volume was the existing host-bounce collective because this machine did not
provide peer-to-peer transport for the two devices.

## Muon implementation

- `optimizer.type = "muon"` is accepted by classification and Wiki training.
- Matrix parameters use ordinary Muon with momentum 0.95, Nesterov momentum,
  and five Newton–Schulz iterations on every optimizer update. Auxiliary
  parameters continue to use AdamW.
- CUDA has dedicated Nesterov moment/statistic kernels for float32, bfloat16,
  mix16_32, bfp8, and mix8_32. The block-32 mix8 path keeps optimizer state in
  BFP8 and adds no persistent state allocation relative to NekoMuon.
- One-GPU, two-GPU, gradient accumulation, and checkpoint resume paths use the
  same policy. Resume reapplies the configured Muon policy after decoding old
  compatible NekoMuon state.
- Existing NekoMuon behavior and checkpoint compatibility remain unchanged.

## Amdahl profile and adopted changes

The initial synchronized diagnostic measured 1,809.24 ms per microbatch:
forward 534.88 ms, backward 1,167.41 ms, all-reduce 10.32 ms, clipping 0.29 ms,
Muon-family update 64.57 ms, and AdamW 17.56 ms. Forward plus backward was
94.1% of the measured time, so work proceeded in those paths first.

1. Direct block-32 BFP8 fused residual/dropout/LayerNorm for width 512 removed
   an intermediate decode. LayerNorm forward improved 17.8% and backward
   23.9% in the synchronized section; compiled-graph storage fell by 32 MiB.
2. Attention backward now consumes its BFP8 output directly while retaining
   the Tensor Core query-owner/key-owner DKV reduction. This removes a full
   BF16 output reconstruction and preserves the established BF16 rounding.
3. Linear bias backward uses a column-owner block reduction for production
   row counts, eliminating atomics and temporary workspace. Same-binary
   A-B-A-B testing reduced end-to-end p50 by 5.02% for this change alone.

Diagnostic rollback switches are
`NNTRAIN_DISABLE_DIRECT_BFP8_LAYERNORM_BLOCK32X512`,
`NNTRAIN_DISABLE_DIRECT_BFP8_ATTENTION_OUTPUT`, and
`NNTRAIN_DISABLE_LINEAR_BIAS_BLOCK_REDUCTION`.

## Verification

- Release solution build: 0 warnings, 0 errors
- Core tests: 1,142 passed
- Integration tests: 329 passed
- Benchmark tests: 26 passed
- Total: 1,497 passed
- Native ABI 29 build: SM80, SM86, SM89, SM90, plus compute-90 PTX
- Production two-GPU mix8_32 accumulation run: finite loss, 16/16 shards, no
  OOM, illegal access, graph fallback, or timed allocation
