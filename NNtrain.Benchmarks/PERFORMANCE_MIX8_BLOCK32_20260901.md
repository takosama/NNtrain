# Mix8 block-32 Transformer optimization — 2026-09-01

## Production conditions

- Windows x64, .NET 10, 2 x NVIDIA GeForce RTX 3070 Ti
- `training.transformer.json`
- microbatch 140, gradient accumulation 4, effective batch 560
- sequence 512, width 384, 12 heads, hidden 1536, 16 layers
- vocabulary 11,500, `mix8_32`, BFP8 block size 32
- NekoMuon fixed Newton–Schulz depth 5 plus AdamW

Reproduction command:

```powershell
dotnet run --project .\NNtrain.Benchmarks\NNtrain.Benchmarks.csproj -c Release --no-build -- --profile-transformer-json .\training.transformer.json 5 10 0 0 mix8_32
```

## End-to-end result

| Build | Step p50 | Step mean | Step p95 | Throughput |
|---|---:|---:|---:|---:|
| Frozen pre-optimization graph path | 4,691.60 ms | 4,697.53 ms | — | 61,113 tokens/s |
| Final, 5 warmup + 10 measured | 2,467.52 ms | 2,467.60 ms | 2,483.08 ms | 116,194 tokens/s |

The final path is 47.4% shorter per optimizer update and provides 1.90x the
token throughput. All ten measured steps used CUDA Graph replay, had zero CUDA
malloc/free calls, zero graph fallbacks, and an even 70/70 shard split.

Final compiled-graph reservation was 9,195,777,048 bytes across both GPUs.
Reported VRAM use was 6,630.5 MiB on GPU 0 and 7,458.3 MiB on GPU 1; GPU 1
also hosted desktop graphics allocations. Non-collective D2H traffic was only
70 scalar copies / 320 bytes over the ten measured updates.

## Adopted A/B changes

- Accumulated CUDA Graph replay: pre-native p50 5,977.99 to 4,691.60 ms.
- Block-32 BFP8 codec: F32/BF16 quantization 10.01x/10.37x faster on a
  41,287,680-element production QKV-sized tensor; dequantization improved
  3.8–5.0%.
- Direct width-384/block-32 fused residual/dropout/LayerNorm: forward 57.26 to
  35.74 ms and backward 65.48 to 36.96 ms in the synchronized diagnostic.
- Exclusive FFN intermediate BF16 gradient: exact-graph A-B-A-B p50 average
  2,597.24 to 2,497.37 ms; graph reservation decreased by 210 MiB total and
  observed VRAM decreased by about 100 MiB per GPU.

Two candidates were rejected and reverted: pre-encoded attention dO regressed
compiled-graph p50 by 0.51%, and dX-only cuBLASLt regressed it by 0.64%.

## Verification

- Release solution build: 0 warnings, 0 errors
- Core tests: 1,126 passed
- Integration tests: 316 passed
- Benchmark tests: 24 passed
- Native ABI 26 build: SM80, SM86, SM89, SM90, plus compute-90 PTX
