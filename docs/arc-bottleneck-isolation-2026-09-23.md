# Arc B580 Transformer: current slowdown isolated, 2026-09-23

## Measured condition

Intel Arc B580 (driver `32.0.101.9030`), Windows x64, .NET 10 Release, current `training.transformer.json` SHA-256 `9B298E978EB748C5218CB87A8A9AF31BF13219519C3B28B5A39826F03C312CCD`. Transformer batch 16 × accumulation 8, context 2048, 32 layers, width 512, 16 heads of dimension 32, `mix8_32` block 32, Muon NS5 + AdamW. Each update consumes 262,144 fixed-seed synthetic tokens. The probe excludes FineWeb/tokenization, scheduling, checkpoint/HTML and generation I/O. No `NNtrain.Cli` training process was running during the measurements.

The normal, untraced 2-warmup/4-measurement result was 37,914.20 ms/update (6,914.1 tokens/s). The timeline run used 1 warmup + 2 measured updates, producing a 37,944.43 ms p50; tracing is a separate diagnostic, not a replacement for the normal throughput result.

## Exclusive wall-clock partition

`ArcTimeline.Partition.WallCategories` assigns every instant to one GPU command category, or to the deepest host scope when no measured GPU command is executing. The following are means across two measured updates; categories are disjoint and total the ~37.94-second wall time.

| Exclusive category | ms/update | share of wall |
| --- | ---: | ---: |
| GPU attention (includes ~622 ms of fused QKV decode/pack by name classification) | 24,608.00 | 64.9% |
| GPU GEMM | 7,186.77 | 18.9% |
| GPU pack/decode/publication | 2,962.10 | 7.8% |
| GPU normalization/reduction | 2,138.35 | 5.6% |
| GPU other kernels + loss + transfers | ~478.9 | ~1.3% |
| **GPU not executing a measured command** | **570.32** | **1.5%** |

GPU-command busy time was 37,374.04 ms/update (98.5%). This means the primary delay is **within GPU-executed work**, not host/VRAM transfer gaps. It does **not** prove high EU occupancy or that every kernel saturates compute or memory bandwidth. Within the 570 ms command-idle interval, queue submission was ~179 ms, partial waits ~156 ms and event collection ~147 ms; exposed native allocation and free were only ~7 and ~12 ms. H2D/D2H/D2D command execution together took ~2 ms. The backend moved 2,097,152 H2D bytes of token/target data and 2,604 D2H bytes of scalars/status per update. Allocation API totals, overlapping waits and queue delays from the older profile must **not** be added to this wall partition.

Each timeline update captured 342,002 device events with 100% wall coverage, zero missing events, zero opaque allocation uploads, zero GPU-event overlap and zero unattributed host time. CPU↔GPU clock-bracket uncertainty was below 0.031 ms. This quality gate matters because a missing event could falsely inflate the idle category.

## Attention broken down by exact operation

`WallOperations` avoids interpreting the name-based category as pure arithmetic. The largest exact GPU commands were:

| Operation | ms/update |
| --- | ---: |
| backward dK/dV, `attention_dkv_block_slm_causal` | 5,121.57 |
| backward probability derivative | 4,877.95 |
| backward dQ | 3,512.21 |
| forward PV | 2,754.55 |
| forward probability | 2,301.39 |
| backward dP | 2,161.48 |
| backward saved-probability replay | 1,593.52 |

The top three alone occupy 13.51 seconds/update. At T2048, two FP32 T² score/derivative buffers and the 64 MiB attention workspace select **two heads per tile**. Across eight microbatches and 32 layers, each main attention stage launches 32,768 times; attention accounts for 295,424 of ~340,307 total kernel launches/update. This establishes repeated T² processing as the expensive code path. It does **not** establish whether an individual kernel is limited by memory bandwidth, arithmetic throughput, occupancy, or local-memory use; hardware counters would be needed for that claim. Earlier 128/256 MiB workspace trials reduced launch count but worsened full-update wall time, so launch count alone is not the cause.

## Controlled dK/dV A/B/A

The same configuration SHA, seed, shape, precision, optimizer, Release binary SHA and timeline settings were used. The **only** Arc option difference was `BlockIoAttention` on/off/on, selecting the SLM block-I/O dK/dV kernel or its older dK/dV alternative. Across the two measured steps, ON/OFF loss differed by at most `2.9e-5` and gradient norm by at most `1.821e-5`; this diagnostic is not a bitwise attention-equivalence test.

| Run | update p50 ms | dK/dV GPU ms | GPU busy ms | attention GPU ms |
| --- | ---: | ---: | ---: | ---: |
| ON | 37,944.43 | 5,121.57 | 37,374.04 | 24,608.00 |
| OFF | 39,737.38 | 6,927.37 | 39,177.79 | 26,410.63 |
| ON again | 37,922.87 | 5,122.19 | 37,375.90 | 24,609.97 |

Turning off block I/O costs **1,803.73 ms/update** relative to mean ON, matching the dK/dV kernel increase of **1,805.49 ms**. The other large stages and host idle are approximately unchanged. Thus dK/dV is a real, isolated contributor and its current SLM path already saves ~1.8 s; it still consumes ~5.12 s/update. No dispatch default was changed by this diagnostic.

## Consequence and next falsifiable hypotheses

- Amdahl bound: halving the entire measured attention category would improve total time by at most ~1.48×; halving dK/dV alone by ~1.07×. Eliminating **all** measured GPU-idle time yields only ~1.015×. A CPU transfer or allocator-only rewrite cannot explain a multi-fold slowdown under this condition.
- The next high-impact experiments should target the FP32 attention derivative (~4.88 s), dK/dV (~5.12 s) and dQ (~3.51 s) **one at a time**, retaining current BF16/BFP8 quantization and FP32 accumulation semantics. Compare full updates as well as resident-input kernel probes; a faster microkernel alone is not adoption evidence.
- Non-attention work is still substantial: GEMM ~7.19 s (3.463 s in fixed-tile streamed GEMM) and pack/decode/publication ~2.96 s. The existing `--expanded-xmx` flag does not change streamed tiles, so testing that particular hypothesis requires a separate opt-in implementation. Pool size 4 vs 6 GiB has only an unmatched ~0.27% prior signal and needs same-binary A/B/A before adoption.

## Reproduce

Use new output names each time; the probe intentionally refuses to overwrite artifacts. Do not run another GPU workload concurrently.

```powershell
$tag = Get-Date -Format 'yyyyMMdd-HHmmss'
dotnet build .\NNtrain.slnx -c Release
dotnet run --configuration Release --no-build --project .\NNtrain.Benchmarks\NNtrain.Benchmarks.csproj -- --probe-arc-transformer .\training.transformer.json ".\benchmark-results\arc-cause-on-$tag.json" --warmup 1 --steps 2 --timeline --block-io-attention on
dotnet run --configuration Release --no-build --project .\NNtrain.Benchmarks\NNtrain.Benchmarks.csproj -- --probe-arc-transformer .\training.transformer.json ".\benchmark-results\arc-cause-off-$tag.json" --warmup 1 --steps 2 --timeline --block-io-attention off
dotnet run --configuration Release --no-build --project .\NNtrain.Benchmarks\NNtrain.Benchmarks.csproj -- --probe-arc-transformer .\training.transformer.json ".\benchmark-results\arc-cause-on-repeat-$tag.json" --warmup 1 --steps 2 --timeline --block-io-attention on
powershell -NoProfile -ExecutionPolicy Bypass -File .\analyze-arc-bottleneck.ps1 -On ".\benchmark-results\arc-cause-on-$tag.json" -Off ".\benchmark-results\arc-cause-off-$tag.json" -OnRepeat ".\benchmark-results\arc-cause-on-repeat-$tag.json"
```

The saved artifacts from this run are `benchmark-results/arc-current-cause-timeline-20260923.json`, `arc-current-cause-dkv-alternative-timeline-20260923.json`, and `arc-current-cause-timeline-repeat-20260923.json`, each with `.step-2/3.trace.json.gz` files. Run `analyze-arc-bottleneck.ps1` on those three JSON files to verify hashes, the one-option A/B, timeline quality and the numbers above.
