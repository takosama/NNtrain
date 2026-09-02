# GPU高速・小メモリ化候補1–6 検証結果

実施日: 2026-08-28

## 条件

- Windows x64 / .NET 10 / CUDA 13 build
- GPU: RTX 3070 Ti 8 GiB × 2
- CUDA peer access: `0->1=no`, `1->0=no`
- 設定: `training.transformer.json`
- 設定SHA-256: `02550BAF43A52C04F2336087FD02E6B122A60AEE59AB7A225A01E1E857EB366A`
- model: batch 150、sequence 512、width 512、heads 16、hidden 1538、layers 16、vocabulary 11500
- precision: `mix8_32`
- optimizer: NekoMuon + AdamW
- 各性能値: warmup 3、10 measured steps。step p50を独立processで3回測定した中央値

## 採否

| # | 候補 | 判定 | 根拠 |
|---|---|---|---|
| 1 | ZeRO-1 + gradient reduce-scatter + BFP8 weight all-gather | 不採用 | 現all-reduceはbackwardと重なり、同期待ちは約31.8 ms。optimizer全体約51.6 msの半減上限は25.8 msだが、P2Pなしでoptimizer後に約1 model分のweight all-gatherが露出するため速度利得が成立しない。 |
| 2 | CUDA Graph liveness arena | 不採用・試作をrevert | 単純best-fit再利用ではautogradの最終利用時点より前にbufferを再利用し、graph captureが `unreturned input or target buffer` で失敗。eager fallback p50 1368.83 ms（基準比約+54%）。採用にはCUDA eventを含む真の最終利用解析が必要。 |
| 3 | block-scaled BFP8 INT8 Tensor Core GEMM | 不採用 | 実shard相当 `[38400,512]×[512,512]` はINT8 1.788 ms、block128 BF16 1.788 ms。同FFN整列形状 `[38400,512]×[512,1536]` はINT8 4.564 ms、BF16 4.452 msでBF16が2.5%高速。block scaleを保つK-block分割と再量子化を足す余地がない。 |
| 4 | vectorized attention/BFP8 block128 codec | 採用 | full blockをlaneあたり4要素の`char4` load/storeとBF16x2 storeへ変更。production step p50 888.71→876.27 ms（約1.4%短縮）。tailは従来scalar経路を保持し、CPU codecとのbit一致testに合格。 |
| 5 | 2-stage pipeline parallel | 不採用 | 8/8層分割後も各GPUの演算量は現DPと同じでbubbleが増える。境界activationはforwardだけで `150×512×512×2 = 75 MiB`、backward込み最低150 MiB/stepをP2Pなしで追加転送する。 |
| 6 | blockwise BFP8 optimizer state | 小メモリpresetとして採用 | FP32 master/gradientを維持し、AdamWのm/vとNekoMuonのfast/slowだけblock128 BFP8化。融合kernelでdecode、EMA、scale reduction、requantize、updateを統合。速度差+2.11%、GPU全体VRAMを約308–342 MiB削減。既存mix8_32のFP32 optimizer統計意味論を守るため明示opt-in。 |

## 最終A/B

### 速度preset（既定、FP32 optimizer state）

- 10-step p50: `883.53 / 881.24 / 878.89 ms`
- 3-run中央値: **881.24 ms**
- throughput中央値: **87,350 tokens/s**
- VRAM中央値: GPU0 **7896.5 MiB**、GPU1 **7960.3 MiB**
- 10-step loss（最終run）: `8.815428 -> 8.367090`
- measured CUDA malloc/free: `0/0`
- CUDA Graph: compiled replay 10/10、fallback 0

### 小メモリpreset（block128 BFP8 optimizer state）

- 10-step p50: `899.80 / 901.50 / 897.03 ms`
- 3-run中央値: **899.80 ms**（速度preset比 **+2.11%**）
- throughput中央値: **85,528 tokens/s**（速度preset比 **-2.09%**）
- VRAM中央値: GPU0 **7554.5 MiB**、GPU1 **7652.7 MiB**
- VRAM削減: GPU0 **342.0 MiB**、GPU1 **307.6 MiB**
- optimizer moment自体: 2本合計 `8 B/element -> 2.0625 B/element`、**74.2%削減**
- measured CUDA malloc/free: `0/0`
- CUDA Graph: compiled replay 10/10、fallback 0

30-step比較でもlossは既定 `8.815356 -> 8.503164`、小メモリpreset `8.813790 -> 8.503310`で、最終差は0.000146だった。p50は883.83 ms対909.52 ms（+2.91%）。warmup後のVRAM増加とCUDA allocationは観測されなかった。

小メモリpresetはprocess開始前に次を設定する。

```powershell
$env:NNTRAIN_ENABLE_BLOCK_BFP8_OPTIMIZER_STATE = '1'
dotnet run --project .\NNtrain.Cli\NNtrain.Cli.csproj -c Release --no-build -- --config .\training.transformer.json
```

速度presetへ戻す場合は、新しいprocessで環境変数を外す。

```powershell
Remove-Item Env:NNTRAIN_ENABLE_BLOCK_BFP8_OPTIMIZER_STATE -ErrorAction SilentlyContinue
```

## 検証

- Release build: warning 0 / error 0
- Core tests: 1056/1056
- Integration tests: 310/310
- Benchmark tests: 22/22
- 合計: **1388/1388、失敗0**
- native ABI: 1.23
