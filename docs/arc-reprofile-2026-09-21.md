# Arc再プロファイルと追加最適化（2026-09-21）

## 条件と測定範囲

Intel Arc B580 / driver 32.0.101.9030 / Windows / .NET 10 Release。
`training.transformer.json`（SHA256 `EC42E94F3A9BA43DBDB90E5A44368798C422F76AD5330BC936F701261E7D64C5`）を読み、モデル・optimizer factoryを実際のCLIと共有するprobeで比較した。
batch64、sequence1024、width512、heads16、hidden1536、32層、語彙11500、mix8_32 block32、dropout0.1、tied embeddings、90,507,500 parameters。Muon NS5 / AdamW、LR0.001/0.0003、weight decay0.01、seed1234。

固定seedの合成token、fresh model、固定LR。実データ読み込み、checkpoint、HTML生成は除外。元のJSON・学習データ・checkpointは変更していない。短い候補比較は累積1、正式比較は設定どおり累積4（実効batch256）。異なる累積回数の時間を速度倍率として混ぜない。

GPU時間はOpenCL event、phaseは同期wall時間。allocation/queue-delayは重なるのでphase/GPU時間へ足し算しない。メモリはlive+cached+retiredのOpenCL buffer会計で、driverのdedicated VRAM実測ではない。

## 開始時の再測定

`benchmark-results/arc-reprofile-baseline-20260921.json`、1 warmup + 3 measured：p50 **34,112.63 ms/update**。backward **76.8%**。
主要GPU区間（平均/update）：DKV 3,554.57ms、attention derivatives 2,434.91ms、大きなstreamed dW GEMM 2,371.72ms、dQ 1,516.32ms。optimizerは約202msにすぎず、ここを倍速化しても全体改善は約0.3%。

## 比較して採用した変更

1. **大きなstreamed dWの並列split-K**：全Kに対するpartial上限を64→96MiB。1536×512 / 512×1536 / K65536が並列経路に入る。2048単位のFP32 partialをdevice上でreduce。BF16 operand・FP32蓄積は変更しない。ただしserial経路とは加算の結合順が変わるため、bit-exactとは主張せず既存1e-4の許容誤差でproduction shapeを検査する。無制限のworkspaceやatomicは使わない。
2. **forwardとbackwardでAttention head tileを分離**：T1024/D32/SG16のbackwardだけ上限4heads（32MiBのP+DS）にする。forwardは8headsのまま。user指定workspaceを超えず、他shape/deviceは既存経路。headを跨ぐslice・dense/causal・3精度の出力と反復加算gradientをbit比較する。
3. **packed ReLU backward**：ReLU判定だけに使っていたBF16/BFP8 activationのFP32展開を廃止。native storageからgateを読み、従来の丸めを施したBF16 gradient operandを一度作り、dX/dW/bias reductionで共有する。最終gradient・partial・masterはFP32。biasの加算順も従来と同じ。非mixed・非XMX・unsupported shapeはfallback。
4. **BFP8 publicationの連続読込**：block32は4 lanes/block、block128はSG16/block。最大値・nonfinite検出・scale・nearest-even quantizationを保持。NaN/Inf/zero/underflow/tailを含むpayload/scale/status検査。未測定block64/256や非SG16には既存codecを残す。

累積1の候補比較（各1 warmup、基準2測定・変更版3測定）：

| 構成 | p50 ms/update | 生JSONの接尾辞 |
|---|---:|---|
| 今回基準 | 8,693.32 | `workspace-64` |
| split-K 96MiB対象（probe上限128MiB） | 8,399.44 | `parallel-weight` |
| 上記 + backward専用tile | 8,247.28 | `cache-backward` |
| 上記 + packed ReLU + codec | 8,113.80 | `packed-relu` |

各JSONは`benchmark-results/arc-reprofile-{接尾辞}-20260921.json`。単独のkernel倍率を学習全体の倍率とはしない。

## 不採用候補も保存

- serial GEMMのK2048 dispatchを4096/8192/16384に統合：bit-exactだがproduction形状で改善不安定～悪化。16384では1536×512が2.27→2.59ms、512×1536が2.00→2.58ms（GPU）。`arc-serial-gemm-20260921.json`。
- register-only DKV key owner：bit比較合格だがcausal 0.213→1.178msへ悪化。現行SLM tileを維持。`arc-dkv-owner-20260921.json`。
- forward/backward両方のworkspace縮小：32MiBは8,682.42msでほぼ同じ、16MiBは10,224.31ms、8MiBは13,369.76msへ悪化。小さすぎるtileはlaunchとoccupancyの損失が大きい。
- block32をSG16/blockにするcodec：100,663,296要素で1.95→2.74msと悪化。4 lanes/blockなら1.85ms。block128は5.89→1.79ms。`arc-bfp8-quads-20260921.json`（3 warmup + 10 measured）。固定private32、2/8 lanesも測定した。
- pool4→3.5GiB：p50 8,113.80→8,120.74ms、peakは10,527.74MiBのまま、native allocationは286→314回/update。idle保持だけは約494MiB減るが、peakも速度も改善せず再確保が増えるため4GiBを維持。`arc-reprofile-pool3584-20260921.json`。

## 最終検証

最終`arc-reprofile-final-20260921.json`は1 warmup + 5 measured、累積4。p50 **31,780.40ms/update**、今回基準から **6.84%短縮 / 1.0734倍**、8,248.6 tokens/s。5点は31,767.71 / 31,795.22 / 31,781.56 / 31,780.40 / 31,765.47ms。

| 指標 | 今回基準 | 最終 |
|---|---:|---:|
| forward平均 ms/update | 7,693.75 | 7,641.97 |
| backward平均 ms/update | 26,181.28 | 23,906.28 |
| optimizer平均 ms/update | 202.20 | 200.76 |
| GPU event合計 ms/update | 32,667.32 | 29,852.28 |
| H2D bytes/update | 2,097,152 | 2,097,152 |
| D2H bytes/update | 2,588 | 2,588 |
| buffer会計peak MiB | 10,239.49 | 10,527.74 |
| 更新後active MiB | 1,478.14 | 1,478.14 |
| 更新後cached MiB | 4,056.44 | 4,047.44 |
| native allocation回数/update | 1,010.0 | 1,069.6 |
| native allocation累積bytes/update | 20,828,533,349 | 22,075,288,848 |
| allocation wall ms/update | 177.77 | 163.17 |
| kernel launches/update | 256,291 | 360,611 |

今回の高速化はallocation回数の削減ではない。小さなbackward tileで起動数が増えてもGPU実行時間が減った。一方で会計peakは**288.25MiB増加**し、native要求の累積量も増える。これを「VRAMが減った」「毎stepの再確保を解消した」とは報告しない。5測定step間のactive/peak増加はともに0、同期後retiredは0。D2Hはloss/norm/optimizer diagnostics等を含み、重み・activation・gradientのhost移動を増やしていない。

基準と共通の3 measured stepsでlossの最大差0.000014、gradient normの最大差0.00007789。個別packed ReLU・Attention head tile・codecはbit比較、split-Kのproduction shapeは既存1e-4基準で合格した。これは長期収束同一性の証明ではない。

batch16/累積4も1 warmup + 3 measuredで **7,132.36ms/update**。以前の同条件7,515.90msに対し約5.1%短縮（以前の保存結果との比較）。`arc-reprofile-b16-20260921.json`。

Release solution buildはwarning/error 0。Arc関連358件に合格し、その後追加したpublication dispatch 4ケースを含むcodec10件も全合格（Arc合計362ケース、重複除外）。非Arc Coreは1,195 passed / 109 skipped、Integrationは427 passed / 6 skipped、Benchmarks.Testsは35 passed。skipはCUDAが利用できない条件を含む。初回の新規テストでは未対応のpure bfloat16 modeを誤ってfixtureに含めて4件失敗したため、Arcの正式対応3mode（float32/mix16_32/mix8_32）で再実行した。数値許容誤差は変更していない。

TRXは`benchmark-results/test-results/arc-reprofile-{core,codec-dispatch,nonarc,integration,benchtests}-20260921.trx`。長時間soak、実コーパスでの収束、CUDA実機との同時期比較は未実行。

## CUDA FlashAttentionとの相違（未完了部分）

現行Arc T1024の選択経路は`Tensor.ArcResidentOperations.cs`の`ArcBatchedAttention`。`StreamingAttention=true`というoption名だけではFlashAttentionを使ったことにならない。QKはBF16 XMXだが、P/V、dP、dQ、dK/dVは現行のFP32 tile kernel。PとDSをhead tileごとのglobal bufferへ書き出し、softmax/微分も別kernelで実行する。

CUDAの`native/flash_attention.cu`には、QK→online softmax→PVをshared memoryで完結させるTensor Core経路、K/V ping-pongとcp.async、backward key-ownerの実装がある。Arcで使えているのはcausal領域省略、tile化、QKの行列命令、softmax統計保存と再計算、DKV融合とatomic回避等であり、**CUDAのFlashAttention全体は移植済みではない**。

最終profileのDKVは3,695.91ms/update、dQは1,467.13ms、PVは1,194.29ms、softmax微分は1,133.39ms。次の大きな候補は、Arc向けonline-softmax融合とXMX活用でglobal中間行列の往復を消すこと。単体の高速化見込みをCUDA同等性能の達成と取り違えない。

再現例（出力先は未使用名を指定）：

```powershell
dotnet run --project .\NNtrain.Benchmarks -c Release --no-build -- --probe-arc-transformer .\training.transformer.json .\benchmark-results\arc-repeat-new.json --warmup 1 --steps 5 --profile
```

旧経路とのA/Bには`--weight-workspace 64 --cache-backward off --packed-relu off --coalesced-codec off`を指定する。
