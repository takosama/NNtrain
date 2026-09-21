# Arc B580: microbatchとXMX epilogueの実測最適化（2026-09-21）

最終defaultで **10,644.4 tokens/s**、p50 **24.627秒/update**。
前回最終版8,833.4 tokens/sから **20.50%向上**、update時間は17.01%短縮。
peak割当は10,591.49 → 5,365.87 MiB（49.34%削減）。
本番JSONの16×16を維持し、kernel側の採用経路を既定有効にした。
最初の31.918秒基準比でも1.296倍であり、要求された2.5倍には未達。

## 条件と判断基準

前回の [排他的wall timeline報告](arc-exclusive-timeline-2026-09-21.md) を起点にした。
今回のユーザー指定は「microbatchが小さくなってもtokens/sが上がればよい」。
実効batchは256に固定し、microbatch × gradient accumulationだけを交換する。
層数・モデル幅・文脈長を減らして速く見せる比較はしていない。

- Intel Arc B580 12 GB、driver 32.0.101.9030、Windows、.NET 10 Release。
- Transformer: 90,507,500 parameters、32層、width 512、hidden 1536、16 heads、D32、語彙11,500。
- sequence 1024、tied embedding、dropout 0.1、seed 1234。
- `mix8_32` block32。BFP8保存、BF16 XMX operand、FP32 accumulation/gradient/master。
- Muon固定NS5、LR 0.001、補助AdamW LR 0.0003、weight decay 0.01。
- 各候補は1 warmup + 3 measured optimizer updates。GPU仕事は直列。
- 固定seedの合成full-length tokenを使う**訓練ベンチ**。
  実FineWebの読み込み・tokenization・checkpoint・HTML・生成・LR scheduleは含めない。
  tokens/s = 256 × 1024 / update秒。loss/勾配の有限性も確認。
- 元のB64/A4 JSON SHA256:
  `EC42E94F3A9BA43DBDB90E5A44368798C422F76AD5330BC936F701261E7D64C5`。
  作業中にproduction JSONはB16/A16となっていたため、その変更を保持した。
  現在のSHA256:
  `D52B38CE5D91EF53A1643D8E7061D0E974872BA6D1DDD69C5047DDB10B0C9E39`。
  各JSON artifactに実効shape、全Arc options、binary hashを記録している。

## 段階別結果

| 経路 | microbatch × accumulation | p50 ms/update | tokens/s | peak割当 MiB |
|---|---:|---:|---:|---:|
| 前回最終版、詳細profileなし | 64 × 4 | 29,676.34 | 8,833.4 | 10,591.49 |
| microbatch交換のみ | 32 × 8 | 26,398.60 | 9,930.2 | 9,107.24 |
| microbatch交換のみ | 16 × 16 | 26,239.32 | 9,990.5 | 5,365.87 |
| 16×32 GEMMとBFP8保存を融合 | 16 × 16 | 25,936.30 | 10,107.2 | 5,365.87 |
| 同上 | 8 × 32 | 25,809.03 | 10,157.1 | 3,491.93 |
| 広いGEMMタイルと融合を併用 | 16 × 16 | 24,632.69 | 10,642.1 | 5,365.87 |
| 同上 | 8 × 32 | 24,909.80 | 10,523.7 | 3,491.93 |
| 最終default、2 warmup + 4 timeline測定 | 16 × 16 | 24,627.35 | 10,644.4 | 5,365.87 |

後続候補は詳細profile有効。前回baselineには詳細profileなしの結果を掲げ、
今回に有利なprofiler overheadの比較にならないようにした。
8×32は初期融合版では僅かに速かったが、最終タイルと組み合わせると16×16が速い。
**組み合わせを実測した上で16×16を選択**。batchを小さくするほど速いわけではない。

対応artifact（すべて `benchmark-results/`）:

- `arc-wall-unprofiled-final-20260921.json`
- `arc-next-b32-a8-20260921.json`
- `arc-next-b16-a16-20260921.json`
- `arc-next-b16-fused-20260921.json`
- `arc-next-b8-a32-20260921.json`
- `arc-next-b16-wide-20260921.json`
- `arc-next-b8-wide-20260921.json`
- `arc-throughput-final-timeline-20260921.json`

## 採用した実装

### 1. microbatch交換でactivation再計算と割当圧力を除去

B64は11層のfull-block再計算と全FFNの再計算が必要だった。
B16ではsaved activation推定3,267 MiBで、両方とも不要になる。
実効batch256を維持するためoptimizer更新頻度は変更しない。
ただしdropout seedの消費順、浮動小数点の加算順はmicrobatch変更で変わるため、
旧B64とのbitwise同一学習軌跡や収束改善は主張しない。

### 2. GEMMのFP32結果をVRAMへ往復させずBFP8を生成

`xmx_bfp8_epilogue.cl` はDPASのFP32 accumulatorからblock32のmax/scale/RNEを計算し、
packed payloadとscaleへ直接書く。従来のFP32出力allocation・write・readと
独立publication kernelを除去する。BF16丸め、DPAS加算順、bias/ReLU順序は維持。
非有限値とscale underflowは従来どおりnumeric statusに記録してoptimizer commitを拒否する。
block128、FP32/BF16出力、非対応shapeは既存経路を使う。暗黙CPU fallbackは追加していない。

`arc-bfp8-epilogue-wide-20260921.json`: 2 warmup + 8 measured、両経路とも同じpacked operand。
packingはこの単体比較には含めず、従来側にはGEMM + publicationを含める。
payloadと**全scaleのbit一致**を計測外で確認。

| M × N × K (ReLUなし) | 従来GPU ms | 融合16×64 GPU ms | 短縮 |
|---|---:|---:|---:|
| 16,384 × 1,536 × 512 | 1.5735 | 1.1731 | 25.4% |
| 16,384 × 512 × 1,536 | 1.3844 | 0.9046 | 34.7% |
| 65,536 × 1,536 × 512 | 6.1522 | 4.5599 | 25.9% |

小shapeでは遅くなるので全shapeに強制しない。既定有効の`FusedBfp8Linear`で切替可能。

### 3. backwardにも形状別XMXタイルを適用

`arc-xmx-expanded-20260921.json` は6候補を旧16×32と比較。
3 warmup + 10 measured、GPU packing/allocation込み、測定中のH2D/D2Hなし。
tail、transpose、gate、bias、ReLU、非ゼロ初期値、二重accumulation、split-Kを全配列比較。

- 大きい通常行列: 16×64/WG16（従来16×32）。
- 大きいdW: 32×32/WG16。既存の2048幅splitとreduction順を変えない。
- 大語彙lossの短いM: 16×64/WG4。
- 小さいoptimizer行列、長い非転置reduction: 従来tileを維持。
- 非融合streamed fallbackは従来tileのまま。融合forwardはrow slice/offsetをサポート。

`ExpandedXmxTiles`で独立A/B可能。生のDPAS/BF16演算を変えていない。
大きいdWのdevice時間は1,442 → 1,153 ms/update、
大きいdXは1,293 → 1,064 ms/updateになった。

## 不採用: K/V backwardの16-keyタイル

最大kernelのDKVは引き続き約3,710 ms/updateを占める。
SLM量を下げて並列workgroupを増やすK16/Q16・Q32・Q64を追加し、実測した。
`arc-dkv-narrow-20260921.json`:
causal K32/Q32は0.1085 msに対し、候補は0.1682 / 0.1505 / 0.1608 ms。
意味論テストは通るが遅いので**学習dispatchへは採用しない**。
前回のFlash/XMX attention実験も有効化していない。

## メモリ・転送

B16最終候補はupdate境界でactive 1,478.14 MiB、cache 3,887.73 MiB。
2回目のupdateまでにpoolが安定し、3・4回目はnative allocation/freeがともに0。
「要求バイト数」はpoolからの再利用も数えるため、native allocationとは区別する。
peak 5,365.87 MiBはOpenCL ownerの割当集計であり、Task Managerの物理VRAM測定ではない。
スクリーンショットのすべての波形の原因を、この値だけで断定はしない。

転送はH2D 2,097,152 bytes/update（token/target）、D2H 2,636 bytes/update
（loss、clip、既存Muon scalar diagnostics）。weight/activation/gradientのreadbackは追加なし。
旧B64からD2Hが48 bytes増えたのはmicrobatch lossが4個から16個になった分。

## 最終検証

- 全solution Release build: warning 0 / error 0。
- Arc Core回帰 **442件合格、失敗0、skip0**:
  `arc-throughput-final-core-20260921.trx`。前回から26件追加。
  既存の許容誤差を広げず、forward・累積gradient・master/optimizer更新、
  block/dtype fallback、streamed row slice/tail、非有限値、転送・resource寿命を検証。
- 無効のFlash/XMX実験用 `Category=ArcAttentionExperimentalAcceptance` 6件は
  前回同様に明示的に除外。未解決の実験経路を合格扱いにはしていない。
- Arc CLI integration **19件合格**:
  `arc-throughput-final-integration-20260921.trx`。
  隔離した小さなFineWeb fixtureで訓練、保存、auto-resume、生成、HTML継続を確認。
- Benchmark tests **46件合格**:
  `arc-throughput-final-benchtests-20260921.trx`。
  microbatch/accumulation交換の実効batch上限、正数、overflow回避を含む11件追加。
- CUDAは現在の環境で使用できず、対CUDAの新しい比較は未実行。
  210/2100-step soakや本番corpusの収束検証ではない。
- dataset、tokenizer、checkpoint、loss履歴は変更していない。commit/pushは行っていない。

### 最終wall timeline

`arc-throughput-final-timeline-20260921.json` と4本の `.step-3`～`.step-6.trace.json.gz`。
2 warmup後の4 updatesは24,629.58 / 24,627.09 / 24,626.30 / 24,627.62 ms。
外側stopwatch合計98,510.5899 ms、明示trace窓の合計98,510.5048 ms。
差0.0851 msはtrace窓の開始・終了処理であり、未計測kernelではない。
各窓の排他的coverageは100%、314,106 device events、missing/opaque upload/GPU overlap/未帰属host時間はすべて0。
clock bracket uncertaintyは0.02845～0.02860 ms。

| 排他的区間 | 平均 ms/update | wall比率 |
|---|---:|---:|
| GPU attention | 10,932.84 | 44.39% |
| GPU GEMM（融合epilogue込み） | 6,472.48 | 26.28% |
| GPU pack/decode/publication | 3,784.83 | 15.37% |
| GPU normalization/reduction | 2,278.62 | 9.25% |
| GPUその他/loss | 866.35 | 3.52% |
| GPU転送H2D/D2H/D2D | 2.13 | 0.01% |
| 計測laneのGPU idle（host/queue待ち等） | 290.38 | 1.18% |

丸め前の全内訳はJSONに保存した。host wait中にGPUが実行している時間を
追加でwall時間へ足していない。融合codecの時間はGEMM側に入るため、
pack/publicationカテゴリの減少をすべて「消滅した仕事」とは解釈しない。
4 updatesともnative allocation/freeは **0 bytes**、peak割当5,365.87 MiBで一定。
H2D/D2Hも全4回でそれぞれ2,097,152 / 2,636 bytesと一致した。

残る単独最大kernelはDKVの3,710.8 ms（約15.1%）。Amdahlの式では、これだけを
2倍速にしても全体は約1.081倍に留まる。さらに大きな改善にはattention全体と
pack/normalizationの改善が必要で、意味論テストに落ちる低精度経路を有効にして
目標達成扱いにはしない。短時間ベンチは長期収束や長時間VRAM安定性の証明ではない。

## 再測定

学習を停止し、既存artifactと異なる出力名を使う。

```powershell
dotnet build .\NNtrain.Benchmarks -c Release
dotnet run --project .\NNtrain.Benchmarks -c Release --no-build -- `
  --probe-arc-transformer .\training.transformer.json .\benchmark-results\arc-throughput-repeat.json `
  --warmup 2 --steps 4 --timeline
```

今回のkernel変更だけをOFFにして同じmicrobatchで比較するには
`--fused-bfp8-linear off --expanded-xmx off`を付ける。
旧historical modeも明示的に両機能をOFFにしており、default変更に追従させていない。
