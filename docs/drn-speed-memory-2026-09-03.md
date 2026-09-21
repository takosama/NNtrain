# DRN：実測に基づく高速化・小メモリ化（2026-09-03）

## 最終結果

現在のDRN JSONを凍結してA/B比較。batch・context・モデル・精度・optimizerは変更していない。
前回の高速化済み実装を今回の基準とし、さらに速度とメモリを改善した。

| 指標 | 今回の変更前 | 変更後（既定） | 改善 |
| --- | ---: | ---: | ---: |
| step p50（40 steps × 2回、80点を合わせた中央値） | 796.42 ms | 543.83 ms | 31.7%短縮、1.46倍 |
| step平均 | 803.77 ms | 551.35 ms | 31.4%短縮 |
| GPU0全使用メモリ（測定終了時） | 6300.5 MiB | 5260.5 MiB | 1040 MiB、16.5%削減 |
| lane allocator所有量／GPU | 4892.27 MiB | 3852.27 MiB | 1040 MiB、21.3%削減 |
| Graph固定領域／GPU | 3755.72 MiB | 2715.72 MiB | 1040 MiB削減 |

各回のp50：旧793.68 / 800.30 ms、新544.19 / 542.97 ms。
GPU1もallocator所有量・Graph固定領域は同じ1040 MiB削減。
GPU1の全使用量には画面描画等の変動があるため、それをコードによる削減量に含めていない。
VRAMは半減ではない。上表はsteady-stateの実測であり、起動・保存・生成を含む全期間のピーク測定ではない。

## 固定条件と測定範囲

- Windows x64、.NET 10 Release、2 × RTX 3070 Ti 8 GiB（SM86）。
- `training.forgetmemorydrn-wiki-jp.json` の作業開始時コピーを使用。
- batch 32（各GPU 16）、gradient accumulation 1、context 1024。
- 語彙4096、幅512、hidden 1536、32層、K/V各16、dropout 0.1、seed 1234。
- `mix8_32`、block 128。FP32 recurrent state・parameter gradient/master・optimizer統計を維持。既存のFFN専用BF16中間勾配経路もそのまま。
- NekoMuon LR 0.003、beta fast 0.9、adaptive NS、5更新ごと。AdamW LR 0.001、beta2 0.95、weight decay 0.01、clip 1。
- 最終比較は20 warmup後に40 stepsを各2回。各40 stepsにNS更新8回を含む。
- 合成・全長・全target有効の固定seed tokenを使用。同じバッチを反復するので、loss低下は実コーパスの収束改善の証拠ではない。
- 実際のCLIの設定ロード・モデル生成・optimizer生成・data parallel・CUDA Graphを使用。
- データセット読込、tokenizer、checkpoint、生成、実コーパス進捗のschedulerは測定対象外。LRは設定値で固定。
- 学習本体、checkpoint、tokenizer、loss履歴、本番JSONにはこの作業で書き込んでいない。

## アムダールの法則による優先付け

通常経路の後に、同期を追加した診断5 stepsを各回実行した。以下は2回分、計10 stepsの平均。
通常Graph実行の時間と同期付き診断時間は混同しない。

| 加算可能な同期付き区間 | 変更前 | 変更後 |
| --- | ---: | ---: |
| zero grad | 0.10 ms | 0.11 ms |
| forward + backward + reduce | 758.69 ms | 504.19 ms |
| clip | 1.17 ms | 1.21 ms |
| NekoMuon | 26.74 ms | 32.70 ms |
| AdamW | 22.58 ms | 21.73 ms |
| 合計 | 809.29 ms | 559.93 ms |

基準のforward/backward/reduceは93.75%。`S = 1 / ((1-p) + p/s)`より、optimizerだけを無限に高速化しても全体の改善は約1.065倍にとどまる。
そこでforward/backwardを優先した。NekoMuon単独の速度改善は主張しない。

追加同期のあるeager演算別診断では、DRN backwardが365.98 msで最大だった。
warp化のみで69.61 msへ低下。最終版では再計算分を含め127.75 ms、forwardは78.72 → 61.64 ms。
これらは順位・原因の診断用であり、採用判定は上記の通常経路A/Bで行った。

## 採用した実装と実験経過

| 順番（各5 warmup + 20 stepsの探索測定） | p50 | GPU0使用量 |
| --- | ---: | ---: |
| 変更前 | 802.68 ms | 6300.5 MiB |
| DRN backwardのwarp化 | 505.08 ms | 6300.5 MiB |
| ＋FP32履歴の再計算 | 573.03 ms | 5788.5 MiB |
| ＋不要なlinear出力の早期解放 | 575.37 ms | 5260.5 MiB |
| ＋forwardのQ/K正規化並列化 | 544.38 ms | 5260.5 MiB |

1. **DRN backward**：1 warpが1 value行を所有し、keyをlaneに分配。読み込みを連続化し、recurrent adjointをレジスタに保持する。K<=32に適用し、幅が大きい場合とV2/V3は既存CUDA経路を維持。Q/KのFP32 atomic加算は残っている。
2. **DRN履歴再計算**：全層の`B*T*K*V` FP32履歴を保持せず、既存の小さなBF16 projectionからbackward直前に再生成する。FP32履歴をBF16/8bitへ落とす変更ではない。元forwardのdispatch policyも保持し、再計算時に別backendへ切り替えない。
3. **linear出力の寿命短縮**：ForgetMemory層内部の、residual加算にしか使わない非ReLU linear出力は、加算後にpayloadを返却する。backward用のnode・gradient・入力・重みは保持。公開の汎用Add/Dropoutは値を破棄しない。誤って破棄後の値を読むと明示エラーになる。
4. **Q/K正規化**：K<=32では1スレッドでのtanh/norm計算をwarpへ分散。BF16 Tensor Core operand、FP32 norm/蓄積/状態は維持する。
5. **値不要な加算backward**：同サイズBFP8加算では入力のBF16 decodeをせず、勾配だけを処理する。dropout 0の境界でも早期解放と両立する。

再計算は無料ではない。探索測定では512 MiBの削減に約68 msを要した。既定では速度とメモリの両方を改善する構成を選び、この代償も隠さず記録した。
warp reductionは加算順を変更するため、変更前後の全学習traceのビット一致は保証しない。

### 速度優先の比較設定

最終版から再計算だけを無効化した構成も、同じ20 warmup + 40 stepsで1回測定した。
`speed-priority-1.json`：p50 **488.75 ms**、平均493.87 ms、GPU0 **5772.5 MiB**。
旧版に対して約38.6%短縮（1.63倍）、528 MiB削減。
既定の小メモリ構成より512 MiB多く保持して約55 ms速い。こちらは1回測定なので、2回比較した既定構成とは測定回数が違う。

```powershell
# 速度優先：FP32履歴は保持。不要なlinear出力の早期解放は有効のまま。
$env:NNTRAIN_DISABLE_DRN_STATE_RECOMPUTATION = '1'
dotnet run -c Release --project .\NNtrain.Cli -- --config .\training.forgetmemorydrn-wiki-jp.json

# 次回起動を既定の小メモリ構成へ戻す
Remove-Item Env:\NNTRAIN_DISABLE_DRN_STATE_RECOMPUTATION -ErrorAction SilentlyContinue
```

この比較設定を本番環境変数へ恒久設定してはいない。既定ではより小メモリの構成を使う。

## 検証

- Core **1186件**、Integration **343件**、benchmark設定等 **31件**、合計 **1560件合格・スキップ0**。
- `NNTRAIN_DRN_PRODUCTION_TEST=1`を指定して、本番56M形状の2GPU BF16/mix8_32、7更新・端数batch・Graph再実行・session解放を検証。1GPUは小型構成で検証した。
- CPU/CUDA forwardとgradient比較はK=1/16/17/32/33等を追加し、既存の許容誤差を広げていない。
- 保存履歴と再計算履歴をK/V=16/16、17/13、48/32（長さ257/33/65）で**FP32ビット一致**検証。
- 早期解放の有無でloss完全一致、gradient差1e-5以内、live allocation削減をdropout 0と0.2で検証。
- ReLU出力の誤解放・破棄後の値再利用は拒否。解放の二重呼び出しとその後のbackwardも検証。
- Release buildは警告0・エラー0。native DLLはSM80/86/89/90 + PTXを生成、既存export検査合格。SM86以外の実機性能は未測定。
- 最終A/Bの各40 stepsでGraph capture/fallback増加0、replay40、native allocation/free **0/0**。
- 210/2100-step soak、長時間の保存・生成を含む本番run、実コーパス収束品質の比較は今回実施していない。

## 記録と再実行

[凍結設定・生JSON・ログ](benchmarks/drn-memory-2026-09-03/)に保存した。
最終値は`baseline-final-1/2.json`と`candidate-final-1/2.json`の`Samples.Total`から集計。
同期付き内訳は`SynchronizedPhases`、メモリは`Vram[].Allocator`と`GraphAfter`を参照する。

```powershell
dotnet run -c Release --project .\NNtrain.Benchmarks -- `
  --profile-drn-json .\docs\benchmarks\drn-memory-2026-09-03\config.json `
  20 40 nodetail .\drn-benchmark-new-result.json
```

出力先は未作成のファイルを指定。既定では小メモリ化2種を有効化する。
診断用の`NNTRAIN_DISABLE_DRN_STATE_RECOMPUTATION=1`で履歴保持との比較ができる。
`NNTRAIN_DISABLE_EXCLUSIVE_LINEAR_OUTPUT_RETIREMENT=1`で早期解放を無効にできる。
どちらも起動時に一度だけ読むため、通常stepの環境変数参照は増やしていない。
基準の再実行はこの2つを無効化し、ベンチ専用出力フォルダのDLLだけを旧版へ差し替えた。各回の終了時に新版へ戻している。

- 作業開始HEAD：`e15c2b0` + 既存未コミット修正。
- 旧DLL SHA256：`AD256EBD58001055BF2FD433479131BB2452A67BF77B4E7F860957DA93BC5B41`。
- 新DLL SHA256：`BA332071A841604AA574DE4034D7C3C4FA80943AA069C7E2F555C4C89AE2573E`。
- 学習CLIにも新版を配置済み。実行中のプロセスではなく、次回起動から反映される。
