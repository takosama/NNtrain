# DRN追加高速化：2026-09-06

## 採用結果

前回22.59%短縮した版を基準に、さらに **step p50 1,038.20 → 1,012.36 ms
（2.49%短縮）**。モデル数式、量子化精度、optimizer、本番JSONは変更していない。
実FineWebの収束やloss停滞の改善を測った結果ではない。

| 指標 | 今回の旧版 | 採用版 |
|---|---:|---:|
| step中央値（各40測定点） | 1,038.20 ms | 1,012.36 ms |
| step平均 | 1,039.13 ms | 1,013.05 ms |
| tokens/s（平均時間から算出） | 31,534 | 32,346 |
| GPU0 allocator所有量 | 5,684.75 MiB | 5,694.87 MiB |
| GPU0物理VRAM使用量（測定終了時） | 7,276.5 MiB | 7,290.5 MiB |
| 測定stepのnative allocation/free | 0/0 | 0/0 |
| 各runのGraph capture/replay/fallback | 0/20/0 | 0/20/0 |

物理VRAMは約14 MiB増加。GPU1にはデスクトップの使用量・変動もある。
この表は長時間学習全体や保存時のピークVRAM保証ではない。

## 条件と再測定

- Windows x64、.NET 10 Release、2×RTX 3070 Ti、CUDA 13でnative build。
- 凍結設定：`docs/benchmarks/drn-speed-2026-09-06/config.json`。
- DRN 48層、width512、hidden1536、K/V32、vocabulary11500。
- batch16（8/8）、context2048、accumulation1、dropout0.1、seed1234。
- mix8_32 block128。BF16 operands / FP32状態・勾配・optimizerを維持。
- 通常Muon、momentum .95、Nesterov、NS5毎更新、LR .001。
  補助AdamW LR .0003、beta .9/.95、weight decay .01、clip1。
- 同一managedバイナリ、native DLLのみ交換。各run新規プロセス、10 warmup + 20 measured。
- 実行順 A1 → B1 → B2 → A2。各run p50：
  A1=1038.1974、B1=1009.3184、B2=1013.7373、A2=1037.1253 ms。
- 同じ固定seedの合成token/target、全長・全target有効、LR固定。
  dataset/BPE読込、scheduler、checkpoint、HTML、生成時間は測定に含めない。
- 各測定後の同期付き診断5 stepsは上表の通常step統計から除外。
- H2D 190,493,664 bytes / D2H 190,231,552 bytes per stepは前後同じ。
  この構成のGPU間host-staging通信は残っており、token/lossだけの転送ではない。

同期付き診断の10 steps平均（通常経路の重なりを外すので上表とは別の測定）：

| 区間 | 旧版 ms | 採用版 ms |
|---|---:|---:|
| forward + backward + reduce | 913.60 | 888.40 |
| clip | 1.75 | 1.88 |
| Muon | 93.59 | 96.95 |
| AdamW | 26.82 | 30.59 |
| 合計 | 1,035.93 | 1,018.01 |

今回の変更対象はforward/backward側。optimizerは変更しておらず、表の変動は
別実行での測定値である。カーネル単体12%短縮を学習全体12%とは報告しない。

## 採用した実装

1. **DRN backwardのtoken単位事前計算**
   - value行ごと・time loop内で繰り返していたQ/Kのtanh・正規化係数と
     V/gate/betaの変換を、各tokenにつき一度だけFP32で計算。
   - recurrent adjointとQ/K atomicの式・蓄積型はそのまま。
   - 追加scratchはこの形状で10.125 MiB。transient leaseで再利用、上限16 MiB。
   - K16–32かつbatch×V>=128、scratch上限内のBFP8/混合経路のみ使用。
     その他の形状、V2/V3、旧ABIは既存CUDA経路へfallback（CPUへ戻さない）。
   - ABI 1.33 `nntrain_drn_backward_prepared` を追加。既存exportは維持。
2. **狭い幅のbias勾配を8列tile化**
   - rows>=4096、width128–512だけ32列tile→8列tile。
   - 各列を単一blockが所有し、FP32部分和をblock内で集約。
     atomic/global workspace追加なし、既存bias gradientへの加算を維持。
   - width1536は実測で悪化したため適用しない。

## 単体ベンチと不採用案

CUDA0、DRN backwardは5 warmup + 20 measured、biasは5 + 50。
同じ入力を使用し、gradientのzeroingも両条件の時間に含む。

| 案・形状 | 旧版 ms | 候補 ms | 判断 |
|---|---:|---:|---|
| DRN B8/T2048/K32/V32、4行をまとめてatomic削減 | 3.8567 | 7.3442 | 不採用。K1では誤差基準も超過 |
| 同形状、Q/Kの二段階集約 | 3.8567 | 4.3608 | 不採用。128 MiB scratch、約13%遅い |
| 同形状、token単位事前計算 | 3.8567 | 3.3908 | 採用。12.08%短縮 |
| BF16 bias rows16384/width512、8列tile | 0.06941 | 0.04038 | 採用 |
| BF16 bias rows16384/width1536、8列tile | 0.10263 | 0.22822 | 不採用。32列tileのまま |

不採用カーネルはソースから除去済み。実験用DLLはbenchmark-results内だけで、
本番に配置していない。単体ベンチの小さい差にはlaunch/clock変動も含まれるため、
採用判定は通常学習経路の上記A/B再測定による。

## 数値・回帰検証と既知の未解決事項

- 新旧DRN backwardの6形状を全要素比較。既存gradientと非ゼロstate gradientを含む。
  採用経路の最大絶対差1.94e-7、相対L2差4.77e-8以下。
  fallbackを含む全6形状も既存6e-5基準内。許容誤差は広げていない。
- bias float32/BF16、端数列・端数行、既存gradientへの加算を検証。
  ランダム値のFP64 reference比較も1e-5以内。
- 関連テスト：142合格、1件は下記の既知問題として明示skip。
  production形状の2GPU session、長いpromptの生成→学習再開を含む。
  全solutionテストや長期soakを実施したという意味ではない。
- Release buildはwarning/error 0。SM80/86/89/90とPTXをビルド。
  実機検証はSM86のみ。

追加した **B8/T65/K1/V19のCPU対CUDA比較** は最大絶対差0.00152587891で不合格。
新旧DLLとも同じ差が再現する既存経路の数値差で、今回のprepared経路はK1を対象外にしている。
原因の詳細修正は未実施。テストは `ScalarKeyLongSequenceMatchesCpuKnownIssue` として
理由付きskipで残し、元の許容誤差6e-5は維持した。
初回の142合格/1失敗と旧DLL再現のTRXも保存。全件合格とは扱わない。

## 成果物・再現

- 生JSON：`docs/benchmarks/drn-backward-2026-09-06/`。
- 実験バイナリ、全要素比較binary、TRX：`benchmark-results/drn-backward-2026-09-06/`。
- 旧DLL SHA256：`6D1115B44194C1302CDF0163D2988DB4C152DA9E8517EC8375DE7B4E55437F82`。
- 採用DLL SHA256：`6390C6AC349401D05C4B8D5524E4FAF9DC637B44E134607F8BD57994718B05BF`。

```powershell
dotnet run -c Release --project .\NNtrain.Benchmarks -- --profile-drn-json `
  .\docs\benchmarks\drn-speed-2026-09-06\config.json 10 20 nodetail <新規の出力.json>
```

既存runのモデル、tokenizer、checkpoint、loss履歴には触れていない。
