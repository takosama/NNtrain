# DRN CUDA高速化 — 2026-09-03

## 結果

モデルの数式・精度・optimizer設定を変えず、DRNのCUDAカーネルを変更した。

| 条件 | 変更前 p50 | 変更後 p50 | 時間短縮 | 速度比 |
| --- | ---: | ---: | ---: | ---: |
| batch 32、各20 steps × 2回（計40点の中央値） | 1,421.36 ms | 805.71 ms | 43.31% | 1.76倍 |
| batch 48、各20 steps × 1回 | 1,559.14 ms | 1,054.53 ms | 32.36% | 1.48倍 |

batch 32の各回のp50は、旧版1,420.05 / 1,426.12 ms、新版807.83 / 802.37 ms。
測定中にユーザーのJSONがbatch 32から48に変更されたため、32の再測定には保存済み設定を使い、48は別条件として比較した。本番JSONは変更していない。

## 条件と範囲

- Windows x64、.NET 10 Release、2 × NVIDIA GeForce RTX 3070 Ti（各8 GiB）。
- ForgetMemoryDRN、語彙4096、幅512、hidden 1536、32層、context 1024、K/V各16、dropout 0.1、seed 1234。
- `mix8_32`、block 128（この時点の設定解決値）、勾配累積1。
- NekoMuon LR 0.003 / beta fast 0.9 / adaptive NS、5更新ごと。AdamW LR 0.001 / beta2 0.95。weight decay 0.01、clip norm 1。
- 各実行は5 warmup後に20 steps。NS更新4回を含む。CLIと同じ設定読込・モデル・optimizer生成、CUDA Graph、data parallelを使用。
- 同一seedの全長・全target有効の合成トークン。LRはJSONの値で固定。データセット読込、tokenizer、checkpoint、生成、実コーパス進捗に応じたLR scheduleは測定対象外。
- 20 stepsの通常経路の測定後、別の5 stepsに同期を追加して内訳を取得。さらに初回ペアのみ、eager演算別診断を1 microbatch実行した。

これは学習stepのA/B比較であり、保存・生成を含む長時間の実コーパス所要時間や収束品質を保証する測定ではない。

## アムダールの法則による優先順位

`S = 1 / ((1 - p) + p / s)`。変更前の加算可能な同期付き診断は以下。

| 区間 | 変更前 | 変更後（初回ペア） |
| --- | ---: | ---: |
| zero grad | 0.14 ms | 0.10 ms |
| forward + backward + reduce | 1,367.39 ms | 755.71 ms |
| clip | 1.50 ms | 1.31 ms |
| NekoMuon | 30.75 ms | 29.96 ms |
| AdamW | 28.67 ms | 25.11 ms |
| 合計 | 1,428.45 ms | 812.19 ms |

optimizer合計は約4.16%。そこだけ無限に速くしても全体は約1.043倍まで。一方、forward/backward/reduceは95.73%を占めるため、そちらを優先した。

同期を多用するeager演算別診断では、DRN backwardが最大だった。

| 演算別診断（遅い側のGPU、32層合計） | 変更前 | 変更後 |
| --- | ---: | ---: |
| DRN backward | 951.62 ms | 364.28 ms |
| DRN forward | 101.40 ms | 78.51 ms |

演算別診断はeager実行で追加同期がある。951.62 msを通常Graph実行の1,422.95 msで割って占有率を計算していない。順位の特定に使い、採用判断は通常経路の実測A/Bで行った。

## 採用した変更

1. `tensor_kernels.cu`: recurrent backwardのblockを256 threadsから32 threadsへ変更。各workerは引き続き1つのvalue行を全時刻にわたって所有する。batch 32では1GPUあたり16 × 16 = 256 workersが従来1 CTAに集中していたが、8 CTAに分散する。Q/Kの共有勾配は引き続きatomicで安全に合算する。
2. `flash_attention.cu`: DRNのK/Qを同じTensor Core行列積の別列に格納し、同じ更新前stateに対する2回の行列積を1回に統合。BF16 operand、FP32蓄積、read-before-writeを維持。
3. state tileを最大幅128のゼロ埋めではなく実際のK幅で格納。K=16で不要だった共有メモリ書込みを削減。16の倍数の対応形状を維持し、24や3070 Ti専用の分岐は追加していない。

native DLLをSM80/86/89/90 + PTX向けに再ビルドし、既存export検査を通した。実機性能はSM86でのみ検証した。

## 検証と注意点

- Core関連72件、benchmark設定テスト5件、合計77件合格、スキップ0。Releaseビルド警告0・エラー0。
- V2/V3/DRN、CPU対CUDA、BF16/mix8_32、1/2GPU、Graph capture/replay、勾配累積、端数batchを含む。
- Tensor Core経路はK=48/V=32、Q列0/17/47、非ゼロ初期stateで出力のBF16ビットを完全一致検証。既存の誤差許容値は広げていない。
- 全測定区間でGraph capture/fallback増加0、20 replay。step内native allocation/freeは0/0。転送にはdata parallelの明示的なhost-staged collectiveが含まれるため、転送量ゼロとはしていない。
- batch 32のVRAM使用はGPU0約6,300 MiB、GPU1約6,300–6,875 MiB（表示用途等も含む）。今回の変更による大きなVRAM削減はない。
- batch 48ではGPU0約8,144 MiB、GPU1約8,191 MiB。短時間測定は完走したが余裕はほぼなく、長時間学習・保存・生成時のOOM安全性は未保証。設定を自動的に下げてはいない。
- 210/2100-step soakは今回実施していない。量子化・atomicを含む複数GPU学習の全traceについてbitwise一致を保証していない。

## 再現と記録

```powershell
dotnet run -c Release --project .\NNtrain.Benchmarks -- `
  --profile-drn-json .\docs\benchmarks\drn-amdahl-2026-09-03\config-batch32.json `
  5 20 detail .\drn-benchmark-new.json
```

出力先は未作成のJSONを指定する。このコマンドは本番のcheckpoint・tokenizer・metricsには書き込まない。

[生の測定結果と凍結設定](benchmarks/drn-amdahl-2026-09-03/)に、batch 32の旧/新版各2回とbatch 48の旧/新版各1回を保存した。各JSONの`Samples.Total`が通常経路、`SynchronizedPhases`が加算可能な内訳。通常経路の`AdamW`欄はcomposite optimizer全体のhost境界時間なので、AdamW単独の分析には使用しない。

測定開始時HEAD: `e15c2b0` + 既存の未コミット修正。
旧native DLL SHA256: `4F3115032E28B6CF56DF2CC6F7967AE74E334ED49DE8ACC6EA3CECCC1544B3D7`。
新native DLL SHA256: `AD256EBD58001055BF2FD433479131BB2452A67BF77B4E7F860957DA93BC5B41`。
