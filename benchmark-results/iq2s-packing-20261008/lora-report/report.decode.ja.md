# Qwen GGUF 生成速度の比較

## 測定範囲

同じベースモデル・2 GPU（GPU 0: Intel(R) Arc(TM) B580 Graphics; GPU 1: Intel(R) Arc(TM) B580 Graphics）・16 tokenの同じprompt・greedy生成・LoRA適用（lora_rintya_qa142_Qwen3.8-27B-Uncensored-noMTP-IQ2_M_2epoch.gguf、SHA-256 `9bc5079fc5eb289358f3d39de0b029655382de7559d1b6baac061a0b4172dbfb`）で比較。各runで128 tokenを生成。EOSによる早期終了はしない。
最初の出力tokenから最後の出力tokenまでをdecode時間とし、`(生成数−1)×1000 / decode ms`でtok/sを再計算した。prompt処理と最初のtoken生成、モデルロード、logits保存はこの指標に含めない。
先頭1 runを速度集計から除外。除外runも生成ID・時刻・logits整合性の検証対象。実行順は各JSONの時刻に記録されており、別プロセス/別セッション間の変動を含む。
全モデルのdecode測定であり、既存のprefillやIQ2演算単体の倍率とは別の値。

カーネルtiming収集：False、詳細profiling：False。旧/新で同じ設定であることを検査。

## 生成速度

| 経路 | 採用run数 | 中央値 tok/s | 平均 tok/s | 母標準偏差 | 最小–最大 tok/s | 中央値の旧比 | token間隔中央値 ms | token間隔p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 3 | 16.665 | 16.663 | 0.004 | 16.657–16.668 | 基準 | 59.983 | 60.448 |
| packed | 3 | 17.480 | 17.444 | 0.073 | 17.342–17.509 | 1.049× | 57.187 | 57.594 |

tok/sの統計単位はrun。token間隔は採用runの全区間をまとめ、p95は昇順の位置 `(n−1)×0.95` を線形補間して計算。倍率1超が高速化を示す。少数runの記述統計であり、統計的有意差や普遍的な高速化を示すものではない。

## 一致検証と設定差

### packed

全run・全生成token ID：旧と完全一致。
全語彙logits：16 snapshotの全バイトが旧と一致。SHA-256、サイズ、有限値、入力continuation、argmax IDを再検査。

| 設定 | 旧 | 候補 |
| --- | --- | --- |
| InferencePackedIq2Decode | False | True |

## 証拠と再現条件

モデルSHA-256：`676db8ccd38035117e705858fd591964005cb0a921b39255b0691d8339a8dfe2`

| 経路 | 生JSON | SHA-256 | 開始UTC | 終了UTC |
| --- | --- | --- | --- | --- |
| original | [final-lora-original.json](<../final-lora-original.json>) | d313108f2857a1096bc1aa734457a6f3015a390d46a14967c7901a02876661c3 | 2026-10-11T04:17:35.7234829+00:00 | 2026-10-11T04:18:24.7621542+00:00 |
| packed | [final-lora-packed.json](<../final-lora-packed.json>) | e2862265d3a3032178b0def6f64bccba29dd95f9ee51a73d9c380dde1b774004 | 2026-10-11T04:18:25.1722315+00:00 | 2026-10-11T04:19:14.2044752+00:00 |

| 測定バイナリ | SHA-256 |
| --- | --- |
| NNtrain.Core | 5cea29710163128134c6031dd678cfcddc808cdbcf33a31dfb9e3a0907c9d72d |
| NNtrain.Arc | 582be5070694db7cd743365c232a5ea7aae327a62e85f5d8dcf8e746208217c2 |
| NNtrain.Benchmarks | f38b2438707b1265a67258d9973520aa868b7ba861c278874782b34c4b8a29a5 |

元JSONとlogitsファイルは変更していない。全rawデータ、再計算した各run・token間隔、設定差、完全一致の結果は以下に保存。
[decode-summary.json](<decode-summary.json>)
