# Qwen GGUF 生成速度の比較

## 測定範囲

同じベースモデル・2 GPU（GPU 0: Intel(R) Arc(TM) B580 Graphics; GPU 1: Intel(R) Arc(TM) B580 Graphics）・16 tokenの同じprompt・greedy生成・LoRAなしで比較。各runで128 tokenを生成。EOSによる早期終了はしない。
最初の出力tokenから最後の出力tokenまでをdecode時間とし、`(生成数−1)×1000 / decode ms`でtok/sを再計算した。prompt処理と最初のtoken生成、モデルロード、logits保存はこの指標に含めない。
先頭1 runを速度集計から除外。除外runも生成ID・時刻・logits整合性の検証対象。実行順は各JSONの時刻に記録されており、別プロセス/別セッション間の変動を含む。
全モデルのdecode測定であり、既存のprefillやIQ2演算単体の倍率とは別の値。

カーネルtiming収集：False、詳細profiling：False。旧/新で同じ設定であることを検査。

## 生成速度

| 経路 | 採用run数 | 中央値 tok/s | 平均 tok/s | 母標準偏差 | 最小–最大 tok/s | 中央値の旧比 | token間隔中央値 ms | token間隔p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 3 | 17.792 | 17.792 | 0.009 | 17.781–17.803 | 基準 | 56.147 | 56.704 |
| packed | 3 | 19.722 | 19.720 | 0.016 | 19.700–19.739 | 1.108× | 50.649 | 51.107 |

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
| original | [pilot-original.json](<../pilot-original.json>) | dc347585dd4c2182b3438c6fc60aca0df5d712a25d4b92041720222b0c429ab1 | 2026-10-08T08:48:30.7802264+00:00 | 2026-10-08T08:49:31.9191103+00:00 |
| packed | [pilot-packed.json](<../pilot-packed.json>) | 81c009703034ebc4290bcbac21cc349adb1e9f2b86ba27bf200ca3e452103f23 | 2026-10-08T08:49:32.2077134+00:00 | 2026-10-08T08:50:13.7955865+00:00 |

| 測定バイナリ | SHA-256 |
| --- | --- |
| NNtrain.Core | 9f2d833d1b435e03921104c538d8f4220faec7356830b809d086608285bed8a4 |
| NNtrain.Arc | 684c0f73c28e467a95abda738afee83a0db70af566a39b0de3f95c872e3113ed |
| NNtrain.Benchmarks | a1df3b1af67c618cb29d0642f418cf897744cfa2ca6061f6bed8b8fad5abd668 |

元JSONとlogitsファイルは変更していない。全rawデータ、再計算した各run・token間隔、設定差、完全一致の結果は以下に保存。
[decode-summary.json](<decode-summary.json>)
