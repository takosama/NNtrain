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
| original | 3 | 17.393 | 17.395 | 0.003 | 17.392–17.399 | 基準 | 57.461 | 57.849 |
| split | 3 | 17.811 | 17.813 | 0.007 | 17.806–17.823 | 1.024× | 56.145 | 56.602 |

tok/sの統計単位はrun。token間隔は採用runの全区間をまとめ、p95は昇順の位置 `(n−1)×0.95` を線形補間して計算。倍率1超が高速化を示す。少数runの記述統計であり、統計的有意差や普遍的な高速化を示すものではない。

## 一致検証と設定差

### split

全run・全生成token ID：旧と完全一致。
全語彙logits：16 snapshotの全バイトが旧と一致。SHA-256、サイズ、有限値、入力continuation、argmax IDを再検査。

| 設定 | 旧 | 候補 |
| --- | --- | --- |
| InferenceSplitOutputHead | False | True |

## 証拠と再現条件

モデルSHA-256：`676db8ccd38035117e705858fd591964005cb0a921b39255b0691d8339a8dfe2`

| 経路 | 生JSON | SHA-256 | 開始UTC | 終了UTC |
| --- | --- | --- | --- | --- |
| original | [final-greedy-original.json](<../final-greedy-original.json>) | 29ebda3aef2a19ed7a695ac0799f39576160851f947a96805d1ff632c3a6b93c | 2026-10-08T07:58:08.0923061+00:00 | 2026-10-08T07:58:51.3475915+00:00 |
| split | [final-greedy-split.json](<../final-greedy-split.json>) | 230b067eeca93f638902e980cb47eb11c0e12ccd64a06ad5c2e12825950425f5 | 2026-10-08T07:57:24.6786722+00:00 | 2026-10-08T07:58:07.8343306+00:00 |

| 測定バイナリ | SHA-256 |
| --- | --- |
| NNtrain.Core | d1a35498a0c5fbb6ab58896c30a490d935be935225c12510b8e94d74e74c7cef |
| NNtrain.Arc | b5da050044efffbca061b864fd59d1d18d1b2c9cf448df9e5047f89377ee61b9 |
| NNtrain.Benchmarks | 1f4c6dfd85edf4b832467474c9843ee802e335a24c20f48274a8dcc74abe6b85 |

元JSONとlogitsファイルは変更していない。全rawデータ、再計算した各run・token間隔、設定差、完全一致の結果は以下に保存。
[decode-summary.json](<decode-summary.json>)
