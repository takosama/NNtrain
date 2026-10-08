# Qwen GGUF 生成速度の比較

## 測定範囲

同じベースモデル・2 GPU（GPU 0: Intel(R) Arc(TM) B580 Graphics; GPU 1: Intel(R) Arc(TM) B580 Graphics）・16 tokenの同じprompt・greedy生成・LoRAなしで比較。各runで64 tokenを生成。EOSによる早期終了はしない。
最初の出力tokenから最後の出力tokenまでをdecode時間とし、`(生成数−1)×1000 / decode ms`でtok/sを再計算した。prompt処理と最初のtoken生成、モデルロード、logits保存はこの指標に含めない。
先頭1 runを速度集計から除外。除外runも生成ID・時刻・logits整合性の検証対象。実行順は各JSONの時刻に記録されており、別プロセス/別セッション間の変動を含む。
全モデルのdecode測定であり、既存のprefillやIQ2演算単体の倍率とは別の値。

カーネルtiming収集：False、詳細profiling：False。旧/新で同じ設定であることを検査。

## 生成速度

| 経路 | 採用run数 | 中央値 tok/s | 平均 tok/s | 母標準偏差 | 最小–最大 tok/s | 中央値の旧比 | token間隔中央値 ms | token間隔p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 3 | 17.451 | 17.451 | 0.005 | 17.444–17.457 | 基準 | 57.285 | 57.486 |
| rms | 3 | 17.330 | 17.331 | 0.001 | 17.330–17.333 | 0.993× | 57.692 | 57.881 |
| grid3 | 3 | 10.347 | 10.346 | 0.002 | 10.344–10.348 | 0.593× | 96.651 | 96.941 |

tok/sの統計単位はrun。token間隔は採用runの全区間をまとめ、p95は昇順の位置 `(n−1)×0.95` を線形補間して計算。倍率1超が高速化を示す。少数runの記述統計であり、統計的有意差や普遍的な高速化を示すものではない。

## 一致検証と設定差

### rms

全run・全生成token ID：旧と完全一致。
全語彙logits：16 snapshotの全バイトが旧と一致。SHA-256、サイズ、有限値、入力continuation、greedy IDを再検査。

| 設定 | 旧 | 候補 |
| --- | --- | --- |
| InferenceFusedResidualRms | False | True |

### grid3

全run・全生成token ID：旧と完全一致。
全語彙logits：16 snapshotの全バイトが旧と一致。SHA-256、サイズ、有限値、入力continuation、greedy IDを再検査。

| 設定 | 旧 | 候補 |
| --- | --- | --- |
| InferenceResidentIq2Panels | False | True |
| InferenceXmxGgufBslmPrefill | True | False |

## 参考：単一DeltaNet更新の実験

以下は別の単体測定。上の全モデルtok/sには加算・乗算しない。遅い候補は高速化候補として採用しない。

出典：[iq2-prefill-decode-final.trx](<C:/Users/takos/source/repos/NNtrain/benchmark-results/iq2s-production-20261008/tests/iq2-prefill-decode-final.trx>)

| profiling | 旧wall ms/更新 | 候補wall ms/更新 | 旧/候補倍率 | 出力・状態 |
| --- | --- | --- | --- | --- |
| False | 0.037734 | 0.045066 | 0.837× | bit一致 |
| True | 0.058041 | 0.066375 | 0.874× | bit一致 |

## 証拠と再現条件

モデルSHA-256：`676db8ccd38035117e705858fd591964005cb0a921b39255b0691d8339a8dfe2`

| 経路 | 生JSON | SHA-256 | 開始UTC | 終了UTC |
| --- | --- | --- | --- | --- |
| original | [original-provisional.json](<C:/Users/takos/source/repos/NNtrain/benchmark-results/iq2s-production-20261008/decode/original-provisional.json>) | b1d5d7c2e5e9abee9710546090e9b0746778c1caed995291f658e81ca29a3f6b | 2026-10-08T05:13:49.8386348+00:00 | 2026-10-08T05:14:18.5433134+00:00 |
| rms | [rms-provisional.json](<C:/Users/takos/source/repos/NNtrain/benchmark-results/iq2s-production-20261008/decode/rms-provisional.json>) | 8d104151317e93e5587466b060400b979ebfa5604d102e445bdbfc5315367777 | 2026-10-08T05:12:46.5758451+00:00 | 2026-10-08T05:13:15.4183848+00:00 |
| grid3 | [grid3-provisional.json](<C:/Users/takos/source/repos/NNtrain/benchmark-results/iq2s-production-20261008/decode/grid3-provisional.json>) | fbb95d505a5b4e2c079353282607aa6ceafe396c51a9507ac9777601bcbb7ffe | 2026-10-08T05:23:05.8725975+00:00 | 2026-10-08T05:24:01.0304155+00:00 |

| 測定バイナリ | SHA-256 |
| --- | --- |
| NNtrain.Core | 33258948b33d3e40503c4a150108a9572870064c415134219728bc40c1232a62 |
| NNtrain.Arc | f72cfead8bc43c963ba4c080e4d89edc52064eb125cad6e41c6fdea534357a63 |
| NNtrain.Benchmarks | a01248e385d1577acb36420071090a2922b0d4e9c52dce2c8a83b3531e37369e |

元JSONとlogitsファイルは変更していない。全rawデータ、再計算した各run・token間隔、設定差、完全一致の結果は以下に保存。
[decode-summary.json](<C:/Users/takos/source/repos/NNtrain/benchmark-results/iq2s-production-20261008/decode-rejected-candidates/decode-summary.json>)
