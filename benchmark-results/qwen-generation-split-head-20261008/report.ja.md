# Qwen 27B テキスト生成速度の改善

## 結果

Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf の生成中 tok/s を改善した。
Q5_K 出力層の語彙を2 GPUで分担する経路を追加し、GUIの2 GPU推論で有効にした。

| 生成設定 | 旧 tok/s 中央値 | 新 tok/s 中央値 | 改善 | 旧平均 ± 母標準偏差 | 新平均 ± 母標準偏差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 17.393 | 17.811 | **+2.40%** | 17.395 ± 0.003 | 17.813 ± 0.007 |
| GUI通常sampling | 17.216 | 17.648 | **+2.51%** | 17.213 ± 0.004 | 17.642 ± 0.011 |

GUI通常samplingは temperature=0.6、top-p=0.95、top-k=20、seed=5429。
2×Intel Arc B580、driver 32.0.101.9030、Windows 10.0.26200、.NET 10.0.11。
同じ16 tokenのpromptから128 tokenを4回生成し、先頭1回を除いた3回を集計した。
LoRAなし、EOSによる早期終了なし、profilingなし。旧/新の測定バイナリは同一で、設定差は `InferenceSplitOutputHead` のみ。

速度は最初の出力から最後の出力までの `(128−1)/秒`。
モデルロード・prompt処理・最初のtokenまでの待ち時間は含まない。
少数runでの実測値であり、他GPU・他モデル・長い文脈で同じ倍率を保証するものではない。
この測定のprompt長は16 tokenであり、4096/8192 tokenのprefill改善率を示す値でもない。

## 実装

- 248,320語彙のQ5_K出力層を124,160行ずつ処理する。同じ量子化デコード式・FMA順序・SG16 reductionを維持。
- 正規化後の入力をpeerへ送り、元GPUの後半とpeerの前半を並行して計算する。結果は従来と同じ全語彙logitsへ結合する。
- peer側に量子化重み437,043,200 bytes（**416.8 MiB**）を追加常駐。入力・bias・出力用も含む追加live allocationは438,056,960 bytes（417.8 MiB）。
- 元の出力重みを保持し、1 GPU・非対応形式・出力層LoRA・学習・メモリ予算不足では既存経路を使用する。
- 画像モデルなどの外部VRAM予約がpeerの余裕を圧迫する場合、補助headの4バッファを解放し、以後は既存経路へ戻る。
- Core APIの `InferenceSplitOutputHead` は既定false。GUIの `InferenceSession` は2 GPU以上で既定true。呼び出し側は `inferenceSplitOutputHead: false` で無効にできる。

## 検証

- **37テスト合格、失敗0、スキップ0**。公開対象だけをexportしたソースから実機GPUで実行。
- 奇数語彙・不均等分割・K=512/5120・bias/offset・範囲外書き込み・GPU逆順・1 GPU・別量子化形式・tied output・LoRA・prefix復元・外部予約・解放を検証。
- 旧/新で、各生成設定の**全4 run×128個の生成token IDが完全一致**。
- 各設定の16地点で**248,320語彙すべてのlogitsが全バイト一致**。SHA-256・有限値・continuation・argmaxも検査した。128地点すべてのlogitsを保存したわけではない。
- GUIとBenchmarksのReleaseビルド成功（警告0・エラー0）。
- 初回のgreedy比較でも17.404→17.813 tok/s。最終版では新→旧の順へ逆転して17.393→17.811 tok/sを確認した。表は外部予約解放を含む最終版の値。

## 証拠

- [Greedyの統計・一致検証](final-greedy-report/report.decode.ja.md) / [再計算JSON](final-greedy-report/decode-summary.json)
- [Samplingの統計・一致検証](final-sampling-report/report.decode.ja.md) / [再計算JSON](final-sampling-report/decode-summary.json)
- [旧Greedy](final-greedy-original.json) / [新Greedy](final-greedy-split.json)
- [旧Sampling](final-sampling-original.json) / [新Sampling](final-sampling-split.json)
- [GPUテスト結果](tests/split-output-head-final.trx)
- [旧設定](original-options.json) / [新設定](split-options.json)
- [ソース・バイナリ・証拠ファイルのハッシュ](source-provenance.json)

GGUFモデルと各15.9 MBの `.logits.f32` はローカルに保持し、Gitへは追加していない。
logitsのSHA-256、サイズ、比較結果は上記JSONに保存した。

## 再現

リポジトリ直下でReleaseビルド後、例として新sampling経路を実行する。
出力先は既存ファイルを上書きしない新しいパスを指定する。

```powershell
dotnet build NNtrain.Benchmarks/NNtrain.Benchmarks.csproj -c Release
dotnet NNtrain.Benchmarks/bin/Release/net10.0/NNtrain.Benchmarks.dll --qwen35-generation-probe --model models/Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf --output benchmark-results/repro-sampling-split.json --devices 0,1 --tokens 128 --runs 4 --logits 16 --options benchmark-results/qwen-generation-split-head-20261008/split-options.json --profile off --temperature 0.6 --top-p 0.95 --top-k 20 --seed 5429
```

旧経路は `original-options.json`、greedyはsamplingの4引数を省略する。
両方の結果と隣接するlogitsファイルを揃え、以下で統計・全バイト一致を再検査する。

```powershell
python tools/iq2s-production-bench/summarize_decode.py benchmark-results/repro-sampling-original.json --candidate split=benchmark-results/repro-sampling-split.json --allow-option-change InferenceSplitOutputHead --require-logits --output benchmark-results/repro-sampling-report
dotnet test NNtrain.Core.Tests/NNtrain.Core.Tests.csproj -c Release --filter FullyQualifiedName~Qwen35SplitOutputHeadTests
```
