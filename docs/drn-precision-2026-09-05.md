# DRN mix8_32 / mix16_32 精度・batch 切り分け（2026-09-05）

## 結論

6条件、計1,600 optimizer updatesを完走。今回の縮小モデルでは、mix8だけが
収束しない現象は再現しなかった。**本番のloss 4.4停滞の原因は未確定**。
学習済みFP32 masterを固定した比較では、重みだけのBFP8丸めより、活性も
BFP8にする経路の勾配誤差が大きかった。block128→32で全6条件の勾配誤差が
減ったが、これはblock32の長期収束改善や本番での解決を証明しない。

本番JSON・checkpoint・tokenizer・metricsは変更していない。本体プロセスも
停止していない。診断コマンドとレポートを追加し、以前のdepth-scaling報告の
「原因確定」と読める表現を訂正した。

## 凍結条件

- RTX 3070 Ti GPU0のみ、既存学習とは別プロセス。CPU/2GPU速度比較ではない。
- 実FineWeb、既存fineweb-bpe.jsonを読み取り専用で使用。vocabulary 11,500。
- 新規DRN、width64 / hidden192 / 32layers / K16 / V16、depth-scaled初期化。
  本番width512ではなく、停滞時checkpointも残っていなかったため使用していない。
- seed1234、dropout0.1、Muon(momentum .95、Nesterov、NS5毎更新) LR .05、
  補助AdamW(beta1 .9 / beta2 .95) LR .015、weight decay .01、clip1。
- 定数LR、accumulationなし、学習mix8はblock128。batch5:12でユーザー申告の
  batch50:120の**比率だけ**を再現。絶対batch・GPU分割・幅の効果は再現していない。
- 系列長ごとに同一初期重み・同一順序の学習トークンを使用。batchを変えると
  optimizer更新回数とdropout乱数の割当ては変わる。1seed/各1回で統計的検定なし。
- 評価は学習に使わない別文書8 batches × 4 sequences。文書ハッシュで重複排除。
  文書上限512 tokens、packing/target shiftは本番関数を使用。
- evalでdropoutを停止するがautogradは有効にして、学習と同じloss経路を測る。
  生成/no-gradモード切替問題を検証する実験ではない。

## 学習結果

評価lossは各モデルが学習した精度で算出。train欄は最後20更新の平均なので、
異なるbatch同士では対象トークン範囲が異なる。batch間は同じheld-out評価で見る。

| Context | Batch | 精度 | 学習tokens | 更新数 | 最終train loss | 評価loss | 学習時間(s) | allocator最大(MiB) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
|32|5|mix8_32|76,800|480|8.37179|8.55411|27.12|84.48|
|32|5|mix16_32|76,800|480|8.37002|8.58131|19.14|101.90|
|32|12|mix8_32|76,800|200|8.38257|8.46228|12.97|95.19|
|32|12|mix16_32|76,800|200|8.38809|8.47459|21.62|138.51|
|512|5|mix8_32|307,200|120|8.08657|8.01852|16.75|199.10|
|512|5|mix16_32|307,200|120|8.08844|8.02185|28.07|494.03|

時刻によって本体のGPU負荷が変わったため、上記時間から速度優劣は判断しない。
warmup除外もしていない。メモリ値は学習session allocatorの所有量で、CUDA
context/native libraryを含めたプロセスVRAM総量ではない。最初の長系列mix16試行は
診断側の350 MiB安全上限で停止（CUDA OOMではない）。上限512 MiBへ変更後に完走。
別途12更新のsmokeも完走。本番の2GPUや長時間resume/soakは実施していない。

## 同じ学習済みmasterでの比較

各学習後のFP32 masterを取り出し、以下の別モデルを作る。

1. BF16重み・BF16活性のmix16（比較基準）。
2. masterをCPU codecでBFP8 block128に一度丸めてからBF16モデルへ設定。
3. 同じmasterからmix8 block128。
4. 同じmasterからmix8 block32。

勾配誤差は最初のheld-out batchの全parameterを連結した
`||g_variant - g_reference||₂ / ||g_reference||₂`。層別の最大誤差ではなく、
大きい勾配のparameterに支配され得る。勾配cosine・8 batches平均loss差も生JSONに保存。

| 学習元(context/batch/precision) | 重みだけ丸め 誤差 | mix8 block128 誤差 | mix8 block32 誤差 |
|---|---:|---:|---:|
|32 / 5 / mix8|4.96%|34.16%|20.53%|
|32 / 5 / mix16|2.90%|4.97%|2.09%|
|32 / 12 / mix8|4.55%|9.49%|6.96%|
|32 / 12 / mix16|9.22%|27.93%|11.74%|
|512 / 5 / mix8|4.76%|14.47%|11.91%|
|512 / 5 / mix16|3.89%|12.79%|9.63%|

これらの比較で同じmasterのmix8 block128対mix16評価loss差の絶対値は最大0.00290。
勾配差は存在しても、今回の短い訓練ではmix8固有のloss悪化に結び付かなかった。
block32は評価時だけの比較であり、block32で再学習した結果ではない。
重みだけの比較と完全mix8経路の差には活性丸めに加えてkernel演算経路差も含まれる。
residualのみをBF16にした厳密なablationではない。

## 次に必要な切り分け

本番の停滞時checkpointで、同じ入力・同じbatch・dropout停止の精度切替比較を行う。
今回の縮小・短期・新規初期化実験だけを根拠に本番LRや初期化を再変更しない。
既存mix8経路はFP32 master/gradient/optimizer stateを保持する契約なので、
「すべての微小更新が毎step丸め捨てられる」とも断定しない。
優先調査候補は活性/残差の量子化と層別勾配だが、原因確定には本番重みが必要。

## 再現・成果物

Release build: warning/error 0。全solutionの回帰テストや210-step性能試験は未実施。
隔離出力へbuildし、本番CLI出力先のDLLは更新していない。

```powershell
dotnet build .\NNtrain.Benchmarks -c Release --no-restore `
  -o .\benchmark-results\drn-precision-20260905\bin-long
$env:CUDA_MODULE_LOADING = 'LAZY'
dotnet .\benchmark-results\drn-precision-20260905\bin-long\NNtrain.Benchmarks.dll `
  --probe-drn-precision `
  .\benchmark-results\drn-precision-20260905\probe.config.json `
  .\benchmark-results\drn-precision-20260905\new-result.json `
  mix8_32 5 600 512
```

出力ファイルは新規名必須。引数はprecision、batch(5/12)、sequence例数(60の倍数、
最大6000)、任意context(32/512)。小規模診断用の固定幅で、本番学習の代替ではない。

生データ: `benchmark-results/drn-precision-20260905/mix{8,16}_32-b{5,12}.json`、
`mix{8,16}_32-b5-s512.json`。固定入力設定は同フォルダの`probe.config.json`。

入力+target SHA256（同contextの全条件で一致）:

- context32: `34786536DC0E80E13F4FEF20D95C783CBC8A34789447F5D84AB5DBE3DA0CA1F0`
- context512: `BB5A416A4F38C3083DCBFF49D8068C90E3C00373B23392F3F5B1F32CCDBEC540`
