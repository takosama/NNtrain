# DRN LoRA: 完成順キューによるオンライン DPO

```powershell
dotnet run -c Release --project .\NNtrain.Cli -- lora --model .\checkpoints\training.forgetmemorydrn-wiki-jp.model.json --config .\loss-traning-dpo.json
```

元モデルと同じ tokenizer を `tokenizerPath` に指定してください。同梱設定は事前学習と同じ **`dataset: fineweb`、`dataPath: data/fineweb`、`textColumn: text`** です。既存の Parquet shard を直接読み、JSONL への変換やダウンロードは行いません。

別途 JSONL を使う場合だけ `dataset: jsonl` とファイルの `dataPath` を指定します。形式は 1 行 1 件の text です。

```json
{"text":"ここに学習対象の文書全体を書きます。前半をprefix、後半をchosenとして使います。"}
```

SFT の prompt/response 形式や、既に chosen/rejected を持つ preference データ形式とは異なります。FineWeb reader は事前学習と共通で、shard 名順に text 列をストリーミングします。全 corpus の token 化や全ファイル同時 open はしません。再開時は読込済み文書数まで読み飛ばします。巨大 corpus 全体を起動時に hash しないため、FineWeb の同一性確認には shard 名・サイズ・更新時刻のハッシュを使用します（内容 hash ではありません）。

## 実行方式

- 文書を BOS/EOS 付きで tokenize し、context + 1 tokens までに制限。切り詰めた末尾に偽の EOS は足しません。
- `cutMinimum`～`cutMaximum` の割合から token 境界を選び、prefix と chosen に分割します。
- `generationSlots` 個の**専用スレッド**が、それぞれ独立した execution session・CUDA stream・モデル複製・DRN recurrent state・RNG を持ちます。`generationDeviceIndices` に指定した GPU へ順番に振り分けます。既定の 4 worker / `[0, 1]` は配置 `[0, 1, 0, 1]`（各 GPU 2 worker）です。prefill 最大 128 tokens、以後 1 token ずつ各 worker が独立して進みます。
- EOS または chosen と同じ token 数で rejected 生成を終了し、完成順キューへ token 配列だけを渡します。logits、計算グラフは保存しません。短い Task の終了時に長い Task を待つ batch barrier はありません。
- `batchSize` pairs が揃うたびに学習します。累積分の生成完了を一括で待たず、その間に worker が次のペアを生成します。chosen/rejected の hidden は 2B 行で計算しますが、語彙 projection・CE は各行の completion 区間だけで実行し、prompt/padding の巨大 logits を作りません。`gradientAccumulationSteps` 回ごとに optimizer を更新し、端数は実際の pair 数による平均に補正してから clip します。
- `completedQueueCapacity` は完成分 + 生成中の予約枠の合計上限です。満杯なら worker を待機させ、trainer が取り出せば再開します。trainer の forward/backward 中も生成 worker は専用 stream で動けます。**交互実行ではなく複数 stream への並列投入**です。実際の kernel overlap や速度向上は GPU の空き SM、帯域、VRAM、kernel の占有率に依存します。

## 重みと loss

### 精度設定と旧 checkpoint の再開

`loss-traning-dpo.json` の `precisionMode` / `bfp8BlockSize` が `lora.json` より優先されます。省略すると `lora.json` を継承します。対応値は `float32`、`mix16_32`、`mix8_32` です。同梱 DPO 設定は次の値です。

```json
"precisionMode": "mix8_32",
"bfp8BlockSize": 32,
"allowPrecisionConversionOnResume": true
```

`mix8_32` は base と adapter の演算用重み・activation を block BFP8 で保持し、master weight・gradient・AdamW 統計は FP32 を使います。LoRA A/B は FP32 で初期化してから量子化するため、小さい optimizer 更新を master に残せます。GPU の GEMM は BF16 operand / FP32 蓄積です。FP8 Tensor Core 命令がない RTX 3070 Ti でも動作します。FP32 の状態・workspace も残るため、総 VRAM が BF16 の半分になるという意味ではありません。

旧 checkpoint に `bfp8BlockSize` がなくても、非 BFP8 精度では比較に影響しません。精度または BFP8 block size を変えて再開する場合だけ `allowPrecisionConversionOnResume: true` が必要です（既定 false、DPO 用）。adapter、optimizer、step、data cursor と完成済みペアを復元して変換します。base/reference の量子化が変わるため、切替直後の loss や生成の完全一致は保証しません。base/tokenizer/data、rank、beta、LR、有効 batch 等の不一致は引き続き拒否し、元 checkpoint を書き換えません。

`deviceIndex` は **学習・optimizer・参照評価・`--generate`** の GPU です。`generationDeviceIndices` は rejected 生成専用で、2 GPU の勾配 all-reduce ではありません。既定は GPU 0 で学習し、GPU 0/1 で生成します。`generationDeviceIndices: [1]` にすると GPU 0 を学習、GPU 1 を生成専用にできます。省略/null は従来どおり全生成を `deviceIndex` に配置します。`generationSlots` は GPU ごとでなく全体の worker 数です。指定 device が利用不可なら開始前にエラーにし、片 GPU や CPU へ自動 fallback しません。

生成用モデルは各 worker 専用の policy スナップショットです。trainer は commit 後に adapter の CPU スナップショットだけを公開し、worker は次の Task を開始する直前に自身の adapter を更新します。生成途中の重みを変更せず、他の worker を待つ barrier もありません。生成済みペアは学習時の最新 policy で再評価します。

参照 policy は **開始時の固定 base checkpoint** です。base は optimizer から除外されるため、学習モデルの adapter を一時的に無効化した no-grad forward で参照値を計算します。生成 worker の別モデルには影響しません。別の reference モデルを追加で常駐させません。参照を移動平均や毎 step の policy に更新することはありません。

loss は `softplus(-beta * ((logP_chosen - logP_rejected) - (logRef_chosen - logRef_rejected)))` の pair 平均です。log probability は回答 token の**合計**で、prompt/padding は対象外。式は [DPO 原論文](https://arxiv.org/abs/2305.18290) に基づきます。`dpoBeta` は optimizer の momentum とは別です。

現在の scalar DPO 結合処理では、各系列の CE scalar を CPU に読み、解析勾配を scalar として戻します。logits・activation・大きな gradient を CPU へ移す実装ではありませんが、転送回数は batch size に比例します。CUDA 上での loss 一括融合は今後の最適化項目です。

## 保存・再開

### 定期サンプル

`sampleEverySteps: 100` で 100 global optimizer steps ごとに、直前の学習ペアの prefix・chosen・**更新後モデルで新しく生成した続き**をコンソールに表示します（0 で無効）。生成はストリーミング表示し、EOS または `min(maxNewTokens, chosen の token 数)` で終了します。学習用 rejected の再表示ではありません。間隔の変更は既存 checkpoint の再開に影響しません。

生成 worker の進行中 Task が終わるのを待ってから、学習モデルを no-grad/eval で一時使用します。追加のモデル複製は作らず、独立 RNG・recurrent state を使い、終了後に学習モードへ戻します。完成キューは保持します。保存と同じ step では先に checkpoint を保存します。サンプル生成時間は学習 step の計測とは別です。

### Loss グラフ

`lossGraphPath: "loss-traning-dpo.loss.html"` へ DPO loss の HTML を出力します。横軸は global step。`lossGraphEverySteps: 10` で 10 optimizer steps ごとに更新し、最初の step・checkpoint 保存・正常終了時にも更新します。ブラウザで開くと 1 秒ごとに再読み込みします。設定を省略した場合は adapterPath と同じ場所の `<adapter名>.loss.html` です。

全 step の loss は隣接する `.metrics.jsonl` に追記します。HTML の描画点は最大 2000 点に間引きますが、JSONL には全点が残ります。自動再開では checkpoint までの履歴を残し、checkpoint より後の再実行対象の点だけを除去します。古い checkpoint に metrics がない場合、過去の loss は復元できないため再開後から記録します。グラフパス・更新間隔の変更は checkpoint の互換性を壊しません。新規 run で同じパスを使う場合は、旧 HTML と metrics を別名で退避してから新しい履歴を作ります。

### モデル

保存境界では新規生成を止め、進行中 Task を完成させてから adapter、optimizer、完成キュー、読込済み cursor を一緒に原子的に保存します。保存と定期サンプル時には drain の待機があります。古い JSON テンプレートとは異なり、ここで DPO 専用の checkpoint を使います。SFT adapter の直接 resume は未対応です。

同梱設定では **`autoResume: true`** です。同じコマンドを実行すると `adapterPath` の checkpoint があれば自動再開し、なければ新規 adapter で開始します。中断マーカーは不要です。adapter 重み・optimizer・global step・読込 cursor・完成キューを復元し、開始ログに再開位置を表示します。壊れた checkpoint や base/tokenizer/data/主要設定の不一致はエラーにし、新規学習へ fallback しません。保存途中の `.tmp` は採用しません。

`resume: true` を明示した場合は checkpoint 必須です（なければエラー）。新規でやり直す場合は別の `adapterPath` を指定してください。`autoResume: false` / `resume: false` のまま既存 adapter を上書きすることも拒否します。`maxSteps` は累積 optimizer steps で、到達済みの場合はその旨を表示して学習を開始しません。継続するときは値を増やしてください。Ctrl+C は処理境界で停止し、最後に完了した checkpoint 以降は再実行します。中途半端な optimizer 更新を checkpoint として公開しません。

adapter がない新規 run で同じ `lossGraphPath` に以前の HTML/metrics がある場合は、両方を `*.previous-<UTC日時>-<ID>.html` / `.metrics.jsonl` にコピーして保存し、元のパスには新規 run の履歴を作ります。退避先を起動ログに表示します。コピーが失敗した場合は元の履歴を置き換えず停止します。旧グラフの存在を理由に新規学習を拒否せず、旧 loss を新しい重みに結び付けて再開したことにもしません。adapter が失われた場合、HTML/metrics だけから重みや optimizer state を復元することはできません。

有効 batch は `batchSize * gradientAccumulationSteps` です。例えば保存時が `2 * 8 = 16` なら、再開時の batch を 4 にする場合は累積を 4 にして `4 * 4 = 16` を維持します。`4 * 8 = 32` は更新の意味が変わるため拒否します。`allowPrecisionConversionOnResume` は精度変更だけの許可で、有効 batch・DPO beta 等の不一致を無視する設定ではありません。

完成順と生成時の snapshot の古さは実際のスレッド scheduling に依存します。保存済み queue/cursor の整合性は維持しますが、別々に起動した連続実行と再開実行の bitwise 一致は保証しません。旧交互実行版からは実行方式の contract が変わるため、旧 DPO checkpoint の直接再開は拒否します。元の事前学習 checkpoint は変更しません。

並列 stream 版 checkpoint は GPU 配置を変更しても再開できます。`deviceIndex` / `generationDeviceIndices` は数式や adapter 形状を変えないため checkpoint contract に含めません。CPU の adapter スナップショットを各生成 worker の GPU に配布します（重み全体の毎 step 転送や P2P 必須化はしません）。

生成確認は同じコマンドに `--generate "文書の書き出し"` を追加します。DPO は生文書の続きの学習なので promptPrefix/responsePrefix は空文字にします。

## 制限と検証

### 2026-09-10 の最適化

DPO の base linear（memory/FFN/語彙 head）は dX だけを計算し、更新しない dW/db の GEMM・reduction・gradient buffer を省略します。LoRA A/B の勾配、参照 policy、loss 式は維持します。embedding と LayerNorm の base 勾配はまだ計算します。

mix8_32 では、LoRA 加算後の private base 出力・倍率適用済み adapter 出力、残差加算後の private Add 出力を早期解放します。これらの backward は出力値を読まず勾配だけを使うため、演算内容を変えずに layer activation を削減できます。ReLU に必要な値と A/B 学習の入力は保持し、公開 Tensor 演算の値再利用には影響しません。[条件付き実測と不採用実験](benchmarks/dpo-mix8-2026-09-10.md)を記録しています。

`generationSlots`・`completedQueueCapacity` の変更は既存 checkpoint から再開できます。`batchSize * gradientAccumulationSteps` が同じなら microbatch の分割変更も許可します。ただし完了順・snapshot の古さが変わるので、変更前後の生成や更新の bitwise 一致は保証しません。新しい容量より保存済みキューが大きい場合は、既存ペアを捨てずに消費し、容量以内になるまで新規生成を待機します。beta・LR 等の意味論が変わる設定の検査は維持します。

同じ step の終了時 checkpoint 二重保存を省略します。学習モデルは 1 個、生成モデルは slot ごとに 1 個のままです。定期サンプルでも進行中生成の drain 待機があります。

DRN、単 GPU / 複数 GPU 分散生成または CPU、float32/mix16_32/mix8_32 に対応。学習・optimizer 自体は単 device です。**学習モデル 1 個 + generationSlots 個の生成モデル**を持つため、生成 worker 数に応じて VRAM が増えます。OOM の場合は `generationSlots` を減らしてください（指定 GPU 数以上が必要）。base の embedding/LayerNorm 勾配は残ります。今回の設定は batch 2 / accumulation 8 / slots 4 / queue 8。小さなキューの待ち時間を減らしますが、全データでの速度や収束の最適値を保証しません。データの続きが生成文より必ず良いわけでもありません。

数値安定性、有限差分勾配、実 worker の重複実行、長い Task を待たない完了順、容量制限・異常終了・端数処理、FineWeb 形式 Parquet を使った CPU/1GPU/2GPU の学習→保存→再開→生成をテストします。CUDA の結合テストでは 4 本の異なる stream と各 worker の device 配置を確認し、保存後の生成 GPU 配置変更も検証します。起動ログに worker/thread/device/stream、保存ログに同時生成 worker 数の peak を出します。これらは GPU kernel overlap の実測値そのものではありません。ログは生成待ち・学習・step 合計の実測 ms を分離します。本番サイズでの収束評価・長時間 soak は未実施です。
