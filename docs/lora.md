# DRN LoRA

`loss-traning-dpo.json` はオンライン生成付き DPO の設定です。起動・データ形式・制限は [DPO の説明](dpo.md) を参照してください。

現在は ForgetMemoryDRNGpt の **SFT（回答部分だけの教師あり学習）**、オンライン生成付き DPO と adapter 付き生成に対応します。他のモデル、multi-GPU、adapter の重みへの merge は未対応です。以下は SFT の説明です。

## 設定と起動

`lora.json` は rank、alpha、対象の Linear、精度を指定します。対応する精度は `float32`、`mix16_32`、`mix8_32`。学習 JSON の `precisionMode` / `bfp8BlockSize` で上書きできます。mix8_32 は block BFP8 保持、FP32 master/gradient/AdamW 統計です。元 checkpoint から指定精度へ読み込んで学習し、元 checkpoint は書き換えません。DPO の明示的な精度変更再開は [DPO の説明](dpo.md) を参照してください。SFT の adapter 再開は同一精度が必要です。

`traning-lora.json`（この綴りが正式なファイル名）は tokenizer、データ、保存先、学習条件を指定します。現在は `deviceIndex` で指定した **1 GPU** を使います。初期設定は microbatch 2 × 勾配累積 8、context 512、AdamW LR 0.00005、最大 100 optimizer steps です。LoRA の倍率は `alpha / rank`、B はゼロ初期化で、開始時の出力を維持します。

学習データ `data/lora/train.jsonl` を用意してください。1 行 1 サンプルです（実データは同梱していません）。

```json
{"prompt":"日本の首都は？","response":"日本の首都は東京です。"}
```

```powershell
dotnet run --configuration Release --project .\NNtrain.Cli -- lora --model .\checkpoints\training.forgetmemorydrn-wiki-jp.model.json --config .\traning-lora.json
```

`--model` は既存のモデル metadata JSON を指定します。safetensors 単体ではありません。best ではなく **current weights** を使い、元 optimizer は読み込みません。元モデルに使った tokenizer を必ず指定してください。語彙数だけでは tokenizer の同一性は保証できません。tokenizer の再学習は行いません。

設定内のパスは `traning-lora.json` の場所を基準に解決します。`--model` と `--config` はカレントディレクトリ基準です。

## 保存・再開・生成

SFT でも `autoResume: true` を設定すると、`adapterPath` が存在すれば自動再開、なければ新規学習になります。既定は false（DPO の同梱設定のみ true）です。壊れたファイルや設定不一致は明示エラーにし、上書きしません。`resume: true` の明示指定は checkpoint 必須です。

adapter A/B と adapter 用 AdamW state だけを別ファイルに保存します。再開は `resume: true` に変更し、同じコマンドで実行します。最大 steps は累積 optimizer steps なので、既に到達していれば `maxSteps` を増やしてください。base checkpoint、tokenizer、データ、主要な学習条件が変わった再開は拒否します。既存 adapter がある状態で `resume: false` にすると上書きを拒否します。

生成は保存済み adapter を使用します（学習データは不要）。

```powershell
dotnet run --configuration Release --project .\NNtrain.Cli -- lora --model .\checkpoints\training.forgetmemorydrn-wiki-jp.model.json --config .\traning-lora.json --generate "日本の首都は？"
```

prompt 部分と padding は loss 対象外です。長すぎる prompt はエラー、回答は context の残りで切り詰めます。勾配累積は有効な回答 token 数で重み付けします。dropout は無効、CUDA Graph は無効です。

**現段階の制限:** base weights は optimizer から除外して更新しませんが、base の勾配計算自体は残っています。そのため一般的な完全凍結 LoRA と同等の VRAM 削減・速度はまだ保証しません。

## 検証

`LoraDrnTests` はゼロ初期化時の出力一致、adapter の更新、base weights 不変、未対応モデル・対象の拒否を検証します。`LoraCommandTests` は CPU/float32 と CUDA/mix16_32 で小型 DRN の学習→adapter 保存→再開→生成を検証します。本番モデルの収束・性能ベンチではありません。
