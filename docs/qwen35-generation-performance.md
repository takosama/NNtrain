# Qwen3.5 27B 生成高速化の測定結果

作成日時（UTC）: 2026-09-27T08:34:37.291579+00:00

## 測定結果

モデル: `Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf`。GPU: Intel Arc B580 × 2（デバイス 0,1）。
LoRA はつくよみちゃん学習済みチェックポイント（1536 更新、rank 8、alpha 16、496 対象行列）を使用しました。

| 条件 | 変更前 tok/s | 変更後 tok/s | 速度比 | 最初のトークン・変更前 ms | 変更後 ms | 最終ピーク GiB/GPU |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LoRA なし | 14.725 | 17.123 | 1.163× | 786.3 | 514.7 | 4.834 / 4.910 |
| LoRA あり | 3.652 | 16.159 | 4.425× | 3789.8 | 569.6 | 4.943 / 5.019 |

各条件はモデルを読み込んだ後に 3 回測定し、初回を除いた 2 回の中央値を表にしています。入力は同じ 16 トークン、出力は毎回 256 トークンです。

- ベンチマークは EOS 後も含めて 256 トークンを強制生成します。通常の会話終了までの速度や回答品質の評価とは条件が異なります。
- 生成速度は最初から最後の出力コールバックまでの 255 トークン分で計算します。モデル・LoRA の読み込み時間は含みません。
- 最初のトークンの時間には状態リセット、入力の処理、最初の出力層計算・選択を含みます。
- OS・GPU ドライバーのキャッシュは消去していません。ピーク値はアプリ側の GPU メモリ割り当てカウンターで、ドライバーの実測 VRAM 値ではありません。

### 各回の速度

| 条件 | 変更前の 3 回 tok/s | 変更後の 3 回 tok/s |
| --- | --- | --- |
| LoRA なし | 14.697, 14.729, 14.721 | 17.096, 17.126, 17.119 |
| LoRA あり | 3.649, 3.651, 3.653 | 16.117, 16.161, 16.157 |

## 採用した変更

- LoRA の A 側積和を 1024 スレッドの協調計算に変更し、B 側の加算を量子化射影カーネルに統合しました。
- GPU 上の最大値選択と DeltaNet の Q/K 正規化を並列化しました。
- GGUF の半精度スケールを GPU の半精度読み込み命令で FP32 に変換します。重みの再量子化や FP16 へのモデル展開は行いません。
- 通常の生成ではカーネル単位の計時イベントを省き、キューの上限を 4096 にしました。完了確認とメモリ寿命の同期は維持しています。
- KV キャッシュと DeltaNet の継続状態は GPU 上で再利用します。演算の蓄積と LoRA パラメーターは FP32、ベース重みは元の量子化形式のままです。
- 訓練用の設定とカーネルを分け、訓練の既定キュー上限 512 を維持しました。

Q4 ループの明示的展開、射影ワークグループ 64/128、LoRA の SG16 縮約は既定に採用していません。1 GPU でも動作・比較しましたが、同等範囲の速度であり、既定の案内は 2 GPU としています。

## 探索中の比較

以下は主に 64 トークンの探索結果です。組み合わせた設定も含むため、各行の差を単独の変更効果として足し合わせることはできません。最終判断には上の同条件 256 トークン比較を使用します。

| 試行 | 記録名 | 出力数 | 初回を除く測定数 | 中央値 tok/s |
| --- | --- | ---: | ---: | ---: |
| LoRA 協調計算 | `coop-lora` | 64 | 2 | 12.768 |
| 射影 WG64 | `group64-base` | 64 | 1 | 14.787 |
| 射影 WG128 | `group128-base` | 64 | 1 | 14.761 |
| 並列 argmax・Q/K 正規化 | `reduce-base` | 64 | 1 | 15.380 |
| LoRA B 側統合 | `fused-lora` | 64 | 1 | 12.898 |
| LoRA SG16 | `lora16` | 64 | 1 | 11.947 |
| LoRA 256 | `lora256` | 64 | 1 | 14.154 |
| LoRA 512 | `lora512` | 64 | 1 | 14.586 |
| LoRA 1024 | `lora1024` | 64 | 2 | 16.275 |
| 1 GPU・LoRA なし | `reduce-single-base` | 64 | 1 | 15.384 |
| 1 GPU・LoRA あり | `reduce-fused-single-lora` | 64 | 1 | 13.208 |
| 計時イベント省略・LoRA なし | `noevents-base` | 64 | 1 | 15.778 |
| 計時イベント省略・LoRA あり | `noevents-lora512` | 64 | 2 | 14.989 |
| キュー 4096・LoRA なし | `queue4096-base` | 64 | 1 | 17.229 |
| キュー 4096・LoRA あり | `queue4096-lora` | 64 | 1 | 16.016 |
| 半精度スケール読み込み | `nativehalf-base` | 64 | 1 | 16.784 |
| Q4 明示的ループ展開 | `unroll-base` | 64 | 1 | 15.397 |

## 確認結果

| 条件 | 変更前 3 回の 256 トークン | 変更後 3 回の 256 トークン |
| --- | --- | --- |
| LoRA なし | 全回一致 | 全回一致 |
| LoRA あり | 全回一致 | 全回一致 |

比較対象は各モードの変更前 1 回目の全文トークン ID です。FP32 の縮約順序は一部変わるため、この入力以外での同一出力やビット単位の logits 一致を保証するものではありません。

同一の固定継続入力で 4 ステップ × 248,320 語彙の logits を別途比較しました（基準64トークン測定の保存値との比較）。非有限値は 0、Top-1 は両モードとも 4/4 一致です。LoRA なしの最大絶対差は 3.528595e-5・相対 L2 は 1.761212e-6、LoRA ありは 1.716614e-5・1.051756e-6 でした。

量子化重みの GPU 常駐サイズは各測定時点で維持されました: GPU 0 = 5,035,468,800 bytes、GPU 1 = 5,116,405,760 bytes。ベース GGUF と LoRA の SHA256 は各モードで変更前後一致しています。

### 通常のストリーム生成

通常 CLI のログでエラーがなく、生成と時間表示が完了したことを確認しています。この短い試行の速度は上の強制 256 トークン結果と直接比較しません。

| 条件 | 生成数 | decode tok/s | 最初のトークン ms |
| --- | ---: | ---: | ---: |
| LoRA なし | 64 | 17.24 | 566.0 |
| LoRA あり | 22 | 16.30 | 622.0 |

### テスト

下表は最終 TRX の各テスト結果から取得しています。

| 対象 | 総数 | 成功 | 失敗 | スキップ |
| --- | ---: | ---: | ---: | ---: |
| Core / Qwen3.5 | 158 | 158 | 0 | 0 |
| Integration | 51 | 45 | 0 | 6 |

Integration の 6 件は CUDA が利用できないためスキップされました。Arc の追加テストはすべて実行されています。Release ビルドは警告 0・エラー 0 で成功しました。


## 未確認の範囲

異なる入力、さらに長いコンテキスト、他のモデル・GPU、回答品質の変化はこの測定だけでは評価していません。LoRA の再学習は行っていません。

## 生成コマンド

NNtrain のリポジトリ直下で PowerShell に貼り付けます。文字列は順次表示されます。

```powershell
$prompt = "<|im_start|>user`nこんにちは。自己紹介をしてください。<|im_end|>`n<|im_start|>assistant`n"

# LoRA なし
dotnet .\NNtrain.Cli\bin\Release\net10.0\NNtrain.Cli.dll qwen-gguf --model ".\models\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf" --devices 0,1 --prompt $prompt --max-new-tokens 256

# 学習済み LoRA あり
dotnet .\NNtrain.Cli\bin\Release\net10.0\NNtrain.Cli.dll qwen-gguf --model ".\models\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf" --adapter ".\checkpoints\tuku-qwen35\adapter.bin" --devices 0,1 --prompt $prompt --max-new-tokens 256
```

NNtrain の `--adapter` には学習済み `adapter.bin` を指定します。GGUF 形式の LoRA アダプターはこのコマンドでは使用しません。

## ベンチマークの再実行

出力先にはまだ存在しないファイル名を指定してください。最適化前の数値は保存済み旧バイナリによる測定です。以下は現在の実装を測定します。

```powershell
$benchPrompt = "<|im_start|>user`n国会議事堂への行き方を教えて<|im_end|>`n<|im_start|>assistant`n"

dotnet run -c Release --no-build --project NNtrain.Benchmarks -- --qwen35-generation-probe --model ".\models\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf" --output ".\benchmark-results\generation-base-256-new.json" --devices 0,1 --prompt $benchPrompt --tokens 256 --runs 3

dotnet run -c Release --no-build --project NNtrain.Benchmarks -- --qwen35-generation-probe --model ".\models\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf" --adapter ".\checkpoints\tuku-qwen35\adapter.bin" --output ".\benchmark-results\generation-lora-256-new.json" --devices 0,1 --prompt $benchPrompt --tokens 256 --runs 3
```

## 再現情報

モデルと実行バイナリの SHA256（実測 JSON より）:

```json
{
  "baseline256-base": {
    "ModelSha256": "676db8ccd38035117e705858fd591964005cb0a921b39255b0691d8339a8dfe2",
    "AdapterSha256": null,
    "BinarySha256": {
      "NNtrain.Core": "b29c91a407ed4d215c1c466d27d5e13bc7ecb6df0a2751a04090914bcaa9bb1a",
      "NNtrain.Arc": "3576e2603f5d09aa326c45be6e936e1d2a0842ddc14d65467d1b72d366912880",
      "InferenceProbe": "4358d6788d4d5d10dfa595f30abb123273980e13a9cbecec6df7647c9466a90a"
    },
    "Options": {
      "QuantizedKernel": "Auto",
      "LoraTraining": false,
      "TrainingBatchGradientNorm": true,
      "TrainingNormSplits": 8,
      "TrainingCooperativeLora": true,
      "TrainingCooperativeDelta": true,
      "TrainingResponseOnlyHead": true,
      "TrainingTransposeRows": 16,
      "TrainingForwardRows": 4,
      "TrainingBufferPoolMiB": 512,
      "DetailedProfiling": false,
      "FusedDelta": true,
      "QueuedKernelLimit": 512
    }
  },
  "baseline256-lora": {
    "ModelSha256": "676db8ccd38035117e705858fd591964005cb0a921b39255b0691d8339a8dfe2",
    "AdapterSha256": "723b5caf1484f6bac85c454bf33ea302e2d51607d1a47f602352f6daa738ba0a",
    "BinarySha256": {
      "NNtrain.Core": "b29c91a407ed4d215c1c466d27d5e13bc7ecb6df0a2751a04090914bcaa9bb1a",
      "NNtrain.Arc": "3576e2603f5d09aa326c45be6e936e1d2a0842ddc14d65467d1b72d366912880",
      "InferenceProbe": "4358d6788d4d5d10dfa595f30abb123273980e13a9cbecec6df7647c9466a90a"
    },
    "Options": {
      "QuantizedKernel": "Auto",
      "LoraTraining": false,
      "TrainingBatchGradientNorm": true,
      "TrainingNormSplits": 8,
      "TrainingCooperativeLora": true,
      "TrainingCooperativeDelta": true,
      "TrainingResponseOnlyHead": true,
      "TrainingTransposeRows": 16,
      "TrainingForwardRows": 4,
      "TrainingBufferPoolMiB": 512,
      "DetailedProfiling": false,
      "FusedDelta": true,
      "QueuedKernelLimit": 512
    }
  },
  "final256-base": {
    "ModelSha256": "676db8ccd38035117e705858fd591964005cb0a921b39255b0691d8339a8dfe2",
    "AdapterSha256": null,
    "BinarySha256": {
      "NNtrain.Core": "dc547a96b50468810fcb3fb11f8774c5b173dc4beda254e38b3b983d74e7dcda",
      "NNtrain.Arc": "9b08ef209c5279872816ff2fbb85677467672f4e3fadd39fe255017267ef1042",
      "NNtrain.Benchmarks": "f45e0161fd484210124bf75c20958a5078b26462ae993a0a55e7be347546bb0d"
    },
    "Options": {
      "QuantizedKernel": "Auto",
      "LoraTraining": false,
      "InferenceCooperativeLora": true,
      "ProjectionWorkgroupSize": 32,
      "ParallelArgmax": true,
      "ParallelDeltaNorm": true,
      "InferenceFusedLora": true,
      "LoraReductionSize": 1024,
      "UnrollQ4": false,
      "NativeHalfScale": true,
      "CollectKernelTimings": false,
      "TrainingBatchGradientNorm": true,
      "TrainingNormSplits": 8,
      "TrainingCooperativeLora": true,
      "TrainingCooperativeDelta": true,
      "TrainingResponseOnlyHead": true,
      "TrainingTransposeRows": 16,
      "TrainingForwardRows": 4,
      "TrainingBufferPoolMiB": 512,
      "DetailedProfiling": false,
      "FusedDelta": true,
      "QueuedKernelLimit": 4096
    }
  },
  "final256-lora": {
    "ModelSha256": "676db8ccd38035117e705858fd591964005cb0a921b39255b0691d8339a8dfe2",
    "AdapterSha256": "723b5caf1484f6bac85c454bf33ea302e2d51607d1a47f602352f6daa738ba0a",
    "BinarySha256": {
      "NNtrain.Core": "dc547a96b50468810fcb3fb11f8774c5b173dc4beda254e38b3b983d74e7dcda",
      "NNtrain.Arc": "9b08ef209c5279872816ff2fbb85677467672f4e3fadd39fe255017267ef1042",
      "NNtrain.Benchmarks": "f45e0161fd484210124bf75c20958a5078b26462ae993a0a55e7be347546bb0d"
    },
    "Options": {
      "QuantizedKernel": "Auto",
      "LoraTraining": false,
      "InferenceCooperativeLora": true,
      "ProjectionWorkgroupSize": 32,
      "ParallelArgmax": true,
      "ParallelDeltaNorm": true,
      "InferenceFusedLora": true,
      "LoraReductionSize": 1024,
      "UnrollQ4": false,
      "NativeHalfScale": true,
      "CollectKernelTimings": false,
      "TrainingBatchGradientNorm": true,
      "TrainingNormSplits": 8,
      "TrainingCooperativeLora": true,
      "TrainingCooperativeDelta": true,
      "TrainingResponseOnlyHead": true,
      "TrainingTransposeRows": 16,
      "TrainingForwardRows": 4,
      "TrainingBufferPoolMiB": 512,
      "DetailedProfiling": false,
      "FusedDelta": true,
      "QueuedKernelLimit": 4096
    }
  }
}
```


## 測定ファイル

[比較・検証結果](benchmarks/qwen35-generation-2026-09-27/validation.json)

- [baseline256-base](benchmarks/qwen35-generation-2026-09-27/baseline256-base.json)
- [baseline256-lora](benchmarks/qwen35-generation-2026-09-27/baseline256-lora.json)
- [final256-base](benchmarks/qwen35-generation-2026-09-27/final256-base.json)
- [final256-lora](benchmarks/qwen35-generation-2026-09-27/final256-lora.json)
