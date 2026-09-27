# Ternary Bonsai 27B の GPU 生成対応

2026-09-27、Intel Arc B580 × 2、Windows、.NET 10 Release で確認。

## 結果

| モデル | 生成速度 tok/s | 最初のトークン ms | 重みの GPU 常駐 GiB/台 | アプリのピーク割り当て GiB/台 |
| --- | ---: | ---: | ---: | --- |
| Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf | 7.615 | 1411.8 | 3.346 | 3.440 / 3.441 |
| Ternary-Bonsai-2-27B-PTQ1_0.gguf | 13.643 | 828.9 | 2.759 | 2.854 / 2.854 |

同じ20トークンの入力から greedy で64トークンを3回生成し、初回を除く2回の中央値を示しています。モデル読み込み時間を除外し、最初から最後の出力コールバックまでの63トークン分で decode 速度を計算します。EOS後も指定数まで生成するベンチマークであり、回答品質の比較ではありません。OS・ドライバーのキャッシュは消去していません。ピーク値はアプリの割り当てカウンターであり、ドライバー計測のVRAM値ではありません。

- PQ2_0 各回: 7.602 / 7.614 / 7.615 tok/s。
- PTQ1_0 各回: 13.637 / 13.624 / 13.663 tok/s。
- 通常CLIも両モデルで成功。PQ2_0は64トークン・7.60 tok/s、PTQ1_0はEOSまで54トークン・13.61 tok/sでした。
- 重みは元の PQ2_0 / PTQ1_0 / BF16 のままGPUへ配置します。モデル全体のFP32展開や再量子化はしません。
- KVキャッシュ、DeltaNetの継続状態、Prism変換をGPUで処理し、生成文をストリーム出力します。

## 実行コマンド

PowerShellでリポジトリのルートから実行します。プロンプト内の質問を書き換えられます。

```powershell
$prompt = "<|im_start|>user`nこんにちは。日本語で自己紹介してください。<|im_end|>`n<|im_start|>assistant`n<think>`n`n</think>`n"

dotnet .\NNtrain.Cli\bin\Release\net10.0\NNtrain.Cli.dll qwen-gguf --model ".\models\Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf" --devices 0,1 --prompt $prompt --max-new-tokens 256

dotnet .\NNtrain.Cli\bin\Release\net10.0\NNtrain.Cli.dll qwen-gguf --model ".\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf" --devices 0,1 --prompt $prompt --max-new-tokens 256
```

別環境で取得したソースを使う場合は、先に `dotnet build .\NNtrain.Cli -c Release` を実行してください。

## 実装

- GGUF type 142（PQ2_0: 128要素/34 bytes）、143（PTQ1_0: 128要素/28 bytes）、30（BF16行列）の読み込み・サイズ検証・GPU射影を追加。
- Prism metadata version 1に従う1024要素の正規化Hadamard変換、明示的符号ベクトル、埋め込みの逆変換、GDN Vの並べ替えを追加。
- 対応しない変換・不足メタデータ・不正な寸法を読み込み時に拒否。
- type 4の追加特殊トークンを分割しない処理とテストを公開ブランチへ追加。通常の作業リポジトリに既存のUnicode関連変更は維持。
- PTQ1_0はパックされたブロックの読み込みとスケール計算を再利用する専用SG16処理、PQ2_0は協調計算カーネルをAutoで選択。

### 速度探索

以下は32トークン×2回の探索で、2回目の値です。最終64トークン測定と条件が異なります。

| 方式 | PQ2_0 tok/s | PTQ1_0 tok/s | 判断 |
| --- | ---: | ---: | --- |
| 最初の正しい汎用SG16実装 | 6.382 | 3.789 | 比較基準 |
| ブロック単位の専用SG16 | 2.167 | 13.653 | PTQ1_0のみ採用 |
| PQ2_0専用SG16のループ展開抑制 | 2.167 | — | 不採用 |
| 協調計算 | 7.747 | — | PQ2_0で採用 |

## 正しさの確認

形式の根拠は[作者の llama.cpp 固定コミット](https://github.com/PrismML-Eng/llama.cpp/tree/adfffbe41b2cabcd51fff326ab045662265062bb)です。[量子化参照データの説明](../NNtrain.Core.Tests/Fixtures/PrismQuantReference/README.md)に配布バイナリー、チェックサム、再現手順があります。

- 作者の公式CPUデコーダーを実行して得た300ブロック・38,400値をGPUテストの参照に使用。
- PQ2の全コード、PTQ1の全バイト値と全格納位置、FP16スケール境界、実モデルのサンプルを確認。
- Hadamard変換は独立した行列計算との比較、逆変換、GDN並べ替えを確認。
- 作者のCPU実行と入力20トークンのIDが一致。最初の8生成トークンを復号した文章が両モデルで一致（コンソール外側の空白は除外）。作者との全文・全logitsの一致を確認したものではありません。
- 最適化前後で先頭32生成IDが一致。最終64トークンのIDは3回とも一致。
- NNtrainの正しい汎用版と最終版の4ステップ×248,320語彙を比較。PQ2_0の最大絶対差は9.059906e-6、相対L2は7.458911e-7。PTQ1_0は全値一致。両モデルで全値有限、Top-1は4/4一致。
- 同一モデルの比較前後でSHA256と量子化重みのGPU常駐量が一致。

### ビルド・テスト

| 対象 | 成功 | 失敗 | スキップ |
| --- | ---: | ---: | ---: |
| Releaseビルド | 成功（警告0・エラー0） | — | — |
| Core / Qwen35 | 208 | 0 | 0 |
| 通常リポジトリ Integration 全体 | 491 | 1 | 6 |

全体テストの失敗は `WikiDTypeCheckpointTests.V8Bfp8CheckpointUsesPrecisionArtifactsAndResumes(mode: Mix8_16, blockSize: 4)`。AdamWがBF16 masterにFP32 `DataBuffer`としてアクセスする例外です。**今回の変更前に退避した実行ファイルでも同じ失敗を再現**しました（該当Theoryは2成功・1失敗）。失敗経路の7ソースファイルは作業前のSHA256と一致し、今回のPrism対応に含めていません。6スキップはCUDA用のテストです。全体テストがすべて成功したとは扱いません。

```powershell
dotnet build .\NNtrain.Benchmarks\NNtrain.Benchmarks.csproj -c Release --no-restore
dotnet test .\NNtrain.Core.Tests\NNtrain.Core.Tests.csproj -c Release --no-restore --filter "FullyQualifiedName~Qwen35"
dotnet test .\NNtrain.IntegrationTests\NNtrain.IntegrationTests.csproj -c Release --no-restore

# 上記の $prompt を使用。PTQ1_0もモデル名と出力先を変えて同条件で測定。
dotnet .\NNtrain.Benchmarks\bin\Release\net10.0\NNtrain.Benchmarks.dll --qwen35-generation-probe --model ".\models\Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf" --devices 0,1 --prompt $prompt --tokens 64 --runs 3 --logits 4 --output PQ2.final64.json
```

数値、全生成ID、モデル・実行ファイルSHA256、メモリ量は[保存した測定記録](benchmarks/ternary-bonsai-2026-09-27/validation.json)と同ディレクトリのJSON・ログにあります。大容量のモデル、参照DLL、全logitsバイナリーはGitへ追加していません。

## 制限と未確認事項

- 今回の2モデルは生成対応です。PrismモデルへのLoRA装着・LoRA訓練は未対応で、明示的に拒否します。既存Qwenモデルの学習済みLoRAは従来のベースモデルと組み合わせてください。
- 長いコンテキスト、他のPrism変換バージョン、別GPUでの速度・数値一致は未確認です。
- 元のQwenモデルのLoRAあり／なしの高速化結果と実行コマンドは[先行レポート](qwen35-generation-performance.md)にあります。
