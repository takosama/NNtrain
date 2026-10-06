# NNtrain

C# / .NET 10 で実装した、ローカル推論・学習用のニューラルネットワーク実行環境です。Tensor、自動微分、言語モデル、最適化、チェックポイントを独自実装し、Windows GUI から Intel Arc 上の GGUF モデルへチャット、画像、音声からの入力を送れます。

モデルの重みとデータセットは付属しません。推論に外部 LLM API やクラウド音声認識は使いません。Hugging Face からのダウンロードには通信が必要です。

## できること

| 用途 | 現在の対応 |
| --- | --- |
| GUI チャット | `general.architecture=qwen35` の GGUF、ストリーミング、Thinking、LoRA、Arc 1 台 / 2 台 |
| CLI GGUF 推論 | Qwen3.5 と Qwen2 / Qwen2.5。Qwen2 系は単一 Arc |
| 画像入力 | Qwen3.5 用の対応 mmproj GGUF と PNG / JPEG の静止画 |
| 日本語音声入力 | Nemotron 3.5 ASR のキャッシュ付き認識、Parakeet 日本語モデルの CTC 認識、確定文の編集・送信 |
| 学習 | Transformer、ForgetMemory 系、ForgetScan、Hyena、CIFAR-100。CPU / CUDA / Arc の対応範囲は異なります |
| LoRA | 独自 DRN の LoRA / DPO、Arc 上の Qwen3.5 GGUF LoRA 学習・再開・推論 |
| ライブラリ | `torch`、`nn`、`optim`、`lr_scheduler`、`datasets`、`tokenizers`、`safetensors` の C# API |

モデル名だけでは互換性を判定しません。GGUF の architecture、テンソル形式、形状を検証します。GUI は Qwen2 系や Qwen3.5 MoE を汎用的に扱うものではありません。

## 必要な環境

- 開発・ビルド: [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)。GUI は Windows / WPF が必要です。
- Intel Arc: OpenCL を含む Intel グラフィックスドライバー。Arc 経路に CUDA Toolkit や oneAPI は不要です。
- CUDA: 対応する NVIDIA GPU、ドライバー、CUDA バックエンドのネイティブ依存物。詳細は [精度モード](docs/precision-modes.md) を確認してください。
- RAM・VRAM・ディスク: モデルサイズ、コンテキスト、画像、学習設定によって変わります。量子化重みのサイズだけでは必要 VRAM を見積もれません。

以下のコマンドはリポジトリのルートで実行します。実行中の GUI や学習が使用する出力を上書きしないよう、開発用ビルドは Debug を利用できます。

```powershell
dotnet restore NNtrain.slnx
dotnet build NNtrain.slnx -c Debug --no-restore
```

## GUI を使う

```powershell
dotnet run --project NNtrain.Gui -c Debug
```

1. GGUF を選び、使用する Arc GPU を指定します。必要なら LoRA と mmproj を選びます。
2. 「プリロード」で読み込みます。モデルは後続の会話でも保持されます。
3. 入力して送信します。Stream、Thinking、temperature、top-p、top-k、最大出力数を変更できます。
4. 音声はチャット内の音声設定から ASR を読み込み、「録音」または PCM16 WAV の選択で入力します。確定後に編集して、通常の送信ボタンを押します。

GUI は別プロセスのローカル API サーバーを起動します。サーバーは `127.0.0.1` に限定して待ち受け、セッションごとの Bearer 認証を使います。GUI 終了時に自身のサーバーを終了します。詳細は [GUI 操作ガイド](NNtrain.Gui/README.md) と [ローカル API](docs/gui-openai-api.md) を参照してください。

配布用の自己完結実行ファイルを作る場合:

```powershell
dotnet publish NNtrain.Gui/NNtrain.Gui.csproj -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true
```

出力は `NNtrain.Gui/bin/Release/net10.0-windows/win-x64/publish/NNtrain.Gui.exe` です。重みは別途配置します。

### Hugging Face・画像・prefill

Hugging Face タブでリポジトリを調べ、GGUF / mmproj をダウンロードできます。コミットを固定し、サイズと取得可能なハッシュを検証します。中断した同じコミットの部分ファイルは再開できます。トークンは任意で、設定ファイルに保存しません。モデルのライセンス・アクセス条件は利用者が確認してください。ASR の SafeTensors / `.nemo` はこのタブの取得対象ではありません。

画像にはベースモデルと互換性のある Qwen3.5 mmproj が必要です。画像準備と既知の会話 prefix を先に処理し、送信時に利用できます。会話の KV / DeltaNet 状態は一致する prefix のみ再利用し、不一致・キャンセル時は再計算します。画像や長いコンテキストの準備にも VRAM と時間が必要です。

## GGUF を CLI で推論する

`models/model.gguf` は利用者が用意した実際のファイルに置き換えてください。CLI の `--prompt` は生のプロンプトで、GUI のようにチャットテンプレートを自動挿入しません。

```powershell
dotnet run --project NNtrain.Cli -c Release -- qwen-gguf --model models/model.gguf --prompt "こんにちは" --devices 0,1 --max-new-tokens 64
```

単一 GPU は `--device 0`、逐次出力を止める場合は `--no-stream`、Qwen3.5 のアダプターは `--adapter checkpoints/qwen35.adapter.bin` を指定します。このコマンドは greedy 生成です。

Qwen3.5 の行列形式は Q4_K / Q5_K / Q6_K / IQ2_S / IQ3_S / BF16、Bonsai の PQ2_0 / PTQ1_0 に対応します。ファイル名の IQ2_M などは混合テンソル構成を表し、各テンソルの形式を検証します。分割 GGUF は事前に結合が必要です。Bonsai PQ2_0 / PTQ1_0 の LoRA はサポート対象外です。[GGUF の詳細](docs/gguf-qwen.md) と [Bonsai 推論](docs/ternary-bonsai-generation.md) を参照してください。

## 日本語音声認識

| モデル | 必要なローカルファイル | 認識方式 |
| --- | --- | --- |
| [NVIDIA Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | `config.json`、`tokenizer.json`、`model.safetensors` | FastConformer / RNNT。日本語プロンプトとキャッシュを使うストリーミング |
| [NVIDIA Parakeet TDT/CTC 0.6B JA](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) | `parakeet-tdt_ctc-0.6b-ja.nemo` | CTC 分岐。読み込み時に必要な設定・語彙を取り出し、FP16 SafeTensors に変換 |

公式配布元から取得し、各ライセンスを確認してください。Parakeet の変換先には原本と変換後の両方のディスク容量が必要です。Python / NeMo は本番認識には不要です。

重みは FP16 で保持・読み込み、累積演算は FP32 です。元の Nemotron 配布ファイルが FP32 でも全モデルの FP32 コピーを常駐させません。CPU に加えて Arc 専用 OpenCL 経路があります。Nemotron は密行列を Arc、前処理・畳み込み・キャッシュを CPU で処理します。Parakeet は密行列・全体 attention・FFN を Arc、前処理・畳み込みを CPU で処理します。

PCM16 WAV は mono / stereo、8–192 kHz に対応し、連続した sinc リサンプリングで 16 kHz に変換します。録音は利用者の開始操作だけで行い、対応する 48 kHz デバイス形式を優先します。途中結果は未確定で、後から変わる場合があります。停止は残りの音声を処理して確定し、キャンセルは元の入力文を復元します。

Parakeet の途中表示は、入力済み音声全体を再認識する方式です。TDT とキャッシュ付き encoder ストリーミングは未実装で、1 発話は最大 30 秒です。停止時は不要な途中認識を中断し、音声末尾を含む最終認識を優先します。完全に同じ音声まで認識済みの場合だけ確定結果を再利用します。

### 検証済みの性能と限界

両 ASR の CPU ロードを高速化しました。同じ Debug 実行環境で、Nemotron は初回 21.889 → 6.784 秒、再読込 21.790 → 6.625 秒、Parakeet は初回 15.235 → 1.211 秒、再読込 15.226 → 1.236 秒でした。既存のモデルファイルから設定・語彙・重みの検証と FP16 変換までを含み、GPU 準備と Parakeet の初回 `.nemo` 変換は除外します。OS キャッシュは消去しておらず、初回は測定プロセス内の最初の実モデルロードです。詳細は [ASR 読み込み測定](docs/benchmarks/asr-loading-2026-10-06.md) を参照してください。

2026-10-06 のローカル作業ツリーでは、Ryzen 7 5700X / RAM 64 GB / Arc B580 12 GB の単一 ASR GPU、公開 FLEURS 日本語 PCM16 3 本で Parakeet を検証しました。11.10 / 22.86 / 7.98 秒の音声を 1.619 / 3.173 / 1.101 秒で認識しました。重み読み込み・アップロード時間は除外しています。以前のネイティブ実装に対して認識文は完全一致し、正規化 CER は 4.35% / 6.17% / 0% です。

22.86 秒を実時間で供給する最新 GUI テストでは、最初の表示は 1.510 秒、停止から確定までは 2.963 秒でした。先行の改善段階では停止後 6.078 秒でした。ドライバー割り当てを含む専用 GPU メモリの観測ピークは約 1.35 GB で、ASR 単体の 2 GB 目標を満たしました。共有メモリと LLM の同時使用量は別です。

この測定は未公開の高速化差分を含みます。公開済み `c1b53d2` の結果と区別してください。実マイク、30 秒の連続入力、雑音、LLM と同時使用、長時間の連続発話は未検証です。3 本での一致は一般的な認識精度を保証しません。[測定方法・数値誤差・テスト結果](docs/benchmarks/parakeet-performance-2026-10-06.md) に詳細があります。

## 学習とチェックポイント

学習は CLI から JSON 設定を指定します。引数なしの CLI は既定の学習を開始します。設定のデータパス・モデル規模・GPU・精度・出力先を確認してから実行してください。

| 設定例 | 用途 |
| --- | --- |
| [training.transformer.json](training.transformer.json) | Transformer の日本語 Wikipedia 学習、Arc の例 |
| [training.example.json](training.example.json) | Wikipedia 言語モデルの設定例。小規模な動作確認用ではありません |
| [training.forgetmemoryv2-wiki-jp.json](training.forgetmemoryv2-wiki-jp.json) | ForgetMemoryV2 |
| [training.forgetmemorydrn-wiki-jp.json](training.forgetmemorydrn-wiki-jp.json) | ForgetMemory DRN |
| [training.forgetscan-wiki-jp.json](training.forgetscan-wiki-jp.json) | ForgetScan |
| [training.hyena-wiki-jp.json](training.hyena-wiki-jp.json) | Hyena |
| [training.cifar100.json](training.cifar100.json) | CIFAR-100 分類 |

```powershell
dotnet run --project NNtrain.Cli -c Release -- --config training.transformer.json
# 保存済みチェックポイントから再開
dotnet run --project NNtrain.Cli -c Release -- --config training.transformer.json --resume
```

`--auto-resume` は再開可能な状態があれば利用します。チェックポイントはモデル重みの SafeTensors と設定・学習状態を保存します。データセットは別途準備します。[データ設計](docs/data-design.md)、[チェックポイントと学習構造](docs/project-architecture.md) を参照してください。

学習済みの独自モデルを生成に使う場合:

```powershell
dotnet run --project NNtrain.Cli -c Release -- --config training.transformer.json --generate "日本の歴史は"
dotnet run --project NNtrain.Cli -c Release -- --generate-config generate.json
```

[generate.json](generate.json) の tokenizer / checkpoint パスは実際の学習結果に合わせます。サンプリングは greedy / topK、temperature、seed を設定できます。Arc Transformer の推論は `inferenceDeviceIndices` と `arcInferenceMode` で single / tensorParallel / auto を選びます。

### バックエンドと精度

CPU は scalar / SIMD 経路を持ちます。CUDA は対応する独自モデルの GPU 経路を持ちます。Arc の汎用学習経路は Transformer 用で、DRN の LoRA / DPO をその経路では実行できません。別経路として Qwen3.5 GGUF の Arc LoRA 学習があります。

Arc Transformer は `device: "arc"` と `deviceIndices: [0]` または `[0,1]` を指定し、float32 / mix16_32 / mix8_32 / mix8_16 を選びます。2 台の学習はモデル複製とホスト経由の勾配集約を使います。推論はモデルに応じた tensor parallel や層・状態の分散を使います。Intel XMX 経路はドライバーの拡張対応によって選択されます。未対応の組み合わせは明示的に失敗します。

低精度の保存形式と演算精度は異なります。設定名だけで全演算が FP16 / INT8 になるとは限りません。[精度モード](docs/precision-modes.md)、[低ビット保存](docs/low-bit-storage-design.md)、[Arc バックエンド](docs/arc-backend-2026-09-20.md)、[Arc 推論の測定](docs/arc-generation-tuning-2026-09-24.md) を参照してください。

## LoRA / DPO

Qwen3.5 GGUF のベース重みを固定して Arc 上で LoRA を学習できます。まず [qwen-lora.example.json](qwen-lora.example.json) と [JSONL データ例](qwen-lora.example.jsonl) のパス・GPU・出力先を編集してください。

```powershell
# CPU でデータ・設定の整合性だけ確認する
dotnet run --project NNtrain.Cli -c Release -- qwen-lora --model models/model.gguf --config qwen-lora.example.json --dry-run
# 学習 / 再開
dotnet run --project NNtrain.Cli -c Release -- qwen-lora --model models/model.gguf --config qwen-lora.example.json
dotnet run --project NNtrain.Cli -c Release -- qwen-lora --model models/model.gguf --config qwen-lora.example.json --resume
```

`adapter.bin` は NNtrain 固有形式で、学習再開に必要な状態を保存します。PEFT アダプターを汎用的に読めるものではありません。NNtrain が出力した Qwen3.5 F32 GGUF LoRA は GUI で読み込めますが、学習再開には `.bin` を使います。

`messages` JSONL を整形する CPU 専用コマンドもあります。既存ファイルへの上書きを避け、新しい出力ディレクトリを指定します。

```powershell
dotnet run --project NNtrain.Cli -c Release -- qwen-lora-prepare --model models/model.gguf --data messages.jsonl --output prepared-data --epochs 2
```

変換は未対応フィールドを捨てずに失敗し、限定した既知の JSON エスケープ問題だけを修復します。詳細は [Qwen3.5 LoRA](docs/qwen35-lora.md)、独自 DRN は [LoRA](docs/lora.md) と [DPO](docs/dpo.md) を参照してください。

## プロジェクト構成と検証

| プロジェクト | 役割 |
| --- | --- |
| NNtrain.Runtime / Core | 数値実行環境、Tensor、自動微分、モデル、ASR |
| NNtrain.Arc / Cuda | GPU バックエンド |
| NNtrain.Training / Data | 学習基盤、データ読み込み、tokenizer |
| NNtrain.Cli / Gui | コマンドと Windows チャット |
| Core.Tests / Gui.Tests / IntegrationTests | 数値・状態・統合検証 |
| Benchmarks / Benchmarks.Tests | 性能測定と測定コードの検証 |

ライブラリの詳細は [Tensor](docs/tensor-semantics.md)、[自動微分](docs/autograd-design.md)、[Module](docs/module-design.md) を参照してください。

```powershell
dotnet test NNtrain.Core.Tests/NNtrain.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~AsrInputTests|FullyQualifiedName~AsrResamplerTests|FullyQualifiedName~ParakeetCtcTests"
```

GPU・実モデルのテストには対象ハードウェア、ローカル重み、明示的な実行条件が必要です。全テストを実行するときは学習・推論との競合に注意してください。測定した環境では GUI Debug ビルドは警告・エラー 0、ASR CPU 34 件と Core CPU 1,502 件が通過しました。GUI テストの通常 restore は権限問題が残り、既存 SDK を使うオフラインの 11 件を検証しています。これは全 solution の通常テスト完了を意味しません。

公開済みの統合状態は [開発統合の記録](docs/development-integration-status-2026-10-06.md)、API の変更は [セキュリティ修正記録](docs/security-scan-current-2026-10-06.md) を参照してください。ベンチマークの生ログ・重み・録音・私用データは配布物に含めません。
