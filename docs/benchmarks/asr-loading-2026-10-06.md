# ASR 読み込み高速化と検証（2026-10-06）

基準は `main c1b53d23d5bf750659595d71ceb45bbc7cf6f505` と、その後の未コミット Parakeet 認識・GUI・README 差分です。今回の読み込み変更も未コミットです。公開・merge・Release 反映はレビュー再評価まで保留しています。

## 条件と改善

Ryzen 7 5700X、RAM 64 GB、Intel Arc B580 12 GB、.NET 10 の Debug ビルドで計測しました。既存のローカル配布モデルを使い、重みの追加取得・マイク・LLM・学習を起動していません。Nemotron は約 2.55 GB の F32 SafeTensors、Parakeet は既に変換済みの約 1.25 GB の F16 SafeTensors です。Parakeet の `.nemo` 抽出・変換は今回のロード計測に含みません。

共通ローダーの工程別計測では、基準の Nemotron 初回 21.889 秒のうち読込は 0.496 秒、変換・有限値検証が 21.345 秒でした。Parakeet 初回 15.235 秒も読込 0.263 秒に対して有限値検証が 14.882 秒でした。要素ごとの処理が約 97% を占めました。

F32 には既存の管理 SIMD FP16 変換を使用し、F16 は最終配列へ直接読み込みます。有限値は SIMD と末尾のスカラー検証で確認します。ヘッダー、名前、形状、範囲、重複・欠落、全データの使用、保持メモリ予算の検証を維持し、NaN・Infinity・FP16 オーバーフローを拒否します。64 KiB 単位でキャンセルを確認し、最後のブロック後にも確認します。モデル全体の F32 コピー、永続・メモリキャッシュ、新しいネイティブ依存は追加していません。Windows では読み込み中の原本への書込みを禁止し、毎回現在のファイルを再検証します。

## CPU ロード

設定・語彙・ヘッダー・重みの読込、F32→FP16 変換、有限値検証、CPU モデル初期化を含みます。GPU は開いていません。

| モデル | 回 | 基準秒 | 改善後秒 | 倍率 |
| --- | --- | ---: | ---: | ---: |
| Nemotron | プロセス内初回 | 21.889 | 6.784 | 3.23 |
| Nemotron | 再読込 | 21.790 | 6.625 | 3.29 |
| Parakeet | プロセス内初回 | 15.235 | 1.211 | 12.58 |
| Parakeet | 再読込 | 15.226 | 1.236 | 12.32 |

OS キャッシュは消去していません。初回は測定プロセス内でその実モデルを最初にロードした回であり、冷えたディスクの保証ではありません。基準・改善後は別のテストプロセスで測り、各モデルを2回ずつ新しくロードしました。ロード後のハッシュ計算、GC、認識は CPU ロード時間から除外します。

4 回すべてで、全 FP16 重みを同じ名前順で連結した SHA256 と保持バイト数が基準と一致しました。保持重みは Nemotron `1,275,994,176`、Parakeet `1,245,077,710` バイトです。改善後の管理割当はそれぞれ約 1.28 / 1.25 GB で、基準より増えていません。これはプロセス全体のピーク RAM を保証する測定ではありません。

## 単一 Arc の準備と認識

公開 FLEURS 日本語 index 2 の 7.98 秒 PCM16 だけを使い、両モデルを各2回、新規 CPU ロード・Arc 初期化・認識・解放しました。CPU・GPU を合算した値はこの別の GPU 試験内の実測です。改善前の GPU 合算値を今回再測定したものではありません。

| モデル | 回 | CPU秒 | GPU準備秒 | 合計秒 | 認識秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| Nemotron | 初回 | 6.537 | 2.754 | 9.291 | 5.081 |
| Nemotron | 再読込 | 6.512 | 0.997 | 7.509 | 4.991 |
| Parakeet | 初回 | 1.168 | 0.957 | 2.125 | 1.183 |
| Parakeet | 再読込 | 1.202 | 0.966 | 2.169 | 1.233 |

最初の Nemotron の OpenCL lane 初期化は約 1.712 秒、後続3回は約 0.007–0.008 秒でした。GPU への重み転送は各回約 0.502–0.512 秒、CPU の準備・GPU バッファ割当は約 0.448–0.532 秒でした。初回 Parakeet の GPU 初期化も、先に Nemotron を実行した同じプロセス内での値です。

4 回とも以前の承認済み同一音声の認識文と完全一致し、CPU 基準の重み SHA256 とも一致しました。専用 GPU メモリの周期観測ピークは最大 `1,339,834,368` バイト、共有は最大 `14,934,016` バイトです。未観測の瞬間ピーク、LLM 同時負荷、異なる音声・話者の精度は保証しません。

## 回帰と GUI

- 読み込み・ASR CPU 回帰 36 件が成功。有限な F16 全ビット、F32 丸め、SIMD と 64 KiB 境界での非有限・オーバーフロー、範囲の重複・欠落・切詰め、予算、原本変更、読込中の書込み拒否、キャンセルと正常な再読込を検証しました。
- 最終 Core CPU の関連 ASR・パーサー・トークナイザー・SafeTensors 回帰は 92 件成功、失敗・skip 0。Wiki Parquet の対象回帰も7件成功。
- ASR 単独 Arc の opt-in profiler 1件成功。その中で両モデル各2回のロード・認識文一致を検証しました。
- 既存オフライン GUI 状態検証11件と、新しい実モデル CPU ロード検証6件が成功。読込中のキャンセル、再読込、途中のバックエンド切替、Parakeet 再読込、切替後の旧モデル除去を確認しました。音声デバイス選択に CPU を追加し、既定の Arc 選択は維持しています。
- 製品 GUI・Core.Tests の Debug ビルドは警告・エラー0。CLI と IntegrationTests も対象 Integration 実行時に正常ビルドしました。標準 Gui.Tests の restore 制約は残り、既存 SDK・xUnit によるオフライン検証を使用しました。オフラインコンパイラには既知の .NET 8/10 参照互換警告4件があります。
- 既存の隔離ローカル HTTP セキュリティチェックも成功。認証拒否、Origin、JSON 要件、ネットワークパス拒否、同時受付上限を確認し、モデル・GPU・マイクは起動していません。

## セキュリティと公開範囲

[既存の11項目の修正記録](../security-scan-current-2026-10-06.md)に対応する GGUF/BPE/SafeTensors/Parquet/ローカル API のソースとセキュリティ試験は、基準 `c1b53d2` から変更していません。今回の変更で検証を省略していません。外部スキャン未再実施、悪意ある Parquet デコードの fuzz 未実施、第三者展開処理の網羅性不足、標準 Gui.Tests restore、実マイク・LLM 同時負荷は残る制約です。包括的な監査完了や公開承認を意味しません。

公開候補は下記24ファイルに限定します。元の未コミット変更を含み、Git の全追加操作は行っていません。

```text
NNtrain.Arc/Kernels/asr_linear.cl
NNtrain.Core/Audio/AsrArcLinear.cs
NNtrain.Core/Audio/AsrCpuMath.cs
NNtrain.Core/Audio/AsrHalfCheckpoint.cs
NNtrain.Core/Audio/ILocalAsrModel.cs
NNtrain.Core/Audio/NemotronAsrModel.cs
NNtrain.Core/Audio/ParakeetCtcEncoder.cs
NNtrain.Core/Audio/ParakeetCtcModel.cs
NNtrain.Core.Tests/AsrArcTests.cs
NNtrain.Core.Tests/AsrLoadPerformanceTests.cs
NNtrain.Core.Tests/AsrLoadSafetyTests.cs
NNtrain.Core.Tests/ParakeetCtcTests.cs
NNtrain.Core.Tests/ParakeetPerformanceTests.cs
NNtrain.Gui/MainWindow.Audio.cs
NNtrain.Gui.Tests/AsrLoadingGuiChecks.cs
NNtrain.Gui.Tests/AudioGuiTests.cs
NNtrain.Gui.Tests/OfflineRunner.cs
NNtrain.Gui.Tests/RealtimeAudioGuiChecks.cs
NNtrain.Gui/README.md
README.md
docs/gui-openai-api.md
docs/benchmarks/parakeet-performance-2026-10-06.md
docs/development-integration-status-2026-10-06.md
docs/benchmarks/asr-loading-2026-10-06.md
```

モデル、録音、認識文を含む生測定JSON、ログ、TRX、コンパイル済みバイナリ、bin/obj、キャッシュ、個人 Library 識別ファイル、学習データ・生成物、作業用バックアップは除外します。候補ソース・資料の秘密値形式と個人の絶対パスを対象にした文字列検査で該当0を確認し、別のローカル証跡一覧に最終ファイルハッシュを残しました。これは網羅的な秘密情報検出を保証する検査ではありません。

## ローカル証跡

生成物は公開対象外です。親レビューは `benchmark-results/asr-loading-20261006` の次の証跡を検証してください。

- `cpu-baseline.json` / `cpu-optimized.json`: CPU の工程別比較と重み SHA256。
- `arc-optimized.json` / `arc-optimized.trx`: 各2回の GPU 準備・認識文一致とメモリ。
- `load-safety.trx` / `cpu-final-security.trx` / `parquet-security.trx`: 読み込み・関連セキュリティ回帰。
- `gui-loading.json`: 実モデル CPU のキャンセル・再読込・切替6件。
- `gui-check/compile.log`: 最新オフライン GUI コンパイル。
- `review-manifest.json`: 公開差分候補と除外、最終ファイルハッシュ、修正維持の確認。

今回の新規ロード性能は認識速度・停止後確定とは別です。既存の認識高速化は [Parakeet 性能報告](parakeet-performance-2026-10-06.md) を参照してください。
