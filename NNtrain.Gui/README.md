# NNtrain Windows チャット GUI

Intel Arc 上のローカル GGUF と会話する WPF アプリです。モデル・LoRA・画像・日本語音声入力を同じチャットに接続します。学習は [CLI](../README.md#学習とチェックポイント) から行います。

## 起動

Windows と .NET 10 SDK、Intel Arc の OpenCL ドライバーを用意し、リポジトリのルートで実行します。

```powershell
dotnet run --project NNtrain.Gui -c Debug
```

配布用ビルド:

```powershell
dotnet publish NNtrain.Gui/NNtrain.Gui.csproj -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true
```

`NNtrain.Gui/bin/Release/net10.0-windows/win-x64/publish/NNtrain.Gui.exe` を実行します。モデル重みは実行ファイルに含みません。

起動時の設定例です。ファイルのパスは実際のものに置き換えてください。

```powershell
.\NNtrain.Gui.exe --model C:\models\base.gguf --lora C:\models\adapter.bin --top_p 0.95 --top_k 20 --temperature 0.6 --maxtokens 512 --stream on --think on
```

`--lora` は省略可能です。`--top-p` / `--top-k`、`--max_tokens` も利用できます。Stream / Thinking の値は `on` / `off` です。モデル・LoRA のパスに親ディレクトリ参照 `..` は指定できません。

## モデルを読み込んで会話する

1. 「モデル・推論設定」を開き、GGUF と使用 GPU を選びます。Arc 2 台を検出すると両方を選択できます。
2. 必要なら LoRA と mmproj を選びます。
3. 「プリロード」を押します。読み込み状態は GUI、詳細な処理はサーバーのコンソールで確認します。
4. 入力欄に質問を書き、送信します。生成途中は停止できます。

現在の GUI は `general.architecture=qwen35` の対応 GGUF を扱います。モデル名に Qwen3.8 などを含む場合でも、内部 architecture とテンソルを検証します。Qwen2 / Qwen2.5 は CLI 経路を使います。Qwen3.5 MoE や任意の GGUF を読み込めるわけではありません。

LoRA は NNtrain の `adapter.bin` と NNtrain がエクスポートした Qwen3.5 F32 GGUF LoRA に対応します。ベースモデルの識別情報・テンソル形状を検証します。学習再開には `.bin` が必要です。Bonsai PQ2_0 / PTQ1_0 の LoRA は対象外です。

Stream On は逐次表示、Off は完了後にまとめて表示します。Thinking は初期状態で On、推論内容は折りたたんで表示します。過去の思考本文は次の質問へ含めず、回答本文を会話として保持します。temperature / top-p / top-k / 最大出力数は設定から変更します。

モデルは会話間で常駐し、最後のプリロードまたは生成完了から 5 分間操作がなければ解放します。次の送信では再読み込みします。モデル・LoRA・GPU の切り替えも再読み込みを伴います。

## Hugging Face から取得する

Hugging Face タブにモデルのリポジトリ ID または URL を指定して調べ、GGUF を選びます。mmproj は同じリポジトリ、または追加の mmproj リポジトリから選択できます。

- 対象は GGUF / mmproj です。ASR の `.nemo` / SafeTensors は別途取得します。
- 取得先は検出した models ディレクトリ内の `huggingface/<owner>/<repository>` です。
- リポジトリのコミットを固定し、サイズと利用可能なハッシュを検証してから完成ファイルにします。同一コミットの部分ファイルは再開できます。
- トークンは任意で、設定ファイルには保存しません。アクセス制限とライセンス条件は公式配布元で確認してください。
- 分割 GGUF は読み込む前に結合してください。

## 画像を入力する

対応する Qwen3.5 mmproj GGUF を選択し、PNG / JPEG の静止画を添付して送信します。ベースモデルと mmproj の互換性が必要です。動画や外部画像 URL の汎用入力ではありません。

プリロードは mmproj も初期化します。読み込み済みなら画像添付時に画像処理と既知の会話 prefix の prefill を進めます。送信が先に押された場合は準備完了を待ちます。画像はモデルが指定する前処理に従って処理します。

テキスト会話は一致する prefix の KV / DeltaNet 状態を再利用します。画像付き会話はトークン・画像埋め込み・位置の一致も検証します。準備範囲外の入力、不一致、キャンセルは安全に再計算します。入力長・再利用数・最初の出力までの時間は状態欄で確認できます。

## 音声からプロンプトを作る

音声設定はチャット画面にあります。ASR 用 GPU は LLM と独立して選び、単一 Arc または CPU を利用します。音声認識は C# / OpenCL のローカル処理です。

### 重みを準備する

公式配布元のアクセス条件・ライセンスを確認し、次のファイルを同じフォルダーに置きます。

| 選択する ASR | 配布元と必要なファイル |
| --- | --- |
| Nemotron 3.5 | [公式モデル](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) の `config.json`、`tokenizer.json`、`model.safetensors` |
| Parakeet 日本語 | [公式モデル](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) の `parakeet-tdt_ctc-0.6b-ja.nemo` |

既定のフォルダー名は `models/nemotron-3.5-asr-streaming-0.6b` と `models/parakeet-tdt_ctc-0.6b-ja` です。別の場所は参照ボタンから指定します。

「FP16 読み込み」で重みを読み込みます。Nemotron の元ファイルは約 2.55 GB ですが、保持する重みは約 1.28 GB の FP16 です。Parakeet は初回に `.nemo` から設定・語彙を抽出し、約 1.25 GB の FP16 SafeTensors に変換します。原本と変換後を置ける空き容量が必要です。前処理・累積演算・キャッシュは重みとは別の容量を使います。GUI の重み表示だけでは総 VRAM を表しません。

ASR のデバイス選択には CPU もあります。2026-10-06 の Debug CPU 測定では、Nemotron は初回 21.889 → 6.784 秒、再読込 21.790 → 6.625 秒、Parakeet は初回 15.235 → 1.211 秒、再読込 15.226 → 1.236 秒へ短縮しました。設定・語彙・重みの検証を含み、GPU 準備と `.nemo` 変換は別計測です。OS キャッシュは消去していません。単一 Arc を含む最新の準備時間は Nemotron 9.291 / 7.509 秒、Parakeet 2.125 / 2.169 秒でした。最初の Nemotron は OpenCL 初期化を含み、後続ロードは同じプロセス内です。[測定条件と検証](../docs/benchmarks/asr-loading-2026-10-06.md) を参照してください。

### 入力・停止・編集・送信

1. ASR のモデル・フォルダー・GPU を選んで読み込みます。
2. 「録音」を押すか、16bit PCM WAV を選びます。自動的に周囲を録音することはありません。
3. 途中の認識文が既存の下書きに反映されます。認識中は確定前の編集を防ぎます。
4. 「停止」で末尾まで処理して確定します。キャンセルは元の下書きを復元します。
5. 確定文を確認・修正し、通常の送信ボタンで現在のローカル LLM に送ります。ASR が勝手に送信することはありません。

WAV は PCM16、mono / stereo、8–192 kHz を受け付け、16 kHz にリサンプリングします。録音は対応する 48 kHz stereo / mono を優先し、16 kHz の形式へフォールバックします。Float WAV、MP3 等を直接認識する入力ではありません。キューは有限で、処理が追いつかない場合は黙って音声を落とさずエラーにします。

Nemotron は日本語 RNNT とキャッシュ付き encoder ストリーミングです。Parakeet は CTC 分岐を使い、入力済み音声全体の再認識で途中表示します。TDT とキャッシュ付き encoder ストリーミングは未実装で、最大 30 秒 / 発話です。途中結果は書き換わる場合があります。

Parakeet の停止は任意の途中認識を中断し、残りのサンプルを取り込んだ最終認識を優先します。最後の完全な入力と同じところまで既に認識済みなら結果を再利用します。短い途中結果を最終結果とみなして末尾を捨てることはありません。開始・停止の連打、キャンセル、モデル切り替えには世代管理と状態制御があります。

### 測定と未検証事項

2026-10-06、Ryzen 7 5700X / RAM 64 GB / 単一 Arc B580 12 GB、公開 FLEURS 日本語 3 本で検証しました。未公開の高速化差分を含む作業ツリーの値です。

| 音声長 | ファイル認識時間 | 正規化 CER |
| --- | ---: | ---: |
| 11.10 秒 | 1.619 秒 | 4.35% |
| 22.86 秒 | 3.173 秒 | 6.17% |
| 7.98 秒 | 1.101 秒 | 0% |

読み込み・アップロード時間は除外し、以前のネイティブ認識文と 3 本すべて完全一致しました。22.86 秒を実時間供給する最新 GUI テストは最初の文字表示 1.510 秒、停止後確定 2.963 秒でした。ASR 単体の専用 GPU メモリ観測ピークは約 1.35 GB、共有メモリは別計測です。LLM の同時メモリは含みません。

確定文の編集と既存 Send 経路への接続は、実 ASR と模擬ローカル HTTP で検証しています。この模擬送信を実 LLM の回答生成とは区別します。実マイク、30 秒の連続入力、雑音、長時間運用、LLM 同時負荷は未検証です。詳細は [性能・状態検証レポート](../docs/benchmarks/parakeet-performance-2026-10-06.md) を参照してください。

## ローカル API と困ったとき

GUI は別プロセスの OpenAI 互換ローカル API に送信します。セッションごとの Bearer 認証があり、GUI は自分の認証情報をサーバーに渡します。外部からアクセスする場合は、自分で起動したサーバーの認証情報を使います。[API ガイド](../docs/gui-openai-api.md) に起動・HTTP 例があります。

読み込みに失敗したら architecture、テンソル形式、ベースと LoRA / mmproj の組み合わせを確認します。VRAM は重み以外に KV・画像・一時領域を使います。状態欄だけで解決しない場合はサーバーのコンソールを確認してください。

開発時の通常 GUI テスト restore には現在の環境で権限問題が残ります。製品 GUI の Debug ビルドと既存 SDK によるオフライン 11 件の状態検証は通過しました。通常の全 solution テストを完了したという意味ではありません。
