# NNtrain チャット

Windows用のローカル推論GUIです。起動時に同じ実行ファイルから別プロセスのOpenAI API互換サーバーを立ち上げ、GUIは `http://127.0.0.1:<自動選択ポート>/v1/chat/completions` を呼びます。サーバーの処理とログは別のCUIウィンドウに表示されます。GUIを閉じると所有するサーバーも終了します。

GGUFを選び、必要ならNNtrainのLoRAチェックポイント（.bin）またはエクスポート済みGGUF LoRA（.gguf）を選んで、Intel Arc GPUで会話できます。学習コマンドや学習画面はありません。

- モデルの「プリロード」で重みをGPUに読み込み、後続の送信でも保持します。mmproj を選んでいる場合は画像モデルも同時に準備するため、最初の画像送信中に画像モデルの初期化を待つ必要がありません。
- モデルが読み込み済みなら、画像の添付時から画像処理を開始し、既知の会話履歴と画像の最後の埋め込み行まで言語モデルも事前計算します。入力欄で編集中の質問は含めません。「画像準備済み」の表示後は保存できた状態を再利用し、質問の続きから計算します。準備中に送信すると、準備の完了を待ってから生成します。画像の解像度や内容は変更しません。mmproj だけを切り替えた場合は画像モデルを交換し、大きな言語モデルの重みは読み込み直しません。
- 「モデル・生成設定」をクリックすると設定欄を折りたたみ、チャット表示を広げられます。設定値は保持されます。
- 「使用GPU」でArc 0＋1または各GPU単独を選べます。2台見つかった場合は両方を初期選択し、モデルの層とKVキャッシュを分散します。選択を変えると読み込み直します。
- StreamのOn/Offを切り替えられます。OnならOpenAI形式のSSEで逐次表示し、Offなら完了後にまとめて表示します。送信の停止、モデルやLoRAの切替にも対応します。
- ThinkingのOn/OffはQwen3.8のGGUFチャットテンプレートにある生成開始部分を使います。
- Thinkingは初期状態でオンです。オフにするとモデルへ思考しないよう指示します。
- 生成の初期値は temperature 0.6、top-p 0.95、top-k 20 です。GUIで変更できます。temperature 0 はgreedyです。
- Thinking中は「思考中…」を表示します。見出しをクリックすると内容を確認でき、生成が終わると自動で折りたたみます。回答は思考文と分けて表示します。
- 過去ターンの思考本文は次の推論プロンプトに含めず、回答本文だけを履歴に残します。テンプレートの空の思考区切りは保持します。
- Thinkingをオンにした生成で回答に到達しない場合は、一度だけThinkingオフで回答を生成し直します。最初の思考文は折りたたんで残します。
- Thinkingオフでもモデルが `</think>` を余分に出した場合は、その前を折りたたみ可能な思考として扱い、回答には終端後の文だけを表示します。思考だけで生成が終わった場合は、回答が未生成であることを表示します。
- 最後のプリロードまたは生成完了から5分間操作がなければサーバーがモデルを解放し、GPUメモリを空けます。GUIはサーバーの状態を表示し、次の送信時は選択中のモデルを再読み込みします。
- 現在のモデル読込経路は `general.architecture=qwen35` のモデルです。Qwen3.8-27BやBonsai PQ2_0/PTQ1_0を使用できます。対応する Qwen3.5 mmproj GGUF を選ぶと PNG / JPEG の静止画像を NNtrain 本体で処理できます。Qwen2.5 はこのGUIでは扱いません。
- NNtrainの `adapter.bin` と、NNtrainが出力したQwen3.5 F32 GGUF LoRAを直接読み込めます。GGUFではベースモデルSHA-256、A/Bテンソル名・形状・有限値を検査します。GGUFにはAdam状態がないため、学習再開には `adapter.bin` を使用します。Bonsai PQ2_0/PTQ1_0のLoRAはサポート対象外です。
- モデルは会話間で常駐します。テキスト会話では前回プロンプトのトークン列が次のプロンプトの先頭と一致すれば、GPU上のKVとDeltaNet状態を保持して続きだけを計算します。画像では添付時に明示的に準備した範囲だけを再利用し、トークンID、画像の全埋め込み値のビット列、MRoPE位置の一致を確認します。表示用の会話履歴には思考本文を残さず、回答だけを入れます。
- 一致しない場合や生成を停止した場合は、安全のため全プロンプトを再計算します。最初の質問もプリロードだけでは事前計算されません。生成完了時の状態欄に入力数、再利用数、最初の出力までの時間を表示します。
- KVキャッシュの確保領域は生成後も保持し、モデル解放時に解放します。長い会話では使用量が増えます。GPUの空き容量が少ない場合は追加の状態保存を見送り、通常の推論を続けます。

ビルド例（`NNtrain` フォルダーでPowerShellから実行）：

```powershell
dotnet publish .\NNtrain.Gui\NNtrain.Gui.csproj -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true
```

実行ファイルは `NNtrain.Gui\bin\Release\net10.0-windows\win-x64\publish\NNtrain.Gui.exe` です。モデルとLoRAは実行ファイルに含めず、既存の `models`、`checkpoints` フォルダーから選びます。

起動引数で初期設定を指定できます。例えば：

```powershell
.\NNtrain.Gui.exe --model C:\models\base.gguf --lora C:\models\adapter.gguf --top_p 0.95 --top_k 20 --temperature 0.6 --maxtokens 512 --stream on --think on
```

`--lora` は省略できます。`--tempreture` も `--temperature` の別名として受け付けます。モデル・LoRAのパスには `../` または `..\` を含む親ディレクトリ参照を指定できません。

APIサーバーだけを起動する場合は `NNtrain.Gui.exe --server --port 8000` を使います。`--lora C:\models\adapter.gguf` または `--lora C:\models\adapter.bin` で既定のLoRAを指定でき、`POST /internal/shutdown` で終了します。

## Local Japanese ASR (Nemotron 3.5)

The chat composer loads `nvidia/nemotron-3.5-asr-streaming-0.6b` from a local directory containing `config.json`, `tokenizer.json`, and `model.safetensors`. The published FP32 checkpoint is converted incrementally to FP16 storage; no ASR service, external LLM API, or Python runtime is used. Keep the downloaded checkpoint outside version control.

Choose an individual Arc GPU, load FP16 weights, then select a PCM16 WAV or explicitly press the recording button in the chat composer. No separate speech tab is required. Recognized text appears directly in the existing chat input while recording. Existing typed text (including a selection) is preserved; voice text is inserted after the caret/selection. Stop drains recognition and enables editing. Cancel restores the original draft and selection and rejects stale callbacks. The regular chat Send button sends the edited prompt to the selected local GGUF/LoRA through the existing server. Recording and chat generation are serialized; every utterance receives fresh Nemotron caches.

WAV input supports PCM16 mono/stereo at 8-192 kHz. The recording API queries 48 kHz stereo, then 48 kHz mono, then falls back to 16 kHz mono if needed. It displays the format accepted by WinMM, which may differ from the microphone hardware/mixer rate because Windows can convert formats. A bounded background worker mixes channels and performs explicit continuous conversion to 16 kHz. Sinc filter history, fractional phase and a split stereo frame carry across callbacks; the final tail is flushed exactly once. Physical capture remains untested: tests inject approved files or synthesized PCM and never open the microphone.
Arc executes dense projections and decoder matrices using FP16 resident weights with FP32 accumulation. Audio features, convolutions, attention/cache bookkeeping and nonlinearities currently run on CPU. ASR uses one Arc device even when the selected LLM uses both. Memory coexistence was verified with the existing 27B IQ2_M model on both cards; other model sizes remain unverified; the ASR allocation is additional to a resident LLM. Recording is bounded to five minutes, and a full audio queue reports an error rather than silently dropping audio. File input is useful on systems where recognition runs slower than realtime.

Validation on 2026-10-06: 17 CPU/Arc checks passed, including FP16 projections on both B580 cards and miniature Conformer/RNNT cache parity. The final real-model test recognized one 11.1 s public FLEURS Japanese utterance in 6.60 s (a GUI integration run took 6.00 s), down from 12.29 s before batching projections and caching relative-position projections across chunks. Checkpoint loading on CPU took 21.72 s; Arc upload is additional. Recognition text stayed unchanged: two incorrect characters (`帰国` recognized as `企画`), normalized CER 4.35%. This is one sample, not a general accuracy or latency benchmark. FP16 host weights: 1,275,994,176 bytes; CPU caches: 23,608,320 bytes, including the new bounded position cache. GPU buffer peak: 1,259,554,500 bytes. Windows GPU Process Memory sampled every 100 ms reported ASR dedicated peak 1,341,927,424 bytes and shared peak 14,934,016 bytes (84 samples). Dedicated memory stayed below 2 GB including driver-managed allocations in this workload.

Evidence: `benchmark-results/asr-20261006/real-arc-japanese-final.json`, `benchmark-results/test-results/asr-real-arc-final.trx`, `benchmark-results/asr-20261006/real-gui-llm.json` and `gui-check/real-gui-log-final.txt`. The file recognition reached the hidden GUI transcript, was edited with a summary instruction, and sent through the actual local HTTP server to the existing Qwen3.8 27B IQ2_M model distributed across Arc 0/1. The real response was 「海外ギャップイヤーコースは、大学進学を有利にする。」. ASR stayed resident during LLM loading and generation; both GPUs together peaked at 12,534,050,816 bytes dedicated and 1,107,550,208 bytes shared (336 process-memory samples). This validates this model combination, not every LLM size. The GUI integration used the real controls, event handlers and local server without showing the window or opening a microphone. Test speech is [Google FLEURS](https://huggingface.co/datasets/google/fleurs), CC BY 4.0; provenance is recorded alongside the sample. Its float WAV was explicitly converted to PCM16. Four additional hidden-window checks cover transcript editing, stale callbacks, active-operation guards and mock chat transport. Manual microphone/device operation and visual GUI interaction remain unverified. Full numerical parity against the [official model](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) Python implementation remains unverified.

CPU checks:
```powershell
dotnet test NNtrain.Core.Tests/NNtrain.Core.Tests.csproj --filter 'FullyQualifiedName~AsrInputTests|FullyQualifiedName~NemotronMelTests|FullyQualifiedName~NemotronModelTests'
dotnet test NNtrain.Gui.Tests/NNtrain.Gui.Tests.csproj
```
GPU checks are opt-in:
```powershell
$env:NNTRAIN_ASR_ARC_TEST='1'
dotnet test NNtrain.Core.Tests/NNtrain.Core.Tests.csproj --filter 'FullyQualifiedName~AsrArcTests'
```
Real-model checks are separately opt-in; set `NNTRAIN_ASR_REAL_TEST=1`, `NNTRAIN_ASR_MODEL`, `NNTRAIN_ASR_TEST_WAV`, `NNTRAIN_ASR_REPORT` and optionally `NNTRAIN_ASR_REFERENCE`. They read an existing file and never activate a microphone. New test-project restore was blocked in this execution environment by access to the user NuGet configuration; the same four GUI checks were compiled with installed SDK/reference assemblies and existing xUnit binaries instead.

### Paced live-input validation

Recording feeds PCM16 chunks every 320 ms into a bounded queue. The recognition worker consumes each chunk immediately with persistent frontend/convolution/attention/RNNT caches and posts text onto the WPF dispatcher before recording stops. The microphone and the paced-file check use the same `ProcessAudioAsync` consumer. Encoder conditioning projections are now batched across the current chunk; the stateful RNNT symbol loop remains sequential. Lookahead stays at 3 encoder frames (240 ms), and greedy decoding is unchanged. No latency/accuracy setting was reduced.

`benchmark-results/asr-20261006/realtime-gui.json` records a real-time simulation made from five repetitions of the permitted Japanese recording, separated by one second of silence (60.5 s). Input was released according to its sample clock, rather than pushed as a batch. The real WPF transcript control changed 112 times before stop. First nonempty text appeared 2.724 s after input start; median interval between changed texts was 0.324 s. The maximum text interval was 5.125 s, including silence and intervals when the decoder emitted no new characters; updates are not promised every 320 ms. The queue peaked at one chunk and drained to zero. Sampled processing lag (wall clock minus completely processed input duration) peaked at 0.518 s, including the last 20 seconds; it did not accumulate. Stop-to-final time was 0.342 s. A cancelled session cleared its transcript, rejected stale callbacks, and the next fresh session produced only its own text; restart first text was 2.730 s and stop-to-final time 0.179 s. Duplicate start calls were rejected. Six separate GUI state/transport checks and 17 CPU/Arc checks passed; GUI build has zero warnings/errors.

The long repeated input had normalized CER 7.39%; the restarted single sentence had CER 4.35%. This repetition/silence test is not a general transcription benchmark, and the long stream was not reset at silence boundaries. First-text latency is measured from input start, not annotated speech onset. Measured dedicated GPU peak was 1,341,894,656 bytes, shared peak 14,934,016 bytes (928 samples at approximately 100 ms). No microphone or local LLM server was opened by this live test. Physical microphone timing and visible-window rendering remain user/manual checks. The finalization guard added after the paced run was independently checked to ensure late partial callbacks cannot overwrite confirmed or edited text.

To use live transcription: load FP16 weights, press the Record button beside the chat input, then Stop to confirm and edit. Use the regular Send button. Cancel restores the preceding typed draft. The Windows default recording device is used; there is no in-app input-device selector.

### Chat integration and continuous conversion validation

The latest hidden-window checks cover ten chat integration, cancellation, model-switch, insertion, transport and layout cases, including 760x580 content rendering. `asr-chat-resample-tail-cpu.trx` records 23 passing CPU checks. Continuous PCM tests compare every sample against whole-file conversion for 48 kHz mono/stereo, 44.1 kHz stereo, 16 kHz mono and 8 kHz stereo with arbitrary odd chunk lengths; output lengths, antialias filtering and finalization are checked. The official Nemotron generator consumes all encoder frames of a padded final chunk; the C# path now follows that behavior. Earlier real-model latency/CER results above predate this tail change and chat integration; they are not fresh measurements of these revisions.

Parakeet Japanese is selectable alongside Nemotron. The official 2,490,951,680-byte NeMo archive was downloaded with approval and its SHA256 matches `a2ed2aab9c82cce4e9b699dca669184d7525c7b5e1eb8c3f7b789421bc3637e8`. Keep its CC BY 4.0 attribution in the model directory. The C# data-only reader does not execute pickle functions and converts ZIP tensor storage directly inside the tar to FP16 safetensors. Conversion took 29.70 s; CPU loading took 15.24 s; host FP16 weights are 1,245,077,710 bytes. `asr-dual-model-cpu.trx` records 31 CPU checks; ten GUI checks also passed, including model switching and collapsed/expanded layout renders.

The implemented Parakeet decoder is the model's supported **CTC** branch, not its default TDT branch. It uses the maintained NeMo 3.0.0 evaluation frontend: 80 bands, centered zero STFT padding, normalization over valid `floor(samples / hop)` frames, a zero terminal frame, and length-aware intermediate subsampling masks. The encoder has noncausal subsampling, global relative attention and evaluation BatchNorm. Global attention requires reprocessing the current utterance for preview; unlike Nemotron it does not reuse streaming encoder caches. Preview is requested after each additional four seconds of input and may revise previous text. A Parakeet utterance is currently bounded to 30 seconds. Recorded-file use is preferable on this implementation; real-time microphone performance has not been established.

`parakeet-real-arc-japanese.json` records actual Arc CTC recognition of the approved 11.1 s Japanese recording in 9.04 s, CER 4.35%, GPU upload 1.02 s, dedicated process peak 1,330,778,112 bytes and shared peak 8,642,560 bytes (93 approximately-100ms PDH samples). A separate real-ASR composer check produced three text updates, retained the original typed draft, allowed editing and sent the exact edited prompt through the existing Send handler into a **mock** local transport. With preview recomputation this took 15.55 s for 11.1 s of audio, so this mode exceeded real time; its dedicated peak was 1,336,610,816 bytes. It started no microphone or LLM server. Earlier real Nemotron-to-LLM evidence remains separate from this mocked-LLM check.

`realtime-chat-integrated-final.json` reruns Nemotron after chat integration and final-padding correction: 60.5 s paced input, first text 2.730 s, 112 updates before stop, median changed-text interval 0.324 s, queue maximum one/final zero, maximum processing lag 0.521 s and stop-to-final 0.343 s. Cancellation and restart passed; restart first text 2.717 s and stop-to-final 0.183 s. Long-input CER remained 7.39%, restarted single-utterance CER 4.35%. Dedicated peak was 1,339,830,272 bytes, shared 14,934,016 bytes (923 samples). These ASR process measurements include driver allocations and meet 2 GB in these tested conditions, not for every input or concurrent model combination.

Later official comparisons cover three distinct public FLEURS Japanese validation recordings (indices 0, 1, 2). Both C# Arc models match their official FP16-rounded CPU reference transcripts after punctuation/spacing normalization on all three. This small sample does not establish general accuracy equivalence or broad accuracy improvement. The original FP32 reference is measured separately because FP16 rounding can change recognition. Nemotron frontend max absolute error is 6.77e-5; its two cached short encoder chunks differ by at most 2.56e-6 on Arc. After correcting Parakeet terminal normalization, STFT boundary padding and intermediate subsampling masks, its full 11.1 s encoder differs by at most 5.01e-6 (mean 2.46e-7). Earlier Parakeet reports above predate those fixes; current results are `parakeet-real-arc-japanese-final-{0,1,2}.json`, `parakeet-layer-comparison.json`, and `official-asr-accuracy-summary.json`.

Parakeet's 22.86 s file takes about 31.6 s in native C#/Arc, versus about 1.32 s in the official four-thread CPU reference. CPU convolution/attention still dominate this C# implementation, so this case exceeds real time even without repeated preview. Tested ASR dedicated process peaks remain below 1.36 GB, including driver allocations; combined LLM+ASR memory is a separate measurement. Physical microphone behavior and native device mixing remain unverified. The approved workspace-only Python CPU environment is a validation tool; it is not used by the production C# GUI. See `tools/asr-reference/README.md` for reproducible reference settings. Current CPU ASR regression coverage is 32 checks and GUI state/layout/mock-transport coverage is ten checks.
