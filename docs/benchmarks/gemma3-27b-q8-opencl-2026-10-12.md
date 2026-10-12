# Gemma 3 27B Q8_0: OpenCL 実装と測定

この記録は当初の 40 層版の測定です。現在の同梱 DLL と GUI 設定は、[43 層・XMX GEMM・初期化再利用の追加最適化](gemma3-opencl-aggressive-2026-10-12.md)を参照してください。

2026-10-12。NNtrain の `gemma-gguf` と GUI 推論セッションに、ローカルの OpenCL ネイティブバックエンドを接続した。推論はプロセス内で実行し、外部 LLM サーバーに依存しない。

## 実装

- `NNtrain.Gemma/Gemma3LocalModel.cs` で CPU / OpenCL を明示的に選び、OpenCL 指定時だけ GPU 層を割り当てる。Q8_0 を含む Gemma 3 GGUF の architecture とテンソル型を読込前に検証する。
- `NNtrain.Gemma/gemma-opencl/` に Windows x64 OpenCL 対応のネイティブ DLL と再ビルド手順を収めた。LLamaSharp 0.27.0 と ABI を合わせるため、llama.cpp のコミット `3f7c29d318e317b63f54c558bc69803963d7d88c` を使った。OpenCL SDK は Khronos Headers / ICD Loader v2024.10.24。
- `NNtrain.Cli` に `--backend cpu|opencl` と `--opencl-all-devices` を追加した。`--gpu-layers N` を指定すると、明示的に CPU を要求していない限り OpenCL を選ぶ。CPU と GPU の矛盾する指定はエラーにする。`--opencl-all-devices` は Intel OpenCL 上の複数 GPU を利用可能にする。
- Windows の C# プロセスでは `Environment.SetEnvironmentVariable` だけでは OpenCL 側の UCRT `getenv()` に選択が反映されなかったため、ネイティブ初期化前に `ucrtbase.dll` の `_putenv_s` でも `GGML_OPENCL_PLATFORM` / `GGML_OPENCL_DEVICE` を設定する。修正前は [NVIDIA プラットフォームを選んで失敗](../../benchmark-results/gemma3-opencl-dual-nntrain-cli-ctx4096-20261012.log)し、修正後は [Intel 2 台で生成](../../benchmark-results/gemma3-opencl-dual-nntrain-cli-ctx4096-fixed-20261012.log)した。
- GUI の Gemma 3 モデル設定に「OpenCL GPU（自動）」を追加した。B580 が 2 台とも総 VRAM 11 GiB 以上なら 40 層を 2 台に、そうでなければ単一 GPU に 20 層を割り当てる設定である。CPU と OpenCL を切り替える際はローカル推論サーバーを再起動する。Gemma 3 に対する GPU 番号指定、LoRA、画像入力は現時点で対象外。

## 対象と測定条件

- モデル: `models/huggingface/ggml-org/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q8_0.gguf`、28,707,604,864 バイト、SHA-256 `794d14393b9a15de7f08da16dc35c503edb3eb48dd58fa5beb37b7cf2a188379`。
- GPU: Intel Arc B580、OpenCL 3.0 NEO、ドライバー 32.0.101.9030。`GGML_OPENCL_PLATFORM=Intel` と `GGML_OPENCL_DEVICE=B580` で単一 B580 を選択した。CPU は 8 スレッド。
- 同じ入力（1 から 50 まで数える指示、28 prompt tokens）、context 512、最大 40 新規 tokens、temperature 0 で `llama-completion` をそれぞれ別プロセスで実行。OpenCL 側は 20/63 層を GPU に置き、CPU 側は 0/63 層。どちらも 39 decode runs。測定中に他のモデル実行・ネイティブビルドを行わなかった。
- 生ログ: [CPU](../../benchmark-results/gemma3-27b-q8-cpu-long-20261012.log)、[OpenCL 20 層](../../benchmark-results/gemma3-27b-q8-opencl20-long-20261012.log)。以下は各条件 1 回の実測であり、複数回測定の統計値ではない。

| 経路 | Prompt eval | Decode | 合計 |
| --- | ---: | ---: | ---: |
| CPU | 28 tokens / 8,367.04 ms = 3.35 tokens/秒 | 39 runs / 41,302.79 ms = 0.94 tokens/秒 | 49,702.84 ms |
| OpenCL、B580 1 台、20 層 | 28 tokens / 4,393.94 ms = 6.37 tokens/秒 | 39 runs / 25,668.56 ms = 1.52 tokens/秒 | 30,086.78 ms |

OpenCL はこの入力で prompt eval が約 1.90 倍、decode が約 1.62 倍だった。GPU モデルバッファ 9,378.43 MiB、KV 76 MiB、計算バッファ 522.50 MiB で、OpenCL 側の自己使用はログ上約 9,976 MiB。CPU には未配置層の重みが残るため、この方式は CPU メモリも必要とする。

### 単一 B580 の GPU 層数を比較

[llama-bench の結果 JSON](../../benchmark-results/gemma3-opencl-ngl-tuning-20261012.json)で、同じ Gemma 3 Q8_0、OpenCL の単一 B580、CPU 8 スレッド、`pp32` / `tg32` を各 2 回測定した。表はトークン/秒の平均 ± 標準偏差で、`tg32` の範囲は 2 回の最小値から最大値。短いベンチなので、長文生成の速度や統計的な優劣を確定する値ではない。

| GPU 層数 | pp32 平均 ± SD | tg32 平均 ± SD | tg32 範囲 |
| ---: | ---: | ---: | ---: |
| 18 | 7.81 ± 0.83 | 1.35 ± 0.02 | 1.34–1.36 |
| 20 | 7.91 ± 0.22 | 1.37 ± 0.08 | 1.32–1.43 |
| 21 | 9.33 ± 0.80 | 0.91 ± 0.40 | 0.63–1.19 |
| 22 | 7.41 ± 0.43 | 1.32 ± 0.04 | 1.29–1.35 |

21 層の `tg32` は 2 回で大きくばらついた。20 層はこの小規模比較で 18 / 22 層と同程度であり、別途 CLI・GUI の生成まで確認済みなので、実運用の単一 GPU 設定は 20 層を維持した。

## NNtrain CLI での生成確認

別の日本語入力 `2足す3はいくつですか。数字だけで答えてください。` を、context 4096・OpenCL 20 層・CPU 8 スレッドで実行し、`5` を生成した。読み込み 8.577 秒、最初の token まで 3,856.4 ms、decode 1.21 tokens/秒、生成 3 tokens。`sched_reserve` の graph splits は 2。生ログは [CLI OpenCL context 4096](../../benchmark-results/gemma3-27b-q8-opencl-cli-4096-20261012.log)。

CPU 指定の既存経路も、context 512・CPU 8 スレッドで同じ質問に `5` と回答した。[CPU CLI の確認ログ](../../benchmark-results/gemma3-27b-q8-cpu-cli-smoke-20261012.log)では graph splits は 1、生成は 3 tokens。コンテキスト長が異なるため、この確認値を上の OpenCL 速度と直接比較しない。

GUI と同じ認証付きローカル API でも `POST /internal/load` に `gemma_backend: opencl` と `devices: []` を指定し、続く `POST /v1/chat/completions` に同じ日本語入力を送った。HTTP 200 で `5` を生成し、完了理由は `stop` / `end_of_message`。モデル読込 9.22 秒、リクエストから最初の文字まで 3.95 秒、GPU 計算グラフの分割は 2。検証後は `/internal/shutdown` で一時サーバーを終了した。[再実行スクリプト](../../tools/gemma-opencl-gui-smoke.ps1)、[応答 JSON](../../benchmark-results/gemma3-27b-q8-opencl-gui-20261012.response.json)、[サーバーログ](../../benchmark-results/gemma3-27b-q8-opencl-gui-20261012.server.log)に記録した。これは GUI が使う API の検証で、実画面のクリック操作ではない。

`dotnet restore` 後の GUI Release ビルド、Gemma プロンプトのテスト 3 件、win-x64 の新規 publish が成功した。publish には `gemma-opencl` 内の 5 DLL とライセンス・説明書が含まれ、Vulkan DLL とパッチの差分ファイルは含まれない。同梱した `ggml-opencl.dll` には後述の 3 修正を反映している。

## 複数 GPU の配置問題と修正

元の OpenCL バックエンドでは GPU の空き VRAM が 0 MiB と報告され、同種の B580 2 台を同時選択した場合、重みの配置と実行デバイスが食い違い、graph split が増えて遅くなった。2 台が 1 つの OpenCL context を共有し、元の `supports_buft` は他方の GPU に属する重みバッファも利用可能と判定する。さらに、llama.cpp のモデルローダー、KV キャッシュ、再帰メモリの `ggml_backend_buft_comparator` はポインターではなく**バッファ型名**を比較する。両 GPU の型名が同じ `OpenCL` だと別デバイスのコンテキストが同じキーに畳み込まれるため、`OpenCL0` / `OpenCL1` の一意な型名が必要だった。

DXGI を使うメモリ報告パッチを別ビルドで試すと、各 B580 は 11,875 MiB total / 11,347 MiB free と列挙された。[デバイス一覧の記録](../../benchmark-results/gemma3-opencl-dxgi-devices-excerpt-20261012.txt)は成功時の出力抜粋で、完全な生ログではない。一方、重みバッファのデバイス局所性を厳格にするだけの試行では、2 台・32 層の投影メモリが GPU1 約 14.9 GiB・GPU0 約 0.18 GiB に偏り、decode の graph splits が 223 に増えた。[失敗した試行のログ](../../benchmark-results/gemma3-27b-q8-opencl-dual-locality-failed-20261012.log)も保存した。局所性の修正だけでは足りず、バッファ型名の衝突も解消する必要があった。3 つの修正と再ビルド方法は [ネイティブビルド手順](../../NNtrain.Gemma/gemma-opencl/README.md) に記録した。

### 2 台の B580 への配置と生成

上の失敗を受け、DXGI メモリ報告・デバイスごとのバッファ局所性・デバイス別の一意なバッファ型名を組み合わせた OpenCL ビルドを試した。コンテキスト 512、CPU 8 スレッドで、重みが両 GPU に配置され、計算グラフの分割も 3 に収まった。以下は別々の 1 回の生成で、32 層は 8 トークンのスモークテスト、40 層は 40 トークンの生成である。この 3 つの修正を含む `ggml-opencl.dll` を同梱版に採用した。

| GPU 層数 | OpenCL0 / OpenCL1 の重み | graph splits | Decode |
| ---: | ---: | ---: | ---: |
| 32 | 6,695.08 / 7,704.66 MiB | 3 | 7 runs、1.76 トークン/秒 |
| 40 | 8,368.85 / 9,378.43 MiB | 3 | 39 runs、2.51 トークン/秒 |

[32 層の生ログ](../../benchmark-results/gemma3-opencl-dual-unique-ngl32-smoke-20261012.log)と[40 層の生ログ](../../benchmark-results/gemma3-opencl-dual-unique-ngl40-40tokens-20261012.log)に配置、分割数、生成時間を保存した。40 層の実行後に報告された空き VRAM は OpenCL0 が 2,592 MiB、OpenCL1 が 1,212 MiB。

### 条件を揃えた 40 トークン比較

同じモデル、同じ英語の数列生成入力、コンテキスト 512、最大 40 新規トークン、CPU 8 スレッド、temperature 0 を使い、単一 B580・20 層と 2 台の B580・40 層を各 1 回測定した。両方とも prompt 25 トークン、decode 39 runs。単一 GPU は元の OpenCL ビルド、2 台は上記の 3 修正を含むビルドであり、GPU 数だけの純粋な比較ではない。

| 構成 | Prompt eval | Decode | 合計 |
| --- | ---: | ---: | ---: |
| B580 1 台、20 層 | 25 tokens / 3,217.29 ms = 7.77 tokens/秒 | 39 runs / 25,386.45 ms = 1.54 tokens/秒 | 28,626.38 ms |
| B580 2 台、40 層 | 25 tokens / 2,351.04 ms = 10.63 tokens/秒 | 39 runs / 15,517.53 ms = 2.51 tokens/秒 | 17,892.82 ms |

この 1 回ずつの比較では decode が約 **1.63 倍**だった。反復測定による分散や他の入力長での速度は未評価である。[単一 GPU の生ログ](../../benchmark-results/gemma3-opencl-single-original-ngl20-matched-40tokens-20261012.log)と[2 台の生ログ](../../benchmark-results/gemma3-opencl-dual-unique-ngl40-40tokens-20261012.log)を参照。

コンテキスト 4096 でも 2 台・40 層を読み込み、graph splits 3 で短い生成を完了した。[ネイティブのスモークテスト](../../benchmark-results/gemma3-opencl-dual-unique-ngl40-ctx4096-smoke-20261012.log)に加え、[NNtrain CLI の実行ログ](../../benchmark-results/gemma3-opencl-dual-nntrain-cli-ctx4096-fixed-20261012.log)には `--opencl-all-devices` による読み込みと日本語の問いへの応答 `5` が記録されている。CLI の生成は 3 トークンだけなので、その decode 値は上記の 40 トークン比較には使わない。

GUI と同じ認証付きローカル API でも、`POST /internal/load` から `POST /v1/chat/completions` まで 2 台の B580 で実行し、応答 `5` を確認した。40/63 層をオフロードし、OpenCL0 / OpenCL1 の重みは 8,368.85 / 9,378.43 MiB、コンテキスト 4096 の graph splits は 3。[サーバーの標準出力](../../benchmark-results/gemma3-opencl-dual-gui-smoke-20261012-final.server.log)、[ネイティブログ](../../benchmark-results/gemma3-opencl-dual-gui-smoke-20261012-final.server-error.log)、[応答 JSON](../../benchmark-results/gemma3-opencl-dual-gui-smoke-20261012-final.response.json)に記録した。これは GUI が使用する API 経路の検証で、画面上のボタン操作は未検証である。速度測定の数値はモデル、コンテキスト長、CPU スレッド数、VRAM 状態で変わる。

外部で `GGML_OPENCL_PLATFORM` / `GGML_OPENCL_DEVICE` を設定せず、Windows PowerShell 5.1 から UTF-8 の日本語リクエストを送る検証も `5` で成功した。[環境変数なしの応答](../../benchmark-results/gemma3-opencl-dual-gui-no-env-utf8-20261012.response.json)と[ネイティブログ](../../benchmark-results/gemma3-opencl-dual-gui-no-env-utf8-20261012.server-error.log)に、Intel OpenCL の自動選択と 2 台への配置が残っている。最初の PowerShell 5.1 試行では文字列 Body が日本語を `?` に変えたため、[再実行スクリプト](../../tools/gemma-opencl-gui-smoke.ps1)を UTF-8 バイト送信に修正した。

`dotnet publish -c Release -r win-x64 --self-contained false` で Windows 配布ビルドを作成した。配布先は `NNtrain.Gui/bin/Release/net10.0-windows/win-x64/publish-opencl-dual-20261012/`。同梱 `ggml-opencl.dll` の SHA-256 はソース側と同じ `723A972EC12D68E09628887BAAC1866DA46D660CAAC9C2F3F947A17AAFE636EE`。[配布版 API の応答](../../benchmark-results/gemma3-opencl-dual-published-gui-20261012.response.json)も `5` だった。

追加の Q8_0 サブグループ幅 4 候補は、単一 B580・20 層・`pp20` / `tg32` を各 2 回比較した。[基準](../../benchmark-results/gemma3-opencl-q8-sg4-baseline-20261012.jsonl)の decode 1.509 ± 0.016 に対し、[候補](../../benchmark-results/gemma3-opencl-q8-sg4-candidate-20261012.jsonl)は 1.522 ± 0.006 トークン/秒（約 +0.86%）で、差が小さいため同梱 DLL には採用していない。候補の短い生成は既存ログと先頭 39 文字が一致したが、ネイティブ CLI が標準入力 EOF で終了したため、全出力の一致は確認できていない。
