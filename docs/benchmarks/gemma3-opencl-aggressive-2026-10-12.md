# Gemma 3 27B Q8_0: 変換・GEMM・GPU配置の最適化

2026-10-12。NNtrain のローカル OpenCL 経路に、Q8_0 の XMX GEMM、複数 GPU のプログラム再利用、B580 2 台向けの 43 層・53:47 配置を組み込んだ。

## 採用構成と測定結果

同じ実行ファイル・CPU 側 DLL・OpenCL DLLで、XMX の有効/無効と GPU 層数を切り替えて比較した。GPU 比率は両条件とも 53:47。基準は比較用の 40 層・従来 F32 GEMM であり、以前の GUI の自動比率とは異なる。

各条件 2 回、値はトークン/秒の平均 ± 標準偏差。短い測定であり、全入力に同じ改善率を保証する値ではない。

| 処理 | 40 層・従来 GEMM | 43 層・M16×N32 XMX | 改善率 |
| --- | ---: | ---: | ---: |
| Prefill 32 | 14.98 ± 0.63 | 17.45 ± 2.05 | +16.5% |
| Prefill 64 | 18.56 ± 0.61 | 20.93 ± 0.43 | +12.7% |
| Prefill 128 | 19.01 ± 0.94 | 21.52 ± 1.59 | +13.2% |
| Prefill 512 | 20.092 ± 0.006 | 22.986 ± 0.200 | +14.4% |
| Decode 32 | 2.5387 ± 0.0023 | 2.8151 ± 0.0032 | +10.9% |

[基準の生ログ](../../benchmark-results/gemma3-opencl-m16n32-reference-20261012.log)、[採用候補の生ログ](../../benchmark-results/gemma3-opencl-m16n32-candidate-20261012.log)、[全サンプルと比較 JSON](../../benchmark-results/gemma3-opencl-aggressive-summary-20261012.json)を保存した。

## 測定条件

- モデル: `models/huggingface/ggml-org/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q8_0.gguf`。
- 元 GGUF は 28,707,604,864 バイト、SHA-256 `794d14393b9a15de7f08da16dc35c503edb3eb48dd58fa5beb37b7cf2a188379`。元モデルの再量子化は行っていない。
- GPU: Intel Arc B580 2 台、OpenCL 3.0 NEO、ドライバー 32.0.101.9030。
- CPU: AMD Ryzen 7 5700X、8 スレッド。native ABI は llama.cpp `3f7c29d318e317b63f54c558bc69803963d7d88c`、LLamaSharp 0.27.0。
- `llama-bench -ngl 40,43 -ts 53/47 -p 32,64,128,512 -n 32 -r 2 -t 8 -o jsonl`。batch 2048、ubatch 512、KV F16、flash attention 無効、mmap 有効。context は各ベンチケースの自動設定。
- `GGML_OPENCL_PLATFORM=Intel`、device 環境変数なし。program cache は両条件とも有効。基準は `GGML_OPENCL_Q8_XMX_F16=0`、候補は同変数と `GGML_OPENCL_Q8_XMX_F16_M16N32` を `1` にした。
- 同じ `llama-bench.exe` と core DLL を使い、両条件の `ggml-opencl.dll` も同じ SHA-256 `AB682510EC1877AE47D05BDEB6C9C38E4AFE49CFC4B149BF2B69D9E9AF4F1865`。測定中は他のモデル実行・CPU ビルドを重ねていない。

512 トークンは prefill の実入力長。別途 GUI/CLI で context 4096 の割り当てと短文生成を検証した。4096/8192 トークンで埋まった context 全体の速度はこの測定に含まない。

## GEMM を書き直した効果

既存の Q8_0 SoA（量子値とスケールの別配列）を維持した。GPU へ全重みの F16 コピーを作らず、必要なパネルだけをレジスタ内で展開する。

新しい GEMM は Intel subgroup16 の行列命令を使い、16 行×32 列を処理する。8 行の A パネル 2 組と 16 列の B パネル 2 組を共有し、K16 の行列積 4 回へ供給する。重みと F32 活性値を F16 に丸め、累積は F32。Q8_0 の変換・復元は、32 回のバイトコピーを 2 回の `vload16/vstore16` に置き換えた。

層数を揃えた GEMM だけの比較は以下。decode の行列ベクトル積は変更しておらず、decode の主な改善は次の GPU 配置による。

| GPU 層数 | Prefill | 従来 GEMM | M16×N32 | 平均の差 |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 32 | 14.98 ± 0.63 | 16.36 ± 1.69 | +9.2% |
| 40 | 64 | 18.56 ± 0.61 | 19.14 ± 0.15 | +3.1% |
| 40 | 128 | 19.01 ± 0.94 | 19.88 ± 0.12 | +4.5% |
| 40 | 512 | 20.092 ± 0.006 | 21.063 ± 0.068 | +4.8% |
| 43 | 32 | 15.97 ± 0.58 | 17.45 ± 2.05 | +9.3% |
| 43 | 64 | 19.36 ± 0.39 | 20.93 ± 0.43 | +8.1% |
| 43 | 128 | 21.44 ± 0.10 | 21.52 ± 1.59 | +0.4% |
| 43 | 512 | 22.53 ± 0.33 | 22.99 ± 0.20 | +2.0% |

32 トークンや 43 層の 128 トークンはばらつきが大きい。40 層・512 トークンの差は今回の 2 サンプルで安定した。初期の M8×N16 は大きめの prefill に改善が確認できなかったため、共有範囲を広げた M16×N32 を採用した。

## CPU に残る層を減らす

最後の出力層で GPU1 側のモデルバッファが約 1 GiB 多くなるため、53:47 に割り当てて 43 層を常駐させた。従来 GEMM のまま層数だけを変えた独立試験では、40 層の decode が 2.5269 ± 0.0029、43 層が 2.8001 ± 0.0111 トークン/秒で、約 10.8% 改善した。[独立した配置比較](../../benchmark-results/gemma3-opencl-balanced40-43-20261012.log)。

context 4096 の CLI は `5` を生成し、graph splits は 3。[CLI の記録](../../benchmark-results/gemma3-opencl-balanced43-ctx4096-smoke-20261012.log)。論理バッファ割り当ては以下だった。

| バッファ | OpenCL0 | OpenCL1 |
| --- | ---: | ---: |
| モデル | 9,624.17 MiB | 9,378.43 MiB |
| KV | 608.00 MiB | 512.00 MiB |
| 計算 | 155.02 MiB | 522.50 MiB |
| 合計 | 10,387.19 MiB | 10,412.93 MiB |

GUI は総 VRAM 11 GiB 以上の B580 がちょうど 2 台の構成に、この 43 層・53:47・context 4096 の設定を使う。CLI には `--tensor-split 53,47` を追加した。単一 B580 の 20 層設定は既存の測定値を使う。27B Q8_0 全体と実行バッファは 2 台の VRAM に収まらず、採用構成でも 20 層と入力埋め込みは CPU に残る。

## 初期化の重複コンパイルを省く

両 GPU は OpenCL context を共有し、`clBuildProgram(..., 0, NULL, ...)` が context の全デバイスをビルドする。2 枚目の初期化で同じ source/options をビルドし直していたため、完全一致のプログラムを再利用する。各 backend の kernel handle は別々に作る。

同一 DLL/core で、モデルを読まない `llama-bench --list-devices` を ON/OFF 各 1 回測った。

| 初期化 | 秒 |
| --- | ---: |
| 再利用なし | 253.757 |
| 再利用あり | 127.924 |

約 49.6% 短縮、約 1.98 倍。ON では 117 miss/117 hit。これは 1 回ずつの初期化測定であり、反復統計ではない。最終版は全デバイスの初期化後に追加のプログラム参照 117 個を解放するため、キャッシュが旧 context を保持し続けない。

[再利用 OFF のログ](../../benchmark-results/gemma3-opencl-initialization-20261012/program-cache-jit-off.log)、[ON のログ](../../benchmark-results/gemma3-opencl-initialization-20261012/program-cache-jit-on.log)。時間はドライバーのキャッシュ状態や実行環境で変わる。これらは寿命改善前の同一 DLL による比較で、最終版でも同じ 117 hit と初期化後の解放を確認した。

## 数値検証と採用しなかった経路

M16×N32 の Q8_0×F32 を CPU と照合し、両 B580 で各 5 件、計 10 件が合格した。

| M | N | K |
| ---: | ---: | ---: |
| 97 | 33 | 256 |
| 512 | 64 | 512 |
| 5,376 | 32 | 5,376 |
| 21,504 | 32 | 5,376 |
| 5,376 | 33 | 21,504 |

`test-backend-ops` の MUL_MAT 許容 NMSE は 5×10⁻⁴。実際の NMSE 値は OK 時に表示されないので、それ以下とだけ確認できる。[GPU 数値検証ログ](../../benchmark-results/gemma3-opencl-xmx-m16n32-real-k-numeric-20261012.log)、[再現用テストケースのパッチ](../../NNtrain.Gemma/gemma-opencl/patches/ggml-opencl-q8-xmx-numeric-cases.patch)。F16 丸めにより、すべての入力で F32 経路と同じ logits/文字列になる保証はない。

CPU で Q8_0 を SoA に再配置して直接転送する案は、短文の出力が一致したが、重みロードが 10.76→33.08 秒へ悪化した。decode の 8 行タイル案は 2.5364 ± 0.0006 に対し基準が 2.5193 ± 0.0247 トークン/秒で、約 +0.68% に留まったため採用していない。[不採用候補のログ](../../benchmark-results/gemma3-opencl-q8-gemv-r8-20261012/)。

初回の M16×N32 数値試験は Adreno 設定により Intel GPU が拒否され、CPU がスキップされた。これは合格として扱わず、Adreno を OFF にした上記 DLL で再実行した。

DXGI budget 由来の free と論理バッファ自己使用量は別系統であり、単発の初期化候補試行の memory-breakdown 表示で unsigned underflow が起きた。上のメモリ表は論理割り当て値で、DXGI free と合計が一致する保証はない。

## 同梱ファイル

採用 DLL: `NNtrain.Gemma/gemma-opencl/ggml-opencl.dll`、889,856 バイト。

SHA-256: `AB682510EC1877AE47D05BDEB6C9C38E4AFE49CFC4B149BF2B69D9E9AF4F1865`。

再ビルド手順と適用順は [native README](../../NNtrain.Gemma/gemma-opencl/README.md)。GUI/CLI の複数 GPU 経路は XMX・M16×N32・program cache を既定で有効にし、明示した環境変数 `0` は尊重する。

## 配布版の検証

CLI/GUI の Release ビルドは警告・エラーなしで成功し、Gemma プロンプトの既存テスト 3 件も合格した。[テスト結果](../../benchmark-results/test-results/gemma3-opencl-xmx-prompt-20261012.trx)。

`dotnet publish -c Release -r win-x64 --self-contained false` の出力先は `NNtrain.Gui/bin/Release/net10.0-windows/win-x64/publish-opencl-xmx-20261012/`。ソース、CLI 出力、配布版の OpenCL DLL は上記 SHA-256 で一致した。

配布版で GUI と同じ認証付きローカル API を使い、`POST /internal/load` から `POST /v1/chat/completions` を検証した。`2足す3はいくつですか。数字だけで答えてください。` に `5` を返し、23 prompt tokens / 3 completion tokens、正常 stop だった。実画面のクリック操作による検証ではない。テスト用プロセスは終了済み。

[応答 JSON](../../benchmark-results/gemma3-opencl-xmx-published-20261012.response.json)、[native ログ](../../benchmark-results/gemma3-opencl-xmx-published-20261012.server-error.log)で、両 B580 の M16×N32 XMX 有効、43/63 層、graph splits 3、117 program hit と初期化後の 117 参照解放を確認した。計算はプロセス内 OpenCL で行う。

通常のサンドボックスでは publish がログなしで終了し、API 検証も localhost のソケット制限で拒否されたため、許可された通常環境で再実行して成功を確認した。
