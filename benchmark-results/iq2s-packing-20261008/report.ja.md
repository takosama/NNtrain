# IQ2_S 圧縮配置・展開・格納の高速化

## 結果

対象: Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf、Intel Arc B580 12GB × 2。
前回採用したprefill・分割出力ヘッドを両条件で有効にし、今回の差は `InferencePackedIq2Decode` だけ。
GUIの標準推論で新経路を有効化した。Core APIでは明示オプションとして提供する。

| 条件 | 従来 tok/s | 新経路 tok/s | 改善 | 従来 平均±母SD | 新経路 平均±母SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 通常生成（greedy） | 17.783 | 19.759 | +11.11% | 17.781 ± 0.010 | 19.750 ± 0.024 |
| GUI相当サンプリング | 17.640 | 19.545 | +10.80% | 17.641 ± 0.004 | 19.544 ± 0.004 |
| 実LoRA適用・greedy | 16.665 | 17.480 | +4.89% | 16.663 ± 0.004 | 17.444 ± 0.073 |

各条件は16 tokenの同じpromptから128 tokenを4回生成し、先頭1回を除いた3回で集計。
decode速度は最初から最後の生成tokenまでの127区間で計算し、モデル読込・prefillを除外。
サンプリングはtemperature 0.6 / top-p 0.95 / top-k 20 / seed 5429。
LoRA条件は既存の `lora_rintya_qa142_Qwen3.8-27B-Uncensored-noMTP-IQ2_M_2epoch.gguf` を使用。
全4回の生成IDと、各条件16 snapshot × 全248,320語彙のFP32 logitsが従来と全バイト一致。
全128ステップのlogitsを保存したという意味ではない。GPUプロファイルを無効にして計測した。

## 圧縮・パッキング・展開・格納

- **82 bytes / 256 weightsを維持**。再量子化も重み精度の変更も行わない。
- grid下位indexとsignをushortへ、上位indexとscaleを共有byteへ再配置する。
- モデル読込時にCPUで一度だけ変換し、GPUへ1コピーだけ転送・常駐させる。
- decodeは整列したushortと共有byteから展開し、FP32 FMA/reduction順を維持する。
- rows2/4/8/16、BSLM/K32rの長文prefill、fused LoRAも新配置から直接計算する。
- 学習、embedding、出力ヘッド、他の量子化形式、未対応デバイス、非既定の非BSLM XMXは従来経路。
- 生成ごとのデバイス/形式判定は読込時に確定し、hot pathから除いた。

モデル読込直後のlive GPU allocationは新旧ともGPU0 `5,559,294,976` bytes、
GPU1 `5,202,195,456` bytes。今回の配置変更による追加常駐VRAMは0 bytes。
CPU変換はロード時だけのコスト。ロード時間はJSONに分離記録し、キャッシュ・コンパイル条件の影響があるためdecode倍率に含めない。

## 行列単体

| 実モデルの重み | K×N | 従来 ms | 新経路 ms | 倍率 |
| --- | ---: | ---: | ---: | ---: |
| blk.0.attn_gate.weight | 5120×6144 | 0.055931 | 0.048438 | 1.155× |
| blk.0.ffn_gate.weight | 5120×17408 | 0.145850 | 0.123537 | 1.181× |
| blk.8.ffn_down.weight | 17408×5120 | 0.137531 | 0.117481 | 1.171× |

1 token投影、3 warmup、9 sample、各sample16 dispatch。順序を交互に反転。
全出力・末尾guard・独立CPU基準点を検証。元GGUF全バイトの復元とGPU/CPU packing一致も検証。
別候補の128KiB符号lookup tableは実測で遅く、production経路には採用していない。

## 長文入力

| prefix token数 | 従来 中央値 | 新経路 中央値 | 全語彙logits |
| --- | ---: | ---: | --- |
| 4,096 | 30.837秒 | 30.825秒 | 完全一致 |
| 8,192 | 108.868秒 | 108.998秒 | 完全一致 |

各1 warmup + 3 sample。reset、prefix処理、次の1 tokenの全logits取得を含む。
8,192 tokenでは約0.12%遅く、長文prefillの高速化は今回確認できなかった。
長文データはGPUカーネルが同一の導入検証ビルドで採取し、最終版ではCPU側のdecode経路選択を読込時に移した。
計測バイナリSHAは各JSONとprovenanceに記録。512 tokenの別profileでpacked K32rの実行も確認した。

## 検証と証拠

- 公開対象だけをGit indexから独立ディレクトリへ取り出し、GUI・benchmarkをReleaseビルド。
- 最終GPU/CPUテスト: **55 passed / 0 failed / 0 skipped**。
- 全65,536種類のhalf bit表現、packing往復、odd tail、WG32/64/128、native-half設定、LoRA、prefill範囲外fallbackを検証。
- 小型実モデルでrouting、学習のloss/gradient一致、非BSLM fallbackのVRAM peak一致を検証。
- 元GGUF SHA-256: `676db8ccd38035117e705858fd591964005cb0a921b39255b0691d8339a8dfe2`。

[通常生成](greedy-report/report.decode.ja.md) / [サンプリング](sampling-report/report.decode.ja.md) / [LoRA](lora-report/report.decode.ja.md)

[行列測定](micro-release/results.json) / [長文測定](prefill-long/model-results.json) / [経路profile](prefill-profile/model-results.json)

[テスト結果](tests/packed-iq2-release.trx) / [ソース・バイナリ対応](source-provenance.json) / [再現手順](../../tools/iq2s-packed-decode-bench/README.md)

今回の倍率は上記モデル・GPU・ドライバ・入力・設定での実測。学習全体の速度や他デバイスの倍率には外挿しない。
GGUF、LoRA重み、実行バイナリ、logitsバイナリはコミット対象に含めず、ソースと検証メタデータ・統計を保存する。
