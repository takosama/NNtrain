# IQ2_S 本番演算経路の比較

## 測定の範囲

FP32の入力とGGUFのIQ2_S重みを使う投影演算を比較した。入力をIQ2_Sへ量子化する測定ではない。
主指標はhost wall時間の中央値。各経路に必要なpack、作業バッファの確保・再利用、GPU同期を含む。
初期入力転送、モデル読込、コンパイル、検証用の出力読戻しは投影演算の時間から除外した。逆伝播のGPU内コピーはhost wallに含み、GPU列のカーネル時間合計には含まれない。
倍率は旧時間÷新時間で、1倍を超えれば高速化。GPU列はイベント時間の合計であり、host wallとは別の指標。

候補が複数あるケースは、host wall中央値が最小だった候補を各行に明記した。これは測定内の候補選択であり、製品の既定値を決定したという意味ではない。

以下の投影演算の倍率を、LoRA学習全体やモデル全体の倍率として扱わない。

## 組込み時の経路選択

推論は既存BSLMの演算順を維持し、入力high/residualの配置をrow-majorに変更する。学習は別の16×32経路を使用する。以下は各候補を直接呼んだマイクロベンチ。

- 推論：M≥512で `q35l_prefill_xmx_iq2_s_gguf_bslm_k32r` とrow-major packを使用。M<512では旧IQ2経路を維持。比較した候補は計測前後の全出力で旧BSLMとbit一致を必須とした。
- 学習forward・逆伝播dX：M≥128で16×32新経路の対象。
- 全モデルの候補はbatched text attentionも有効化。256 tokenでもattentionは変更されるため、そのモデル速度差をIQ2単独の効果とは解釈しない。
- メモリ・デバイス対応・resident panel等による既存経路への分岐は残る。表は形状条件を示す。
- Mは実際に投影する行数。GUIのprefix chunkが1024なら、8192 tokenのpromptでも各chunkのMは最大1024となる。

## 推論 forward：旧GGUF BSLMとの比較

| テンソル / M×K→N | 新候補 | 旧wall ms | 新wall ms | wall倍率 | 旧GPU ms | 新GPU ms | 新wall 平均±σ (最小–最大) ms | 本番の形状判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blk.0.attn_gate.weight / 256×5120→6144 | inference-bslm-k32r | 0.828 | 0.820 | 1.010× | 0.781 | 0.774 | 0.825 ± 0.011 (0.815–0.841) | 旧IQ2経路を維持 |
| blk.0.attn_gate.weight / 512×5120→6144 | inference-bslm-k32r | 1.511 | 1.470 | 1.028× | 1.446 | 1.409 | 1.488 ± 0.036 (1.456–1.575) | row-major BSLMの対象 |
| blk.0.attn_gate.weight / 1024×5120→6144 | inference-bslm-k32r | 3.011 | 2.847 | 1.058× | 2.962 | 2.807 | 2.851 ± 0.013 (2.831–2.877) | row-major BSLMの対象 |
| blk.0.attn_gate.weight / 2048×5120→6144 | inference-bslm-k32r | 6.173 | 5.628 | 1.097× | 6.097 | 5.566 | 5.613 ± 0.027 (5.556–5.643) | row-major BSLMの対象 |
| blk.0.attn_gate.weight / 4096×5120→6144 | inference-bslm-k32r | 11.592 | 10.896 | 1.064× | 11.509 | 10.850 | 10.901 ± 0.024 (10.868–10.956) | row-major BSLMの対象 |
| blk.0.attn_gate.weight / 8192×5120→6144 | inference-bslm-k32r | 24.752 | 21.483 | 1.152× | 24.588 | 21.429 | 21.417 ± 0.175 (21.060–21.586) | row-major BSLMの対象 |
| blk.0.ffn_gate.weight / 256×5120→17408 | inference-bslm-k32r | 1.909 | 1.956 | 0.976× | 1.868 | 1.914 | 1.976 ± 0.033 (1.939–2.029) | 旧IQ2経路を維持 |
| blk.0.ffn_gate.weight / 512×5120→17408 | inference-bslm-k32r | 4.183 | 3.979 | 1.051× | 4.136 | 3.934 | 3.979 ± 0.007 (3.969–3.989) | row-major BSLMの対象 |
| blk.0.ffn_gate.weight / 1024×5120→17408 | inference-bslm-k32r | 8.856 | 7.932 | 1.116× | 8.802 | 7.883 | 7.945 ± 0.034 (7.905–8.014) | row-major BSLMの対象 |
| blk.0.ffn_gate.weight / 2048×5120→17408 | inference-bslm-k32r | 18.425 | 15.525 | 1.187× | 18.346 | 15.472 | 15.523 ± 0.023 (15.488–15.556) | row-major BSLMの対象 |
| blk.0.ffn_gate.weight / 4096×5120→17408 | inference-bslm-k32r | 36.807 | 30.730 | 1.198× | 36.686 | 30.670 | 30.733 ± 0.022 (30.698–30.776) | row-major BSLMの対象 |
| blk.0.ffn_gate.weight / 8192×5120→17408 | inference-bslm-k32r | 74.149 | 61.483 | 1.206× | 73.996 | 61.357 | 61.499 ± 0.048 (61.444–61.575) | row-major BSLMの対象 |
| blk.8.ffn_down.weight / 256×17408→5120 | inference-bslm-k32r | 2.326 | 2.239 | 1.039× | 2.272 | 2.196 | 2.242 ± 0.010 (2.233–2.269) | 旧IQ2経路を維持 |
| blk.8.ffn_down.weight / 512×17408→5120 | inference-bslm-k32r | 4.992 | 4.404 | 1.133× | 4.936 | 4.353 | 4.414 ± 0.025 (4.392–4.479) | row-major BSLMの対象 |
| blk.8.ffn_down.weight / 1024×17408→5120 | inference-bslm-k32r | 10.526 | 8.719 | 1.207× | 10.460 | 8.672 | 8.730 ± 0.021 (8.709–8.777) | row-major BSLMの対象 |
| blk.8.ffn_down.weight / 2048×17408→5120 | inference-bslm-k32r | 21.447 | 17.261 | 1.242× | 21.286 | 17.200 | 17.264 ± 0.013 (17.250–17.286) | row-major BSLMの対象 |
| blk.8.ffn_down.weight / 4096×17408→5120 | inference-bslm-k32r | 42.399 | 34.641 | 1.224× | 42.311 | 34.571 | 34.656 ± 0.032 (34.618–34.706) | row-major BSLMの対象 |
| blk.8.ffn_down.weight / 8192×17408→5120 | inference-bslm-k32r | 90.967 | 77.334 | 1.176× | 85.722 | 69.107 | 76.131 ± 1.811 (73.539–78.486) | row-major BSLMの対象 |

旧経路を含む全候補の平均・母標準偏差・最小・最大と全サンプルはsummary.jsonに保存。

## 学習 forward：旧FP32 rows4との比較

| テンソル / M×K→N | 新候補 | 旧wall ms | 新wall ms | wall倍率 | 旧GPU ms | 新GPU ms | 新wall 平均±σ (最小–最大) ms | 本番の形状判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blk.0.attn_gate.weight / 256×5120→6144 | tiled-16x32-panel0 | 7.732 | 1.228 | 6.296× | 7.707 | 1.157 | 1.241 ± 0.021 (1.224–1.286) | 16×32新経路の対象 |
| blk.0.attn_gate.weight / 512×5120→6144 | tiled-16x32-panel0 | 15.417 | 1.882 | 8.193× | 15.369 | 1.774 | 1.895 ± 0.035 (1.851–1.982) | 16×32新経路の対象 |
| blk.0.attn_gate.weight / 1024×5120→6144 | tiled-16x32-panel0 | 30.821 | 3.324 | 9.272× | 30.793 | 3.243 | 3.346 ± 0.063 (3.296–3.511) | 16×32新経路の対象 |
| blk.0.attn_gate.weight / 2048×5120→6144 | tiled-16x32-panel0 | 61.613 | 6.348 | 9.706× | 61.561 | 6.232 | 6.360 ± 0.070 (6.280–6.521) | 16×32新経路の対象 |
| blk.0.attn_gate.weight / 4096×5120→6144 | tiled-16x32-panel0 | 123.183 | 12.127 | 10.158× | 123.141 | 11.984 | 12.099 ± 0.049 (12.034–12.163) | 16×32新経路の対象 |
| blk.0.attn_gate.weight / 8192×5120→6144 | tiled-16x32-panel0 | 246.455 | 23.535 | 10.472× | 246.400 | 23.360 | 23.495 ± 0.064 (23.408–23.560) | 16×32新経路の対象 |
| blk.0.ffn_gate.weight / 256×5120→17408 | tiled-16x32-panel0 | 18.684 | 3.149 | 5.934× | 18.663 | 3.080 | 3.171 ± 0.046 (3.132–3.252) | 16×32新経路の対象 |
| blk.0.ffn_gate.weight / 512×5120→17408 | tiled-16x32-panel0 | 37.347 | 5.377 | 6.946× | 37.294 | 5.286 | 5.367 ± 0.023 (5.323–5.402) | 16×32新経路の対象 |
| blk.0.ffn_gate.weight / 1024×5120→17408 | tiled-16x32-panel0 | 74.591 | 9.233 | 8.079× | 74.543 | 9.124 | 9.239 ± 0.054 (9.188–9.375) | 16×32新経路の対象 |
| blk.0.ffn_gate.weight / 2048×5120→17408 | tiled-16x32-panel0 | 149.041 | 17.025 | 8.754× | 148.988 | 16.862 | 17.007 ± 0.048 (16.922–17.057) | 16×32新経路の対象 |
| blk.0.ffn_gate.weight / 4096×5120→17408 | tiled-16x32-panel0 | 298.153 | 32.804 | 9.089× | 298.024 | 32.630 | 32.782 ± 0.061 (32.686–32.863) | 16×32新経路の対象 |
| blk.0.ffn_gate.weight / 8192×5120→17408 | tiled-16x32-panel0 | 596.130 | 64.257 | 9.277× | 595.974 | 64.063 | 64.255 ± 0.075 (64.141–64.387) | 16×32新経路の対象 |
| blk.8.ffn_down.weight / 256×17408→5120 | tiled-16x32-panel0 | 32.075 | 3.476 | 9.227× | 32.038 | 3.388 | 3.506 ± 0.056 (3.449–3.604) | 16×32新経路の対象 |
| blk.8.ffn_down.weight / 512×17408→5120 | tiled-16x32-panel0 | 64.051 | 6.115 | 10.475× | 64.005 | 6.016 | 6.095 ± 0.094 (5.889–6.204) | 16×32新経路の対象 |
| blk.8.ffn_down.weight / 1024×17408→5120 | tiled-16x32-panel0 | 128.113 | 9.440 | 13.572× | 128.069 | 9.312 | 9.450 ± 0.064 (9.356–9.532) | 16×32新経路の対象 |
| blk.8.ffn_down.weight / 2048×17408→5120 | tiled-16x32-panel0 | 256.180 | 17.005 | 15.065× | 256.121 | 16.802 | 16.975 ± 0.078 (16.854–17.055) | 16×32新経路の対象 |
| blk.8.ffn_down.weight / 4096×17408→5120 | tiled-16x32-panel0 | 512.067 | 34.032 | 15.046× | 511.964 | 33.911 | 33.987 ± 0.464 (33.225–34.894) | 16×32新経路の対象 |
| blk.8.ffn_down.weight / 8192×17408→5120 | tiled-16x32-panel0 | 1024.258 | 75.492 | 13.568× | 1024.069 | 67.915 | 75.390 ± 0.718 (74.358–76.773) | 16×32新経路の対象 |

旧経路を含む全候補の平均・母標準偏差・最小・最大と全サンプルはsummary.jsonに保存。

## 逆伝播 dX：旧vec8 / rows8 / reduceとの比較

| テンソル / M×K→N | 新候補 | 旧wall ms | 新wall ms | wall倍率 | 旧GPU ms | 新GPU ms | 新wall 平均±σ (最小–最大) ms | 本番の形状判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blk.0.attn_gate.weight / 256×5120→6144 | tiled-transpose-16x32-panel0 | 4.976 | 2.488 | 2.000× | 4.907 | 2.396 | 2.531 ± 0.085 (2.420–2.674) | 16×32新経路の対象 |
| blk.0.attn_gate.weight / 512×5120→6144 | tiled-transpose-16x32-panel0 | 9.625 | 3.530 | 2.727× | 9.581 | 3.423 | 3.518 ± 0.083 (3.404–3.635) | 16×32新経路の対象 |
| blk.0.attn_gate.weight / 1024×5120→6144 | tiled-transpose-16x32-panel0 | 18.955 | 5.550 | 3.416× | 18.876 | 5.419 | 5.541 ± 0.070 (5.421–5.665) | 16×32新経路の対象 |
| blk.0.attn_gate.weight / 2048×5120→6144 | tiled-transpose-16x32-panel0 | 37.770 | 9.483 | 3.983× | 37.698 | 9.393 | 9.516 ± 0.087 (9.401–9.703) | 16×32新経路の対象 |
| blk.0.attn_gate.weight / 4096×5120→6144 | tiled-transpose-16x32-panel0 | 85.552 | 19.073 | 4.486× | 75.337 | 16.726 | 18.451 ± 1.583 (16.687–21.044) | 16×32新経路の対象 |
| blk.0.attn_gate.weight / 8192×5120→6144 | tiled-transpose-16x32-panel0 | 157.906 | 34.926 | 4.521× | 151.103 | 32.298 | 33.950 ± 1.489 (32.052–35.651) | 16×32新経路の対象 |
| blk.0.ffn_gate.weight / 256×5120→17408 | tiled-transpose-16x32-panel0 | 14.831 | 8.222 | 1.804× | 14.792 | 8.089 | 8.234 ± 0.044 (8.189–8.339) | 16×32新経路の対象 |
| blk.0.ffn_gate.weight / 512×5120→17408 | tiled-transpose-16x32-panel0 | 29.596 | 10.051 | 2.944× | 29.510 | 9.908 | 10.054 ± 0.046 (9.994–10.140) | 16×32新経路の対象 |
| blk.0.ffn_gate.weight / 1024×5120→17408 | tiled-transpose-16x32-panel0 | 60.744 | 19.539 | 3.109× | 59.019 | 19.298 | 19.906 ± 0.578 (19.424–21.197) | 16×32新経路の対象 |
| blk.0.ffn_gate.weight / 2048×5120→17408 | tiled-transpose-16x32-panel0 | 123.455 | 24.127 | 5.117× | 117.987 | 22.281 | 23.638 ± 1.167 (22.306–24.983) | 16×32新経路の対象 |
| blk.0.ffn_gate.weight / 4096×5120→17408 | tiled-transpose-16x32-panel0 | 242.354 | 68.481 | 3.539× | 235.929 | 64.906 | 68.062 ± 2.109 (63.676–71.187) | 16×32新経路の対象 |
| blk.0.ffn_gate.weight / 8192×5120→17408 | tiled-transpose-16x32-panel0 | 480.100 | 129.072 | 3.720× | 471.797 | 122.613 | 128.437 ± 2.460 (124.494–132.001) | 16×32新経路の対象 |
| blk.8.ffn_down.weight / 256×17408→5120 | tiled-transpose-16x32-panel0 | 15.955 | 6.628 | 2.407× | 15.913 | 6.515 | 6.654 ± 0.149 (6.403–6.963) | 16×32新経路の対象 |
| blk.8.ffn_down.weight / 512×17408→5120 | tiled-transpose-16x32-panel0 | 31.827 | 9.263 | 3.436× | 31.728 | 9.088 | 9.259 ± 0.119 (9.066–9.423) | 16×32新経路の対象 |
| blk.8.ffn_down.weight / 1024×17408→5120 | tiled-transpose-16x32-panel0 | 63.951 | 14.822 | 4.315× | 63.423 | 14.656 | 14.828 ± 0.116 (14.659–15.034) | 16×32新経路の対象 |
| blk.8.ffn_down.weight / 2048×17408→5120 | tiled-transpose-16x32-panel0 | 132.123 | 26.537 | 4.979× | 126.773 | 24.772 | 25.976 ± 1.143 (24.559–27.622) | 16×32新経路の対象 |
| blk.8.ffn_down.weight / 4096×17408→5120 | tiled-transpose-16x32-panel0 | 256.722 | 46.175 | 5.560× | 253.608 | 45.266 | 45.890 ± 0.514 (45.201–46.623) | 16×32新経路の対象 |
| blk.8.ffn_down.weight / 8192×17408→5120 | tiled-transpose-16x32-panel0 | 515.046 | 87.947 | 5.856× | 507.155 | 85.862 | 87.428 ± 1.462 (85.612–90.317) | 16×32新経路の対象 |

旧経路を含む全候補の平均・母標準偏差・最小・最大と全サンプルはsummary.jsonに保存。

## 全モデルでの確認

GUIの推論設定に合わせ、LoRAなし・画像なしで、旧モデルと新モデルを順にロードした。各回Reset後にprefixを処理し、次の1 tokenの全語彙logitsを取得。
prefixチェックポイントの保存とlogits読戻しをhost wallに含む。モデルロード・tokenizeは除外。旧→新の順で測定しているため、時間変化の影響は残る。実際の画面上のTTFTを直接測った値ではない。

モデル測定：GPU 0: Intel(R) Arc(TM) B580 Graphics; GPU 1: Intel(R) Arc(TM) B580 Graphics。投影マイクロベンチとGPU構成が異なる場合、それぞれの旧/新同条件の比較として扱う。
この全モデル比較の変更範囲はIQ2 row-major BSLMとbatched text attention。投影表のIQ2単独の倍率とは別の測定。

主測定ではカーネルprofilingを無効にしてhost wallを測定した。GPU合計時間は未測定のため「—」。同じバイナリ・モデル・設定の別profile測定で使用カーネルと数値一致を確認した。

| prefix token (+次1) | 旧wall ms | 新wall ms | wall倍率 | 旧GPU合計 ms | 新GPU合計 ms | logits相対L2 | Top1一致 | 実行経路 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 1254.561 | 1017.854 | 1.233× | — | — | 0.000e+00 | はい | 旧IQ2 + batched text attention（別profileで確認） |
| 512 | 2594.026 | 1979.134 | 1.311× | — | — | 0.000e+00 | はい | IQ2 row-major + batched text attention（別profileで確認） |
| 1024 | 5946.477 | 4355.098 | 1.365× | — | — | 0.000e+00 | はい | IQ2 row-major + batched text attention（別profileで確認） |
| 2048 | 16869.854 | 10718.596 | 1.574× | — | — | 0.000e+00 | はい | IQ2 row-major + batched text attention（別profileで確認） |
| 4096 | 67152.808 | 30807.387 | 2.180× | — | — | 0.000e+00 | はい | IQ2 row-major + batched text attention（別profileで確認） |
| 8192 | 272696.339 | 108824.951 | 2.506× | — | — | 0.000e+00 | はい | IQ2 row-major + batched text attention（別profileで確認） |

全logitsファイルのSHA-256・長さ・有限値を再検査し、誤差とTop1を再計算した。複数GPUのイベント合計は実経過時間とは異なる。
256 tokenでIQ2が旧経路でもattentionは新経路になる。速度差にはattentionの効果と別セッション間の変動を含む。

### Host wallの区間別中央値

追加同期を入れずに連続するhost区間を記録。各回の3区間の和はwallと一致するが、各区間の中央値の和はwall中央値と必ずしも一致しない。

| prefix token | 経路 | Reset ms | PrimePrefix ms | LastToken ms |
| --- | --- | --- | --- | --- |
| 256 | original | 0.666 | 1194.867 | 59.127 |
| 256 | candidate | 0.627 | 958.236 | 58.939 |
| 512 | original | 0.565 | 2532.858 | 60.604 |
| 512 | candidate | 0.571 | 1917.705 | 60.449 |
| 1024 | original | 0.596 | 5881.073 | 64.134 |
| 1024 | candidate | 0.601 | 4290.169 | 64.319 |
| 2048 | original | 0.623 | 16795.822 | 73.699 |
| 2048 | candidate | 0.586 | 10643.987 | 73.755 |
| 4096 | original | 0.653 | 67062.325 | 89.878 |
| 4096 | candidate | 0.612 | 30716.386 | 90.256 |
| 8192 | original | 0.701 | 272573.372 | 122.539 |
| 8192 | candidate | 0.668 | 108700.878 | 123.763 |

### 別profileによる経路監査

この監査は使用カーネルと全logitsの確認用。サンプル数が少ない場合もあり、速度倍率の主測定には使用しない。

| prefix token | IQ2候補実行 | attention候補実行 | logits相対L2 | Top1一致 |
| --- | --- | --- | --- | --- |
| 1024 | はい | はい | 0.000e+00 | はい |

## 数値検証

計測前後の全出力について有限値・ガード領域・旧経路との相対L2を検査し、独立したCPU double内積との誤差をサンプル点で確認した。以下は新候補の前後2回の最大値。

| ケース | CPU点数 | CPU最大絶対誤差 | CPU相対L2 | 旧参照との全出力相対L2 | CPU許容比 ≤1 | 旧参照とbit一致 |
| --- | --- | --- | --- | --- | --- | --- |
| forward / blk.0.attn_gate.weight / M=256 / inference-bslm-k32r | 1024 | 2.420e-06 | 3.277e-07 | 0.000e+00 | 0.016 | はい |
| forward / blk.0.attn_gate.weight / M=256 / tiled-16x32-panel0 | 1024 | 7.469e-07 | 1.437e-07 | 3.480e-07 | 0.005 | いいえ |
| forward / blk.0.attn_gate.weight / M=512 / inference-bslm-k32r | 1024 | 2.683e-06 | 3.432e-07 | 0.000e+00 | 0.020 | はい |
| forward / blk.0.attn_gate.weight / M=512 / tiled-16x32-panel0 | 1024 | 9.861e-07 | 1.383e-07 | 3.481e-07 | 0.005 | いいえ |
| forward / blk.0.attn_gate.weight / M=1024 / inference-bslm-k32r | 1024 | 2.196e-06 | 3.018e-07 | 0.000e+00 | 0.017 | はい |
| forward / blk.0.attn_gate.weight / M=1024 / tiled-16x32-panel0 | 1024 | 9.328e-07 | 1.364e-07 | 3.479e-07 | 0.005 | いいえ |
| forward / blk.0.attn_gate.weight / M=2048 / inference-bslm-k32r | 1024 | 2.503e-06 | 3.187e-07 | 0.000e+00 | 0.018 | はい |
| forward / blk.0.attn_gate.weight / M=2048 / tiled-16x32-panel0 | 1024 | 7.666e-07 | 1.403e-07 | 3.482e-07 | 0.006 | いいえ |
| forward / blk.0.attn_gate.weight / M=4096 / inference-bslm-k32r | 1024 | 3.421e-06 | 3.461e-07 | 0.000e+00 | 0.026 | はい |
| forward / blk.0.attn_gate.weight / M=4096 / tiled-16x32-panel0 | 1024 | 8.519e-07 | 1.367e-07 | 3.480e-07 | 0.006 | いいえ |
| forward / blk.0.attn_gate.weight / M=8192 / inference-bslm-k32r | 1024 | 2.655e-06 | 3.221e-07 | 0.000e+00 | 0.019 | はい |
| forward / blk.0.attn_gate.weight / M=8192 / tiled-16x32-panel0 | 1024 | 9.460e-07 | 1.348e-07 | 3.481e-07 | 0.005 | いいえ |
| forward / blk.0.ffn_gate.weight / M=256 / inference-bslm-k32r | 1024 | 1.622e-06 | 3.200e-07 | 0.000e+00 | 0.017 | はい |
| forward / blk.0.ffn_gate.weight / M=256 / tiled-16x32-panel0 | 1024 | 4.819e-07 | 1.370e-07 | 3.474e-07 | 0.005 | いいえ |
| forward / blk.0.ffn_gate.weight / M=512 / inference-bslm-k32r | 1024 | 2.302e-06 | 3.198e-07 | 0.000e+00 | 0.028 | はい |
| forward / blk.0.ffn_gate.weight / M=512 / tiled-16x32-panel0 | 1024 | 4.818e-07 | 1.370e-07 | 3.470e-07 | 0.005 | いいえ |
| forward / blk.0.ffn_gate.weight / M=1024 / inference-bslm-k32r | 1024 | 1.397e-06 | 3.130e-07 | 0.000e+00 | 0.017 | はい |
| forward / blk.0.ffn_gate.weight / M=1024 / tiled-16x32-panel0 | 1024 | 4.465e-07 | 1.360e-07 | 3.471e-07 | 0.005 | いいえ |
| forward / blk.0.ffn_gate.weight / M=2048 / inference-bslm-k32r | 1024 | 1.242e-06 | 3.059e-07 | 0.000e+00 | 0.015 | はい |
| forward / blk.0.ffn_gate.weight / M=2048 / tiled-16x32-panel0 | 1024 | 4.620e-07 | 1.397e-07 | 3.468e-07 | 0.005 | いいえ |
| forward / blk.0.ffn_gate.weight / M=4096 / inference-bslm-k32r | 1024 | 1.674e-06 | 3.234e-07 | 0.000e+00 | 0.020 | はい |
| forward / blk.0.ffn_gate.weight / M=4096 / tiled-16x32-panel0 | 1024 | 4.872e-07 | 1.395e-07 | 3.470e-07 | 0.006 | いいえ |
| forward / blk.0.ffn_gate.weight / M=8192 / inference-bslm-k32r | 1024 | 1.799e-06 | 3.215e-07 | 0.000e+00 | 0.023 | はい |
| forward / blk.0.ffn_gate.weight / M=8192 / tiled-16x32-panel0 | 1024 | 5.128e-07 | 1.338e-07 | 3.471e-07 | 0.006 | いいえ |
| forward / blk.8.ffn_down.weight / M=256 / inference-bslm-k32r | 1024 | 5.855e-06 | 6.010e-07 | 0.000e+00 | 0.018 | はい |
| forward / blk.8.ffn_down.weight / M=256 / tiled-16x32-panel0 | 1024 | 1.681e-06 | 1.823e-07 | 6.215e-07 | 0.005 | いいえ |
| forward / blk.8.ffn_down.weight / M=512 / inference-bslm-k32r | 1024 | 6.203e-06 | 6.114e-07 | 0.000e+00 | 0.019 | はい |
| forward / blk.8.ffn_down.weight / M=512 / tiled-16x32-panel0 | 1024 | 1.565e-06 | 1.810e-07 | 6.207e-07 | 0.005 | いいえ |
| forward / blk.8.ffn_down.weight / M=1024 / inference-bslm-k32r | 1024 | 6.484e-06 | 6.090e-07 | 0.000e+00 | 0.020 | はい |
| forward / blk.8.ffn_down.weight / M=1024 / tiled-16x32-panel0 | 1024 | 2.226e-06 | 1.860e-07 | 6.213e-07 | 0.007 | いいえ |
| forward / blk.8.ffn_down.weight / M=2048 / inference-bslm-k32r | 1024 | 7.023e-06 | 5.606e-07 | 0.000e+00 | 0.022 | はい |
| forward / blk.8.ffn_down.weight / M=2048 / tiled-16x32-panel0 | 1024 | 1.632e-06 | 1.882e-07 | 6.210e-07 | 0.005 | いいえ |
| forward / blk.8.ffn_down.weight / M=4096 / inference-bslm-k32r | 1024 | 6.113e-06 | 5.871e-07 | 0.000e+00 | 0.019 | はい |
| forward / blk.8.ffn_down.weight / M=4096 / tiled-16x32-panel0 | 1024 | 2.323e-06 | 1.806e-07 | 6.211e-07 | 0.007 | いいえ |
| forward / blk.8.ffn_down.weight / M=8192 / inference-bslm-k32r | 1024 | 4.686e-06 | 5.900e-07 | 0.000e+00 | 0.015 | はい |
| forward / blk.8.ffn_down.weight / M=8192 / tiled-16x32-panel0 | 1024 | 1.223e-06 | 1.873e-07 | 6.210e-07 | 0.004 | いいえ |
| transpose / blk.0.attn_gate.weight / M=256 / tiled-transpose-16x32-panel0 | 1024 | 4.713e-06 | 5.973e-07 | 8.393e-07 | 0.028 | いいえ |
| transpose / blk.0.attn_gate.weight / M=512 / tiled-transpose-16x32-panel0 | 1024 | 7.269e-06 | 6.029e-07 | 8.383e-07 | 0.043 | いいえ |
| transpose / blk.0.attn_gate.weight / M=1024 / tiled-transpose-16x32-panel0 | 1024 | 4.932e-06 | 6.151e-07 | 8.385e-07 | 0.030 | いいえ |
| transpose / blk.0.attn_gate.weight / M=2048 / tiled-transpose-16x32-panel0 | 1024 | 5.998e-06 | 6.166e-07 | 8.378e-07 | 0.037 | いいえ |
| transpose / blk.0.attn_gate.weight / M=4096 / tiled-transpose-16x32-panel0 | 1024 | 8.474e-06 | 6.336e-07 | 8.371e-07 | 0.052 | いいえ |
| transpose / blk.0.attn_gate.weight / M=8192 / tiled-transpose-16x32-panel0 | 1024 | 5.392e-06 | 6.269e-07 | 8.375e-07 | 0.033 | いいえ |
| transpose / blk.0.ffn_gate.weight / M=256 / tiled-transpose-16x32-panel0 | 1024 | 9.952e-06 | 1.072e-06 | 1.203e-06 | 0.033 | いいえ |
| transpose / blk.0.ffn_gate.weight / M=512 / tiled-transpose-16x32-panel0 | 1024 | 8.316e-06 | 1.064e-06 | 1.202e-06 | 0.028 | いいえ |
| transpose / blk.0.ffn_gate.weight / M=1024 / tiled-transpose-16x32-panel0 | 1024 | 1.030e-05 | 1.083e-06 | 1.202e-06 | 0.035 | いいえ |
| transpose / blk.0.ffn_gate.weight / M=2048 / tiled-transpose-16x32-panel0 | 1024 | 1.284e-05 | 1.049e-06 | 1.203e-06 | 0.043 | いいえ |
| transpose / blk.0.ffn_gate.weight / M=4096 / tiled-transpose-16x32-panel0 | 1024 | 1.102e-05 | 1.088e-06 | 1.203e-06 | 0.037 | いいえ |
| transpose / blk.0.ffn_gate.weight / M=8192 / tiled-transpose-16x32-panel0 | 1024 | 6.430e-06 | 1.011e-06 | 1.203e-06 | 0.022 | いいえ |
| transpose / blk.8.ffn_down.weight / M=256 / tiled-transpose-16x32-panel0 | 1024 | 3.406e-06 | 5.797e-07 | 8.157e-07 | 0.037 | いいえ |
| transpose / blk.8.ffn_down.weight / M=512 / tiled-transpose-16x32-panel0 | 1024 | 3.200e-06 | 5.607e-07 | 8.160e-07 | 0.035 | いいえ |
| transpose / blk.8.ffn_down.weight / M=1024 / tiled-transpose-16x32-panel0 | 1024 | 2.336e-06 | 5.421e-07 | 8.157e-07 | 0.025 | いいえ |
| transpose / blk.8.ffn_down.weight / M=2048 / tiled-transpose-16x32-panel0 | 1024 | 3.537e-06 | 5.754e-07 | 8.155e-07 | 0.039 | いいえ |
| transpose / blk.8.ffn_down.weight / M=4096 / tiled-transpose-16x32-panel0 | 1024 | 4.048e-06 | 5.738e-07 | 8.156e-07 | 0.044 | いいえ |
| transpose / blk.8.ffn_down.weight / M=8192 / tiled-transpose-16x32-panel0 | 1024 | 2.918e-06 | 5.611e-07 | 8.156e-07 | 0.031 | いいえ |

forwardの全出力参照は旧推論BSLM。学習rows4も同じ参照およびCPU doubleで検証した。逆伝播の参照は旧transpose経路。bit一致と許容誤差内の一致を区別して記録した。

## 入力・再現条件

| 方向 / テンソル | M | GGUF形状 [K,N] | FP32 seed | 入力範囲・種類 | 重みpayload SHA-256 |
| --- | --- | --- | --- | --- | --- |
| forward / blk.0.attn_gate.weight | 256 | [5120, 6144] | 782528 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | d2e134e379f6ad09d29821492a83a339b4cc5449afe2c6d8b030cb313ea31eb4 |
| forward / blk.0.attn_gate.weight | 512 | [5120, 6144] | 782784 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | d2e134e379f6ad09d29821492a83a339b4cc5449afe2c6d8b030cb313ea31eb4 |
| forward / blk.0.attn_gate.weight | 1024 | [5120, 6144] | 783296 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | d2e134e379f6ad09d29821492a83a339b4cc5449afe2c6d8b030cb313ea31eb4 |
| forward / blk.0.attn_gate.weight | 2048 | [5120, 6144] | 784320 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | d2e134e379f6ad09d29821492a83a339b4cc5449afe2c6d8b030cb313ea31eb4 |
| forward / blk.0.attn_gate.weight | 4096 | [5120, 6144] | 786368 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | d2e134e379f6ad09d29821492a83a339b4cc5449afe2c6d8b030cb313ea31eb4 |
| forward / blk.0.attn_gate.weight | 8192 | [5120, 6144] | 790464 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | d2e134e379f6ad09d29821492a83a339b4cc5449afe2c6d8b030cb313ea31eb4 |
| forward / blk.0.ffn_gate.weight | 256 | [5120, 17408] | 793792 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | 6950b3f8a593a76b2b6b85f3c4a9357fa2475d70001987dc7497c81040c16541 |
| forward / blk.0.ffn_gate.weight | 512 | [5120, 17408] | 794048 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | 6950b3f8a593a76b2b6b85f3c4a9357fa2475d70001987dc7497c81040c16541 |
| forward / blk.0.ffn_gate.weight | 1024 | [5120, 17408] | 794560 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | 6950b3f8a593a76b2b6b85f3c4a9357fa2475d70001987dc7497c81040c16541 |
| forward / blk.0.ffn_gate.weight | 2048 | [5120, 17408] | 795584 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | 6950b3f8a593a76b2b6b85f3c4a9357fa2475d70001987dc7497c81040c16541 |
| forward / blk.0.ffn_gate.weight | 4096 | [5120, 17408] | 797632 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | 6950b3f8a593a76b2b6b85f3c4a9357fa2475d70001987dc7497c81040c16541 |
| forward / blk.0.ffn_gate.weight | 8192 | [5120, 17408] | 801728 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | 6950b3f8a593a76b2b6b85f3c4a9357fa2475d70001987dc7497c81040c16541 |
| forward / blk.8.ffn_down.weight | 256 | [17408, 5120] | 793792 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | bb263ad290c0acebacaf713c42ebaa203f8f442b39ec07eef62210a15e478aa7 |
| forward / blk.8.ffn_down.weight | 512 | [17408, 5120] | 794048 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | bb263ad290c0acebacaf713c42ebaa203f8f442b39ec07eef62210a15e478aa7 |
| forward / blk.8.ffn_down.weight | 1024 | [17408, 5120] | 794560 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | bb263ad290c0acebacaf713c42ebaa203f8f442b39ec07eef62210a15e478aa7 |
| forward / blk.8.ffn_down.weight | 2048 | [17408, 5120] | 795584 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | bb263ad290c0acebacaf713c42ebaa203f8f442b39ec07eef62210a15e478aa7 |
| forward / blk.8.ffn_down.weight | 4096 | [17408, 5120] | 797632 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | bb263ad290c0acebacaf713c42ebaa203f8f442b39ec07eef62210a15e478aa7 |
| forward / blk.8.ffn_down.weight | 8192 | [17408, 5120] | 801728 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | bb263ad290c0acebacaf713c42ebaa203f8f442b39ec07eef62210a15e478aa7 |
| transpose / blk.0.attn_gate.weight | 256 | [5120, 6144] | 492528 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | d2e134e379f6ad09d29821492a83a339b4cc5449afe2c6d8b030cb313ea31eb4 |
| transpose / blk.0.attn_gate.weight | 512 | [5120, 6144] | 492784 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | d2e134e379f6ad09d29821492a83a339b4cc5449afe2c6d8b030cb313ea31eb4 |
| transpose / blk.0.attn_gate.weight | 1024 | [5120, 6144] | 493296 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | d2e134e379f6ad09d29821492a83a339b4cc5449afe2c6d8b030cb313ea31eb4 |
| transpose / blk.0.attn_gate.weight | 2048 | [5120, 6144] | 494320 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | d2e134e379f6ad09d29821492a83a339b4cc5449afe2c6d8b030cb313ea31eb4 |
| transpose / blk.0.attn_gate.weight | 4096 | [5120, 6144] | 496368 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | d2e134e379f6ad09d29821492a83a339b4cc5449afe2c6d8b030cb313ea31eb4 |
| transpose / blk.0.attn_gate.weight | 8192 | [5120, 6144] | 500464 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | d2e134e379f6ad09d29821492a83a339b4cc5449afe2c6d8b030cb313ea31eb4 |
| transpose / blk.0.ffn_gate.weight | 256 | [5120, 17408] | 503792 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | 6950b3f8a593a76b2b6b85f3c4a9357fa2475d70001987dc7497c81040c16541 |
| transpose / blk.0.ffn_gate.weight | 512 | [5120, 17408] | 504048 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | 6950b3f8a593a76b2b6b85f3c4a9357fa2475d70001987dc7497c81040c16541 |
| transpose / blk.0.ffn_gate.weight | 1024 | [5120, 17408] | 504560 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | 6950b3f8a593a76b2b6b85f3c4a9357fa2475d70001987dc7497c81040c16541 |
| transpose / blk.0.ffn_gate.weight | 2048 | [5120, 17408] | 505584 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | 6950b3f8a593a76b2b6b85f3c4a9357fa2475d70001987dc7497c81040c16541 |
| transpose / blk.0.ffn_gate.weight | 4096 | [5120, 17408] | 507632 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | 6950b3f8a593a76b2b6b85f3c4a9357fa2475d70001987dc7497c81040c16541 |
| transpose / blk.0.ffn_gate.weight | 8192 | [5120, 17408] | 511728 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | 6950b3f8a593a76b2b6b85f3c4a9357fa2475d70001987dc7497c81040c16541 |
| transpose / blk.8.ffn_down.weight | 256 | [17408, 5120] | 503792 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | bb263ad290c0acebacaf713c42ebaa203f8f442b39ec07eef62210a15e478aa7 |
| transpose / blk.8.ffn_down.weight | 512 | [17408, 5120] | 504048 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | bb263ad290c0acebacaf713c42ebaa203f8f442b39ec07eef62210a15e478aa7 |
| transpose / blk.8.ffn_down.weight | 1024 | [17408, 5120] | 504560 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | bb263ad290c0acebacaf713c42ebaa203f8f442b39ec07eef62210a15e478aa7 |
| transpose / blk.8.ffn_down.weight | 2048 | [17408, 5120] | 505584 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | bb263ad290c0acebacaf713c42ebaa203f8f442b39ec07eef62210a15e478aa7 |
| transpose / blk.8.ffn_down.weight | 4096 | [17408, 5120] | 507632 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | bb263ad290c0acebacaf713c42ebaa203f8f442b39ec07eef62210a15e478aa7 |
| transpose / blk.8.ffn_down.weight | 8192 | [17408, 5120] | 511728 | 決定的乱数・通常範囲 [-2,2) / 実GGUF重み | bb263ad290c0acebacaf713c42ebaa203f8f442b39ec07eef62210a15e478aa7 |

入力SHA-256、重みoffset/byte数、bias、全CLI引数、デバイス・ドライバー、サンプル数・warmup回数はsummary.jsonのcases/evidenceに保存。

- [results.json](<C:/Users/takos/source/repos/NNtrain/benchmark-results/iq2s-production-20261008/final-forward-v3/results.json>)：GPU 0: Intel(R) Arc(TM) B580 Graphics。warmup 3回、9サンプル。round robinで奇数roundは逆順。
- [results.json](<C:/Users/takos/source/repos/NNtrain/benchmark-results/iq2s-production-20261008/final-backward-v3/results.json>)：GPU 0: Intel(R) Arc(TM) B580 Graphics。warmup 3回、9サンプル。round robinで奇数roundは逆順。
- [model-results.json](<C:/Users/takos/source/repos/NNtrain/benchmark-results/iq2s-production-20261008/final-model-v3/model-results.json>)：モデルwarmup 1回、3サンプル。
- [model-results.json](<C:/Users/takos/source/repos/NNtrain/benchmark-results/iq2s-production-20261008/final-model-profile-v3/model-results.json>)：別profile監査。warmup 0回、1サンプル。

## 極小値・範囲外の単体試験

- [iq2-final-guards-compact132.trx](<C:/Users/takos/source/repos/NNtrain/benchmark-results/iq2s-production-20261008/tests/iq2-final-guards-compact132.trx>)：30件すべて成功。
  tiny-normal両方向の試験振幅：1.00e-20, 1.00e-09。これは上記の速度測定とは別の単体試験。

## 来歴とファイル

入力JSONのcomplete、前後の数値検証、全サンプルの統計値、記録されたカーネル時間合計を再検査済み。共通の実装バイナリおよびソースのSHA-256は入力間で一致。失敗した測定は含めない。

| 証拠JSON | JSON SHA-256 |
| --- | --- |
| [results.json](<C:/Users/takos/source/repos/NNtrain/benchmark-results/iq2s-production-20261008/final-forward-v3/results.json>) | 2013d914ac7553e5074bf56f30f4fe52ef1cc86ab05c564942ca88360f8b5027 |
| [results.json](<C:/Users/takos/source/repos/NNtrain/benchmark-results/iq2s-production-20261008/final-backward-v3/results.json>) | 0a19d7d0dd7e2ddcebda308b1719bec45769ea0c79d7f233a830f41a9c2f844e |
| [model-results.json](<C:/Users/takos/source/repos/NNtrain/benchmark-results/iq2s-production-20261008/final-model-v3/model-results.json>) | 528c73245fed52698d7b370392b5005e5cb815ea4e94a535ccb0cd5e5f551430 |
| [model-results.json](<C:/Users/takos/source/repos/NNtrain/benchmark-results/iq2s-production-20261008/final-model-profile-v3/model-results.json>) | d35434e53f8b606750340f35d70426cb806105cf2b9153390b32b59dd4c37e61 |

| バイナリ / ソース | SHA-256 |
| --- | --- |
| NNtrain.Arc.dll | 19ca6278bf5abdb09f3f1eb349f67d90c08b864e46d3f283e14c280ce7dd70ec |
| NNtrain.Core.dll | 36d4318bfc0db71a53b79300f37a31666ec4ea764c45379f1ac3b4916ea23368 |
| NNtrain.Arc/Kernels/qwen35_attention.cl | 32d49983ed32f9790a87163b5b5748ab9afe182eecd1da906a0e757e44696169 |
| NNtrain.Arc/Kernels/qwen35_iq2_tiles.cl | 99c0e10525dfce277de40c1c5efd2406c6ee2bbb95316813f99aae0d602749a6 |
| NNtrain.Arc/Kernels/qwen35_prefill_xmx.cl | e4df934496be68ffeb5970c460f18f4d0ffb9b21e33471eb6d59c66fc48665ee |
| NNtrain.Arc/ArcExecutionLane.Iq2.cs | 3657b53fff141aafc24f4f9747d98289eb87881b33da5f75e7b63978d79d87b5 |
| NNtrain.Core/Modules/Qwen35ExecutionOptions.cs | 7cf0e2eb03e6d336b8226df6c890dd2b0f309c2f95b058f2f2f4e2182f40545f |
| NNtrain.Core/Modules/Qwen35QuantizedModel.cs | 5aa00d909c8785fc41b0edb95d8bd342ea8e52efa7aaba947811540e268ad65d |
| NNtrain.Core/Modules/Qwen35QuantizedModel.Prefill.cs | 9089bef30fd6f2412f6e51ef8624b93320080bb24d4d373dafd52d137ed6b359 |

全数値・全候補・全サンプル・入力/ソースSHA一覧：[summary.json](<C:/Users/takos/source/repos/NNtrain/benchmark-results/iq2s-production-20261008/final-report-v3/summary.json>)
