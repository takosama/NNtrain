# Arc B580 2GPU 学習: Attention を含む最適化（2026-09-24）

## 対象と計測方法

`training.transformer.json` の `mix8_16`、Arc B580 2枚、batch 16 × sequence 2048 × accumulation 8、32層、幅512、16 heads、FFN1536、語彙11500、Muonを使用。**実効262,144 tokens/updateは維持**した。固定seedの合成tokenと新規モデルを使い、実データ読み込み・tokenizer・checkpoint保存・計測後のfinite-gradient走査はtok/sに含めない。計測対象はforward/backward、GPU間勾配還元、clip、optimizer、replica同期の同期wall時間。

前回採用した7件の非Attention最適化を有効にした状態を今回の基準にした。初期の2更新profileは15,242.37 ms/update、17,198.4 tokens/s。設定SHA-256は `E876C55B7BFBE4F0BA5054D0A86070E525C776257E8C260A976A8F4305C1A50C`。学習設定や実効バッチは変更していない。

## 最終 A/B/A 結果

A1/B/A2は各1 warmup＋3測定更新。同じ設定とCore/Arc/Benchmarksバイナリで、今回の6フラグだけを切り替えた。

| 経路 | 測定更新数 | 同期wall中央値 | tokens/s |
| --- | ---: | ---: | ---: |
| A1: 6候補無効 | 3 | 15,223.63 ms | 17,219.5 |
| A2: 6候補無効 | 3 | 15,219.18 ms | 17,224.6 |
| A1+A2の6更新をプール | 6 | **15,221.41 ms** | **17,222.1** |
| B: 6候補有効 | 3 | **14,602.80 ms** | **17,951.6** |

**618.60 ms/update短縮、tokens/sは4.24%増**。採用した6フラグは `Mix8_16NativeExpAttention`、`Mix8_16DpDsK32`、`Mix8_16DkvPackedSlm`、`Mix8_16PvPackedSlm`、`Mix8_16DqPackedSlm`、`Mix8_16FusedNormParameterGradients`。既定で有効、`ArcExecutionOptions.Reference`では全て無効とした。

GPU 0/1のpeak backend allocationはA/Bとも6,091.1/5,558.9 MiB、各更新のlive allocation差は0 bytes。これはbackendのnative allocationであり、driverが報告するVRAM値ではない。更新あたりのH2D/D2HはGPU 0が183,112,152/102,411,320 bytes、GPU 1が103,459,340/182,063,608 bytesで、A/B間で一致した。

| 更新 | A loss | B loss | loss差 | A gradient norm | B gradient norm | norm相対差 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 9.349764 | 9.349768 | +0.000004 | 0.07547713 | 0.07544427 | −0.04354% |
| 3 | 9.312968 | 9.312960 | −0.000008 | 0.07395657 | 0.07394024 | −0.02208% |
| 4 | 9.276879 | 9.276876 | −0.000003 | 0.08118565 | 0.08121390 | +0.03480% |

全測定更新の勾配はfinite。短期の合成データ検証であり、長期の収束を確認した結果ではない。最終Releaseビルドは警告・エラー0、精度モード、Transformer、2GPU data parallel、新旧mix8_16経路を含む**回帰テスト117件が成功、失敗・skipは0**。集計は `benchmark-results/arc-overall-final-summary-20260924.json` と `NNtrain.Core.Tests/TestResults/arc-overall-final-core-20260924.trx`。

### 最終profile

Bを別途1 warmup＋2測定、`--profile --timeline`で測った。wall中央値は14,624.59 msで、速度の比較値は上のA/B/Aを用いる。両GPU・両更新ともtimeline coverage 100%、missing event 0、opaque allocation copy 0。device event数はGPU 0/1で174,782/164,303件。

| GPU | Attention kernel event: 初期 → 最終 | その他のkernel event: 初期 → 最終 |
| --- | ---: | ---: |
| 0 | 9,197.30 → 8,626.89 ms | 5,495.83 → 5,468.41 ms |
| 1 | 9,198.92 → 8,639.14 ms | 5,383.57 → 5,354.46 ms |

GPU 0の最終排他的wall区分はAttention 8,627.24 ms、GEMM 3,247.40 ms、pack/decode/publication 1,128.38 ms、normalization/reduction 835.23 ms。旧norm dX約179 msとparameter scan約107 msは、融合dX/partial 229.27 msとpartial縮約33.42 msになった。個々のfusion kernelが長くなっても、周辺の処理を含めると短縮している。

最終4run（A1/B/A2/profile）のSHA-256は一致:

```text
NNtrain.Core       F1C732D8CE85ECBED6F8F1CDB7C666569097760671DF66573C0D8095A4B275F9
NNtrain.Arc        C68906913DDE798DB9033DAD98336F30FBBA37665C0B6E724A5F550323BA944E
NNtrain.Benchmarks 0C32DD369DC4FF3B062D0FC45D3B75B953A1149B33A61FB161D011B6CEE8DB3E
```

## 初期profileと仮説

GPU 0、1更新あたりのkernel event時間。これらを2GPU分足してwall時間とは比較しない。最終採否には同じバイナリの同期wallを用いる。

| 処理 | GPU 0 / 更新 | 改善の仮説 |
| --- | ---: | --- |
| Attention dK/dV | 2,120.3 ms | P/dSをSLM内でもBF16の組として保持し、SLM使用量と読出しを減らす |
| Attention dP/dS | 1,464.8 ms | K16を2回読む代わりにK32を一度読み、barrierを4回から1回へ減らす |
| Attention dQ | 1,361.3 ms | Kの2チャネルを1つのuintにまとめ、SLM上のFP32展開を避ける |
| Attention PV | 1,321.7 ms | Vも同様に2チャネルをまとめる |
| softmax forward | 1,048.3 ms | `native_exp`を使い、reductionの順序とFP32 workspaceを維持する |
| softmax saved replay | 637.1 ms | forwardと同じ指数関数で再計算し、保存したmaximum/inverseを使う |
| norm parameter gradient scan | 約107 ms | dXの計算中にgamma/betaのpartialを作り、x/dy/statsの再走査を省く |

`mix8_16`はINT8の重みを優先し、速ければBF16の重み演算を許可する。重み以外は速度を優先する既存の方針を維持した。今回の変更は重みの保存形式やGPU間転送の精度を変更しない。

## カーネル単体の比較

T2048/D32、2 heads、入力はGPU上に常駐。3 warmup＋15測定をbaseline/candidateの交互順で実施し、D2D resetと同期は計時外にした。表はGPU eventの中央値で、学習全体の速度とは区別する。

| 候補 | baseline → candidate | kernel時間の短縮 | 判定 |
| --- | ---: | ---: | --- |
| native_exp forward | 0.083333 → 0.075104 ms | 9.87% | 全体でも確認 |
| native_exp saved replay | 0.065625 → 0.059479 ms | 9.37% | 全体でも確認 |
| PV packed-B SLM | 0.086041 → 0.079583 ms | 7.51% | 全体でも確認 |
| dP/dS K32 | 0.096562 → 0.091354 ms | 5.39% | 全体でも確認 |
| dQ packed-B SLM | 0.103958 → 0.098645 ms | 5.11% | 全体でも確認 |
| dK/dV packed P/dS SLM | 0.147083 → 0.141979 ms | 3.47% | 全体でも確認 |
| PV M128 | 0.086250 → 0.190729 ms | −121.14% | 不採用 |
| dQ M128 | 0.104270 → 0.160208 ms | −53.65% | 不採用 |
| dQ packed-A+B SLM（別run） | 0.104479 → 約0.1115 ms | 約−6.7% | 不採用 |

ここで比較した全出力はbitwise一致した。native_expの単体ベンチ入力は狭い分布であり、一般の入力でbitwise一致するという意味ではない。

driverが報告したSLMは、PV/dQのpacked-Bで12,672→10,624 bytes、dK/dVで16,640→12,416 bytes。K32 dP/dSは8,512→16,768 bytesへ増えるが、同期回数削減が勝った。M128は21,120 bytesに増えて大幅に遅くなった。dQのAまで圧縮する案は6,784 bytesまで減ったが、BF16からの復元処理が増え、Bだけの圧縮より遅かった。SLM削減だけで採用を判断しない。

## 全形状pilotと採否

各runは1 warmup＋2測定、2GPU・32層。最終A/B/Aとは別の候補選別測定。

| 経路 | ms/update | tokens/s |
| --- | ---: | ---: |
| K32 dP/dS + packed dK/dV | 15,064.97 | 17,400.9 |
| 上記 + native_exp | 14,947.09 | 17,538.1 |
| 上記 + PV/dQ packed-B | 14,640.44 | 17,905.5 |
| 上記 + norm parameter fusion | 14,598.24 | 17,957.2 |
| 同じ6候補、workspace128 MiB（4 heads） | 17,078.78 | 15,349.1 |

ヘッド数は1/2/3/4の単体比較も行った。2と4はkernel中央値を頭数で割った指標では近かったが、全形状では4 headsが明確に悪化した。**workspace64 MiB、2 headsを維持**する。M128とdQ packed-A+Bも既定では無効。

norm融合は幅512・512行以上のmix8_16 parallel residual backwardだけに適用する。4行partialを作り、256行partialへ還元して既存finish kernelへ渡す。dX/residualの計算順序は維持し、gammaの各寄与も従来の `dy*(x-mean)*inverse` と同じ乗算順序にした。gamma/betaの加算順序は変わる。

以前試したdWの84通りのタイル/split-K、loss chunk拡大、panel cache拡大、BF16 XMX Attentionへの全面置換などは、以前の計測で勝たなかった同一実装を繰り返さなかった。Fc1 epilogueからFc2用BF16 panelを同時生成する案も確認したが、現packの削減上限は88 ms/update（約0.6%）。追加48 MiB/呼出しの書き込みとpanelの所有権管理が必要で、今回は実装対象に含めなかった。

## 数値と回帰検証

- packed SLM、K32 dP/dS、M128、dQ packed-A+Bは元のFP32 FMA順序を維持し、単体で全出力のbitwise一致を検査。
- native_expは因果マスク0/1/2、定数行、±10000の有限値、広い/狭いlogit分布、masked signaling NaN、head offsetを検査。double安定softmaxとの最大絶対差は6.43984×10⁻⁸、最大相対差は9.2702×10⁻⁷。事前の許容幅 `2e-6 + 1e-5*|reference|` と行和誤差2e-5以内に収まり、fresh/savedの出力はbitwise一致。
- Tensor経由でもbatch2/heads2、workspace96 MiBによるbatch境界をまたぐ3 headsのlaunchと末尾headを検査。5 Attention候補のON/OFFで出力・入力勾配の相対RMSは0、mix8_32のfallbackもbitwise一致。
- normは2回の勾配蓄積、branch alias、513/1025の末尾行、幅65のfallback、mix8_32を検査。dXはbitwise比較、gamma/betaはBF16の丸め幅とFP32寄与の誤差見積り、相対RMS上限5e-4で判定。測定したケースの誤差は0だった。

native関数の精度と入力範囲はOpenCL仕様では実装依存。今回の採用はB580の実測に基づく。[Khronos OpenCL C specification](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html)。Intelも速度と精度の要件に応じたnative mathの検討を案内している。[Intel native math guide](https://www.intel.com/content/www/us/en/docs/opencl-sdk/developer-guide-processor-graphics/2019-4/considering-native-and-half-versions-of-math-built.html)。全kernelにfast-relaxed-mathを設定する変更は行わず、該当softmaxの指数関数だけを明示的に変更した。

## 再現

Releaseビルドを作り、同じバイナリでA/B/Aを実行する。出力ファイルは未使用の名前を指定する。

```powershell
dotnet build NNtrain.slnx -c Release

# A: 今回の6候補を無効にする。前回の非Attention最適化は有効。
dotnet NNtrain.Benchmarks/bin/Release/net10.0/NNtrain.Benchmarks.dll --probe-arc-transformer training.transformer.json benchmark-results/repro-overall-a.json --warmup 1 --steps 3 --profile --mix8-16-native-exp off --mix8-16-dpds-k32 off --mix8-16-dkv-packed-slm off --mix8-16-pv-packed-slm off --mix8-16-dq-packed-slm off --mix8-16-norm-parameter-gradients off

# B: 採用した6候補は既定で有効。
dotnet NNtrain.Benchmarks/bin/Release/net10.0/NNtrain.Benchmarks.dll --probe-arc-transformer training.transformer.json benchmark-results/repro-overall-b.json --warmup 1 --steps 3 --profile

# GPU単体の候補比較。末尾は常駐head数（1〜4）。
dotnet NNtrain.Benchmarks/bin/Release/net10.0/NNtrain.Benchmarks.dll --probe-arc-overall-attention benchmark-results/repro-overall-micro.json 2
```

実測JSONは `benchmark-results/arc-overall-*-20260924.json`、テストTRXは `NNtrain.Core.Tests/TestResults/arc-overall-*-20260924.trx` に保存。
