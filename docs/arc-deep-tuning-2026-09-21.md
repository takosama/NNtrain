# Arc B580: 直接XMX・低精度storage・Attention・allocatorの再調整

## 目的と測定条件

直前の改善版11,657.0838 ms/updateから、さらに3倍（3,885.7 ms/update以下）を目標に調査した記録。
3倍は未達。中間の小規模ベンチや数値検査不合格の候補を最終成果として扱わない。

- Intel Arc B580、driver 32.0.101.9030、Windows x64、.NET 10 Release。
- `training.transformer.json`のモデル・optimizer factoryを使用。
- batch16、勾配累積4、実効batch64、sequence1024、32層、幅512、16 heads、head幅32、hidden1536、vocabulary11500。
- 90,507,500 parameters、mix8_32/block32、tied embedding、dropout0.1、seed1234。
- Muon固定NS5（LR0.001）＋AdamW（LR0.0003）、weight decay0.01。
- 合成固定token列・初期化直後のモデル・固定LR。corpus、tokenizer、checkpoint、生成、HTMLのI/Oは含めない。
- forward/backward/clip/optimizerを同期wall timeで計測。全勾配の有限性検査のhost転送は測定外。
- GPUイベントの時間はhost waitやphase wall timeに重なるため、両者を足してはならない。
- 本測定時の`training.transformer.json`のSHA256: `952ECFEFCB4A20463B2F83A9231CABC5B0651C02F5F02BD1E464D2FC451AEAF9`。測定後、ユーザー指定で既定batchSizeは64へ変更された。以下の再現コマンドは`--batch 16`で測定時の条件を固定する。

## 同一binaryでの最終比較（3 warmup＋10 updatesずつ）

| 指標 | 今回の変更を無効化 | 採用版 |
|---|---:|---:|
| step p50 | 12,064.38 ms | 8,276.29 ms |
| step mean | 12,067.54 ms | 8,276.81 ms |
| tokens/s（p50換算） | 5,432.19 | 7,918.53 |
| forward mean | 3,211.03 ms | 2,308.37 ms |
| backward mean | 8,512.56 ms | 5,744.84 ms |
| clip mean | 20.49 ms | 20.38 ms |
| optimizer mean | 317.30 ms | 197.01 ms |
| backend buffer peak | 5,386.63 MiB | 5,300.28 MiB |
| 更新後active buffer | 1,478.14 MiB | 1,478.14 MiB |
| 更新後cached buffer | 497.99 MiB | 505.74 MiB |
| 更新後retired buffer | 0 | 0 |
| native allocations/update | 2,924 | 2,494 |
| allocation wall/update | 344.89 ms | 282.78 ms |
| kernel launches/update | 61,663 | 62,171 |
| H2D/update | 524,288 bytes | 524,288 bytes |
| D2H/update | 2,588 bytes／261 scalar copies | 同じ |

同一binary比較で**1.458倍、時間31.40%削減**。
前回保存値11,657.08 msに対しては**1.408倍、時間29.00%削減**。
再測定baselineは前回値と一致しないため、比較起点を明示して両方記載する。
3倍目標3,885.7 msには届いていない。

全更新でloss・norm・勾配は有限。同じstepのloss最大絶対差0.000083、norm最大絶対差0.00041015。
小規模の厳密kernel比較・既存許容差での学習比較は通るが、この長めの合成系列全体がbit一致するという意味ではない。
通常学習のJSONを変更せず、新経路が既定で使われる。

- [変更を無効化した測定](../benchmark-results/arc-next-final-baseline-full-20260921.json)
- [採用版の測定](../benchmark-results/arc-next-final-optimized-full-20260921.json)

この比較後、解放待ちbufferのexact-size再利用も追加測定し、採用した。
3 warmup＋10更新でp50 **8,268.37 ms**、mean8,271.36 ms、native確保**2,227回/update**（267回を再利用）、allocation wall259.41 ms、buffer peak**5,285.87 MiB**。
H2D/D2Hと更新後active/cached量は変わらず、測定10更新内のactive/peak増分は0 MiB。
時間差は0.1%程度なので、速度上の明確な上積みではなく再確保・ピーク削減として採用する。
この追加版ではloss最大差0.000088、norm最大差0.00044145。全勾配有限。
[追加版測定](../benchmark-results/arc-next-final-retired-full-20260921.json)が通常起動の最終feature構成に対応する。

残るGPUイベントはAttention約3,597 ms、GEMM約1,889 ms、packing約724 ms、codec約723 ms、norm/bias約648 ms/update。
backwardはwallの69.4%。ここを2倍にしても全体は約1.53倍。
optimizerは約2.4%なので、そこだけをゼロにしても全体約1.02倍に留まる。
3倍到達にはAttentionと行列積・その入出力をさらにまとめて削る必要があり、今回の結果をハードウェアの限界とは見なさない。

## 実装した主要経路

1. **直接XMX GEMM**：BF16 panelからDPASへ直接投入。SLM往復を削減し、形状別に256×32/128×32 tileを選択。対応外形状は従来Arc kernelへfallbackし、CPUへ戻さない。
2. **低精度storageの直接packing**：BFP8/BF16からBF16 operand panelへ変換。中間FP32 activation/weight全体のdecodeを除去。FP32 gradient・accumulationの意味論は維持。
3. **packingのレイアウト改善**：転置はcoalesced 32×32、非転置Aはvector4。gate・端数・offset・block scaleを含むpanel全要素のbit比較を実施。
4. **gradient丸めの統合**：単独のmatrix-gradient BF16丸めpassをpackingへ統合。biasは従来の丸め境界、LayerNormはFP32 reductionを維持。
5. **loss-head weight panel再利用**：forward/backward内だけでweight panelを再利用。optimizerを跨ぐcacheは作らず、古いweightの使用を防止。32 chunksの場合、更新あたり372 packing launchesを削減。
6. **FP32 Attentionのtile変更**：dP/PV/dQを調整し、D32専用の整列版とtail版を追加。FP32 FMAの順序を維持。
7. **DKV統合**：key-owner方式でdK/dVを同時に計算。D32 causalはK32/Q32、denseはK64/Q32。atomicなし。
8. **固定容量LRU buffer pool**：解放済みbufferのみをLRU管理。最初に入った冷たいサイズが512 MiB poolを占有し続ける問題を解消。実行中bufferの寿命・deferred release fenceを維持。
9. **勾配書込みのみの融合**：LayerNorm自体の加算順・scratchを変えず、その後の連続アドレス書込み2passを統合。同一Tensorが残差とbranchの両方に使われる場合も更新順を維持。負のゼロ・subnormal・既存勾配への2回加算をbit単位で比較。
10. **解放待ちbufferの同一queue再利用**：pool miss時、native参照を保持しているretired bufferをexact-sizeで再利用。古いowner/borrowは無効のまま、新ownerへ物理byte計上を移す。in-order queue内の順序は維持し、早すぎるnative解放やVRAM計上漏れを起こさない。

DPAS使用時のdriver resource queryでは、採用したdirect GEMM・D32 Attention候補はspill 0。手書きassemblyを投入したとの主張ではなく、Intel subgroup matrix intrinsicを使ったnative GPU kernelの変更である。

## Amdahlに基づく原因と途中経過

直前版のGPUイベント集計はAttention約4,700 ms、GEMM約4,028 ms、codec約1,261 ms/update。
Attentionだけをゼロにしても全体最大約1.68倍、GEMMだけでも約1.53倍にしかならない。
optimizerだけの改善では3倍は狙えないため、行列積・Attention・転送形式・allocationを順に変更した。

- direct GEMM導入後、行列積算術は約1,901 msまで減ったが、新しいpackingとnative allocationが増加。
- 中間版ではallocationが2,928→3,664回/update、allocation時間338→500 ms、wallとGPUイベント合計の差が約721→1,460 msへ増加。
- LRU化は容量512 MiBのまま、中間構成のp50を9,410→8,414 msへ短縮。allocationも約3,700→2,494回へ減少。
- LRU640/768 MiBは中間構成で約0.5%差に留まり、不採用。event queue上限512も改善せず、128を維持。
- workspace64/128/256 MiBも比較。大きいworkspaceは遅く、64 MiBを維持。

上記8,414 msの中間版は、後にLayerNormの学習全体数値検査で不合格となった並列reductionを含む。
**最終的にそのLayerNormは採用していない。**

## 不採用にした候補

- dP→dS→dQ全融合、softmax→PV融合：小規模実測でも遅い。
- 4 subgroup/rowのreduction：barrierは減ったが遅い。
- 1 subgroup/rowのreductionも追加検証：4本のprivate部分和で旧64-lane treeを再現し、15条件でprobability/statistics/derivative/下流勾配がbit一致。しかしprobability 0.0875→0.1177 ms、derivative 0.1376→0.1456 msへ悪化したため不採用。SLM/barrierが0でも速いとは限らなかった。
- wide direct GEMM：driver上で4,224 bytesのspillがあり、速くない。
- branchless BF16 rounding：393,216 patternの検査は通るが、速度優位なし。
- LayerNormの並列sum：単体検査は通っても、2step学習全体で既存許容差を超える。許容差を広げず棄却。
- 厳密な加算順を保つ残差LayerNorm全面融合：数値検査は通るが、strided gradient書込等によりnorm backwardが約240→1,912 msとなる。全体9,985 msに悪化したため既定無効。
- TF32等への追加精度変更は行っていない。

## 正しさ・回帰検証

- Release solution build：warning/errorとも0。
- Core Arc tests：197合格、失敗0、skip0。`benchmark-results/test-results/arc-deep-core-verified-20260921.trx`。
- 関連Integration tests：202合格、失敗0、skip0。`benchmark-results/test-results/arc-deep-integration-verified-20260921.trx`。FineWeb小規模訓練・保存・再開・生成・HTML継続、設定、dtype checkpoint、epoch境界を含む。Coreと合わせ399件。
- SLM比較テスト・probeはDirectXmxMatrices=falseを明示し、最適化既定値の変更で比較先が同じ経路になる問題を防止。
- buffer budget-fenceテストもretired再利用を明示的に無効化し、旧解放経路を引き続き検証。新再利用は独立6ケースで検証。
- inline gradient／typed storage／tail・転置／loss head／norm alias／buffer寿命／2step学習のloss・全勾配・全master updateを検証。既存の許容差を広げていない。
- microprobeは全panel・全配列を比較したうえで測定。特殊値・低精度境界の検証数は各JSONに記録。
- これは対象を絞った回帰検証であり、全solutionの全テストやCUDA実機の合格を意味しない。

## 再現

出力先は新規ファイルが必須。同じ名前を上書きしない。

```powershell
dotnet run --project .\NNtrain.Benchmarks -c Release --no-build -- `
  --probe-arc-transformer .\training.transformer.json .\benchmark-results\new-baseline.json `
  --batch 16 --warmup 3 --steps 10 --profile --next-features none

dotnet run --project .\NNtrain.Benchmarks -c Release --no-build -- `
  --probe-arc-transformer .\training.transformer.json .\benchmark-results\new-optimized.json `
  --batch 16 --warmup 3 --steps 10 --profile
```

個別probe：`--probe-arc-xmx-next`、`--probe-arc-xmx-direct-tune`、`--probe-arc-xmx-pack-tune`、`--probe-arc-storage-pack-linear`、`--probe-arc-attention-fp32-tiles`、`--probe-arc-attention-fp32-special`、`--probe-arc-dkv-special`。
全て続けて新規JSON出力パスを指定する。

主な保存済みmicro結果：

- [GEMM tile/resource](../benchmark-results/arc-next-direct-tune-20260921.json)
- [転置packing](../benchmark-results/arc-next-pack-tune-20260921.json)
- [storage vector packing](../benchmark-results/arc-next-linear-pack-20260921.json)
- [Attention D32](../benchmark-results/arc-next-attention-special-20260921.json)
- [DKV](../benchmark-results/arc-next-dkv-special-20260921.json)
- [不採用の1 subgroup/row](../benchmark-results/arc-next-row-subgroup-20260921.json)

## 検証の限界

本番形状であっても合成データの短期ベンチであり、実corpusの収束速度改善は未検証。
長時間soak、210/2100-step、現在接続されていないCUDA GPUとの実機比較も未実施。
memory値はbackendが所有するbufferの計測で、driver全体のVRAM使用量とは異なる。
本番checkpoint・tokenizer・dataset・学習履歴の書換えは行わない。
