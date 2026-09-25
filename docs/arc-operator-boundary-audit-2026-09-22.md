# Arc Transformer: 演算境界の型変換・並び替え監査

## 結論

削減候補はある。特に **ReLU backwardの行優先BF16中間バッファ、attentionのQKV全体FP32展開、LayerNorm前後のFP32中間バッファ** は、消費側へ処理を統合する余地がある。

ただし、現在のLinearはすでにBFP8からBF16のXMX用配列へ直接変換し、大きなforwardではBFP8出力まで融合している。全GEMMが `8→32→16→32→8` の巨大な中間配列を作っているわけではない。必要なBF16丸め、BFP8量子化、DPASの入力配列形式まで「無駄」として削除すると数値意味論が変わる。

現在条件で変換・整形関連のGPUカーネル時間は合計 **4,906.78 ms/update**。これは必要な計算も含む処理群の時間であり、削減見込みではない。一方、attention関連は **24,861.79 ms/update**。大幅な高速化には変換削減だけでなく、attention内部の中間行列の読み書き・演算自体の改善が必要。

今回はコード調査と既存プロファイルの再集計のみ。実装・設定変更、新規GPUベンチ、テスト再実行は行っていない。

## 1. 対象・測定条件・限界

- コード: `12f6aa8c3313338d32a5c58fb74081e6332fd745`。
- 対象: 現在の `training.transformer.json` が選択する **Arc / Transformer / mix8_32** のforward、loss、backward、勾配累積、clip、Muon/AdamW、常駐buffer管理。
- GPU: Intel Arc B580、driver `32.0.101.9030`。
- batch 16 × accumulation 8、context 2048、262,144 tokens/update。
- 32層、width 512、hidden 1536、16 heads、head dimension 32、語彙11,500、91,031,788 parameters。
- BFP8 block32保存、行列operandはBF16、累積・勾配・masterはFP32。Muon固定NS5＋補助AdamW。
- 設定SHA256: `9B298E978EB748C5218CB87A8A9AF31BF13219519C3B28B5A39826F03C312CCD`。調査時にも一致を確認。
- 元データ: [arc-config-t2048-b16-a8-20260922.json](../benchmark-results/arc-config-t2048-b16-a8-20260922.json)。2 warmup＋4測定、Release、固定seedの新規モデル、合成full-length token。
- step p50 **38,989.10 ms**、平均 **38,990.68 ms**、**6,723.52 tokens/s**。
- 平均forward 9,889.45 ms、backward 28,874.02 ms、optimizer 199.74 ms、clip 20.57 ms、zero-grad 6.58 ms。

実コーパス読込、tokenizer、保存、HTML、サンプル生成、schedulerは計測外。収束速度を評価したベンチではない。CPU/CUDA/DRN/DPOや任意の別shapeを全実行した監査でもない。現在経路の分岐先と不採用fallbackを区別している。

以下のカーネル時間は4 updateの平均。Profileの各recordの `Count` を合計して回数を求めた。この計測の `Timeline` はnullであり、排他的なwall-time内訳ではない。host-submit、host-wait、queue-delay、GPU時間を足して100%にしてはいけない。候補間でも同じカーネル群を参照するため、候補の数値は加算不可。

## 2. 現在経路を端から端まで追った結果

モデル入口は [GptRinWikiJp.cs](../NNtrain.Core/Modules/GptRinWikiJp.cs) の `ForwardLoss` / `ForwardHidden`、blockは [TransformerBlock.cs](../NNtrain.Core/Modules/TransformerBlock.cs)。attentionは [MultiHeadAttention.cs](../NNtrain.Core/Modules/MultiHeadAttention.cs)、FFNは [FeedForward.cs](../NNtrain.Core/Modules/FeedForward.cs)。

| 境界 | 実際のデータ経路 | 判定 |
|---|---|---|
| token/position embedding | BFP8の表全体→FP32表→gather/addのFP32出力→BFP8 | 表全体の展開は必須ではない。gather時に必要値だけdecode可能 |
| embedding dropout | BFP8→FP32全体→dropoutのFP32出力→BFP8 | typed load/storeによる中間削減候補。embedding出力の量子化境界は保持する |
| QKV / Wo / Fc1 / Fc2 forward | BFP8→BF16 DPAS panel→FP32レジスタ累積→bias/ReLU→直接BFP8出力 | 大きなprojectionは既に融合済み。行列全体のFP32 decodeなし |
| QKV→attention | BFP8 QKV→BF16丸め済みのFP32全体→Q/KのBF16 panel。V等はFP32配列から読む | FP32全体のmaterializeと再packに削減余地 |
| attention内部 | QK→FP32 score→同じbufferを確率へ上書き→FP32 PV出力→BFP8 | 大きな中間行列の読み書き。現在Flashは無効 |
| residual/dropout→LayerNorm | 2つのBFP8をdecodeしdropout/addを融合→FP32和→FP32 norm出力→BFP8 | decode用の2配列は既に不要。和とnorm出力の中間配列は残る |
| LayerNorm backward | 同じresidual和を再計算→FP32 dx一時配列→2枝の勾配へscatter/加算 | 一時dxと別scatterを融合可能。再計算の全廃はVRAMと交換 |
| Linear backward | FP32 dY→dX用BF16 panel、および転置dW用BF16 panel | 同じ源を2回走査。異なる配列形式は必要だが同時生成は検討可能 |
| ReLU backward | FP32 dY＋量子化gate→行優先BF16配列→上記2種類のpanel＋bias reduction | 行優先BF16を間に置かずgateとpanel生成を統合する候補 |
| loss head | 512行chunk、BFP8入力→BF16 panel→FP32 logits→BF16値へ丸めたFP32 logits→CE | 独立したlogits丸めpassを融合可能。重みpanelはchunk間で既に再利用 |
| loss backward | logitsを再計算→FP32 dLogits→2種類のpanel。dX chunk→全体dXへ加算コピー | 再計算はメモリ節約の設計。chunkコピーはoffset対応で除去可能 |
| grad accumulation / clip | 常駐FP32勾配、FP32 reduction、clip結果だけreadback | FP32を一律BF16へ落とす最適化は不可 |
| Muon | 常駐FP32 moment→正規化/実転置→各NS行列積でBF16 panel→FP32結果→master更新→BFP8再量子化 | 転置・packの融合候補はあるが全optimizerが0.20秒と小さい |
| AdamW | 常駐FP32 state/gradient/master更新→BFP8再量子化 | optimizer stateの毎step host往復なし |

### 転送について

測定updateのH2Dは **2,097,152 bytes、16回**。`16 × 2048 × 8 × (token 4 bytes + target 4 bytes)` に一致する。

D2Hは **2,604 bytes、265回**。コードとprofileの回数から、Muonの128個のconfidence統計（各16 bytes）、128個のnorm（各4 bytes）、8個のloss（各4 bytes）、clip（12 bytes）に一致する。したがって、このstepの重量級weight/activation/gradientの暗黙D2Hは見えていない。ただし「lossだけ1回」でもなく、scalar同期の整理余地はある。checkpointや明示的なTensor読出しは別範囲。

根拠: [Tensor.ArcResidency.cs](../NNtrain.Core/Tensors/Tensor.ArcResidency.cs) `EnsureArcPacked` / `DecodeArcReplica` / `SynchronizeArcHostData`、[ArcMuonMath.cs](../NNtrain.Core/Optimization/ArcMuonMath.cs) `Confidence` / `SumSquares`、[ArcTrainingMath.cs](../NNtrain.Core/Optimization/ArcTrainingMath.cs) `ClipResident`。

## 3. 変換・整形関連の実測総量

この表の行同士はカーネル名で分離しており重複しない。ただし算術を含む複合カーネルもあるため、「純粋な不要castの費用」とは解釈しない。

| GPUカーネル群 | 回/update | 平均ms/update |
|---|---:|---:|
| XMX panel pack全種＋Q/K pack | 15,384 | 2,819.06 |
| `decode_bfp8` | 3,152 | 587.38 |
| `norm_packed_residual_input` | 1,024 | 719.58 |
| `linear_relu_grad_packed` | 256 | 412.64 |
| `resident_bfp8_quad4_32` | 1,181 | 271.67 |
| `round_bf16_values` | 1,024 | 91.80 |
| optimizer `transpose_scale` | 192 | 4.65 |
| **合計** | **22,213** | **4,906.78** |

これらを仮に全て無料化する単純なアムダール試算でも、38.99→34.08秒、約 **1.14倍 / 7,691 tokens/s**。実際には必要なpack/quantizeが残る。一方、融合に伴うallocationやlaunchの削減波及はこの試算には含めていない。正確な改善率は実装後の同条件A/Bが必要。

## 4. 高速化候補リスト

番号は演算境界の改善を試す推奨順。時間は「触る処理群の現状費用」であって削減見込みではない。重複範囲や未分離費用を明記した。

| 順 | 候補・現在の余分な中間処理 | 実装方針 | 測れた関連費用 | 精度・メモリ上の条件 |
|---:|---|---|---|---|
| 1 | ReLU backward: 行優先BF16を作ってから2形式へ再pack | gate判定＋BF16丸めをdX/dW panel生成へ統合。必要ならタイル単位の同時生成とbias partialも比較 | encode 412.64＋BF16 A-pack 218.80＋転置A-pack 192.79 = **824.23 ms**。bias partialは別111.50 ms | `32768×1536×2` = **96 MiB**の中間を省く候補。丸め後の値でbiasを集計する順序、NaN・zeroのgate意味論を維持。panel自体は必要 |
| 2 | attention: QKV全体をFP32へ展開してQ/Kを再びBF16へ詰める | BFP8 payload/scaleからQ/K panelを直接作る。Vを読むPV/dP等とQ/Kを読むdQ/dKVもtyped loadへ移行し、全体FP32配列を不要にする | Q/K pack **468.31 ms**。全decode 587.38 msのうちQKVの分はprofile上未分離 | QKVのFP32一時配列 **192 MiB/呼出し**が対象。BF16で丸めた値をFP32で演算する現在の意味論を維持。typed decodeを各内積で繰り返すと逆に遅くなるためSLM/registerで再利用 |
| 3 | LayerNorm: residual和→norm→量子化、backward一時dx→scatter | typed residual/dropout読込をnormへ統合。forwardのBFP8出力生成、backwardの2枝加算を同じカーネルに統合する案を個別評価 | residual入力 **719.58 ms**、scatter **251.16 ms**。publication全体271.67 msの一部 | residual和/output/dxは各 **64 MiB**。FP32の和・分散・加算順とdropout seedを維持。既存 `BlockResidualNorm` は遅かったため、そのままONにする提案ではない |
| 4 | 非ReLU/CEの同じFP32勾配を2回読みBF16化 | 共有タイルから通常Aと転置Aの両panelを生成、または勾配を作るkernelのepilogueへpackを統合 | backward FP32 A-pack **516.20＋460.40 = 976.60 ms**（Linearとloss合計） | `dX` と `dW` では配列形式が異なるので単一panelの使い回しは不可。同時panelの保持によるピーク増とspillを計測。FP32勾配本体は必要な消費者のため保持 |
| 5 | dW向けactivationのB-packなど、GEMM直前の全体配列生成 | streamedタイル単位でproducer/consumerとpackを連結する。直接BFP8読込＋SLM packもA/B | backward `xmx_storage_pack_b_bfp8` **593.80 ms**。weightとactivation用途は未分離 | これは「全部weight pack」ではない。全体cacheで消せる数字として扱わない。小タイルでdecode回数が増えないことを確認 |
| 6 | forward出力BFP8→直後のLinearでBF16 A-pack | BFP8 publication時に、量子化後の値から次consumer用BF16 panelも作る。単一consumer・短寿命の境界に限定 | forward BFP8 A-pack **333.99 ms**。publicationと候補3に重複 | FP32計算結果を直接BF16で渡すのは禁止。必ずBFP8 payload/scaleを介した値と同じにする。panel保持でピークが悪化するため逐次比較 |
| 7 | single-consumer activation勾配も初回zero＋加算 | Arcにも「初回書込み/加算」の所有状態を設け、証明できる辺だけGEMMのoverwriteを選択。CUDAのexclusive FFN hintを参照 | backward `resident_zero` **206.42 ms**全体の一部 | FFN専用hintは現在CUDAだけが使用。residual、tied embedding、parameterの8回勾配累積では加算を残す。全zeroの削除は不可 |
| 8 | loss logitsの独立BF16丸めpass | GEMM epilogueで現在と同じBF16丸めを行い、CEが受け取るFP32値はそのまま維持 | **91.80 ms、1,024回** | BF16丸めの削除ではなく移動。bias加算後の丸め、NaN/Infとloss/gradient一致を検査 |
| 9 | embedding表全体decode＋embedding/dropoutの中間 | packed tableから直接gather。dropoutはpacked input/outputに対応。融合する場合も2つの量子化境界を再現 | 個別decode費用は未分離。embedding本体2.99 ms、dropout本体2.90 msは小さい | 表のFP32一時領域約 **26.46 MiB**、activation一時領域64 MiBが対象。embeddingのblock量子化を飛ばしてdropout後だけ量子化する変更は不可 |
| 10 | bias、gamma、beta等を呼出しごとに小さなFP32配列へdecode | consumer内typed load、または同一parameter generation内の小配列cache | 全decode **3,152回/587.38 ms**の一部。小parameter分の時間は未分離 | 8microbatch内は再利用可。optimizer/resume/precision変更で必ず無効化。大きなactivationまで同じcache方針にしない |
| 11 | loss dX chunkを全体dxへコピー、target chunkのD2D copy | GEMMにdestination offsetを渡して直接加算、CEにtarget offsetを渡す | dX `copy_range` **1.72 ms、512回**。target copyはGPU時間未分離 | 低優先。chunk tail、ignoreIndex、tied weight累積、既存split-K加算順を維持 |
| 12 | MuonのFP32実転置→BF16 panelとNS各積のpack | normalization/transposeと初回packを統合、更新時の逆転置を読み出し側で吸収。NS内の同じ値の2形式packも検討 | transpose **4.65 ms**、optimizer内pack **19.07 ms**。optimizer全体wall **199.74 ms** | NS5、FP32統計・master、confidence診断を維持。optimizerを全て無料にしても全体改善は約0.5%なので最後 |
| 13 | 同じweightを8microbatchで再pack | generation付き・予算付きの選択的cache、特に小parameterや高再利用の形式だけを再検討 | 全weight単独の時間は未分離。前回T1024の512 MiB cacheは **11,069.9 vs 11,078.1 tokens/s**で改善なし | **現状では不採用維持**。その試験ではpeak+320 MiB、112.52 MiB/updateのnative確保が再発。T2048で再試験する場合も全stepで判定 |

### 根拠となる実装位置

行番号は調査時のソース。各ファイルを開き、記載のメソッド/行を参照。

| 候補 | ファイル・行 |
|---|---|
| 1 | [Tensor.ArcPackedReluBackward.cs](../NNtrain.Core/Tensors/Tensor.ArcPackedReluBackward.cs):16,17,21,22,32 / [packed_relu_backward.cl](../NNtrain.Arc/Kernels/packed_relu_backward.cl):4 |
| 2 | [Tensor.ArcBatchedAttention.cs](../NNtrain.Core/Tensors/Tensor.ArcBatchedAttention.cs):158,164,184,189 / [Tensor.ArcResidency.cs](../NNtrain.Core/Tensors/Tensor.ArcResidency.cs):66 |
| 3 | [Tensor.ArcResidentOperations.cs](../NNtrain.Core/Tensors/Tensor.ArcResidentOperations.cs):107,118,123,125,129,131,135 / [norm_packed_input.cl](../NNtrain.Arc/Kernels/norm_packed_input.cl):8 |
| 4–6 | [Tensor.ArcPackedMatrix.cs](../NNtrain.Core/Tensors/Tensor.ArcPackedMatrix.cs):41,42 / [ArcXmxStorageOperand.cs](../NNtrain.Core/Optimization/ArcXmxStorageOperand.cs):51,54,125,126 / [Tensor.ArcFusedBfp8Linear.cs](../NNtrain.Core/Tensors/Tensor.ArcFusedBfp8Linear.cs):33 |
| 7 | [Tensor.ArcResidency.cs](../NNtrain.Core/Tensors/Tensor.ArcResidency.cs):119 / [Tensor.LinearLastDim.cs](../NNtrain.Core/Tensors/Tensor.LinearLastDim.cs):116,185 |
| 8,11 | [Tensor.ArcPackedLossHead.cs](../NNtrain.Core/Tensors/Tensor.ArcPackedLossHead.cs):58,62,118,120,121 |
| 9,10 | [Tensor.ArcResidentOperations.cs](../NNtrain.Core/Tensors/Tensor.ArcResidentOperations.cs):13,53,54,76,118,129 |
| 12 | [NekoMuon.Arc.cs](../NNtrain.Core/Optimization/NekoMuon.Arc.cs):34,40,55 / [ArcDirectXmxMath.cs](../NNtrain.Core/Optimization/ArcDirectXmxMath.cs):20 |
| 13 | [ArcMatrixPanelCache.cs](../NNtrain.Arc/ArcMatrixPanelCache.cs):10 / [Tensor.ArcResidency.cs](../NNtrain.Core/Tensors/Tensor.ArcResidency.cs):84,207 |

## 5. 「無駄なcast/transpose」ではなかったもの

1. **BF16→FP32→BF16の二重変換はBF16 panel packでは行っていない。** `xmx_storage_read_bf16` とvector版は `ushort` の読出し・格納であり、NaNのquiet化以外はbitを保つ。ReLUの問題は二重丸めではなく、中間配列と複数回の読み書き。[xmx_storage_pack.cl](../NNtrain.Arc/Kernels/xmx_storage_pack.cl):11、[xmx_storage_pack_linear_candidate.cl](../NNtrain.Arc/Kernels/xmx_storage_pack_linear_candidate.cl):16。
2. **BFP8→BF16変換そのものは現在の行列演算契約に必要。** scaleとの乗算をFP32で計算してBF16 RNEを行う処理は、全体FP32配列なしでpackに融合済み。これを生のsigned byteや別FP8形式の演算へ交換するのは同じ意味論ではない。
3. **attentionの `Transpose()` はstrideの交換だけ。** Tensor全体の並び替えkernelではない。[Tensor.ArcBatchedAttention.cs](../NNtrain.Core/Tensors/Tensor.ArcBatchedAttention.cs):9。
4. **matrix operandの `Slice()` はoffsetを持つborrow。** streamed GEMMの各chunkで元activation全体をコピーしていない。scale indexも元配列の位置を保つ。[ArcXmxStorageOperand.cs](../NNtrain.Core/Optimization/ArcXmxStorageOperand.cs):42。
5. **現在のmulti-head attentionはTensorのsplit/transpose/concat列を作っていない。** QKVからfused attentionへ直接渡す。旧fallbackの `attention_pack` も現在のT2048 profileには出ていない。
6. **大projectionのFP32出力→BFP8の別passは既に除去済み。** 現在は `gemm_xmx_bfp8_epilogue_16x64`。これを新規改善として数え直さない。
7. **loss重みpanelは512行ごとに再packしていない。** forward全chunkで1形式、backward全chunkで2形式を保持する。
8. **現行mixed gradientの非ReLU経路は独立 `matrix_gradient_bf16` を使わない。** `InlineMatrixGradient=true` でpackへ丸めを統合。ReLUだけはgate付き中間BF16を使う。
9. **softmax、LayerNorm統計、reduction、grad/masterのFP32は無駄な型変換ではない。** 特にFP32のP/dS/dYを単にBF16にしてXMXへ送ると精度契約が変わる。
10. **attention/lossのbackward再計算は保存容量との交換。** Q/K panelや全Pを32層分保持して再計算を消す案は、小さなVRAM増加では済まない。現在でもQ/KのBF16 panelは合計64 MiB/層で、32層を跨いで保持すると約2 GiB増える。丸ごと保存ではなく短寿命融合を優先する。

## 6. 型変換以外で、優先順位を誤らないための大きな所見

### 6.1 attentionのT2048経路

現在のattention GPU時間合計は24,861.79 ms（Q/K packを含むため3節とは重複）。特に:

- dK/dV: **5,122.01 ms**。
- softmax derivative: **4,877.73 ms**。
- dQ: **3,504.72 ms**。
- PV: **2,749.45 ms**。
- forward/backward probability: **4,274.78 ms**。
- dP: **2,206.14 ms**。

T512/T1024だけがregister-cache softmaxの対象で、**T2048はgeneric kernel**に戻る。[Tensor.ArcBatchedAttention.cs](../NNtrain.Core/Tensors/Tensor.ArcBatchedAttention.cs):31。generic確率kernelはscore読込、exp書込、正規化の再読込/書込、derivativeもP/dPの複数走査を行う。[attention_batched.cl](../NNtrain.Arc/Kernels/attention_batched.cl) の `attention_probabilities` / `attention_derivatives`。

64 MiB workspaceで現在head tileは2。attention関連だけで **295,424回/update**、全体は342,867回。候補はT2048向けexact-orderの行内再利用、T²中間を小タイル内に閉じる融合、head tile/workspaceのA/B。既存Flash/XMX実験は数値テスト未合格なので単に有効化しない。T1024で不採用だった融合も、T2048で速いと仮定せず改めて測る。

### 6.2 streamed GEMMは拡張タイル選択を共有していない

panel上限96 MiBを越えるshapeはstreamed経路へ入り、その `RunStreamedPanels` は8×32/16×32固定。通常経路の `ExpandedXmxTiles` の選択を使用しない。現在streamed kernel合計は **3,468.37 ms/update**。

これは無意味なcastではなく、**整形用bufferの上限によって別のGEMM実装が選ばれる**問題。streamedにも検証済みのタイル選択を対応させる案、panel予算の小幅変更、K分割のA/Bを行う価値がある。split-Kの既存partial順序とscratch上限は保持する。

根拠: [ArcXmxStorageOperand.Streamed.cs](../NNtrain.Core/Optimization/ArcXmxStorageOperand.Streamed.cs):8,112,117 / [ArcXmxStorageOperand.cs](../NNtrain.Core/Optimization/ArcXmxStorageOperand.cs):141。

### 6.3 allocatorの入替えが新条件で再発

- native allocation / release: **2,484個、各27.84 GiB/update**（累積量。27.84 GiB同時使用ではない）。
- native allocationのhost API時間: **279.40 ms/update**。GPU stallを含む排他的費用とは言えない。
- cache evictionも2,484個、27.84 GiB/update。native release等と同じbufferなので足さない。
- peak native ownership **9,147.80 MiB**。update境界active 1,486.70 MiB、cache 4,085.30 MiB。ドライバの専用VRAM使用量とは別指標。
- forwardの48 MiB allocationが264個、16 MiBが697個。48/16 MiBのBFP8 activationや対応scaleと一致するサイズ群が支配的だが、profileにはTensor IDがないため個別所有者まで断定しない。

poolは4 GiB、saved activation見積りは6,534.50 MiB、prefix/FFN checkpointは0/false。短寿命workspaceと長寿命saved activationが同じexact-size LRU poolで競合する。lifetimesを持つbuffer plan、activationとscratchのpool分離、予算内arena/slot再利用を検討する。poolを無制限に増やす提案ではない。

根拠: [ArcExecutionLane.cs](../NNtrain.Arc/ArcExecutionLane.cs):274,533,586,618 / [ArcExecutionOptions.cs](../NNtrain.Arc/ArcExecutionOptions.cs) `BufferPoolBytes`。

前回T1024の最終ベンチはnative確保/解放0だった。現在はcontextが変わったので、前回の「確保問題は解消済み」をそのまま流用できない。

## 7. 実装する場合の合格条件・順番

1. 現条件のp50/tokens/sを基準に固定し、候補1→2→3を独立A/B。その後4〜6を選ぶ。attentionのT2048行内再利用とstreamed GEMMタイルは、全体改善の別重点項目として扱う。
2. kernel単独の高速化だけで採用しない。同じbatch/context/accumulation、同一seedで全stepを測り、native allocation、peak ownership、H2D/D2Hも併記する。
3. 現行payload/scaleのbit一致、forward/gradient/updateの既存許容差を維持。NaN/Inf、gateの0、tail shape、block32/128、dropout seed、tied weights、8回累積を検証する。FMA/reduction順序が変わる案は「単なるcast除去」と呼ばない。
4. producer→consumerごとにTensor ID/shape、source/target dtype、panel形式、read/write bytesをprofileへ付与し、現在未分離のdecode/pack費用を分ける。現在の関数名だけの集計から個々の高速化率を捏造しない。
5. host wait/idle/driver影響を判定する場合はT2048で排他的timelineを追加取得。現在の重複するwait値を最大ボトルネックとして扱わない。
6. 候補7〜11は小さな補助改善、12は最後。13の大容量weight cacheは全stepの再ベンチで改善を確認するまで有効化しない。

参考: [前回の採用/不採用ベンチ一覧](arc-15000-target-2026-09-21.md)、[forward量子化epilogueと拡張GEMMタイル](arc-throughput-fusion-2026-09-21.md)。
