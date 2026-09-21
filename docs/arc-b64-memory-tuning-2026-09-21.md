# Arc B64: メモリ寿命・型付きGEMM・LayerNormの実測改善（2026-09-21）

## 結論

本来のJSON条件 **B64・accumulation 4・32層**で、step p50は **70.45555 s → 34.10463 s**、**2.066倍・時間51.59%減**。2.5倍目標（28.18222 s以下）は未達で、さらに約17.4%の時間削減が必要。採用経路は通常起動で有効になり、JSONは変更していない。

基準は1 warmup＋3測定。まず改善版を1＋10測定し34.68556 s、その後softmax register cacheを追加した最終版を1＋5測定し34.10463 sとなった。生JSONはすべて保持した。これを長時間学習・収束の改善やCUDA同等性能の証明とはしない。CUDA実機比較・長時間soak・実コーパスでの収束は未検証。

候補選定用のB64・accumulation 1比較は17.9304055→8.83895945 s。accumulation 4の70.45555 sをaccumulation 1の8.83896 sで割るような、実効batchの違う比較はしていない。

## 固定条件と測定範囲

GPUはIntel Arc B580、driver `32.0.101.9030`。設定は `training.transformer.json`、SHA256 `EC42E94F3A9BA43DBDB90E5A44368798C422F76AD5330BC936F701261E7D64C5`。各JSONにbinary SHA256、変更した引数、実際のArc optionsを保存した。既定値を変更した後に過去条件を再現する場合は、ファイル名ではなく保存された `ArcOptions` も固定する必要がある。

共通形状はmicrobatch 64、sequence 1024、width 512、heads 16（D32）、hidden 1536、32層、vocabulary 11,500、dropout 0.1、tied embedding、90,507,500 parameters。精度は `mix8_32`、BFP8 block 32。Muon（LR 0.001、固定NS5、毎step）＋AdamW（LR 0.0003）、weight decay 0.01、clip norm 1、seed 1234。8層への縮小試験は表中で明示する。

固定synthetic full-length tokenによるforward/backward/clip/optimizerを計測した。モデル・optimizerはCLI factory経由。tokenizer、dataset reader、checkpoint保存、評価、生成、LR schedule、HTML更新は対象外。実データのpadding率や転送以外のI/O待ちはこの数値に含まれない。

`StepP50Ms`は同期を含むwall time。`KernelGpuMs`はOpenCL event duration。allocation時間にはupload、transfer時間には先行kernel待ちが含まれるため、これらを単純に足し合わせてstep時間の分解にしてはいけない。Amdahlのphase比率は同期したphase wall timeを使う。

## 原因: 保存activationがデバイス容量より大きい

問題はweightだけではない。shape-based plannerの見積もりでは、32層分のQKV、attention output、residual/LN、FFN中間値およびFP32 statisticsを全保持すると、保存activationだけで **13,703,839,744 bytes = 12.7627 GiB**。さらにFP32 master/gradient/optimizer moments等のpersistent state約1.44 GiBと、GEMM/attentionの作業領域が必要になる。

これは推定量でありphysical VRAMの測定値ではない。ただし実際のwarm baselineのOpenCL buffer会計ピークも **16,103.90 MiB** に達し、12 GiB級GPUの容量を超える保持要求が存在した。driver paging量・PCIeページフォルト量は未計測なので、「何GiBがホストへページングされた」とは断言しない。

小さい512 MiB poolと巨大な一時領域の組合せでは、acc4 baselineで1 update当たり `clCreateBuffer` 要求量が合計 **684,387,261,556 bytes（637.39 GiB）**、native allocation 7,767回だった。この数値は同時使用量ではなく、再確保を含む累積churn。学習後にbufferを解放していても、毎stepの再確保コストと作業セット圧力は残る。

## 今回の採用経路

### 1. 型付きstorageからのstreamed XMX GEMM

従来のfull-panel条件はA/B pack合計96 MiB以下。B64の大行列でこの条件を外れると、高速な型付きpanel経路から外れることがあった。

`ArcXmxStorageOperand.Streamed.cs`はFP32/BF16/BFP8のresident storageとscale原点を保持したsliceを使い、96 MiB以内のpanelで順次実行する。通常のlinear/dXは行方向、dWはtoken方向の連続sliceを扱う。原則最大16,384行/縮約要素で区切り、出力は元bufferのoffsetへ直接書く。全体のhost複製やFP32 decode用の巨大な中間tensorは作らない。元のBF16 operand丸め、FP32 accumulation、gate、epilogue、split-reduction条件を保持する。

8層の短時間試験は3.95486035 s → streamed 3.5607038 s。packed residual入力を併用すると3.5099567 s。異なる層数の32層結果とは直接比較しない。

### 2. 演算順を変えないSG16 LayerNorm

元の `norm_dx_set` は1 work-itemが1 rowを直列処理するため、同時実行lane間のloadがwidth間隔になっていた。単純なparallel reductionは和順を変えるため採用しない。新経路は1 SG16で1 rowをcoalesced loadし、shuffleで元のchannel 0→width−1順に値を渡す。元のFP32式・積和順は保持する。

採用はworkgroup 64（4 rows/WG）、rows≥128、width64〜2048、SG16/XMX capability条件付き。対応外は従来kernelへ戻る。追加global scratch、SLM、workgroup barrierはいずれも不要。

`arc-norm-coalesced-20260921.json`、65,536×512、3 warmup＋10測定のwall p50:

| 演算 | 元のserial | column-major転置候補 | 採用SG64 | SG128候補 |
|---|---:|---:|---:|---:|
| forward | 5.91280 ms | 3.97655 ms | 1.09380 ms | 1.09085 ms |
| backward dX | 48.09845 ms | 4.93170 ms | 1.14455 ms | 1.15445 ms |

column-major案はforward128 MiB/backward256 MiBのscratchと転置が必要だったため不採用。SG64はdX単体で約42倍だが、これを学習全体の倍率と呼ばない。probeは30 fixturesおよびproduction行列で出力・stats・反復加算結果を全配列bit比較した。`ArcOrderedNormTests`は3精度の2-stepモデル比較、packed入力との併用、shape/capability gateを検査し、既存の`1e-4`許容誤差を維持する。

### 3. Packed residual入力

`PackedNormInput`はBFP8/BF16/FP32のstorageから、dropout＋residualのFP32和bufferへ直接decodeする。入力2本をそれぞれ全面FP32 materializeする中間bufferを省く。storage丸め→dropout→加算の順は従来通りで、LayerNorm統計の意味も変えない。

### 4. Q/Kのdevice-side packとdirect DPAS

Q/KのBF16 RNEとpadded-32縮約順を維持し、同一値の再pack・SLM経由を減らした。pack、allocation、releaseを含めて比較した。

`arc-qk-direct-20260921.json`、B64/H16/T1024/D32、causal、全head分、3 warmup＋10測定:

| QK経路 | wall p50 | GPU p50（pack含む） | pack GPU p50 |
|---|---:|---:|---:|
| `attention_xmx_panel` | 10.29985 ms | 9.70698 ms | 0 |
| direct128×64 | 7.80230 ms | 6.10343 ms | 2.47250 ms |
| direct128×32 | 7.82550 ms | 6.21353 ms | 2.40010 ms |
| 採用direct256×32 | 7.53005 ms | 5.79286 ms | 2.47953 ms |

device panelは8-head tile当たり1 MiB、採用kernelはSLM/spillとも0。wall時間は約26.9%短縮。tail、batchをまたぐhead、反復、accumulate、non-finite/signed zeroもprobeの全配列検査対象。通常のFP32 attention経路をBF16へ変更する最適化ではない。

### 5. 必要な層だけ再計算し、poolで再利用

FFN再計算は最も大きいhidden activationを比較的小さい追加計算で除去する。なお足りない分だけ先頭prefixをfull-block checkpointにし、attentionの無用な再計算を抑える。checkpointのdropout seedを記録・再生し、入力gradientの加算順とparameter version検査を保持する。full-block内でFFN checkpointを重ねない。ここでのcheckpointはactivation再計算であり、学習状態のディスク保存とは別物。

今回のB580が報告するglobal memory量でauto plannerを実行した実測値は **prefix 11層＋FFN再計算**。手動8層候補や仮の12 GiB値から得る10層ではない。保存activation見積もりは12.7627 GiB → **6.5840 GiB**、activation budgetは6.6642 GiB。autoはbatch、accumulation、LR、dtypeを変更せず、graphの保持/再計算だけを選ぶ。これはfree VRAM queryでも、他プロセスを含めたOOM回避保証でもない。

poolは4 GiBへ拡大。8 GiB候補は4 GiB比でwall改善が約0.23%しかなく、余分な保持を正当化しないため既定値にはしない。LRUとmemory budgetでidle cacheだけをtrimし、GPU完了待ちのretired bufferを必要に応じてfenceする。live tensorを勝手に解放/host退避する処理ではない。

## B64実測一覧

表の32層acc1候補は同一モデル/実効batchで比較できる。ただし短いrunであり、異なるwarmup回数やcold runは分けて解釈する。メモリ列は全て`LaneSnapshot.PeakAllocatedBytes`、すなわち**live＋cached＋retiredのOpenCL buffer会計ピーク**で、physical VRAMではない。

| JSON（`benchmark-results/`） | 層 / acc | warmup＋測定 | step p50 s | 会計peak MiB |
|---|---:|---:|---:|---:|
| `arc-b64-acc4-baseline-20260921.json` | 32 / 4 | 1＋3 | 70.4555489 | 16,103.90 |
| `arc-b64-memory-baseline-full-20260921.json`（cold） | 32 / 1 | 0＋1 | 18.0986452 | 14,748.38 |
| `arc-b64-memory-baseline-warm-20260921.json` | 32 / 1 | 1＋2 | 17.9304055 | 16,103.90 |
| `arc-b64-memory-baseline-short-20260921.json` | 8 / 1 | 1＋2 | 3.95486035 | 5,354.00 |
| `arc-b64-streamed-short-20260921.json` | 8 / 1 | 1＋3 | 3.5607038 | 5,041.98 |
| `arc-b64-streamed-packed-short-20260921.json` | 8 / 1 | 1＋3 | 3.5099567 | 5,041.98 |
| `arc-b64-three-paths-full-20260921.json`（stream/packed/ordered） | 32 / 1 | 1＋2 | 13.53901665 | 15,791.89 |
| `arc-b64-checkpoint16-full-20260921.json`（pool512 MiB） | 32 / 1 | 1＋2 | 10.8481304 | 9,887.89 |
| `arc-b64-checkpoint16-pool4096-20260921.json` | 32 / 1 | 1＋2 | 9.08932785 | 10,554.49 |
| `arc-b64-checkpoint8-pool4096-20260921.json` | 32 / 1 | 1＋2 | 10.041586 | 13,506.49 |
| `arc-b64-hybrid8-pool4096-20260921.json` | 32 / 1 | 1＋2 | 8.91084565 | 11,022.49 |
| `arc-b64-hybrid8-pool8192-20260921.json` | 32 / 1 | 1＋2 | 8.8900897 | 11,022.49 |
| `arc-b64-hybrid8-direct-qk-20260921.json` | 32 / 1 | 1＋2 | 8.7087575 | 10,676.88 |
| `arc-b64-auto-memory-20260921.json`（prefix11＋FFN、自動） | 32 / 1 | 1＋2 | 8.83895945 | 10,239.49 |
| `arc-b64-acc4-final-20260921.json`（softmax追加前） | 32 / 4 | 1＋10 | 34.6855644 | 10,239.49 |
| `arc-b64-acc4-softmax-final-20260921.json`（最終既定値） | 32 / 4 | 1＋5 | 34.1046284 | 10,239.49 |

手動hybrid8の8.7088 sがauto11の8.8390 sより僅かに速いが、autoは小さい保持見積もり・予算余裕を優先する。手動8層を全shapeに固定する判断はしない。

## 転送とメモリcounterの読み方

warm updateの明示的host↔device転送は次の通り。

| 条件 | H2D/update | D2H/update |
|---|---:|---:|
| B64 / acc4 baseline | 2,097,152 bytes | 2,588 bytes |
| B64 / acc4 最終版 | 2,097,152 bytes | 2,588 bytes |
| B64 / acc1・32層のwarm各候補 | 524,288 bytes | 2,576 bytes |
| B64 / acc1・8層候補 | 524,288 bytes | 656 bytes |

H2Dは`64×1024×2×sizeof(int)×accumulation`でtoken/targetと一致する。D2Hはloss、clip、既存Muon統計等の小さいdiagnosticであり、weights/activation/gradient全体を返していない。probeの追加finite検査はstep計測・LaneDelta取得後に実施するため、この表には含まない。ただし現在のMuon診断にはparameter数依存の少量readbackが残るので、「全モデルに対してscalar定数個だけ」とは表現しない。これらはAPIで観測する明示copyのcounterであり、driver内部のmigrationまで含む帯域計ではない。

auto acc1 runの各step完了時はlive約1,478.14 MiB、cached約4,056.44 MiB、retired 0。会計peakは10,239.49 MiBで2測定step間で増加なし。1 update当たりnative新規要求量は約4.864 GiB（約272.5回）で、同量がほぼ解放された。これは長時間leak不在の証明ではない。古いJSONに存在しない `NativeAllocatedBytes` 等は未計測であって0ではない。

`PhysicalBufferBudgetBytes`というoption名でも、使用している情報はdriver報告global capacityとOpenCL buffer会計。既定0はreported capacityの90%をcache trim用目安にするだけで、現在のdedicated VRAM使用量を計測しているわけではない。プロセスworking setもVRAMとして掲載しない。

## 不採用: dP＋dS register融合

`arc-dp-ds-register-20260921.json`。D32/T≤1024/SG16限定で、dPをregister保持しdSへ直接変換。元のchannel FMA順と64-lane derivative reduction treeを保持した。15 tail/causal fixtures＋productionの全配列bit比較は合格、SLM 8,320 bytes、spill 0、追加global scratch 0。

T1024、8 heads、causal mode2、3 warmup＋10測定:

| 経路 | wall p50 | GPU p50 |
|---|---:|---:|
| 現採用tuned dP＋既存derivatives | 0.21350 ms | 0.1821340 ms |
| register融合候補 | 0.20325 ms | 0.1799995 ms |

GPU時間の改善は約1.17%。対象の全体割合を約10%とみても、Amdahlで全体改善は約0.12%に留まる見込み。短時間wall差だけで通常dispatchを増やす根拠にはしない。候補kernel/probeは保存し、**既定経路には採用しない**。

## 追加候補: softmaxのrow register cache

`arc-row-register-20260921.json`。T512/1024だけ、各laneの8/16要素をregisterに保持する。64-lane reduction treeとFMA順は変更せず、probabilityのglobal読込3回・書込2回を読込1回・書込1回へ減らす。12 fixturesでfresh/saved probability、stats、derivative、2回加算した下流gradientを全配列bit比較した。masked signaling NaN、未書込み領域も検査。SLMは従来どおり256 bytes、spill 0。

T1024・causal、3 warmup＋10測定のGPU p50は、fresh probability 0.0856→0.0719 ms、saved probability 0.0714→0.0634 ms。通常経路への候補はprobabilityだけで、`CachedAttentionProbabilities`により個別比較できる。SG16の確認済みdevice・T512/1024に限定し、明示的subgroup指定や未対応sequenceは元の経路を維持する。

derivativeも同じ方針で読込を削ったが、0.1379→0.1367 msに留まったため通常dispatchには採用しない。最終モデル全体の追加効果は下の最終検証欄で評価する。

## 次に見る区間とAmdahl

auto acc1 runのphase平均はbackward 6.6077 s（74.76%）、forward 1.9983 s（22.61%）、optimizer 0.2034 s（2.30%）。この状態でoptimizerを理想的に0にしても全体倍率の上限は約1.024倍。backwardを2倍にすると理論上の全体倍率は約1.597倍である。

同runの最初の測定stepにはattention dKV 889.16 ms、derivatives 610.53 ms、dP 253.65 msが残る。一方norm dXは74.75 msまで下がった。次の候補はこの更新後のprofileから選び、既に十分小さくなった区間だけを反復しない。profileの `queue-delay-overlapping` は複数kernelで重複する待機時間なので、合計して追加の直列costと見なさない。

## 最終モデル全体の結果

| 指標（B64 / acc4） | 基準 | 最終既定値 |
|---|---:|---:|
| step p50 | 70,455.55 ms | 34,104.63 ms |
| step mean | 70,565.20 ms | 34,189.13 ms |
| tokens/s | 3,720.70 | 7,686.46 |
| forward mean | 18,459.67 ms | 7,700.89 ms |
| backward mean | 51,719.44 ms | 26,257.82 ms |
| optimizer mean | 355.85 ms | 201.60 ms |
| native allocation回数/update | 7,767 | 1,009.6 |
| native新規確保の累積量/update | 637.39 GiB | 19.40 GiB |
| allocation wall/update | 2,460.54 ms | 174.95 ms |
| OpenCL buffer会計peak | 15.7265 GiB | 9.9995 GiB |
| 更新後active | 1,478.14 MiB | 1,478.14 MiB |
| 更新後cached | 351.43 MiB | 4,056.44 MiB |
| 更新後retired | 0 | 0 |

最終5測定は34,536.40 / 34,118.78 / 34,104.63 / 34,096.14 / 34,089.72 ms。初回の高めの値も除外せず保存・平均に含めた。softmax追加前の10測定と最終5測定の両方で、初回測定から最終測定までactive/peak増分は0 bytes。すべてloss/gradientは有限。ただし5〜10 updatesの安定性は長時間leak不在の保証ではない。

基準と最終版の共通3測定stepではloss最大差0.000016、grad norm最大差0.000114965。全モデルの学習系列がbit一致するとの主張ではない。下位kernelのbit比較、およびモデル/optimizerの既存許容誤差による回帰テストは別途実行し、許容誤差は広げていない。

最後のsoftmax追加はモデル全体で約1.70%短縮。backwardは最終wallの76.8%、optimizerは約0.59%。optimizerだけをゼロにしても全体は約1.006倍なので、2.5倍到達の次の対象はAttentionとGEMM/packing。これをハードウェアの限界とは断定しない。

小さいshapeの退行確認も実施。B16 / acc4 / その他同条件の最終版は1 warmup＋3測定で **7,515.90 ms/update**（前回保存値8,268.37 ms比1.100倍）。autoはprefix0、FFN再計算なしを選び、buffer peakは5,285.87 MiB。H2D524,288 / D2H2,588 bytes/update。B64とtoken数の違うupdate時間を直接比較しない。

- [凍結基準](../benchmark-results/arc-b64-acc4-baseline-20260921.json)
- [改善版10測定](../benchmark-results/arc-b64-acc4-final-20260921.json)
- [softmax追加後・最終5測定](../benchmark-results/arc-b64-acc4-softmax-final-20260921.json)
- [B16退行確認・最終3測定](../benchmark-results/arc-b16-softmax-final-20260921.json)

## 回帰検証・再現

既定値は `StreamedXmxMatrices` / `DirectAttentionQk` / `CachedAttentionProbabilities` / `OrderedTiledNorm` / `PackedNormInput` / `AutomaticTransformerMemoryPlan` がtrue、`BufferPoolBytes` が4 GiB。明示checkpoint指定はautoより優先する。CPU/CUDAモデル式、JSONのbatch/accumulation/LR/dtypeは変更していない。

- solution Release build: warning 0 / error 0。
- CoreのArc関連: 328 passed / 0 skipped / 0 failed。float32・mix16_32・mix8_32、grad/update、転送、解放、反復backward、dropout/Train-Eval切替、shape fallbackを含む。
- CoreのArc以外: 1,195 passed / 109 skipped / 0 failed。
- Integration全体: 427 passed / 6 skipped / 0 failed。
- Benchmarks.Tests: 35 passed / 0 skipped / 0 failed。
- Float16 operation manifest: 9 passed（CPU）。内部helper追加の厳密inventoryを更新。既存の未登録Arc loss/frozen linear/DPOも、非対応理由または実際の精度契約を明示した。

CUDA実機が使えないためCUDAテストは検証済みに数えない。GPU不在のまま実行してerror 801となっていた既存3テストには明示skip条件を追加し、ABIのみの検査を残した。CUDA実行のassertionや数値許容誤差は削除していない。

```powershell
dotnet build .\NNtrain.slnx -c Release --no-restore
dotnet run --project .\NNtrain.Benchmarks\NNtrain.Benchmarks.csproj -c Release --no-build -- `
  --probe-arc-transformer .\training.transformer.json .\benchmark-results\arc-b64-recheck.json `
  --warmup 1 --steps 5 --profile
```

出力は未使用のファイル名にする。元の学習JSON、dataset、tokenizer、checkpoint、metricsはprobeで変更しない。個別A/Bには`--streamed-xmx`、`--direct-qk`、`--cached-prob`、`--ordered-norm`、`--packed-norm`、`--auto-memory`のon/offと`--pool-mib`を使用する。

通常の学習コマンドは従来通り。今回の追加はArc sessionの既定値なので、別の起動フラグは不要。
