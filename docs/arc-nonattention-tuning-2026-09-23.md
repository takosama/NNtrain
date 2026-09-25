# Arc B580 2GPU 学習: Attention 以外の高速化調査（2026-09-23）

## 最終 A/B/A 結果

Arc B580 2枚で `training.transformer.json` の合成データ学習を測った。`mix8_16`、B16×T2048×累積8、32層・幅512・FFN1536・語彙11500で、1更新は262,144 tokens。A1/A2 は今回の7候補を全て無効、B は選択した7候補を既定設定で有効にした。各runは1 warmup＋3測定更新で、設定 SHA-256 と Core/Arc/Benchmarks の各バイナリ SHA-256 は全runで一致する。FineWeb 読み込み、tokenizer、checkpoint 保存は測定対象外。

| 経路 | 測定更新 | 同期 wall 中央値 | tokens/s |
| --- | ---: | ---: | ---: |
| A1: 7候補無効 | 3 | 15,775.53 ms | 16,617.1 |
| A2: 7候補無効 | 3 | 15,768.02 ms | 16,625.0 |
| A1＋A2: 6更新をプール | 6 | 15,771.78 ms | 16,621.1 |
| B: 7候補有効 | 3 | 15,218.40 ms | 17,225.5 |

同一バイナリの A/B/A で、選択した経路は **553.37 ms／更新短縮、tokens/s は3.64%増**。7候補は線形層 dual gradient pack、ReLU dual gradient pack、norm 後向き dX＋residual 累積融合、norm BF16直接出力、アンカー補正付き並列 norm reduction、BF16 logits キャッシュ、bias-only 勾配還元である。Attention の演算経路は変更していない。loss chunk は512行、logits／panel workspace は32／96 MiBを維持した。

| 更新 | A の loss | B の loss | A の gradient norm | B の gradient norm | gradient norm 相対差 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 9.349761 | 9.349764 | 0.075518854 | 0.07547713 | −0.05525% |
| 3 | 9.3129635 | 9.312968 | 0.07392036 | 0.07395657 | ＋0.04899% |
| 4 | 9.276874 | 9.276879 | 0.081232436 | 0.08118565 | −0.05760% |

loss の差は順に＋3×10⁻⁶、＋4.5×10⁻⁶、＋5×10⁻⁶。全更新で勾配は finite だった。この短期測定では長期の学習収束までは検証していない。GPU 0／1 の peak backend native ownership は A の5,726.9／5,194.7 MiBから B の6,091.1／5,558.9 MiBへ、各GPUで364.2 MiB増えた。これは backend の native allocation であり、driver が報告する VRAM 使用量ではない。各更新後の live allocation 差は両GPUとも0 B、H2D／D2H転送量は A/B で同一だった。

別途、B を1 warmup＋2更新で `--profile --timeline` 計測した。両GPUの各更新で trace coverage 100%、missing event 0、opaque allocation copy 0。1更新あたりの device event は GPU 0／1 で174,782／164,303件、初期 profile の176,598／166,119件よりそれぞれ1,816件少ない。GPU 0 の Attention 以外の kernel イベント時間は6,086.13→5,495.81 ms（−9.70%）、GPU 1 は5,978.88→5,384.79 ms（−9.94%）。GPU 0 の Attention kernel は9,212.41→9,199.26 msで、経路変更はない。kernel イベント時間は重複や待ちを含む wall の内訳ではないため、上の A/B/A の同期 wall を速度判断に使う。

| GPU 0 の排他的 wall 区分 | 初期 profile | 最終 B profile |
| --- | ---: | ---: |
| Attention | 9,210.9 ms | 9,199.6 ms |
| GEMM | 3,307.1 ms | 3,244.6 ms |
| pack／decode／publication | 1,509.1 ms | 1,128.2 ms |
| normalization／reduction | 1,002.3 ms | 865.6 ms |
| loss | 99.7 ms | 92.3 ms |

この表は各 profile のタイムライン上で排他的に割り当てた経過時間である。最終 profile の同期 wall 中央値は15,231.23 msで、上表の A/B/A 測定値とは別runのため直接混ぜない。最終テストは core の複数回の実行と checkpoint integration をテスト名で重複排除して **136件成功、失敗・skip とも0件**（core 134件、integration 2件）。最終の focused `core-r3` は6件すべて成功し、製品バイナリの SHA-256 は上記の A/B/A 測定時から変わっていない。

アンカー補正後の並列 norm を含む実機検証では、通常出力の最大相対 RMS は6.00416×10⁻⁵、勾配は5.04511×10⁻⁵で、判定上限5×10⁻⁴に収まった。テストは BF16 丸めだけでなく、行幅と勾配の最大寄与から見積もる FP32 和の誤差も許容範囲に含めている。定数行は厳密一致し、大きなオフセットの入力も許容範囲内だった。全ての norm 入力で bitwise 一致するという主張ではない。以前の norm 相殺誤差境界と参照 BFP8／BF16 比較条件による失敗はテスト前提を直して解消した。

同じ Release バイナリで A と B を再現する。出力 JSON は存在しない新しいパスを指定する。

```powershell
dotnet run --configuration Release --no-build --project .\NNtrain.Benchmarks -- `
  --probe-arc-transformer .\training.transformer.json .\benchmark-results\repro-nonattention-a.json `
  --warmup 1 --steps 3 --profile `
  --mix8-16-linear-dual-pack off --mix8-16-relu-dual-pack off `
  --mix8-16-norm-backward off --mix8-16-norm-output off `
  --mix8-16-parallel-norm off --mix8-16-cached-loss off `
  --mix8-16-bias-only-reduction off

dotnet run --configuration Release --no-build --project .\NNtrain.Benchmarks -- `
  --probe-arc-transformer .\training.transformer.json .\benchmark-results\repro-nonattention-b.json `
  --warmup 1 --steps 3 --profile
```

## 初期基準測定

選択候補を入れる前の同じ形状で `--profile --timeline` により1 warmup＋2更新を測った。設定 SHA-256 は `E876C55B7BFBE4F0BA5054D0A86070E525C776257E8C260A976A8F4305C1A50C`。

| 指標 | GPU 0 / 更新 |
| --- | ---: |
| 同期 wall p50 | 15,830.04 ms |
| tokens/s | 16,559.9 |
| Attention の exclusive wall | 9,210.9 ms |
| Attention 以外の exclusive GPU wall | GEMM 3,307.1 ms、pack/decode/publication 1,509.1 ms、normalization/reduction 1,002.3 ms、loss 99.7 ms |
| replica 同期 phase の wall | 約99.9 ms |
| GPU 0 device events / 更新 | 176,598、欠損 0 |

この表の exclusive wall 区分はタイムライン上の排他的な経過時間である。個々の GPU kernel のイベント時間や host-submit 時間を足して wall と比較しない。キュー待ちの `queue-delay-overlapping` は先行する GPU 作業と重複し、全イベントで合計すると実時間を大幅に超える。`partial-wait` も GPU 実行と重なる。最終採否には同じ設定・バイナリによる A/B/A の同期 wall と tokens/sを使い、kernel event、転送量、メモリ、loss／勾配を併せて確認した。

基準の非 Attention kernel では FFN GEMM 群が最も大きい。`norm_packed_residual_input` は275.8 ms／1024回、`norm_dx_row_sg16_w64_candidate` は156.6 ms／520回、`norm_row_sg16_w64_candidate` は145.7 ms／520回、`norm_residual_back_accumulate` は119.3 ms／512回。FP32 A-panel pack の通常／転置は236.0／214.1 ms、ReLU 勾配 pack の転置／bias は195.3／191.9 ms。これらは kernel の累積イベント時間で、wall の独立した節約可能時間ではない。

## 初期の3候補

1. **線形層の二方向勾配 pack**: 同じ FP32 勾配から dX 用と dW 用の BF16 XMX panel を1回の読み取りで作る。既存 loss head の dual pack は同様の構造を使う。`--mix8-16-linear-dual-pack on` で、通常／転置 pack の二重走査と起動回数を減らした。
2. **正規化の後向き融合**: norm dX と residual の2枝への勾配累積を1つの kernel にまとめ、中間 FP32 dX buffer と queue submit を減らす。residual/dropout 入力の再構築 `Input()` は従来通り残る。`--mix8-16-norm-backward on` を使う。
3. **正規化結果の BF16 直接保存**: `mix8_16` の BF16 activation 選択時、FP32 一時結果を作ってから BF16 に変換する手順を減らす。`resident_bf16` 全体は初期測定で138.0 ms／792回だったが、そのうち正規化出力分だけが対象である。`--mix8-16-norm-output on` を使う。

## 同一バイナリの初回 pilot

control と下記3候補は設定 SHA-256 `E876C55B...A86070E525C776257E8C260A976A8F4305C1A50C`、Core/Arc/Benchmarks の各バイナリ SHA-256 が一致する。1 warmup＋2測定の短い選別結果で、最終採否には冒頭の A/B/A と勾配検証を用いた。

| 候補 | 同期 wall p50 | tokens/s | GPU 0の関連pack kernelイベント合計 | GPU 0 peak native ownership |
| --- | ---: | ---: | ---: | ---: |
| control | 15,784.78 ms | 16,607.4 | 488.45 ms（通常＋転置＋loss dual） | 5,993,209,908 B |
| 線形層 dual pack | 15,630.89 ms | 16,770.9 | 317.42 ms（統合 dual） | 6,043,541,556 B |
| norm dX＋2枝累積 | 15,689.98 ms | 16,707.7 | 対象外 | 5,959,655,476 B |
| norm BF16直接出力 | 15,696.58 ms | 16,700.7 | 対象外 | 5,993,209,908 B |

線形層 dual pack の pilot では control より wall が153.89 ms短く、関連pack kernelイベント合計は171.03 ms減った。peak native ownership は50,331,648 B増えた。norm 後向き融合は control より wall が94.80 ms短く、`norm_dx_row_sg16_w64_candidate`＋`norm_residual_back_accumulate` のGPU時間275.97 msが、融合 kernel＋残り8回の既存 dX kernel で203.49 msとなった。norm BF16直接出力は wall が88.21 ms短く、`resident_bf16` 呼び出しが792→272回となった。4経路の2更新の loss はともに `9.349761`、`9.3129635` で一致した。kernelイベント差をそのまま wall の内訳とみなさない。

## dW GEMM タイルの単体測定

GPU 0 の常駐 BF16 panel を使い、dW の parallel-slice GEMM と finish を測った。3形状×7 kernel 変種×4 SplitK（1024、2048、4096、8192）の84組を各3回 warmup＋7回測定した。pack と reset は含まず、2回の勾配蓄積を既存の `32x32_wg16`／SplitK 2048 と比較した。表は GPU 時間の中央値で、形状ごとに実運用の基準と最速の候補を示す。

| dW 形状 M×N×K | 既存 `32x32_wg16`、SplitK 2048 | 最速候補 | GPU 時間短縮 | 精度 |
| --- | ---: | ---: | ---: | --- |
| 1536×512×16384 | 1.200833 ms | `32x32_wg8`、SplitK 2048: 1.199166 ms | 0.14% | bitwise 一致 |
| 512×1536×16384 | 1.192187 ms | `32x32_wg8`、SplitK 2048: 1.185937 ms | 0.52% | bitwise 一致 |
| 512×512×16384 | 0.473229 ms | `32x32_wg8`、SplitK 4096: 0.455520 ms | 3.74% | 相対 RMS 5.68×10⁻⁷、bitwise 不一致 |

SplitK 2048 の21組は基準と bitwise 一致した。他の63組は演算順序が変わり bitwise 不一致だが、相対 RMS は最大1.14×10⁻⁶だった。全84組で報告された spill memory は0 B。最速候補の差は最初の2形状では小さく、3つ目も実形状の256回／更新に換算すると約4.5 ms（全体の約0.03%）の短縮にとどまる。演算順序の変更は mix8_16 の速度方針で許容される。今回のタイルを据え置いた理由は、期待できる全体の短縮が小さいためである。測定は GPU 0 の単体 GEMM＋finish のみで、2GPU・pack・reset・optimizer を含む1更新の短縮を示さない。この結果から production のタイルは変更せず、現行の `32x32_wg16`／SplitK 2048 を維持する。再検討する場合は full-step A/B で同期 wall と数値差を確認する。

## 実機テストの確認

初回候補のテストは7件実行・7件成功、次の候補の再実行 `second-candidates-r2` は16件実行・16件成功。この2回では合計23件が成功した。`second-candidates` の旧 TRX はビルド失敗後に古いバイナリで走った0件の記録なので、検証判定から除外した。最終の重複排除したテスト結果は冒頭に記載した。

## 同一バイナリの第2回 pilot

`combined-control` は初回の3候補（線形層 dual pack、norm 後向き融合、norm BF16直接出力）を有効にした基準である。第2回はこれに ReLU dual pack、bias-only 勾配還元、BF16 logits キャッシュを1つずつ追加し、最後に全6候補を同時に有効にした。5経路は設定 SHA-256 と Core/Arc/Benchmarks の各バイナリ SHA-256 が一致し、同じ1 warmup＋2測定を実施した。

| 経路 | 同期 wall p50 | tokens/s | `combined-control` 比 | GPU 0 peak native ownership |
| --- | ---: | ---: | ---: | ---: |
| `combined-control` | 15,474.92 ms | 16,939.9 | — | 6,021,821,164 B |
| ＋ReLU dual pack | 15,307.68 ms | 17,125.0 | −167.24 ms | 6,021,821,164 B |
| ＋bias-only 勾配還元 | 15,454.58 ms | 16,962.2 | −20.35 ms | 6,021,821,164 B |
| ＋BF16 logits キャッシュ | 15,436.98 ms | 16,981.6 | −37.94 ms | 6,386,958,140 B |
| 全6候補 | 15,280.42 ms | 17,155.6 | −194.51 ms | 6,386,958,140 B |

ReLU 勾配の通常 bias pack＋転置 pack は GPU 0 のイベント時間平均で `192.959＋193.930＝386.889 ms`、統合 kernel は `234.590 ms` だった。約152.30 ms の kernel 時間削減があり、単独 pilot の wall も167.24 ms短い。bias-only 勾配還元は `gradient_rows` と新しい `gradient_rows_bias_bf16` の合計が `271.433→268.252 ms` で約3.18 ms短いが、wall の20.35 ms差は短い2更新測定の揺れを含む可能性がある。

BF16 logits キャッシュでは、GPU 0 の `cross_entropy_rows` が `61.43→29.82 ms`、関連する `gemm_xmx_direct_block_16x64_wg4` が `143.37→69.18 ms` となり、後向きの再計算を減らした。一方、保存／復元の `loss_head_pack_bf16_at` と `loss_head_unpack_bf16_at` は合計45.10 msで、GPU 0 の peak native ownership は365,136,976 B（348.2 MiB）増えた。単独 pilot の wall 差は37.94 msにとどまる。5経路とも2更新の loss は `9.349761`、`9.3129635`、gradient norm は `0.075518854`、`0.07392036` で一致した。kernelイベントの増減は wall と重なるため、表の wall 差と直接加算しない。

全6候補の15,280.42 ms／17,155.6 tokens/sは短い pilot の結果で、アンカー補正付き並列 normalization reduction を加えた最終 A/B/A は冒頭に記載した。

## loss chunk の比較

全6候補を有効にしたまま、loss head の chunk と logits／panel workspace を変えた。3経路は同じ設定 SHA-256 と Core/Arc/Benchmarks の各バイナリ SHA-256 で、1 warmup＋2更新を測定した。peak は各 GPU の native ownership である。

| loss chunk | logits／panel workspace | 同期 wall p50 | tokens/s | GPU 0／1 peak |
| ---: | ---: | ---: | ---: | ---: |
| 512行 | 32／96 MiB | 15,280.42 ms | 17,155.6 | 6,091.1／5,558.9 MiB |
| 1024行 | 64／96 MiB | 15,296.17 ms | 17,137.9 | 6,160.7／5,628.4 MiB |
| 2048行 | 128／128 MiB | 15,315.30 ms | 17,116.5 | 6,294.6／5,762.4 MiB |

1024行は512行より15.76 ms、2048行は34.88 ms遅く、peak も増えた。loss は各更新で最大約1×10⁻⁶の差、gradient norm は表示値で一致した。各測定更新の `LaneDelta.AllocatedBytes` は両 GPU とも0 Bで、更新をまたぐ live allocation の増加は見られない。この設定では拡大した chunk を採用せず、512行と32／96 MiB workspace を維持する。

## 並列 normalization reduction の暫定 pilot

アンカー補正前の並列 reduction 版を、同一バイナリの無効／有効で1 warmup＋2更新ずつ比較した。この組のバイナリ SHA-256 は互いに一致するが、上記の第2回 pilot とは別バイナリである。

| 経路 | 同期 wall p50 | tokens/s | GPU 0 forward norm kernel | GPU 0 backward norm kernel | GPU 0／1 peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| 並列 reduction 無効 | 15,288.45 ms | 17,146.5 | 155.678 ms | 201.212 ms | 6,091.1／5,558.9 MiB |
| 並列 reduction 有効 | 15,200.47 ms | 17,245.8 | 100.320 ms | 179.080 ms | 6,091.1／5,558.9 MiB |

有効時の wall は87.98 ms短く、forward norm kernel は55.36 ms、backward norm kernel は22.13 ms短い。一方、FP32 reduction の順序変更で数値は bitwise 一致しない。loss は更新2で `9.349761→9.349764`（＋3×10⁻⁶）、更新3で `9.3129635→9.31296`（−3.5×10⁻⁶）。gradient norm は更新2で `0.075518854→0.07549861`（−2.0244×10⁻⁵）、更新3で `0.07392036→0.073886506`（−3.3854×10⁻⁵）だった。このアンカー補正前の実機テストは17件中16件成功・1件失敗で、513行×幅512・residual alias・dropout 0.75 の勾配が BF16 許容差を超えた。平均計算をアンカー補正した改訂版を別バイナリで測定・検証し、冒頭の最終結果に採用した。

## 候補の制約と不採用の経路

- **loss head の BF16 logits 保持**: 前向きの BF16 値に丸めた logits を物理 BF16 で保存し、後向きの再計算を省く。実形状で保存量は `16384×11500×2 = 376,832,000 B`（359.4 MiB）/GPUと FP32 row stats 128 KiB。512 MiB 上限、現在の live allocation、GPU 容量、1 GiB の後向き scratch reserve で admission を制限する。`--mix8-16-cached-loss on` を含む最終 B の peak native ownership と loss／勾配差は冒頭に示した。
- **loss chunk**: 512／1024／2048行の比較は上記のとおり。`LossChunkRows` だけを増やすと FP32 logits workspace 32 MiB によって実際の上限は729行になるため、測定では `--loss-logits-mib` も拡大した。dual gradient pack では通常／転置の2つの A-panel と dW 用 B-panel が同時に生き、2048行は96 MiB panel 上限を超えるため128 MiBにした。
- **GPU 間通信と embedding**: 更新後の packed weight 同期は約100 ms／更新で、全 wall の約0.63%。91,031,788 parameter の BF16 勾配は182,063,576 B（173.6 MiB）で、GPU 1からの読み出しとGPU 0への転送が必要。既存の勾配転送 pipeline は opt-in であり、通信短縮の上限は GEMM／Attention より小さい。embedding forward/backward は計約4.6 ms／更新なので、この設定では後回しにする。

資料: [基準 profile](../benchmark-results/arc-nonattention-baseline-profile-20260923.json)、[基準要約](../benchmark-results/arc-nonattention-baseline-summary-20260923.json)、[pilot control](../benchmark-results/arc-nonattention-pilot-control-20260923.json)、[pilot linear dual](../benchmark-results/arc-nonattention-pilot-linear-dual-20260923.json)、[pilot norm backward](../benchmark-results/arc-nonattention-pilot-norm-backward-20260923.json)、[pilot norm output](../benchmark-results/arc-nonattention-pilot-norm-output-20260923.json)、[dW タイル単体測定](../benchmark-results/arc-nonattention-dw-tiles-20260923.json)、[初回テスト](../benchmark-results/test-results/arc-nonattention-initial-candidates-20260923.trx)、[次候補の再実行テスト](../benchmark-results/test-results/arc-nonattention-second-candidates-r2-20260923.trx)。第2回 pilot: [combined-control](../benchmark-results/arc-nonattention-combined-control-20260923.json)、[ReLU dual pack](../benchmark-results/arc-nonattention-combined-relu-dual-20260923.json)、[bias-only](../benchmark-results/arc-nonattention-combined-bias-only-20260923.json)、[BF16 logits キャッシュ](../benchmark-results/arc-nonattention-combined-cached-loss-20260923.json)、[全6候補](../benchmark-results/arc-nonattention-all-six-20260923.json)、[chunk 1024](../benchmark-results/arc-nonattention-chunk1024-20260923.json)、[chunk 2048](../benchmark-results/arc-nonattention-chunk2048-20260923.json)。並列 norm のアンカー補正前: [control](../benchmark-results/arc-nonattention-parallel-norm-control-20260923.json)、[有効](../benchmark-results/arc-nonattention-parallel-norm-on-20260923.json)、[実機テスト](../benchmark-results/test-results/arc-nonattention-parallel-norm-20260923.trx)。

最終資料: [A1](../benchmark-results/arc-nonattention-final-a1-20260923.json)、[B](../benchmark-results/arc-nonattention-final-b-20260923.json)、[A2](../benchmark-results/arc-nonattention-final-a2-20260923.json)、[profile／timeline](../benchmark-results/arc-nonattention-final-profile-20260923.json)、[profile 要約](../benchmark-results/arc-nonattention-final-summary-20260923.json)。テスト: [core 広域](../benchmark-results/test-results/arc-nonattention-final-core-20260923.trx)、[core 再実行1](../benchmark-results/test-results/arc-nonattention-final-core-r2-20260923.trx)、[core 最終再実行](../benchmark-results/test-results/arc-nonattention-final-core-r3-20260923.trx)、[checkpoint integration](../benchmark-results/test-results/arc-nonattention-checkpoint-20260923.trx)。
