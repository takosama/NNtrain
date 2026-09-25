# Arc B580 2GPU Transformer 学習の高速化（2026-09-23）

## 対象と仮説

`training.transformer.json` の Transformer、B16×T2048×累積8、32層、幅512、16 heads、FFN1536、`mix8_32` block32、Muon NS5 + AdamWを対象にした。1更新は262,144トークン。2枚の Arc B580（driver `32.0.101.9030`）で固定シードの合成入力を使用し、FineWeb、tokenizer、checkpoint、metrics、生成のI/Oは計測から除外した。各測定は1 warmup＋2更新で、数値は同期wall timeのp50。

従来の2GPU実装は更新後、GPU 0のFP32 masterから`state_dict`を作り、GPU 1へ`load_state_dict`していた。GPU 1は前後計算だけを担当し、optimizerやcheckpointを保持しない。このため更新後に公開された演算用のBFP8 payloadとscaleだけをGPU 1に渡せば、次更新の計算を実行できると仮定した。初期同期と完全な状態復元には従来の経路を残した。

## 採用した変更

- 各parameterの公開済み演算用重みをGPU 0からホスト経由でGPU 1へ転送する。BFP8、BF16、FP32を扱い、GPU 1の既存bufferと型別のホスト中継配列を再利用する。
- GPU 1のweight panel cacheとデータ世代を更新し、古いpanelを使わない。GPU 1のFP32 masterは破棄する。GPU 1はoptimizerとcheckpointの主体ではない。
- 勾配集約で、GPU 1からダウンロード済みの`float[]`をGPU 0へアップロードする際の余分な`ToArray()`を除いた。
- ベンチに`--replica-sync full|packed`を追加した。通常の2GPU学習とベンチの既定は`packed`。

## 同一バイナリ A/B/A

3本の結果は同じ設定SHA-256とCore/Arc/BenchmarksバイナリSHA-256を持つ。変更したのは`--replica-sync`だけ。1更新あたりの転送量は計測2回目のlane delta。

| 経路 | 更新p50 | tokens/s | 同期平均 | GPU 0 D2H | GPU 1 H2D |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全状態 A | 19,328.55 ms | 13,562.5 | 660.79 ms | 466,542,076 B | 467,586,492 B |
| 圧縮済み B | **18,613.50 ms** | **14,083.5** | **91.52 ms** | **102,413,368 B** | **103,459,340 B** |
| 全状態 A再測定 | 19,325.20 ms | 13,564.9 | 630.19 ms | 466,542,076 B | 467,586,492 B |

全状態2本のp50平均に対して、圧縮済み同期は**713.37 ms/update（3.69%）短縮**し、p50換算のthroughputは約3.83%増えた。GPU 0 D2Hは364,128,708 B、GPU 1 H2Dは364,127,152 B減った。同期以外にも、GPU 1の次更新での重み再アップロードが消え、前後計算・勾配集約phaseが短くなった。転送・kernelの累積カウンタをwall timeへ加算していない。

全更新のlossと勾配normは有限。全状態Aと圧縮済みBの同じ更新番号では、lossの最大差は約`4e-6`、勾配normの最大差は約`2.1e-5`だった。公開済みGPU量子化重みをそのまま複製するため、CPU上でFP32 masterを再量子化する従来経路との丸め差はあり得る。

結果: [全状態 A](../benchmark-results/arc-dual-sync-full-a-20260923.json)、[圧縮済み B](../benchmark-results/arc-dual-sync-packed-b-20260923.json)、[全状態 A再測定](../benchmark-results/arc-dual-sync-full-a-repeat-20260923.json)。

## 試して採用しなかった候補

| 仮説 | 同じ262,144トークン/更新の結果 | 判断 |
| --- | --- | --- |
| microbatch B32×累積4 | 20,166.99 ms | B16×累積8より遅い。 |
| microbatch B8×累積16 | 19,786.57 ms | B16×累積8より遅い。 |
| `mix16_32` | 19,416.47 ms、peak backend GPU 0 約7,865 MiB | `mix8_32`より遅く、数値軌道も変わる。 |
| streamed XMX tile拡大 | 19,321.68 → 19,319.54 ms | 現行の2GPU形状はstreamed GEMMを実行せず、効果なし。 |
| dK/dV K16/Q32 SLM tile | resident T2048 causal GPU 0.2814 → 0.4805 ms | ビット一致だが70.8%遅い。実験コードを撤去。 |

結果: [B32×4](../benchmark-results/arc-train-dual-b32-a4-candidate-20260923.json)、[B8×16](../benchmark-results/arc-train-dual-b8-a16-candidate-20260923.json)、[mix16](../benchmark-results/arc-train-dual-mix16-candidate-20260923.json)、[streamed拡大](../benchmark-results/arc-dual-opt-expanded-streamed-20260923.json)、[K16](../benchmark-results/arc-dkv-k16-b580-20260923.json)。これらの数値は候補選別用で、異なるshape・精度のlossを同等と主張しない。

## 検証と範囲

- Release solution build: 警告0、エラー0。
- `ArcDataParallelEngineTests`: 9/9合格。圧縮済み同期後の連続する2更新を従来の全状態同期と比較し、`float32`、`mix16_32`、`mix8_32`でlossと全parameter勾配が各要素1e-5以内。既存の2GPU勾配、dropout再開、完全状態同期も合格。
- `ArcTrainingIntegrationTests`: 20/20合格。2GPU FineWeb小規模学習、保存・再開、生成、単GPU経路を含む。
- GPU 1の内部FP32 masterは圧縮済み同期の後に保持しない。チェックポイントはGPU 0から保存し、開始時は従来の全状態同期を使う。
- この短い合成ベンチは実データ供給・長時間学習・収束品質を測っていない。backendのpeak allocated bytesはdriver全体のVRAM使用量ではない。
