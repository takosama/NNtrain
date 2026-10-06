# 画像読み込み高速化 — 2026-10-05

これは初回の高速化の記録。以降の採用版は [画像から最初の文字までの高速化](vision-ttft-v2-2026-10-05.md) を参照。

## 採用した設定

GUI・NNtrain 内蔵 API の `InferenceSession` に反映済み。

- Vision: FP32 タイル演算。Q/K/V と重みをワークグループ内で再利用。
- 画像・テキスト混在 prefill: 16 token 単位、量子化重みの復号を4行で共有。
- 因果 attention・三軸 MRoPE・KV キャッシュを chunk 単位で処理。
- Bicubic 補間と patch 配置: 最大8 worker。ピクセルごとの加算順序を維持。
- PNG/JPEG デコード時の encoded byte 配列の複製を除去。
- Vision 専用 OpenCL プログラム、プログラムと引数のキャッシュ、256 MiB の buffer pool。

画像の解像度上限・補間方式・mmproj の重みは変更していない。
LoRA の共有処理は画像混在時に適用し、テキスト会話の fused LoRA 経路を維持。

## 実測

Intel Arc B580 ×2、Release、実 GGUF/mmproj。エンコードは各サイズ3回の中央値。

| 処理 | 参照 | 採用版 | 改善 |
|---|---:|---:|---:|
| 256×256 の画像エンコード | 190.13 ms | 90.64 ms | 2.10倍 |
| 768×768 の画像エンコード | 5,134.95 ms | 1,576.26 ms | 3.26倍 |
| 4096×3072 RGB の前処理、1 worker → 8 worker | 114.15 ms | 22.59 ms | 5.05倍 |
| 新しい768画像、最初の文字表示まで | 24.48秒 | 20.45秒 | 16.4%短縮 |
| 同一画像の embedding キャッシュ使用、最初の文字まで | 18.52秒 | 16.94秒 | 8.5%短縮 |

文字表示の計測は、読み込み済み27Bモデルへ赤い768×768 PNGと
「この画像の色を日本語で答えて。」を渡したもの。新しい画像の行は
mmproj の初期化・画像デコード・Vision・LLM prefill を含む。ベース GGUF の初期ロードは含まない。
prompt 599 token、temperature 0、top-k 1、生成10 token。参照と採用版は同じ日本語回答。
画像エンコードの3.26倍は、その工程の速度であり、回答全体の速度とは分けている。

モデル: `models/Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf`

mmproj: `models/huggingface/unsloth/Qwen3.5-27B-GGUF/mmproj-F16.gguf`

mmproj SHA-256: `e47ac1b901cd3d0ed805fb1cb39d75948e02ab12077ff778219f54acce88e54b`

## 出力・回帰確認

- 256画像の327,680個、768画像の2,949,120個の最終 embedding を全要素比較。**ビット単位で完全一致**。
- 参照・採用版の入力 patch SHA-256 も一致。
- 全画像サイズの attention scores 84,934,656個、softmax後の確率、context を全要素でビット比較。
- GPU の量子復号共有・chunk 境界・複数 GPU・三軸位置・追加 decode・LoRA の回帰確認。
- LoRA なしの mixed logits はビット一致。画像混在 LoRA は token 一致、logits は絶対1e-6＋相対1e-5以内。
- テキスト会話の fused LoRA・EOS は token stream と全 logits の完全一致を確認。
- 並列前処理は1/2/4/8 workerで patch の全バイト一致。追加 managed allocation は約36 KB。

採否・比較値: `benchmark-results/vision-speed-20261005/completion-audit.json`

正式な比較は同じビルドの `reference-clean.json` と `final.json`、
embedding 照合は `reference-clean-vs-final.json`。
途中の `reference.json` はGPUテストとの同時実行があったため、正式な比較には使用しない。

## 比較した候補

- 復号共有なしの16行 chunk は LLM prefill が約26.3秒へ悪化。採用版は4行共有で約16.9秒。
- chunk 64 は16とほぼ同じ。復号共有8行は約18.6秒で4行より遅かったため、16 token / 4行を選択。
- XMX事前packは768画像を約1.15秒に短縮。ただし最終 embedding の相対L2差は約7.0e-6。
  採用版は FP32 の完全一致経路。XMX は `--xmx` で比較できる任意候補で、GUI既定では無効。
- この実行環境では OpenCL のディスクキャッシュ hit は false。
  表の時間にキャッシュ hit による短縮は計上していない。

## 再実行

リポジトリのルートから、上記のモデルとmmprojが存在する状態で実行。
GPU計測中は他の学習・推論・GPUテストを同時に実行しない。

```powershell
dotnet run --project tools/vision-bench/VisionBench.csproj -c Release --no-restore -- reference-clean --reference
dotnet run --project tools/vision-bench/VisionBench.csproj -c Release --no-restore -- final
python tools/vision-bench/compare_embeddings.py reference-clean final
dotnet run --project tools/vision-preprocess-bench/VisionPreprocessBench.csproj -c Release --no-restore
```

`compare_embeddings.py` は NumPy が必要。正式な比較は許容差0で終了コード0。
前処理ツールはウォームアップ2回・計測6回で、全出力のSHAを確認する。
