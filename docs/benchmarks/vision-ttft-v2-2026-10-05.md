# 画像から最初の文字までの高速化 — 2026-10-05

## 測定条件

Intel Arc B580 12GB ×2、Release、読み込み済みの
`Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf` と
`models/huggingface/unsloth/Qwen3.5-27B-GGUF/mmproj-F16.gguf`。
赤・緑・青の異なる768×768 PNGを各1回、合計3回使用する。
画像の解像度・全576個の画像埋め込み・質問は維持する。
入力599 token、temperature 0、top-k 1、thinking off、回答は10/11/10 token。
学習・他のGPUテストとの同時実行は避ける。

計測は生成APIの開始から最初の空でないテキストまで。
GGUFのロードは別に記録する。旧経路は最初の画像送信時にmmprojを
初期化し、新経路は「プリロード」中に初期化する。
添付時の先行処理は、通常の新しい画像の測定には含めていない。
先行処理を使う測定では、その処理時間と送信後の待ち時間を別々に記録する。

## 結果

| 経路 | 最初の文字まで・3画像の中央値 | 最初の文字以降・実測ms/tokenの中央値 |
| --- | ---: | ---: |
| 旧経路・新しい画像 | 18.534秒 | 60.66 |
| 採用経路・新しい画像 | **2.834秒** | **60.62** |
| 採用経路・添付時の先行処理が完了後 | **0.422秒** | 60.86 |

新しい画像では待ち時間が旧経路の**15.3%（約6.54倍の速度）**となり、
「1/5」の目標を達成した。3画像の値は2.917 / 2.834 / 2.824秒。
送信前の先行処理の時間は2.808 / 2.741 / 2.745秒で、0.422秒に含めない。
いずれも赤・緑・青への最終回答とEOS終了が一致し、画像キャッシュなしの測定では
プロンプトの再利用は0、先行処理ありでは580 / 599 tokenを再利用した。

GGUF等の読み込みは旧経路14.703秒、新経路16.481秒で、上表には含まない。
モデルを未読み込みの状態から全てを含む待ち時間が1/5になったという結果ではない。
入力内容、文脈の長さ、thinkingやサンプリング設定で所要時間は変わる。

## 採用した処理

- IQ2は元のGGUF格納を維持する。prefill中に整数グリッド・符号・元の係数を
  workgroupごとに解読し、4.5 KiBのB行列SLMを16 subgroup間で共有する。
  入力AはGPU上で並べ替える。大きなA用SLMとIQ2の一時重み展開を省く。
- XMXの行列演算で整数グリッドを使う。
  入力はFP16の上位・残差に分け、係数と加算はFP32で扱う。
  FP16範囲外の入力はGPU内で元のFP32計算へ戻す。
- prefillは最大1024 token。DeltaNetは状態の加算順序を維持した一括処理と
  SG16のRMS計算を使い、行ごとのworkgroup barrierを削減する。
- Visionの線形演算・attentionも上位・残差のXMX演算を使う。
  softmaxは全キーに対するFP32。72次元の実モデルでは80次元用のタイルを使う。
- 言語モデルのbuffer poolは512 MiB、遅延解放は256 MiB。
  画像モデルの重みと作業領域を言語モデル側のメモリ予算から予約し、
  後続のKV拡張・状態保存も同じ上限で判定する。
- 添付時に画像と既知の履歴を処理し、最後の画像行までの状態を保存する。
  編集中の質問は事前計算に含めない。送信時にトークンID、三軸位置、画像の
  全埋め込みバイトが一致した場合だけ再利用する。

GUIと内蔵APIの既定値に反映する。CoreのXMX設定は明示的な選択を維持する。
resident再配置の候補は追加VRAMと生成速度の低下があったため、既定では使わない。
言語モデルの格納容量と単一tokenのGEMVを維持する。
外部の推論サーバー・llama.cppは使用しない。従来の画像解像度上限や補間方式は変更しない。

## 数値と回帰確認

XMXを含む全処理はFP32参照と近い結果であり、全体のビット一致は主張しない。
一方、IQ2のGGUF BSLMとresident BSLMの比較、範囲外FP32フォールバック、
単一行GEMV・fused LoRA・DeltaNetの出力と状態は、全要素のビット一致を確認する。

- 256/768の最終Vision埋め込み327,680 / 2,949,120個を全要素比較。
  相対L2差は6.77e-6 / 7.59e-6。入力patch SHA-256も一致。
- 実GGUFの3画像について、FP32参照と生成トークン列、全語彙logitsを比較する。
  実LoRAを付けた場合も同じ比較を行う。双方とも各画像の最初の6生成tokenが一致。
  logitsの最大相対L2差はベース1.97e-5、LoRAあり2.23e-5、
  最大絶対差はそれぞれ0.000343 / 0.000568。実測値はcompletion auditに保存する。
- 量子化projectionの端数列、chunk境界、非ゼロLoRA、100,000の入力による
  範囲外フォールバックを確認する。
- 画像差し替え、呼び出し元の配列変更、停止、リセット、繰り返し送信時の
  状態の再利用・無効化を確認する。文字だけのLoRA・EOSも確認する。
- GGUF BSLMの比較2/2、外部メモリ予約の比較3/3、Deltaの基本7/7と
  SG16 RMS比較2/2、Vision XMX attention6/6が成功。混合prefix・LoRA・EOSの
  追加比較7/7も成功（この実行はコンソール記録のみ）。

## 再実行と記録

`benchmark-results/vision-ttft-v2-20261005/completion-audit.json` が測定値と採否、
数値比較、ソースとGUI実行ファイルのSHA-256をまとめる。旧版の記録は
`docs/benchmarks/vision-speed-2026-10-05.md` に保存する。
更新したGUIは `NNtrain.Gui/bin/Release/net10.0-windows/win-x64/publish/NNtrain.Gui.exe`。

```powershell
dotnet run --project tools/vision-bench/VisionBench.csproj -c Release --no-restore -- legacy-formal --v2 --legacy --generation-only --runs=3 --new-images
dotnet run --project tools/vision-bench/VisionBench.csproj -c Release --no-restore -- gguf-bslm-final --v2 --generation-only --runs=3 --new-images --xmx-prefill --factored-prefill --gguf-bslm --prefill=1024 --pool=512 --deferred=256 --xmx --xmx-attention
dotnet run --project tools/vision-bench/VisionBench.csproj -c Release --no-restore -- gguf-bslm-prepared-final --v2 --generation-only --runs=3 --new-images --prepared-image --xmx-prefill --factored-prefill --gguf-bslm --prefill=1024 --pool=512 --deferred=256 --xmx --xmx-attention
dotnet run --project tools/vision-bench/VisionBench.csproj -c Release --no-restore -- parity --v2 --parity --reuse-reference --factored-prefill --gguf-bslm --prefill=1024 --pool=512 --deferred=256 --xmx --xmx-attention
dotnet run --project tools/vision-bench/VisionBench.csproj -c Release --no-restore -- parity --v2 --parity --lora --reuse-reference --factored-prefill --gguf-bslm --prefill=1024 --pool=512 --deferred=256 --xmx --xmx-attention
python tools/vision-bench/create_ttft_audit.py
```

`--reuse-reference` はモデル・LoRAのファイル情報と、参照用プロンプトの全バイトを
照合する。同じ参照がなければ、そのオプションを外して再作成する。
遅かったK64/大型タイル、一時F16の重み全展開、i8、FlashAttentionの候補は
既定で無効。実験専用カーネルは通常のOpenCLプログラムへ含めない。
