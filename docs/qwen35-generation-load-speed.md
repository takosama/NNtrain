# Qwen3.5 27B：生成とロードの追加高速化（2026-09-27）

## 結果

前回の生成・訓練高速化を含む `6c003b7` との比較です。今回は **ロード時間の短縮が主な成果** です。生成速度の差は小さく、このPC・プロンプトでの実測値として扱います。

| 条件 | 生成・変更前 tok/s | 変更後 tok/s | 差 | 起動・変更前 s | 変更後 s | 起動時間短縮 |
|---|---:|---:|---:|---:|---:|---:|
| LoRAなし | 17.120 | 17.270 | 0.88% | 6.950 | 4.387 | 36.9% |
| LoRAあり | 16.145 | 16.178 | 0.21% | 18.142 | 8.044 | 55.7% |

## 測定条件

- Intel Arc B580 12GiB × 2、ドライバー32.0.101.9030、Windows、.NET 10.0.11、Release。GPU実行は直列化。
- `Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf`（10,173,451,200 bytes）。LoRAはつくよみちゃん1536更新の `adapter.bin`、rank8、alpha16、496対象。
- 同じ16入力トークン、256出力トークン。各版・各LoRA条件で独立2プロセス×3生成。生成値は各プロセスの初回を除く計4回の中央値。起動値は2プロセスの平均。
- 生成速度は最初と最後の出力コールバック間の255トークン分。EOSでも止めず固定長を測定。回答品質・通常会話終了までの時間は別条件。
- 起動はGGUF解析・tokenizer・モデル・LoRAロードの合計。ベンチマークのデバイス列挙は合計の外。CLIのloadはコマンド内の開始から準備完了までを計測。
- OS/ドライバーキャッシュは消去していない。新版の主表はOpenCLバイナリキャッシュヒット時。新規インストール・ドライバー変更後の初回と同一条件ではない。
- LoRAロード時は完全なベースモデルSHA256を毎回検証。SHAの永続キャッシュやチェックの省略は行っていない。LoRAのSHA処理はモデルロードと重ねるため、その時間はモデルロード側に含む。

### 起動内訳と初回トークン

| 条件 | 版 | モデルロード s | アダプター追加ロード s | 初回トークン ms |
|---|---|---:|---:|---:|
| LoRAなし | 変更前 | 6.569 | 0.000 | 516.7 |
| LoRAなし | 変更後 | 4.151 | 0.000 | 509.9 |
| LoRAあり | 変更前 | 6.300 | 11.482 | 569.9 |
| LoRAあり | 変更後 | 6.829 | 0.959 | 566.1 |

初回トークン時間はロード後の状態リセット、prefill、最初の選択を含む。アダプター単体の時間だけを比較するとSHA移動の効果が混入するため、主表では準備完了までの合計で比較した。

### 1 GPUの確認

1 GPUは128出力×2回の別条件。2回目の値を記載し、256出力の主表との厳密な比較には使用しない。

| 条件 | 生成 tok/s | 起動 s |
|---|---:|---:|
| LoRAなし | 17.550 | 6.538 |
| LoRAあり | 16.408 | 7.698 |

## 採用した変更

1. GGUFの解析結果とtokenizer読込用readerを共有。位置指定読込でpacked tensorをGPUごとに並列ロード。デバイス順序・例外時の後片付けを維持。
2. OpenCLのコンパイル済みバイナリを保存。ソース・ビルド設定・デバイス・ドライバー情報をキーにし、SHAで破損を検知。使用不能時は元ソースから再構築。保存先は `%LOCALAPPDATA%\NNtrain\OpenCLCache`。
3. 完全なモデルSHAを4MiBバッファで処理し、ロードと並行化。単独測定は従来平均10.828秒→5.870秒。同じ全ファイルSHAを得た。
4. 推論用LoRAはA/Bのみ保持。Adam状態も全値の有限性・2次モーメント非負・チェックサムを読みながら検証し、一時領域で破棄。訓練用は全状態を保持。
5. 推論時のOpenCL引数設定をキャッシュし、バッファ解放時に無効化。層の重み参照とカーネル名をロード時に準備し、毎トークンの文字列生成・辞書検索を削減。
6. IQ2/IQ3の2出力行で入力を共有する射影と、加算順を維持したRMSNormを採用。Q4とLoRAの2行射影は測定で不利だったため既定では無効。
7. 既存のKVキャッシュ・DeltaNet状態再利用、LoRA融合を維持。学習用の生成カーネル切替は無効。ベース重みの再量子化・全体FP16展開は行っていない。

OpenCLバイナリのロードと取得は公式APIに従う：[clCreateProgramWithBinary](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clCreateProgramWithBinary.html)、[clGetProgramInfo](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetProgramInfo.html)。

## 試して採用しなかった経路

探索は主に128出力×2回。設定の組合せ・計測時刻が異なるため、効果を加算したり、主表の精度で比較したりしない。全試行の詳細は `benchmarks/qwen35-generation-load-2026-09-27/trial-summary.json`。

| 試行 | 記録名 | 最終回 tok/s | 起動 s |
|---|---|---:|---:|
| IQ2の無損失GPU再配置 | repack-base | 15.370 | 6.593 |
| 同上・LoRAあり | repack-lora | 14.677 | 7.745 |
| 幅固定IQ2カーネル | specialized-base | 16.787 | 4.316 |
| 幅固定・WG16 | specialized16-base | 16.790 | 4.363 |
| WG16 | wg16-base | 17.267 | 7.092 |
| 残差加算＋RMS融合 | fused-base | 17.180 | 4.025 |
| 同上・LoRAあり | fused-lora | 16.156 | 7.826 |
| Q4の2行射影 | pair-q4-base | 17.224 | 4.275 |
| LoRAの2行射影 | pair-lora | 16.174 | 12.501 |
| 直列ロード | serial-base | 17.286 | 5.628 |
| 引数キャッシュ無効 | nocacheargs-base | 17.280 | 4.279 |
| プログラムキャッシュ未命中 | cache-first-base | 17.257 | 6.311 |
| プログラムキャッシュ命中 | cache-warm-base | 17.255 | 4.098 |

IQ2の再配置は約1.93GBの追加VRAMも必要だったため撤去。幅固定/WG16/残差融合も遅く、製品コードから撤去。引数キャッシュは壁時計の改善を単独では確認できず、ホスト側の重複呼出削減として保持。

## 正しさと通常環境の検証

- 2 GPUの各版・各条件・各回256トークンIDが完全一致。1 GPUは先頭128トークンが一致。
- 固定teacherの4ステップ×語彙248,320のFP32 logitsは、各条件で変更前とビット一致。最大絶対差0、全値有限。LoRAありとなしをそれぞれの変更前に照合。
- Core関連327件成功。Integration関連48件成功、CUDA専用6件スキップ（計375成功）。通常リポジトリでも関連Integrationを再実行し61件成功・CUDA専用6件スキップ。
- 通常リポジトリのCLIをReleaseビルドし、LoRAあり・なし双方で実モデルの日本語入力からのストリーム生成を確認（LoRAなしは64トークンで思考出力中に終了、LoRAありは日本語回答・EOSまで20トークン）。ログを同じ証跡フォルダーに保存。
- 量子化ベース重みの常駐量はGPU0が5,035,468,800 bytes、GPU1が5,116,405,760 bytesで変更なし。
- 完成済みadapter.bin、GGUF LoRA、学習設定のSHAを照合し変更なし。通常リポジトリの既存変更を保持し、今回の25ソース変更を照合した。
- キャッシュ破損/拒否/無効時のfallback、引数キャッシュ寿命、並列ロードと例外、LoRA壊れた形式/値/ハッシュ、RMS/射影数値をテスト。
- 検証したのはこのモデル・GPU・ドライバー・入力。一般的な回答品質、長いコンテキスト全域、他GPUでの速度、OSキャッシュ完全消去時の起動は未測定。

## 生成コマンド

`C:\Users\takos\source\repos\NNtrain` でPowerShellから実行。ストリーム出力とKVキャッシュは既定で有効。GGUF LoRAファイルはエクスポート成果物であり、現在のNNtrainの `--adapter` には検証済みの `adapter.bin` を指定する。

```powershell
dotnet build .\NNtrain.Cli\NNtrain.Cli.csproj -c Release
$prompt = "<|im_start|>user`nこんにちは。自己紹介をしてください。<|im_end|>`n<|im_start|>assistant`n"

# LoRAなし
dotnet .\NNtrain.Cli\bin\Release\net10.0\NNtrain.Cli.dll qwen-gguf --model ".\models\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf" --devices 0,1 --prompt $prompt --max-new-tokens 1000

# LoRAあり
dotnet .\NNtrain.Cli\bin\Release\net10.0\NNtrain.Cli.dll qwen-gguf --model ".\models\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf" --adapter ".\checkpoints\tuku-qwen35\adapter.bin" --devices 0,1 --prompt $prompt --max-new-tokens 1000
```

## 測定・テストの再実行

```powershell
dotnet build .\NNtrain.Benchmarks\NNtrain.Benchmarks.csproj -c Release
dotnet .\NNtrain.Benchmarks\bin\Release\net10.0\NNtrain.Benchmarks.dll --qwen35-generation-probe --model ".\models\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf" --devices 0,1 --tokens 256 --runs 3 --logits 4 --output .\generation-base-new.json
# LoRAは同じ引数に --adapter .\checkpoints\tuku-qwen35\adapter.bin を追加。出力先は未作成の別名にする。
dotnet test .\NNtrain.Core.Tests\NNtrain.Core.Tests.csproj -c Release --filter "FullyQualifiedName~Qwen35|FullyQualifiedName~Gguf|FullyQualifiedName~ArcKernelArgumentCache|FullyQualifiedName~ArcProgramBinaryCache|FullyQualifiedName~QwenArcKernelTests"
dotnet test .\NNtrain.IntegrationTests\NNtrain.IntegrationTests.csproj -c Release --filter "FullyQualifiedName~Qwen|FullyQualifiedName~Lora"
```

再現用の各回ID、時間、バイナリSHA、GPU情報、検証結果は [証跡](benchmarks/qwen35-generation-load-2026-09-27/verification.json) と同ディレクトリのJSONに保存。生のlogitsはローカル作業記録に保存し、Gitには照合結果とSHAを保存した。
