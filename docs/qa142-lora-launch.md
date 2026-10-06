# QA142 LoRA の起動方法

学習はまだ開始していない。`checkpoints/qa142-prepared-20261001/` に、CPUで
準備・検証済みの新規run設定を作成した。既存checkpoint・export済みGGUF adapterは使わず、
新しく `adapter.bin` を作る。既存ファイルがあれば新規開始は拒否する。

## この準備済みrunを開始

PowerShellで次を実行する（これを実行するとGPU学習が始まる）：

```powershell
Set-Location 'C:\Users\takos\source\repos\NNtrain'
.\tools\Start-Qa142Lora.ps1 -Start -RunDirectory '.\checkpoints\qa142-prepared-20261001'
```

スクリプト実行が環境のExecutionPolicyで制限される場合は、ポリシーを変更せず
既にbuild済みのCLIを直接起動できる：

```powershell
Set-Location 'C:\Users\takos\source\repos\NNtrain'
dotnet .\NNtrain.Cli\bin\Release\net10.0\NNtrain.Cli.dll qwen-lora --model .\models\Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf --config .\checkpoints\qa142-prepared-20261001\train.json 2>&1 | Tee-Object -FilePath .\checkpoints\qa142-prepared-20261001\training-console.log -Append
```

出力は同runディレクトリの `adapter.bin`（optimizer込み、再開用）、`loss.html`、
lossのmetrics sidecar、`training-console.log`。5更新ごとに保存し、終了時にも保存する。
失敗・中断時は最後に正常保存できたcheckpointまで戻る。未保存の更新は再開できない。
export済みGGUF adapterだけではresumeできない。

## CPU確認のみ／明示的な再開

```powershell
# CPU validation only; GPU/model weights/training are not started
.\tools\Start-Qa142Lora.ps1 -DryRun -RunDirectory '.\checkpoints\qa142-prepared-20261001'

# Resume only after this run has actually saved adapter.bin
.\tools\Start-Qa142Lora.ps1 -Start -Resume -RunDirectory '.\checkpoints\qa142-prepared-20261001'
```

CLIへ直接resumeする場合は同じmodel/configのコマンドに `--resume` を加える。
`--dry-run` ならCPUで設定・データ・token長を検証してGPU load前に終了する。
resume dry-runはcheckpointの形式/TrainingIdentityヘッダーも確認するが、
全optimizer payloadのintegrityチェックは実際のcheckpoint loader側で行う。

新しい独立runの準備は次で行う。毎回一意の新規ディレクトリになる：

```powershell
.\tools\Start-Qa142Lora.ps1 -DryRun -Epochs 2
```

CPU確認後に表示されたRunDirectoryを `-Start -RunDirectory` に渡す。
指定RunDirectoryが存在する場合はprepared manifestを確認し、他の設定へ書き換えない。
同じrunの重複起動はファイルロックで拒否する。スクリプトは別のNNtrain CLI/probeが
動いていれば開始を拒否し、そのプロセスを停止しない。

## データと長さ

元の `data/rintya/rintya_qa_142.jsonl` は未変更。
元バイトのコピー `source.jsonl` と、修正を記録した `normalized.messages.jsonl`、
CLI用 `training.jsonl` をrun内に保存した。142例／284メッセージ。
全history role/contentをChatML promptへ順に保存し、最後のassistant contentをresponseにする。
responseはtrimしない。既存QA80のChatMLに合わせた空think assistant prefixを用いる。
追加のpromptPrefix/responsePrefixは空、EOSはCLIが1個追加する。

修正は次の3点だけ。元ファイルはそのまま残す：

- 23行目：Markdown `\*` のliteralバックスラッシュ64個を正しいJSONで表現。
- 100行目：LaTeXのliteralバックスラッシュ151個を正しいJSONで表現。
  `\beta`、`\neq`、`\times`等を制御文字・改行へ誤decodeしない。
  正常なJSONの `\n` は改行のまま。数式の内容やMarkdown記号を削除しない。
- 76行目：user質問の後の最終回答role `assisutant` の明らかな綴り誤りを
  `assistant` に訂正。回答contentは不変、RoleRepairsに記録。

CPUで本モデルのGGUF tokenizerと本CLIのReadExamplesを使用した実長は、
**75〜6530 tokens（EOS込み）**。**17例が4096を超える**。
contextLengthは上限であり、固定padding長ではない。全例を切り捨てずに使うため
提案上限を **8192** とした。padding・packing・truncate・合成token反復はしていない。
上限4096を明示すると6530-token例を拒否し、勝手に短縮しない。
`preparation.json` のLengthsに全142例のprompt/response/total token数がある。

## 設定と高速化の範囲

モデル：`Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf`、Arc 0/1。
**exact精度、rank8、alpha16、LR0.0001、weight decay0、clip1、seed1、2epoch=284更新**。
全対象層・12 LoRA targets、output adapterなし。shuffleはseed固定のepoch単位。
rank/LR等は以前のQA設定を引き継いだ提案値であり、過去QA142 runの設定を復元した値とは
称していない。2epochは過去QA設定とQA142 adapter名にも整合する提案。
保存間隔5、pool2048 MiB/Arc、projection cache0を明示。元のユーザー設定は書き換えていない。

以前に承認済みの `hostCheckpointGpuGradients=true` と `packedAttentionScores=true` は維持。
最新の固定4096/8192合成ベンチで有力だった4設定は、QA142が変動長なので
**fusedAttentionRows=false、hostCheckpointBufferHandoff=false、streamedAttentionTileRows=0、
iq2RollingTranspose=false** で固定した。resumeでも同じ値を保存/loadする。
`useMeasuredLengthDefaults=false` にして自動判断の再適用も避ける。
未測定長へ高速化を拡張せず、FP16/XMXへ精度を変更していない。
したがって「QA142で3.60%／7.68%速い」とは主張しない。

今回検証はCPUのみ。GPU VRAM/完走、全epochの品質、heldout、生成品質は未検証。
2epochの所要時間は、exact・変動長データの実測がないため提示しない。

## CPU検証と変更

元ファイルSHA256とsourceコピーが一致。元の正常JSON140行のcontentに変更0、
全142例のChatML変換とnormalized contentの対応に不一致0。roleは既知の綴り修正1件のみ。
normalizationのliteral/math/newline保持、複数role長文保持、未知escape・未知fields拒否、
設定保存/load、length defaults、dry-runとresumeのCPU検証を実施した。
script dry-runは142例のCPU検証を通過し、adapter.binは作成されていない。
CPU既存変更の3ハッシュは一致。学習/GPU測定/push/uploadは実施していない。

変更：`NNtrain.Cli/QwenLoraPreparation.cs`（新規）、Program dispatch、QwenLoraCommandの
CPU `--dry-run`、`tools/Start-Qa142Lora.ps1`（新規）、CPU integration tests。
証跡：`benchmark-results/qa142-launcher-20261001/` とrun内 `preparation.json`、`dry-run.log`。
