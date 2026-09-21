# DRN retention floor修正（2026-09-11）

DRNで無視されていた層別`retentionFloor`を復元した。

```text
s = sigmoid(raw_gate)
f = retentionFloor + (1 - retentionFloor) * s
read = M_previous * q
M = f * M_previous + beta * (v - M_previous * k) * k^T
d_raw_gate = d_f * (1 - retentionFloor) * s * (1 - s)
```

Q/KのL2正規化、tanh(v)、beta、read-before-write、保存・蓄積精度は変更していない。
既存の層別floor（標準0.5→0.99）をそのまま使用する。JSONとcheckpointは変更していない。
保存形式は維持するが、以前のcheckpointから再開しても以後の計算ではfloorが有効になるため、以前と同一のloss・出力にはならない。

## 変更範囲

- CPU forward、backward、履歴再計算、recurrent continuation。
- CUDA generic、Tensor Core、prepared/unprepared warp backward。
- BF16/mix8 chunk forward、parallel forward、adjoint境界計算、replay backward。
- managed呼び出しから全kernelまでfloorを引き渡す。
- native ABIを1.35へ更新。旧exportのシグネチャは維持し、新しいfloor付きexportを追加。
- DRN実行時にABI 1.35を要求し、古いDLLによるfloor無視を拒否。

高floorでは並列chunkの近似反復が規定回数で完全一致しない場合がある。
その場合は従来からの逐次fallbackを使い、出力・stateの完全一致を維持する。

## 検証結果

- Windows x64、.NET 10 Release全solutionビルド：警告0／エラー0。
- CUDA 13 nativeビルド：SM80/86/89/90およびPTX、export検査合格。
- GPU実行検査：RTX 3070 Ti（device 0）。他SMの実機実行は未検証。
- 最終の関連テスト150件：149成功、失敗0、既存の既知問題によるskip 1。
- 対象：DRN/V2/V3、floor解析式・有限差分、層別配分、CUDA chunk/prepared/再計算、continuation、ABI、LoRA/DPO関連。
- floor 0／0.5／0.99のforward・gradient、およびparallel floor 0／0.37／0.99を検査。
- 既存の数値許容差は変更していない。新規TC参照計算はBF16 state operandとFP32保持・勾配を区別する。
- 既存skip：`ForgetMemoryDRNTests.ScalarKeyLongSequenceMatchesCpuKnownIssue`（K=1長系列での既知CPU/CUDA数値差）。

最終結果：`benchmark-results/drn-retention-floor-tests-20260911/drn-floor-final.trx`。
更新前DLLは`benchmark-results/drn-retention-floor-native-20260911/NNtrain.CudaKernels.pre-floor.dll`へ保存。
本番学習・長時間収束検証・checkpoint書き込みは行っていない。
