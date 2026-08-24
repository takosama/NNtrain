# Tensor Float16 operation manifest

`TensorDType.Float16` は値を IEEE 754 binary16 (`Half[]`) として物理保存する。
演算時の展開、reduction、gradient と optimizer の master value は Float32 のままで
ある。この組み合わせにより、値の保存帯域を半分にしつつ、backward の蓄積精度を
保つ。

この文書は「Float16 を一部のモデルだけで使える」ではなく、Tensor API 全体で
どこまで使えるかを監査するための対応表である。実行可能な正本は
`TensorFloat16OperationManifest` で、
`TensorFloat16OperationManifestTests` が reflection により、追加された
Tensor-returning public/internal member が表にない状態をテスト失敗にする。

## 判定規則

- **保持**: 入力 Tensor がすべて Float16 なら出力の物理 storage も Float16。
  演算・勾配蓄積は Float32。
- **reduction**: Float16 入力を読むが出力は意図的に Float32。
- **factory/conversion**: 呼び出し側が dtype を明示して Float16 を選ぶ。
- Float16 と Float32 を混在させた値演算は Float32 へ promote する。
- Float8/Float4/Float2/1.58-bit は enum と dispatch 境界だけを予約済みであり、
  codec/scaling policy が未実装のため現在は明示的に拒否する。

## public Tensor-returning API

| 分類 | member id | Float16 結果 | 実行テスト |
| --- | --- | --- | --- |
| Factory | `FromOwnedData`, `Scalar`, `tensor`, `Zeros(dtype, ...)`, `From1D`, `From2D` | 明示した dtype。Float16 を指定可能 | `FactoriesAndConversionsSupportFloat16` |
| Factory default | `Zeros(...)` | 既定は後方互換の Float32。`Zeros(Float16, ...)` で Float16 | `FactoriesAndConversionsSupportFloat16` |
| Conversion | `To`, `to`, `Half`, `half` | Float16 へ明示変換 | `FactoriesAndConversionsSupportFloat16` |
| Conversion | `ToFloat32` | 意図的に Float32 | `FactoriesAndConversionsSupportFloat16` |
| Elementwise | `op_Addition`, `op_Subtraction`, `op_UnaryNegation`, `op_Multiply`, `op_Division`, `Pow` | 保持 | `TensorFloat16BasicOperationTests` |
| Reduction/loss | `Sum`, `Mean`, `CrossEntropyWithLogits` | 意図的に Float32 | `TensorFloat16BasicOperationTests`, `TensorFloat16ActivationAndLossTests` |
| Shape/copy | `Reshape`, `Slice`, `Concat`, `Transpose` | 保持（同 dtype copy/view） | `TensorFloat16BasicOperationTests` |
| Dense | `MatMul`（rank1×rank1/rank2×rank1/rank2×rank2）, `MatMulTransposedRight`, `MatMulTransposedRightAddRow`, `MatMulTransposedRightAddRowRelu` | 保持 | `TensorFloat16OperationManifestTests`, `TensorFloat16FusedOperationTests` |
| Batched dense | `BatchedMatMul`, `BatchedMatMulTransposedRight` | 保持 | `TensorFloat16OperationManifestTests` |
| Activation | `Sin`, `Relu`, `AddRowWise`, `SoftmaxLastDim`, `LogSoftmaxLastDim`, `LayerNormLastDim`, `AddLayerNormLastDim`, `CausalMask`, `AddBatchWise` | 保持 | `TensorFloat16ActivationAndLossTests`, `TensorFloat16FusedOperationTests`, `TensorFloat16BasicOperationTests` |
| Regularization | `Dropout`, `AddDropout` | 保持 | `TensorFloat16ActivationAndLossTests` |
| Embedding | `EmbeddingLookup`, `EmbeddingLookupWithPositions` | 保持 | `TensorFloat16ActivationAndLossTests` |
| Fused sequence | `FusedMultiHeadAttention`, `FusedCausalHyenaOrder2`（Direct/Fft）, `FusedForgetScan`, `ForgetMemoryV2/V3/DRN` | 保持。内部 recurrence/state/gradient は Float32 | `TensorFloat16FusedOperationTests`, `ForgetMemoryV3Tests`, `ForgetMemoryDRNTests` |

`member id` の完全な overload 単位の一覧とテスト名は
[`TensorFloat16OperationManifest.cs`](../NNtrain.Core/Tensors/TensorFloat16OperationManifest.cs)
にあり、上表は読解用の分類である。reflection contract が overload を含む完全性を
保証するため、表だけを更新して実装/API と乖離させることはできない。

## internal Tensor node

| member id | Float16 結果 | 実行テスト |
| --- | --- | --- |
| `LinearLastDim(Tensor, Tensor, Boolean)` | 保持 | `LinearLastDimTests.Float16ProjectionMatchesFormerFloat16Graph` |

これは rank 3 以上の `Linear` が flatten/reshape の graph を作らずに使う内部 node
である。public API と同様に reflection contract の対象に含める。

## Tensor を返さない API

`item`, `Backward`, `ZeroGrad`, `DataString`, `GradString` は Tensor を返さないが、
Float16 のユーザー契約に含まれる。`Float16AuxiliaryApisReadAndClearGradients` が
packed storage の値読み出し、Float32 gradient の backward、gradient clear、文字列化
を一度に検証する。

## 実行方法

```powershell
dotnet test NNtrain.Core.Tests --no-restore --filter FullyQualifiedName~TensorFloat16
```

このテスト群は SIMD が有効な長さと scalar tail の両方を通す。SIMD の有無は結果の
仕様を変えず、codec と kernel の最適化だけを切り替える。
