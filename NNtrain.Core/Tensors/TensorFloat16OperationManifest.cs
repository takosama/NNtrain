namespace NNtrain;

/// <summary>
/// Auditable inventory of Tensor-returning operations and their Float16
/// contract. The inventory deliberately lives beside Tensor rather than in a
/// test-only list: adding a new Tensor operation requires choosing its result
/// dtype policy and a regression test at the same time.
/// </summary>
/// <remarks>
/// Float16 describes physical value storage. Arithmetic and gradient
/// accumulation remain Float32, and reductions intentionally return Float32.
/// Member ids use the format emitted by <see cref="System.Reflection.MethodInfo"/>
/// in <c>TensorFloat16OperationManifestTests</c>.
/// </remarks>
internal static class TensorFloat16OperationManifest
{
    internal static IReadOnlyList<TensorFloat16OperationManifestEntry>
        PublicTensorReturningMembers { get; } =
        [
            Factory("FromOwnedData(Single[],Int32[],String,TensorDType)"),
            Factory("Scalar(Single,String,TensorDType)"),
            Factory("tensor(Single[],Int32[],String,TensorDType)"),
            Factory("Zeros(Int32[])"),
            Factory("Zeros(TensorDType,Int32[])"),
            Factory("From1D(Single[],String,TensorDType)"),
            Factory("From2D(Single[,],String,TensorDType)"),

            Conversion("To(TensorDType)"),
            Conversion("To(TensorDevice)"),
            Conversion("to(TensorDType)"),
            Conversion("to(TensorDevice)"),
            Conversion("Half()"),
            Conversion("half()"),
            Conversion("ToFloat32()"),

            Preserve(
                "op_Addition(Tensor,Tensor)",
                "TensorFloat16BasicOperationTests.EveryBinaryOperationUsesFloat16StorageWithSimdSizedInputs"),
            Preserve(
                "op_Subtraction(Tensor,Tensor)",
                "TensorFloat16BasicOperationTests.EveryBinaryOperationUsesFloat16StorageWithSimdSizedInputs"),
            Preserve(
                "op_UnaryNegation(Tensor)",
                "TensorFloat16BasicOperationTests.ArithmeticAndBroadcastingPreserveFloat16StorageAndGradients"),
            Preserve(
                "op_Multiply(Tensor,Tensor)",
                "TensorFloat16BasicOperationTests.EveryBinaryOperationUsesFloat16StorageWithSimdSizedInputs"),
            Preserve(
                "op_Division(Tensor,Tensor)",
                "TensorFloat16BasicOperationTests.EveryBinaryOperationUsesFloat16StorageWithSimdSizedInputs"),
            Preserve(
                "Pow(Single)",
                "TensorFloat16BasicOperationTests.ArithmeticAndBroadcastingPreserveFloat16StorageAndGradients"),
            Reduction(
                "Sum()",
                "TensorFloat16BasicOperationTests.ReductionsReturnFloat32AndAccumulateInFloat32"),
            Reduction(
                "Mean()",
                "TensorFloat16BasicOperationTests.ReductionsReturnFloat32AndAccumulateInFloat32"),
            Reduction(
                "CrossEntropyWithLogits(Int32[],Single,Int32)",
                "TensorFloat16ActivationAndLossTests.CrossEntropyReducesFloat16LogitsToFloat32"),

            Preserve(
                "Reshape(Int32[])",
                "TensorFloat16BasicOperationTests.ShapeOperationsPreserveFloat16ValuesAndGradientRouting"),
            Preserve(
                "Slice(Int32,Int32,Int32)",
                "TensorFloat16BasicOperationTests.SliceAndConcatCoverEveryRankThreeAxisForFloat16"),
            Preserve(
                "Concat(Int32,Tensor[])",
                "TensorFloat16BasicOperationTests.SliceAndConcatCoverEveryRankThreeAxisForFloat16"),
            Preserve(
                "Transpose()",
                "TensorFloat16BasicOperationTests.ShapeOperationsPreserveFloat16ValuesAndGradientRouting"),

            Preserve(
                "MatMul(Tensor)",
                "TensorFloat16OperationManifestTests.MatrixAndBatchedOperationsPreserveFloat16StorageAndFloat32Gradients"),
            Preserve(
                "MatMulTransposedRight(Tensor)",
                "TensorFloat16OperationManifestTests.MatrixAndBatchedOperationsPreserveFloat16StorageAndFloat32Gradients"),
            Preserve(
                "MatMulTransposedRightAddRow(Tensor,Tensor)",
                "TensorFloat16OperationManifestTests.MatrixAndBatchedOperationsPreserveFloat16StorageAndFloat32Gradients"),
            Preserve(
                "MatMulTransposedRightAddRowRelu(Tensor,Tensor)",
                "TensorFloat16FusedOperationTests.FusedLinearReluAndResidualLayerNormUseFloat16Storage"),
            Preserve(
                "BatchedMatMul(Tensor)",
                "TensorFloat16OperationManifestTests.MatrixAndBatchedOperationsPreserveFloat16StorageAndFloat32Gradients"),
            Preserve(
                "BatchedMatMulTransposedRight(Tensor)",
                "TensorFloat16OperationManifestTests.MatrixAndBatchedOperationsPreserveFloat16StorageAndFloat32Gradients"),

            Preserve(
                "Sin()",
                "TensorFloat16ActivationAndLossTests.ElementwiseAndMaskedActivationsPreserveFloat16"),
            Preserve(
                "Relu()",
                "TensorFloat16ActivationAndLossTests.ActivationPipelinePreservesFloat16AndFloat32Gradients"),
            Preserve(
                "AddRowWise(Tensor)",
                "TensorFloat16ActivationAndLossTests.ElementwiseAndMaskedActivationsPreserveFloat16"),
            Preserve(
                "SoftmaxLastDim()",
                "TensorFloat16ActivationAndLossTests.ActivationPipelinePreservesFloat16AndFloat32Gradients"),
            Preserve(
                "LogSoftmaxLastDim()",
                "TensorFloat16ActivationAndLossTests.ElementwiseAndMaskedActivationsPreserveFloat16"),
            Preserve(
                "LayerNormLastDim(Tensor,Tensor,Single)",
                "TensorFloat16ActivationAndLossTests.ActivationPipelinePreservesFloat16AndFloat32Gradients"),
            Preserve(
                "AddLayerNormLastDim(Tensor,Tensor,Tensor,Single)",
                "TensorFloat16FusedOperationTests.FusedLinearReluAndResidualLayerNormUseFloat16Storage"),
            Preserve(
                "CausalMask(Single)",
                "TensorFloat16ActivationAndLossTests.ElementwiseAndMaskedActivationsPreserveFloat16"),
            Preserve(
                "AddBatchWise(Tensor)",
                "TensorFloat16BasicOperationTests.BatchWiseAdditionReadsFloat16StorageAndReducesGradient"),
            Preserve(
                "Dropout(Single,Random)",
                "TensorFloat16ActivationAndLossTests.EmbeddingAndDropoutPreserveFloat16Storage"),
            Preserve(
                "AddDropout(Tensor,Single,Random)",
                "TensorFloat16ActivationAndLossTests.PositionalEmbeddingAndResidualDropoutPreserveFloat16"),

            Preserve(
                "EmbeddingLookup(Int32[],Int32[])",
                "TensorFloat16ActivationAndLossTests.EmbeddingAndDropoutPreserveFloat16Storage"),
            Preserve(
                "EmbeddingLookupWithPositions(Tensor,Int32[],Int32,Int32)",
                "TensorFloat16ActivationAndLossTests.PositionalEmbeddingAndResidualDropoutPreserveFloat16"),

            Preserve(
                "FusedMultiHeadAttention(Int32,Boolean)",
                "TensorFloat16FusedOperationTests.AttentionUsesPackedFloat16InputsAndFloat32Gradients"),
            Preserve(
                "FusedCausalHyenaOrder2(Tensor,Tensor,Tensor,HyenaConvolutionAlgorithm)",
                "TensorFloat16FusedOperationTests.HyenaDirectAndFftReadPackedFloat16Filters"),
            Preserve(
                "FusedForgetScan()",
                "TensorFloat16FusedOperationTests.ForgetScanKeepsRecurrenceAndBackwardAccumulationInFloat32"),
            Preserve(
                "ForgetMemoryV2(Int32,Int32,Single)",
                "TensorFloat16FusedOperationTests.ForgetMemoryV2KeepsMatrixStateAndGradientsInFloat32"),
        ];

    internal static IReadOnlyList<TensorFloat16OperationManifestEntry>
        InternalTensorReturningMembers { get; } =
        [
            Preserve(
                "ForgetMemoryV2Continue(Int32,Int32,Single,Single[])",
                "ForgetMemoryV2Tests.GptSchedulesShortToLongMemoryAndTrains"),
            Preserve(
                "LinearLastDim(Tensor,Tensor,Boolean)",
                "LinearLastDimTests.Float16ProjectionMatchesFormerFloat16Graph"),
        ];

    /// <summary>
    /// Operations which do not return a Tensor, but are part of the Float16
    /// user contract. They are listed separately so the reflection inventory
    /// remains precise about Tensor-producing nodes.
    /// </summary>
    internal static IReadOnlyList<TensorFloat16OperationManifestEntry>
        AuxiliaryMembers { get; } =
        [
            new(
                "item()",
                TensorFloat16ResultPolicy.Auxiliary,
                "TensorFloat16OperationManifestTests.Float16AuxiliaryApisReadAndClearGradients"),
            new(
                "Backward(Single[])",
                TensorFloat16ResultPolicy.Auxiliary,
                "TensorFloat16OperationManifestTests.Float16AuxiliaryApisReadAndClearGradients"),
            new(
                "ZeroGrad()",
                TensorFloat16ResultPolicy.Auxiliary,
                "TensorFloat16OperationManifestTests.Float16AuxiliaryApisReadAndClearGradients"),
            new(
                "DataString()",
                TensorFloat16ResultPolicy.Auxiliary,
                "TensorFloat16OperationManifestTests.Float16AuxiliaryApisReadAndClearGradients"),
            new(
                "GradString()",
                TensorFloat16ResultPolicy.Auxiliary,
                "TensorFloat16OperationManifestTests.Float16AuxiliaryApisReadAndClearGradients"),
        ];

    private static TensorFloat16OperationManifestEntry Factory(string memberId)
        => new(
            memberId,
            TensorFloat16ResultPolicy.Factory,
            "TensorFloat16OperationManifestTests.FactoriesAndConversionsSupportFloat16");

    private static TensorFloat16OperationManifestEntry Conversion(string memberId)
        => new(
            memberId,
            TensorFloat16ResultPolicy.Conversion,
            "TensorFloat16OperationManifestTests.FactoriesAndConversionsSupportFloat16");

    private static TensorFloat16OperationManifestEntry Preserve(
        string memberId,
        string verification)
        => new(memberId, TensorFloat16ResultPolicy.PreserveFloat16, verification);

    private static TensorFloat16OperationManifestEntry Reduction(
        string memberId,
        string verification)
        => new(memberId, TensorFloat16ResultPolicy.ReduceFloat32, verification);
}

internal sealed record TensorFloat16OperationManifestEntry(
    string MemberId,
    TensorFloat16ResultPolicy ResultPolicy,
    string Verification);

internal enum TensorFloat16ResultPolicy
{
    /// <summary>Creates a Tensor in an explicitly selected dtype.</summary>
    Factory,

    /// <summary>Changes storage dtype explicitly.</summary>
    Conversion,

    /// <summary>All-Float16 Tensor inputs produce Float16 value storage.</summary>
    PreserveFloat16,

    /// <summary>Reads Float16 inputs but intentionally returns Float32.</summary>
    ReduceFloat32,

    /// <summary>Non-Tensor-returning public contract such as Backward.</summary>
    Auxiliary,
}
