using NNtrain.Runtime.Execution;

namespace NNtrain;

internal enum CudaBfp8ScaleContract
{
    None = 0,
    TensorWide = 1,
    Block = 2,
}

internal enum CudaPrecisionComputeContract
{
    MatrixOperand = 0,
    Elementwise = 1,
    Reduction = 2,
    Storage = 3,
}

internal sealed record CudaPrecisionOperationRoute(
    PrecisionMode Mode,
    // Logical tensor storage at operation boundaries. Backend may name a
    // short-lived fused intermediate (for example direct BF16 loss logits).
    NumericFormat Storage,
    // Logical policy format. Backend describes the physical CUDA operand
    // chosen after BFP8 decode/packing.
    NumericFormat Compute,
    NumericFormat Accumulation,
    NumericFormat Gradient,
    CudaBfp8ScaleContract ScaleContract,
    string Backend,
    bool UsesTensorCoreWhenEligible,
    bool AllowsCpuFallback);

internal sealed record CudaPrecisionOperationEntry(
    string Operation,
    CudaPrecisionComputeContract ComputeContract,
    IReadOnlyList<CudaPrecisionOperationRoute> Routes);

/// <summary>
/// Executable documentation for Transformer training precision dispatch. The
/// manifest is deliberately independent of physical tensor dtype checks: its
/// values describe the model policy contract which every CUDA dispatcher must
/// preserve. Tests require a resident route for all five supported modes.
/// </summary>
internal static class CudaPrecisionOperationManifest
{
    internal static IReadOnlyList<CudaPrecisionOperationEntry> Entries { get; }
        =
        [
            Entry(
                "linear/GEMM",
                "cuBLAS FP32",
                "cuBLASLt BF16 Tensor Core; resident CUDA fallback",
                "cuBLASLt BF16 Tensor Core; FP32 gradient/master",
                "cuBLASLt INT8 Tensor Core; BF16 Tensor Core fallback",
                "block decode + BF16 Tensor Core + block requantize"),
            Entry(
                "attention",
                "CUDA FlashAttention FP32",
                "CUDA BF16 Tensor Core FlashAttention",
                "CUDA BF16 Tensor Core FlashAttention; FP32 gradient",
                "tensor-scale decode + BF16 Tensor Core FlashAttention",
                "block decode + BF16 Tensor Core FlashAttention"),
            Entry(
                "LayerNorm",
                "CUDA block/warp reduction FP32",
                "CUDA BF16 IO + FP32 block/warp reduction",
                "CUDA BF16 IO + FP32 block/warp reduction",
                "tensor-scale decode + FP32 reduction + requantize",
                "block decode + FP32 reduction + block requantize",
                tensorCore: false,
                computeContract: CudaPrecisionComputeContract.Reduction),
            Entry(
                "residual/dropout/LayerNorm",
                "CUDA fused FP32",
                "CUDA fused BF16 IO + FP32 reduction",
                "CUDA fused BF16 IO + FP32 reduction/gradient",
                "CUDA fused tensor-scale BFP8 + FP32 reduction",
                "CUDA fused block-scale BFP8 + FP32 reduction",
                tensorCore: false,
                computeContract: CudaPrecisionComputeContract.Reduction),
            Entry(
                "cross-entropy loss",
                "CUDA stable FP32 reduction",
                "CUDA BF16 logits + FP32 stable reduction",
                "CUDA BF16 logits + FP32 stable reduction/gradient",
                "direct BF16 loss-head intermediate or tensor-scale decode " +
                    "+ FP32 stable reduction",
                "direct BF16 loss-head intermediate or block decode + " +
                    "FP32 stable reduction",
                tensorCore: false,
                computeContract: CudaPrecisionComputeContract.Reduction),
            Entry(
                "classification accuracy",
                "CUDA FP32 argmax + Int32 block reduction",
                "CUDA BF16 logits + FP32 compare + Int32 block reduction",
                "CUDA BF16 logits + FP32 compare + Int32 block reduction",
                "CUDA tensor-scale decode + FP32 compare + Int32 reduction",
                "CUDA block-scale decode + FP32 compare + Int32 reduction",
                tensorCore: false,
                computeContract: CudaPrecisionComputeContract.Reduction),
            Entry(
                "embedding",
                "CUDA resident gather/reduce",
                "CUDA resident BF16 gather + FP32 gradient reduction",
                "CUDA resident BF16 gather + FP32 gradient reduction",
                "CUDA tensor-scale BFP8 gather/requantize",
                "CUDA block-scale BFP8 gather/requantize",
                tensorCore: false,
                computeContract: CudaPrecisionComputeContract.Storage),
            Entry(
                "ForgetMemory",
                "CUDA recurrent FP32",
                "CUDA BF16 Tensor Core when aligned; resident BF16 fallback",
                "CUDA BF16 Tensor Core when aligned; FP32 state/gradient",
                "tensor-scale decode + BF16 Tensor Core/resident fallback",
                "block decode + BF16 Tensor Core/resident fallback"),
            Entry(
                "AdamW",
                "CUDA FP32 state/update",
                "CUDA BF16 state/update",
                "CUDA FP32 state/master update",
                "CUDA tensor-scale BFP8 state/update",
                "CUDA FP32 state/master + block requantize",
                tensorCore: false,
                computeContract: CudaPrecisionComputeContract.Elementwise),
            Entry(
                "NekoMuon",
                "CUDA FP32 block reduction/NS5",
                "CUDA BF16 state + FP32 block reduction/NS5",
                "CUDA FP32 state/master + block reduction/NS5",
                "CUDA tensor-scale BFP8 state + block reduction/NS5",
                "CUDA FP32 state/master + block reduction/NS5 + block requantize",
                tensorCore: false,
                computeContract: CudaPrecisionComputeContract.Reduction),
            Entry(
                "elementwise/scalar",
                "CUDA resident FP32 elementwise",
                "CUDA resident BF16 elementwise + BF16 gradient",
                "CUDA resident BF16 elementwise + FP32 gradient",
                "tensor-scale decode/compute/requantize",
                "block decode/compute/block requantize",
                tensorCore: false,
                computeContract: CudaPrecisionComputeContract.Elementwise),
            Entry(
                "activations",
                "CUDA resident FP32 ReLU/GELU/tanh/exp/log/sin/pow",
                "CUDA resident BF16 activation + BF16 gradient",
                "CUDA resident BF16 activation + FP32 gradient",
                "tensor-scale decode/activation/requantize",
                "block decode/activation/block requantize",
                tensorCore: false,
                computeContract: CudaPrecisionComputeContract.Elementwise),
            Entry(
                "sum/mean/max",
                "CUDA block FP32 reduction",
                "CUDA BF16 IO + FP32 reduction + BF16 gradient",
                "CUDA BF16 IO + FP32 reduction/gradient",
                "tensor-scale decode + FP32 reduction",
                "block decode + FP32 reduction",
                tensorCore: false,
                computeContract: CudaPrecisionComputeContract.Reduction),
            Entry(
                "Slice/Concat/Transpose/Reshape/Select",
                "CUDA resident device copy/permutation",
                "CUDA resident BF16 copy/permutation + BF16 gradient",
                "CUDA resident BF16 copy/permutation + FP32 gradient",
                "tensor-scale resident copy/requantize",
                "block-scale resident copy/requantize",
                tensorCore: false,
                computeContract: CudaPrecisionComputeContract.Storage),
            Entry(
                "indexed broadcast",
                "CUDA resident FP32 indexed broadcast",
                "CUDA resident BF16 indexed broadcast + BF16 gradient",
                "CUDA resident BF16 indexed broadcast + FP32 gradient",
                "tensor-scale decode/broadcast/requantize",
                "block decode/broadcast/block requantize",
                tensorCore: false,
                computeContract: CudaPrecisionComputeContract.Elementwise),
            Entry(
                "softmax/logsoftmax/causal-mask",
                "CUDA FP32 row/block reduction",
                "CUDA BF16 IO + FP32 row/block reduction + BF16 gradient",
                "CUDA BF16 IO + FP32 row/block reduction/gradient",
                "tensor-scale decode + FP32 row/block reduction",
                "block decode + FP32 row/block reduction",
                tensorCore: false,
                computeContract: CudaPrecisionComputeContract.Reduction),
            Entry(
                "rank/batched/transposed GEMM",
                "cuBLAS FP32 resident GEMM/dot/matvec",
                "cuBLASLt BF16 Tensor Core; resident CUDA tail",
                "cuBLASLt BF16 Tensor Core; FP32 gradient/master",
                "INT8 Tensor Core or BF16 Tensor Core fallback",
                "block decode + BF16 Tensor Core + block requantize"),
            Entry(
                "ForgetScan",
                "CUDA recurrent FP32",
                "CUDA BF16 IO + FP32 recurrence + BF16 gradient",
                "CUDA BF16 IO + FP32 recurrence/gradient",
                "tensor-scale decode + FP32 recurrence/requantize",
                "block decode + FP32 recurrence/block requantize",
                tensorCore: false,
                computeContract: CudaPrecisionComputeContract.Reduction),
            Entry(
                "Hyena",
                "CUDA direct/parallel causal convolution FP32",
                "CUDA BF16 direct/parallel convolution + BF16 gradient",
                "CUDA BF16 direct/parallel convolution + FP32 gradient",
                "tensor-scale decode + parallel convolution/requantize",
                "block decode + parallel convolution/block requantize",
                tensorCore: false,
                computeContract: CudaPrecisionComputeContract.Reduction),
            Entry(
                "dtype conversion",
                "CUDA resident storage conversion",
                "CUDA resident BF16 encode/decode + BF16 gradient",
                "CUDA resident BF16 encode/decode + FP32 gradient",
                "CUDA tensor-scale quantize/dequantize",
                "CUDA block-scale quantize/dequantize",
                tensorCore: false,
                computeContract: CudaPrecisionComputeContract.Storage),
        ];

    private static CudaPrecisionOperationEntry Entry(
        string operation,
        string float32Backend,
        string bfloat16Backend,
        string mix16Backend,
        string bfp8Backend,
        string mix8Backend,
        bool tensorCore = true,
        CudaPrecisionComputeContract computeContract =
            CudaPrecisionComputeContract.MatrixOperand)
        => new(
            operation,
            computeContract,
            [
                Route(PrecisionPolicy.Float32, float32Backend,
                    tensorCore: false, computeContract),
                Route(PrecisionPolicy.BFloat16, bfloat16Backend,
                    tensorCore, computeContract),
                Route(PrecisionPolicy.Mix16_32, mix16Backend,
                    tensorCore, computeContract),
                Route(PrecisionPolicy.Bfp8, bfp8Backend,
                    tensorCore, computeContract),
                Route(PrecisionPolicy.Mix8_32, mix8Backend,
                    tensorCore, computeContract),
            ]);

    private static CudaPrecisionOperationRoute Route(
        PrecisionPolicy policy,
        string backend,
        bool tensorCore,
        CudaPrecisionComputeContract computeContract)
        => new(
            policy.Mode,
            policy.ActivationStorage,
            computeContract switch
            {
                CudaPrecisionComputeContract.MatrixOperand =>
                    policy.MatrixOperand,
                CudaPrecisionComputeContract.Elementwise =>
                    policy.ElementwiseCompute,
                CudaPrecisionComputeContract.Reduction => policy.Reduction,
                CudaPrecisionComputeContract.Storage =>
                    policy.ActivationStorage,
                _ => throw new InvalidOperationException(
                    "Unknown CUDA precision compute contract."),
            },
            policy.Accumulation,
            policy.Gradient,
            policy.Mode switch
            {
                PrecisionMode.Bfp8 => CudaBfp8ScaleContract.TensorWide,
                PrecisionMode.Mix8_32 => CudaBfp8ScaleContract.Block,
                _ => CudaBfp8ScaleContract.None,
            },
            backend,
            tensorCore,
            AllowsCpuFallback: false);
}
