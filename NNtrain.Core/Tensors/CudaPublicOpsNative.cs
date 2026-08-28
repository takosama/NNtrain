using NNtrain.Cuda.Interop;

namespace NNtrain;

internal enum CudaPublicBinaryOperation
{
    Add = 0,
    Subtract = 1,
    Multiply = 2,
    Divide = 3,
}

internal enum CudaPublicUnaryOperation
{
    Relu = 0,
    Gelu = 1,
    Tanh = 2,
    Exp = 3,
    Log = 4,
    Negate = 5,
    Sin = 6,
    Pow = 7,
}

internal enum CudaPublicReductionOperation
{
    Sum = 0,
    Mean = 1,
    Max = 2,
}

internal static class CudaPublicOpsNative
{
    internal static void Binary(
        int deviceIndex,
        nint left,
        nint right,
        nint output,
        int length,
        bool leftScalar,
        bool rightScalar,
        CudaPublicBinaryOperation operation,
        bool bfloat16,
        nint stream)
        => Check(
            CudaNativeGateway.PublicBinary(
                deviceIndex,
                left,
                right,
                output,
                length,
                leftScalar,
                rightScalar,
                (int)operation,
                bfloat16,
                stream),
            $"CUDA resident {operation}");

    internal static void BinaryBackward(
        int deviceIndex,
        nint left,
        nint right,
        nint outputGradient,
        nint leftGradient,
        nint rightGradient,
        int length,
        bool leftScalar,
        bool rightScalar,
        bool sameParent,
        CudaPublicBinaryOperation operation,
        bool bfloat16,
        nint stream)
        => Check(
            CudaNativeGateway.PublicBinaryBackward(
                deviceIndex,
                left,
                right,
                outputGradient,
                leftGradient,
                rightGradient,
                length,
                leftScalar,
                rightScalar,
                sameParent,
                (int)operation,
                bfloat16,
                stream),
            $"CUDA resident {operation} backward");

    internal static void BinaryBackwardBFloat16Gradient(
        int deviceIndex,
        nint left,
        nint right,
        nint outputGradient,
        nint leftGradient,
        nint rightGradient,
        int length,
        bool leftScalar,
        bool rightScalar,
        bool sameParent,
        CudaPublicBinaryOperation operation,
        nint stream)
        => Check(
            CudaNativeGateway.PublicBinaryBackwardBFloat16Gradient(
                deviceIndex,
                left,
                right,
                outputGradient,
                leftGradient,
                rightGradient,
                length,
                leftScalar,
                rightScalar,
                sameParent,
                (int)operation,
                stream),
            $"pure-BF16 CUDA resident {operation} backward");

    internal static void Unary(
        int deviceIndex,
        nint input,
        nint output,
        int length,
        CudaPublicUnaryOperation operation,
        float parameter,
        bool bfloat16,
        nint stream)
        => Check(
            CudaNativeGateway.PublicUnary(
                deviceIndex,
                input,
                output,
                length,
                (int)operation,
                parameter,
                bfloat16,
                stream),
            $"CUDA resident {operation}");

    internal static void UnaryBackward(
        int deviceIndex,
        nint input,
        nint output,
        nint outputGradient,
        nint inputGradient,
        int length,
        CudaPublicUnaryOperation operation,
        float parameter,
        bool bfloat16,
        nint stream)
        => Check(
            CudaNativeGateway.PublicUnaryBackward(
                deviceIndex,
                input,
                output,
                outputGradient,
                inputGradient,
                length,
                (int)operation,
                parameter,
                bfloat16,
                stream),
            $"CUDA resident {operation} backward");

    internal static void UnaryBackwardBFloat16Gradient(
        int deviceIndex,
        nint input,
        nint output,
        nint outputGradient,
        nint inputGradient,
        int length,
        CudaPublicUnaryOperation operation,
        float parameter,
        nint stream)
        => Check(
            CudaNativeGateway.PublicUnaryBackwardBFloat16Gradient(
                deviceIndex,
                input,
                output,
                outputGradient,
                inputGradient,
                length,
                (int)operation,
                parameter,
                stream),
            $"pure-BF16 CUDA resident {operation} backward");

    internal static void Reduce(
        int deviceIndex,
        nint input,
        nint output,
        int length,
        CudaPublicReductionOperation operation,
        bool bfloat16,
        nint stream)
        => Check(
            CudaNativeGateway.PublicReduce(
                deviceIndex,
                input,
                output,
                length,
                (int)operation,
                bfloat16,
                stream),
            $"CUDA resident {operation} reduction");

    internal static void ReduceBackward(
        int deviceIndex,
        nint input,
        nint reduced,
        nint outputGradient,
        nint inputGradient,
        int length,
        CudaPublicReductionOperation operation,
        bool bfloat16,
        nint stream)
        => Check(
            CudaNativeGateway.PublicReduceBackward(
                deviceIndex,
                input,
                reduced,
                outputGradient,
                inputGradient,
                length,
                (int)operation,
                bfloat16,
                stream),
            $"CUDA resident {operation} reduction backward");

    internal static void ReduceBackwardBFloat16Gradient(
        int deviceIndex,
        nint input,
        nint reduced,
        nint outputGradient,
        nint inputGradient,
        int length,
        CudaPublicReductionOperation operation,
        nint stream)
        => Check(
            CudaNativeGateway.PublicReduceBackwardBFloat16Gradient(
                deviceIndex,
                input,
                reduced,
                outputGradient,
                inputGradient,
                length,
                (int)operation,
                stream),
            $"pure-BF16 CUDA resident {operation} reduction backward");

    internal static void ForgetScan(
        int deviceIndex,
        nint projected,
        nint output,
        nint memory,
        nint forget,
        nint input,
        nint value,
        int batch,
        int sequence,
        int width,
        bool saveContext,
        bool bfloat16,
        nint stream)
        => Check(
            CudaNativeGateway.PublicForgetScan(
                deviceIndex,
                projected,
                output,
                memory,
                forget,
                input,
                value,
                batch,
                sequence,
                width,
                saveContext,
                bfloat16,
                stream),
            "CUDA resident ForgetScan");

    internal static void ForgetScanBackward(
        int deviceIndex,
        nint outputGradient,
        nint memory,
        nint forget,
        nint input,
        nint value,
        nint projectedGradient,
        int batch,
        int sequence,
        int width,
        nint stream)
        => Check(
            CudaNativeGateway.PublicForgetScanBackward(
                deviceIndex,
                outputGradient,
                memory,
                forget,
                input,
                value,
                projectedGradient,
                batch,
                sequence,
                width,
                stream),
            "CUDA resident ForgetScan backward");

    internal static void ForgetScanBackwardBFloat16Gradient(
        int deviceIndex,
        nint outputGradient,
        nint memory,
        nint forget,
        nint input,
        nint value,
        nint projectedGradient,
        int batch,
        int sequence,
        int width,
        nint stream)
        => Check(
            CudaNativeGateway.PublicForgetScanBackwardBFloat16Gradient(
                deviceIndex,
                outputGradient,
                memory,
                forget,
                input,
                value,
                projectedGradient,
                batch,
                sequence,
                width,
                stream),
            "pure-BF16 CUDA resident ForgetScan backward");

    internal static void Hyena(
        int deviceIndex,
        nint projected,
        nint shortFilter,
        nint longFilter,
        nint diagonal,
        nint output,
        nint savedShort,
        nint savedGated,
        nint savedConvolved,
        int batch,
        int sequence,
        int width,
        bool bfloat16,
        bool parallelLong,
        nint stream)
        => Check(
            CudaNativeGateway.PublicHyena(
                deviceIndex,
                projected,
                shortFilter,
                longFilter,
                diagonal,
                output,
                savedShort,
                savedGated,
                savedConvolved,
                batch,
                sequence,
                width,
                bfloat16,
                parallelLong,
                stream),
            "CUDA resident Hyena");

    internal static void HyenaBackward(
        int deviceIndex,
        nint projected,
        nint shortFilter,
        nint longFilter,
        nint diagonal,
        nint outputGradient,
        nint savedShort,
        nint savedGated,
        nint savedConvolved,
        nint projectedGradient,
        nint shortFilterGradient,
        nint longFilterGradient,
        nint diagonalGradient,
        nint shortGradient,
        nint convolutionGradient,
        nint gatedGradient,
        int batch,
        int sequence,
        int width,
        bool bfloat16,
        bool parallelLong,
        bool bfloat16Gradient,
        nint stream)
        => Check(
            CudaNativeGateway.PublicHyenaBackward(
                deviceIndex,
                projected,
                shortFilter,
                longFilter,
                diagonal,
                outputGradient,
                savedShort,
                savedGated,
                savedConvolved,
                projectedGradient,
                shortFilterGradient,
                longFilterGradient,
                diagonalGradient,
                shortGradient,
                convolutionGradient,
                gatedGradient,
                batch,
                sequence,
                width,
                bfloat16,
                parallelLong,
                bfloat16Gradient,
                stream),
            "CUDA resident Hyena backward");

    internal static void BroadcastAdd(
        int deviceIndex,
        nint input,
        nint addend,
        nint output,
        int length,
        int repeatLength,
        bool bfloat16,
        nint stream)
        => Check(
            CudaNativeGateway.PublicBroadcastAdd(
                deviceIndex,
                input,
                addend,
                output,
                length,
                repeatLength,
                bfloat16,
                stream),
            "CUDA resident indexed broadcast addition");

    internal static void ShapeAccumulateBFloat16Gradient(
        int deviceIndex,
        nint source,
        nint destination,
        int length,
        int sourceOffset,
        int destinationOffset,
        nint stream)
        => Check(
            CudaNativeGateway.PublicShapeAccumulateBFloat16Gradient(
                deviceIndex,
                source,
                destination,
                length,
                sourceOffset,
                destinationOffset,
                stream),
            "pure-BF16 CUDA shape gradient accumulation");

    internal static void TransposeBFloat16(
        int deviceIndex,
        nint input,
        nint output,
        int rows,
        int columns,
        nint stream)
        => Check(
            CudaNativeGateway.PublicTransposeBFloat16(
                deviceIndex, input, output, rows, columns, stream),
            "CUDA resident BF16 transpose");

    internal static void TransposeBackwardBFloat16Gradient(
        int deviceIndex,
        nint outputGradient,
        nint inputGradient,
        int rows,
        int columns,
        nint stream)
        => Check(
            CudaNativeGateway.PublicTransposeBackwardBFloat16Gradient(
                deviceIndex,
                outputGradient,
                inputGradient,
                rows,
                columns,
                stream),
            "pure-BF16 CUDA transpose gradient");

    internal static void BroadcastAddBackward(
        int deviceIndex,
        nint outputGradient,
        nint inputGradient,
        nint addendGradient,
        int length,
        int repeatLength,
        nint stream)
        => Check(
            CudaNativeGateway.PublicBroadcastAddBackward(
                deviceIndex,
                outputGradient,
                inputGradient,
                addendGradient,
                length,
                repeatLength,
                stream),
            "CUDA resident indexed broadcast addition backward");

    internal static void BroadcastAddBackwardBFloat16Gradient(
        int deviceIndex,
        nint outputGradient,
        nint inputGradient,
        nint addendGradient,
        int length,
        int repeatLength,
        nint stream)
        => Check(
            CudaNativeGateway.PublicBroadcastAddBackwardBFloat16Gradient(
                deviceIndex,
                outputGradient,
                inputGradient,
                addendGradient,
                length,
                repeatLength,
                stream),
            "pure-BF16 CUDA indexed broadcast addition backward");

    internal static void CausalMask(
        int deviceIndex,
        nint input,
        nint output,
        int length,
        int rows,
        int columns,
        float fillValue,
        bool bfloat16,
        nint stream)
        => Check(
            CudaNativeGateway.PublicCausalMask(
                deviceIndex,
                input,
                output,
                length,
                rows,
                columns,
                fillValue,
                bfloat16,
                stream),
            "CUDA resident causal mask");

    internal static void CausalMaskBackward(
        int deviceIndex,
        nint outputGradient,
        nint inputGradient,
        int length,
        int rows,
        int columns,
        nint stream)
        => Check(
            CudaNativeGateway.PublicCausalMaskBackward(
                deviceIndex,
                outputGradient,
                inputGradient,
                length,
                rows,
                columns,
                stream),
            "CUDA resident causal mask backward");

    internal static void CausalMaskBackwardBFloat16Gradient(
        int deviceIndex,
        nint outputGradient,
        nint inputGradient,
        int length,
        int rows,
        int columns,
        nint stream)
        => Check(
            CudaNativeGateway.PublicCausalMaskBackwardBFloat16Gradient(
                deviceIndex,
                outputGradient,
                inputGradient,
                length,
                rows,
                columns,
                stream),
            "pure-BF16 CUDA causal mask backward");

    internal static void Softmax(
        int deviceIndex,
        nint input,
        nint output,
        nint probabilities,
        int rows,
        int columns,
        bool logSoftmax,
        bool bfloat16,
        nint stream)
        => Check(
            CudaNativeGateway.PublicSoftmax(
                deviceIndex,
                input,
                output,
                probabilities,
                rows,
                columns,
                logSoftmax,
                bfloat16,
                stream),
            logSoftmax
                ? "CUDA resident log-softmax"
                : "CUDA resident softmax");

    internal static void SoftmaxBackward(
        int deviceIndex,
        nint probabilities,
        nint outputGradient,
        nint inputGradient,
        int rows,
        int columns,
        bool logSoftmax,
        nint stream)
        => Check(
            CudaNativeGateway.PublicSoftmaxBackward(
                deviceIndex,
                probabilities,
                outputGradient,
                inputGradient,
                rows,
                columns,
                logSoftmax,
                stream),
            logSoftmax
                ? "CUDA resident log-softmax backward"
                : "CUDA resident softmax backward");

    internal static void SoftmaxBackwardBFloat16Gradient(
        int deviceIndex,
        nint probabilities,
        nint outputGradient,
        nint inputGradient,
        int rows,
        int columns,
        bool logSoftmax,
        nint stream)
        => Check(
            CudaNativeGateway.PublicSoftmaxBackwardBFloat16Gradient(
                deviceIndex,
                probabilities,
                outputGradient,
                inputGradient,
                rows,
                columns,
                logSoftmax,
                stream),
            logSoftmax
                ? "pure-BF16 CUDA log-softmax backward"
                : "pure-BF16 CUDA softmax backward");

    private static void Check(int status, string operation)
        => NativeCudaRuntime.Check(status, operation);
}
