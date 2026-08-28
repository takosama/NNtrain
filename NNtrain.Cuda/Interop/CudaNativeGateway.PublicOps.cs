using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// CUDA-resident public tensor primitives.  Storage format is selected by the
/// caller while reductions and all gradient accumulation remain Float32.
/// </summary>
public static partial class CudaNativeGateway
{
    public static int PublicBinary(
        int device,
        nint left,
        nint right,
        nint output,
        int length,
        bool leftScalar,
        bool rightScalar,
        int operation,
        bool bfloat16,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA public tensor operations");
        int status = bfloat16
            ? PublicOpsNativeMethods.BinaryBFloat16(
                left, right, output, length, leftScalar ? 1 : 0,
                rightScalar ? 1 : 0, operation, stream)
            : PublicOpsNativeMethods.BinaryFloat32(
                left, right, output, length, leftScalar ? 1 : 0,
                rightScalar ? 1 : 0, operation, stream);
        return Complete(status, CudaNativeOperation.PublicTensorOps, device);
    }

    public static int PublicBinaryBackward(
        int device,
        nint left,
        nint right,
        nint outputGradient,
        nint leftGradient,
        nint rightGradient,
        int length,
        bool leftScalar,
        bool rightScalar,
        bool sameParent,
        int operation,
        bool bfloat16,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA public tensor operation gradients");
        int status = bfloat16
            ? PublicOpsNativeMethods.BinaryBackwardBFloat16(
                left, right, outputGradient, leftGradient, rightGradient,
                length, leftScalar ? 1 : 0, rightScalar ? 1 : 0,
                sameParent ? 1 : 0, operation, stream)
            : PublicOpsNativeMethods.BinaryBackwardFloat32(
                left, right, outputGradient, leftGradient, rightGradient,
                length, leftScalar ? 1 : 0, rightScalar ? 1 : 0,
                sameParent ? 1 : 0, operation, stream);
        return Complete(status, CudaNativeOperation.PublicTensorOps, device);
    }

    public static int PublicBinaryBackwardBFloat16Gradient(
        int device,
        nint left,
        nint right,
        nint outputGradient,
        nint leftGradient,
        nint rightGradient,
        int length,
        bool leftScalar,
        bool rightScalar,
        bool sameParent,
        int operation,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "pure-BF16 CUDA public tensor operation gradients");
        return Complete(
            PublicOpsNativeMethods.BinaryBackwardBFloat16Gradient(
                left, right, outputGradient, leftGradient, rightGradient,
                length, leftScalar ? 1 : 0, rightScalar ? 1 : 0,
                sameParent ? 1 : 0, operation, stream),
            CudaNativeOperation.PublicTensorOps,
            device);
    }

    public static int PublicUnary(
        int device,
        nint input,
        nint output,
        int length,
        int operation,
        float parameter,
        bool bfloat16,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA public tensor operations");
        int status = bfloat16
            ? PublicOpsNativeMethods.UnaryBFloat16(
                input, output, length, operation, parameter, stream)
            : PublicOpsNativeMethods.UnaryFloat32(
                input, output, length, operation, parameter, stream);
        return Complete(status, CudaNativeOperation.PublicTensorOps, device);
    }

    public static int PublicUnaryBackward(
        int device,
        nint input,
        nint output,
        nint outputGradient,
        nint inputGradient,
        int length,
        int operation,
        float parameter,
        bool bfloat16,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA public tensor operation gradients");
        int status = bfloat16
            ? PublicOpsNativeMethods.UnaryBackwardBFloat16(
                input, output, outputGradient, inputGradient, length,
                operation, parameter, stream)
            : PublicOpsNativeMethods.UnaryBackwardFloat32(
                input, output, outputGradient, inputGradient, length,
                operation, parameter, stream);
        return Complete(status, CudaNativeOperation.PublicTensorOps, device);
    }

    public static int PublicUnaryBackwardBFloat16Gradient(
        int device,
        nint input,
        nint output,
        nint outputGradient,
        nint inputGradient,
        int length,
        int operation,
        float parameter,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "pure-BF16 CUDA public tensor operation gradients");
        return Complete(
            PublicOpsNativeMethods.UnaryBackwardBFloat16Gradient(
                input, output, outputGradient, inputGradient,
                length, operation, parameter, stream),
            CudaNativeOperation.PublicTensorOps,
            device);
    }

    public static int PublicReduce(
        int device,
        nint input,
        nint output,
        int length,
        int operation,
        bool bfloat16,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA public tensor reductions");
        int status = bfloat16
            ? PublicOpsNativeMethods.ReduceBFloat16(
                input, output, length, operation, stream)
            : PublicOpsNativeMethods.ReduceFloat32(
                input, output, length, operation, stream);
        return Complete(status, CudaNativeOperation.PublicTensorOps, device);
    }

    public static int PublicReduceBackward(
        int device,
        nint input,
        nint reduced,
        nint outputGradient,
        nint inputGradient,
        int length,
        int operation,
        bool bfloat16,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA public tensor reduction gradients");
        int status = bfloat16
            ? PublicOpsNativeMethods.ReduceBackwardBFloat16(
                input, reduced, outputGradient, inputGradient, length,
                operation, stream)
            : PublicOpsNativeMethods.ReduceBackwardFloat32(
                input, reduced, outputGradient, inputGradient, length,
                operation, stream);
        return Complete(status, CudaNativeOperation.PublicTensorOps, device);
    }

    public static int PublicReduceBackwardBFloat16Gradient(
        int device,
        nint input,
        nint reduced,
        nint outputGradient,
        nint inputGradient,
        int length,
        int operation,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "pure-BF16 CUDA public tensor reduction gradients");
        return Complete(
            PublicOpsNativeMethods.ReduceBackwardBFloat16Gradient(
                input, reduced, outputGradient, inputGradient,
                length, operation, stream),
            CudaNativeOperation.PublicTensorOps,
            device);
    }

    public static int PublicForgetScan(
        int device,
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
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA resident ForgetScan");
        int status = bfloat16
            ? PublicOpsNativeMethods.ForgetScanBFloat16(
                projected, output, memory, forget, input, value,
                batch, sequence, width, saveContext ? 1 : 0, stream)
            : PublicOpsNativeMethods.ForgetScanFloat32(
                projected, output, memory, forget, input, value,
                batch, sequence, width, saveContext ? 1 : 0, stream);
        return Complete(status, CudaNativeOperation.PublicTensorOps, device);
    }

    public static int PublicForgetScanBackward(
        int device,
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
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA resident ForgetScan gradient");
        return Complete(
            PublicOpsNativeMethods.ForgetScanBackward(
                outputGradient, memory, forget, input, value,
                projectedGradient, batch, sequence, width, stream),
            CudaNativeOperation.PublicTensorOps,
            device);
    }

    public static int PublicForgetScanBackwardBFloat16Gradient(
        int device,
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
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "pure-BF16 CUDA ForgetScan gradient");
        return Complete(
            PublicOpsNativeMethods.ForgetScanBackwardBFloat16Gradient(
                outputGradient, memory, forget, input, value,
                projectedGradient, batch, sequence, width, stream),
            CudaNativeOperation.PublicTensorOps,
            device);
    }

    public static int PublicHyena(
        int device,
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
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA resident Hyena");
        int status = (bfloat16, parallelLong) switch
        {
            (true, true) => PublicOpsNativeMethods.HyenaParallelBFloat16(
                projected, shortFilter, longFilter, diagonal, output,
                savedShort, savedGated, savedConvolved,
                batch, sequence, width, stream),
            (true, false) => PublicOpsNativeMethods.HyenaBFloat16(
                projected, shortFilter, longFilter, diagonal, output,
                savedShort, savedGated, savedConvolved,
                batch, sequence, width, stream),
            (false, true) => PublicOpsNativeMethods.HyenaParallelFloat32(
                projected, shortFilter, longFilter, diagonal, output,
                savedShort, savedGated, savedConvolved,
                batch, sequence, width, stream),
            _ => PublicOpsNativeMethods.HyenaFloat32(
                projected, shortFilter, longFilter, diagonal, output,
                savedShort, savedGated, savedConvolved,
                batch, sequence, width, stream),
        };
        return Complete(status, CudaNativeOperation.PublicTensorOps, device);
    }

    public static int PublicHyenaBackward(
        int device,
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
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA resident Hyena gradient");
        int status;
        if (bfloat16Gradient)
        {
            if (!bfloat16)
                throw new ArgumentException(
                    "BF16 Hyena gradients require BF16 operands.",
                    nameof(bfloat16Gradient));
            status = PublicOpsNativeMethods.HyenaBackwardBFloat16Gradient(
                projected, shortFilter, longFilter, diagonal,
                outputGradient, savedShort, savedGated, savedConvolved,
                projectedGradient, shortFilterGradient, longFilterGradient,
                diagonalGradient, shortGradient, convolutionGradient,
                gatedGradient, batch, sequence, width, stream);
        }
        else if (parallelLong && bfloat16)
        {
            status = PublicOpsNativeMethods.HyenaBackwardParallelBFloat16(
                projected, shortFilter, longFilter, diagonal,
                outputGradient, savedShort, savedGated, savedConvolved,
                projectedGradient, shortFilterGradient, longFilterGradient,
                diagonalGradient, shortGradient, convolutionGradient,
                gatedGradient, batch, sequence, width, stream);
        }
        else if (parallelLong)
        {
            status = PublicOpsNativeMethods.HyenaBackwardParallelFloat32(
                projected, shortFilter, longFilter, diagonal,
                outputGradient, savedShort, savedGated, savedConvolved,
                projectedGradient, shortFilterGradient, longFilterGradient,
                diagonalGradient, shortGradient, convolutionGradient,
                gatedGradient, batch, sequence, width, stream);
        }
        else
        {
            status = bfloat16
                ? PublicOpsNativeMethods.HyenaBackwardBFloat16(
                    projected, shortFilter, longFilter, diagonal,
                    outputGradient, savedShort, savedGated, savedConvolved,
                    projectedGradient, shortFilterGradient,
                    longFilterGradient, diagonalGradient, shortGradient,
                    convolutionGradient, gatedGradient,
                    batch, sequence, width, stream)
                : PublicOpsNativeMethods.HyenaBackwardFloat32(
                    projected, shortFilter, longFilter, diagonal,
                    outputGradient, savedShort, savedGated, savedConvolved,
                    projectedGradient, shortFilterGradient,
                    longFilterGradient, diagonalGradient, shortGradient,
                    convolutionGradient, gatedGradient,
                    batch, sequence, width, stream);
        }
        return Complete(status, CudaNativeOperation.PublicTensorOps, device);
    }

    public static int PublicBroadcastAdd(
        int device,
        nint input,
        nint addend,
        nint output,
        int length,
        int repeatLength,
        bool bfloat16,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA resident indexed broadcast addition");
        int status = bfloat16
            ? PublicOpsNativeMethods.BroadcastAddBFloat16(
                input, addend, output, length, repeatLength, stream)
            : PublicOpsNativeMethods.BroadcastAddFloat32(
                input, addend, output, length, repeatLength, stream);
        return Complete(status, CudaNativeOperation.PublicTensorOps, device);
    }

    public static int PublicShapeAccumulateBFloat16Gradient(
        int device,
        nint source,
        nint destination,
        int length,
        int sourceOffset,
        int destinationOffset,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "pure-BF16 CUDA shape gradient accumulation");
        return Complete(
            PublicOpsNativeMethods.ShapeAccumulateBFloat16Gradient(
                source, destination, length, sourceOffset,
                destinationOffset, stream),
            CudaNativeOperation.PublicTensorOps,
            device);
    }

    public static int PublicTransposeBFloat16(
        int device,
        nint input,
        nint output,
        int rows,
        int columns,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA resident BF16 transpose");
        return Complete(
            PublicOpsNativeMethods.TransposeBFloat16(
                input, output, rows, columns, stream),
            CudaNativeOperation.PublicTensorOps,
            device);
    }

    public static int PublicTransposeBackwardBFloat16Gradient(
        int device,
        nint outputGradient,
        nint inputGradient,
        int rows,
        int columns,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "pure-BF16 CUDA transpose gradient");
        return Complete(
            PublicOpsNativeMethods.TransposeBackwardBFloat16Gradient(
                outputGradient, inputGradient, rows, columns, stream),
            CudaNativeOperation.PublicTensorOps,
            device);
    }

    public static int PublicBroadcastAddBackward(
        int device,
        nint outputGradient,
        nint inputGradient,
        nint addendGradient,
        int length,
        int repeatLength,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA resident indexed broadcast addition gradient");
        return Complete(
            PublicOpsNativeMethods.BroadcastAddBackward(
                outputGradient, inputGradient, addendGradient,
                length, repeatLength, stream),
            CudaNativeOperation.PublicTensorOps,
            device);
    }

    public static int PublicBroadcastAddBackwardBFloat16Gradient(
        int device,
        nint outputGradient,
        nint inputGradient,
        nint addendGradient,
        int length,
        int repeatLength,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "pure-BF16 CUDA indexed broadcast addition gradient");
        return Complete(
            PublicOpsNativeMethods.BroadcastAddBackwardBFloat16Gradient(
                outputGradient, inputGradient, addendGradient,
                length, repeatLength, stream),
            CudaNativeOperation.PublicTensorOps,
            device);
    }

    public static int PublicCausalMask(
        int device,
        nint input,
        nint output,
        int length,
        int rows,
        int columns,
        float fillValue,
        bool bfloat16,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA resident causal mask");
        int status = bfloat16
            ? PublicOpsNativeMethods.CausalMaskBFloat16(
                input, output, length, rows, columns, fillValue, stream)
            : PublicOpsNativeMethods.CausalMaskFloat32(
                input, output, length, rows, columns, fillValue, stream);
        return Complete(status, CudaNativeOperation.PublicTensorOps, device);
    }

    public static int PublicCausalMaskBackward(
        int device,
        nint outputGradient,
        nint inputGradient,
        int length,
        int rows,
        int columns,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA resident causal mask gradient");
        return Complete(
            PublicOpsNativeMethods.CausalMaskBackward(
                outputGradient, inputGradient, length, rows, columns, stream),
            CudaNativeOperation.PublicTensorOps,
            device);
    }

    public static int PublicCausalMaskBackwardBFloat16Gradient(
        int device,
        nint outputGradient,
        nint inputGradient,
        int length,
        int rows,
        int columns,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "pure-BF16 CUDA causal mask gradient");
        return Complete(
            PublicOpsNativeMethods.CausalMaskBackwardBFloat16Gradient(
                outputGradient, inputGradient, length, rows, columns, stream),
            CudaNativeOperation.PublicTensorOps,
            device);
    }

    public static int PublicSoftmax(
        int device,
        nint input,
        nint output,
        nint probabilities,
        int rows,
        int columns,
        bool logSoftmax,
        bool bfloat16,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA resident softmax");
        int status = bfloat16
            ? PublicOpsNativeMethods.SoftmaxBFloat16(
                input, output, probabilities, rows, columns,
                logSoftmax ? 1 : 0, stream)
            : PublicOpsNativeMethods.SoftmaxFloat32(
                input, output, probabilities, rows, columns,
                logSoftmax ? 1 : 0, stream);
        return Complete(status, CudaNativeOperation.PublicTensorOps, device);
    }

    public static int PublicSoftmaxBackward(
        int device,
        nint probabilities,
        nint outputGradient,
        nint inputGradient,
        int rows,
        int columns,
        bool logSoftmax,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "CUDA resident softmax gradient");
        return Complete(
            PublicOpsNativeMethods.SoftmaxBackward(
                probabilities, outputGradient, inputGradient,
                rows, columns, logSoftmax ? 1 : 0, stream),
            CudaNativeOperation.PublicTensorOps,
            device);
    }

    public static int PublicSoftmaxBackwardBFloat16Gradient(
        int device,
        nint probabilities,
        nint outputGradient,
        nint inputGradient,
        int rows,
        int columns,
        bool logSoftmax,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.PublicTensorOpsMinor,
            "pure-BF16 CUDA softmax gradient");
        return Complete(
            PublicOpsNativeMethods.SoftmaxBackwardBFloat16Gradient(
                probabilities, outputGradient, inputGradient,
                rows, columns, logSoftmax ? 1 : 0, stream),
            CudaNativeOperation.PublicTensorOps,
            device);
    }

    private static class PublicOpsNativeMethods
    {
        [DllImport(LibraryName, EntryPoint = "nntrain_public_binary_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BinaryFloat32(
            nint left, nint right, nint output, int length,
            int leftScalar, int rightScalar, int operation, nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_public_binary_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BinaryBFloat16(
            nint left, nint right, nint output, int length,
            int leftScalar, int rightScalar, int operation, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_binary_backward_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BinaryBackwardFloat32(
            nint left, nint right, nint outputGradient,
            nint leftGradient, nint rightGradient, int length,
            int leftScalar, int rightScalar, int sameParent, int operation,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_binary_backward_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BinaryBackwardBFloat16(
            nint left, nint right, nint outputGradient,
            nint leftGradient, nint rightGradient, int length,
            int leftScalar, int rightScalar, int sameParent, int operation,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_binary_backward_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BinaryBackwardBFloat16Gradient(
            nint left, nint right, nint outputGradient,
            nint leftGradient, nint rightGradient, int length,
            int leftScalar, int rightScalar, int sameParent, int operation,
            nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_public_unary_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int UnaryFloat32(
            nint input, nint output, int length, int operation,
            float parameter, nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_public_unary_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int UnaryBFloat16(
            nint input, nint output, int length, int operation,
            float parameter, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_unary_backward_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int UnaryBackwardFloat32(
            nint input, nint output, nint outputGradient,
            nint inputGradient, int length, int operation,
            float parameter, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_unary_backward_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int UnaryBackwardBFloat16(
            nint input, nint output, nint outputGradient,
            nint inputGradient, int length, int operation,
            float parameter, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_unary_backward_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int UnaryBackwardBFloat16Gradient(
            nint input, nint output, nint outputGradient,
            nint inputGradient, int length, int operation,
            float parameter, nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_public_reduce_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ReduceFloat32(
            nint input, nint output, int length, int operation, nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_public_reduce_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ReduceBFloat16(
            nint input, nint output, int length, int operation, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_reduce_backward_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ReduceBackwardFloat32(
            nint input, nint reduced, nint outputGradient,
            nint inputGradient, int length, int operation, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_reduce_backward_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ReduceBackwardBFloat16(
            nint input, nint reduced, nint outputGradient,
            nint inputGradient, int length, int operation, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_reduce_backward_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ReduceBackwardBFloat16Gradient(
            nint input, nint reduced, nint outputGradient,
            nint inputGradient, int length, int operation, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_forget_scan_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ForgetScanFloat32(
            nint projected, nint output, nint memory, nint forget,
            nint input, nint value, int batch, int sequence, int width,
            int saveContext, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_forget_scan_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ForgetScanBFloat16(
            nint projected, nint output, nint memory, nint forget,
            nint input, nint value, int batch, int sequence, int width,
            int saveContext, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_forget_scan_backward",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ForgetScanBackward(
            nint outputGradient, nint memory, nint forget, nint input,
            nint value, nint projectedGradient, int batch, int sequence,
            int width, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_forget_scan_backward_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ForgetScanBackwardBFloat16Gradient(
            nint outputGradient, nint memory, nint forget, nint input,
            nint value, nint projectedGradient, int batch, int sequence,
            int width, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_hyena_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int HyenaFloat32(
            nint projected, nint shortFilter, nint longFilter,
            nint diagonal, nint output, nint savedShort, nint savedGated,
            nint savedConvolved, int batch, int sequence, int width,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_hyena_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int HyenaBFloat16(
            nint projected, nint shortFilter, nint longFilter,
            nint diagonal, nint output, nint savedShort, nint savedGated,
            nint savedConvolved, int batch, int sequence, int width,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_hyena_parallel_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int HyenaParallelFloat32(
            nint projected, nint shortFilter, nint longFilter,
            nint diagonal, nint output, nint savedShort, nint savedGated,
            nint savedConvolved, int batch, int sequence, int width,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_hyena_parallel_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int HyenaParallelBFloat16(
            nint projected, nint shortFilter, nint longFilter,
            nint diagonal, nint output, nint savedShort, nint savedGated,
            nint savedConvolved, int batch, int sequence, int width,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_hyena_backward_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int HyenaBackwardFloat32(
            nint projected, nint shortFilter, nint longFilter,
            nint diagonal, nint outputGradient, nint savedShort,
            nint savedGated, nint savedConvolved, nint projectedGradient,
            nint shortFilterGradient, nint longFilterGradient,
            nint diagonalGradient, nint shortGradient,
            nint convolutionGradient, nint gatedGradient,
            int batch, int sequence, int width, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_hyena_backward_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int HyenaBackwardBFloat16(
            nint projected, nint shortFilter, nint longFilter,
            nint diagonal, nint outputGradient, nint savedShort,
            nint savedGated, nint savedConvolved, nint projectedGradient,
            nint shortFilterGradient, nint longFilterGradient,
            nint diagonalGradient, nint shortGradient,
            nint convolutionGradient, nint gatedGradient,
            int batch, int sequence, int width, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_hyena_backward_parallel_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int HyenaBackwardParallelFloat32(
            nint projected, nint shortFilter, nint longFilter,
            nint diagonal, nint outputGradient, nint savedShort,
            nint savedGated, nint savedConvolved, nint projectedGradient,
            nint shortFilterGradient, nint longFilterGradient,
            nint diagonalGradient, nint shortGradient,
            nint convolutionGradient, nint gatedGradient,
            int batch, int sequence, int width, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_hyena_backward_parallel_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int HyenaBackwardParallelBFloat16(
            nint projected, nint shortFilter, nint longFilter,
            nint diagonal, nint outputGradient, nint savedShort,
            nint savedGated, nint savedConvolved, nint projectedGradient,
            nint shortFilterGradient, nint longFilterGradient,
            nint diagonalGradient, nint shortGradient,
            nint convolutionGradient, nint gatedGradient,
            int batch, int sequence, int width, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_hyena_backward_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int HyenaBackwardBFloat16Gradient(
            nint projected, nint shortFilter, nint longFilter,
            nint diagonal, nint outputGradient, nint savedShort,
            nint savedGated, nint savedConvolved, nint projectedGradient,
            nint shortFilterGradient, nint longFilterGradient,
            nint diagonalGradient, nint shortGradient,
            nint convolutionGradient, nint gatedGradient,
            int batch, int sequence, int width, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_shape_accumulate_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ShapeAccumulateBFloat16Gradient(
            nint source, nint destination, int length,
            int sourceOffset, int destinationOffset, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_transpose_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int TransposeBFloat16(
            nint input, nint output, int rows, int columns, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_transpose_backward_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int TransposeBackwardBFloat16Gradient(
            nint outputGradient, nint inputGradient,
            int rows, int columns, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_broadcast_add_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BroadcastAddFloat32(
            nint input, nint addend, nint output,
            int length, int repeatLength, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_broadcast_add_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BroadcastAddBFloat16(
            nint input, nint addend, nint output,
            int length, int repeatLength, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_broadcast_add_backward",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BroadcastAddBackward(
            nint outputGradient, nint inputGradient, nint addendGradient,
            int length, int repeatLength, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_broadcast_add_backward_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BroadcastAddBackwardBFloat16Gradient(
            nint outputGradient, nint inputGradient, nint addendGradient,
            int length, int repeatLength, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_causal_mask_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CausalMaskFloat32(
            nint input, nint output, int length, int rows, int columns,
            float fillValue, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_causal_mask_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CausalMaskBFloat16(
            nint input, nint output, int length, int rows, int columns,
            float fillValue, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_causal_mask_backward",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CausalMaskBackward(
            nint outputGradient, nint inputGradient,
            int length, int rows, int columns, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_causal_mask_backward_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CausalMaskBackwardBFloat16Gradient(
            nint outputGradient, nint inputGradient,
            int length, int rows, int columns, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_softmax_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int SoftmaxFloat32(
            nint input, nint output, nint probabilities,
            int rows, int columns, int logSoftmax, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_softmax_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int SoftmaxBFloat16(
            nint input, nint output, nint probabilities,
            int rows, int columns, int logSoftmax, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_softmax_backward",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int SoftmaxBackward(
            nint probabilities, nint outputGradient, nint inputGradient,
            int rows, int columns, int logSoftmax, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_public_softmax_backward_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int SoftmaxBackwardBFloat16Gradient(
            nint probabilities, nint outputGradient, nint inputGradient,
            int rows, int columns, int logSoftmax, nint stream);
    }
}
