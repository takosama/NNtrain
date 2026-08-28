using System.Runtime.CompilerServices;
using NNtrain.Cuda.Interop;
using NNtrain.Runtime.Execution;

namespace NNtrain;

/// <summary>Thin cuBLAS binding over native CUDA Runtime buffers.</summary>
internal static unsafe class CudaBlas
{
    private const int Success = 0;
    private const int OperationNone = 0;
    private const int OperationTranspose = 1;
    private const int CudaR32F = 0;
    private const int CudaR16BF = 14;
    private const int Compute32FFast16BFloat = 75;
    private const int GemmDefault = -1;
    private const int GemmDefaultTensorOp = 99;

    private const int FallbackHandleCapacity = 8;
    private static readonly ResettableBoundedDisposableLeaseCache<
        HandleKey,
        LegacyHandle> FallbackHandles = new(FallbackHandleCapacity);
    private static readonly ConditionalWeakTable<
        IStreamExecutionLane,
        Lazy<LaneHandle>> LaneHandles = new();
    private static int _activeLaneHandleCount;

    internal static int ActiveLaneHandleCount =>
        Volatile.Read(ref _activeLaneHandleCount);

    internal static int FallbackHandleCount => FallbackHandles.Count;

    internal static void DisposeFallbackResources()
        => FallbackHandles.Dispose();

    internal static void LinearForward(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<float> input,
        NativeCudaBuffer<float> weight,
        NativeCudaBuffer<float> output,
        int rows,
        int inputWidth,
        int outputWidth,
        bool bfloat16)
    {
        accelerator.Bind();
        using PreparedHandle preparedHandle = PrepareHandle(
            accelerator,
            deviceIndex);
        nint handle = preparedHandle.Handle;
        float alpha = 1f;
        float beta = 0f;

        // The tensors are row-major. cuBLAS is column-major, so calculate
        // Y^T = W * X^T without allocating transposed copies.
        int status = CublasGemmEx(
            handle,
            OperationTranspose,
            OperationNone,
            outputWidth,
            rows,
            inputWidth,
            (nint)(&alpha),
            weight.NativePtr,
            CudaR32F,
            inputWidth,
            input.NativePtr,
            CudaR32F,
            inputWidth,
            (nint)(&beta),
            output.NativePtr,
            CudaR32F,
            outputWidth,
            bfloat16 ? Compute32FFast16BFloat : 68,
            bfloat16 ? GemmDefaultTensorOp : GemmDefault);
        ThrowIfFailed(status, "cublasGemmEx");
    }

    internal static void LinearForwardBFloat16(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> input,
        NativeCudaBuffer<ushort> weight,
        NativeCudaBuffer<ushort> output,
        int rows,
        int inputWidth,
        int outputWidth)
    {
        accelerator.Bind();
        using PreparedHandle preparedHandle = PrepareHandle(
            accelerator,
            deviceIndex);
        nint handle = preparedHandle.Handle;
        float alpha = 1f;
        float beta = 0f;
        int status = CublasGemmEx(
            handle,
            OperationTranspose,
            OperationNone,
            outputWidth,
            rows,
            inputWidth,
            (nint)(&alpha),
            weight.NativePtr,
            CudaR16BF,
            inputWidth,
            input.NativePtr,
            CudaR16BF,
            inputWidth,
            (nint)(&beta),
            output.NativePtr,
            CudaR16BF,
            outputWidth,
            Compute32FFast16BFloat,
            GemmDefaultTensorOp);
        ThrowIfFailed(status, "cublasGemmEx(BF16)");
    }

    internal static void LinearBackwardInputBFloat16(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> outputGradient,
        NativeCudaBuffer<ushort> weight,
        NativeCudaBuffer<float> inputGradient,
        int rows,
        int inputWidth,
        int outputWidth)
    {
        GemmBFloat16ToFloat32(
            accelerator,
            deviceIndex,
            OperationNone,
            OperationNone,
            inputWidth,
            rows,
            outputWidth,
            weight,
            inputWidth,
            outputGradient,
            outputWidth,
            inputGradient,
            inputWidth,
            beta: 1f);
        CudaBlasLt.RecordAccumulatingBackwardCublasExecution();
    }

    internal static void LinearBackwardInputBFloat16Direct(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> outputGradient,
        NativeCudaBuffer<ushort> weight,
        NativeCudaBuffer<ushort> inputGradient,
        int rows,
        int inputWidth,
        int outputWidth)
    {
        if (CudaBlasLt.TryLinearBackwardInputBFloat16Direct(
                accelerator,
                deviceIndex,
                outputGradient,
                weight,
                inputGradient,
                rows,
                inputWidth,
                outputWidth))
        {
            return;
        }
        accelerator.Bind();
        using PreparedHandle preparedHandle = PrepareHandle(
            accelerator,
            deviceIndex);
        nint handle = preparedHandle.Handle;
        float alpha = 1f;
        float beta = 0f;
        int status = CublasGemmEx(
            handle,
            OperationNone,
            OperationNone,
            inputWidth,
            rows,
            outputWidth,
            (nint)(&alpha),
            weight.NativePtr,
            CudaR16BF,
            inputWidth,
            outputGradient.NativePtr,
            CudaR16BF,
            outputWidth,
            (nint)(&beta),
            inputGradient.NativePtr,
            CudaR16BF,
            inputWidth,
            Compute32FFast16BFloat,
            GemmDefaultTensorOp);
        ThrowIfFailed(status, "cublasGemmEx(BF16 direct gradient)");
    }

    internal static void LinearBackwardInputBFloat16Accumulate(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> outputGradient,
        NativeCudaBuffer<ushort> weight,
        NativeCudaBuffer<ushort> inputGradient,
        int rows,
        int inputWidth,
        int outputWidth,
        bool accumulate)
        => GemmBFloat16(
            accelerator,
            deviceIndex,
            OperationNone,
            OperationNone,
            inputWidth,
            rows,
            outputWidth,
            weight,
            inputWidth,
            outputGradient,
            outputWidth,
            inputGradient,
            inputWidth,
            accumulate ? 1f : 0f);

    internal static void LinearBackwardWeightBFloat16(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> input,
        NativeCudaBuffer<ushort> outputGradient,
        NativeCudaBuffer<float> weightGradient,
        int rows,
        int inputWidth,
        int outputWidth)
    {
        GemmBFloat16ToFloat32(
            accelerator,
            deviceIndex,
            OperationNone,
            OperationTranspose,
            inputWidth,
            outputWidth,
            rows,
            input,
            inputWidth,
            outputGradient,
            outputWidth,
            weightGradient,
            inputWidth,
            beta: 1f);
        CudaBlasLt.RecordAccumulatingBackwardCublasExecution();
    }

    internal static void LinearBackwardWeightBFloat16Direct(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> input,
        NativeCudaBuffer<ushort> outputGradient,
        NativeCudaBuffer<ushort> weightGradient,
        int rows,
        int inputWidth,
        int outputWidth,
        bool accumulate)
        => GemmBFloat16(
            accelerator,
            deviceIndex,
            OperationNone,
            OperationTranspose,
            inputWidth,
            outputWidth,
            rows,
            input,
            inputWidth,
            outputGradient,
            outputWidth,
            weightGradient,
            inputWidth,
            accumulate ? 1f : 0f);

    internal static void LinearBackwardInput(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> weight,
        NativeCudaBuffer<float> inputGradient,
        int rows,
        int inputWidth,
        int outputWidth,
        bool bfloat16)
        => Gemm(
            accelerator, deviceIndex,
            OperationNone, OperationNone,
            inputWidth, rows, outputWidth,
            weight, inputWidth,
            outputGradient, outputWidth,
            inputGradient, inputWidth,
            beta: 1f,
            bfloat16);

    internal static void LinearBackwardWeight(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<float> input,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> weightGradient,
        int rows,
        int inputWidth,
        int outputWidth,
        bool bfloat16)
        => Gemm(
            accelerator, deviceIndex,
            OperationNone, OperationTranspose,
            inputWidth, outputWidth, rows,
            input, inputWidth,
            outputGradient, outputWidth,
            weightGradient, inputWidth,
            beta: 1f,
            bfloat16);

    internal static void MuonGram(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<float> source,
        NativeCudaBuffer<float> destination,
        int rows,
        int columns,
        bool bfloat16TensorCores)
        => Gemm(
            accelerator, deviceIndex,
            OperationTranspose, OperationNone,
            rows, rows, columns,
            source, columns,
            source, columns,
            destination, rows,
            beta: 0f,
            bfloat16: bfloat16TensorCores);

    internal static void MuonPolynomialUpdate(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<float> source,
        NativeCudaBuffer<float> coefficients,
        NativeCudaBuffer<float> destination,
        int rows,
        int columns,
        bool bfloat16TensorCores)
        => Gemm(
            accelerator, deviceIndex,
            OperationNone, OperationNone,
            columns, rows, rows,
            source, columns,
            coefficients, rows,
            destination, columns,
            beta: 0f,
            bfloat16: bfloat16TensorCores);

    internal static void MuonGramBatched(
        NativeCudaDevice accelerator,
        int deviceIndex,
        nint source,
        nint destination,
        int rows,
        int columns,
        int batch,
        bool bfloat16TensorCores)
        => GemmStridedPointers(
            accelerator, deviceIndex,
            OperationTranspose, OperationNone,
            rows, rows, columns,
            source, columns, (long)rows * columns,
            source, columns, (long)rows * columns,
            destination, rows, (long)rows * rows,
            batch, beta: 0f, bfloat16TensorCores);

    internal static void MuonPolynomialUpdateBatched(
        NativeCudaDevice accelerator,
        int deviceIndex,
        nint source,
        nint coefficients,
        nint destination,
        int rows,
        int columns,
        int batch,
        bool bfloat16TensorCores)
        => GemmStridedPointers(
            accelerator, deviceIndex,
            OperationNone, OperationNone,
            columns, rows, rows,
            source, columns, (long)rows * columns,
            coefficients, rows, (long)rows * rows,
            destination, columns, (long)rows * columns,
            batch, beta: 0f, bfloat16TensorCores);

    internal static void MatMulForward(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<float> left,
        NativeCudaBuffer<float> right,
        NativeCudaBuffer<float> output,
        int batch,
        int m,
        int k,
        int n)
        => GemmStrided(
            accelerator, deviceIndex,
            OperationNone, OperationNone,
            n, m, k,
            right, n, (long)k * n,
            left, k, (long)m * k,
            output, n, (long)m * n,
            batch, beta: 0f);

    internal static void MatMulForwardBFloat16(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> left,
        NativeCudaBuffer<ushort> right,
        NativeCudaBuffer<ushort> output,
        int batch,
        int m,
        int k,
        int n)
        => GemmStridedBFloat16(
            accelerator, deviceIndex,
            OperationNone, OperationNone,
            n, m, k,
            right, n, (long)k * n,
            left, k, (long)m * k,
            output, n, (long)m * n,
            batch, beta: 0f);

    /// <summary>
    /// Computes row-major <c>left * right^T</c> for one or more batches
    /// without materializing the transpose. BF16 uses Tensor Core GEMMEx.
    /// </summary>
    internal static void MatMulTransposedRightForward(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<float> left,
        NativeCudaBuffer<float> right,
        NativeCudaBuffer<float> output,
        int batch,
        int m,
        int k,
        int n)
        => GemmStrided(
            accelerator, deviceIndex,
            OperationTranspose, OperationNone,
            n, m, k,
            right, k, (long)n * k,
            left, k, (long)m * k,
            output, n, (long)m * n,
            batch, beta: 0f);

    internal static void MatMulTransposedRightForwardBFloat16(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> left,
        NativeCudaBuffer<ushort> right,
        NativeCudaBuffer<ushort> output,
        int batch,
        int m,
        int k,
        int n)
        => GemmStridedBFloat16(
            accelerator, deviceIndex,
            OperationTranspose, OperationNone,
            n, m, k,
            right, k, (long)n * k,
            left, k, (long)m * k,
            output, n, (long)m * n,
            batch, beta: 0f);

    internal static void MatMulBackward(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<float> left,
        NativeCudaBuffer<float> right,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> leftGradient,
        NativeCudaBuffer<float> rightGradient,
        int batch,
        int m,
        int k,
        int n)
    {
        GemmStrided(
            accelerator, deviceIndex,
            OperationTranspose, OperationNone,
            k, m, n,
            right, n, (long)k * n,
            outputGradient, n, (long)m * n,
            leftGradient, k, (long)m * k,
            batch, beta: 1f);
        GemmStrided(
            accelerator, deviceIndex,
            OperationNone, OperationTranspose,
            n, k, m,
            outputGradient, n, (long)m * n,
            left, k, (long)m * k,
            rightGradient, n, (long)k * n,
            batch, beta: 1f);
    }

    internal static void MatMulBackwardBFloat16(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> left,
        NativeCudaBuffer<ushort> right,
        NativeCudaBuffer<ushort> outputGradient,
        NativeCudaBuffer<float> leftGradient,
        NativeCudaBuffer<float> rightGradient,
        int batch,
        int m,
        int k,
        int n)
    {
        GemmStridedBFloat16ToFloat32(
            accelerator, deviceIndex,
            OperationTranspose, OperationNone,
            k, m, n,
            right, n, (long)k * n,
            outputGradient, n, (long)m * n,
            leftGradient, k, (long)m * k,
            batch, beta: 1f);
        GemmStridedBFloat16ToFloat32(
            accelerator, deviceIndex,
            OperationNone, OperationTranspose,
            n, k, m,
            outputGradient, n, (long)m * n,
            left, k, (long)m * k,
            rightGradient, n, (long)k * n,
            batch, beta: 1f);
    }

    internal static void MatMulBackwardLeftBFloat16Direct(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> right,
        NativeCudaBuffer<ushort> outputGradient,
        NativeCudaBuffer<ushort> leftGradient,
        int batch,
        int m,
        int k,
        int n,
        bool accumulate)
        => GemmStridedBFloat16(
            accelerator, deviceIndex,
            OperationTranspose, OperationNone,
            k, m, n,
            right, n, (long)k * n,
            outputGradient, n, (long)m * n,
            leftGradient, k, (long)m * k,
            batch, accumulate ? 1f : 0f);

    internal static void MatMulBackwardRightBFloat16Direct(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> left,
        NativeCudaBuffer<ushort> outputGradient,
        NativeCudaBuffer<ushort> rightGradient,
        int batch,
        int m,
        int k,
        int n,
        bool accumulate)
        => GemmStridedBFloat16(
            accelerator, deviceIndex,
            OperationNone, OperationTranspose,
            n, k, m,
            outputGradient, n, (long)m * n,
            left, k, (long)m * k,
            rightGradient, n, (long)k * n,
            batch, accumulate ? 1f : 0f);

    internal static void MatMulTransposedRightBackward(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<float> left,
        NativeCudaBuffer<float> right,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> leftGradient,
        NativeCudaBuffer<float> rightGradient,
        int batch,
        int m,
        int k,
        int n)
    {
        // dLeft = dOutput * right
        GemmStrided(
            accelerator, deviceIndex,
            OperationNone, OperationNone,
            k, m, n,
            right, k, (long)n * k,
            outputGradient, n, (long)m * n,
            leftGradient, k, (long)m * k,
            batch, beta: 1f);
        // dRight = dOutput^T * left
        GemmStrided(
            accelerator, deviceIndex,
            OperationNone, OperationTranspose,
            k, n, m,
            left, k, (long)m * k,
            outputGradient, n, (long)m * n,
            rightGradient, k, (long)n * k,
            batch, beta: 1f);
    }

    internal static void MatMulTransposedRightBackwardBFloat16(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> left,
        NativeCudaBuffer<ushort> right,
        NativeCudaBuffer<ushort> outputGradient,
        NativeCudaBuffer<float> leftGradient,
        NativeCudaBuffer<float> rightGradient,
        int batch,
        int m,
        int k,
        int n)
    {
        GemmStridedBFloat16ToFloat32(
            accelerator, deviceIndex,
            OperationNone, OperationNone,
            k, m, n,
            right, k, (long)n * k,
            outputGradient, n, (long)m * n,
            leftGradient, k, (long)m * k,
            batch, beta: 1f);
        GemmStridedBFloat16ToFloat32(
            accelerator, deviceIndex,
            OperationNone, OperationTranspose,
            k, n, m,
            left, k, (long)m * k,
            outputGradient, n, (long)m * n,
            rightGradient, k, (long)n * k,
            batch, beta: 1f);
    }

    internal static void MatMulTransposedRightBackwardInputBFloat16Direct(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> right,
        NativeCudaBuffer<ushort> outputGradient,
        NativeCudaBuffer<ushort> leftGradient,
        int batch,
        int m,
        int k,
        int n)
        => GemmStridedBFloat16(
            accelerator, deviceIndex,
            OperationNone, OperationNone,
            k, m, n,
            right, k, (long)n * k,
            outputGradient, n, (long)m * n,
            leftGradient, k, (long)m * k,
            batch, beta: 0f);

    internal static void
        MatMulTransposedRightBackwardInputBFloat16Accumulate(
            NativeCudaDevice accelerator,
            int deviceIndex,
            NativeCudaBuffer<ushort> right,
            NativeCudaBuffer<ushort> outputGradient,
            NativeCudaBuffer<ushort> leftGradient,
            int batch,
            int m,
            int k,
            int n,
            bool accumulate)
        => GemmStridedBFloat16(
            accelerator, deviceIndex,
            OperationNone, OperationNone,
            k, m, n,
            right, k, (long)n * k,
            outputGradient, n, (long)m * n,
            leftGradient, k, (long)m * k,
            batch, accumulate ? 1f : 0f);

    internal static void MatMulTransposedRightBackwardWeightBFloat16(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> left,
        NativeCudaBuffer<ushort> outputGradient,
        NativeCudaBuffer<float> rightGradient,
        int batch,
        int m,
        int k,
        int n)
        => GemmStridedBFloat16ToFloat32(
            accelerator, deviceIndex,
            OperationNone, OperationTranspose,
            k, n, m,
            left, k, (long)m * k,
            outputGradient, n, (long)m * n,
            rightGradient, k, (long)n * k,
            batch, beta: 1f);

    internal static void
        MatMulTransposedRightBackwardWeightBFloat16Direct(
            NativeCudaDevice accelerator,
            int deviceIndex,
            NativeCudaBuffer<ushort> left,
            NativeCudaBuffer<ushort> outputGradient,
            NativeCudaBuffer<ushort> rightGradient,
            int batch,
            int m,
            int k,
            int n,
            bool accumulate)
        => GemmStridedBFloat16(
            accelerator, deviceIndex,
            OperationNone, OperationTranspose,
            k, n, m,
            left, k, (long)m * k,
            outputGradient, n, (long)m * n,
            rightGradient, k, (long)n * k,
            batch, accumulate ? 1f : 0f);

    internal static void TransposeFloat32(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<float> source,
        NativeCudaBuffer<float> destination,
        int sourceRows,
        int sourceColumns)
    {
        accelerator.Bind();
        using PreparedHandle preparedHandle = PrepareHandle(
            accelerator,
            deviceIndex);
        nint handle = preparedHandle.Handle;
        float alpha = 1f;
        float beta = 0f;
        // Row-major [R,C] is column-major [C,R]. Transposing that view into a
        // column-major [R,C] buffer produces row-major [C,R].
        int status = CublasSgeam(
            handle,
            OperationTranspose,
            OperationTranspose,
            sourceRows,
            sourceColumns,
            (nint)(&alpha),
            source.NativePtr,
            sourceColumns,
            (nint)(&beta),
            source.NativePtr,
            sourceColumns,
            destination.NativePtr,
            sourceRows);
        ThrowIfFailed(status, "cublasSgeam(transpose)");
    }

    private static void Gemm(
        NativeCudaDevice accelerator,
        int deviceIndex,
        int transA,
        int transB,
        int m,
        int n,
        int k,
        NativeCudaBuffer<float> a,
        int lda,
        NativeCudaBuffer<float> b,
        int ldb,
        NativeCudaBuffer<float> c,
        int ldc,
        float beta,
        bool bfloat16)
    {
        accelerator.Bind();
        using PreparedHandle preparedHandle = PrepareHandle(
            accelerator,
            deviceIndex);
        nint handle = preparedHandle.Handle;
        float alpha = 1f;
        int status = CublasGemmEx(
            handle, transA, transB, m, n, k,
            (nint)(&alpha),
            a.NativePtr, CudaR32F, lda,
            b.NativePtr, CudaR32F, ldb,
            (nint)(&beta),
            c.NativePtr, CudaR32F, ldc,
            bfloat16 ? Compute32FFast16BFloat : 68,
            bfloat16 ? GemmDefaultTensorOp : GemmDefault);
        ThrowIfFailed(status, "cublasGemmEx");
    }

    private static void GemmBFloat16(
        NativeCudaDevice accelerator,
        int deviceIndex,
        int transA,
        int transB,
        int m,
        int n,
        int k,
        NativeCudaBuffer<ushort> a,
        int lda,
        NativeCudaBuffer<ushort> b,
        int ldb,
        NativeCudaBuffer<ushort> c,
        int ldc,
        float beta)
    {
        accelerator.Bind();
        using PreparedHandle preparedHandle = PrepareHandle(
            accelerator,
            deviceIndex);
        nint handle = preparedHandle.Handle;
        float alpha = 1f;
        int status = CublasGemmEx(
            handle,
            transA,
            transB,
            m,
            n,
            k,
            (nint)(&alpha),
            a.NativePtr,
            CudaR16BF,
            lda,
            b.NativePtr,
            CudaR16BF,
            ldb,
            (nint)(&beta),
            c.NativePtr,
            CudaR16BF,
            ldc,
            Compute32FFast16BFloat,
            GemmDefaultTensorOp);
        ThrowIfFailed(status, "cublasGemmEx(BF16 direct gradient)");
    }

    private static void GemmBFloat16ToFloat32(
        NativeCudaDevice accelerator,
        int deviceIndex,
        int transA,
        int transB,
        int m,
        int n,
        int k,
        NativeCudaBuffer<ushort> a,
        int lda,
        NativeCudaBuffer<ushort> b,
        int ldb,
        NativeCudaBuffer<float> c,
        int ldc,
        float beta)
    {
        accelerator.Bind();
        using PreparedHandle preparedHandle = PrepareHandle(
            accelerator,
            deviceIndex);
        nint handle = preparedHandle.Handle;
        float alpha = 1f;
        int status = CublasGemmEx(
            handle,
            transA,
            transB,
            m,
            n,
            k,
            (nint)(&alpha),
            a.NativePtr,
            CudaR16BF,
            lda,
            b.NativePtr,
            CudaR16BF,
            ldb,
            (nint)(&beta),
            c.NativePtr,
            CudaR32F,
            ldc,
            Compute32FFast16BFloat,
            GemmDefaultTensorOp);
        ThrowIfFailed(status, "cublasGemmEx(BF16->FP32)");
    }

    private static void GemmStrided(
        NativeCudaDevice accelerator,
        int deviceIndex,
        int transA,
        int transB,
        int m,
        int n,
        int k,
        NativeCudaBuffer<float> a,
        int lda,
        long strideA,
        NativeCudaBuffer<float> b,
        int ldb,
        long strideB,
        NativeCudaBuffer<float> c,
        int ldc,
        long strideC,
        int batch,
        float beta)
    {
        accelerator.Bind();
        using PreparedHandle preparedHandle = PrepareHandle(
            accelerator,
            deviceIndex);
        nint handle = preparedHandle.Handle;
        float alpha = 1f;
        int status = CublasGemmStridedBatchedEx(
            handle, transA, transB, m, n, k,
            (nint)(&alpha),
            a.NativePtr, CudaR32F, lda, strideA,
            b.NativePtr, CudaR32F, ldb, strideB,
            (nint)(&beta),
            c.NativePtr, CudaR32F, ldc, strideC,
            batch, 68, GemmDefault);
        ThrowIfFailed(status, "cublasGemmStridedBatchedEx");
    }

    private static void GemmStridedPointers(
        NativeCudaDevice accelerator,
        int deviceIndex,
        int transA,
        int transB,
        int m,
        int n,
        int k,
        nint a,
        int lda,
        long strideA,
        nint b,
        int ldb,
        long strideB,
        nint c,
        int ldc,
        long strideC,
        int batch,
        float beta,
        bool bfloat16)
    {
        accelerator.Bind();
        using PreparedHandle preparedHandle = PrepareHandle(
            accelerator,
            deviceIndex);
        nint handle = preparedHandle.Handle;
        float alpha = 1f;
        int status = CublasGemmStridedBatchedEx(
            handle, transA, transB, m, n, k,
            (nint)(&alpha),
            a, CudaR32F, lda, strideA,
            b, CudaR32F, ldb, strideB,
            (nint)(&beta),
            c, CudaR32F, ldc, strideC,
            batch,
            bfloat16 ? Compute32FFast16BFloat : 68,
            bfloat16 ? GemmDefaultTensorOp : GemmDefault);
        ThrowIfFailed(status, "cublasGemmStridedBatchedEx(NekoMuon)");
    }

    private static void GemmStridedBFloat16(
        NativeCudaDevice accelerator,
        int deviceIndex,
        int transA,
        int transB,
        int m,
        int n,
        int k,
        NativeCudaBuffer<ushort> a,
        int lda,
        long strideA,
        NativeCudaBuffer<ushort> b,
        int ldb,
        long strideB,
        NativeCudaBuffer<ushort> c,
        int ldc,
        long strideC,
        int batch,
        float beta)
    {
        accelerator.Bind();
        using PreparedHandle preparedHandle = PrepareHandle(
            accelerator,
            deviceIndex);
        nint handle = preparedHandle.Handle;
        float alpha = 1f;
        int status = CublasGemmStridedBatchedEx(
            handle, transA, transB, m, n, k,
            (nint)(&alpha),
            a.NativePtr, CudaR16BF, lda, strideA,
            b.NativePtr, CudaR16BF, ldb, strideB,
            (nint)(&beta),
            c.NativePtr, CudaR16BF, ldc, strideC,
            batch, Compute32FFast16BFloat, GemmDefaultTensorOp);
        ThrowIfFailed(status, "cublasGemmStridedBatchedEx(BF16)");
    }

    private static void GemmStridedBFloat16ToFloat32(
        NativeCudaDevice accelerator,
        int deviceIndex,
        int transA,
        int transB,
        int m,
        int n,
        int k,
        NativeCudaBuffer<ushort> a,
        int lda,
        long strideA,
        NativeCudaBuffer<ushort> b,
        int ldb,
        long strideB,
        NativeCudaBuffer<float> c,
        int ldc,
        long strideC,
        int batch,
        float beta)
    {
        accelerator.Bind();
        using PreparedHandle preparedHandle = PrepareHandle(
            accelerator,
            deviceIndex);
        nint handle = preparedHandle.Handle;
        float alpha = 1f;
        int status = CublasGemmStridedBatchedEx(
            handle, transA, transB, m, n, k,
            (nint)(&alpha),
            a.NativePtr, CudaR16BF, lda, strideA,
            b.NativePtr, CudaR16BF, ldb, strideB,
            (nint)(&beta),
            c.NativePtr, CudaR32F, ldc, strideC,
            batch, Compute32FFast16BFloat, GemmDefaultTensorOp);
        ThrowIfFailed(status, "cublasGemmStridedBatchedEx(BF16->FP32)");
    }

    private static nint CreateHandle(nint computeStream)
    {
        int status = CublasCreate(out nint handle);
        ThrowIfFailed(status, "cublasCreate_v2");
        try
        {
            ThrowIfFailed(
                CublasSetStream(handle, computeStream),
                "cublasSetStream_v2");
            return handle;
        }
        catch
        {
            _ = CublasDestroy(handle);
            throw;
        }
    }

    private static PreparedHandle PrepareHandle(
        NativeCudaDevice accelerator,
        int deviceIndex)
    {
        nint computeStream = accelerator.DefaultStream;
        if (TensorExecutionContext.TryGetCudaStreamLane(
                deviceIndex,
                out IStreamExecutionLane lane)
            && lane.ComputeStreamHandle == computeStream)
        {
            LaneHandle owned = LaneHandles.GetValue(
                lane,
                static value => new Lazy<LaneHandle>(
                    () => ExecutionLaneResources.Attach(
                        value,
                        new LaneHandle(value.ComputeStreamHandle)),
                    LazyThreadSafetyMode.ExecutionAndPublication)).Value;
            return new PreparedHandle(owned.Handle, fallbackLease: null);
        }

        BoundedDisposableLeaseCache<HandleKey, LegacyHandle>.Lease lease =
            FallbackHandles.Acquire(
                new HandleKey(deviceIndex, computeStream),
                static key => new LegacyHandle(
                    key.DeviceIndex,
                    key.ComputeStream))
            ?? throw new InvalidOperationException(
                "A legacy cuBLAS handle could not be created.");
        return new PreparedHandle(lease.Value.Handle, lease);
    }

    private readonly record struct HandleKey(
        int DeviceIndex,
        nint ComputeStream);

    private sealed class PreparedHandle(
        nint handle,
        BoundedDisposableLeaseCache<HandleKey, LegacyHandle>.Lease?
            fallbackLease) : IDisposable
    {
        private BoundedDisposableLeaseCache<HandleKey, LegacyHandle>.Lease?
            _fallbackLease = fallbackLease;

        internal nint Handle { get; } = handle;

        public void Dispose()
            => Interlocked.Exchange(ref _fallbackLease, null)?.Dispose();
    }

    private sealed class LaneHandle : IDisposable
    {
        private nint _handle;

        internal LaneHandle(nint computeStream)
        {
            _handle = CreateHandle(computeStream);
            Interlocked.Increment(ref _activeLaneHandleCount);
        }

        internal nint Handle
        {
            get
            {
                nint handle = Volatile.Read(ref _handle);
                ObjectDisposedException.ThrowIf(
                    handle == nint.Zero,
                    this);
                return handle;
            }
        }

        public void Dispose()
        {
            nint handle = Interlocked.Exchange(ref _handle, nint.Zero);
            if (handle == nint.Zero)
                return;
            try
            {
                ThrowIfFailed(CublasDestroy(handle), "cublasDestroy_v2");
            }
            finally
            {
                Interlocked.Decrement(ref _activeLaneHandleCount);
            }
        }
    }

    private sealed class LegacyHandle : IDisposable
    {
        private readonly int _deviceIndex;
        private readonly nint _computeStream;
        private nint _handle;

        internal LegacyHandle(int deviceIndex, nint computeStream)
        {
            _deviceIndex = deviceIndex;
            _computeStream = computeStream;
            _handle = CreateHandle(computeStream);
        }

        internal nint Handle
        {
            get
            {
                nint handle = Volatile.Read(ref _handle);
                ObjectDisposedException.ThrowIf(
                    handle == nint.Zero,
                    this);
                return handle;
            }
        }

        public void Dispose()
        {
            nint handle = Interlocked.Exchange(ref _handle, nint.Zero);
            if (handle == nint.Zero)
                return;
            NativeCudaRuntime.DisposeAfterStreamFence(
                _deviceIndex,
                _computeStream,
                () => ThrowIfFailed(
                    CublasDestroy(handle),
                    "cublasDestroy_v2(legacy fallback)"));
        }
    }

    private static void ThrowIfFailed(int status, string operation)
    {
        if (status != Success)
        {
            throw new InvalidOperationException(
                $"{operation} failed with cuBLAS status {status}.");
        }
    }

    private static int CublasCreate(out nint handle)
        => CudaNativeGateway.CublasCreate(out handle);

    private static int CublasDestroy(nint handle)
        => CudaNativeGateway.CublasDestroy(handle);

    private static int CublasSetStream(nint handle, nint stream)
        => CudaNativeGateway.CublasSetStream(handle, stream);

    private static int CublasGemmEx(
        nint handle,
        int transA,
        int transB,
        int m,
        int n,
        int k,
        nint alpha,
        nint a,
        int aType,
        int lda,
        nint b,
        int bType,
        int ldb,
        nint beta,
        nint c,
        int cType,
        int ldc,
        int computeType,
        int algorithm)
        => CudaNativeGateway.CublasGemmEx(
            handle, transA, transB, m, n, k, alpha, a, aType, lda, b,
            bType, ldb, beta, c, cType, ldc, computeType, algorithm);

    private static int CublasGemmStridedBatchedEx(
        nint handle,
        int transA,
        int transB,
        int m,
        int n,
        int k,
        nint alpha,
        nint a,
        int aType,
        int lda,
        long strideA,
        nint b,
        int bType,
        int ldb,
        long strideB,
        nint beta,
        nint c,
        int cType,
        int ldc,
        long strideC,
        int batchCount,
        int computeType,
        int algorithm)
        => CudaNativeGateway.CublasGemmStridedBatchedEx(
            handle, transA, transB, m, n, k, alpha, a, aType, lda, strideA,
            b, bType, ldb, strideB, beta, c, cType, ldc, strideC,
            batchCount, computeType, algorithm);

    private static int CublasSgeam(
        nint handle,
        int transA,
        int transB,
        int m,
        int n,
        nint alpha,
        nint a,
        int lda,
        nint beta,
        nint b,
        int ldb,
        nint c,
        int ldc)
        => CudaNativeGateway.CublasSgeam(
            handle, transA, transB, m, n, alpha, a, lda, beta, b, ldb, c,
            ldc);
}
