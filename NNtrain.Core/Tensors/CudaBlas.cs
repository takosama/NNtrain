using System.Runtime.InteropServices;

namespace NNtrain;

/// <summary>Thin cuBLAS binding over native CUDA Runtime buffers.</summary>
internal static unsafe class CudaBlas
{
    private const string Library = "cublas64_12";
    private const int Success = 0;
    private const int OperationNone = 0;
    private const int OperationTranspose = 1;
    private const int CudaR32F = 0;
    private const int CudaR16BF = 14;
    private const int Compute32FFast16BFloat = 75;
    private const int GemmDefault = -1;
    private const int GemmDefaultTensorOp = 99;

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<
        int,
        Lazy<nint>> Handles = new();

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
        nint handle = PrepareHandle(accelerator, deviceIndex);
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
        nint handle = PrepareHandle(accelerator, deviceIndex);
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
        if (CudaBlasLt.TryLinearBackwardInputBFloat16(
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
        nint handle = PrepareHandle(accelerator, deviceIndex);
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
        if (CudaBlasLt.TryLinearBackwardWeightBFloat16(
                accelerator,
                deviceIndex,
                input,
                outputGradient,
                weightGradient,
                rows,
                inputWidth,
                outputWidth))
        {
            return;
        }
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
    }

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
        nint handle = PrepareHandle(accelerator, deviceIndex);
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
        nint handle = PrepareHandle(accelerator, deviceIndex);
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
        nint handle = PrepareHandle(accelerator, deviceIndex);
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
        nint handle = PrepareHandle(accelerator, deviceIndex);
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
        nint handle = PrepareHandle(accelerator, deviceIndex);
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
        nint handle = PrepareHandle(accelerator, deviceIndex);
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

    private static nint GetHandle(int deviceIndex)
        => Handles.GetOrAdd(
            deviceIndex,
            static _ => new Lazy<nint>(
                CreateHandle,
                LazyThreadSafetyMode.ExecutionAndPublication)).Value;

    private static nint CreateHandle()
    {
        int status = CublasCreate(out nint handle);
        ThrowIfFailed(status, "cublasCreate_v2");
        return handle;
    }

    private static nint PrepareHandle(
        NativeCudaDevice accelerator,
        int deviceIndex)
    {
        nint handle = GetHandle(deviceIndex);
        return handle;
    }

    private static void ThrowIfFailed(int status, string operation)
    {
        if (status != Success)
        {
            throw new InvalidOperationException(
                $"{operation} failed with cuBLAS status {status}.");
        }
    }

    [DllImport(Library, EntryPoint = "cublasCreate_v2",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int CublasCreate(out nint handle);

    [DllImport(Library, EntryPoint = "cublasSetStream_v2",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int CublasSetStream(nint handle, nint stream);

    [DllImport(Library, EntryPoint = "cublasGemmEx",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int CublasGemmEx(
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
        int algorithm);

    [DllImport(Library, EntryPoint = "cublasGemmStridedBatchedEx",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int CublasGemmStridedBatchedEx(
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
        int algorithm);
}
