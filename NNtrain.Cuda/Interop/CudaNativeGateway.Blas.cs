using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// Central interop boundary for the CUDA 12 cuBLAS and cuBLASLt vendor ABIs.
/// Vendor status values are returned unchanged: they are cuBLAS statuses, not
/// CUDA-runtime statuses, and therefore must not enter the NNtrain native
/// error-ring protocol.
/// </summary>
public static partial class CudaNativeGateway
{
    public const string CublasLibraryName = "cublas64_12";
    public const string CublasLtLibraryName = "cublasLt64_12";

    public static int CublasCreate(out nint handle)
    {
        EnsureCompatibleAbi();
        return BlasNativeMethods.Create(out handle);
    }

    public static int CublasDestroy(nint handle)
        => BlasNativeMethods.Destroy(handle);

    public static int CublasSetStream(nint handle, nint stream)
        => BlasNativeMethods.SetStream(handle, stream);

    public static int CublasGemmEx(
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
        => BlasNativeMethods.GemmEx(
            handle,
            transA,
            transB,
            m,
            n,
            k,
            alpha,
            a,
            aType,
            lda,
            b,
            bType,
            ldb,
            beta,
            c,
            cType,
            ldc,
            computeType,
            algorithm);

    public static int CublasGemmStridedBatchedEx(
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
        => BlasNativeMethods.GemmStridedBatchedEx(
            handle,
            transA,
            transB,
            m,
            n,
            k,
            alpha,
            a,
            aType,
            lda,
            strideA,
            b,
            bType,
            ldb,
            strideB,
            beta,
            c,
            cType,
            ldc,
            strideC,
            batchCount,
            computeType,
            algorithm);

    public static int CublasSgeam(
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
        => BlasNativeMethods.Sgeam(
            handle,
            transA,
            transB,
            m,
            n,
            alpha,
            a,
            lda,
            beta,
            b,
            ldb,
            c,
            ldc);

    public static int CublasLtCreate(out nint handle)
    {
        EnsureCompatibleAbi();
        return BlasLtNativeMethods.Create(out handle);
    }

    public static int CublasLtDestroy(nint handle)
        => BlasLtNativeMethods.Destroy(handle);

    public static int CublasLtMatmulDescCreate(
        out nint descriptor,
        int computeType,
        int scaleType)
        => BlasLtNativeMethods.MatmulDescCreate(
            out descriptor,
            computeType,
            scaleType);

    public static int CublasLtMatmulDescDestroy(nint descriptor)
        => BlasLtNativeMethods.MatmulDescDestroy(descriptor);

    public static int CublasLtMatmulDescSetAttribute(
        nint descriptor,
        int attribute,
        nint value,
        nuint size)
        => BlasLtNativeMethods.MatmulDescSetAttribute(
            descriptor,
            attribute,
            value,
            size);

    public static int CublasLtMatrixLayoutCreate(
        out nint layout,
        int type,
        nuint rows,
        nuint columns,
        long leadingDimension)
        => BlasLtNativeMethods.MatrixLayoutCreate(
            out layout,
            type,
            rows,
            columns,
            leadingDimension);

    public static int CublasLtMatrixLayoutDestroy(nint layout)
        => BlasLtNativeMethods.MatrixLayoutDestroy(layout);

    public static int CublasLtPreferenceCreate(out nint preference)
        => BlasLtNativeMethods.PreferenceCreate(out preference);

    public static int CublasLtPreferenceDestroy(nint preference)
        => BlasLtNativeMethods.PreferenceDestroy(preference);

    public static int CublasLtPreferenceSetAttribute(
        nint preference,
        int attribute,
        nint value,
        nuint size)
        => BlasLtNativeMethods.PreferenceSetAttribute(
            preference,
            attribute,
            value,
            size);

    public static int CublasLtMatmulAlgoGetHeuristic(
        nint handle,
        nint operation,
        nint first,
        nint second,
        nint outputC,
        nint outputD,
        nint preference,
        int requestedCount,
        nint results,
        nint returnedCount)
        => BlasLtNativeMethods.MatmulAlgoGetHeuristic(
            handle,
            operation,
            first,
            second,
            outputC,
            outputD,
            preference,
            requestedCount,
            results,
            returnedCount);

    public static int CublasLtMatmulAlgoCapGetAttribute(
        nint algorithm,
        int attribute,
        nint buffer,
        nuint size,
        nint sizeWritten)
        => BlasLtNativeMethods.MatmulAlgoCapGetAttribute(
            algorithm,
            attribute,
            buffer,
            size,
            sizeWritten);

    public static int CublasLtMatmul(
        nint handle,
        nint operation,
        nint alpha,
        nint first,
        nint firstLayout,
        nint second,
        nint secondLayout,
        nint beta,
        nint outputC,
        nint outputCLayout,
        nint outputD,
        nint outputDLayout,
        nint algorithm,
        nint workspace,
        nuint workspaceSize,
        nint stream)
        => BlasLtNativeMethods.Matmul(
            handle,
            operation,
            alpha,
            first,
            firstLayout,
            second,
            secondLayout,
            beta,
            outputC,
            outputCLayout,
            outputD,
            outputDLayout,
            algorithm,
            workspace,
            workspaceSize,
            stream);

    private static class BlasNativeMethods
    {
        [DllImport(CublasLibraryName, EntryPoint = "cublasCreate_v2", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Create(out nint handle);
        [DllImport(CublasLibraryName, EntryPoint = "cublasDestroy_v2", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Destroy(nint handle);
        [DllImport(CublasLibraryName, EntryPoint = "cublasSetStream_v2", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int SetStream(nint handle, nint stream);
        [DllImport(CublasLibraryName, EntryPoint = "cublasGemmEx", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int GemmEx(nint handle, int transA, int transB, int m, int n, int k, nint alpha, nint a, int aType, int lda, nint b, int bType, int ldb, nint beta, nint c, int cType, int ldc, int computeType, int algorithm);
        [DllImport(CublasLibraryName, EntryPoint = "cublasGemmStridedBatchedEx", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int GemmStridedBatchedEx(nint handle, int transA, int transB, int m, int n, int k, nint alpha, nint a, int aType, int lda, long strideA, nint b, int bType, int ldb, long strideB, nint beta, nint c, int cType, int ldc, long strideC, int batchCount, int computeType, int algorithm);
        [DllImport(CublasLibraryName, EntryPoint = "cublasSgeam", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Sgeam(nint handle, int transA, int transB, int m, int n, nint alpha, nint a, int lda, nint beta, nint b, int ldb, nint c, int ldc);
    }

    private static class BlasLtNativeMethods
    {
        [DllImport(CublasLtLibraryName, EntryPoint = "cublasLtCreate", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Create(out nint handle);
        [DllImport(CublasLtLibraryName, EntryPoint = "cublasLtDestroy", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Destroy(nint handle);
        [DllImport(CublasLtLibraryName, EntryPoint = "cublasLtMatmulDescCreate", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int MatmulDescCreate(out nint descriptor, int computeType, int scaleType);
        [DllImport(CublasLtLibraryName, EntryPoint = "cublasLtMatmulDescDestroy", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int MatmulDescDestroy(nint descriptor);
        [DllImport(CublasLtLibraryName, EntryPoint = "cublasLtMatmulDescSetAttribute", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int MatmulDescSetAttribute(nint descriptor, int attribute, nint value, nuint size);
        [DllImport(CublasLtLibraryName, EntryPoint = "cublasLtMatrixLayoutCreate", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int MatrixLayoutCreate(out nint layout, int type, nuint rows, nuint columns, long leadingDimension);
        [DllImport(CublasLtLibraryName, EntryPoint = "cublasLtMatrixLayoutDestroy", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int MatrixLayoutDestroy(nint layout);
        [DllImport(CublasLtLibraryName, EntryPoint = "cublasLtMatmulPreferenceCreate", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int PreferenceCreate(out nint preference);
        [DllImport(CublasLtLibraryName, EntryPoint = "cublasLtMatmulPreferenceDestroy", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int PreferenceDestroy(nint preference);
        [DllImport(CublasLtLibraryName, EntryPoint = "cublasLtMatmulPreferenceSetAttribute", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int PreferenceSetAttribute(nint preference, int attribute, nint value, nuint size);
        [DllImport(CublasLtLibraryName, EntryPoint = "cublasLtMatmulAlgoGetHeuristic", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int MatmulAlgoGetHeuristic(nint handle, nint operation, nint first, nint second, nint outputC, nint outputD, nint preference, int requestedCount, nint results, nint returnedCount);
        [DllImport(CublasLtLibraryName, EntryPoint = "cublasLtMatmulAlgoCapGetAttribute", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int MatmulAlgoCapGetAttribute(nint algorithm, int attribute, nint buffer, nuint size, nint sizeWritten);
        [DllImport(CublasLtLibraryName, EntryPoint = "cublasLtMatmul", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Matmul(nint handle, nint operation, nint alpha, nint first, nint firstLayout, nint second, nint secondLayout, nint beta, nint outputC, nint outputCLayout, nint outputD, nint outputDLayout, nint algorithm, nint workspace, nuint workspaceSize, nint stream);
    }
}
