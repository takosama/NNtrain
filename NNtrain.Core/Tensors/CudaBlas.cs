using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace NNtrain;

/// <summary>Thin cuBLAS binding over buffers allocated by ILGPU.</summary>
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

    private static readonly object Sync = new();
    private static readonly Dictionary<int, nint> Handles = [];

    internal static void LinearForward(
        CudaAccelerator accelerator,
        int deviceIndex,
        MemoryBuffer1D<float, Stride1D.Dense> input,
        MemoryBuffer1D<float, Stride1D.Dense> weight,
        MemoryBuffer1D<float, Stride1D.Dense> output,
        int rows,
        int inputWidth,
        int outputWidth,
        bool bfloat16)
    {
        accelerator.Bind();
        nint handle = GetHandle(deviceIndex);
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
            GemmDefault);
        ThrowIfFailed(status, "cublasGemmEx");
    }

    internal static void LinearForwardBFloat16(
        CudaAccelerator accelerator,
        int deviceIndex,
        MemoryBuffer1D<ushort, Stride1D.Dense> input,
        MemoryBuffer1D<ushort, Stride1D.Dense> weight,
        MemoryBuffer1D<ushort, Stride1D.Dense> output,
        int rows,
        int inputWidth,
        int outputWidth)
    {
        accelerator.Bind();
        nint handle = GetHandle(deviceIndex);
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
            GemmDefault);
        ThrowIfFailed(status, "cublasGemmEx(BF16)");
    }

    internal static void LinearBackwardInputBFloat16(
        CudaAccelerator accelerator,
        int deviceIndex,
        MemoryBuffer1D<ushort, Stride1D.Dense> outputGradient,
        MemoryBuffer1D<ushort, Stride1D.Dense> weight,
        MemoryBuffer1D<float, Stride1D.Dense> inputGradient,
        int rows,
        int inputWidth,
        int outputWidth)
        => GemmBFloat16ToFloat32(
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

    internal static void LinearBackwardWeightBFloat16(
        CudaAccelerator accelerator,
        int deviceIndex,
        MemoryBuffer1D<ushort, Stride1D.Dense> input,
        MemoryBuffer1D<ushort, Stride1D.Dense> outputGradient,
        MemoryBuffer1D<float, Stride1D.Dense> weightGradient,
        int rows,
        int inputWidth,
        int outputWidth)
        => GemmBFloat16ToFloat32(
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

    internal static void LinearBackwardInput(
        CudaAccelerator accelerator,
        int deviceIndex,
        MemoryBuffer1D<float, Stride1D.Dense> outputGradient,
        MemoryBuffer1D<float, Stride1D.Dense> weight,
        MemoryBuffer1D<float, Stride1D.Dense> inputGradient,
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
        CudaAccelerator accelerator,
        int deviceIndex,
        MemoryBuffer1D<float, Stride1D.Dense> input,
        MemoryBuffer1D<float, Stride1D.Dense> outputGradient,
        MemoryBuffer1D<float, Stride1D.Dense> weightGradient,
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
        CudaAccelerator accelerator,
        int deviceIndex,
        MemoryBuffer1D<float, Stride1D.Dense> source,
        MemoryBuffer1D<float, Stride1D.Dense> destination,
        int rows,
        int columns)
        => Gemm(
            accelerator, deviceIndex,
            OperationTranspose, OperationNone,
            rows, rows, columns,
            source, columns,
            source, columns,
            destination, rows,
            beta: 0f,
            bfloat16: false);

    internal static void MuonPolynomialUpdate(
        CudaAccelerator accelerator,
        int deviceIndex,
        MemoryBuffer1D<float, Stride1D.Dense> source,
        MemoryBuffer1D<float, Stride1D.Dense> coefficients,
        MemoryBuffer1D<float, Stride1D.Dense> destination,
        int rows,
        int columns)
        => Gemm(
            accelerator, deviceIndex,
            OperationNone, OperationNone,
            columns, rows, rows,
            source, columns,
            coefficients, rows,
            destination, columns,
            beta: 0f,
            bfloat16: false);

    private static void Gemm(
        CudaAccelerator accelerator,
        int deviceIndex,
        int transA,
        int transB,
        int m,
        int n,
        int k,
        MemoryBuffer1D<float, Stride1D.Dense> a,
        int lda,
        MemoryBuffer1D<float, Stride1D.Dense> b,
        int ldb,
        MemoryBuffer1D<float, Stride1D.Dense> c,
        int ldc,
        float beta,
        bool bfloat16)
    {
        accelerator.Bind();
        nint handle = GetHandle(deviceIndex);
        float alpha = 1f;
        int status = CublasGemmEx(
            handle, transA, transB, m, n, k,
            (nint)(&alpha),
            a.NativePtr, CudaR32F, lda,
            b.NativePtr, CudaR32F, ldb,
            (nint)(&beta),
            c.NativePtr, CudaR32F, ldc,
            bfloat16 ? Compute32FFast16BFloat : 68,
            GemmDefault);
        ThrowIfFailed(status, "cublasGemmEx");
    }

    private static void GemmBFloat16ToFloat32(
        CudaAccelerator accelerator,
        int deviceIndex,
        int transA,
        int transB,
        int m,
        int n,
        int k,
        MemoryBuffer1D<ushort, Stride1D.Dense> a,
        int lda,
        MemoryBuffer1D<ushort, Stride1D.Dense> b,
        int ldb,
        MemoryBuffer1D<float, Stride1D.Dense> c,
        int ldc,
        float beta)
    {
        accelerator.Bind();
        nint handle = GetHandle(deviceIndex);
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
            GemmDefault);
        ThrowIfFailed(status, "cublasGemmEx(BF16->FP32)");
    }

    private static nint GetHandle(int deviceIndex)
    {
        lock (Sync)
        {
            if (Handles.TryGetValue(deviceIndex, out nint handle))
                return handle;

            int status = CublasCreate(out handle);
            ThrowIfFailed(status, "cublasCreate_v2");
            Handles.Add(deviceIndex, handle);
            return handle;
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

    [DllImport(Library, EntryPoint = "cublasCreate_v2",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int CublasCreate(out nint handle);

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
}
