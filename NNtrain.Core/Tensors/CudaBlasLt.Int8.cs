using System.Collections.Concurrent;
using System.Runtime.InteropServices;

namespace NNtrain;

/// <summary>
/// Bounded cuBLASLt signed-Int8 plans used only by tensor-wide BFP8 operands.
/// Int32 output preserves the exact dot product; BFP scale application and
/// output quantization happen in the following resident CUDA kernel.
/// </summary>
internal static unsafe class CudaBlasLtInt8
{
    private const string Library = "cublasLt64_12";
    private const int Success = 0;
    private const int OperationTranspose = 1;
    private const int CudaR8I = 3;
    private const int CudaR32I = 10;
    private const int Compute32I = 72;
    private const int DescTransA = 3;
    private const int PreferenceMaxWorkspaceBytes = 1;
    private const int AlgorithmNumericalImplementationFlags = 15;
    private const ulong NumericalImplementationImma = 0x04UL;
    private const int HeuristicCandidateCount = 16;
    private const int WorkspaceLimitBytes = 4 * 1024 * 1024;
    private const int PlanCacheCapacity = 24;

    private static readonly ConcurrentDictionary<int, Lazy<nint>> Handles = [];
    private static readonly BoundedDisposableLeaseCache<PlanKey, Int8Plan>
        Plans = new(PlanCacheCapacity);
    private static long _executionCount;
    private static long _lastNumericalImplementationFlags;

    internal static long ExecutionCount => Interlocked.Read(ref _executionCount);
    internal static ulong LastNumericalImplementationFlags => unchecked(
        (ulong)Interlocked.Read(ref _lastNumericalImplementationFlags));
    internal static bool LastExecutionUsedInt8TensorCores =>
        (LastNumericalImplementationFlags & NumericalImplementationImma) != 0;

    internal static bool TryMatMul(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<sbyte> left,
        NativeCudaBuffer<sbyte> right,
        NativeCudaBuffer<int> output,
        int m,
        int k,
        int n)
    {
        ValidateBuffers(deviceIndex, left, right, output, m, k, n);
        return Execute(
            accelerator,
            new PlanKey(deviceIndex, Int8Operation.MatMul, m, n, k),
            right.NativePtr,
            left.NativePtr,
            output.NativePtr);
    }

    internal static bool TryLinear(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<sbyte> input,
        NativeCudaBuffer<sbyte> weight,
        NativeCudaBuffer<int> output,
        int rows,
        int inputWidth,
        int outputWidth)
    {
        ValidateBuffers(
            deviceIndex,
            input,
            weight,
            output,
            rows,
            inputWidth,
            outputWidth);
        return Execute(
            accelerator,
            new PlanKey(
                deviceIndex,
                Int8Operation.Linear,
                rows,
                outputWidth,
                inputWidth),
            weight.NativePtr,
            input.NativePtr,
            output.NativePtr);
    }

    private static void ValidateBuffers(
        int deviceIndex,
        NativeCudaBuffer<sbyte> left,
        NativeCudaBuffer<sbyte> right,
        NativeCudaBuffer<int> output,
        int m,
        int k,
        int n)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(m);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(n);
        if (left.Device.Index != deviceIndex
            || right.Device.Index != deviceIndex
            || output.Device.Index != deviceIndex)
        {
            throw new ArgumentException(
                "cuBLASLt Int8 buffers must reside on the requested CUDA device.");
        }
        if (left.Length != checked(m * k)
            || right.Length != checked(k * n)
            || output.Length != checked(m * n))
        {
            throw new ArgumentException("cuBLASLt Int8 GEMM buffer length mismatch.");
        }
        // 127*127*K must fit the exact Int32 accumulator.
        if (k > int.MaxValue / (127 * 127))
        {
            throw new NotSupportedException(
                "BFP8 Int8 GEMM K exceeds the exact Int32 accumulation range.");
        }
    }

    private static bool Execute(
        NativeCudaDevice accelerator,
        PlanKey key,
        nint first,
        nint second,
        nint output)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        try
        {
            accelerator.Bind();
            using BoundedDisposableLeaseCache<PlanKey, Int8Plan>.Lease? lease =
                Plans.Acquire(key, static value => CreatePlan(value));
            Int8Plan? plan = lease?.Value;
            if (plan is null || !plan.IsSupported)
                return false;

            lock (plan.ExecutionSync)
            {
                int alpha = 1;
                int beta = 0;
                MatmulAlgorithm algorithm = plan.Algorithm;
                int status = Matmul(
                    plan.Handle,
                    plan.Operation,
                    (nint)(&alpha),
                    first,
                    plan.First,
                    second,
                    plan.Second,
                    (nint)(&beta),
                    output,
                    plan.Output,
                    output,
                    plan.Output,
                    (nint)(&algorithm),
                    plan.Workspace?.NativePtr ?? 0,
                    plan.WorkspaceSize,
                    accelerator.DefaultStream);
                if (status != Success)
                {
                    throw new InvalidOperationException(
                        $"cublasLtMatmul(Int8) failed with cuBLAS status {status}.");
                }
            }
            Interlocked.Increment(ref _executionCount);
            Interlocked.Exchange(
                ref _lastNumericalImplementationFlags,
                unchecked((long)plan.NumericalImplementationFlags));
            return true;
        }
        catch (Exception exception) when (exception is DllNotFoundException
            or EntryPointNotFoundException
            or BadImageFormatException)
        {
            throw new NotSupportedException(
                "The CUDA runtime has no compatible cuBLASLt Int8 backend. " +
                "CPU fallback is forbidden.",
                exception);
        }
    }

    private static Int8Plan CreatePlan(PlanKey key)
    {
        nint handle = GetHandle(key.DeviceIndex);
        nint operation = 0;
        nint first = 0;
        nint second = 0;
        nint output = 0;
        nint preference = 0;
        NativeCudaBuffer<byte>? workspace = null;
        bool ownershipTransferred = false;
        try
        {
            if (MatmulDescCreate(
                    out operation,
                    Compute32I,
                    CudaR32I) != Success)
            {
                return Int8Plan.CreateUnsupported();
            }

            // Regular-order IMMA is supported only as TN. The first buffer is
            // a column-major [K,N] view (cached row-to-column transform for a
            // MatMul right operand, direct weight view for Linear); the second
            // row-major [M,K] buffer is already column-major [K,M]. The output
            // column-major [N,M] view is the requested row-major [M,N].
            int transposeA = OperationTranspose;
            int firstRows = key.K;
            int firstColumns = key.N;
            int firstLeadingDimension = key.K;

            if (MatmulDescSetAttribute(
                    operation,
                    DescTransA,
                    (nint)(&transposeA),
                    sizeof(int)) != Success
                || MatrixLayoutCreate(
                    out first,
                    CudaR8I,
                    (nuint)firstRows,
                    (nuint)firstColumns,
                    firstLeadingDimension) != Success
                || MatrixLayoutCreate(
                    out second,
                    CudaR8I,
                    (nuint)key.K,
                    (nuint)key.M,
                    key.K) != Success
                || MatrixLayoutCreate(
                    out output,
                    CudaR32I,
                    (nuint)key.N,
                    (nuint)key.M,
                    key.N) != Success
                || PreferenceCreate(out preference) != Success)
            {
                return Int8Plan.CreateUnsupported();
            }

            nuint maximumWorkspaceBytes = WorkspaceLimitBytes;
            if (PreferenceSetAttribute(
                    preference,
                    PreferenceMaxWorkspaceBytes,
                    (nint)(&maximumWorkspaceBytes),
                    (nuint)sizeof(nuint)) != Success)
            {
                return Int8Plan.CreateUnsupported();
            }

            HeuristicResult* heuristics = stackalloc
                HeuristicResult[HeuristicCandidateCount];
            int count = 0;
            int status = MatmulAlgoGetHeuristic(
                handle,
                operation,
                first,
                second,
                output,
                output,
                preference,
                HeuristicCandidateCount,
                (nint)heuristics,
                (nint)(&count));
            if (status != Success || count == 0)
                return Int8Plan.CreateUnsupported();

            int selected = -1;
            float selectedWaves = float.PositiveInfinity;
            nuint selectedWorkspace = nuint.MaxValue;
            ulong selectedImplementationFlags = 0;
            for (int index = 0; index < count; index++)
            {
                ref HeuristicResult candidate = ref heuristics[index];
                if (candidate.State != Success
                    || candidate.WorkspaceSize > maximumWorkspaceBytes)
                {
                    continue;
                }
                ulong implementationFlags = 0;
                nuint bytesWritten = 0;
                int capabilityStatus;
                fixed (MatmulAlgorithm* candidateAlgorithm =
                    &candidate.Algorithm)
                {
                    capabilityStatus = MatmulAlgoCapGetAttribute(
                        (nint)candidateAlgorithm,
                        AlgorithmNumericalImplementationFlags,
                        (nint)(&implementationFlags),
                        (nuint)sizeof(ulong),
                        (nint)(&bytesWritten));
                }
                if (capabilityStatus != Success
                    || bytesWritten != (nuint)sizeof(ulong)
                    || (implementationFlags & NumericalImplementationImma) == 0)
                {
                    continue;
                }
                float waves = float.IsFinite(candidate.WavesCount)
                    && candidate.WavesCount > 0f
                        ? candidate.WavesCount
                        : float.MaxValue;
                if (selected < 0
                    || waves < selectedWaves
                    || (waves == selectedWaves
                        && candidate.WorkspaceSize < selectedWorkspace))
                {
                    selected = index;
                    selectedWaves = waves;
                    selectedWorkspace = candidate.WorkspaceSize;
                    selectedImplementationFlags = implementationFlags;
                }
            }
            if (selected < 0)
                return Int8Plan.CreateUnsupported();

            if (selectedWorkspace > 0)
            {
                workspace = ForgetMemoryV2Cuda.GetAccelerator(key.DeviceIndex)
                    .Allocate1D<byte>(checked((int)selectedWorkspace));
            }
            var plan = new Int8Plan(
                handle,
                operation,
                first,
                second,
                output,
                workspace,
                heuristics[selected].Algorithm,
                selectedWorkspace,
                selectedImplementationFlags);
            ownershipTransferred = true;
            workspace = null;
            return plan;
        }
        finally
        {
            if (preference != 0)
                _ = PreferenceDestroy(preference);
            if (!ownershipTransferred)
            {
                workspace?.Dispose();
                DestroyDescriptors(operation, first, second, output);
            }
        }
    }

    private static nint GetHandle(int deviceIndex)
        => Handles.GetOrAdd(
            deviceIndex,
            static _ => new Lazy<nint>(
                CreateHandle,
                LazyThreadSafetyMode.ExecutionAndPublication)).Value;

    private static nint CreateHandle()
    {
        if (Create(out nint handle) != Success)
            throw new InvalidOperationException("cublasLtCreate failed.");
        return handle;
    }

    private static void DestroyDescriptors(
        nint operation,
        nint first,
        nint second,
        nint output)
    {
        if (output != 0)
            _ = MatrixLayoutDestroy(output);
        if (second != 0)
            _ = MatrixLayoutDestroy(second);
        if (first != 0)
            _ = MatrixLayoutDestroy(first);
        if (operation != 0)
            _ = MatmulDescDestroy(operation);
    }

    private enum Int8Operation
    {
        MatMul,
        Linear,
    }

    private readonly record struct PlanKey(
        int DeviceIndex,
        Int8Operation Operation,
        int M,
        int N,
        int K);

    private sealed class Int8Plan(
        nint handle,
        nint operation,
        nint first,
        nint second,
        nint output,
        NativeCudaBuffer<byte>? workspace,
        MatmulAlgorithm algorithm,
        nuint workspaceSize,
        ulong numericalImplementationFlags,
        bool isSupported = true) : IDisposable
    {
        private int _disposed;

        internal nint Handle { get; } = handle;
        internal nint Operation { get; } = operation;
        internal nint First { get; } = first;
        internal nint Second { get; } = second;
        internal nint Output { get; } = output;
        internal NativeCudaBuffer<byte>? Workspace { get; } = workspace;
        internal MatmulAlgorithm Algorithm = algorithm;
        internal nuint WorkspaceSize { get; } = workspaceSize;
        internal ulong NumericalImplementationFlags { get; } =
            numericalImplementationFlags;
        internal bool IsSupported { get; } = isSupported;
        internal object ExecutionSync { get; } = new();

        internal static Int8Plan CreateUnsupported() => new(
            0,
            0,
            0,
            0,
            0,
            null,
            default,
            0,
            0,
            isSupported: false);

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            try
            {
                DestroyDescriptors(Operation, First, Second, Output);
            }
            finally
            {
                Workspace?.Dispose();
            }
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private unsafe struct MatmulAlgorithm
    {
        internal fixed ulong Data[8];
    }

    [StructLayout(LayoutKind.Sequential)]
    private unsafe struct HeuristicResult
    {
        internal MatmulAlgorithm Algorithm;
        internal nuint WorkspaceSize;
        internal int State;
        internal float WavesCount;
        internal fixed int Reserved[4];
    }

    [DllImport(Library, EntryPoint = "cublasLtCreate",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int Create(out nint handle);

    [DllImport(Library, EntryPoint = "cublasLtMatmulDescCreate",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int MatmulDescCreate(
        out nint descriptor,
        int computeType,
        int scaleType);

    [DllImport(Library, EntryPoint = "cublasLtMatmulDescDestroy",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int MatmulDescDestroy(nint descriptor);

    [DllImport(Library, EntryPoint = "cublasLtMatmulDescSetAttribute",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int MatmulDescSetAttribute(
        nint descriptor,
        int attribute,
        nint value,
        nuint size);

    [DllImport(Library, EntryPoint = "cublasLtMatrixLayoutCreate",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int MatrixLayoutCreate(
        out nint layout,
        int type,
        nuint rows,
        nuint columns,
        long leadingDimension);

    [DllImport(Library, EntryPoint = "cublasLtMatrixLayoutDestroy",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int MatrixLayoutDestroy(nint layout);

    [DllImport(Library, EntryPoint = "cublasLtMatmulPreferenceCreate",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int PreferenceCreate(out nint preference);

    [DllImport(Library, EntryPoint = "cublasLtMatmulPreferenceDestroy",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int PreferenceDestroy(nint preference);

    [DllImport(Library, EntryPoint = "cublasLtMatmulPreferenceSetAttribute",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int PreferenceSetAttribute(
        nint preference,
        int attribute,
        nint value,
        nuint size);

    [DllImport(Library, EntryPoint = "cublasLtMatmulAlgoGetHeuristic",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int MatmulAlgoGetHeuristic(
        nint handle,
        nint operation,
        nint first,
        nint second,
        nint outputC,
        nint outputD,
        nint preference,
        int requestedCount,
        nint results,
        nint returnedCount);

    [DllImport(Library, EntryPoint = "cublasLtMatmulAlgoCapGetAttribute",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int MatmulAlgoCapGetAttribute(
        nint algorithm,
        int attribute,
        nint buffer,
        nuint size,
        nint sizeWritten);

    [DllImport(Library, EntryPoint = "cublasLtMatmul",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int Matmul(
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
        nint stream);
}
