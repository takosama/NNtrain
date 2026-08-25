using System.Runtime.InteropServices;

namespace NNtrain;

/// <summary>
/// Cached cuBLASLt plans for BF16 linear projections with fused epilogues.
/// </summary>
internal static unsafe class CudaBlasLt
{
    private const string Library = "cublasLt64_12";
    private const int Success = 0;
    private const int OperationNone = 0;
    private const int OperationTranspose = 1;
    private const int CudaR32F = 0;
    private const int CudaR16BF = 14;
    private const int Compute32FFast16BFloat = 75;
    private const int DescTransA = 3;
    private const int DescEpilogue = 7;
    private const int DescBiasPointer = 8;
    private const uint EpilogueBias = 4;
    private const uint EpilogueReluBias = 6;

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<
        int,
        Lazy<nint>> Handles = new();
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<
        PlanKey,
        Lazy<Plan?>> Plans = new();
    private static int _availability;

    internal static bool BackendActive => Volatile.Read(ref _availability) > 0;

    internal static bool TryLinearForwardBFloat16(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> input,
        NativeCudaBuffer<ushort> weight,
        NativeCudaBuffer<ushort> bias,
        NativeCudaBuffer<ushort> output,
        int rows,
        int inputWidth,
        int outputWidth,
        bool applyRelu)
    {
        if (Environment.GetEnvironmentVariable("NNTRAIN_DISABLE_CUBLASLT") == "1"
            || Volatile.Read(ref _availability) < 0)
            return false;
        try
        {
            accelerator.Bind();
            PlanKey key = new(deviceIndex, rows, inputWidth, outputWidth,
                applyRelu);
            Plan? plan = Plans.GetOrAdd(
                key,
                static value => new Lazy<Plan?>(
                    () => CreatePlan(value),
                    LazyThreadSafetyMode.ExecutionAndPublication)).Value;
            if (plan is null)
                return false;

            nint biasPointer = bias.NativePtr;
            int status = MatmulDescSetAttribute(
                plan.Operation,
                DescBiasPointer,
                (nint)(&biasPointer),
                (nuint)sizeof(nint));
            if (status != Success)
                return false;

            float alpha = 1f;
            float beta = 0f;
            MatmulAlgorithm algorithm = plan.Algorithm;
            nint stream = accelerator.DefaultStream;
            status = Matmul(
                plan.Handle,
                plan.Operation,
                (nint)(&alpha),
                weight.NativePtr,
                plan.Weight,
                input.NativePtr,
                plan.Input,
                (nint)(&beta),
                output.NativePtr,
                plan.Output,
                output.NativePtr,
                plan.Output,
                (nint)(&algorithm),
                0,
                0,
                stream);
            if (status != Success)
                return false;
            Volatile.Write(ref _availability, 1);
            return true;
        }
        catch (Exception exception) when (exception is DllNotFoundException
            or EntryPointNotFoundException
            or BadImageFormatException)
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }

    private static Plan? CreatePlan(PlanKey key)
    {
        nint handle = GetHandle(key.DeviceIndex);
        if (MatmulDescCreate(
                out nint operation,
                Compute32FFast16BFloat,
                CudaR32F) != Success)
        {
            return null;
        }
        nint weight = 0;
        nint input = 0;
        nint output = 0;
        nint preference = 0;
        try
        {
            int transpose = OperationTranspose;
            uint epilogue = key.ApplyRelu
                ? EpilogueReluBias
                : EpilogueBias;
            if (MatmulDescSetAttribute(
                    operation,
                    DescTransA,
                    (nint)(&transpose),
                    sizeof(int)) != Success
                || MatmulDescSetAttribute(
                    operation,
                    DescEpilogue,
                    (nint)(&epilogue),
                    sizeof(uint)) != Success
                || MatrixLayoutCreate(
                    out weight,
                    CudaR16BF,
                    (nuint)key.InputWidth,
                    (nuint)key.OutputWidth,
                    key.InputWidth) != Success
                || MatrixLayoutCreate(
                    out input,
                    CudaR16BF,
                    (nuint)key.InputWidth,
                    (nuint)key.Rows,
                    key.InputWidth) != Success
                || MatrixLayoutCreate(
                    out output,
                    CudaR16BF,
                    (nuint)key.OutputWidth,
                    (nuint)key.Rows,
                    key.OutputWidth) != Success
                || PreferenceCreate(out preference) != Success)
            {
                return null;
            }

            HeuristicResult heuristic = default;
            int count = 0;
            int status = MatmulAlgoGetHeuristic(
                handle,
                operation,
                weight,
                input,
                output,
                output,
                preference,
                1,
                (nint)(&heuristic),
                (nint)(&count));
            if (status != Success || count == 0 || heuristic.State != Success
                || heuristic.WorkspaceSize != 0)
            {
                return null;
            }
            return new Plan(
                handle,
                operation,
                weight,
                input,
                output,
                heuristic.Algorithm);
        }
        finally
        {
            if (preference != 0)
                _ = PreferenceDestroy(preference);
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

    private readonly record struct PlanKey(
        int DeviceIndex,
        int Rows,
        int InputWidth,
        int OutputWidth,
        bool ApplyRelu);

    private sealed class Plan(
        nint handle,
        nint operation,
        nint weight,
        nint input,
        nint output,
        MatmulAlgorithm algorithm)
    {
        internal nint Handle { get; } = handle;
        internal nint Operation { get; } = operation;
        internal nint Weight { get; } = weight;
        internal nint Input { get; } = input;
        internal nint Output { get; } = output;
        internal MatmulAlgorithm Algorithm = algorithm;
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

    [DllImport(Library, EntryPoint = "cublasLtMatmulPreferenceCreate",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int PreferenceCreate(out nint preference);

    [DllImport(Library, EntryPoint = "cublasLtMatmulPreferenceDestroy",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int PreferenceDestroy(nint preference);

    [DllImport(Library, EntryPoint = "cublasLtMatmulAlgoGetHeuristic",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int MatmulAlgoGetHeuristic(
        nint handle,
        nint operation,
        nint weight,
        nint input,
        nint outputC,
        nint outputD,
        nint preference,
        int requestedCount,
        nint results,
        nint returnedCount);

    [DllImport(Library, EntryPoint = "cublasLtMatmul",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int Matmul(
        nint handle,
        nint operation,
        nint alpha,
        nint weight,
        nint weightLayout,
        nint input,
        nint inputLayout,
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
