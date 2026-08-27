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
    private const int DescTransB = 4;
    private const int DescEpilogue = 7;
    private const int DescBiasPointer = 8;
    private const int PreferenceMaxWorkspaceBytes = 1;
    private const uint EpilogueBias = 4;
    private const uint EpilogueReluBias = 6;
    private const int HeuristicCandidateCount = 16;
    private const int AutotuneCandidateCount = 8;
    private const int WorkspaceBytes = 32 * 1024 * 1024;
    private const int PlanCacheCapacity = 128;
    private const int BackwardPlanCacheCapacity = 128;

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<
        int,
        Lazy<nint>> Handles = new();
    private static readonly BoundedDisposableLeaseCache<
        PlanKey,
        CachedPlan<Plan>> Plans = new(PlanCacheCapacity);
    private static readonly BoundedDisposableLeaseCache<
        BackwardPlanKey,
        CachedPlan<BackwardPlan>> BackwardPlans =
            new(BackwardPlanCacheCapacity);
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<
        int,
        Lazy<NativeCudaBuffer<byte>>> Workspaces = new();
    private static int _availability;
    private static int _backwardAvailability;

    internal static bool BackendActive => Volatile.Read(ref _availability) > 0;
    internal static bool BackwardBackendActive
        => Volatile.Read(ref _backwardAvailability) > 0;

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
            using BoundedDisposableLeaseCache<
                PlanKey,
                CachedPlan<Plan>>.Lease? lease = Plans.Acquire(
                    key,
                    static value => new CachedPlan<Plan>(CreatePlan(value)));
            if (lease is null)
                return false;
            Plan? plan = lease.Value.Value;
            if (plan is null)
                return false;

            // The bias pointer is mutable state on the shared operation
            // descriptor. Keep it paired with the enqueue that consumes it.
            lock (plan.ExecutionSync)
            {
                nint biasPointer = bias.NativePtr;
                int status = MatmulDescSetAttribute(
                    plan.Operation,
                    DescBiasPointer,
                    (nint)(&biasPointer),
                    (nuint)sizeof(nint));
                if (status != Success)
                    return false;

                status = ExecutePlan(
                    plan, accelerator, input, weight, output);
                if (status != Success)
                    return false;
            }
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

    internal static bool TryLinearBackwardInputBFloat16(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> outputGradient,
        NativeCudaBuffer<ushort> weight,
        NativeCudaBuffer<float> inputGradient,
        int rows,
        int inputWidth,
        int outputWidth)
        => TryLinearBackwardBFloat16(
            accelerator,
            new BackwardPlanKey(
                deviceIndex, rows, inputWidth, outputWidth,
                BackwardOperation.InputFloat32),
            weight.NativePtr,
            outputGradient.NativePtr,
            inputGradient.NativePtr,
            beta: 1f);

    internal static bool TryLinearBackwardInputBFloat16Direct(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> outputGradient,
        NativeCudaBuffer<ushort> weight,
        NativeCudaBuffer<ushort> inputGradient,
        int rows,
        int inputWidth,
        int outputWidth)
        => TryLinearBackwardBFloat16(
            accelerator,
            new BackwardPlanKey(
                deviceIndex, rows, inputWidth, outputWidth,
                BackwardOperation.InputBFloat16),
            weight.NativePtr,
            outputGradient.NativePtr,
            inputGradient.NativePtr,
            beta: 0f);

    internal static bool TryLinearBackwardWeightBFloat16(
        NativeCudaDevice accelerator,
        int deviceIndex,
        NativeCudaBuffer<ushort> input,
        NativeCudaBuffer<ushort> outputGradient,
        NativeCudaBuffer<float> weightGradient,
        int rows,
        int inputWidth,
        int outputWidth)
        => TryLinearBackwardBFloat16(
            accelerator,
            new BackwardPlanKey(
                deviceIndex, rows, inputWidth, outputWidth,
                BackwardOperation.WeightFloat32),
            input.NativePtr,
            outputGradient.NativePtr,
            weightGradient.NativePtr,
            beta: 1f);

    private static bool TryLinearBackwardBFloat16(
        NativeCudaDevice accelerator,
        BackwardPlanKey key,
        nint left,
        nint right,
        nint destination,
        float beta)
    {
        if (Environment.GetEnvironmentVariable("NNTRAIN_DISABLE_CUBLASLT") == "1"
            || Environment.GetEnvironmentVariable(
                "NNTRAIN_DISABLE_CUBLASLT_BACKWARD") == "1"
            || Volatile.Read(ref _availability) < 0)
        {
            return false;
        }
        try
        {
            accelerator.Bind();
            using BoundedDisposableLeaseCache<
                BackwardPlanKey,
                CachedPlan<BackwardPlan>>.Lease? lease = BackwardPlans.Acquire(
                    key,
                    static value => new CachedPlan<BackwardPlan>(
                        CreateBackwardPlan(value)));
            if (lease is null)
                return false;
            BackwardPlan? plan = lease.Value.Value;
            if (plan is null)
                return false;
            lock (plan.ExecutionSync)
            {
                float alpha = 1f;
                MatmulAlgorithm algorithm = plan.Algorithm;
                int status = Matmul(
                    plan.Handle,
                    plan.Operation,
                    (nint)(&alpha),
                    left,
                    plan.Left,
                    right,
                    plan.Right,
                    (nint)(&beta),
                    destination,
                    plan.Destination,
                    destination,
                    plan.Destination,
                    (nint)(&algorithm),
                    plan.Workspace.NativePtr,
                    plan.WorkspaceSize,
                    accelerator.DefaultStream);
                if (status != Success)
                    return false;
            }
            Volatile.Write(ref _availability, 1);
            Volatile.Write(ref _backwardAvailability, 1);
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
        nint operation = 0;
        if (MatmulDescCreate(
                out operation,
                Compute32FFast16BFloat,
                CudaR32F) != Success)
        {
            if (operation != 0)
                _ = MatmulDescDestroy(operation);
            return null;
        }
        nint weight = 0;
        nint input = 0;
        nint output = 0;
        nint preference = 0;
        bool ownershipTransferred = false;
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

            NativeCudaBuffer<byte> workspace = GetWorkspace(key.DeviceIndex);
            nuint maximumWorkspaceBytes = checked((nuint)workspace.Length);
            if (PreferenceSetAttribute(
                    preference,
                    PreferenceMaxWorkspaceBytes,
                    (nint)(&maximumWorkspaceBytes),
                    (nuint)sizeof(nuint)) != Success)
            {
                return null;
            }

            HeuristicResult* heuristics = stackalloc
                HeuristicResult[HeuristicCandidateCount];
            int count = 0;
            int status = MatmulAlgoGetHeuristic(
                handle,
                operation,
                weight,
                input,
                output,
                output,
                preference,
                HeuristicCandidateCount,
                (nint)heuristics,
                (nint)(&count));
            if (status != Success || count == 0)
            {
                return null;
            }

            var candidates = new List<AlgorithmCandidate>(count);
            for (int index = 0; index < count; index++)
            {
                ref HeuristicResult candidate = ref heuristics[index];
                if (candidate.State != Success
                    || candidate.WorkspaceSize > maximumWorkspaceBytes)
                {
                    continue;
                }
                float waves = float.IsFinite(candidate.WavesCount)
                    && candidate.WavesCount > 0f
                        ? candidate.WavesCount
                        : float.MaxValue;
                candidates.Add(new AlgorithmCandidate(
                    candidate.Algorithm,
                    candidate.WorkspaceSize,
                    waves));
            }
            if (candidates.Count == 0)
                return null;
            AlgorithmCandidate[] ordered = candidates
                .OrderBy(candidate => candidate.EstimatedWaves)
                .Take(AutotuneCandidateCount)
                .ToArray();
            var plan = new Plan(
                handle,
                operation,
                weight,
                input,
                output,
                workspace,
                ordered);
            ownershipTransferred = true;
            return plan;
        }
        finally
        {
            if (preference != 0)
                _ = PreferenceDestroy(preference);
            if (!ownershipTransferred)
                DestroyDescriptors(operation, weight, input, output);
        }
    }

    private static BackwardPlan? CreateBackwardPlan(BackwardPlanKey key)
    {
        nint handle = GetHandle(key.DeviceIndex);
        nint operation = 0;
        if (MatmulDescCreate(
                out operation,
                Compute32FFast16BFloat,
                CudaR32F) != Success)
        {
            if (operation != 0)
                _ = MatmulDescDestroy(operation);
            return null;
        }
        nint left = 0;
        nint right = 0;
        nint destination = 0;
        nint preference = 0;
        bool ownershipTransferred = false;
        try
        {
            bool weightGradient =
                key.Operation == BackwardOperation.WeightFloat32;
            int transposeA = OperationNone;
            int transposeB = weightGradient
                ? OperationTranspose
                : OperationNone;
            int leftRows = key.InputWidth;
            int leftColumns = weightGradient ? key.Rows : key.OutputWidth;
            int leftLeadingDimension = key.InputWidth;
            int rightRows = key.OutputWidth;
            int rightColumns = key.Rows;
            int rightLeadingDimension = key.OutputWidth;
            int destinationRows = key.InputWidth;
            int destinationColumns = weightGradient
                ? key.OutputWidth
                : key.Rows;
            int destinationLeadingDimension = key.InputWidth;
            int destinationType =
                key.Operation == BackwardOperation.InputBFloat16
                    ? CudaR16BF
                    : CudaR32F;

            if (MatmulDescSetAttribute(
                    operation,
                    DescTransA,
                    (nint)(&transposeA),
                    sizeof(int)) != Success
                || MatmulDescSetAttribute(
                    operation,
                    DescTransB,
                    (nint)(&transposeB),
                    sizeof(int)) != Success
                || MatrixLayoutCreate(
                    out left,
                    CudaR16BF,
                    (nuint)leftRows,
                    (nuint)leftColumns,
                    leftLeadingDimension) != Success
                || MatrixLayoutCreate(
                    out right,
                    CudaR16BF,
                    (nuint)rightRows,
                    (nuint)rightColumns,
                    rightLeadingDimension) != Success
                || MatrixLayoutCreate(
                    out destination,
                    destinationType,
                    (nuint)destinationRows,
                    (nuint)destinationColumns,
                    destinationLeadingDimension) != Success
                || PreferenceCreate(out preference) != Success)
            {
                return null;
            }

            NativeCudaBuffer<byte> workspace = GetWorkspace(key.DeviceIndex);
            nuint maximumWorkspaceBytes = checked((nuint)workspace.Length);
            if (PreferenceSetAttribute(
                    preference,
                    PreferenceMaxWorkspaceBytes,
                    (nint)(&maximumWorkspaceBytes),
                    (nuint)sizeof(nuint)) != Success)
            {
                return null;
            }

            HeuristicResult* heuristics = stackalloc
                HeuristicResult[HeuristicCandidateCount];
            int count = 0;
            int status = MatmulAlgoGetHeuristic(
                handle,
                operation,
                left,
                right,
                destination,
                destination,
                preference,
                HeuristicCandidateCount,
                (nint)heuristics,
                (nint)(&count));
            if (status != Success || count == 0)
                return null;

            int selected = -1;
            float selectedWaves = float.PositiveInfinity;
            for (int index = 0; index < count; index++)
            {
                ref HeuristicResult candidate = ref heuristics[index];
                if (candidate.State != Success
                    || candidate.WorkspaceSize > maximumWorkspaceBytes)
                {
                    continue;
                }
                float waves = float.IsFinite(candidate.WavesCount)
                    && candidate.WavesCount > 0f
                        ? candidate.WavesCount
                        : float.MaxValue;
                if (selected < 0 || waves < selectedWaves)
                {
                    selected = index;
                    selectedWaves = waves;
                }
            }
            if (selected < 0)
                return null;
            HeuristicResult heuristic = heuristics[selected];
            var plan = new BackwardPlan(
                handle,
                operation,
                left,
                right,
                destination,
                workspace,
                heuristic.Algorithm,
                heuristic.WorkspaceSize);
            ownershipTransferred = true;
            return plan;
        }
        finally
        {
            if (preference != 0)
                _ = PreferenceDestroy(preference);
            if (!ownershipTransferred)
            {
                DestroyDescriptors(
                    operation, left, right, destination);
            }
        }
    }

    private static void DestroyDescriptors(
        nint operation,
        nint first,
        nint second,
        nint third)
    {
        if (third != 0)
            _ = MatrixLayoutDestroy(third);
        if (second != 0)
            _ = MatrixLayoutDestroy(second);
        if (first != 0)
            _ = MatrixLayoutDestroy(first);
        if (operation != 0)
            _ = MatmulDescDestroy(operation);
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

    private static NativeCudaBuffer<byte> GetWorkspace(int deviceIndex)
        => Workspaces.GetOrAdd(
            deviceIndex,
            static index => new Lazy<NativeCudaBuffer<byte>>(
                () => ForgetMemoryV2Cuda.GetAccelerator(index)
                    .Allocate1D<byte>(WorkspaceBytes),
                LazyThreadSafetyMode.ExecutionAndPublication)).Value;

    private static int ExecutePlan(
        Plan plan,
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> input,
        NativeCudaBuffer<ushort> weight,
        NativeCudaBuffer<ushort> output)
    {
        int selected = Volatile.Read(ref plan.SelectedCandidate);
        if (selected < 0)
        {
            accelerator.Synchronize();
            long bestTicks = long.MaxValue;
            int best = -1;
            for (int index = 0;
                 index < plan.Candidates.Length;
                 index++)
            {
                AlgorithmCandidate candidate = plan.Candidates[index];
                // One untimed launch removes first-use setup from the
                // shape-specific measurement.
                int status = ExecuteCandidate(
                    plan, candidate, accelerator.DefaultStream,
                    input, weight, output);
                if (status != Success)
                    continue;
                accelerator.Synchronize();
                long started = System.Diagnostics.Stopwatch.GetTimestamp();
                status = ExecuteCandidate(
                    plan, candidate, accelerator.DefaultStream,
                    input, weight, output);
                if (status != Success)
                    continue;
                accelerator.Synchronize();
                long elapsed =
                    System.Diagnostics.Stopwatch.GetTimestamp() - started;
                if (elapsed < bestTicks)
                {
                    bestTicks = elapsed;
                    best = index;
                }
            }
            if (best < 0)
                return 1;
            Volatile.Write(ref plan.SelectedCandidate, best);
            selected = best;
        }
        return ExecuteCandidate(
            plan,
            plan.Candidates[selected],
            accelerator.DefaultStream,
            input,
            weight,
            output);
    }

    private static int ExecuteCandidate(
        Plan plan,
        AlgorithmCandidate candidate,
        nint stream,
        NativeCudaBuffer<ushort> input,
        NativeCudaBuffer<ushort> weight,
        NativeCudaBuffer<ushort> output)
    {
        float alpha = 1f;
        float beta = 0f;
        MatmulAlgorithm algorithm = candidate.Algorithm;
        return Matmul(
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
            plan.Workspace.NativePtr,
            candidate.WorkspaceSize,
            stream);
    }

    private readonly record struct PlanKey(
        int DeviceIndex,
        int Rows,
        int InputWidth,
        int OutputWidth,
        bool ApplyRelu);

    private enum BackwardOperation
    {
        InputFloat32,
        InputBFloat16,
        WeightFloat32,
    }

    private readonly record struct BackwardPlanKey(
        int DeviceIndex,
        int Rows,
        int InputWidth,
        int OutputWidth,
        BackwardOperation Operation);

    private sealed class CachedPlan<TPlan>(TPlan? value) : IDisposable
        where TPlan : class, IDisposable
    {
        internal TPlan? Value { get; } = value;

        public void Dispose() => Value?.Dispose();
    }

    private sealed class Plan(
        nint handle,
        nint operation,
        nint weight,
        nint input,
        nint output,
        NativeCudaBuffer<byte> workspace,
        AlgorithmCandidate[] candidates) : IDisposable
    {
        private int _disposed;

        internal nint Handle { get; } = handle;
        internal nint Operation { get; } = operation;
        internal nint Weight { get; } = weight;
        internal nint Input { get; } = input;
        internal nint Output { get; } = output;
        internal NativeCudaBuffer<byte> Workspace { get; } = workspace;
        internal AlgorithmCandidate[] Candidates { get; } = candidates;
        internal object ExecutionSync { get; } = new();
        internal int SelectedCandidate = -1;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            DestroyDescriptors(Operation, Weight, Input, Output);
        }
    }

    private readonly record struct AlgorithmCandidate(
        MatmulAlgorithm Algorithm,
        nuint WorkspaceSize,
        float EstimatedWaves);

    private sealed class BackwardPlan(
        nint handle,
        nint operation,
        nint left,
        nint right,
        nint destination,
        NativeCudaBuffer<byte> workspace,
        MatmulAlgorithm algorithm,
        nuint workspaceSize) : IDisposable
    {
        private int _disposed;

        internal nint Handle { get; } = handle;
        internal nint Operation { get; } = operation;
        internal nint Left { get; } = left;
        internal nint Right { get; } = right;
        internal nint Destination { get; } = destination;
        internal NativeCudaBuffer<byte> Workspace { get; } = workspace;
        internal MatmulAlgorithm Algorithm = algorithm;
        internal nuint WorkspaceSize { get; } = workspaceSize;
        internal object ExecutionSync { get; } = new();

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            DestroyDescriptors(Operation, Left, Right, Destination);
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
