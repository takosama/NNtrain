using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using NNtrain.Cuda.Interop;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain;

internal readonly record struct CudaBlasLtTelemetrySnapshot(
    long ForwardTensorCoreExecutions,
    long DirectBackwardTensorCoreExecutions,
    long AccumulatingBackwardCublasExecutions,
    long RejectedNonTensorCoreCandidates,
    ulong LastForwardNumericalImplementationFlags)
{
    public static CudaBlasLtTelemetrySnapshot operator -(
        CudaBlasLtTelemetrySnapshot left,
        CudaBlasLtTelemetrySnapshot right)
        => new(
            left.ForwardTensorCoreExecutions -
                right.ForwardTensorCoreExecutions,
            left.DirectBackwardTensorCoreExecutions -
                right.DirectBackwardTensorCoreExecutions,
            left.AccumulatingBackwardCublasExecutions -
                right.AccumulatingBackwardCublasExecutions,
            left.RejectedNonTensorCoreCandidates -
                right.RejectedNonTensorCoreCandidates,
            left.LastForwardNumericalImplementationFlags);
}

/// <summary>
/// Cached cuBLASLt plans for BF16 linear projections with fused epilogues.
/// </summary>
internal static unsafe class CudaBlasLt
{
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
    private const int AlgorithmNumericalImplementationFlags = 15;
    private const ulong NumericalImplementationHmma = 0x02UL;
    private const uint EpilogueBias = 4;
    private const uint EpilogueReluBias = 6;
    private const int HeuristicCandidateCount = 16;
    private const int AutotuneCandidateCount = 8;
    private const int WorkspaceBytes = 32 * 1024 * 1024;
    private const int PlanCacheCapacity = 128;
    private const int BackwardPlanCacheCapacity = 128;
    private const int FallbackResourceCapacity = 4;

    private static readonly ResettableBoundedDisposableLeaseCache<
        StreamKey,
        LaneResources> FallbackResources =
            new(FallbackResourceCapacity);
    private static readonly ConditionalWeakTable<
        IStreamExecutionLane,
        Lazy<LaneResources>> LaneResourceTable = new();
    private static int _activeLaneResourceCount;
    private static int _availability;
    private static int _backwardAvailability;
    private static long _forwardTensorCoreExecutions;
    private static long _directBackwardTensorCoreExecutions;
    private static long _accumulatingBackwardCublasExecutions;
    private static long _rejectedNonTensorCoreCandidates;
    private static long _lastForwardNumericalImplementationFlags;

    internal static bool BackendActive => Volatile.Read(ref _availability) > 0;
    internal static bool BackwardBackendActive
        => Volatile.Read(ref _backwardAvailability) > 0;
    internal static int ActiveLaneResourceCount =>
        Volatile.Read(ref _activeLaneResourceCount);
    internal static int FallbackResourceCount => FallbackResources.Count;
    internal static CudaBlasLtTelemetrySnapshot Telemetry => new(
        Interlocked.Read(ref _forwardTensorCoreExecutions),
        Interlocked.Read(ref _directBackwardTensorCoreExecutions),
        Interlocked.Read(ref _accumulatingBackwardCublasExecutions),
        Interlocked.Read(ref _rejectedNonTensorCoreCandidates),
        unchecked((ulong)Interlocked.Read(
            ref _lastForwardNumericalImplementationFlags)));

    internal static void RecordAccumulatingBackwardCublasExecution()
        => Interlocked.Increment(
            ref _accumulatingBackwardCublasExecutions);

    internal static void DisposeFallbackResources()
        => FallbackResources.Dispose();

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
        if (CudaDispatchPolicy.Current.DisableCublasLt
            || Volatile.Read(ref _availability) < 0)
            return false;
        try
        {
            accelerator.Bind();
            nint computeStream = accelerator.DefaultStream;
            PlanKey key = new(deviceIndex, rows, inputWidth, outputWidth,
                applyRelu)
            {
                ComputeStream = computeStream,
            };
            LaneResources? laneResources = GetLaneResources(
                deviceIndex,
                computeStream);
            using BoundedDisposableLeaseCache<
                StreamKey,
                LaneResources>.Lease? fallbackLease = laneResources is null
                    ? FallbackResources.Acquire(
                        new StreamKey(deviceIndex, computeStream),
                        static value => new LaneResources(
                            value.DeviceIndex,
                            value.ComputeStream,
                            laneOwned: false))
                    : null;
            LaneResources resources = laneResources
                ?? fallbackLease?.Value
                ?? throw new InvalidOperationException(
                    "cuBLASLt fallback resources could not be created.");
            BoundedDisposableLeaseCache<
                PlanKey,
                CachedPlan<Plan>>.Lease? lease = resources.Plans.Acquire(
                    key,
                    resources.ForwardPlanFactory);
            using (lease)
            {
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
                int selected = Volatile.Read(ref plan.SelectedCandidate);
                if (selected >= 0)
                {
                    ulong implementationFlags =
                        plan.Candidates[selected]
                            .NumericalImplementationFlags;
                    Interlocked.Exchange(
                        ref _lastForwardNumericalImplementationFlags,
                        unchecked((long)implementationFlags));
                    if ((implementationFlags &
                        NumericalImplementationHmma) != 0)
                    {
                        Interlocked.Increment(
                            ref _forwardTensorCoreExecutions);
                    }
                }
            }
            Volatile.Write(ref _availability, 1);
            return true;
            }
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
        CudaDispatchPolicy dispatch = CudaDispatchPolicy.Current;
        if (dispatch.DisableCublasLt
            || dispatch.DisableCublasLtBackward
            || Volatile.Read(ref _availability) < 0)
        {
            return false;
        }
        try
        {
            accelerator.Bind();
            key = key with
            {
                ComputeStream = accelerator.DefaultStream,
            };
            LaneResources? laneResources = GetLaneResources(
                key.DeviceIndex,
                key.ComputeStream);
            using BoundedDisposableLeaseCache<
                StreamKey,
                LaneResources>.Lease? fallbackLease = laneResources is null
                    ? FallbackResources.Acquire(
                        new StreamKey(
                            key.DeviceIndex,
                            key.ComputeStream),
                        static value => new LaneResources(
                            value.DeviceIndex,
                            value.ComputeStream,
                            laneOwned: false))
                    : null;
            LaneResources resources = laneResources
                ?? fallbackLease?.Value
                ?? throw new InvalidOperationException(
                    "cuBLASLt fallback resources could not be created.");
            BoundedDisposableLeaseCache<
                BackwardPlanKey,
                CachedPlan<BackwardPlan>>.Lease? lease =
                    resources.BackwardPlans.Acquire(
                        key,
                        resources.BackwardPlanFactory);
            using (lease)
            {
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
                    plan.ComputeStream);
                if (status != Success)
                    return false;
                if ((plan.NumericalImplementationFlags &
                    NumericalImplementationHmma) != 0)
                {
                    Interlocked.Increment(
                        ref _directBackwardTensorCoreExecutions);
                }
            }
            Volatile.Write(ref _availability, 1);
            Volatile.Write(ref _backwardAvailability, 1);
            return true;
            }
        }
        catch (Exception exception) when (exception is DllNotFoundException
            or EntryPointNotFoundException
            or BadImageFormatException)
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }

    private static Plan? CreatePlan(
        PlanKey key,
        LaneResources resources)
    {
        nint handle = resources.Handle;
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

            NativeCudaBuffer<byte> workspace = resources.Workspace;
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
                _ = TryGetNumericalImplementationFlags(
                    ref candidate.Algorithm,
                    out ulong implementationFlags);
                candidates.Add(new AlgorithmCandidate(
                    candidate.Algorithm,
                    candidate.WorkspaceSize,
                    waves,
                    implementationFlags));
            }
            if (candidates.Count == 0)
                return null;
            AlgorithmCandidate[] tensorCoreCandidates = candidates
                .Where(candidate =>
                    (candidate.NumericalImplementationFlags &
                        NumericalImplementationHmma) != 0)
                .ToArray();
            IEnumerable<AlgorithmCandidate> eligible =
                tensorCoreCandidates.Length > 0
                    ? tensorCoreCandidates
                    : candidates;
            if (tensorCoreCandidates.Length > 0)
            {
                Interlocked.Add(
                    ref _rejectedNonTensorCoreCandidates,
                    candidates.Count - tensorCoreCandidates.Length);
            }
            AlgorithmCandidate[] ordered = eligible
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
                ordered,
                key.ComputeStream,
                resources);
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

    private static BackwardPlan? CreateBackwardPlan(
        BackwardPlanKey key,
        LaneResources resources)
    {
        nint handle = resources.Handle;
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

            NativeCudaBuffer<byte> workspace = resources.Workspace;
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
                _ = TryGetNumericalImplementationFlags(
                    ref candidate.Algorithm,
                    out ulong implementationFlags);
                candidates.Add(new AlgorithmCandidate(
                    candidate.Algorithm,
                    candidate.WorkspaceSize,
                    waves,
                    implementationFlags));
            }
            if (candidates.Count == 0)
                return null;
            AlgorithmCandidate[] tensorCoreCandidates = candidates
                .Where(candidate =>
                    (candidate.NumericalImplementationFlags &
                        NumericalImplementationHmma) != 0)
                .ToArray();
            IEnumerable<AlgorithmCandidate> eligible =
                tensorCoreCandidates.Length > 0
                    ? tensorCoreCandidates
                    : candidates;
            if (tensorCoreCandidates.Length > 0)
            {
                Interlocked.Add(
                    ref _rejectedNonTensorCoreCandidates,
                    candidates.Count - tensorCoreCandidates.Length);
            }
            AlgorithmCandidate selected = eligible
                .OrderBy(candidate => candidate.EstimatedWaves)
                .ThenBy(candidate => candidate.WorkspaceSize)
                .First();
            var plan = new BackwardPlan(
                handle,
                operation,
                left,
                right,
                destination,
                workspace,
                selected.Algorithm,
                selected.WorkspaceSize,
                selected.NumericalImplementationFlags,
                key.ComputeStream,
                resources);
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

    private static bool TryGetNumericalImplementationFlags(
        ref MatmulAlgorithm algorithm,
        out ulong implementationFlags)
    {
        ulong flags = 0;
        nuint bytesWritten = 0;
        MatmulAlgorithm algorithmCopy = algorithm;
        int status = MatmulAlgoCapGetAttribute(
            (nint)(&algorithmCopy),
            AlgorithmNumericalImplementationFlags,
            (nint)(&flags),
            (nuint)sizeof(ulong),
            (nint)(&bytesWritten));
        implementationFlags = flags;
        return status == Success
            && bytesWritten == (nuint)sizeof(ulong);
    }

    private static LaneResources? GetLaneResources(
        int deviceIndex,
        nint computeStream)
    {
        if (!TensorExecutionContext.TryGetCudaStreamLane(
                deviceIndex,
                out IStreamExecutionLane lane)
            || lane.ComputeStreamHandle != computeStream)
        {
            return null;
        }
        return LaneResourceTable.GetValue(
            lane,
            static value => new Lazy<LaneResources>(
                () => ExecutionLaneResources.Attach(
                    value,
                    new LaneResources(
                        value.DeviceIndex,
                        value.ComputeStreamHandle,
                        laneOwned: true)),
                LazyThreadSafetyMode.ExecutionAndPublication)).Value;
    }

    private static nint CreateHandle()
    {
        if (Create(out nint handle) != Success)
            throw new InvalidOperationException("cublasLtCreate failed.");
        return handle;
    }

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
            NativeCudaRuntime.SynchronizeComputeStream(
                accelerator,
                plan.ComputeStream);
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
                    plan, candidate, plan.ComputeStream,
                    input, weight, output);
                if (status != Success)
                    continue;
                NativeCudaRuntime.SynchronizeComputeStream(
                    accelerator,
                    plan.ComputeStream);
                long started = System.Diagnostics.Stopwatch.GetTimestamp();
                status = ExecuteCandidate(
                    plan, candidate, plan.ComputeStream,
                    input, weight, output);
                if (status != Success)
                    continue;
                NativeCudaRuntime.SynchronizeComputeStream(
                    accelerator,
                    plan.ComputeStream);
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
            plan.ComputeStream,
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
        bool ApplyRelu)
    {
        internal nint ComputeStream { get; init; }
    }

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
        BackwardOperation Operation)
    {
        internal nint ComputeStream { get; init; }
    }

    private readonly record struct StreamKey(
        int DeviceIndex,
        nint ComputeStream);

    private sealed class LaneResources : IDisposable
    {
        private readonly Lazy<nint> _handle;
        private readonly Lazy<NativeCudaBuffer<byte>> _workspace;
        private readonly bool _laneOwned;
        private int _disposed;

        internal LaneResources(
            int deviceIndex,
            nint computeStream,
            bool laneOwned)
        {
            DeviceIndex = deviceIndex;
            ComputeStream = computeStream;
            _laneOwned = laneOwned;
            _handle = new Lazy<nint>(
                CreateHandle,
                LazyThreadSafetyMode.ExecutionAndPublication);
            _workspace = new Lazy<NativeCudaBuffer<byte>>(
                () => ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
                    .Allocate1D<byte>(
                        WorkspaceBytes,
                        CudaMemoryKind.Workspace),
                LazyThreadSafetyMode.ExecutionAndPublication);
            Plans = new(PlanCacheCapacity);
            BackwardPlans = new(BackwardPlanCacheCapacity);
            ForwardPlanFactory = CreateForwardPlan;
            BackwardPlanFactory = CreateBackwardPlan;
            if (laneOwned)
                Interlocked.Increment(ref _activeLaneResourceCount);
        }

        internal int DeviceIndex { get; }
        internal nint ComputeStream { get; }
        internal nint Handle => _handle.Value;
        internal NativeCudaBuffer<byte> Workspace => _workspace.Value;
        internal BoundedDisposableLeaseCache<
            PlanKey,
            CachedPlan<Plan>> Plans { get; }
        internal BoundedDisposableLeaseCache<
            BackwardPlanKey,
            CachedPlan<BackwardPlan>> BackwardPlans { get; }
        internal Func<PlanKey, CachedPlan<Plan>?> ForwardPlanFactory { get; }
        internal Func<BackwardPlanKey, CachedPlan<BackwardPlan>?>
            BackwardPlanFactory { get; }
        internal bool IsDisposing => Volatile.Read(ref _disposed) != 0;

        private CachedPlan<Plan> CreateForwardPlan(PlanKey key)
            => new(CudaBlasLt.CreatePlan(key, this));

        private CachedPlan<BackwardPlan> CreateBackwardPlan(
            BackwardPlanKey key)
            => new(CudaBlasLt.CreateBackwardPlan(key, this));

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            if (!_laneOwned)
            {
                NativeCudaRuntime.DisposeAfterStreamFence(
                    DeviceIndex,
                    ComputeStream,
                    DisposeOwnedResources);
                return;
            }
            DisposeOwnedResources();
        }

        internal void DisposeChildAfterFence(Action dispose)
        {
            ArgumentNullException.ThrowIfNull(dispose);
            if (IsDisposing)
            {
                dispose();
                return;
            }
            NativeCudaRuntime.DisposeAfterStreamFence(
                DeviceIndex,
                ComputeStream,
                dispose);
        }

        private void DisposeOwnedResources()
        {
            List<Exception>? failures = null;
            TryDispose(Plans.Dispose, ref failures);
            TryDispose(BackwardPlans.Dispose, ref failures);
            if (_workspace.IsValueCreated)
                TryDispose(_workspace.Value.Dispose, ref failures);
            if (_handle.IsValueCreated)
            {
                TryDispose(
                    () =>
                    {
                        int status = Destroy(_handle.Value);
                        if (status != Success)
                        {
                            throw new InvalidOperationException(
                                $"cublasLtDestroy failed with cuBLAS status {status}.");
                        }
                    },
                    ref failures);
            }
            if (_laneOwned)
                Interlocked.Decrement(ref _activeLaneResourceCount);
            if (failures is not null)
            {
                throw new AggregateException(
                    $"cuBLASLt stream resources failed to dispose on device {DeviceIndex}.",
                    failures);
            }
        }

        private static void TryDispose(
            Action action,
            ref List<Exception>? failures)
        {
            try
            {
                action();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
    }

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
        AlgorithmCandidate[] candidates,
        nint computeStream,
        LaneResources owner) : IDisposable
    {
        private int _disposed;

        internal nint Handle { get; } = handle;
        internal nint Operation { get; } = operation;
        internal nint Weight { get; } = weight;
        internal nint Input { get; } = input;
        internal nint Output { get; } = output;
        internal NativeCudaBuffer<byte> Workspace { get; } = workspace;
        internal AlgorithmCandidate[] Candidates { get; } = candidates;
        internal nint ComputeStream { get; } = computeStream;
        internal object ExecutionSync { get; } = new();
        internal int SelectedCandidate = -1;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            owner.DisposeChildAfterFence(
                () => DestroyDescriptors(
                    Operation,
                    Weight,
                    Input,
                    Output));
        }
    }

    private readonly record struct AlgorithmCandidate(
        MatmulAlgorithm Algorithm,
        nuint WorkspaceSize,
        float EstimatedWaves,
        ulong NumericalImplementationFlags);

    private sealed class BackwardPlan(
        nint handle,
        nint operation,
        nint left,
        nint right,
        nint destination,
        NativeCudaBuffer<byte> workspace,
        MatmulAlgorithm algorithm,
        nuint workspaceSize,
        ulong numericalImplementationFlags,
        nint computeStream,
        LaneResources owner) : IDisposable
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
        internal ulong NumericalImplementationFlags { get; } =
            numericalImplementationFlags;
        internal nint ComputeStream { get; } = computeStream;
        internal object ExecutionSync { get; } = new();

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            owner.DisposeChildAfterFence(
                () => DestroyDescriptors(
                    Operation,
                    Left,
                    Right,
                    Destination));
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

    private static int Create(out nint handle)
        => CudaNativeGateway.CublasLtCreate(out handle);

    private static int Destroy(nint handle)
        => CudaNativeGateway.CublasLtDestroy(handle);

    private static int MatmulDescCreate(
        out nint descriptor,
        int computeType,
        int scaleType)
        => CudaNativeGateway.CublasLtMatmulDescCreate(
            out descriptor, computeType, scaleType);

    private static int MatmulDescDestroy(nint descriptor)
        => CudaNativeGateway.CublasLtMatmulDescDestroy(descriptor);

    private static int MatmulDescSetAttribute(
        nint descriptor,
        int attribute,
        nint value,
        nuint size)
        => CudaNativeGateway.CublasLtMatmulDescSetAttribute(
            descriptor, attribute, value, size);

    private static int MatrixLayoutCreate(
        out nint layout,
        int type,
        nuint rows,
        nuint columns,
        long leadingDimension)
        => CudaNativeGateway.CublasLtMatrixLayoutCreate(
            out layout, type, rows, columns, leadingDimension);

    private static int MatrixLayoutDestroy(nint layout)
        => CudaNativeGateway.CublasLtMatrixLayoutDestroy(layout);

    private static int PreferenceCreate(out nint preference)
        => CudaNativeGateway.CublasLtPreferenceCreate(out preference);

    private static int PreferenceDestroy(nint preference)
        => CudaNativeGateway.CublasLtPreferenceDestroy(preference);

    private static int PreferenceSetAttribute(
        nint preference,
        int attribute,
        nint value,
        nuint size)
        => CudaNativeGateway.CublasLtPreferenceSetAttribute(
            preference, attribute, value, size);

    private static int MatmulAlgoGetHeuristic(
        nint handle,
        nint operation,
        nint weight,
        nint input,
        nint outputC,
        nint outputD,
        nint preference,
        int requestedCount,
        nint results,
        nint returnedCount)
        => CudaNativeGateway.CublasLtMatmulAlgoGetHeuristic(
            handle, operation, weight, input, outputC, outputD, preference,
            requestedCount, results, returnedCount);

    private static int MatmulAlgoCapGetAttribute(
        nint algorithm,
        int attribute,
        nint buffer,
        nuint size,
        nint sizeWritten)
        => CudaNativeGateway.CublasLtMatmulAlgoCapGetAttribute(
            algorithm, attribute, buffer, size, sizeWritten);

    private static int Matmul(
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
        nint stream)
        => CudaNativeGateway.CublasLtMatmul(
            handle, operation, alpha, weight, weightLayout, input,
            inputLayout, beta, outputC, outputCLayout, outputD,
            outputDLayout, algorithm, workspace, workspaceSize, stream);

}
