namespace NNtrain;

/// <summary>
/// Owns the reusable workspace boundary for CUDA embedding gradients. All
/// numeric modes accumulate their gradients in Float32 before the precision
/// policy publishes BF16/BFP8 state, so one dispatch serves float32, bfloat16,
/// bfp8, and mix8_32 without a host fallback.
/// </summary>
internal static class CudaEmbeddingBackwardDispatcher
{
    internal static int GetWorkspaceIntCount(int positionCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(positionCount);
        if (positionCount > 1 << 29)
        {
            throw new ArgumentOutOfRangeException(
                nameof(positionCount),
                "Embedding position count exceeds the CUDA hash workspace limit.");
        }

        int requested = checked(positionCount * 2);
        int hashCapacity = 2;
        while (hashCapacity < requested)
            hashCapacity = checked(hashCapacity * 2);
        return checked(2 * hashCapacity + 2 * positionCount + 1);
    }

    internal static void Backward(
        int deviceIndex,
        nint indices,
        nint outputGradient,
        nint tableGradient,
        int length,
        int width)
    {
        ValidateShape(length, width);
        int positionCount = length / width;
        int workspaceInts = GetWorkspaceIntCount(positionCount);
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<int> workspace =
            Tensor.RentCudaIntBuffer(deviceIndex, workspaceInts);
        try
        {
            CudaTensorNative.EmbeddingBackwardReduced(
                deviceIndex,
                indices,
                outputGradient,
                tableGradient,
                workspace.NativePtr,
                workspaceInts,
                length,
                width);
            CudaEmbeddingBackwardTelemetry.Record(
                includesPositions: false,
                positionCount,
                width,
                workspaceInts);
        }
        finally
        {
            Tensor.ReturnCudaIntBuffer(accelerator, workspace);
        }
    }

    internal static void BackwardBFloat16Gradient(
        int deviceIndex,
        nint indices,
        nint outputGradient,
        nint tableGradient,
        int length,
        int width)
    {
        ValidateShape(length, width);
        int positionCount = length / width;
        int workspaceInts = GetWorkspaceIntCount(positionCount);
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<int> workspace =
            Tensor.RentCudaIntBuffer(deviceIndex, workspaceInts);
        try
        {
            CudaPureBFloat16GradientNative.EmbeddingBackwardReduced(
                deviceIndex,
                indices,
                outputGradient,
                tableGradient,
                workspace.NativePtr,
                workspaceInts,
                length,
                width,
                accelerator.DefaultStream);
            CudaEmbeddingBackwardTelemetry.Record(
                includesPositions: false,
                positionCount,
                width,
                workspaceInts);
        }
        finally
        {
            Tensor.ReturnCudaIntBuffer(accelerator, workspace);
        }
    }

    internal static void BackwardWithPositions(
        int deviceIndex,
        nint indices,
        nint outputGradient,
        nint tokenGradient,
        nint positionGradient,
        int length,
        int sequence,
        int width)
    {
        ValidateShape(length, width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sequence);
        int positionCount = length / width;
        if (positionCount % sequence != 0)
        {
            throw new ArgumentException(
                "Embedding position count must be divisible by sequence.",
                nameof(sequence));
        }
        int workspaceInts = GetWorkspaceIntCount(positionCount);
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<int> workspace =
            Tensor.RentCudaIntBuffer(deviceIndex, workspaceInts);
        try
        {
            CudaTensorNative.EmbeddingPositionsBackwardReduced(
                deviceIndex,
                indices,
                outputGradient,
                tokenGradient,
                positionGradient,
                workspace.NativePtr,
                workspaceInts,
                length,
                sequence,
                width);
            CudaEmbeddingBackwardTelemetry.Record(
                includesPositions: true,
                positionCount,
                width,
                workspaceInts);
        }
        finally
        {
            Tensor.ReturnCudaIntBuffer(accelerator, workspace);
        }
    }

    internal static void BackwardWithPositionsBFloat16Gradient(
        int deviceIndex,
        nint indices,
        nint outputGradient,
        nint tokenGradient,
        nint positionGradient,
        int length,
        int sequence,
        int width)
    {
        ValidateShape(length, width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sequence);
        int positionCount = length / width;
        if (positionCount % sequence != 0)
        {
            throw new ArgumentException(
                "Embedding position count must be divisible by sequence.",
                nameof(sequence));
        }
        int workspaceInts = GetWorkspaceIntCount(positionCount);
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<int> workspace =
            Tensor.RentCudaIntBuffer(deviceIndex, workspaceInts);
        try
        {
            CudaPureBFloat16GradientNative
                .EmbeddingPositionsBackwardReduced(
                    deviceIndex,
                    indices,
                    outputGradient,
                    tokenGradient,
                    positionGradient,
                    workspace.NativePtr,
                    workspaceInts,
                    length,
                    sequence,
                    width,
                    accelerator.DefaultStream);
            CudaEmbeddingBackwardTelemetry.Record(
                includesPositions: true,
                positionCount,
                width,
                workspaceInts);
        }
        finally
        {
            Tensor.ReturnCudaIntBuffer(accelerator, workspace);
        }
    }

    private static void ValidateShape(int length, int width)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        if (length % width != 0)
        {
            throw new ArgumentException(
                "Embedding gradient length must be divisible by width.",
                nameof(length));
        }
    }
}

internal readonly record struct CudaEmbeddingBackwardTelemetrySnapshot(
    long ReducedLookupExecutions,
    long ReducedLookupWithPositionsExecutions,
    long GradientValuesAccumulated,
    long LegacyTableAtomicAddsAvoided,
    long ReducedTableAtomicAdds,
    long HashBookkeepingAtomicLowerBound,
    long WorkspaceIntsRented)
{
    public static CudaEmbeddingBackwardTelemetrySnapshot operator -(
        CudaEmbeddingBackwardTelemetrySnapshot left,
        CudaEmbeddingBackwardTelemetrySnapshot right)
        => new(
            left.ReducedLookupExecutions - right.ReducedLookupExecutions,
            left.ReducedLookupWithPositionsExecutions
                - right.ReducedLookupWithPositionsExecutions,
            left.GradientValuesAccumulated - right.GradientValuesAccumulated,
            left.LegacyTableAtomicAddsAvoided
                - right.LegacyTableAtomicAddsAvoided,
            left.ReducedTableAtomicAdds - right.ReducedTableAtomicAdds,
            left.HashBookkeepingAtomicLowerBound
                - right.HashBookkeepingAtomicLowerBound,
            left.WorkspaceIntsRented - right.WorkspaceIntsRented);
}

/// <summary>
/// Exact gradient-table atomic telemetry. Hash insertion uses a much smaller
/// O(position-count) number of bookkeeping atomics; the reported lower bound
/// excludes data-dependent linear-probe retries.
/// </summary>
internal static class CudaEmbeddingBackwardTelemetry
{
    private static long _reducedLookupExecutions;
    private static long _reducedLookupWithPositionsExecutions;
    private static long _gradientValuesAccumulated;
    private static long _legacyTableAtomicAddsAvoided;
    private static long _hashBookkeepingAtomicLowerBound;
    private static long _workspaceIntsRented;

    internal static CudaEmbeddingBackwardTelemetrySnapshot Snapshot => new(
        Volatile.Read(ref _reducedLookupExecutions),
        Volatile.Read(ref _reducedLookupWithPositionsExecutions),
        Volatile.Read(ref _gradientValuesAccumulated),
        Volatile.Read(ref _legacyTableAtomicAddsAvoided),
        ReducedTableAtomicAdds: 0,
        Volatile.Read(ref _hashBookkeepingAtomicLowerBound),
        Volatile.Read(ref _workspaceIntsRented));

    internal static void Record(
        bool includesPositions,
        int positionCount,
        int width,
        int workspaceInts)
    {
        long values = checked((long)positionCount * width);
        if (includesPositions)
            Interlocked.Increment(ref _reducedLookupWithPositionsExecutions);
        else
            Interlocked.Increment(ref _reducedLookupExecutions);
        Interlocked.Add(ref _gradientValuesAccumulated, values);
        Interlocked.Add(
            ref _legacyTableAtomicAddsAvoided,
            checked(values * (includesPositions ? 2 : 1)));
        // One successful CAS and one head exchange per position, plus at most
        // one unique-list counter increment per position.
        Interlocked.Add(
            ref _hashBookkeepingAtomicLowerBound,
            checked((long)positionCount * 2));
        Interlocked.Add(ref _workspaceIntsRented, workspaceInts);
    }
}
