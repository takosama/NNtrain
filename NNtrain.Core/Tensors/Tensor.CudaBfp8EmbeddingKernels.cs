using System.Runtime.ExceptionServices;

namespace NNtrain;

internal readonly record struct CudaBfp8EmbeddingTelemetrySnapshot(
    long LookupExecutions,
    long LookupWithPositionsExecutions,
    long SelectedValuesDecoded,
    long ReductionWorkspaceElements)
{
    public static CudaBfp8EmbeddingTelemetrySnapshot operator -(
        CudaBfp8EmbeddingTelemetrySnapshot left,
        CudaBfp8EmbeddingTelemetrySnapshot right)
        => new(
            left.LookupExecutions - right.LookupExecutions,
            left.LookupWithPositionsExecutions -
                right.LookupWithPositionsExecutions,
            left.SelectedValuesDecoded - right.SelectedValuesDecoded,
            left.ReductionWorkspaceElements -
                right.ReductionWorkspaceElements);
}

internal static class CudaBfp8EmbeddingTelemetry
{
    private static long _lookupExecutions;
    private static long _lookupWithPositionsExecutions;
    private static long _selectedValuesDecoded;
    private static long _reductionWorkspaceElements;

    internal static CudaBfp8EmbeddingTelemetrySnapshot Snapshot => new(
        Interlocked.Read(ref _lookupExecutions),
        Interlocked.Read(ref _lookupWithPositionsExecutions),
        Interlocked.Read(ref _selectedValuesDecoded),
        Interlocked.Read(ref _reductionWorkspaceElements));

    internal static void Record(
        bool includesPositions,
        int outputLength,
        int workspaceLength)
    {
        if (includesPositions)
            Interlocked.Increment(ref _lookupWithPositionsExecutions);
        else
            Interlocked.Increment(ref _lookupExecutions);
        Interlocked.Add(
            ref _selectedValuesDecoded,
            includesPositions
                ? checked((long)outputLength * 2)
                : outputLength);
        Interlocked.Add(ref _reductionWorkspaceElements, workspaceLength);
    }
}

internal static partial class TensorCudaKernels
{
    internal static Bfp8EmbeddingResidentContext
        EmbeddingForwardBfp8Resident(
            Tensor table,
            int[] indices,
            int width,
            Bfp8QuantizationDescriptor outputDescriptor)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(outputDescriptor);
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        int outputLength = checked(indices.Length * width);
        NativeCudaBuffer<int>? indicesBuffer = null;
        CudaBfp8OwnedBuffers? output = null;
        NativeCudaBuffer<float>? workspace = null;
        try
        {
            CudaBfp8BufferView tableView =
                table.EnsureCudaBfp8Buffer(deviceIndex);
            indicesBuffer = Tensor.RentCudaIntBuffer(deviceIndex, indices);
            output = CudaBfp8OwnedBuffers.Allocate(
                accelerator, outputLength, outputDescriptor);
            int workspaceLength =
                CudaBfp8EmbeddingNative.GetWorkspaceLength(
                    outputLength, output.Scales.Length);
            if (workspaceLength != 0)
            {
                workspace = Tensor.RentCudaFloatBuffer(
                    deviceIndex, workspaceLength);
            }
            CudaBfp8EmbeddingNative.EmbeddingForward(
                deviceIndex,
                tableView,
                indicesBuffer,
                width,
                output,
                workspace,
                accelerator.DefaultStream);
            CudaBfp8EmbeddingTelemetry.Record(
                includesPositions: false,
                outputLength,
                workspaceLength);
            return new Bfp8EmbeddingResidentContext(
                output,
                indicesBuffer,
                workspace,
                accelerator,
                sourceTableLength: table.Numel,
                selectedOutputLength: outputLength);
        }
        catch (Exception failure)
        {
            RollbackEmbeddingResources(
                failure, output, workspace, indicesBuffer, accelerator);
            throw;
        }
    }

    internal static void EmbeddingBackwardBfp8Resident(
        Tensor output,
        Tensor table,
        Bfp8EmbeddingResidentContext context,
        int width)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(context);
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaBuffer<float> outputGradient =
            output.EnsureCudaGradientBuffer(deviceIndex);
        NativeCudaBuffer<float> tableGradient =
            table.EnsureCudaGradientBuffer(deviceIndex);
        CudaEmbeddingBackwardDispatcher.Backward(
            deviceIndex,
            context.Indices.NativePtr,
            outputGradient.NativePtr,
            tableGradient.NativePtr,
            output.Numel,
            width);
        table.MarkCudaGradientMutated(deviceIndex);
    }

    internal static Bfp8EmbeddingPositionsResidentContext
        EmbeddingWithPositionsForwardBfp8Resident(
            Tensor tokenTable,
            Tensor positionTable,
            int[] indices,
            int sequenceLength,
            int width,
            Bfp8QuantizationDescriptor outputDescriptor)
    {
        ArgumentNullException.ThrowIfNull(tokenTable);
        ArgumentNullException.ThrowIfNull(positionTable);
        ArgumentNullException.ThrowIfNull(indices);
        ArgumentNullException.ThrowIfNull(outputDescriptor);
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        int outputLength = checked(indices.Length * width);
        NativeCudaBuffer<int>? indicesBuffer = null;
        CudaBfp8OwnedBuffers? output = null;
        NativeCudaBuffer<float>? workspace = null;
        try
        {
            CudaBfp8BufferView tokenView =
                tokenTable.EnsureCudaBfp8Buffer(deviceIndex);
            CudaBfp8BufferView positionView =
                positionTable.EnsureCudaBfp8Buffer(deviceIndex);
            indicesBuffer = Tensor.RentCudaIntBuffer(deviceIndex, indices);
            output = CudaBfp8OwnedBuffers.Allocate(
                accelerator, outputLength, outputDescriptor);
            int workspaceLength =
                CudaBfp8EmbeddingNative.GetWorkspaceLength(
                    outputLength, output.Scales.Length);
            if (workspaceLength != 0)
            {
                workspace = Tensor.RentCudaFloatBuffer(
                    deviceIndex, workspaceLength);
            }
            CudaBfp8EmbeddingNative.EmbeddingWithPositionsForward(
                deviceIndex,
                tokenView,
                positionView,
                indicesBuffer,
                sequenceLength,
                width,
                output,
                workspace,
                accelerator.DefaultStream);
            CudaBfp8EmbeddingTelemetry.Record(
                includesPositions: true,
                outputLength,
                workspaceLength);
            return new Bfp8EmbeddingPositionsResidentContext(
                output,
                indicesBuffer,
                workspace,
                accelerator,
                tokenTableLength: tokenTable.Numel,
                positionTableLength: positionTable.Numel,
                selectedOutputLength: outputLength);
        }
        catch (Exception failure)
        {
            RollbackEmbeddingResources(
                failure, output, workspace, indicesBuffer, accelerator);
            throw;
        }
    }

    internal static void EmbeddingWithPositionsBackwardBfp8Resident(
        Tensor output,
        Tensor tokenTable,
        Tensor positionTable,
        Bfp8EmbeddingPositionsResidentContext context,
        int sequenceLength,
        int width)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(tokenTable);
        ArgumentNullException.ThrowIfNull(positionTable);
        ArgumentNullException.ThrowIfNull(context);
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaBuffer<float> outputGradient =
            output.EnsureCudaGradientBuffer(deviceIndex);
        NativeCudaBuffer<float> tokenGradient =
            tokenTable.EnsureCudaGradientBuffer(deviceIndex);
        NativeCudaBuffer<float> positionGradient =
            positionTable.EnsureCudaGradientBuffer(deviceIndex);
        CudaEmbeddingBackwardDispatcher.BackwardWithPositions(
            deviceIndex,
            context.Indices.NativePtr,
            outputGradient.NativePtr,
            tokenGradient.NativePtr,
            positionGradient.NativePtr,
            output.Numel,
            sequenceLength,
            width);
        tokenTable.MarkCudaGradientMutated(deviceIndex);
        positionTable.MarkCudaGradientMutated(deviceIndex);
    }

    private static void RollbackEmbeddingResources(
        Exception failure,
        CudaBfp8OwnedBuffers? output,
        NativeCudaBuffer<float>? workspace,
        NativeCudaBuffer<int>? indices,
        NativeCudaDevice accelerator)
    {
        var failures = new List<Exception> { failure };
        if (output is not null)
            TryReleaseEmbeddingResource(output.Dispose, failures);
        if (workspace is not null)
        {
            TryReleaseEmbeddingResource(
                () => Tensor.ReturnCudaFloatBuffer(accelerator, workspace),
                failures);
        }
        if (indices is not null)
        {
            TryReleaseEmbeddingResource(
                () => Tensor.ReturnCudaIntBuffer(accelerator, indices),
                failures);
        }
        if (failures.Count == 1)
            ExceptionDispatchInfo.Capture(failure).Throw();
        throw new AggregateException(
            "CUDA BFP8 embedding launch and resource rollback failed.",
            failures);
    }

    private static void TryReleaseEmbeddingResource(
        Action? release,
        List<Exception> failures)
    {
        if (release is null)
            return;
        try
        {
            release();
        }
        catch (Exception cleanupFailure)
        {
            if (cleanupFailure is AggregateException aggregate)
                failures.AddRange(aggregate.Flatten().InnerExceptions);
            else
                failures.Add(cleanupFailure);
        }
    }

    internal sealed class Bfp8EmbeddingResidentContext : IDisposable
    {
        private readonly NativeCudaDevice _accelerator;
        private int _disposed;

        internal Bfp8EmbeddingResidentContext(
            CudaBfp8OwnedBuffers output,
            NativeCudaBuffer<int> indices,
            NativeCudaBuffer<float>? workspace,
            NativeCudaDevice accelerator,
            int sourceTableLength,
            int selectedOutputLength)
        {
            Output = output ?? throw new ArgumentNullException(nameof(output));
            Indices = indices ?? throw new ArgumentNullException(nameof(indices));
            Workspace = workspace;
            _accelerator = accelerator ??
                throw new ArgumentNullException(nameof(accelerator));
            SourceTableLength = sourceTableLength;
            SelectedOutputLength = selectedOutputLength;
        }

        internal CudaBfp8OwnedBuffers Output { get; }
        internal NativeCudaBuffer<int> Indices { get; }
        internal NativeCudaBuffer<float>? Workspace { get; }
        internal int WorkspaceLength => Workspace?.Length ?? 0;
        internal int SourceTableLength { get; }
        internal int SelectedOutputLength { get; }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            var failures = new List<Exception>();
            TryReleaseEmbeddingResource(Output.Dispose, failures);
            if (Workspace is not null)
            {
                TryReleaseEmbeddingResource(
                    () => Tensor.ReturnCudaFloatBuffer(
                        _accelerator, Workspace),
                    failures);
            }
            TryReleaseEmbeddingResource(
                () => Tensor.ReturnCudaIntBuffer(_accelerator, Indices),
                failures);
            if (failures.Count != 0)
            {
                throw new AggregateException(
                    "CUDA BFP8 embedding resources failed to dispose.",
                    failures);
            }
            GC.SuppressFinalize(this);
        }
    }

    internal sealed class Bfp8EmbeddingPositionsResidentContext : IDisposable
    {
        private readonly NativeCudaDevice _accelerator;
        private int _disposed;

        internal Bfp8EmbeddingPositionsResidentContext(
            CudaBfp8OwnedBuffers output,
            NativeCudaBuffer<int> indices,
            NativeCudaBuffer<float>? workspace,
            NativeCudaDevice accelerator,
            int tokenTableLength,
            int positionTableLength,
            int selectedOutputLength)
        {
            Output = output ?? throw new ArgumentNullException(nameof(output));
            Indices = indices ?? throw new ArgumentNullException(nameof(indices));
            Workspace = workspace;
            _accelerator = accelerator ??
                throw new ArgumentNullException(nameof(accelerator));
            TokenTableLength = tokenTableLength;
            PositionTableLength = positionTableLength;
            SelectedOutputLength = selectedOutputLength;
        }

        internal CudaBfp8OwnedBuffers Output { get; }
        internal NativeCudaBuffer<int> Indices { get; }
        internal NativeCudaBuffer<float>? Workspace { get; }
        internal int WorkspaceLength => Workspace?.Length ?? 0;
        internal int TokenTableLength { get; }
        internal int PositionTableLength { get; }
        internal int SelectedOutputLength { get; }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            var failures = new List<Exception>();
            TryReleaseEmbeddingResource(Output.Dispose, failures);
            if (Workspace is not null)
            {
                TryReleaseEmbeddingResource(
                    () => Tensor.ReturnCudaFloatBuffer(
                        _accelerator, Workspace),
                    failures);
            }
            TryReleaseEmbeddingResource(
                () => Tensor.ReturnCudaIntBuffer(_accelerator, Indices),
                failures);
            if (failures.Count != 0)
            {
                throw new AggregateException(
                    "CUDA BFP8 embedding/position resources failed to dispose.",
                    failures);
            }
            GC.SuppressFinalize(this);
        }
    }
}
