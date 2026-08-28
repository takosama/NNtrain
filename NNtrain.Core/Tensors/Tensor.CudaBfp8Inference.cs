namespace NNtrain;

/// <summary>
/// Narrows BF16 compute over BFP8 storage to an explicitly requested
/// generation session. Ordinary no-grad calls retain their public BFP8 output
/// contract and continue to use the regular BFP8 kernels.
/// </summary>
internal static class CudaBfp8InferenceComputeScope
{
    private static readonly AsyncLocal<int> Depth = new();

    internal static bool IsActive => Depth.Value > 0;

    internal static IDisposable Begin(bool enabled)
    {
        if (!enabled)
            return DisabledScope.Instance;
        Depth.Value++;
        return new ActiveScope();
    }

    private sealed class ActiveScope : IDisposable
    {
        private bool _disposed;

        public void Dispose()
        {
            if (_disposed)
                return;
            int depth = Depth.Value;
            if (depth <= 0)
            {
                throw new InvalidOperationException(
                    "BFP8 inference compute scopes must be disposed in the " +
                    "execution context where they are active.");
            }
            Depth.Value = depth - 1;
            _disposed = true;
        }
    }

    private sealed class DisabledScope : IDisposable
    {
        internal static readonly DisabledScope Instance = new();

        public void Dispose()
        {
        }
    }
}

internal readonly record struct CudaBfp8InferenceTelemetrySnapshot(
    long EmbeddingExecutions,
    long EmbeddingWithPositionsExecutions,
    long MixedLinearExecutions,
    long MixedLayerNormExecutions,
    long MixedResidualLayerNormExecutions,
    long KvCachePrefillExecutions,
    long KvCacheIncrementalExecutions)
{
    public static CudaBfp8InferenceTelemetrySnapshot operator -(
        CudaBfp8InferenceTelemetrySnapshot left,
        CudaBfp8InferenceTelemetrySnapshot right)
        => new(
            left.EmbeddingExecutions - right.EmbeddingExecutions,
            left.EmbeddingWithPositionsExecutions -
                right.EmbeddingWithPositionsExecutions,
            left.MixedLinearExecutions - right.MixedLinearExecutions,
            left.MixedLayerNormExecutions - right.MixedLayerNormExecutions,
            left.MixedResidualLayerNormExecutions -
                right.MixedResidualLayerNormExecutions,
            left.KvCachePrefillExecutions - right.KvCachePrefillExecutions,
            left.KvCacheIncrementalExecutions -
                right.KvCacheIncrementalExecutions);
}

/// <summary>
/// Dispatch counters for the no-grad BFP8-storage/BF16-compute generation
/// path. They make it possible to distinguish a real KV-cache execution from
/// the correct but much slower full-window fallback.
/// </summary>
internal static class CudaBfp8InferenceTelemetry
{
    private static long _embeddingExecutions;
    private static long _embeddingWithPositionsExecutions;
    private static long _mixedLinearExecutions;
    private static long _mixedLayerNormExecutions;
    private static long _mixedResidualLayerNormExecutions;
    private static long _kvCachePrefillExecutions;
    private static long _kvCacheIncrementalExecutions;

    internal static CudaBfp8InferenceTelemetrySnapshot Snapshot => new(
        Interlocked.Read(ref _embeddingExecutions),
        Interlocked.Read(ref _embeddingWithPositionsExecutions),
        Interlocked.Read(ref _mixedLinearExecutions),
        Interlocked.Read(ref _mixedLayerNormExecutions),
        Interlocked.Read(ref _mixedResidualLayerNormExecutions),
        Interlocked.Read(ref _kvCachePrefillExecutions),
        Interlocked.Read(ref _kvCacheIncrementalExecutions));

    internal static void RecordEmbedding(bool includesPositions)
    {
        if (includesPositions)
            Interlocked.Increment(ref _embeddingWithPositionsExecutions);
        else
            Interlocked.Increment(ref _embeddingExecutions);
    }

    internal static void RecordMixedLinear()
        => Interlocked.Increment(ref _mixedLinearExecutions);

    internal static void RecordMixedLayerNorm(bool residual)
    {
        if (residual)
            Interlocked.Increment(ref _mixedResidualLayerNormExecutions);
        else
            Interlocked.Increment(ref _mixedLayerNormExecutions);
    }

    internal static void RecordKvCachePrefill()
        => Interlocked.Increment(ref _kvCachePrefillExecutions);

    internal static void RecordKvCacheIncremental()
        => Interlocked.Increment(ref _kvCacheIncrementalExecutions);
}

internal static partial class TensorCudaKernels
{
    internal static BFloat16EmbeddingResidentContext
        EmbeddingForwardBfp8StorageBFloat16Inference(
            Tensor table,
            int[] indices,
            int width)
    {
        RequireNoGradBfp8InferenceOperand(table, nameof(table));
        ArgumentNullException.ThrowIfNull(indices);
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<int>? indicesBuffer = null;
        NativeCudaBuffer<ushort>? output = null;
        try
        {
            indicesBuffer = Tensor.RentCudaIntBuffer(deviceIndex, indices);
            output = Tensor.RentCudaBFloat16Buffer(
                deviceIndex,
                checked(indices.Length * width));
            CudaTensorNative.Embedding(
                deviceIndex,
                table.EnsureCudaBfp8BFloat16Buffer(deviceIndex).NativePtr,
                indicesBuffer.NativePtr,
                output.NativePtr,
                output.Length,
                width,
                bfloat16: true);
            var context = new BFloat16EmbeddingResidentContext(
                output,
                indicesBuffer,
                accelerator);
            output = null;
            indicesBuffer = null;
            CudaBfp8InferenceTelemetry.RecordEmbedding(
                includesPositions: false);
            return context;
        }
        catch (Exception failure)
        {
            ThrowAfterInferenceRollback(
                "BFP8 embedding BF16 inference setup failed.",
                failure,
                () => ReturnBFloat16(accelerator, output),
                () => ReturnInt(accelerator, indicesBuffer));
            throw;
        }
    }

    internal static BFloat16EmbeddingPositionsResidentContext
        EmbeddingWithPositionsForwardBfp8StorageBFloat16Inference(
            Tensor tokenTable,
            Tensor positionTable,
            int[] indices,
            int sequenceLength,
            int width)
    {
        RequireNoGradBfp8InferenceOperand(tokenTable, nameof(tokenTable));
        RequireNoGradBfp8InferenceOperand(
            positionTable,
            nameof(positionTable));
        ArgumentNullException.ThrowIfNull(indices);
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<int>? indicesBuffer = null;
        NativeCudaBuffer<ushort>? output = null;
        try
        {
            indicesBuffer = Tensor.RentCudaIntBuffer(deviceIndex, indices);
            output = Tensor.RentCudaBFloat16Buffer(
                deviceIndex,
                checked(indices.Length * width));
            CudaTensorNative.EmbeddingPositions(
                deviceIndex,
                tokenTable.EnsureCudaBfp8BFloat16Buffer(deviceIndex).NativePtr,
                positionTable.EnsureCudaBfp8BFloat16Buffer(
                    deviceIndex).NativePtr,
                indicesBuffer.NativePtr,
                output.NativePtr,
                output.Length,
                sequenceLength,
                width,
                bfloat16: true);
            var context = new BFloat16EmbeddingPositionsResidentContext(
                output,
                indicesBuffer,
                accelerator);
            output = null;
            indicesBuffer = null;
            CudaBfp8InferenceTelemetry.RecordEmbedding(
                includesPositions: true);
            return context;
        }
        catch (Exception failure)
        {
            ThrowAfterInferenceRollback(
                "BFP8 embedding/position BF16 inference setup failed.",
                failure,
                () => ReturnBFloat16(accelerator, output),
                () => ReturnInt(accelerator, indicesBuffer));
            throw;
        }
    }

    internal static NativeCudaBuffer<ushort>
        LinearForwardBFloat16ActivationBfp8ParametersInference(
            Tensor input,
            Tensor weight,
            Tensor bias,
            int rows,
            int inputWidth,
            int outputWidth,
            bool applyRelu)
    {
        RequireNoGradBFloat16InferenceActivation(input, nameof(input));
        RequireNoGradBfp8InferenceOperand(weight, nameof(weight));
        RequireNoGradBfp8InferenceOperand(bias, nameof(bias));
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        using CudaBfp8BFloat16Lease weightOperand =
            weight.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        using CudaBfp8BFloat16Lease biasOperand =
            bias.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        NativeCudaBuffer<ushort> output = Tensor.RentCudaBFloat16Buffer(
            deviceIndex,
            checked(rows * outputWidth));
        try
        {
            NativeCudaBuffer<ushort> inputBuffer =
                input.EnsureCudaBFloat16Buffer(deviceIndex);
            if (!CudaBlasLt.TryLinearForwardBFloat16(
                    accelerator,
                    deviceIndex,
                    inputBuffer,
                    weightOperand.Buffer,
                    biasOperand.Buffer,
                    output,
                    rows,
                    inputWidth,
                    outputWidth,
                    applyRelu))
            {
                CudaBlas.LinearForwardBFloat16(
                    accelerator,
                    deviceIndex,
                    inputBuffer,
                    weightOperand.Buffer,
                    output,
                    rows,
                    inputWidth,
                    outputWidth);
                CudaTensorNative.LinearBias(
                    deviceIndex,
                    output.NativePtr,
                    biasOperand.Buffer.NativePtr,
                    output.Length,
                    outputWidth,
                    applyRelu,
                    bfloat16: true);
            }
            CudaBfp8InferenceTelemetry.RecordMixedLinear();
            return output;
        }
        catch
        {
            Tensor.ReturnCudaBFloat16Buffer(accelerator, output);
            throw;
        }
    }

    internal static BFloat16LayerNormResidentContext
        LayerNormForwardBFloat16ActivationBfp8ParametersInference(
            Tensor input,
            Tensor gamma,
            Tensor beta,
            int rows,
            int columns,
            float epsilon)
    {
        RequireNoGradBFloat16InferenceActivation(input, nameof(input));
        RequireNoGradBfp8InferenceOperand(gamma, nameof(gamma));
        RequireNoGradBfp8InferenceOperand(beta, nameof(beta));
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        using CudaBfp8BFloat16Lease gammaOperand =
            gamma.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        using CudaBfp8BFloat16Lease betaOperand =
            beta.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        NativeCudaBuffer<ushort>? output = null;
        NativeCudaBuffer<float>? means = null;
        NativeCudaBuffer<float>? inverses = null;
        try
        {
            output = Tensor.RentCudaBFloat16Buffer(deviceIndex, input.Numel);
            means = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
            inverses = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
            if (!CudaLayerNorm.TryForwardBFloat16(
                    accelerator,
                    input.EnsureCudaBFloat16Buffer(deviceIndex),
                    gammaOperand.Buffer,
                    betaOperand.Buffer,
                    output,
                    means,
                    inverses,
                    rows,
                    columns,
                    epsilon))
            {
                throw new PlatformNotSupportedException(
                    "Mixed BFP8/BF16 inference LayerNorm requires the " +
                    "resident BF16 CUDA reduction kernel.");
            }
            var context = new BFloat16LayerNormResidentContext(
                output,
                means,
                inverses,
                accelerator,
                native: true);
            output = null;
            means = null;
            inverses = null;
            CudaBfp8InferenceTelemetry.RecordMixedLayerNorm(
                residual: false);
            return context;
        }
        catch (Exception failure)
        {
            ThrowAfterInferenceRollback(
                "Mixed BFP8/BF16 inference LayerNorm setup failed.",
                failure,
                () => ReturnFloat(accelerator, inverses),
                () => ReturnFloat(accelerator, means),
                () => ReturnBFloat16(accelerator, output));
            throw;
        }
    }

    internal static BFloat16LayerNormResidentContext
        ResidualDropoutLayerNormForwardMixedBFloat16Inference(
            Tensor residual,
            Tensor branch,
            Tensor gamma,
            Tensor beta,
            int rows,
            int columns,
            uint seed,
            uint dropThreshold,
            float dropoutScale,
            float epsilon,
            CudaGraphDropoutToken? graphToken = null)
    {
        RequireNoGradBFloat16InferenceActivation(residual, nameof(residual));
        RequireNoGradBFloat16InferenceActivation(branch, nameof(branch));
        RequireNoGradBfp8InferenceOperand(gamma, nameof(gamma));
        RequireNoGradBfp8InferenceOperand(beta, nameof(beta));
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        using CudaBfp8BFloat16Lease gammaOperand =
            gamma.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        using CudaBfp8BFloat16Lease betaOperand =
            beta.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        NativeCudaBuffer<ushort>? output = null;
        NativeCudaBuffer<float>? means = null;
        NativeCudaBuffer<float>? inverses = null;
        try
        {
            output = Tensor.RentCudaBFloat16Buffer(
                deviceIndex,
                residual.Numel);
            means = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
            inverses = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
            bool succeeded = graphToken is { } token
                ? CudaLayerNorm.TryFusedForwardBFloat16Graph(
                    accelerator,
                    residual.EnsureCudaBFloat16Buffer(deviceIndex),
                    branch.EnsureCudaBFloat16Buffer(deviceIndex),
                    gammaOperand.Buffer,
                    betaOperand.Buffer,
                    output,
                    means,
                    inverses,
                    rows,
                    columns,
                    token,
                    dropThreshold,
                    dropoutScale,
                    epsilon)
                : CudaLayerNorm.TryFusedForwardBFloat16(
                    accelerator,
                    residual.EnsureCudaBFloat16Buffer(deviceIndex),
                    branch.EnsureCudaBFloat16Buffer(deviceIndex),
                    gammaOperand.Buffer,
                    betaOperand.Buffer,
                    output,
                    means,
                    inverses,
                    rows,
                    columns,
                    seed,
                    dropThreshold,
                    dropoutScale,
                    epsilon);
            if (!succeeded)
            {
                throw new PlatformNotSupportedException(
                    "Mixed BFP8/BF16 inference residual LayerNorm requires " +
                    "the resident fused BF16 CUDA kernel.");
            }
            var context = new BFloat16LayerNormResidentContext(
                output,
                means,
                inverses,
                accelerator,
                native: true);
            output = null;
            means = null;
            inverses = null;
            CudaBfp8InferenceTelemetry.RecordMixedLayerNorm(
                residual: true);
            return context;
        }
        catch (Exception failure)
        {
            ThrowAfterInferenceRollback(
                "Mixed BFP8/BF16 residual LayerNorm setup failed.",
                failure,
                () => ReturnFloat(accelerator, inverses),
                () => ReturnFloat(accelerator, means),
                () => ReturnBFloat16(accelerator, output));
            throw;
        }
    }

    private static void RequireNoGradBfp8InferenceOperand(
        Tensor tensor,
        string parameterName)
    {
        ArgumentNullException.ThrowIfNull(tensor, parameterName);
        if (AutogradContext.IsRecordingEnabled
            || !CudaBfp8InferenceComputeScope.IsActive
            || Tensor.ExecutionDevice != TensorDevice.Cuda
            || tensor.DType != TensorDType.Bfp8)
        {
            throw new InvalidOperationException(
                "The BFP8-storage/BF16-compute path is restricted to " +
                "no-grad CUDA inference.");
        }
    }

    private static void RequireNoGradBFloat16InferenceActivation(
        Tensor tensor,
        string parameterName)
    {
        ArgumentNullException.ThrowIfNull(tensor, parameterName);
        if (AutogradContext.IsRecordingEnabled
            || !CudaBfp8InferenceComputeScope.IsActive
            || Tensor.ExecutionDevice != TensorDevice.Cuda
            || tensor.DType != TensorDType.BFloat16)
        {
            throw new InvalidOperationException(
                "Mixed BFP8 parameter execution requires a resident BF16 " +
                "activation during no-grad CUDA inference.");
        }
    }

    private static void ReturnBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort>? buffer)
    {
        if (buffer is not null)
            Tensor.ReturnCudaBFloat16Buffer(accelerator, buffer);
    }

    private static void ReturnFloat(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float>? buffer)
    {
        if (buffer is not null)
            Tensor.ReturnCudaFloatBuffer(accelerator, buffer);
    }

    private static void ReturnInt(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<int>? buffer)
    {
        if (buffer is not null)
            Tensor.ReturnCudaIntBuffer(accelerator, buffer);
    }

    private static void ThrowAfterInferenceRollback(
        string message,
        Exception operationFailure,
        params Action[] cleanup)
    {
        var failures = new List<Exception> { operationFailure };
        foreach (Action release in cleanup)
        {
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
        if (failures.Count == 1)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(operationFailure)
                .Throw();
        }
        throw new AggregateException(message, failures);
    }
}

public partial class Tensor
{
    private Tensor EmbeddingLookupBfp8StorageBFloat16Inference(
        int[] indices,
        int width,
        int[] resultShape)
    {
        TensorCudaKernels.BFloat16EmbeddingResidentContext context =
            TensorCudaKernels
                .EmbeddingForwardBfp8StorageBFloat16Inference(
                    this,
                    indices,
                    width);
        Tensor result;
        try
        {
            result = FromCudaResult(
                context.Output,
                CudaDeviceIndex,
                resultShape,
                [this],
                TensorDType.BFloat16);
        }
        catch (Exception adoptionFailure)
        {
            DisposeUnadoptedInferenceEmbedding(
                context.Output,
                context,
                adoptionFailure);
            throw;
        }
        if (!CudaInferenceScope.TrackResource(context))
            context.Dispose();
        return result;
    }

    private Tensor EmbeddingWithPositionsBfp8StorageBFloat16Inference(
        Tensor positionTable,
        int[] indices,
        int batchSize,
        int sequenceLength,
        int width)
    {
        TensorCudaKernels.BFloat16EmbeddingPositionsResidentContext context =
            TensorCudaKernels
                .EmbeddingWithPositionsForwardBfp8StorageBFloat16Inference(
                    this,
                    positionTable,
                    indices,
                    sequenceLength,
                    width);
        Tensor result;
        try
        {
            result = FromCudaResult(
                context.Output,
                CudaDeviceIndex,
                [batchSize, sequenceLength, width],
                [this, positionTable],
                TensorDType.BFloat16);
        }
        catch (Exception adoptionFailure)
        {
            DisposeUnadoptedInferenceEmbedding(
                context.Output,
                context,
                adoptionFailure);
            throw;
        }
        if (!CudaInferenceScope.TrackResource(context))
            context.Dispose();
        return result;
    }

    private static void DisposeUnadoptedInferenceEmbedding(
        NativeCudaBuffer<ushort> output,
        IDisposable context,
        Exception adoptionFailure)
    {
        List<Exception>? failures = null;
        NativeCudaDevice accelerator = output.Device;
        try
        {
            ReturnCudaBFloat16Buffer(accelerator, output);
        }
        catch (Exception cleanupFailure)
        {
            (failures ??= []).Add(cleanupFailure);
        }
        try
        {
            context.Dispose();
        }
        catch (Exception cleanupFailure)
        {
            (failures ??= []).Add(cleanupFailure);
        }
        if (failures is not null)
        {
            failures.Insert(0, adoptionFailure);
            throw new AggregateException(
                "BF16 inference embedding adoption and cleanup failed.",
                failures);
        }
    }
}
