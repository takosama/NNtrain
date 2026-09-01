using System.Runtime.CompilerServices;
using NNtrain.Cuda.Interop;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain;

internal static class CudaLayerNorm
{
    // Parameter gradients use one atomic-free two-stage partial per row tile.
    private const int ParameterRowsPerTile = 1024;
    private const int ParameterScratchCacheCapacity = 16;
    private const int FallbackResourceCapacity = 4;
    private static int _availability;
    private static readonly ResettableBoundedDisposableLeaseCache<
        StreamKey,
        LaneParameterScratch> FallbackScratch =
            new(FallbackResourceCapacity);
    private static readonly ConditionalWeakTable<
        IStreamExecutionLane,
        Lazy<LaneParameterScratch>> LaneScratch = new();
    private static int _activeLaneScratchResourceCount;

    internal static int ActiveLaneScratchResourceCount =>
        Volatile.Read(ref _activeLaneScratchResourceCount);
    internal static int FallbackScratchResourceCount => FallbackScratch.Count;

    internal static void DisposeFallbackResources()
        => FallbackScratch.Dispose();

    internal static bool TryForward(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> input,
        NativeCudaBuffer<float> gamma,
        NativeCudaBuffer<float> beta,
        NativeCudaBuffer<float> output,
        NativeCudaBuffer<float> normalized,
        NativeCudaBuffer<float> inverses,
        int rows,
        int columns,
        float epsilon)
    {
        if (Volatile.Read(ref _availability) < 0)
            return false;
        try
        {
            accelerator.Bind();
            int status = CudaNativeGateway.LayerNormForward(
                accelerator.Index,
                input.NativePtr, gamma.NativePtr, beta.NativePtr,
                output.NativePtr, normalized.NativePtr, inverses.NativePtr,
                rows, columns, epsilon, Stream(accelerator));
            return Complete(status);
        }
        catch (Exception exception) when (IsUnavailable(exception))
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }

    internal static bool TryForwardBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> input,
        NativeCudaBuffer<ushort> gamma,
        NativeCudaBuffer<ushort> beta,
        NativeCudaBuffer<ushort> output,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        int rows,
        int columns,
        float epsilon)
    {
        if (Volatile.Read(ref _availability) < 0)
            return false;
        try
        {
            accelerator.Bind();
            int status = CudaNativeGateway.LayerNormForwardBFloat16(
                accelerator.Index,
                input.NativePtr, gamma.NativePtr, beta.NativePtr,
                output.NativePtr, means.NativePtr, inverses.NativePtr,
                rows, columns, epsilon, Stream(accelerator));
            return Complete(status);
        }
        catch (Exception exception) when (IsUnavailable(exception))
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }

    internal static bool TryFusedForward(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> residual,
        NativeCudaBuffer<float> branch,
        NativeCudaBuffer<float> gamma,
        NativeCudaBuffer<float> beta,
        NativeCudaBuffer<float> output,
        NativeCudaBuffer<float> normalized,
        NativeCudaBuffer<float> inverses,
        int rows,
        int columns,
        uint seed,
        uint dropThreshold,
        float dropoutScale,
        float epsilon)
    {
        if (Volatile.Read(ref _availability) < 0)
            return false;
        try
        {
            accelerator.Bind();
            int status = CudaNativeGateway.ResidualDropoutLayerNormForward(
                accelerator.Index,
                residual.NativePtr, branch.NativePtr, gamma.NativePtr,
                beta.NativePtr, output.NativePtr, normalized.NativePtr,
                inverses.NativePtr, rows, columns, seed, dropThreshold,
                dropoutScale, epsilon, Stream(accelerator));
            return Complete(status);
        }
        catch (Exception exception) when (IsUnavailable(exception))
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }

    internal static bool TryFusedForwardBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> residual,
        NativeCudaBuffer<ushort> branch,
        NativeCudaBuffer<ushort> gamma,
        NativeCudaBuffer<ushort> beta,
        NativeCudaBuffer<ushort> output,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        int rows,
        int columns,
        uint seed,
        uint dropThreshold,
        float dropoutScale,
        float epsilon)
    {
        if (Volatile.Read(ref _availability) < 0)
            return false;
        try
        {
            accelerator.Bind();
            int status = CudaNativeGateway
                .ResidualDropoutLayerNormForwardBFloat16(
                accelerator.Index,
                residual.NativePtr, branch.NativePtr, gamma.NativePtr,
                beta.NativePtr, output.NativePtr, means.NativePtr,
                inverses.NativePtr, rows, columns, seed, dropThreshold,
                dropoutScale, epsilon, Stream(accelerator));
            return Complete(status);
        }
        catch (Exception exception) when (IsUnavailable(exception))
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }

    internal static void FusedForwardBfp8Block128x512(
        NativeCudaDevice accelerator,
        CudaBfp8BufferView residual,
        CudaBfp8BufferView branch,
        CudaBfp8BufferView gamma,
        CudaBfp8BufferView beta,
        CudaBfp8BufferView output,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        int rows,
        int columns,
        uint seed,
        uint dropThreshold,
        float dropoutScale,
        float epsilon,
        CudaGraphDropoutToken? graphToken = null)
    {
        accelerator.Bind();
        int blockSize = output.Descriptor.GetEffectiveBlockSize(
            output.Payload.Length);
        if (graphToken is { } token)
        {
            token.RngState
                .EnqueueResidualDropoutLayerNormForwardBfp8Block128x512(
                    residual.Payload.NativePtr, residual.Scales.NativePtr,
                    branch.Payload.NativePtr, branch.Scales.NativePtr,
                    gamma.Payload.NativePtr, gamma.Scales.NativePtr,
                    beta.Payload.NativePtr, beta.Scales.NativePtr,
                    output.Payload.NativePtr, output.Scales.NativePtr,
                    means.NativePtr, inverses.NativePtr,
                    rows, columns, blockSize, dropThreshold,
                    dropoutScale, epsilon, token.OperationSeed);
            return;
        }
        ThrowIfFailed(
            CudaNativeGateway
                .ResidualDropoutLayerNormForwardBfp8Block128x512(
                    accelerator.Index,
                    residual.Payload.NativePtr, residual.Scales.NativePtr,
                    branch.Payload.NativePtr, branch.Scales.NativePtr,
                    gamma.Payload.NativePtr, gamma.Scales.NativePtr,
                    beta.Payload.NativePtr, beta.Scales.NativePtr,
                    output.Payload.NativePtr, output.Scales.NativePtr,
                    means.NativePtr, inverses.NativePtr,
                    rows, columns, blockSize, seed, dropThreshold,
                    dropoutScale, epsilon, Stream(accelerator)));
    }

    internal static bool TryFusedForwardGraph(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> residual,
        NativeCudaBuffer<float> branch,
        NativeCudaBuffer<float> gamma,
        NativeCudaBuffer<float> beta,
        NativeCudaBuffer<float> output,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        int rows,
        int columns,
        CudaGraphDropoutToken token,
        uint dropThreshold,
        float dropoutScale,
        float epsilon)
    {
        if (Volatile.Read(ref _availability) < 0)
            return false;
        try
        {
            accelerator.Bind();
            token.RngState.EnqueueResidualDropoutLayerNormForwardFloat32(
                residual.NativePtr,
                branch.NativePtr,
                gamma.NativePtr,
                beta.NativePtr,
                output.NativePtr,
                means.NativePtr,
                inverses.NativePtr,
                rows,
                columns,
                dropThreshold,
                dropoutScale,
                epsilon,
                token.OperationSeed);
            return true;
        }
        catch (Exception exception) when (IsUnavailable(exception))
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }

    internal static bool TryFusedForwardBFloat16Graph(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> residual,
        NativeCudaBuffer<ushort> branch,
        NativeCudaBuffer<ushort> gamma,
        NativeCudaBuffer<ushort> beta,
        NativeCudaBuffer<ushort> output,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        int rows,
        int columns,
        CudaGraphDropoutToken token,
        uint dropThreshold,
        float dropoutScale,
        float epsilon)
    {
        if (Volatile.Read(ref _availability) < 0)
            return false;
        try
        {
            accelerator.Bind();
            token.RngState.EnqueueResidualDropoutLayerNormForwardBFloat16(
                residual.NativePtr,
                branch.NativePtr,
                gamma.NativePtr,
                beta.NativePtr,
                output.NativePtr,
                means.NativePtr,
                inverses.NativePtr,
                rows,
                columns,
                dropThreshold,
                dropoutScale,
                epsilon,
                token.OperationSeed);
            return true;
        }
        catch (Exception exception) when (IsUnavailable(exception))
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }

    internal static void Backward(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> input,
        NativeCudaBuffer<float> gamma,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> inputGradient,
        NativeCudaBuffer<float> gammaGradient,
        NativeCudaBuffer<float> betaGradient,
        int rows,
        int columns)
    {
        accelerator.Bind();
        using ParameterScratchLease parameterScratch = GetParameterScratch(
            accelerator, rows, columns);
        ThrowIfFailed(CudaNativeGateway.LayerNormBackward(
            accelerator.Index,
            input.NativePtr, gamma.NativePtr, means.NativePtr,
            inverses.NativePtr,
            outputGradient.NativePtr, inputGradient.NativePtr,
            gammaGradient.NativePtr, betaGradient.NativePtr,
            parameterScratch.Buffer.NativePtr,
            rows, columns, Stream(accelerator)));
    }

    internal static void BackwardBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> input,
        NativeCudaBuffer<ushort> gamma,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> inputGradient,
        NativeCudaBuffer<float> gammaGradient,
        NativeCudaBuffer<float> betaGradient,
        int rows,
        int columns)
    {
        accelerator.Bind();
        using ParameterScratchLease parameterScratch = GetParameterScratch(
            accelerator, rows, columns);
        ThrowIfFailed(CudaNativeGateway.LayerNormBackwardBFloat16(
            accelerator.Index,
            input.NativePtr, gamma.NativePtr, means.NativePtr,
            inverses.NativePtr,
            outputGradient.NativePtr, inputGradient.NativePtr,
            gammaGradient.NativePtr, betaGradient.NativePtr,
            parameterScratch.Buffer.NativePtr,
            rows, columns, Stream(accelerator)));
    }

    internal static void FusedBackward(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> residual,
        NativeCudaBuffer<float> branch,
        NativeCudaBuffer<float> gamma,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> residualGradient,
        NativeCudaBuffer<float> branchGradient,
        NativeCudaBuffer<float> gammaGradient,
        NativeCudaBuffer<float> betaGradient,
        int rows,
        int columns,
        bool sameParent,
        uint seed,
        uint dropThreshold,
        float dropoutScale)
    {
        accelerator.Bind();
        using ParameterScratchLease parameterScratch = GetParameterScratch(
            accelerator, rows, columns);
        ThrowIfFailed(CudaNativeGateway.ResidualDropoutLayerNormBackward(
            accelerator.Index,
            residual.NativePtr, branch.NativePtr, gamma.NativePtr,
            means.NativePtr, inverses.NativePtr,
            outputGradient.NativePtr, residualGradient.NativePtr,
            branchGradient.NativePtr, gammaGradient.NativePtr,
            betaGradient.NativePtr, parameterScratch.Buffer.NativePtr,
            rows, columns, sameParent ? 1 : 0,
            seed, dropThreshold, dropoutScale, Stream(accelerator)));
    }

    internal static void FusedBackwardBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> residual,
        NativeCudaBuffer<ushort> branch,
        NativeCudaBuffer<ushort> gamma,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> residualGradient,
        NativeCudaBuffer<float> branchGradient,
        NativeCudaBuffer<float> gammaGradient,
        NativeCudaBuffer<float> betaGradient,
        int rows,
        int columns,
        bool sameParent,
        uint seed,
        uint dropThreshold,
        float dropoutScale)
    {
        accelerator.Bind();
        using ParameterScratchLease parameterScratch = GetParameterScratch(
            accelerator, rows, columns);
        bool oneScan = columns == 512
            && CudaDispatchPolicy.Current.EnableLayerNormOneScan512;
        ThrowIfFailed(oneScan
            ? CudaNativeGateway
                .ResidualDropoutLayerNormBackwardBFloat16OneScan512(
                    accelerator.Index,
                    residual.NativePtr, branch.NativePtr, gamma.NativePtr,
                    means.NativePtr, inverses.NativePtr,
                    outputGradient.NativePtr, residualGradient.NativePtr,
                    branchGradient.NativePtr, gammaGradient.NativePtr,
                    betaGradient.NativePtr, parameterScratch.Buffer.NativePtr,
                    rows, columns, sameParent ? 1 : 0,
                    seed, dropThreshold, dropoutScale, Stream(accelerator))
            : CudaNativeGateway
                .ResidualDropoutLayerNormBackwardBFloat16(
                    accelerator.Index,
                    residual.NativePtr, branch.NativePtr, gamma.NativePtr,
                    means.NativePtr, inverses.NativePtr,
                    outputGradient.NativePtr, residualGradient.NativePtr,
                    branchGradient.NativePtr, gammaGradient.NativePtr,
                    betaGradient.NativePtr, parameterScratch.Buffer.NativePtr,
                    rows, columns, sameParent ? 1 : 0,
                    seed, dropThreshold, dropoutScale, Stream(accelerator)));
    }

    internal static void FusedBackwardBfp8Block128x512(
        NativeCudaDevice accelerator,
        CudaBfp8BufferView residual,
        CudaBfp8BufferView branch,
        CudaBfp8BufferView gamma,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> residualGradient,
        NativeCudaBuffer<float> branchGradient,
        NativeCudaBuffer<float> gammaGradient,
        NativeCudaBuffer<float> betaGradient,
        int rows,
        int columns,
        bool sameParent,
        uint seed,
        uint dropThreshold,
        float dropoutScale,
        CudaGraphDropoutToken? graphToken = null)
    {
        accelerator.Bind();
        using ParameterScratchLease parameterScratch = GetParameterScratch(
            accelerator, rows, columns);
        int blockSize = residual.Descriptor.GetEffectiveBlockSize(
            residual.Payload.Length);
        if (graphToken is { } token)
        {
            token.RngState
                .EnqueueResidualDropoutLayerNormBackwardBfp8Block128x512(
                    residual.Payload.NativePtr, residual.Scales.NativePtr,
                    branch.Payload.NativePtr, branch.Scales.NativePtr,
                    gamma.Payload.NativePtr, gamma.Scales.NativePtr,
                    means.NativePtr, inverses.NativePtr,
                    outputGradient.NativePtr,
                    residualGradient.NativePtr, branchGradient.NativePtr,
                    gammaGradient.NativePtr, betaGradient.NativePtr,
                    parameterScratch.Buffer.NativePtr,
                    rows, columns, blockSize, sameParent, dropThreshold,
                    dropoutScale, token.OperationSeed);
            return;
        }
        ThrowIfFailed(
            CudaNativeGateway
                .ResidualDropoutLayerNormBackwardBfp8Block128x512(
                    accelerator.Index,
                    residual.Payload.NativePtr, residual.Scales.NativePtr,
                    branch.Payload.NativePtr, branch.Scales.NativePtr,
                    gamma.Payload.NativePtr, gamma.Scales.NativePtr,
                    means.NativePtr, inverses.NativePtr,
                    outputGradient.NativePtr,
                    residualGradient.NativePtr, branchGradient.NativePtr,
                    gammaGradient.NativePtr, betaGradient.NativePtr,
                    parameterScratch.Buffer.NativePtr,
                    rows, columns, blockSize, sameParent ? 1 : 0,
                    seed, dropThreshold, dropoutScale,
                    Stream(accelerator)));
    }

    internal static void FusedBackwardGraph(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> residual,
        NativeCudaBuffer<float> branch,
        NativeCudaBuffer<float> gamma,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> residualGradient,
        NativeCudaBuffer<float> branchGradient,
        NativeCudaBuffer<float> gammaGradient,
        NativeCudaBuffer<float> betaGradient,
        int rows,
        int columns,
        bool sameParent,
        CudaGraphDropoutToken token,
        uint dropThreshold,
        float dropoutScale)
    {
        accelerator.Bind();
        using ParameterScratchLease parameterScratch = GetParameterScratch(
            accelerator, rows, columns);
        token.RngState.EnqueueResidualDropoutLayerNormBackwardFloat32(
            residual.NativePtr,
            branch.NativePtr,
            gamma.NativePtr,
            means.NativePtr,
            inverses.NativePtr,
            outputGradient.NativePtr,
            residualGradient.NativePtr,
            branchGradient.NativePtr,
            gammaGradient.NativePtr,
            betaGradient.NativePtr,
            parameterScratch.Buffer.NativePtr,
            rows,
            columns,
            sameParent,
            dropThreshold,
            dropoutScale,
            token.OperationSeed);
    }

    internal static void FusedBackwardBFloat16Graph(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> residual,
        NativeCudaBuffer<ushort> branch,
        NativeCudaBuffer<ushort> gamma,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> residualGradient,
        NativeCudaBuffer<float> branchGradient,
        NativeCudaBuffer<float> gammaGradient,
        NativeCudaBuffer<float> betaGradient,
        int rows,
        int columns,
        bool sameParent,
        CudaGraphDropoutToken token,
        uint dropThreshold,
        float dropoutScale)
    {
        accelerator.Bind();
        using ParameterScratchLease parameterScratch = GetParameterScratch(
            accelerator, rows, columns);
        token.RngState.EnqueueResidualDropoutLayerNormBackwardBFloat16(
            residual.NativePtr,
            branch.NativePtr,
            gamma.NativePtr,
            means.NativePtr,
            inverses.NativePtr,
            outputGradient.NativePtr,
            residualGradient.NativePtr,
            branchGradient.NativePtr,
            gammaGradient.NativePtr,
            betaGradient.NativePtr,
            parameterScratch.Buffer.NativePtr,
            rows,
            columns,
            sameParent,
            dropThreshold,
            dropoutScale,
            token.OperationSeed,
            columns == 512
                && CudaDispatchPolicy.Current.EnableLayerNormOneScan512);
    }

    internal static void FusedBackwardBFloat16DirectBranch(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> residual,
        NativeCudaBuffer<ushort> branch,
        NativeCudaBuffer<ushort> gamma,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        NativeCudaBuffer<float>? outputGradient,
        NativeCudaBuffer<ushort>? outputGradientBFloat16,
        NativeCudaBuffer<float> residualGradient,
        NativeCudaBuffer<ushort> branchGradientBFloat16,
        NativeCudaBuffer<float> gammaGradient,
        NativeCudaBuffer<float> betaGradient,
        int rows,
        int columns,
        uint seed,
        uint dropThreshold,
        float dropoutScale)
    {
        accelerator.Bind();
        using ParameterScratchLease parameterScratch = GetParameterScratch(
            accelerator, rows, columns);
        int status;
        if (outputGradientBFloat16 is not null)
        {
            status = CudaNativeGateway
                .ResidualDropoutLayerNormBackwardBFloat16IoGradient(
                accelerator.Index,
                residual.NativePtr, branch.NativePtr, gamma.NativePtr,
                means.NativePtr, inverses.NativePtr,
                outputGradientBFloat16.NativePtr, residualGradient.NativePtr,
                branchGradientBFloat16.NativePtr, gammaGradient.NativePtr,
                betaGradient.NativePtr, parameterScratch.Buffer.NativePtr,
                rows, columns, seed, dropThreshold, dropoutScale,
                Stream(accelerator));
        }
        else
        {
            if (outputGradient is null)
            {
                throw new ArgumentNullException(
                    nameof(outputGradient),
                    "LayerNorm backward requires an output gradient.");
            }
            status = CudaNativeGateway
                .ResidualDropoutLayerNormBackwardBFloat16BranchGradient(
                accelerator.Index,
                residual.NativePtr, branch.NativePtr, gamma.NativePtr,
                means.NativePtr, inverses.NativePtr,
                outputGradient.NativePtr, residualGradient.NativePtr,
                branchGradientBFloat16.NativePtr, gammaGradient.NativePtr,
                betaGradient.NativePtr, parameterScratch.Buffer.NativePtr,
                rows, columns, seed, dropThreshold, dropoutScale,
                Stream(accelerator));
        }
        ThrowIfFailed(status);
    }

    internal static void FusedBackwardBFloat16DirectBranchGraph(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> residual,
        NativeCudaBuffer<ushort> branch,
        NativeCudaBuffer<ushort> gamma,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        NativeCudaBuffer<float>? outputGradient,
        NativeCudaBuffer<ushort>? outputGradientBFloat16,
        NativeCudaBuffer<float> residualGradient,
        NativeCudaBuffer<ushort> branchGradientBFloat16,
        NativeCudaBuffer<float> gammaGradient,
        NativeCudaBuffer<float> betaGradient,
        int rows,
        int columns,
        CudaGraphDropoutToken token,
        uint dropThreshold,
        float dropoutScale)
    {
        accelerator.Bind();
        using ParameterScratchLease parameterScratch = GetParameterScratch(
            accelerator, rows, columns);
        if (outputGradientBFloat16 is not null)
        {
            token.RngState
                .EnqueueResidualDropoutLayerNormBackwardBFloat16IoGradient(
                    residual.NativePtr,
                    branch.NativePtr,
                    gamma.NativePtr,
                    means.NativePtr,
                    inverses.NativePtr,
                    outputGradientBFloat16.NativePtr,
                    residualGradient.NativePtr,
                    branchGradientBFloat16.NativePtr,
                    gammaGradient.NativePtr,
                    betaGradient.NativePtr,
                    parameterScratch.Buffer.NativePtr,
                    rows,
                    columns,
                    dropThreshold,
                    dropoutScale,
                    token.OperationSeed);
            return;
        }
        if (outputGradient is null)
        {
            throw new ArgumentNullException(
                nameof(outputGradient),
                "LayerNorm backward requires an output gradient.");
        }
        token.RngState
            .EnqueueResidualDropoutLayerNormBackwardBFloat16BranchGradient(
                residual.NativePtr,
                branch.NativePtr,
                gamma.NativePtr,
                means.NativePtr,
                inverses.NativePtr,
                outputGradient.NativePtr,
                residualGradient.NativePtr,
                branchGradientBFloat16.NativePtr,
                gammaGradient.NativePtr,
                betaGradient.NativePtr,
                parameterScratch.Buffer.NativePtr,
                rows,
                columns,
                dropThreshold,
                dropoutScale,
                token.OperationSeed);
    }

    private static nint Stream(NativeCudaDevice accelerator) =>
        accelerator.DefaultStream;

    private static ParameterScratchLease GetParameterScratch(
        NativeCudaDevice accelerator,
        int rows,
        int columns)
    {
        int rowTiles = checked(
            (rows + ParameterRowsPerTile - 1) / ParameterRowsPerTile);
        int length = checked(2 * rowTiles * columns);
        nint stream = accelerator.DefaultStream;
        LaneParameterScratch? resources = null;
        BoundedDisposableLeaseCache<
            StreamKey,
            LaneParameterScratch>.Lease? fallbackLease = null;
        if (TensorExecutionContext.TryGetCudaStreamLane(
                accelerator.Index,
                out IStreamExecutionLane lane)
            && lane.ComputeStreamHandle == stream)
        {
            resources = LaneScratch.GetValue(
                    lane,
                    static owner => new Lazy<LaneParameterScratch>(
                        () => ExecutionLaneResources.Attach(
                            owner,
                            new LaneParameterScratch(
                                owner.DeviceIndex,
                                owner.ComputeStreamHandle,
                                laneOwned: true)),
                        LazyThreadSafetyMode.ExecutionAndPublication))
                .Value;
        }
        else
        {
            fallbackLease = FallbackScratch.Acquire(
                new StreamKey(accelerator.Index, stream),
                static key => new LaneParameterScratch(
                    key.DeviceIndex,
                    key.ComputeStream,
                    laneOwned: false));
            resources = fallbackLease?.Value;
        }

        if (resources is null)
        {
            fallbackLease?.Dispose();
            throw new InvalidOperationException(
                "LayerNorm CUDA scratch resources could not be created.");
        }
        try
        {
            BoundedDisposableLeaseCache<int, ScratchBuffer>.Lease? scratch =
                resources.Acquire(length);
            if (scratch is null)
            {
                throw new InvalidOperationException(
                    "LayerNorm CUDA parameter scratch could not be allocated.");
            }
            return new ParameterScratchLease(scratch, fallbackLease);
        }
        catch
        {
            fallbackLease?.Dispose();
            throw;
        }
    }

    private readonly record struct StreamKey(
        int DeviceIndex,
        nint ComputeStream);

    private sealed class LaneParameterScratch : IDisposable
    {
        private readonly BoundedDisposableLeaseCache<int, ScratchBuffer>
            _buffers = new(ParameterScratchCacheCapacity);
        private readonly bool _laneOwned;
        private int _disposed;

        internal LaneParameterScratch(
            int deviceIndex,
            nint computeStream,
            bool laneOwned)
        {
            DeviceIndex = deviceIndex;
            ComputeStream = computeStream;
            _laneOwned = laneOwned;
            ScratchFactory = CreateScratch;
            if (laneOwned)
                Interlocked.Increment(ref _activeLaneScratchResourceCount);
        }

        internal int DeviceIndex { get; }
        internal nint ComputeStream { get; }
        internal bool IsDisposing => Volatile.Read(ref _disposed) != 0;
        private Func<int, ScratchBuffer?> ScratchFactory { get; }

        internal BoundedDisposableLeaseCache<int, ScratchBuffer>.Lease?
            Acquire(int length)
        {
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _disposed) != 0,
                this);
            return _buffers.Acquire(length, ScratchFactory);
        }

        private ScratchBuffer CreateScratch(int length)
            => new(this, length);

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            if (!_laneOwned)
            {
                NativeCudaRuntime.DisposeAfterStreamFence(
                    DeviceIndex,
                    ComputeStream,
                    _buffers.Dispose);
                return;
            }
            try
            {
                _buffers.Dispose();
            }
            finally
            {
                Interlocked.Decrement(
                    ref _activeLaneScratchResourceCount);
            }
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
    }

    private sealed class ScratchBuffer(
        LaneParameterScratch owner,
        int length) : IDisposable
    {
        private NativeCudaBuffer<float>? _buffer =
            ForgetMemoryV2Cuda.GetAccelerator(owner.DeviceIndex)
                .Allocate1D<float>(length, CudaMemoryKind.Workspace);

        internal NativeCudaBuffer<float> Buffer => Volatile.Read(ref _buffer)
            ?? throw new ObjectDisposedException(this.GetType().Name);

        public void Dispose()
        {
            NativeCudaBuffer<float>? buffer = Interlocked.Exchange(
                ref _buffer,
                null);
            if (buffer is not null)
                owner.DisposeChildAfterFence(buffer.Dispose);
        }
    }

    private sealed class ParameterScratchLease : IDisposable
    {
        private BoundedDisposableLeaseCache<int, ScratchBuffer>.Lease?
            _scratchLease;
        private BoundedDisposableLeaseCache<
            StreamKey,
            LaneParameterScratch>.Lease? _fallbackLease;

        internal ParameterScratchLease(
            BoundedDisposableLeaseCache<int, ScratchBuffer>.Lease scratchLease,
            BoundedDisposableLeaseCache<
                StreamKey,
                LaneParameterScratch>.Lease? fallbackLease)
        {
            _scratchLease = scratchLease;
            _fallbackLease = fallbackLease;
            Buffer = scratchLease.Value.Buffer;
        }

        internal NativeCudaBuffer<float> Buffer { get; }

        public void Dispose()
        {
            BoundedDisposableLeaseCache<int, ScratchBuffer>.Lease? scratch =
                Interlocked.Exchange(ref _scratchLease, null);
            BoundedDisposableLeaseCache<
                StreamKey,
                LaneParameterScratch>.Lease? fallback =
                    Interlocked.Exchange(ref _fallbackLease, null);
            try
            {
                scratch?.Dispose();
            }
            finally
            {
                fallback?.Dispose();
            }
        }
    }

    private static bool Complete(int status)
    {
        ThrowIfFailed(status);
        Volatile.Write(ref _availability, 1);
        return true;
    }

    private static void ThrowIfFailed(int status)
    {
        if (status != 0)
            throw new InvalidOperationException($"LayerNorm CUDA error {status}.");
    }

    private static bool IsUnavailable(Exception exception) =>
        exception is DllNotFoundException
        or EntryPointNotFoundException
        or BadImageFormatException;
}
