using NNtrain.Cuda.Interop;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain.Cuda.Execution;

/// <summary>
/// Persistent device-side replay counter for capture-safe dropout RNG. Each
/// captured invocation advances the counter on-device before producing a mask.
/// </summary>
public sealed class CudaGraphRngState : IDisposable
{
    private readonly CudaExecutionLane _lane;
    private readonly CudaMemoryLease _counter;
    private int _disposed;

    private CudaGraphRngState(
        CudaExecutionLane lane,
        CudaMemoryLease counter)
    {
        _lane = lane;
        _counter = counter;
    }

    public int DeviceIndex => _lane.DeviceIndex;

    public static CudaGraphRngState Create(
        CudaExecutionLane lane,
        ulong initialCounter = 0)
    {
        ArgumentNullException.ThrowIfNull(lane);
        if (!lane.CudaCapabilities.Supports(CudaKernelFeature.CudaGraphs))
        {
            throw new NotSupportedException(
                $"CUDA Graph RNG is not available on device " +
                $"{lane.DeviceIndex}.");
        }

        CudaMemoryLease counter = lane.Memory.Allocate(
            sizeof(ulong),
            CudaMemoryKind.Persistent);
        var state = new CudaGraphRngState(lane, counter);
        try
        {
            lane.ActivateComputeStream();
            state.SetCounter(initialCounter);
            return ExecutionLaneResources.Attach(lane, state);
        }
        catch
        {
            state.Dispose();
            throw;
        }
    }

    public void SetCounter(ulong value)
    {
        EnsureActive();
        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway.GraphCounterSet(
                DeviceIndex,
                _counter.Pointer,
                value,
                _lane.ComputeStreamHandle),
            "set CUDA Graph RNG counter",
            DeviceIndex);
    }

    /// <summary>Enqueues exactly one step advance, normally at graph head.</summary>
    public void EnqueueAdvance()
    {
        EnsureActive();
        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway.GraphCounterAdvance(
                DeviceIndex,
                _counter.Pointer,
                _lane.ComputeStreamHandle),
            "advance CUDA Graph RNG counter",
            DeviceIndex);
    }

    public void EnqueueDropoutForwardFloat32(
        nint input,
        nint output,
        int length,
        float dropoutProbability,
        ulong operationSeed)
    {
        ValidateDropoutCall(input, output, length, dropoutProbability);
        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway.GraphDropoutForwardFloat32(
                DeviceIndex,
                _counter.Pointer,
                operationSeed,
                dropoutProbability,
                input,
                output,
                length,
                _lane.ComputeStreamHandle),
            "CUDA Graph FP32 dropout forward",
            DeviceIndex);
    }

    public void EnqueueDropoutForwardBFloat16(
        nint input,
        nint output,
        int length,
        float dropoutProbability,
        ulong operationSeed)
    {
        ValidateDropoutCall(input, output, length, dropoutProbability);
        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway.GraphDropoutForwardBFloat16(
                DeviceIndex,
                _counter.Pointer,
                operationSeed,
                dropoutProbability,
                input,
                output,
                length,
                _lane.ComputeStreamHandle),
            "CUDA Graph BF16 dropout forward",
            DeviceIndex);
    }

    public void EnqueueAddDropoutForwardFloat32(
        nint residual,
        nint branch,
        nint output,
        int length,
        float dropoutProbability,
        ulong operationSeed)
    {
        ValidateDropoutCall(residual, branch, length, dropoutProbability);
        if (output == nint.Zero)
            throw new ArgumentException("Output must be a device pointer.", nameof(output));
        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway.GraphAddDropoutForwardFloat32(
                DeviceIndex,
                _counter.Pointer,
                operationSeed,
                dropoutProbability,
                residual,
                branch,
                output,
                length,
                _lane.ComputeStreamHandle),
            "CUDA Graph FP32 add-dropout forward",
            DeviceIndex);
    }

    public void EnqueueAddDropoutForwardBFloat16(
        nint residual,
        nint branch,
        nint output,
        int length,
        float dropoutProbability,
        ulong operationSeed)
    {
        ValidateDropoutCall(residual, branch, length, dropoutProbability);
        if (output == nint.Zero)
            throw new ArgumentException("Output must be a device pointer.", nameof(output));
        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway.GraphAddDropoutForwardBFloat16(
                DeviceIndex,
                _counter.Pointer,
                operationSeed,
                dropoutProbability,
                residual,
                branch,
                output,
                length,
                _lane.ComputeStreamHandle),
            "CUDA Graph BF16 add-dropout forward",
            DeviceIndex);
    }

    public void EnqueueDropoutBackwardFloat32(
        nint outputGradient,
        nint inputGradient,
        int length,
        float dropoutProbability,
        ulong operationSeed)
    {
        ValidateDropoutCall(
            outputGradient, inputGradient, length, dropoutProbability);
        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway.GraphDropoutBackwardFloat32(
                DeviceIndex,
                _counter.Pointer,
                operationSeed,
                dropoutProbability,
                outputGradient,
                inputGradient,
                length,
                _lane.ComputeStreamHandle),
            "CUDA Graph FP32 dropout backward",
            DeviceIndex);
    }

    public void EnqueueDropoutBackwardBFloat16Gradient(
        nint outputGradient,
        nint inputGradient,
        int length,
        float dropoutProbability,
        ulong operationSeed)
    {
        ValidateDropoutCall(
            outputGradient, inputGradient, length, dropoutProbability);
        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway.GraphDropoutBackwardBFloat16Gradient(
                DeviceIndex,
                _counter.Pointer,
                operationSeed,
                dropoutProbability,
                outputGradient,
                inputGradient,
                length,
                _lane.ComputeStreamHandle),
            "CUDA Graph pure-BF16 dropout backward",
            DeviceIndex);
    }

    public void EnqueueAddDropoutBackwardFloat32(
        nint outputGradient,
        nint residualGradient,
        nint branchGradient,
        int length,
        float dropoutProbability,
        ulong operationSeed,
        bool sameParent)
    {
        ValidateDropoutCall(
            outputGradient, residualGradient, length, dropoutProbability);
        if (branchGradient == nint.Zero)
        {
            throw new ArgumentException(
                "Branch gradient must be a device pointer.",
                nameof(branchGradient));
        }
        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway.GraphAddDropoutBackwardFloat32(
                DeviceIndex,
                _counter.Pointer,
                operationSeed,
                dropoutProbability,
                outputGradient,
                residualGradient,
                branchGradient,
                length,
                sameParent,
                _lane.ComputeStreamHandle),
            "CUDA Graph FP32 add-dropout backward",
            DeviceIndex);
    }

    public void EnqueueAddDropoutBackwardBFloat16Gradient(
        nint outputGradient,
        nint residualGradient,
        nint branchGradient,
        int length,
        float dropoutProbability,
        ulong operationSeed,
        bool sameParent)
    {
        ValidateDropoutCall(
            outputGradient, residualGradient, length, dropoutProbability);
        if (branchGradient == nint.Zero)
        {
            throw new ArgumentException(
                "Branch gradient must be a device pointer.",
                nameof(branchGradient));
        }
        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway.GraphAddDropoutBackwardBFloat16Gradient(
                DeviceIndex,
                _counter.Pointer,
                operationSeed,
                dropoutProbability,
                outputGradient,
                residualGradient,
                branchGradient,
                length,
                sameParent,
                _lane.ComputeStreamHandle),
            "CUDA Graph pure-BF16 add-dropout backward",
            DeviceIndex);
    }

    public void EnqueueDropoutMask(
        nint output,
        int length,
        float dropoutProbability,
        uint seed)
    {
        EnsureActive();
        if (output == nint.Zero)
            throw new ArgumentException("Output must be a device pointer.", nameof(output));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        if (!(dropoutProbability >= 0f) || dropoutProbability >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(dropoutProbability));
        }

        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway.GraphDropoutMask(
                DeviceIndex,
                _counter.Pointer,
                seed,
                dropoutProbability,
                output,
                length,
                _lane.ComputeStreamHandle),
            "capture-safe CUDA dropout RNG",
            DeviceIndex);
    }

    public void EnqueueResidualDropoutLayerNormForwardFloat32(
        nint residual,
        nint branch,
        nint gamma,
        nint beta,
        nint output,
        nint means,
        nint inverses,
        int rows,
        int columns,
        uint dropThreshold,
        float dropoutScale,
        float epsilon,
        ulong operationSeed)
    {
        ValidateFusedLayerNormForward(
            residual, branch, gamma, beta, output, means, inverses,
            rows, columns, dropoutScale, epsilon);
        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway.GraphResidualDropoutLayerNormForward(
                DeviceIndex,
                residual,
                branch,
                gamma,
                beta,
                output,
                means,
                inverses,
                rows,
                columns,
                _counter.Pointer,
                operationSeed,
                dropThreshold,
                dropoutScale,
                epsilon,
                _lane.ComputeStreamHandle),
            "CUDA Graph fused FP32 residual/dropout/LayerNorm forward",
            DeviceIndex);
    }

    public void EnqueueResidualDropoutLayerNormForwardBFloat16(
        nint residual,
        nint branch,
        nint gamma,
        nint beta,
        nint output,
        nint means,
        nint inverses,
        int rows,
        int columns,
        uint dropThreshold,
        float dropoutScale,
        float epsilon,
        ulong operationSeed)
    {
        ValidateFusedLayerNormForward(
            residual, branch, gamma, beta, output, means, inverses,
            rows, columns, dropoutScale, epsilon);
        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway
                .GraphResidualDropoutLayerNormForwardBFloat16(
                    DeviceIndex,
                    residual,
                    branch,
                    gamma,
                    beta,
                    output,
                    means,
                    inverses,
                    rows,
                    columns,
                    _counter.Pointer,
                    operationSeed,
                    dropThreshold,
                    dropoutScale,
                    epsilon,
                    _lane.ComputeStreamHandle),
            "CUDA Graph fused BF16 residual/dropout/LayerNorm forward",
            DeviceIndex);
    }

    public void EnqueueResidualDropoutLayerNormBackwardFloat32(
        nint residual,
        nint branch,
        nint gamma,
        nint means,
        nint inverses,
        nint outputGradient,
        nint residualGradient,
        nint branchGradient,
        nint gammaGradient,
        nint betaGradient,
        nint parameterScratch,
        int rows,
        int columns,
        bool sameParent,
        uint dropThreshold,
        float dropoutScale,
        ulong operationSeed)
    {
        ValidateFusedLayerNormBackward(
            residual, branch, gamma, means, inverses, outputGradient,
            residualGradient, branchGradient, gammaGradient, betaGradient,
            parameterScratch, rows, columns, dropoutScale);
        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway.GraphResidualDropoutLayerNormBackward(
                DeviceIndex,
                residual,
                branch,
                gamma,
                means,
                inverses,
                outputGradient,
                residualGradient,
                branchGradient,
                gammaGradient,
                betaGradient,
                parameterScratch,
                rows,
                columns,
                sameParent ? 1 : 0,
                _counter.Pointer,
                operationSeed,
                dropThreshold,
                dropoutScale,
                _lane.ComputeStreamHandle),
            "CUDA Graph fused FP32 residual/dropout/LayerNorm backward",
            DeviceIndex);
    }

    public void EnqueueResidualDropoutLayerNormBackwardBFloat16(
        nint residual,
        nint branch,
        nint gamma,
        nint means,
        nint inverses,
        nint outputGradient,
        nint residualGradient,
        nint branchGradient,
        nint gammaGradient,
        nint betaGradient,
        nint parameterScratch,
        int rows,
        int columns,
        bool sameParent,
        uint dropThreshold,
        float dropoutScale,
        ulong operationSeed)
    {
        ValidateFusedLayerNormBackward(
            residual, branch, gamma, means, inverses, outputGradient,
            residualGradient, branchGradient, gammaGradient, betaGradient,
            parameterScratch, rows, columns, dropoutScale);
        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway.GraphResidualDropoutLayerNormBackwardBFloat16(
                DeviceIndex,
                residual,
                branch,
                gamma,
                means,
                inverses,
                outputGradient,
                residualGradient,
                branchGradient,
                gammaGradient,
                betaGradient,
                parameterScratch,
                rows,
                columns,
                sameParent ? 1 : 0,
                _counter.Pointer,
                operationSeed,
                dropThreshold,
                dropoutScale,
                _lane.ComputeStreamHandle),
            "CUDA Graph fused BF16 residual/dropout/LayerNorm backward",
            DeviceIndex);
    }

    public void
        EnqueueResidualDropoutLayerNormBackwardBFloat16BranchGradient(
            nint residual,
            nint branch,
            nint gamma,
            nint means,
            nint inverses,
            nint outputGradient,
            nint residualGradient,
            nint branchGradient,
            nint gammaGradient,
            nint betaGradient,
            nint parameterScratch,
            int rows,
            int columns,
            uint dropThreshold,
            float dropoutScale,
            ulong operationSeed)
    {
        ValidateFusedLayerNormBackward(
            residual, branch, gamma, means, inverses, outputGradient,
            residualGradient, branchGradient, gammaGradient, betaGradient,
            parameterScratch, rows, columns, dropoutScale);
        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway
                .GraphResidualDropoutLayerNormBackwardBFloat16BranchGradient(
                    DeviceIndex,
                    residual,
                    branch,
                    gamma,
                    means,
                    inverses,
                    outputGradient,
                    residualGradient,
                    branchGradient,
                    gammaGradient,
                    betaGradient,
                    parameterScratch,
                    rows,
                    columns,
                    _counter.Pointer,
                    operationSeed,
                    dropThreshold,
                    dropoutScale,
                    _lane.ComputeStreamHandle),
            "CUDA Graph fused BF16 branch-gradient LayerNorm backward",
            DeviceIndex);
    }

    public void EnqueueResidualDropoutLayerNormBackwardBFloat16IoGradient(
        nint residual,
        nint branch,
        nint gamma,
        nint means,
        nint inverses,
        nint outputGradient,
        nint residualGradient,
        nint branchGradient,
        nint gammaGradient,
        nint betaGradient,
        nint parameterScratch,
        int rows,
        int columns,
        uint dropThreshold,
        float dropoutScale,
        ulong operationSeed)
    {
        ValidateFusedLayerNormBackward(
            residual, branch, gamma, means, inverses, outputGradient,
            residualGradient, branchGradient, gammaGradient, betaGradient,
            parameterScratch, rows, columns, dropoutScale);
        _lane.ActivateComputeStream();
        CudaGraphStatus.Check(
            CudaNativeGateway
                .GraphResidualDropoutLayerNormBackwardBFloat16IoGradient(
                    DeviceIndex,
                    residual,
                    branch,
                    gamma,
                    means,
                    inverses,
                    outputGradient,
                    residualGradient,
                    branchGradient,
                    gammaGradient,
                    betaGradient,
                    parameterScratch,
                    rows,
                    columns,
                    _counter.Pointer,
                    operationSeed,
                    dropThreshold,
                    dropoutScale,
                    _lane.ComputeStreamHandle),
            "CUDA Graph fused BF16 IO-gradient LayerNorm backward",
            DeviceIndex);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _counter.Dispose();
    }

    private void ValidateDropoutCall(
        nint first,
        nint second,
        int length,
        float dropoutProbability)
    {
        EnsureActive();
        if (first == nint.Zero || second == nint.Zero)
        {
            throw new ArgumentException(
                "CUDA Graph dropout operands must be device pointers.");
        }
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        if (!(dropoutProbability >= 0f) || dropoutProbability >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(dropoutProbability));
        }
    }

    private void ValidateFusedLayerNormForward(
        nint residual,
        nint branch,
        nint gamma,
        nint beta,
        nint output,
        nint means,
        nint inverses,
        int rows,
        int columns,
        float dropoutScale,
        float epsilon)
    {
        EnsureActive();
        if (residual == nint.Zero || branch == nint.Zero
            || gamma == nint.Zero || beta == nint.Zero
            || output == nint.Zero || means == nint.Zero
            || inverses == nint.Zero)
        {
            throw new ArgumentException(
                "CUDA Graph fused LayerNorm operands must be device pointers.");
        }
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        if (!(dropoutScale >= 1f) || !float.IsFinite(dropoutScale))
            throw new ArgumentOutOfRangeException(nameof(dropoutScale));
        if (!(epsilon > 0f) || !float.IsFinite(epsilon))
            throw new ArgumentOutOfRangeException(nameof(epsilon));
    }

    private void ValidateFusedLayerNormBackward(
        nint residual,
        nint branch,
        nint gamma,
        nint means,
        nint inverses,
        nint outputGradient,
        nint residualGradient,
        nint branchGradient,
        nint gammaGradient,
        nint betaGradient,
        nint parameterScratch,
        int rows,
        int columns,
        float dropoutScale)
    {
        EnsureActive();
        if (residual == nint.Zero || branch == nint.Zero
            || gamma == nint.Zero || means == nint.Zero
            || inverses == nint.Zero || outputGradient == nint.Zero
            || residualGradient == nint.Zero || branchGradient == nint.Zero
            || gammaGradient == nint.Zero || betaGradient == nint.Zero
            || parameterScratch == nint.Zero)
        {
            throw new ArgumentException(
                "CUDA Graph fused LayerNorm operands must be device pointers.");
        }
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        if (!(dropoutScale >= 1f) || !float.IsFinite(dropoutScale))
            throw new ArgumentOutOfRangeException(nameof(dropoutScale));
    }

    private void EnsureActive()
        => ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
}
