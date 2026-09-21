using NNtrain.Cuda.Memory;

namespace NNtrain;

/// <summary>
/// Resident CUDA ForgetMemory dispatch. Eligible mixed-BF16 DRN sequences use
/// exact checked chunk forward and fused replay/backward; other variants,
/// pure-BF16 gradients and recurrent inference retain the original kernels.
/// </summary>
internal static class ForgetMemoryV2Cuda
{
    private static readonly Lazy<int> CachedDeviceCount = new(
        () => NativeCudaRuntime.DeviceCount,
        LazyThreadSafetyMode.ExecutionAndPublication);

    internal static ResidentForwardResult ForwardResident(
        Tensor projected,
        int batch,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        bool bfloat16Compute,
        bool useV3,
        bool useDrn,
        NativeCudaBuffer<float>? recurrentState = null)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = GetAccelerator(deviceIndex);
        int matrixSize = checked(keyWidth * valueWidth);
        // The exact BF16 chunk kernels also serve mixed BF16 training. Pure
        // BF16 gradients and externally supplied recurrent states retain their
        // existing numerical/continuation contract.
        bool chunked = useDrn && recurrentState is null
            && projected.DType == TensorDType.BFloat16
            && !TensorExecutionContext.UsesBFloat16GradientStorage
            && CudaDrnChunk.CanUse(batch, sequence, keyWidth, valueWidth);
        NativeCudaBuffer<float>? outputFloat32 = null;
        NativeCudaBuffer<ushort>? outputBFloat16 = null;
        int outputLength = checked(batch * sequence * valueWidth);
        if (projected.DType == TensorDType.BFloat16)
        {
            projected.EnsureCudaBFloat16Buffer(deviceIndex);
            outputBFloat16 = Tensor.RentCudaBFloat16Buffer(
                deviceIndex,
                outputLength);
        }
        else
        {
            projected.EnsureCudaFloat32Buffer(deviceIndex);
            outputFloat32 = Tensor.RentCudaFloatBuffer(
                deviceIndex,
                outputLength);
        }
        var statesBuffer = accelerator.Allocate1D<float>(
            chunked ? CudaDrnChunk.CheckpointLength(batch, sequence, keyWidth, valueWidth)
                : checked(batch * sequence * matrixSize),
            CudaMemoryKind.Transient);
        NativeCudaBuffer<float>? ownedStateBuffer = null;
        NativeCudaBuffer<float> stateBuffer;
        if (recurrentState is null)
        {
            ownedStateBuffer = accelerator.Allocate1D<float>(
                checked(batch * matrixSize),
                CudaMemoryKind.Transient);
            ownedStateBuffer.MemSetToZero();
            stateBuffer = ownedStateBuffer;
        }
        else
        {
            if (recurrentState.Device.Index != deviceIndex
                || recurrentState.Length != checked(batch * matrixSize))
            {
                statesBuffer.Dispose();
                outputBFloat16?.Dispose();
                outputFloat32?.Dispose();
                throw new ArgumentException(
                    "Recurrent state must match the CUDA device and batch memory size.",
                    nameof(recurrentState));
            }
            stateBuffer = recurrentState;
        }
        int memoryVariant = useDrn ? 2 : useV3 ? 1 : 0;
        try
        {
            if (chunked)
                CudaDrnChunk.Forward(accelerator,
                    projected.EnsureCudaBFloat16Buffer(deviceIndex), outputBFloat16!,
                    statesBuffer, stateBuffer, batch, sequence, projectionWidth, keyWidth, valueWidth, retentionFloor);
            bool tensorCore = chunked || (outputBFloat16 is not null
                && CudaForgetMemoryTensorCore.TryForward(
                    accelerator,
                    projected.EnsureCudaBFloat16Buffer(deviceIndex),
                    outputBFloat16,
                    statesBuffer,
                    stateBuffer,
                    batch,
                    sequence,
                    projectionWidth,
                    keyWidth,
                    valueWidth,
                    retentionFloor,
                    memoryVariant));
            if (!tensorCore)
            {
                CudaForgetMemoryNative.Forward(
                    accelerator,
                    projected.DType == TensorDType.BFloat16
                        ? 0
                        : projected.EnsureCudaFloat32Buffer(deviceIndex).NativePtr,
                    projected.DType == TensorDType.BFloat16
                        ? projected.EnsureCudaBFloat16Buffer(deviceIndex).NativePtr
                        : 0,
                    outputFloat32?.NativePtr ?? 0,
                    outputBFloat16?.NativePtr ?? 0,
                    statesBuffer.NativePtr,
                    stateBuffer.NativePtr,
                    batch,
                    sequence,
                    projectionWidth,
                    keyWidth,
                    valueWidth,
                    retentionFloor,
                    memoryVariant,
                    bfloat16Compute);
            }
            // A session lane owns one ordered compute stream.  All buffers
            // above are either retained by the result/recurrent state or
            // returned to the lane's event-fenced pool.
            if (!TensorExecutionContext.TryGetCudaStreamLane(
                    deviceIndex,
                    out _))
            {
                accelerator.Synchronize();
            }
            return outputBFloat16 is not null
                ? new ResidentForwardResult(
                    deviceIndex,
                    outputBFloat16,
                    statesBuffer,
                    chunked)
                : new ResidentForwardResult(
                    deviceIndex,
                    outputFloat32!,
                    statesBuffer);
        }
        catch
        {
            statesBuffer.Dispose();
            outputBFloat16?.Dispose();
            outputFloat32?.Dispose();
            throw;
        }
        finally
        {
            ownedStateBuffer?.Dispose();
        }
    }

    internal static void BackwardResident(
        Tensor projected,
        Tensor output,
        ResidentForwardResult forward,
        int batch,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        bool bfloat16Compute,
        bool useV3,
        bool useDrn)
    {
        NativeCudaDevice accelerator = GetAccelerator(forward.DeviceIndex);
        int matrixSize = checked(keyWidth * valueWidth);
        if (projected.DType == TensorDType.BFloat16)
        {
            projected.EnsureCudaBFloat16Buffer(forward.DeviceIndex);
        }
        else
        {
            projected.EnsureCudaFloat32Buffer(forward.DeviceIndex);
        }

        if (projected.DType == TensorDType.BFloat16
            && TensorExecutionContext.UsesBFloat16GradientStorage)
        {
            NativeCudaBuffer<float>? decodedOutputGradient = null;
            NativeCudaBuffer<float>? projectedContribution = null;
            try
            {
                using CudaBFloat16GradientSource outputGradientSource =
                    CudaBFloat16GradientSource.Acquire(
                        output,
                        forward.DeviceIndex);
                using var targets =
                    new CudaPureBFloat16GradientTargetSet(
                        forward.DeviceIndex);
                CudaPureBFloat16GradientTarget projectedGradientTarget =
                    targets.Get(projected);
                decodedOutputGradient = Tensor.RentCudaFloatBuffer(
                    forward.DeviceIndex,
                    output.Numel);
                projectedContribution = Tensor.RentCudaFloatBuffer(
                    forward.DeviceIndex,
                    projected.Numel);
                projectedContribution.MemSetToZero();
                CudaTensorNative.DecodeBFloat16(
                    forward.DeviceIndex,
                    outputGradientSource.Buffer.NativePtr,
                    decodedOutputGradient.NativePtr,
                    output.Numel);
                using var pureStateGradient = accelerator.Allocate1D<float>(
                    checked(batch * matrixSize),
                    CudaMemoryKind.Transient);
                using var purePreviousGradient = accelerator.Allocate1D<float>(
                    checked(batch * matrixSize),
                    CudaMemoryKind.Transient);
                pureStateGradient.MemSetToZero();
                purePreviousGradient.MemSetToZero();
                CudaForgetMemoryNative.Backward(
                    accelerator,
                    0,
                    projected.EnsureCudaBFloat16Buffer(forward.DeviceIndex)
                        .NativePtr,
                    projectedContribution.NativePtr,
                    decodedOutputGradient.NativePtr,
                    forward.States.NativePtr,
                    pureStateGradient.NativePtr,
                    purePreviousGradient.NativePtr,
                    batch,
                    sequence,
                    projectionWidth,
                    keyWidth,
                    valueWidth,
                    retentionFloor,
                    useDrn ? 2 : useV3 ? 1 : 0,
                    bfloat16Compute);
                projectedGradientTarget.AccumulateFloat32(
                    projectedContribution,
                    projected.Numel);
                targets.CommitAll();
                if (!TensorExecutionContext.TryGetCudaStreamLane(
                        forward.DeviceIndex,
                        out _))
                {
                    accelerator.Synchronize();
                }
            }
            finally
            {
                if (projectedContribution is not null)
                {
                    Tensor.ReturnCudaFloatBuffer(
                        accelerator,
                        projectedContribution);
                }
                if (decodedOutputGradient is not null)
                {
                    Tensor.ReturnCudaFloatBuffer(
                        accelerator,
                        decodedOutputGradient);
                }
            }
            return;
        }

        var projectedGradientBuffer = projected.EnsureCudaGradientBuffer(
            forward.DeviceIndex);
        var outputGradientBuffer = output.EnsureCudaGradientBuffer(
            forward.DeviceIndex);
        if (forward.Chunked)
        {
            using var terminalGradient = accelerator.Allocate1D<float>(
                checked(batch * matrixSize), CudaMemoryKind.Transient);
            using var adjoints = accelerator.Allocate1D<float>(
                CudaDrnChunk.AdjointLength(batch, sequence, keyWidth, valueWidth), CudaMemoryKind.Transient);
            terminalGradient.MemSetToZero();
            CudaDrnChunk.Backward(accelerator,
                projected.EnsureCudaBFloat16Buffer(forward.DeviceIndex),
                projectedGradientBuffer, outputGradientBuffer, forward.States,
                terminalGradient, adjoints, batch, sequence, projectionWidth, keyWidth, valueWidth, retentionFloor);
            if (!TensorExecutionContext.TryGetCudaStreamLane(forward.DeviceIndex, out _))
                accelerator.Synchronize();
            projected.MarkCudaGradientMutated(forward.DeviceIndex);
            return;
        }
        using var stateGradientBuffer = accelerator.Allocate1D<float>(
            checked(batch * matrixSize),
            CudaMemoryKind.Transient);
        using var previousGradientBuffer = accelerator.Allocate1D<float>(
            checked(batch * matrixSize),
            CudaMemoryKind.Transient);
        stateGradientBuffer.MemSetToZero();
        previousGradientBuffer.MemSetToZero();
        CudaForgetMemoryNative.Backward(
            accelerator,
            projected.DType == TensorDType.BFloat16
                ? 0
                : projected.EnsureCudaFloat32Buffer(forward.DeviceIndex)
                    .NativePtr,
            projected.DType == TensorDType.BFloat16
                ? projected.EnsureCudaBFloat16Buffer(forward.DeviceIndex)
                    .NativePtr
                : 0,
            projectedGradientBuffer.NativePtr,
            outputGradientBuffer.NativePtr,
            forward.States.NativePtr,
            stateGradientBuffer.NativePtr,
            previousGradientBuffer.NativePtr,
            batch,
            sequence,
            projectionWidth,
            keyWidth,
            valueWidth,
            retentionFloor,
            useDrn ? 2 : useV3 ? 1 : 0,
            bfloat16Compute);
        // Backward is queued on the lane's ordered compute stream.  The next
        // consumer (gradient reduction/optimizer) observes the same stream
        // order, and transient state buffers may safely be reused only by a
        // later launch on that lane.  Do not serialize every ForgetMemory layer
        // with a device-wide wait in the hot training path.
        if (!TensorExecutionContext.TryGetCudaStreamLane(
                forward.DeviceIndex,
                out _))
        {
            accelerator.Synchronize();
        }
        projected.MarkCudaGradientMutated(forward.DeviceIndex);
    }

    internal sealed class ResidentForwardResult : IDisposable
    {
        private bool _disposed;
        internal ResidentForwardResult(
            int deviceIndex,
            NativeCudaBuffer<float> output,
            NativeCudaBuffer<float> states)
        {
            DeviceIndex = deviceIndex;
            OutputFloat32 = output;
            States = states;
        }

        internal ResidentForwardResult(
            int deviceIndex,
            NativeCudaBuffer<ushort> output,
            NativeCudaBuffer<float> states,
            bool chunked = false)
        {
            DeviceIndex = deviceIndex;
            OutputBFloat16 = output;
            States = states;
            Chunked = chunked;
        }

        internal int DeviceIndex { get; }
        internal NativeCudaBuffer<float>? OutputFloat32 { get; }
        internal NativeCudaBuffer<ushort>? OutputBFloat16 { get; }
        internal NativeCudaBuffer<float> States { get; }
        internal bool Chunked { get; }

        internal void Dispose()
        {
            if (_disposed)
                return;
            States.Dispose();
            _disposed = true;
            GC.SuppressFinalize(this);
        }

        void IDisposable.Dispose() => Dispose();

        ~ResidentForwardResult() => Dispose();
    }

    internal static ForwardResult Forward(
        float[] projected,
        int batch,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        bool bfloat16Compute,
        bool useV3 = false,
        bool useDrn = false)
    {
        int[] devices = Tensor.CudaDeviceIndices
            .Take(Math.Min(batch, Tensor.CudaDeviceIndices.Count))
            .ToArray();
        if (devices.Length == 1)
        {
            (float[] singleOutput, ShardContext singleContext) = ForwardSingle(
                GetAccelerator(devices[0]), projected, batch, sequence,
                projectionWidth, keyWidth, valueWidth, retentionFloor,
                bfloat16Compute, useV3, useDrn);
            return new ForwardResult(singleOutput, [singleContext]);
        }

        int projectedStride = checked(sequence * projectionWidth);
        int outputStride = checked(sequence * valueWidth);
        var output = new float[checked(batch * outputStride)];
        var contexts = new ShardContext[devices.Length];
        Parallel.For(0, devices.Length, shard =>
        {
            int start = batch * shard / devices.Length;
            int end = batch * (shard + 1) / devices.Length;
            int shardBatch = end - start;
            float[] shardProjected = projected
                .AsSpan(start * projectedStride, shardBatch * projectedStride)
                .ToArray();
            (float[] shardOutput, ShardContext shardContext) = ForwardSingle(
                GetAccelerator(devices[shard]), shardProjected, shardBatch,
                sequence, projectionWidth, keyWidth, valueWidth,
                retentionFloor, bfloat16Compute, useV3, useDrn);
            shardOutput.CopyTo(output, start * outputStride);
            shardContext.BatchStart = start;
            contexts[shard] = shardContext;
        });
        return new ForwardResult(output, contexts);
    }

    private static (float[] Output, ShardContext Context) ForwardSingle(
        NativeCudaDevice accelerator,
        float[] projected,
        int batch,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        bool bfloat16Compute,
        bool useV3,
        bool useDrn)
    {
        int matrixSize = checked(keyWidth * valueWidth);
        var output = new float[checked(batch * sequence * valueWidth)];

        NativeCudaBuffer<float> projectedBuffer =
            accelerator.Allocate1D(projected);
        using NativeCudaBuffer<float> outputBuffer =
            accelerator.Allocate1D<float>(output.Length);
        NativeCudaBuffer<float> statesBuffer =
            accelerator.Allocate1D<float>(checked(batch * sequence * matrixSize));
        using NativeCudaBuffer<float> stateBuffer =
            accelerator.Allocate1D<float>(checked(batch * matrixSize));
        stateBuffer.MemSetToZero();

        CudaForgetMemoryNative.Forward(
            accelerator,
            projectedBuffer.NativePtr,
            0,
            outputBuffer.NativePtr,
            0,
            statesBuffer.NativePtr,
            stateBuffer.NativePtr,
            batch,
            sequence,
            projectionWidth,
            keyWidth,
            valueWidth,
            retentionFloor,
            useDrn ? 2 : useV3 ? 1 : 0,
            bfloat16: false);
        accelerator.Synchronize();
        outputBuffer.CopyToCPU(output);
        return (output, new ShardContext(
            accelerator, projectedBuffer, statesBuffer, batch));
    }

    internal static float[] Backward(
        ForwardResult forward,
        float[] outputGradient,
        int batch,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        bool bfloat16Compute,
        bool useV3 = false,
        bool useDrn = false)
    {
        ArgumentNullException.ThrowIfNull(forward);
        int projectedStride = checked(sequence * projectionWidth);
        int outputStride = checked(sequence * valueWidth);
        var projectedGradient = new float[checked(batch * projectedStride)];
        Parallel.For(0, forward.Shards.Length, shard =>
        {
            ShardContext context = forward.Shards[shard];
            float[] shardGradient = BackwardSingle(
                context,
                outputGradient.AsSpan(
                    context.BatchStart * outputStride,
                    context.Batch * outputStride).ToArray(),
                sequence, projectionWidth, keyWidth, valueWidth,
                retentionFloor, bfloat16Compute, useV3, useDrn);
            shardGradient.CopyTo(
                projectedGradient,
                context.BatchStart * projectedStride);
        });
        return projectedGradient;
    }

    private static float[] BackwardSingle(
        ShardContext context,
        float[] outputGradient,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        bool bfloat16Compute,
        bool useV3,
        bool useDrn)
    {
        NativeCudaDevice accelerator = context.Accelerator;
        int batch = context.Batch;
        int matrixSize = checked(keyWidth * valueWidth);
        var projectedGradient = new float[checked(batch * sequence * projectionWidth)];

        using NativeCudaBuffer<float> outputGradientBuffer =
            accelerator.Allocate1D(outputGradient);
        using NativeCudaBuffer<float> projectedGradientBuffer =
            accelerator.Allocate1D<float>(projectedGradient.Length);
        using NativeCudaBuffer<float> stateGradientBuffer =
            accelerator.Allocate1D<float>(checked(batch * matrixSize));
        using NativeCudaBuffer<float> previousGradientBuffer =
            accelerator.Allocate1D<float>(checked(batch * matrixSize));
        projectedGradientBuffer.MemSetToZero();
        stateGradientBuffer.MemSetToZero();
        previousGradientBuffer.MemSetToZero();

        CudaForgetMemoryNative.Backward(
            accelerator,
            context.Projected.NativePtr,
            0,
            projectedGradientBuffer.NativePtr,
            outputGradientBuffer.NativePtr,
            context.States.NativePtr,
            stateGradientBuffer.NativePtr,
            previousGradientBuffer.NativePtr,
            batch,
            sequence,
            projectionWidth,
            keyWidth,
            valueWidth,
            retentionFloor,
            useDrn ? 2 : useV3 ? 1 : 0,
            bfloat16: false);
        accelerator.Synchronize();
        projectedGradientBuffer.CopyToCPU(projectedGradient);
        return projectedGradient;
    }

    internal sealed class ForwardResult : IDisposable
    {
        internal ForwardResult(float[] output, ShardContext[] shards)
        {
            Output = output;
            Shards = shards;
        }

        internal float[] Output { get; }
        internal ShardContext[] Shards { get; }

        public void Dispose()
        {
            foreach (ShardContext shard in Shards)
                shard.Dispose();
            GC.SuppressFinalize(this);
        }

        ~ForwardResult() => Dispose();
    }

    internal sealed class ShardContext : IDisposable
    {
        internal ShardContext(
            NativeCudaDevice accelerator,
            NativeCudaBuffer<float> projected,
            NativeCudaBuffer<float> states,
            int batch)
        {
            Accelerator = accelerator;
            Projected = projected;
            States = states;
            Batch = batch;
        }

        internal NativeCudaDevice Accelerator { get; }
        internal NativeCudaBuffer<float> Projected { get; }
        internal NativeCudaBuffer<float> States { get; }
        internal int Batch { get; }
        internal int BatchStart { get; set; }

        public void Dispose()
        {
            Projected.Dispose();
            States.Dispose();
        }
    }

    internal static string DeviceName => string.Join(
        ", ",
        Tensor.CudaDeviceIndices.Select(
            index => $"cuda:{index} {GetAccelerator(index).Name}"));

    internal static int DeviceCount
    {
        get
        {
            try
            {
                return CachedDeviceCount.Value;
            }
            catch
            {
                return 0;
            }
        }
    }

    internal static bool IsAvailable(int deviceIndex)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceIndex);
        try
        {
            return deviceIndex < CachedDeviceCount.Value;
        }
        catch
        {
            return false;
        }
    }

    internal static NativeCudaDevice GetAccelerator()
        => GetAccelerator(Tensor.CudaDeviceIndex);

    internal static NativeCudaDevice GetAccelerator(int requestedIndex)
    {
        int deviceCount = CachedDeviceCount.Value;
        if ((uint)requestedIndex >= (uint)deviceCount)
        {
            throw new InvalidOperationException(
                $"CUDA device index {requestedIndex} is unavailable; " +
                $"detected {deviceCount} CUDA device(s).");
        }
        // NativeCudaRuntime owns the single process-level wrapper per
        // validated physical device. Do not add a second static cache here;
        // lane/session resources are owned below that canonical wrapper.
        NativeCudaDevice accelerator = NativeCudaRuntime.GetDevice(
            requestedIndex);
        accelerator.Bind();
        return accelerator;
    }

}
