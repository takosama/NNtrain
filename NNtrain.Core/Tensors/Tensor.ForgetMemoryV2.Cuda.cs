
namespace NNtrain;

/// <summary>
/// CUDA implementation of the stateful ForgetMemory recurrence. One worker
/// owns a complete batch item, which keeps the time recurrence and all
/// gradient accumulation deterministic while allowing batch items to execute
/// concurrently on the GPU.
/// </summary>
internal static class ForgetMemoryV2Cuda
{
    private static readonly Lazy<int> CachedDeviceCount = new(
        () => NativeCudaRuntime.DeviceCount,
        LazyThreadSafetyMode.ExecutionAndPublication);
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<
        int,
        Lazy<NativeCudaDevice>> Accelerators = new();

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
        bool useDrn)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = GetAccelerator(deviceIndex);
        int matrixSize = checked(keyWidth * valueWidth);
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
            checked(batch * sequence * matrixSize));
        using var stateBuffer = accelerator.Allocate1D<float>(
            checked(batch * matrixSize));
        stateBuffer.MemSetToZero();
        int memoryVariant = useDrn ? 2 : useV3 ? 1 : 0;
        bool tensorCore = outputBFloat16 is not null
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
                memoryVariant);
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
        accelerator.Synchronize();
        return outputBFloat16 is not null
            ? new ResidentForwardResult(
                deviceIndex,
                outputBFloat16,
                statesBuffer)
            : new ResidentForwardResult(
                deviceIndex,
                outputFloat32!,
                statesBuffer);
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
        var projectedGradientBuffer = projected.EnsureCudaGradientBuffer(
            forward.DeviceIndex);
        var outputGradientBuffer = output.EnsureCudaGradientBuffer(
            forward.DeviceIndex);
        using var stateGradientBuffer = accelerator.Allocate1D<float>(
            checked(batch * matrixSize));
        using var previousGradientBuffer = accelerator.Allocate1D<float>(
            checked(batch * matrixSize));
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
        accelerator.Synchronize();
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
            NativeCudaBuffer<float> states)
        {
            DeviceIndex = deviceIndex;
            OutputBFloat16 = output;
            States = states;
        }

        internal int DeviceIndex { get; }
        internal NativeCudaBuffer<float>? OutputFloat32 { get; }
        internal NativeCudaBuffer<ushort>? OutputBFloat16 { get; }
        internal NativeCudaBuffer<float> States { get; }

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
        NativeCudaDevice accelerator = Accelerators.GetOrAdd(
            requestedIndex,
            static index => new Lazy<NativeCudaDevice>(
                () => NativeCudaRuntime.GetDevice(index),
                LazyThreadSafetyMode.ExecutionAndPublication)).Value;
        accelerator.Bind();
        return accelerator;
    }

}
