using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace NNtrain;

/// <summary>
/// CUDA implementation of the stateful ForgetMemory recurrence. One worker
/// owns a complete batch item, which keeps the time recurrence and all
/// gradient accumulation deterministic while allowing batch items to execute
/// concurrently on the GPU.
/// </summary>
internal static class ForgetMemoryV2Cuda
{
    private static readonly object Sync = new();
    private static Context? _context;
    private static readonly Dictionary<int, CudaAccelerator> Accelerators = [];

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
        CudaAccelerator accelerator = GetAccelerator(deviceIndex);
        int matrixSize = checked(keyWidth * valueWidth);
        ArrayView<float> projectedFloat32 = default;
        ArrayView<ushort> projectedBFloat16 = default;
        MemoryBuffer1D<float, Stride1D.Dense>? outputFloat32 = null;
        MemoryBuffer1D<ushort, Stride1D.Dense>? outputBFloat16 = null;
        int outputLength = checked(batch * sequence * valueWidth);
        if (projected.DType == TensorDType.BFloat16)
        {
            projectedBFloat16 = projected
                .EnsureCudaBFloat16Buffer(deviceIndex).View;
            outputBFloat16 = Tensor.RentCudaBFloat16Buffer(
                deviceIndex,
                outputLength);
        }
        else
        {
            projectedFloat32 = projected
                .EnsureCudaFloat32Buffer(deviceIndex).View;
            outputFloat32 = Tensor.RentCudaFloatBuffer(
                deviceIndex,
                outputLength);
        }
        var statesBuffer = accelerator.Allocate1D<float>(
            checked(batch * sequence * matrixSize));
        using var stateBuffer = accelerator.Allocate1D<float>(
            checked(batch * matrixSize));
        stateBuffer.MemSetToZero();
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<ushort>,
            ArrayView<float>,
            ArrayView<ushort>,
            ArrayView<float>,
            ArrayView<float>,
            int,
            int,
            int,
            int,
            float,
            int,
            int>(ForwardKernel);
        kernel(
            checked(batch * valueWidth),
            projectedFloat32,
            projectedBFloat16,
            outputFloat32?.View ?? default,
            outputBFloat16?.View ?? default,
            statesBuffer.View,
            stateBuffer.View,
            sequence,
            projectionWidth,
            keyWidth,
            valueWidth,
            retentionFloor,
            useDrn ? 2 : useV3 ? 1 : 0,
            bfloat16Compute ? 1 : 0);
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
        CudaAccelerator accelerator = GetAccelerator(forward.DeviceIndex);
        int matrixSize = checked(keyWidth * valueWidth);
        ArrayView<float> projectedFloat32 = default;
        ArrayView<ushort> projectedBFloat16 = default;
        if (projected.DType == TensorDType.BFloat16)
        {
            projectedBFloat16 = projected
                .EnsureCudaBFloat16Buffer(forward.DeviceIndex).View;
        }
        else
        {
            projectedFloat32 = projected
                .EnsureCudaFloat32Buffer(forward.DeviceIndex).View;
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
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<ushort>,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<float>,
            int,
            int,
            int,
            int,
            float,
            int,
            int>(BackwardKernel);
        kernel(
            checked(batch * valueWidth),
            projectedFloat32,
            projectedBFloat16,
            projectedGradientBuffer.View,
            outputGradientBuffer.View,
            forward.States.View,
            stateGradientBuffer.View,
            previousGradientBuffer.View,
            sequence,
            projectionWidth,
            keyWidth,
            valueWidth,
            retentionFloor,
            useDrn ? 2 : useV3 ? 1 : 0,
            bfloat16Compute ? 1 : 0);
        accelerator.Synchronize();
        projected.MarkCudaGradientMutated(forward.DeviceIndex);
    }

    internal sealed class ResidentForwardResult : IDisposable
    {
        private bool _disposed;
        internal ResidentForwardResult(
            int deviceIndex,
            MemoryBuffer1D<float, Stride1D.Dense> output,
            MemoryBuffer1D<float, Stride1D.Dense> states)
        {
            DeviceIndex = deviceIndex;
            OutputFloat32 = output;
            States = states;
        }

        internal ResidentForwardResult(
            int deviceIndex,
            MemoryBuffer1D<ushort, Stride1D.Dense> output,
            MemoryBuffer1D<float, Stride1D.Dense> states)
        {
            DeviceIndex = deviceIndex;
            OutputBFloat16 = output;
            States = states;
        }

        internal int DeviceIndex { get; }
        internal MemoryBuffer1D<float, Stride1D.Dense>? OutputFloat32 { get; }
        internal MemoryBuffer1D<ushort, Stride1D.Dense>? OutputBFloat16 { get; }
        internal MemoryBuffer1D<float, Stride1D.Dense> States { get; }

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
        CudaAccelerator accelerator,
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

        MemoryBuffer1D<float, Stride1D.Dense> projectedBuffer =
            accelerator.Allocate1D(projected);
        using MemoryBuffer1D<float, Stride1D.Dense> outputBuffer =
            accelerator.Allocate1D<float>(output.Length);
        MemoryBuffer1D<float, Stride1D.Dense> statesBuffer =
            accelerator.Allocate1D<float>(checked(batch * sequence * matrixSize));
        using MemoryBuffer1D<float, Stride1D.Dense> stateBuffer =
            accelerator.Allocate1D<float>(checked(batch * matrixSize));
        stateBuffer.MemSetToZero();

        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<ushort>,
            ArrayView<float>,
            ArrayView<ushort>,
            ArrayView<float>,
            ArrayView<float>,
            int,
            int,
            int,
            int,
            float,
            int,
            int>(ForwardKernel);
        kernel(
            checked(batch * valueWidth),
            projectedBuffer.View,
            default,
            outputBuffer.View,
            default,
            statesBuffer.View,
            stateBuffer.View,
            sequence,
            projectionWidth,
            keyWidth,
            valueWidth,
            retentionFloor,
            useDrn ? 2 : useV3 ? 1 : 0,
            bfloat16Compute ? 1 : 0);
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
        CudaAccelerator accelerator = context.Accelerator;
        int batch = context.Batch;
        int matrixSize = checked(keyWidth * valueWidth);
        var projectedGradient = new float[checked(batch * sequence * projectionWidth)];

        using MemoryBuffer1D<float, Stride1D.Dense> outputGradientBuffer =
            accelerator.Allocate1D(outputGradient);
        using MemoryBuffer1D<float, Stride1D.Dense> projectedGradientBuffer =
            accelerator.Allocate1D<float>(projectedGradient.Length);
        using MemoryBuffer1D<float, Stride1D.Dense> stateGradientBuffer =
            accelerator.Allocate1D<float>(checked(batch * matrixSize));
        using MemoryBuffer1D<float, Stride1D.Dense> previousGradientBuffer =
            accelerator.Allocate1D<float>(checked(batch * matrixSize));
        projectedGradientBuffer.MemSetToZero();
        stateGradientBuffer.MemSetToZero();
        previousGradientBuffer.MemSetToZero();

        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<float>,
            ArrayView<ushort>,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<float>,
            int,
            int,
            int,
            int,
            float,
            int,
            int>(BackwardKernel);
        kernel(
            checked(batch * valueWidth),
            context.Projected.View,
            default,
            projectedGradientBuffer.View,
            outputGradientBuffer.View,
            context.States.View,
            stateGradientBuffer.View,
            previousGradientBuffer.View,
            sequence,
            projectionWidth,
            keyWidth,
            valueWidth,
            retentionFloor,
            useDrn ? 2 : useV3 ? 1 : 0,
            bfloat16Compute ? 1 : 0);
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
            CudaAccelerator accelerator,
            MemoryBuffer1D<float, Stride1D.Dense> projected,
            MemoryBuffer1D<float, Stride1D.Dense> states,
            int batch)
        {
            Accelerator = accelerator;
            Projected = projected;
            States = states;
            Batch = batch;
        }

        internal CudaAccelerator Accelerator { get; }
        internal MemoryBuffer1D<float, Stride1D.Dense> Projected { get; }
        internal MemoryBuffer1D<float, Stride1D.Dense> States { get; }
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
                EnsureContext();
                return _context!.GetCudaDevices().Count;
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
            using Context context = Context.Create(builder => builder.Cuda());
            return deviceIndex < context.GetCudaDevices().Count;
        }
        catch
        {
            return false;
        }
    }

    internal static CudaAccelerator GetAccelerator()
        => GetAccelerator(Tensor.CudaDeviceIndex);

    internal static CudaAccelerator GetAccelerator(int requestedIndex)
    {
        lock (Sync)
        {
            EnsureContext();
            Context.DeviceCollection<CudaDevice> devices =
                _context!.GetCudaDevices();
            if ((uint)requestedIndex >= (uint)devices.Count)
            {
                throw new InvalidOperationException(
                    $"CUDA device index {requestedIndex} is unavailable; " +
                    $"detected {devices.Count} CUDA device(s).");
            }
            if (!Accelerators.TryGetValue(requestedIndex, out CudaAccelerator? accelerator))
            {
                accelerator = devices[requestedIndex]
                    .CreateCudaAccelerator(_context!);
                Accelerators.Add(requestedIndex, accelerator);
            }
            return accelerator;
        }
    }

    private static void EnsureContext()
    {
        if (_context is null)
        {
            _context = Context.Create(
                builder => builder.Cuda().EnableAlgorithms());
        }
    }

    private static void ForwardKernel(
        Index1D batchIndex,
        ArrayView<float> projected,
        ArrayView<ushort> projectedBFloat16,
        ArrayView<float> output,
        ArrayView<ushort> outputBFloat16,
        ArrayView<float> states,
        ArrayView<float> state,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        int memoryVariant,
        int bfloat16Compute)
    {
        int useV3 = memoryVariant == 1 ? 1 : 0;
        int useDrn = memoryVariant == 2 ? 1 : 0;
        int worker = batchIndex;
        int batch = worker / valueWidth;
        int valueIndex = worker - batch * valueWidth;
        int matrixSize = keyWidth * valueWidth;
        int projectedBatchOffset = batch * sequence * projectionWidth;
        int outputBatchOffset = batch * sequence * valueWidth;
        int stateBatchOffset = batch * matrixSize;
        int statesBatchOffset = batch * sequence * matrixSize;
        float inverseSqrtKeyWidth = 1f / XMath.Sqrt((float)keyWidth);

        for (int time = 0; time < sequence; time++)
        {
            int projectedOffset = projectedBatchOffset + time * projectionWidth;
            int keyOffset = projectedOffset + keyWidth;
            int valueOffset = keyOffset + keyWidth;
            int gateOffset = valueOffset + valueWidth;
            int betaOffset = gateOffset + valueWidth;

            int row = stateBatchOffset + valueIndex * keyWidth;
            float gate = Sigmoid(ReadProjected(
                projected, projectedBFloat16, gateOffset + valueIndex,
                bfloat16Compute));
            float retention = useDrn != 0
                ? gate
                : retentionFloor + (1f - retentionFloor) * gate;
            float beta = Sigmoid(ReadProjected(
                projected, projectedBFloat16, betaOffset + valueIndex,
                bfloat16Compute));
            float write = useV3 != 0 || useDrn != 0
                ? beta
                : (1f - retention) * beta;
            float value = XMath.Tanh(ReadProjected(
                projected, projectedBFloat16, valueOffset + valueIndex,
                bfloat16Compute));
            float keySquaredNorm = useDrn != 0 ? 1e-8f : 1e-6f;
            float querySquaredNorm = 1e-8f;
            if (useV3 != 0 || useDrn != 0)
            {
                for (int key = 0; key < keyWidth; key++)
                {
                    float keyTanh = XMath.Tanh(ReadProjected(
                        projected, projectedBFloat16, keyOffset + key,
                        bfloat16Compute));
                    keySquaredNorm += keyTanh * keyTanh;
                    if (useDrn != 0)
                    {
                        float queryTanh = XMath.Tanh(ReadProjected(
                            projected, projectedBFloat16,
                            projectedOffset + key, bfloat16Compute));
                        querySquaredNorm += queryTanh * queryTanh;
                    }
                }
            }
            float keyScale = useV3 != 0 || useDrn != 0
                ? 1f / XMath.Sqrt(keySquaredNorm)
                : inverseSqrtKeyWidth;
            float queryScale = useDrn != 0
                ? 1f / XMath.Sqrt(querySquaredNorm)
                : inverseSqrtKeyWidth;

            if (useDrn != 0)
            {
                float recalledBeforeWrite = 0f;
                for (int key = 0; key < keyWidth; key++)
                {
                    float normalizedQuery = XMath.Tanh(ReadProjected(
                        projected, projectedBFloat16,
                        projectedOffset + key, bfloat16Compute)) * queryScale;
                    recalledBeforeWrite += state[row + key]
                        * normalizedQuery;
                }
                WriteOutput(
                    output,
                    outputBFloat16,
                    outputBatchOffset + time * valueWidth + valueIndex,
                    recalledBeforeWrite,
                    bfloat16Compute);
            }
            float predicted = 0f;
            for (int key = 0; key < keyWidth; key++)
            {
                float normalizedKey = XMath.Tanh(ReadProjected(
                    projected, projectedBFloat16, keyOffset + key,
                    bfloat16Compute)) * keyScale;
                predicted += state[row + key] * normalizedKey;
            }
            if (useV3 != 0)
                predicted *= retention;
            float delta = write * (value - predicted);
            for (int key = 0; key < keyWidth; key++)
            {
                float normalizedKey = XMath.Tanh(ReadProjected(
                    projected, projectedBFloat16, keyOffset + key,
                    bfloat16Compute)) * keyScale;
                state[row + key] = retention * state[row + key]
                    + delta * normalizedKey;
            }

            if (useDrn == 0)
            {
                float recalled = 0f;
                for (int key = 0; key < keyWidth; key++)
                {
                    float normalizedQuery = XMath.Tanh(ReadProjected(
                        projected, projectedBFloat16,
                        projectedOffset + key, bfloat16Compute)) * queryScale;
                    recalled += state[row + key] * normalizedQuery;
                }
                WriteOutput(
                    output,
                    outputBFloat16,
                    outputBatchOffset + time * valueWidth + valueIndex,
                    recalled,
                    bfloat16Compute);
            }

            int stateTimeOffset = statesBatchOffset + time * matrixSize;
            int rowOffset = valueIndex * keyWidth;
            for (int key = 0; key < keyWidth; key++)
            {
                states[stateTimeOffset + rowOffset + key] =
                    state[stateBatchOffset + rowOffset + key];
            }
        }
    }

    private static void BackwardKernel(
        Index1D batchIndex,
        ArrayView<float> projected,
        ArrayView<ushort> projectedBFloat16,
        ArrayView<float> projectedGradient,
        ArrayView<float> outputGradient,
        ArrayView<float> states,
        ArrayView<float> stateGradient,
        ArrayView<float> previousGradient,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        int memoryVariant,
        int bfloat16Compute)
    {
        int useV3 = memoryVariant == 1 ? 1 : 0;
        int useDrn = memoryVariant == 2 ? 1 : 0;
        int worker = batchIndex;
        int batch = worker / valueWidth;
        int valueIndex = worker - batch * valueWidth;
        int matrixSize = keyWidth * valueWidth;
        int projectedBatchOffset = batch * sequence * projectionWidth;
        int outputBatchOffset = batch * sequence * valueWidth;
        int statesBatchOffset = batch * sequence * matrixSize;
        int gradientBatchOffset = batch * matrixSize;
        float inverseSqrtKeyWidth = 1f / XMath.Sqrt((float)keyWidth);

        for (int time = sequence - 1; time >= 0; time--)
        {
            int projectedOffset = projectedBatchOffset + time * projectionWidth;
            int queryOffset = projectedOffset;
            int keyOffset = queryOffset + keyWidth;
            int valueOffset = keyOffset + keyWidth;
            int gateOffset = valueOffset + valueWidth;
            int betaOffset = gateOffset + valueWidth;
            int currentStateOffset = statesBatchOffset + time * matrixSize;
            int previousStateOffset = statesBatchOffset + (time - 1) * matrixSize;

            int row = valueIndex * keyWidth;
            int gradientRow = gradientBatchOffset + row;
            for (int key = 0; key < keyWidth; key++)
                previousGradient[gradientRow + key] = 0f;

            float recalledGradient = outputGradient[
                outputBatchOffset + time * valueWidth + valueIndex];
            if (useDrn != 0)
            {
                float querySquaredNorm = 1e-8f;
                for (int key = 0; key < keyWidth; key++)
                {
                    float queryTanh = XMath.Tanh(ReadProjected(
                        projected, projectedBFloat16, queryOffset + key,
                        bfloat16Compute));
                    querySquaredNorm += queryTanh * queryTanh;
                }
                float queryScale = 1f / XMath.Sqrt(querySquaredNorm);
                float queryTanhDotGradient = 0f;
                for (int key = 0; key < keyWidth; key++)
                {
                    float previous = time == 0
                        ? 0f
                        : states[previousStateOffset + row + key];
                    float queryGradient = previous * recalledGradient;
                    queryTanhDotGradient +=
                        XMath.Tanh(ReadProjected(
                            projected, projectedBFloat16, queryOffset + key,
                            bfloat16Compute))
                        * queryGradient;
                }
                for (int key = 0; key < keyWidth; key++)
                {
                    float previous = time == 0
                        ? 0f
                        : states[previousStateOffset + row + key];
                    float queryTanh = XMath.Tanh(ReadProjected(
                        projected, projectedBFloat16, queryOffset + key,
                        bfloat16Compute));
                    float queryGradient = previous * recalledGradient;
                    float tanhGradient = queryGradient * queryScale
                        - queryTanh * queryTanhDotGradient
                            * queryScale * queryScale * queryScale;
                    Atomic.Add(
                        ref projectedGradient[queryOffset + key],
                        tanhGradient * (1f - queryTanh * queryTanh));
                    previousGradient[gradientRow + key] +=
                        queryTanh * queryScale * recalledGradient;
                }
            }
            else
            {
                for (int key = 0; key < keyWidth; key++)
                {
                    float queryTanh = XMath.Tanh(ReadProjected(
                        projected, projectedBFloat16, queryOffset + key,
                        bfloat16Compute));
                    float normalizedQuery = queryTanh * inverseSqrtKeyWidth;
                    float queryDerivative =
                        (1f - queryTanh * queryTanh) * inverseSqrtKeyWidth;
                    Atomic.Add(
                        ref projectedGradient[queryOffset + key],
                        states[currentStateOffset + row + key]
                        * recalledGradient * queryDerivative);
                    stateGradient[gradientRow + key] +=
                        normalizedQuery * recalledGradient;
                }
            }

            float gate = Sigmoid(ReadProjected(
                projected, projectedBFloat16, gateOffset + valueIndex,
                bfloat16Compute));
            float retention = useDrn != 0
                ? gate
                : retentionFloor + (1f - retentionFloor) * gate;
            float beta = Sigmoid(ReadProjected(
                projected, projectedBFloat16, betaOffset + valueIndex,
                bfloat16Compute));
            float write = useV3 != 0 || useDrn != 0
                ? beta
                : (1f - retention) * beta;
            float value = XMath.Tanh(ReadProjected(
                projected, projectedBFloat16, valueOffset + valueIndex,
                bfloat16Compute));
            float keySquaredNorm = useDrn != 0 ? 1e-8f : 1e-6f;
            if (useV3 != 0 || useDrn != 0)
            {
                for (int key = 0; key < keyWidth; key++)
                {
                    float keyTanh = XMath.Tanh(ReadProjected(
                        projected, projectedBFloat16, keyOffset + key,
                        bfloat16Compute));
                    keySquaredNorm += keyTanh * keyTanh;
                }
            }
            float keyNorm = useV3 != 0 || useDrn != 0
                ? XMath.Sqrt(keySquaredNorm)
                : XMath.Sqrt((float)keyWidth);
            float keyScale = 1f / keyNorm;
            float predicted = 0f;
            float stateGradientDotKey = 0f;
            float retentionGradient = 0f;
            for (int key = 0; key < keyWidth; key++)
            {
                float previous = time == 0
                    ? 0f
                    : states[previousStateOffset + row + key];
                float gradient = stateGradient[gradientRow + key];
                float keyValue = XMath.Tanh(ReadProjected(
                    projected, projectedBFloat16, keyOffset + key,
                    bfloat16Compute)) * keyScale;
                predicted += previous * keyValue;
                stateGradientDotKey += gradient * keyValue;
                retentionGradient += gradient * previous;
            }

            float retainedPrediction = useV3 != 0
                ? retention * predicted
                : predicted;
            float error = value - retainedPrediction;
            float writeGradient = error * stateGradientDotKey;
            float errorGradient = write * stateGradientDotKey;
            if (useV3 != 0)
                retentionGradient -= errorGradient * predicted;
            else if (useDrn == 0)
                retentionGradient -= writeGradient * beta;
            projectedGradient[valueOffset + valueIndex] +=
                errorGradient * (1f - value * value);
            projectedGradient[gateOffset + valueIndex] +=
                retentionGradient
                    * (useDrn != 0 ? 1f : 1f - retentionFloor) *
                gate * (1f - gate);
            projectedGradient[betaOffset + valueIndex] +=
                writeGradient
                    * (useV3 != 0 || useDrn != 0 ? 1f : 1f - retention)
                    * beta * (1f - beta);

            float keyTanhDotGradient = 0f;
            if (useV3 != 0 || useDrn != 0)
            {
                for (int key = 0; key < keyWidth; key++)
                {
                    float previous = time == 0
                        ? 0f
                        : states[previousStateOffset + row + key];
                    float gradient = stateGradient[gradientRow + key];
                    float keyGradient = gradient * write * error
                        - previous * errorGradient
                            * (useV3 != 0 ? retention : 1f);
                    keyTanhDotGradient +=
                        XMath.Tanh(ReadProjected(
                            projected, projectedBFloat16, keyOffset + key,
                            bfloat16Compute)) * keyGradient;
                }
            }

            for (int key = 0; key < keyWidth; key++)
            {
                float previous = time == 0
                    ? 0f
                    : states[previousStateOffset + row + key];
                float gradient = stateGradient[gradientRow + key];
                float keyTanh = XMath.Tanh(ReadProjected(
                    projected, projectedBFloat16, keyOffset + key,
                    bfloat16Compute));
                float keyValue = keyTanh * keyScale;
                float keyGradient = gradient * write * error
                    - previous * errorGradient
                        * (useV3 != 0 ? retention : 1f);
                float tanhGradient = useV3 != 0 || useDrn != 0
                    ? keyGradient * keyScale
                        - keyTanh * keyTanhDotGradient
                            * keyScale * keyScale * keyScale
                    : keyGradient * keyScale;
                Atomic.Add(
                    ref projectedGradient[keyOffset + key],
                    tanhGradient * (1f - keyTanh * keyTanh));
                float recurrentPreviousGradient = useV3 != 0
                    ? retention * (gradient - keyValue * errorGradient)
                    : gradient * retention - keyValue * errorGradient;
                if (useDrn != 0)
                {
                    previousGradient[gradientRow + key] +=
                        recurrentPreviousGradient;
                }
                else
                {
                    previousGradient[gradientRow + key] =
                        recurrentPreviousGradient;
                }
            }

            for (int key = 0; key < keyWidth; key++)
            {
                stateGradient[gradientRow + key] =
                    previousGradient[gradientRow + key];
            }
        }
    }

    private static float Sigmoid(float value)
        => value >= 0f
            ? 1f / (1f + XMath.Exp(-value))
            : XMath.Exp(value) / (1f + XMath.Exp(value));

    private static float ReadProjected(
        ArrayView<float> projected,
        ArrayView<ushort> projectedBFloat16,
        int index,
        int bfloat16Compute)
        => bfloat16Compute != 0
            ? Interop.IntAsFloat((uint)projectedBFloat16[index] << 16)
            : projected[index];

    private static void WriteOutput(
        ArrayView<float> output,
        ArrayView<ushort> outputBFloat16,
        int index,
        float value,
        int bfloat16Compute)
    {
        if (bfloat16Compute != 0)
        {
            uint bits = Interop.FloatAsInt(value);
            uint roundingBias = 0x7FFFu + ((bits >> 16) & 1u);
            outputBFloat16[index] = (ushort)((bits + roundingBias) >> 16);
        }
        else
        {
            output[index] = value;
        }
    }

    private static float RoundBFloat16(float value)
    {
        uint bits = Interop.FloatAsInt(value);
        uint roundingBias = 0x7FFFu + ((bits >> 16) & 1u);
        return Interop.IntAsFloat((bits + roundingBias) & 0xFFFF0000u);
    }
}
