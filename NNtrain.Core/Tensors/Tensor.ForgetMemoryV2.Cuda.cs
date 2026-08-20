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

    internal static ForwardResult Forward(
        float[] projected,
        int batch,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor)
    {
        int[] devices = Tensor.CudaDeviceIndices
            .Take(Math.Min(batch, Tensor.CudaDeviceIndices.Count))
            .ToArray();
        if (devices.Length == 1)
        {
            (float[] singleOutput, ShardContext singleContext) = ForwardSingle(
                GetAccelerator(devices[0]), projected, batch, sequence,
                projectionWidth, keyWidth, valueWidth, retentionFloor);
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
                retentionFloor);
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
        float retentionFloor)
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
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<float>,
            int,
            int,
            int,
            int,
            float>(ForwardKernel);
        kernel(
            checked(batch * valueWidth),
            projectedBuffer.View,
            outputBuffer.View,
            statesBuffer.View,
            stateBuffer.View,
            sequence,
            projectionWidth,
            keyWidth,
            valueWidth,
            retentionFloor);
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
        float retentionFloor)
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
                retentionFloor);
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
        float retentionFloor)
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
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<float>,
            ArrayView<float>,
            int,
            int,
            int,
            int,
            float>(BackwardKernel);
        kernel(
            checked(batch * valueWidth),
            context.Projected.View,
            projectedGradientBuffer.View,
            outputGradientBuffer.View,
            context.States.View,
            stateGradientBuffer.View,
            previousGradientBuffer.View,
            sequence,
            projectionWidth,
            keyWidth,
            valueWidth,
            retentionFloor);
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
        ArrayView<float> output,
        ArrayView<float> states,
        ArrayView<float> state,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor)
    {
        int worker = batchIndex;
        int batch = worker / valueWidth;
        int valueIndex = worker - batch * valueWidth;
        int matrixSize = keyWidth * valueWidth;
        int projectedBatchOffset = batch * sequence * projectionWidth;
        int outputBatchOffset = batch * sequence * valueWidth;
        int stateBatchOffset = batch * matrixSize;
        int statesBatchOffset = batch * sequence * matrixSize;

        for (int time = 0; time < sequence; time++)
        {
            int projectedOffset = projectedBatchOffset + time * projectionWidth;
            int keyOffset = projectedOffset + keyWidth;
            int valueOffset = keyOffset + keyWidth;
            int gateOffset = valueOffset + valueWidth;
            int betaOffset = gateOffset + valueWidth;

            int row = stateBatchOffset + valueIndex * keyWidth;
            float gate = Sigmoid(projected[gateOffset + valueIndex]);
            float retention = retentionFloor +
                (1f - retentionFloor) * gate;
            float beta = Sigmoid(projected[betaOffset + valueIndex]);
            float write = (1f - retention) * beta;
            float value = XMath.Tanh(projected[valueOffset + valueIndex]);
            float predicted = 0f;
            for (int key = 0; key < keyWidth; key++)
                predicted += state[row + key] * projected[keyOffset + key];
            float delta = write * (value - predicted);
            for (int key = 0; key < keyWidth; key++)
            {
                state[row + key] = retention * state[row + key] +
                    delta * projected[keyOffset + key];
            }

            float recalled = 0f;
            for (int key = 0; key < keyWidth; key++)
                recalled += state[row + key] * projected[projectedOffset + key];
            output[outputBatchOffset + time * valueWidth + valueIndex] =
                recalled;

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
        ArrayView<float> projectedGradient,
        ArrayView<float> outputGradient,
        ArrayView<float> states,
        ArrayView<float> stateGradient,
        ArrayView<float> previousGradient,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor)
    {
        int worker = batchIndex;
        int batch = worker / valueWidth;
        int valueIndex = worker - batch * valueWidth;
        int matrixSize = keyWidth * valueWidth;
        int projectedBatchOffset = batch * sequence * projectionWidth;
        int outputBatchOffset = batch * sequence * valueWidth;
        int statesBatchOffset = batch * sequence * matrixSize;
        int gradientBatchOffset = batch * matrixSize;

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
            for (int key = 0; key < keyWidth; key++)
            {
                Atomic.Add(
                    ref projectedGradient[queryOffset + key],
                    states[currentStateOffset + row + key] * recalledGradient);
                stateGradient[gradientRow + key] +=
                    projected[queryOffset + key] * recalledGradient;
            }

            float gate = Sigmoid(projected[gateOffset + valueIndex]);
            float retention = retentionFloor +
                (1f - retentionFloor) * gate;
            float beta = Sigmoid(projected[betaOffset + valueIndex]);
            float write = (1f - retention) * beta;
            float value = XMath.Tanh(projected[valueOffset + valueIndex]);
            float predicted = 0f;
            float stateGradientDotKey = 0f;
            float retentionGradient = 0f;
            for (int key = 0; key < keyWidth; key++)
            {
                float previous = time == 0
                    ? 0f
                    : states[previousStateOffset + row + key];
                float gradient = stateGradient[gradientRow + key];
                float keyValue = projected[keyOffset + key];
                predicted += previous * keyValue;
                stateGradientDotKey += gradient * keyValue;
                retentionGradient += gradient * previous;
            }

            float error = value - predicted;
            float writeGradient = error * stateGradientDotKey;
            float errorGradient = write * stateGradientDotKey;
            retentionGradient -= writeGradient * beta;
            projectedGradient[valueOffset + valueIndex] +=
                errorGradient * (1f - value * value);
            projectedGradient[gateOffset + valueIndex] +=
                retentionGradient * (1f - retentionFloor) *
                gate * (1f - gate);
            projectedGradient[betaOffset + valueIndex] +=
                writeGradient * (1f - retention) * beta * (1f - beta);

            for (int key = 0; key < keyWidth; key++)
            {
                float previous = time == 0
                    ? 0f
                    : states[previousStateOffset + row + key];
                float gradient = stateGradient[gradientRow + key];
                float keyValue = projected[keyOffset + key];
                Atomic.Add(
                    ref projectedGradient[keyOffset + key],
                    gradient * write * error - previous * errorGradient);
                previousGradient[gradientRow + key] =
                    gradient * retention - keyValue * errorGradient;
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
}
