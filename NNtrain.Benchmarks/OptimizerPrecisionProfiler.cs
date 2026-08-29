using System.Diagnostics;
using NNtrain.Runtime.Execution;

namespace NNtrain.Benchmarks;

/// <summary>
/// Focused optimizer benchmark. Gradient publication is deliberately outside
/// the timed interval so the result measures the resident optimizer update,
/// its required completion barrier, and scalar finite-status readbacks only.
/// </summary>
internal static class OptimizerPrecisionProfiler
{
    private static readonly string[] OptimizerNames =
        ["adamw", "nekomuon", "lion", "gainshareadamw"];

    private static readonly TensorPrecisionMode[] PrecisionModes =
    [
        TensorPrecisionMode.Float32,
        TensorPrecisionMode.BFloat16,
        TensorPrecisionMode.Mix16_32,
        TensorPrecisionMode.Bfp8,
        TensorPrecisionMode.Mix8_32,
    ];

    internal static void Run(
        string optimizerSelector,
        string precisionSelector,
        int deviceCount,
        int warmup,
        int iterations,
        int parameterCount,
        int rows,
        int columns)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceCount);
        ArgumentOutOfRangeException.ThrowIfNegative(warmup);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(iterations);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(parameterCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        if (deviceCount > 0 && Tensor.CudaDeviceCount < deviceCount)
        {
            throw new InvalidOperationException(
                $"Requested {deviceCount} CUDA devices, but only " +
                $"{Tensor.CudaDeviceCount} are available.");
        }

        string[] optimizers = ResolveOptimizers(optimizerSelector);
        TensorPrecisionMode[] precisions = ResolvePrecisions(
            precisionSelector);
        long elements = checked((long)parameterCount * rows * columns);
        Console.WriteLine("Optimizer precision benchmark");
        Console.WriteLine(
            $"device={(deviceCount == 0 ? "cpu" : $"cuda:{deviceCount}")}, " +
            $"parameters={parameterCount}, shape=[{rows},{columns}], " +
            $"elements={elements:N0}, warmup={warmup}, " +
            $"iterations={iterations}, gradient-publication=outside-timer");
        Console.WriteLine(
            "optimizer,precision,device,p50_ms,p95_ms,mean_ms," +
            "h2d_bytes,d2h_bytes,allocs,frees,physical_syncs");

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            foreach (string optimizer in optimizers)
            {
                foreach (TensorPrecisionMode precision in precisions)
                {
                    ScenarioResult result = RunScenario(
                        optimizer,
                        precision,
                        deviceCount,
                        warmup,
                        iterations,
                        parameterCount,
                        rows,
                        columns);
                    Print(optimizer, precision, deviceCount, result);
                }
            }
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    private static ScenarioResult RunScenario(
        string optimizerName,
        TensorPrecisionMode precisionMode,
        int deviceCount,
        int warmup,
        int iterations,
        int parameterCount,
        int rows,
        int columns)
    {
        int[] devices = Enumerable.Range(0, deviceCount).ToArray();
        Tensor.CudaDeviceIndices = deviceCount == 0 ? [0] : devices;
        Tensor.ExecutionDevice = deviceCount == 0
            ? TensorDevice.Cpu
            : TensorDevice.Cuda;
        using IDisposable precision = TensorExecutionContext
            .PushPrecisionPolicy(ResolvePolicy(precisionMode));
        Parameter[] parameters = CreateParameters(
            precisionMode, devices, parameterCount, rows, columns);
        IOptimizer optimizer = CreateOptimizer(optimizerName, parameters);
        float[][] gradients = CreateGradients(parameters, offset: 31);
        var gradientPublisher = new GradientPublisher(
            parameters, gradients, precisionMode, devices);
        try
        {
            optimizer.prepare();
            for (int iteration = 0; iteration < warmup; iteration++)
            {
                gradientPublisher.Publish();
                optimizer.step();
                optimizer.zero_grad();
            }

            long hostToDeviceCopies = 0;
            long hostToDeviceBytes = 0;
            long deviceToHostCopies = 0;
            long deviceToHostBytes = 0;
            long allocationCount = 0;
            long allocationBytes = 0;
            long freeCount = 0;
            long freeBytes = 0;
            long logicalBarriers = 0;
            long requestedSynchronizations = 0;
            long deferredBarriers = 0;
            long physicalSynchronizations = 0;
            long batchStarts = 0;
            long batchCompletions = 0;
            long failureDrains = 0;
            long clipBarriersElided = 0;
            var elapsed = new double[iterations];
            for (int iteration = 0; iteration < iterations; iteration++)
            {
                gradientPublisher.Publish();
                NativeCudaTransferTelemetry transferBefore =
                    NativeCudaRuntime.TransferTelemetry;
                NativeCudaAllocationTelemetry allocationBefore =
                    NativeCudaRuntime.AllocationTelemetry;
                CudaOptimizerSynchronizationTelemetrySnapshot syncBefore =
                    CudaOptimizerSynchronizationTelemetry.Snapshot;
                long started = Stopwatch.GetTimestamp();
                optimizer.step();
                elapsed[iteration] = Stopwatch.GetElapsedTime(started)
                    .TotalMilliseconds;
                NativeCudaTransferTelemetry iterationTransfers =
                    NativeCudaRuntime.TransferTelemetry - transferBefore;
                NativeCudaAllocationTelemetry iterationAllocations =
                    NativeCudaRuntime.AllocationTelemetry - allocationBefore;
                CudaOptimizerSynchronizationTelemetrySnapshot iterationSync =
                    CudaOptimizerSynchronizationTelemetry.Snapshot
                        - syncBefore;
                hostToDeviceCopies +=
                    iterationTransfers.HostToDeviceCopyCount;
                hostToDeviceBytes += iterationTransfers.HostToDeviceBytes;
                deviceToHostCopies +=
                    iterationTransfers.DeviceToHostCopyCount;
                deviceToHostBytes += iterationTransfers.DeviceToHostBytes;
                allocationCount += iterationAllocations.AllocationCount;
                allocationBytes += iterationAllocations.AllocationBytes;
                freeCount += iterationAllocations.FreeCount;
                freeBytes += iterationAllocations.FreeBytes;
                logicalBarriers += iterationSync.LogicalBarrierRequests;
                requestedSynchronizations +=
                    iterationSync.RequestedDeviceSynchronizations;
                deferredBarriers += iterationSync.DeferredBarrierRequests;
                physicalSynchronizations +=
                    iterationSync.PhysicalComputeStreamSynchronizations;
                batchStarts += iterationSync.BatchStarts;
                batchCompletions += iterationSync.BatchCompletions;
                failureDrains += iterationSync.FailureDrains;
                clipBarriersElided += iterationSync.ClipScaleBarriersElided;
                optimizer.zero_grad();
            }
            var transfers = new NativeCudaTransferTelemetry(
                hostToDeviceCopies,
                hostToDeviceBytes,
                deviceToHostCopies,
                deviceToHostBytes);
            var allocations = new NativeCudaAllocationTelemetry(
                allocationCount,
                allocationBytes,
                freeCount,
                freeBytes);
            var sync = new CudaOptimizerSynchronizationTelemetrySnapshot(
                logicalBarriers,
                requestedSynchronizations,
                deferredBarriers,
                physicalSynchronizations,
                batchStarts,
                batchCompletions,
                failureDrains,
                clipBarriersElided);
            Array.Sort(elapsed);
            return new ScenarioResult(
                elapsed.Average(),
                Percentile(elapsed, 0.50),
                Percentile(elapsed, 0.95),
                transfers,
                allocations,
                sync);
        }
        finally
        {
            DisposeOptimizer(optimizer);
            foreach (Parameter parameter in parameters)
                parameter.T.InvalidateCudaBuffers();
        }
    }

    private static Parameter[] CreateParameters(
        TensorPrecisionMode precisionMode,
        IReadOnlyList<int> devices,
        int parameterCount,
        int rows,
        int columns)
    {
        int length = checked(rows * columns);
        var parameters = new Parameter[parameterCount];
        for (int slot = 0; slot < parameterCount; slot++)
        {
            var parameter = new Parameter(
                Values(length, 7 + slot * 13, 0.08f),
                [rows, columns],
                $"matrix.{slot}",
                WeightDecayPolicy.Apply);
            switch (precisionMode)
            {
                case TensorPrecisionMode.BFloat16:
                    parameter.T.ConvertStorageInPlace(
                        TensorDType.BFloat16,
                        preserveFloat32Master: false);
                    break;
                case TensorPrecisionMode.Mix16_32:
                    parameter.T.ConvertStorageInPlace(
                        TensorDType.BFloat16,
                        preserveFloat32Master: true);
                    break;
                case TensorPrecisionMode.Bfp8:
                    parameter.T.ConvertStorageInPlace(
                        TensorDType.Bfp8,
                        Bfp8QuantizationDescriptor.TensorWide,
                        preserveFloat32Master: false);
                    break;
                case TensorPrecisionMode.Mix8_32:
                    parameter.T.ConvertStorageInPlace(
                        TensorDType.Bfp8,
                        Bfp8QuantizationDescriptor.Mix8_32,
                        preserveFloat32Master: true);
                    break;
            }

            foreach (int device in devices)
            {
                switch (precisionMode)
                {
                    case TensorPrecisionMode.BFloat16:
                    case TensorPrecisionMode.Mix16_32:
                        _ = parameter.T.EnsureCudaBFloat16Buffer(device);
                        break;
                    case TensorPrecisionMode.Bfp8:
                    case TensorPrecisionMode.Mix8_32:
                        _ = parameter.T.EnsureCudaBfp8Buffer(device);
                        break;
                    default:
                        _ = parameter.T.EnsureCudaFloat32Buffer(device);
                        break;
                }
            }
            if (devices.Count > 0)
                parameter.T.to(new TorchDevice(TensorDevice.Cuda, devices[0]));
            parameters[slot] = parameter;
        }
        return parameters;
    }

    private static IOptimizer CreateOptimizer(
        string name,
        Parameter[] parameters)
        => name switch
        {
            "adamw" => new AdamW(
                parameters,
                new AdamWOptions
                {
                    LearningRate = 3e-4f,
                    Beta1 = 0.9f,
                    Beta2 = 0.95f,
                    Epsilon = 1e-8f,
                    WeightDecay = 0.01f,
                }),
            "nekomuon" => new NekoMuon(
                parameters,
                new NekoMuonOptions
                {
                    LearningRate = 3e-4f,
                    WeightDecay = 0.01f,
                    MaxNewtonSchulzSteps = 5,
                    NewtonSchulzInterval = 1,
                    NewtonSchulzDepthMode =
                        NekoMuonNewtonSchulzDepthMode.Fixed,
                    NewtonSchulzDepth = 5f,
                }),
            "lion" => new Lion(
                parameters,
                new LionOptions
                {
                    LearningRate = 3e-4f,
                    Beta1 = 0.9f,
                    Beta2 = 0.99f,
                    WeightDecay = 0.01f,
                }),
            "gainshareadamw" => new GainShareAdamW(
                CreateGroups(parameters),
                new GainShareAdamWOptions
                {
                    LearningRate = 3e-4f,
                    Beta1 = 0.9f,
                    Beta2 = 0.95f,
                    Epsilon = 1e-8f,
                    Rho = 0.9f,
                    Gamma = 0.5f,
                    MinScale = 0.5f,
                    MaxScale = 2f,
                    WeightDecay = 0.01f,
                }),
            _ => throw new ArgumentOutOfRangeException(nameof(name)),
        };

    private static IEnumerable<IEnumerable<Parameter>> CreateGroups(
        Parameter[] parameters)
    {
        int groupSize = Math.Max(1, (parameters.Length + 3) / 4);
        return parameters.Chunk(groupSize)
            .Select(group => (IEnumerable<Parameter>)group);
    }

    private static float[][] CreateGradients(
        IReadOnlyList<Parameter> parameters,
        int offset)
        => parameters
            .Select((parameter, index) => Values(
                parameter.T.Numel,
                offset + index * 17,
                0.025f))
            .ToArray();

    private static void DisposeOptimizer(IOptimizer optimizer)
    {
        switch (optimizer)
        {
            case AdamW adamW:
                adamW.DisposeCudaResources();
                break;
            case NekoMuon nekoMuon:
                nekoMuon.DisposeCudaResources();
                break;
            case IDisposable disposable:
                disposable.Dispose();
                break;
        }
    }

    private static PrecisionPolicy ResolvePolicy(TensorPrecisionMode mode)
        => mode switch
        {
            TensorPrecisionMode.Float32 => PrecisionPolicy.Float32,
            TensorPrecisionMode.BFloat16 => PrecisionPolicy.BFloat16,
            TensorPrecisionMode.Mix16_32 => PrecisionPolicy.Mix16_32,
            TensorPrecisionMode.Bfp8 => PrecisionPolicy.Bfp8,
            TensorPrecisionMode.Mix8_32 => PrecisionPolicy.Mix8_32,
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };

    private static string[] ResolveOptimizers(string selector)
    {
        if (string.Equals(selector, "all", StringComparison.OrdinalIgnoreCase))
            return OptimizerNames;
        string normalized = selector.ToLowerInvariant();
        if (!OptimizerNames.Contains(normalized, StringComparer.Ordinal))
        {
            throw new ArgumentException(
                $"Unknown optimizer '{selector}'. Use all, " +
                $"{string.Join(", ", OptimizerNames)}.");
        }
        return [normalized];
    }

    private static TensorPrecisionMode[] ResolvePrecisions(string selector)
        => string.Equals(selector, "all", StringComparison.OrdinalIgnoreCase)
            ? PrecisionModes
            : [TensorPrecisionModeNames.Parse(selector)];

    private static float[] Values(int length, int offset, float scale)
        => Enumerable.Range(0, length)
            .Select(index => MathF.Sin((index + offset) * 0.173f) * scale)
            .ToArray();

    private static double Percentile(double[] sorted, double percentile)
    {
        int index = Math.Clamp(
            (int)Math.Ceiling(percentile * sorted.Length) - 1,
            0,
            sorted.Length - 1);
        return sorted[index];
    }

    private static void Print(
        string optimizer,
        TensorPrecisionMode precision,
        int deviceCount,
        ScenarioResult result)
        => Console.WriteLine(
            $"{optimizer},{TensorPrecisionModeNames.Format(precision)}," +
            $"{(deviceCount == 0 ? "cpu" : $"cuda:{deviceCount}")}," +
            $"{result.P50Milliseconds:F3}," +
            $"{result.P95Milliseconds:F3}," +
            $"{result.MeanMilliseconds:F3}," +
            $"{result.Transfers.HostToDeviceBytes}," +
            $"{result.Transfers.DeviceToHostBytes}," +
            $"{result.Allocations.AllocationCount}," +
            $"{result.Allocations.FreeCount}," +
            $"{result.Synchronization.PhysicalComputeStreamSynchronizations}");

    /// <summary>
    /// Owns stable logical gradient bindings for the duration of a scenario.
    /// Pure BF16 reuses the same native addresses after zero_grad so an
    /// optimizer descriptor plan is not rebuilt by the benchmark harness.
    /// </summary>
    private sealed class GradientPublisher
    {
        private readonly Parameter[] _parameters;
        private readonly float[][] _gradients;
        private readonly TensorPrecisionMode _mode;
        private readonly int[] _devices;
        private readonly NativeCudaBuffer<ushort>?[][] _bfloat16;

        internal GradientPublisher(
            Parameter[] parameters,
            float[][] gradients,
            TensorPrecisionMode mode,
            int[] devices)
        {
            _parameters = parameters;
            _gradients = gradients;
            _mode = mode;
            _devices = devices;
            _bfloat16 = parameters
                .Select(_ => new NativeCudaBuffer<ushort>?[devices.Length])
                .ToArray();
        }

        internal void Publish()
        {
            if (_devices.Length == 0)
            {
                for (int index = 0; index < _parameters.Length; index++)
                {
                    _gradients[index].AsSpan().CopyTo(
                        _parameters[index].T.MutableGrad);
                }
                return;
            }

            CudaGradientReductionStamp reductionStamp =
                CudaGradientReductionStampSource.CreateStandalone();
            for (int index = 0; index < _parameters.Length; index++)
                Publish(index, reductionStamp);
        }

        private void Publish(
            int index,
            CudaGradientReductionStamp reductionStamp)
        {
            Parameter parameter = _parameters[index];
            float[] values = _gradients[index];
            switch (_mode)
            {
                case TensorPrecisionMode.BFloat16:
                    PublishBFloat16(
                        index, parameter, values, reductionStamp);
                    break;
                case TensorPrecisionMode.Bfp8:
                    PublishBfp8(parameter, values, reductionStamp);
                    break;
                default:
                    foreach (int device in _devices)
                        parameter.T.SetCudaGradient(values, device);
                    parameter.T.MarkCudaGradientsSynchronized(
                        _devices, reductionStamp);
                    break;
            }
        }

        private void PublishBFloat16(
            int parameterIndex,
            Parameter parameter,
            float[] values,
            CudaGradientReductionStamp reductionStamp)
        {
            var encoded = new ushort[values.Length];
            TensorStorageCodec.EncodeBFloat16(values, encoded);
            for (int deviceSlot = 0;
                deviceSlot < _devices.Length;
                deviceSlot++)
            {
                int device = _devices[deviceSlot];
                NativeCudaBuffer<ushort>? buffer =
                    _bfloat16[parameterIndex][deviceSlot];
                if (buffer is null)
                {
                    buffer = ForgetMemoryV2Cuda.GetAccelerator(device)
                        .Allocate1D(encoded);
                    parameter.T.AdoptCudaBFloat16GradientBuffer(
                        buffer, device);
                    _bfloat16[parameterIndex][deviceSlot] = buffer;
                }
                else
                {
                    buffer.CopyFromCPU(encoded);
                    parameter.T.MarkCudaBFloat16GradientMutated(device);
                }
            }
            parameter.T.MarkCudaBFloat16GradientsSynchronized(
                _devices, reductionStamp);
        }

        private void PublishBfp8(
            Parameter parameter,
            float[] values,
            CudaGradientReductionStamp reductionStamp)
        {
            Bfp8EncodedStorage encoded = Bfp8QuantizationCodec.Default.Encode(
                values,
                Bfp8QuantizationDescriptor.TensorWide);
            foreach (int device in _devices)
            {
                CudaBfp8BufferView target = parameter.T
                    .PrepareCudaBfp8GradientReplica(device);
                target.Payload.CopyFromCPU(encoded.Payload.Span);
                target.Scales.CopyFromCPU(encoded.Scales.Span);
            }
            parameter.T.MarkCudaBfp8GradientsSynchronized(
                _devices, reductionStamp);
        }
    }

    private sealed record ScenarioResult(
        double MeanMilliseconds,
        double P50Milliseconds,
        double P95Milliseconds,
        NativeCudaTransferTelemetry Transfers,
        NativeCudaAllocationTelemetry Allocations,
        CudaOptimizerSynchronizationTelemetrySnapshot Synchronization);
}
