using System.Runtime.InteropServices;
using NNtrain.Cuda.Interop;
using NNtrain.Runtime.Execution;

namespace NNtrain;

/// <summary>
/// Owns Lion's persistent CUDA state and immutable multi-tensor descriptor
/// plans. Managed momentum arrays are checkpoint shadows only while this
/// object has a resident state; hot steps never synchronize them.
/// </summary>
internal sealed class CudaLionOptimizer : IDisposable
{
    private const int ChunkElements = 4096;

    private readonly IReadOnlyList<Parameter> _parameters;
    private readonly LionParameterState[] _hostStates;
    private readonly ParameterRuntime[] _runtime;
    private readonly Dictionary<int, MultiTensorPlan> _plans = [];
    private readonly Dictionary<int, NativeCudaBuffer<int>> _finiteStatus = [];
    private readonly Dictionary<int, CudaOptimizerFiniteStatusReadback>
        _finiteReadbacks = [];
    private readonly object _sync = new();
    private LionCudaPrecision? _precision;
    private int _disposed;

    internal CudaLionOptimizer(
        IReadOnlyList<Parameter> parameters,
        LionParameterState[] hostStates)
    {
        _parameters = parameters
            ?? throw new ArgumentNullException(nameof(parameters));
        _hostStates = hostStates
            ?? throw new ArgumentNullException(nameof(hostStates));
        if (parameters.Count != hostStates.Length)
        {
            throw new ArgumentException(
                "Lion CUDA state count must match the parameter count.",
                nameof(hostStates));
        }
        _runtime = new ParameterRuntime[parameters.Count];
        for (int index = 0; index < _runtime.Length; index++)
        {
            _runtime[index] = new ParameterRuntime(
                parameters[index], hostStates[index].Momentum);
        }
    }

    internal void Prepare(IReadOnlyList<int> deviceIndices)
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
        int[] devices = NormalizeDevices(deviceIndices);
        lock (_sync)
        {
            if (_runtime.Length == 0)
            {
                _precision = LionCudaPrecision.Float32;
                return;
            }
            LionCudaPrecision precision = ResolvePrecision();
            if (_precision is { } previous && previous != precision)
                ResetResidentResources(devices[0]);
            _precision = precision;

            foreach (ParameterRuntime runtime in _runtime)
            {
                foreach (int deviceIndex in devices)
                {
                    PrepareParameter(runtime, precision, deviceIndex);
                }
            }

            if (precision is LionCudaPrecision.Bfp8
                or LionCudaPrecision.Mix8)
            {
                foreach (int deviceIndex in devices)
                {
                    _ = GetOrCreateFiniteStatus(deviceIndex);
                    _ = GetOrCreateFiniteReadback(deviceIndex);
                }
            }

            // Pure BF16 gradients are normally reducer-owned arena slices and
            // may not exist at the public pre-gradient prepare hook. All
            // other formats have stable optimizer-visible storage here.
            bool gradientsReady = precision switch
            {
                LionCudaPrecision.BFloat16 => _runtime.All(runtime =>
                    devices.All(deviceIndex => runtime.Parameter.T
                        .TryGetCudaBFloat16GradientBuffer(
                            deviceIndex,
                            out _))),
                LionCudaPrecision.Bfp8 => _runtime.All(runtime =>
                    devices.All(deviceIndex => runtime.Parameter.T
                        .TryGetCudaBfp8GradientBuffer(
                            deviceIndex,
                            out _))),
                _ => true,
            };
            if (gradientsReady)
            {
                PreparePlans(devices, precision);
            }
        }
    }

    internal void Step(
        IReadOnlyList<int> deviceIndices,
        LionOptions options,
        int step)
    {
        ArgumentNullException.ThrowIfNull(options);
        int[] devices = NormalizeDevices(deviceIndices);
        Prepare(devices);
        if (_runtime.Length == 0)
            return;

        LionCudaPrecision precision;
        MultiTensorPlan[] plans;
        NativeCudaBuffer<int>[]? statuses = null;
        CudaOptimizerFiniteStatusReadback[]? readbacks = null;
        lock (_sync)
        {
            precision = _precision
                ?? throw new InvalidOperationException(
                    "Lion CUDA precision was not prepared.");
            PreparePlans(devices, precision);
            plans = new MultiTensorPlan[devices.Length];
            for (int slot = 0; slot < devices.Length; slot++)
                plans[slot] = _plans[devices[slot]];

            if (precision is LionCudaPrecision.Bfp8
                or LionCudaPrecision.Mix8)
            {
                statuses = new NativeCudaBuffer<int>[devices.Length];
                readbacks = new CudaOptimizerFiniteStatusReadback[
                    devices.Length];
                for (int slot = 0; slot < devices.Length; slot++)
                {
                    statuses[slot] = GetOrCreateFiniteStatus(devices[slot]);
                    readbacks[slot] = GetOrCreateFiniteReadback(devices[slot]);
                    statuses[slot].MemSetToZero();
                }
            }
        }

        NativeCudaBuffer<int>[]? launchStatuses = statuses;
        Parallel.For(0, devices.Length, slot =>
        {
            plans[slot].Execute(
                options,
                launchStatuses is null ? null : launchStatuses[slot]);
        });

        NativeCudaBuffer<int>[]? completionStatuses = statuses;
        CudaOptimizerFiniteStatusReadback[]? completionReadbacks = readbacks;
        CudaOptimizerStepBatch.CompleteAfterSynchronization(
            devices,
            $"Lion {FormatPrecision(precision)} update",
            queueReadback: completionStatuses is null
                ? null
                : () =>
                {
                    for (int slot = 0; slot < devices.Length; slot++)
                    {
                        completionReadbacks![slot].Begin(
                            completionStatuses[slot]);
                    }
                },
            finalize: () =>
            {
                if (completionReadbacks is not null)
                {
                    for (int slot = 0; slot < devices.Length; slot++)
                    {
                        if (completionReadbacks[slot]
                                .ReadAfterSynchronization() != 0)
                        {
                            throw new InvalidOperationException(
                                $"Non-finite CUDA value detected in pure " +
                                $"BFP8 Lion state on device " +
                                $"{devices[slot]} at optimizer step " +
                                $"{step}.");
                        }
                    }
                }

                foreach (Parameter parameter in _parameters)
                {
                    if (precision is LionCudaPrecision.Bfp8
                        or LionCudaPrecision.Mix8)
                    {
                        parameter.T.MarkCudaBfp8DataReplicasSynchronized(
                            devices);
                    }
                    else
                    {
                        parameter.T.MarkCudaDataReplicasSynchronized(devices);
                    }
                }
            });
    }

    internal void SynchronizeHost(int primaryDevice)
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
        lock (_sync)
        {
            if (_precision is null)
                return;
            foreach (ParameterRuntime runtime in _runtime)
                runtime.SynchronizeHost(primaryDevice, _precision.Value);
        }
    }

    internal LionCudaPrecision? ResidentPrecision => _precision;

    private void PreparePlans(
        IReadOnlyList<int> devices,
        LionCudaPrecision precision)
    {
        foreach (int deviceIndex in devices)
        {
            if (_plans.TryGetValue(
                    deviceIndex,
                    out MultiTensorPlan? plan)
                && plan.Precision == precision
                && plan.HasCurrentGradientBindings())
            {
                continue;
            }

            plan?.Dispose();
            _plans.Remove(deviceIndex);
            _plans.Add(
                deviceIndex,
                MultiTensorPlan.Create(
                    deviceIndex,
                    precision,
                    _runtime,
                    ChunkElements));
        }
    }

    private static void PrepareParameter(
        ParameterRuntime runtime,
        LionCudaPrecision precision,
        int deviceIndex)
    {
        Tensor parameter = runtime.Parameter.T;
        switch (precision)
        {
            case LionCudaPrecision.Float32:
                if (parameter.DType == TensorDType.Float32)
                    _ = parameter.EnsureCudaFloat32Buffer(deviceIndex);
                else
                {
                    _ = parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
                    _ = parameter.EnsureCudaBFloat16Buffer(deviceIndex);
                }
                _ = parameter.EnsureCudaGradientBuffer(deviceIndex);
                _ = runtime.GetOrCreateFloat32(deviceIndex);
                break;
            case LionCudaPrecision.BFloat16:
                _ = parameter.EnsureCudaBFloat16Buffer(deviceIndex);
                _ = runtime.GetOrCreateBFloat16(deviceIndex);
                break;
            case LionCudaPrecision.Mix8:
                _ = parameter.EnsureCudaBfp8Buffer(deviceIndex);
                _ = parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
                _ = parameter.EnsureCudaGradientBuffer(deviceIndex);
                _ = runtime.GetOrCreateFloat32(deviceIndex);
                break;
            case LionCudaPrecision.Bfp8:
                _ = parameter.EnsureCudaBfp8Buffer(deviceIndex);
                _ = parameter.PrepareCudaBfp8GradientReplica(deviceIndex);
                _ = runtime.GetOrCreateBfp8(deviceIndex);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(precision));
        }
    }

    private LionCudaPrecision ResolvePrecision()
    {
        bool allBfp8 = _runtime.All(runtime =>
            runtime.Parameter.T.DType == TensorDType.Bfp8);
        if (allBfp8)
        {
            bool allTensorWide = _runtime.All(runtime =>
                runtime.Parameter.T.Bfp8Quantization
                    == Bfp8QuantizationDescriptor.TensorWide);
            bool allBlock = _runtime.All(runtime =>
                runtime.Parameter.T.Bfp8Quantization?.Granularity
                    == Bfp8ScaleGranularity.Block);
            if (allTensorWide)
                return LionCudaPrecision.Bfp8;
            if (allBlock)
                return LionCudaPrecision.Mix8;
            throw new InvalidOperationException(
                "Lion CUDA parameters cannot mix tensor-wide and block " +
                "BFP8 scale contracts.");
        }
        if (_runtime.Any(runtime =>
                runtime.Parameter.T.DType == TensorDType.Bfp8))
        {
            throw new InvalidOperationException(
                "Lion CUDA parameters must use one physical precision.");
        }

        bool allBFloat16 = _runtime.All(runtime =>
            runtime.Parameter.T.DType == TensorDType.BFloat16);
        bool pureBFloat16 = allBFloat16
            && TensorExecutionContext.ActivePrecisionPolicy?.OptimizerState
                == NumericFormat.BFloat16;
        if (pureBFloat16)
            return LionCudaPrecision.BFloat16;

        bool standard = _runtime.All(runtime =>
            runtime.Parameter.T.DType is TensorDType.Float32
                or TensorDType.BFloat16);
        if (!standard)
        {
            throw new InvalidOperationException(
                "Lion CUDA encountered an unsupported parameter dtype.");
        }
        return LionCudaPrecision.Float32;
    }

    private void ResetResidentResources(int primaryDevice)
    {
        if (_precision is { } precision)
        {
            foreach (ParameterRuntime runtime in _runtime)
                runtime.SynchronizeHost(primaryDevice, precision);
        }
        DisposePlans();
        foreach (ParameterRuntime runtime in _runtime)
            runtime.DisposeDeviceBuffers();
        DisposeFiniteResources();
    }

    private NativeCudaBuffer<int> GetOrCreateFiniteStatus(int deviceIndex)
    {
        if (_finiteStatus.TryGetValue(
                deviceIndex,
                out NativeCudaBuffer<int>? status))
        {
            return status;
        }
        status = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
            .Allocate1D<int>(1);
        _finiteStatus.Add(deviceIndex, status);
        return status;
    }

    private CudaOptimizerFiniteStatusReadback GetOrCreateFiniteReadback(
        int deviceIndex)
    {
        if (_finiteReadbacks.TryGetValue(
                deviceIndex,
                out CudaOptimizerFiniteStatusReadback? readback))
        {
            return readback;
        }
        readback = new CudaOptimizerFiniteStatusReadback(deviceIndex);
        _finiteReadbacks.Add(deviceIndex, readback);
        return readback;
    }

    private static int[] NormalizeDevices(IReadOnlyList<int> deviceIndices)
    {
        ArgumentNullException.ThrowIfNull(deviceIndices);
        int[] devices = deviceIndices.Distinct().ToArray();
        if (devices.Length == 0 || devices.Any(device => device < 0))
        {
            throw new ArgumentException(
                "Lion CUDA requires at least one unique non-negative device.",
                nameof(deviceIndices));
        }
        return devices;
    }

    private static string FormatPrecision(LionCudaPrecision precision)
        => precision switch
        {
            LionCudaPrecision.Float32 => "float32/mix16_32",
            LionCudaPrecision.BFloat16 => "bfloat16",
            LionCudaPrecision.Mix8 => "mix8_32",
            LionCudaPrecision.Bfp8 => "bfp8",
            _ => precision.ToString(),
        };

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        List<Exception>? failures = null;
        TryDispose(DisposePlans, ref failures);
        foreach (ParameterRuntime runtime in _runtime)
            TryDispose(runtime.DisposeDeviceBuffers, ref failures);
        TryDispose(DisposeFiniteResources, ref failures);
        if (failures is not null)
        {
            throw new AggregateException(
                "Lion CUDA resource cleanup failed.", failures);
        }
    }

    private void DisposePlans()
    {
        List<Exception>? failures = null;
        foreach (MultiTensorPlan plan in _plans.Values)
            TryDispose(plan.Dispose, ref failures);
        _plans.Clear();
        if (failures is not null)
            throw new AggregateException("Lion plan cleanup failed.", failures);
    }

    private void DisposeFiniteResources()
    {
        List<Exception>? failures = null;
        foreach (CudaOptimizerFiniteStatusReadback readback
            in _finiteReadbacks.Values)
        {
            TryDispose(readback.Dispose, ref failures);
        }
        _finiteReadbacks.Clear();
        foreach (NativeCudaBuffer<int> status in _finiteStatus.Values)
            TryDispose(status.Dispose, ref failures);
        _finiteStatus.Clear();
        if (failures is not null)
        {
            throw new AggregateException(
                "Lion finite-status cleanup failed.", failures);
        }
    }

    private static void TryDispose(
        Action action,
        ref List<Exception>? failures)
    {
        try
        {
            action();
        }
        catch (AggregateException aggregate)
        {
            (failures ??= []).AddRange(aggregate.Flatten().InnerExceptions);
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
    }

    internal enum LionCudaPrecision
    {
        Float32,
        BFloat16,
        Mix8,
        Bfp8,
    }

    private sealed class ParameterRuntime(
        Parameter parameter,
        float[] hostMomentum)
    {
        private readonly Dictionary<int, NativeCudaBuffer<float>> _float = [];
        private readonly Dictionary<int, NativeCudaBuffer<ushort>> _bf16 = [];
        private readonly Dictionary<int, Bfp8StateBuffer> _bfp8 = [];

        internal Parameter Parameter { get; } = parameter;
        internal float[] HostMomentum { get; } = hostMomentum;

        internal NativeCudaBuffer<float> GetOrCreateFloat32(int deviceIndex)
        {
            if (_float.TryGetValue(
                    deviceIndex,
                    out NativeCudaBuffer<float>? buffer))
            {
                return buffer;
            }
            buffer = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
                .Allocate1D(HostMomentum);
            _float.Add(deviceIndex, buffer);
            return buffer;
        }

        internal NativeCudaBuffer<ushort> GetOrCreateBFloat16(int deviceIndex)
        {
            if (_bf16.TryGetValue(
                    deviceIndex,
                    out NativeCudaBuffer<ushort>? buffer))
            {
                return buffer;
            }
            var encoded = new ushort[HostMomentum.Length];
            TensorStorageCodec.EncodeBFloat16(HostMomentum, encoded);
            buffer = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
                .Allocate1D(encoded);
            _bf16.Add(deviceIndex, buffer);
            return buffer;
        }

        internal Bfp8StateBuffer GetOrCreateBfp8(int deviceIndex)
        {
            if (_bfp8.TryGetValue(
                    deviceIndex,
                    out Bfp8StateBuffer? buffer))
            {
                return buffer;
            }
            Bfp8EncodedStorage encoded = Bfp8QuantizationCodec.Default.Encode(
                HostMomentum,
                Bfp8QuantizationDescriptor.TensorWide);
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            NativeCudaBuffer<sbyte>? payload = null;
            NativeCudaBuffer<float>? scales = null;
            try
            {
                payload = accelerator.Allocate1D(encoded.Payload.Span);
                scales = accelerator.Allocate1D(encoded.Scales.Span);
                buffer = new Bfp8StateBuffer(payload, scales);
                _bfp8.Add(deviceIndex, buffer);
                return buffer;
            }
            catch
            {
                payload?.Dispose();
                scales?.Dispose();
                throw;
            }
        }

        internal void SynchronizeHost(
            int deviceIndex,
            LionCudaPrecision precision)
        {
            switch (precision)
            {
                case LionCudaPrecision.Float32:
                case LionCudaPrecision.Mix8:
                    if (_float.TryGetValue(
                            deviceIndex,
                            out NativeCudaBuffer<float>? floatState))
                    {
                        floatState.CopyToCPU(HostMomentum);
                    }
                    break;
                case LionCudaPrecision.BFloat16:
                    if (_bf16.TryGetValue(
                            deviceIndex,
                            out NativeCudaBuffer<ushort>? bf16State))
                    {
                        var encoded = new ushort[HostMomentum.Length];
                        bf16State.CopyToCPU(encoded);
                        TensorStorageCodec.DecodeBFloat16(
                            encoded,
                            HostMomentum);
                    }
                    break;
                case LionCudaPrecision.Bfp8:
                    if (_bfp8.TryGetValue(
                            deviceIndex,
                            out Bfp8StateBuffer? bfp8State))
                    {
                        var payload = new sbyte[HostMomentum.Length];
                        var scales = new float[1];
                        bfp8State.Payload.CopyToCPU(payload);
                        bfp8State.Scales.CopyToCPU(scales);
                        Bfp8QuantizationCodec.Default.Decode(
                            payload,
                            scales,
                            Bfp8QuantizationDescriptor.TensorWide,
                            HostMomentum);
                    }
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(precision));
            }
        }

        internal void DisposeDeviceBuffers()
        {
            List<Exception>? failures = null;
            foreach (NativeCudaBuffer<float> buffer in _float.Values)
                TryDispose(buffer.Dispose, ref failures);
            _float.Clear();
            foreach (NativeCudaBuffer<ushort> buffer in _bf16.Values)
                TryDispose(buffer.Dispose, ref failures);
            _bf16.Clear();
            foreach (Bfp8StateBuffer buffer in _bfp8.Values)
                TryDispose(buffer.Dispose, ref failures);
            _bfp8.Clear();
            if (failures is not null)
            {
                throw new AggregateException(
                    "Lion parameter-state cleanup failed.", failures);
            }
        }
    }

    private sealed class Bfp8StateBuffer(
        NativeCudaBuffer<sbyte> payload,
        NativeCudaBuffer<float> scales) : IDisposable
    {
        internal NativeCudaBuffer<sbyte> Payload { get; } = payload;
        internal NativeCudaBuffer<float> Scales { get; } = scales;

        public void Dispose()
        {
            List<Exception>? failures = null;
            TryDispose(Payload.Dispose, ref failures);
            TryDispose(Scales.Dispose, ref failures);
            if (failures is not null)
                throw new AggregateException("BFP8 state cleanup failed.", failures);
        }
    }

    private sealed class MultiTensorPlan : IDisposable
    {
        private readonly int _deviceIndex;
        private readonly ParameterRuntime[] _runtime;
        private readonly nint[] _gradientPointers;
        private readonly NativeCudaBuffer<FloatChunk>? _floatChunks;
        private readonly NativeCudaBuffer<BFloat16Chunk>? _bf16Chunks;
        private readonly NativeCudaBuffer<Mix8Block>? _mix8Blocks;
        private readonly NativeCudaBuffer<Bfp8Tensor>? _bfp8Tensors;
        private readonly NativeCudaBuffer<float>? _bfp8Reduction;
        private readonly int _bfp8MaximumChunks;

        private MultiTensorPlan(
            int deviceIndex,
            LionCudaPrecision precision,
            ParameterRuntime[] runtime,
            nint[] gradientPointers,
            NativeCudaBuffer<FloatChunk>? floatChunks = null,
            NativeCudaBuffer<BFloat16Chunk>? bf16Chunks = null,
            NativeCudaBuffer<Mix8Block>? mix8Blocks = null,
            NativeCudaBuffer<Bfp8Tensor>? bfp8Tensors = null,
            NativeCudaBuffer<float>? bfp8Reduction = null,
            int bfp8MaximumChunks = 0)
        {
            _deviceIndex = deviceIndex;
            Precision = precision;
            _runtime = runtime;
            _gradientPointers = gradientPointers;
            _floatChunks = floatChunks;
            _bf16Chunks = bf16Chunks;
            _mix8Blocks = mix8Blocks;
            _bfp8Tensors = bfp8Tensors;
            _bfp8Reduction = bfp8Reduction;
            _bfp8MaximumChunks = bfp8MaximumChunks;
        }

        internal LionCudaPrecision Precision { get; }

        internal static MultiTensorPlan Create(
            int deviceIndex,
            LionCudaPrecision precision,
            ParameterRuntime[] runtime,
            int chunkElements)
        {
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            var gradients = new nint[runtime.Length];
            switch (precision)
            {
                case LionCudaPrecision.Float32:
                {
                    var chunks = new List<FloatChunk>();
                    for (int slot = 0; slot < runtime.Length; slot++)
                    {
                        ParameterRuntime item = runtime[slot];
                        Tensor parameter = item.Parameter.T;
                        NativeCudaBuffer<float> gradient =
                            parameter.EnsureCudaGradientBuffer(deviceIndex);
                        NativeCudaBuffer<float> data =
                            parameter.DType == TensorDType.Float32
                                ? parameter.EnsureCudaFloat32Buffer(deviceIndex)
                                : parameter.EnsureCudaMasterFloat32Buffer(
                                    deviceIndex);
                        NativeCudaBuffer<ushort>? physical =
                            parameter.DType == TensorDType.BFloat16
                                ? parameter.EnsureCudaBFloat16Buffer(deviceIndex)
                                : null;
                        NativeCudaBuffer<float> momentum =
                            item.GetOrCreateFloat32(deviceIndex);
                        gradients[slot] = gradient.NativePtr;
                        AppendFloatChunks(
                            chunks,
                            data.NativePtr,
                            gradient.NativePtr,
                            momentum.NativePtr,
                            physical?.NativePtr ?? 0,
                            parameter.Numel,
                            chunkElements,
                            AppliesWeightDecay(item.Parameter),
                            physical is not null,
                            parameter.Rank == 1);
                    }
                    return new MultiTensorPlan(
                        deviceIndex,
                        precision,
                        runtime,
                        gradients,
                        floatChunks: accelerator.Allocate1D(
                            CollectionsMarshal.AsSpan(chunks)));
                }
                case LionCudaPrecision.BFloat16:
                {
                    var chunks = new List<BFloat16Chunk>();
                    for (int slot = 0; slot < runtime.Length; slot++)
                    {
                        ParameterRuntime item = runtime[slot];
                        Tensor parameter = item.Parameter.T;
                        if (!parameter.TryGetCudaBFloat16GradientBuffer(
                                deviceIndex,
                                out NativeCudaBuffer<ushort>? gradient))
                        {
                            throw new InvalidOperationException(
                                $"Pure BF16 Lion requires a resident BF16 " +
                                $"gradient for '{parameter.Name}' on CUDA " +
                                $"device {deviceIndex}.");
                        }
                        NativeCudaBuffer<ushort> data =
                            parameter.EnsureCudaBFloat16Buffer(deviceIndex);
                        NativeCudaBuffer<ushort> momentum =
                            item.GetOrCreateBFloat16(deviceIndex);
                        gradients[slot] = gradient!.NativePtr;
                        AppendBFloat16Chunks(
                            chunks,
                            data.NativePtr,
                            gradient.NativePtr,
                            momentum.NativePtr,
                            parameter.Numel,
                            chunkElements,
                            AppliesWeightDecay(item.Parameter),
                            parameter.Rank == 1);
                    }
                    return new MultiTensorPlan(
                        deviceIndex,
                        precision,
                        runtime,
                        gradients,
                        bf16Chunks: accelerator.Allocate1D(
                            CollectionsMarshal.AsSpan(chunks)));
                }
                case LionCudaPrecision.Mix8:
                {
                    var blocks = new List<Mix8Block>();
                    for (int slot = 0; slot < runtime.Length; slot++)
                    {
                        ParameterRuntime item = runtime[slot];
                        Tensor parameter = item.Parameter.T;
                        CudaBfp8BufferView encoded =
                            parameter.EnsureCudaBfp8Buffer(deviceIndex);
                        NativeCudaBuffer<float> master =
                            parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
                        NativeCudaBuffer<float> gradient =
                            parameter.EnsureCudaGradientBuffer(deviceIndex);
                        NativeCudaBuffer<float> momentum =
                            item.GetOrCreateFloat32(deviceIndex);
                        gradients[slot] = gradient.NativePtr;
                        int blockSize = encoded.Descriptor.BlockSize;
                        int scaleCount = encoded.Descriptor.GetScaleCount(
                            parameter.Numel);
                        for (int block = 0; block < scaleCount; block++)
                        {
                            int offset = checked(block * blockSize);
                            int length = Math.Min(
                                blockSize,
                                parameter.Numel - offset);
                            blocks.Add(new Mix8Block(
                                master.NativePtr,
                                gradient.NativePtr,
                                momentum.NativePtr,
                                encoded.Payload.NativePtr,
                                encoded.Scales.NativePtr,
                                offset,
                                length,
                                block,
                                AppliesWeightDecay(item.Parameter) ? 1 : 0,
                                parameter.Rank == 1 ? 1 : 0));
                        }
                    }
                    return new MultiTensorPlan(
                        deviceIndex,
                        precision,
                        runtime,
                        gradients,
                        mix8Blocks: accelerator.Allocate1D(
                            CollectionsMarshal.AsSpan(blocks)));
                }
                case LionCudaPrecision.Bfp8:
                {
                    var tensors = new Bfp8Tensor[runtime.Length];
                    for (int slot = 0; slot < runtime.Length; slot++)
                    {
                        ParameterRuntime item = runtime[slot];
                        Tensor parameter = item.Parameter.T;
                        CudaBfp8BufferView data =
                            parameter.EnsureCudaBfp8Buffer(deviceIndex);
                        if (!parameter.TryGetCudaBfp8GradientBuffer(
                                deviceIndex,
                                out CudaBfp8BufferView gradient))
                        {
                            throw new InvalidOperationException(
                                $"Pure BFP8 Lion requires an authoritative " +
                                $"BFP8 gradient for '{parameter.Name}' on " +
                                $"CUDA device {deviceIndex}.");
                        }
                        Bfp8StateBuffer momentum =
                            item.GetOrCreateBfp8(deviceIndex);
                        gradients[slot] = gradient.Payload.NativePtr;
                        tensors[slot] = new Bfp8Tensor(
                            data.Payload.NativePtr,
                            data.Scales.NativePtr,
                            gradient.Payload.NativePtr,
                            gradient.Scales.NativePtr,
                            momentum.Payload.NativePtr,
                            momentum.Scales.NativePtr,
                            parameter.Numel,
                            AppliesWeightDecay(item.Parameter) ? 1 : 0,
                            parameter.Rank == 1 ? 1 : 0);
                    }
                    NativeCudaBuffer<Bfp8Tensor>? tensorBuffer = null;
                    NativeCudaBuffer<float>? reduction = null;
                    try
                    {
                        tensorBuffer = accelerator.Allocate1D(tensors);
                        reduction = accelerator.Allocate1D<float>(
                            checked(runtime.Length * 4));
                        int maximumChunks = runtime.Max(item =>
                            checked((item.Parameter.T.Numel
                                + chunkElements - 1) / chunkElements));
                        return new MultiTensorPlan(
                            deviceIndex,
                            precision,
                            runtime,
                            gradients,
                            bfp8Tensors: tensorBuffer,
                            bfp8Reduction: reduction,
                            bfp8MaximumChunks: maximumChunks);
                    }
                    catch
                    {
                        tensorBuffer?.Dispose();
                        reduction?.Dispose();
                        throw;
                    }
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(precision));
            }
        }

        internal bool HasCurrentGradientBindings()
        {
            for (int slot = 0; slot < _runtime.Length; slot++)
            {
                Tensor parameter = _runtime[slot].Parameter.T;
                nint current;
                switch (Precision)
                {
                    case LionCudaPrecision.BFloat16:
                        if (!parameter.TryGetCudaBFloat16GradientBuffer(
                                _deviceIndex,
                                out NativeCudaBuffer<ushort>? bf16))
                        {
                            return false;
                        }
                        current = bf16!.NativePtr;
                        break;
                    case LionCudaPrecision.Bfp8:
                        if (!parameter.TryGetCudaBfp8GradientBuffer(
                                _deviceIndex,
                                out CudaBfp8BufferView bfp8))
                        {
                            return false;
                        }
                        current = bfp8.Payload.NativePtr;
                        break;
                    default:
                        current = parameter.EnsureCudaGradientBuffer(
                            _deviceIndex).NativePtr;
                        break;
                }
                if (current != _gradientPointers[slot])
                    return false;
            }
            return true;
        }

        internal void Execute(
            LionOptions options,
            NativeCudaBuffer<int>? finiteStatus)
        {
            NativeCudaRuntime.BindDeviceAndComputeStream(_deviceIndex);
            nint stream = ForgetMemoryV2Cuda.GetAccelerator(_deviceIndex)
                .DefaultStream;
            int status = Precision switch
            {
                LionCudaPrecision.Float32 =>
                    CudaNativeGateway.LionMultiTensorFloat32(
                        _deviceIndex,
                        _floatChunks!.NativePtr,
                        _floatChunks.Length,
                        options.Beta1,
                        options.Beta2,
                        options.LearningRate,
                        options.WeightDecay,
                        options.Decay1D,
                        stream),
                LionCudaPrecision.BFloat16 =>
                    CudaNativeGateway.LionMultiTensorBFloat16(
                        _deviceIndex,
                        _bf16Chunks!.NativePtr,
                        _bf16Chunks.Length,
                        options.Beta1,
                        options.Beta2,
                        options.LearningRate,
                        options.WeightDecay,
                        options.Decay1D,
                        stream),
                LionCudaPrecision.Mix8 =>
                    CudaNativeGateway.LionMultiTensorMix8(
                        _deviceIndex,
                        _mix8Blocks!.NativePtr,
                        _mix8Blocks.Length,
                        options.Beta1,
                        options.Beta2,
                        options.LearningRate,
                        options.WeightDecay,
                        options.Decay1D,
                        finiteStatus?.NativePtr
                            ?? throw new ArgumentNullException(
                                nameof(finiteStatus)),
                        stream),
                LionCudaPrecision.Bfp8 =>
                    CudaNativeGateway.LionMultiTensorBfp8(
                        _deviceIndex,
                        _bfp8Tensors!.NativePtr,
                        _bfp8Tensors.Length,
                        options.Beta1,
                        options.Beta2,
                        options.LearningRate,
                        options.WeightDecay,
                        options.Decay1D,
                        _bfp8Reduction!.NativePtr,
                        _bfp8MaximumChunks,
                        finiteStatus?.NativePtr
                            ?? throw new ArgumentNullException(
                                nameof(finiteStatus)),
                        stream),
                _ => throw new ArgumentOutOfRangeException(),
            };
            NativeCudaRuntime.Check(status, $"Lion {Precision} update");
        }

        public void Dispose()
        {
            List<Exception>? failures = null;
            if (_floatChunks is not null)
                TryDispose(_floatChunks.Dispose, ref failures);
            if (_bf16Chunks is not null)
                TryDispose(_bf16Chunks.Dispose, ref failures);
            if (_mix8Blocks is not null)
                TryDispose(_mix8Blocks.Dispose, ref failures);
            if (_bfp8Tensors is not null)
                TryDispose(_bfp8Tensors.Dispose, ref failures);
            if (_bfp8Reduction is not null)
                TryDispose(_bfp8Reduction.Dispose, ref failures);
            if (failures is not null)
                throw new AggregateException("Lion plan cleanup failed.", failures);
        }

        private static bool AppliesWeightDecay(Parameter parameter)
            => parameter.WeightDecay == WeightDecayPolicy.Apply;

        private static void AppendFloatChunks(
            List<FloatChunk> chunks,
            nint data,
            nint gradient,
            nint momentum,
            nint physical,
            int length,
            int chunkElements,
            bool decay,
            bool publishBFloat16,
            bool rankOne)
        {
            for (int offset = 0; offset < length; offset += chunkElements)
            {
                chunks.Add(new FloatChunk(
                    data,
                    gradient,
                    momentum,
                    physical,
                    offset,
                    Math.Min(chunkElements, length - offset),
                    decay ? 1 : 0,
                    publishBFloat16 ? 1 : 0,
                    rankOne ? 1 : 0));
            }
        }

        private static void AppendBFloat16Chunks(
            List<BFloat16Chunk> chunks,
            nint data,
            nint gradient,
            nint momentum,
            int length,
            int chunkElements,
            bool decay,
            bool rankOne)
        {
            for (int offset = 0; offset < length; offset += chunkElements)
            {
                chunks.Add(new BFloat16Chunk(
                    data,
                    gradient,
                    momentum,
                    offset,
                    Math.Min(chunkElements, length - offset),
                    decay ? 1 : 0,
                    rankOne ? 1 : 0));
            }
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private readonly struct FloatChunk(
        nint data,
        nint gradient,
        nint momentum,
        nint physicalBFloat16,
        int offset,
        int length,
        int applyWeightDecay,
        int publishBFloat16,
        int rankOne)
    {
        private readonly nint Data = data;
        private readonly nint Gradient = gradient;
        private readonly nint Momentum = momentum;
        private readonly nint PhysicalBFloat16 = physicalBFloat16;
        private readonly int Offset = offset;
        private readonly int Length = length;
        private readonly int ApplyWeightDecay = applyWeightDecay;
        private readonly int PublishBFloat16 = publishBFloat16;
        private readonly int RankOne = rankOne;
    }

    [StructLayout(LayoutKind.Sequential)]
    private readonly struct BFloat16Chunk(
        nint data,
        nint gradient,
        nint momentum,
        int offset,
        int length,
        int applyWeightDecay,
        int rankOne)
    {
        private readonly nint Data = data;
        private readonly nint Gradient = gradient;
        private readonly nint Momentum = momentum;
        private readonly int Offset = offset;
        private readonly int Length = length;
        private readonly int ApplyWeightDecay = applyWeightDecay;
        private readonly int RankOne = rankOne;
    }

    [StructLayout(LayoutKind.Sequential)]
    private readonly struct Mix8Block(
        nint master,
        nint gradient,
        nint momentum,
        nint payload,
        nint scales,
        int offset,
        int length,
        int scaleIndex,
        int applyWeightDecay,
        int rankOne)
    {
        private readonly nint Master = master;
        private readonly nint Gradient = gradient;
        private readonly nint Momentum = momentum;
        private readonly nint Payload = payload;
        private readonly nint Scales = scales;
        private readonly int Offset = offset;
        private readonly int Length = length;
        private readonly int ScaleIndex = scaleIndex;
        private readonly int ApplyWeightDecay = applyWeightDecay;
        private readonly int RankOne = rankOne;
    }

    [StructLayout(LayoutKind.Sequential)]
    private readonly struct Bfp8Tensor(
        nint dataPayload,
        nint dataScale,
        nint gradientPayload,
        nint gradientScale,
        nint momentumPayload,
        nint momentumScale,
        int length,
        int applyWeightDecay,
        int rankOne)
    {
        private readonly nint DataPayload = dataPayload;
        private readonly nint DataScale = dataScale;
        private readonly nint GradientPayload = gradientPayload;
        private readonly nint GradientScale = gradientScale;
        private readonly nint MomentumPayload = momentumPayload;
        private readonly nint MomentumScale = momentumScale;
        private readonly int Length = length;
        private readonly int ApplyWeightDecay = applyWeightDecay;
        private readonly int RankOne = rankOne;
    }
}
