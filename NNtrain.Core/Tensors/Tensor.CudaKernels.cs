
namespace NNtrain;

/// <summary>CUDA kernels shared by the ForgetMemory training graph.</summary>
internal static partial class TensorCudaKernels
{
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<
        int,
        Lazy<NativeCudaBuffer<double>>> GradientNormScratch = new();
    private static readonly object GradientSquaredSumCacheLock = new();
    private static GradientSquaredSumCacheEntry? _gradientSquaredSumCache;

    internal static AttentionResidentContext
        AttentionForwardResident(
            Tensor projected,
            int batch,
            int sequence,
            int modelWidth,
            int numHeads,
            bool causal)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var input = projected.EnsureCudaFloat32Buffer(deviceIndex);
        var output = Tensor.RentCudaFloatBuffer(
            deviceIndex, checked(batch * sequence * modelWidth));
        int queries = checked(batch * numHeads * sequence);
        var softmaxLogSumExp = Tensor.RentCudaFloatBuffer(
            deviceIndex, queries);
        if (!CudaFlashAttention.TryForward(accelerator, input, output,
            softmaxLogSumExp, batch, sequence, modelWidth, numHeads, causal))
        {
            Tensor.ReturnCudaFloatBuffer(accelerator, output);
            Tensor.ReturnCudaFloatBuffer(accelerator, softmaxLogSumExp);
            throw new PlatformNotSupportedException(
                "CUDA attention requires the native FlashAttention backend.");
        }
        return new AttentionResidentContext(
            output, null, softmaxLogSumExp, accelerator,
            nativeFlashAttention: true);
    }

    internal static void AttentionBackwardResident(
        Tensor projected,
        Tensor output,
        AttentionResidentContext context,
        int batch,
        int sequence,
        int modelWidth,
        int numHeads,
        bool causal)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var input = projected.EnsureCudaFloat32Buffer(deviceIndex);
        var outputGradient = output.EnsureCudaGradientBuffer(deviceIndex);
        var inputGradient = projected.EnsureCudaGradientBuffer(deviceIndex);
        CudaFlashAttention.Backward(accelerator, input, context.Output,
            outputGradient, context.SoftmaxLogSumExp!, inputGradient,
            batch, sequence, modelWidth, numHeads, causal);
        projected.MarkCudaGradientMutated(deviceIndex);
    }

    internal sealed class AttentionResidentContext(
        NativeCudaBuffer<float> output,
        NativeCudaBuffer<float>? probabilities,
        NativeCudaBuffer<float>? softmaxLogSumExp,
        NativeCudaDevice accelerator,
        bool nativeFlashAttention = false) : IDisposable
    {
        private int _disposed;
        internal NativeCudaBuffer<float> Output { get; } = output;
        internal NativeCudaBuffer<float>? Probabilities { get; } = probabilities;
        internal NativeCudaBuffer<float>? SoftmaxLogSumExp { get; }
            = softmaxLogSumExp;
        internal bool NativeFlashAttention { get; } = nativeFlashAttention;
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            if (Probabilities is not null)
                Tensor.ReturnCudaFloatBuffer(accelerator, Probabilities);
            if (SoftmaxLogSumExp is not null)
                Tensor.ReturnCudaFloatBuffer(accelerator, SoftmaxLogSumExp);
        }
    }

    internal static float ClipGradientNormResident(
        IReadOnlyList<Parameter> parameters,
        float maxNorm)
    {
        int[] devices = Tensor.CudaDeviceIndices.ToArray();
        double squaredSumValue;
        if (!TryConsumeGradientSquaredSum(parameters, devices, out squaredSumValue))
        {
            NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
            NativeCudaBuffer<double> squaredSumBuffer = GradientNormScratch
                .GetOrAdd(
                    Tensor.CudaDeviceIndex,
                    static deviceIndex => new Lazy<NativeCudaBuffer<double>>(
                        () => ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
                            .Allocate1D<double>(1),
                        LazyThreadSafetyMode.ExecutionAndPublication)).Value;
            squaredSumBuffer.MemSetToZero();
            var primaryArenas = new HashSet<NativeCudaArena<float>>(
                ReferenceEqualityComparer.Instance);
            foreach (Parameter parameter in parameters)
            {
                Tensor tensor = parameter.T;
                if (!tensor.HasGradientBuffer)
                    continue;
                NativeCudaArena<float>? arena = tensor.GetCudaGradientArena(
                    Tensor.CudaDeviceIndex);
                if (arena is not null)
                {
                    if (primaryArenas.Add(arena))
                    {
                        CudaTensorNative.SquaredSum(
                            Tensor.CudaDeviceIndex,
                            arena.NativePtr,
                            arena.Length,
                            squaredSumBuffer.NativePtr);
                    }
                    continue;
                }
                var gradient = tensor.EnsureCudaGradientBuffer();
                CudaTensorNative.SquaredSum(
                    Tensor.CudaDeviceIndex,
                    gradient.NativePtr,
                    tensor.Numel,
                    squaredSumBuffer.NativePtr);
            }
            accelerator.Synchronize();
            var squaredSum = new double[1];
            squaredSumBuffer.CopyToCPU(squaredSum);
            squaredSumValue = squaredSum[0];
        }
        float totalNorm = (float)Math.Sqrt(squaredSumValue);
        if (totalNorm <= maxNorm)
            return totalNorm;

        float scale = maxNorm / (totalNorm + 1e-6f);
        foreach (int deviceIndex in devices)
        {
            var scaledArenas = new HashSet<NativeCudaArena<float>>(
                ReferenceEqualityComparer.Instance);
            foreach (Parameter parameter in parameters)
            {
                Tensor tensor = parameter.T;
                if (!tensor.HasGradientBuffer)
                    continue;
                NativeCudaArena<float>? arena =
                    tensor.GetCudaGradientArena(deviceIndex);
                if (arena is not null)
                {
                    if (scaledArenas.Add(arena))
                    {
                        CudaTensorNative.Scale(
                            deviceIndex,
                            arena.NativePtr,
                            arena.Length,
                            scale);
                        arena.MarkDirty();
                    }
                    continue;
                }
                NativeCudaBuffer<float> gradient =
                    tensor.EnsureCudaGradientBuffer(deviceIndex);
                CudaTensorNative.Scale(
                    deviceIndex,
                    gradient.NativePtr,
                    tensor.Numel,
                    scale);
            }
        }
        foreach (Parameter parameter in parameters)
            parameter.T.MarkCudaGradientsSynchronized(devices);
        foreach (int deviceIndex in devices)
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex).Synchronize();
        return totalNorm;
    }

    internal static void PublishGradientSquaredSum(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> devices,
        double squaredSum)
    {
        if (!double.IsFinite(squaredSum) || squaredSum < 0d)
            return;
        var entry = new GradientSquaredSumCacheEntry(
            parameters.Select(parameter =>
                new WeakReference<Parameter>(parameter)).ToArray(),
            parameters.Select(parameter => parameter.T.GradientVersion).ToArray(),
            devices.ToArray(),
            squaredSum);
        lock (GradientSquaredSumCacheLock)
            _gradientSquaredSumCache = entry;
    }

    private static bool TryConsumeGradientSquaredSum(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> devices,
        out double squaredSum)
    {
        lock (GradientSquaredSumCacheLock)
        {
            GradientSquaredSumCacheEntry? entry = _gradientSquaredSumCache;
            if (entry is not null
                && entry.Parameters.Length == parameters.Count
                && entry.Devices.SequenceEqual(devices)
                && parameters.Select((parameter, index) =>
                    entry.Parameters[index].TryGetTarget(out Parameter? cached)
                    && ReferenceEquals(parameter, cached)
                    && parameter.T.GradientVersion == entry.Versions[index])
                    .All(value => value))
            {
                _gradientSquaredSumCache = null;
                squaredSum = entry.SquaredSum;
                return true;
            }
        }
        squaredSum = 0d;
        return false;
    }

    private sealed record GradientSquaredSumCacheEntry(
        WeakReference<Parameter>[] Parameters,
        long[] Versions,
        int[] Devices,
        double SquaredSum);

    internal static void AllReduceGradientResident(
        Tensor tensor,
        IReadOnlyList<int> deviceIndices)
    {
        if (deviceIndices.Count < 2)
            return;
        int primaryIndex = deviceIndices[0];
        NativeCudaDevice primary =
            ForgetMemoryV2Cuda.GetAccelerator(primaryIndex);
        NativeCudaBuffer<float> primaryGradient =
            tensor.EnsureCudaGradientBuffer(primaryIndex);
        using var staging = primary.Allocate1D<float>(tensor.Numel);
        for (int index = 1; index < deviceIndices.Count; index++)
        {
            int secondaryIndex = deviceIndices[index];
            NativeCudaDevice secondary =
                ForgetMemoryV2Cuda.GetAccelerator(secondaryIndex);
            NativeCudaBuffer<float> secondaryGradient =
                tensor.EnsureCudaGradientBuffer(secondaryIndex);
            secondary.Synchronize();
            secondaryGradient.View.CopyTo(staging.View);
            primary.Synchronize();
            CudaTensorNative.Accumulate(
                primaryIndex,
                staging.NativePtr,
                primaryGradient.NativePtr,
                tensor.Numel);
        }
        primary.Synchronize();
        for (int index = 1; index < deviceIndices.Count; index++)
        {
            int secondaryIndex = deviceIndices[index];
            NativeCudaDevice secondary =
                ForgetMemoryV2Cuda.GetAccelerator(secondaryIndex);
            NativeCudaBuffer<float> secondaryGradient =
                tensor.EnsureCudaGradientBuffer(secondaryIndex);
            primaryGradient.View.CopyTo(secondaryGradient.View);
            secondary.Synchronize();
        }
        tensor.MarkCudaGradientsSynchronized(deviceIndices);
    }

    internal static void AllReduceGradientsResident(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> deviceIndices,
        FlatGradientPlan? flatPlan = null)
    {
        if (deviceIndices.Count < 2)
            return;
        if (flatPlan is not null
            && flatPlan.Matches(parameters, deviceIndices))
        {
            AllReduceFlatGradientsResident(parameters, deviceIndices, flatPlan);
            return;
        }
        int primaryIndex = deviceIndices[0];
        NativeCudaDevice primary =
            ForgetMemoryV2Cuda.GetAccelerator(primaryIndex);

        for (int device = 1; device < deviceIndices.Count; device++)
        {
            int secondaryIndex = deviceIndices[device];
            NativeCudaDevice secondary =
                ForgetMemoryV2Cuda.GetAccelerator(secondaryIndex);
            secondary.Synchronize();
            foreach (Parameter parameter in parameters)
            {
                Tensor tensor = parameter.T;
                var primaryGradient =
                    tensor.EnsureCudaGradientBuffer(primaryIndex);
                var secondaryGradient =
                    tensor.EnsureCudaGradientBuffer(secondaryIndex);
                var staging = tensor.EnsureCudaStagingBuffer(primaryIndex);
                secondaryGradient.View.CopyTo(staging.View);
                CudaTensorNative.Accumulate(
                    primaryIndex,
                    staging.NativePtr,
                    primaryGradient.NativePtr,
                    tensor.Numel);
            }
            primary.Synchronize();
        }

        foreach (int secondaryIndex in deviceIndices.Skip(1))
        {
            NativeCudaDevice secondary =
                ForgetMemoryV2Cuda.GetAccelerator(secondaryIndex);
            foreach (Parameter parameter in parameters)
            {
                Tensor tensor = parameter.T;
                var primaryGradient =
                    tensor.EnsureCudaGradientBuffer(primaryIndex);
                var secondaryGradient =
                    tensor.EnsureCudaGradientBuffer(secondaryIndex);
                primaryGradient.View.CopyTo(secondaryGradient.View);
            }
            secondary.Synchronize();
        }
        foreach (Parameter parameter in parameters)
            parameter.T.MarkCudaGradientsSynchronized(deviceIndices);
    }

    private static void AllReduceFlatGradientsResident(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> deviceIndices,
        FlatGradientPlan plan)
    {
        Parallel.For(0, deviceIndices.Count, device =>
        {
            int deviceIndex = deviceIndices[device];
            NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            var flat = plan.GetFlatBuffer(deviceIndex);
            flat.MemSetToZero();
            for (int parameter = 0; parameter < parameters.Count; parameter++)
            {
                Tensor tensor = parameters[parameter].T;
                var gradient = tensor.EnsureCudaGradientBuffer(deviceIndex);
                CudaTensorNative.Copy(
                    deviceIndex,
                    gradient.NativePtr,
                    flat.NativePtr,
                    tensor.Numel,
                    destinationOffset: plan.Offsets[parameter]);
            }
            accelerator.Synchronize();
        });

        if (deviceIndices.Count == 2)
        {
            // Preserve both original gradients before either device starts
            // accumulating. This removes the GPU-0 gather/broadcast bottleneck.
            Parallel.For(0, 2, device =>
            {
                int destinationIndex = deviceIndices[device];
                int sourceIndex = deviceIndices[1 - device];
                plan.GetFlatBuffer(sourceIndex).View.CopyTo(
                    plan.GetStagingBuffer(destinationIndex).View);
                ForgetMemoryV2Cuda.GetAccelerator(destinationIndex).Synchronize();
            });

            Parallel.For(0, 2, device =>
            {
                int deviceIndex = deviceIndices[device];
                NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
                CudaTensorNative.Accumulate(
                    deviceIndex,
                    plan.GetStagingBuffer(deviceIndex).NativePtr,
                    plan.GetFlatBuffer(deviceIndex).NativePtr,
                    plan.TotalElements);
                accelerator.Synchronize();
            });
        }
        else
        {
            // Ring all-reduce for N devices. The staging/exchange buffers carry
            // one immutable contribution around the ring while each device
            // accumulates locally into its flat buffer.
            Parallel.For(0, deviceIndices.Count, device =>
            {
                int deviceIndex = deviceIndices[device];
                plan.GetFlatBuffer(deviceIndex).View.CopyTo(
                    plan.GetStagingBuffer(deviceIndex).View);
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex).Synchronize();
            });

            for (int round = 0; round < deviceIndices.Count - 1; round++)
            {
                bool stagingIsSource = (round & 1) == 0;
                Parallel.For(0, deviceIndices.Count, device =>
                {
                    int destinationIndex = deviceIndices[device];
                    int predecessorIndex = deviceIndices[
                        (device + deviceIndices.Count - 1) % deviceIndices.Count];
                    var source = stagingIsSource
                        ? plan.GetStagingBuffer(predecessorIndex)
                        : plan.GetExchangeBuffer(predecessorIndex);
                    var destination = stagingIsSource
                        ? plan.GetExchangeBuffer(destinationIndex)
                        : plan.GetStagingBuffer(destinationIndex);
                    source.View.CopyTo(destination.View);
                    ForgetMemoryV2Cuda.GetAccelerator(destinationIndex).Synchronize();
                });

                Parallel.For(0, deviceIndices.Count, device =>
                {
                    int deviceIndex = deviceIndices[device];
                    NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
                    var received = stagingIsSource
                        ? plan.GetExchangeBuffer(deviceIndex)
                        : plan.GetStagingBuffer(deviceIndex);
                    CudaTensorNative.Accumulate(
                        deviceIndex,
                        received.NativePtr,
                        plan.GetFlatBuffer(deviceIndex).NativePtr,
                        plan.TotalElements);
                    accelerator.Synchronize();
                });
            }
        }

        Parallel.For(0, deviceIndices.Count, device =>
        {
            int deviceIndex = deviceIndices[device];
            NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            var flat = plan.GetFlatBuffer(deviceIndex);
            for (int parameter = 0; parameter < parameters.Count; parameter++)
            {
                Tensor tensor = parameters[parameter].T;
                var gradient = tensor.EnsureCudaGradientBuffer(deviceIndex);
                CudaTensorNative.Copy(
                    deviceIndex,
                    flat.NativePtr,
                    gradient.NativePtr,
                    tensor.Numel,
                    sourceOffset: plan.Offsets[parameter]);
            }
            accelerator.Synchronize();
        });
        foreach (Parameter parameter in parameters)
            parameter.T.MarkCudaGradientsSynchronized(deviceIndices);
    }

    internal sealed class FlatGradientPlan : IDisposable
    {
        private readonly Parameter[] _parameters;
        private readonly int[] _devices;
        private readonly Dictionary<int, NativeCudaBuffer<float>> _flat = [];
        private readonly Dictionary<int, NativeCudaBuffer<float>> _staging = [];
        private readonly Dictionary<int, NativeCudaBuffer<float>> _exchange = [];

        internal FlatGradientPlan(
            IReadOnlyList<Parameter> parameters,
            IReadOnlyList<int> devices)
        {
            _parameters = parameters.ToArray();
            _devices = devices.ToArray();
            Offsets = new int[_parameters.Length];
            int total = 0;
            for (int index = 0; index < _parameters.Length; index++)
            {
                Offsets[index] = total;
                total = checked(total + _parameters[index].T.Numel);
            }
            TotalElements = total;
            foreach (int device in _devices)
            {
                NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(device);
                _flat[device] = accelerator.Allocate1D<float>(total);
                _staging[device] = accelerator.Allocate1D<float>(total);
                if (_devices.Length > 2)
                    _exchange[device] = accelerator.Allocate1D<float>(total);
            }
        }

        internal int[] Offsets { get; }
        internal int TotalElements { get; }
        internal NativeCudaBuffer<float> GetFlatBuffer(int device)
            => _flat[device];
        internal NativeCudaBuffer<float> GetStagingBuffer(int device)
            => _staging[device];
        internal NativeCudaBuffer<float> GetExchangeBuffer(int device)
            => _exchange[device];
        internal bool Matches(
            IReadOnlyList<Parameter> parameters,
            IReadOnlyList<int> devices)
            => parameters.Count == _parameters.Length
                && devices.SequenceEqual(_devices)
                && parameters.Select((parameter, index) =>
                    ReferenceEquals(parameter, _parameters[index])).All(value => value);

        public void Dispose()
        {
            foreach (var buffer in _flat.Values)
                buffer.Dispose();
            foreach (var buffer in _staging.Values)
                buffer.Dispose();
            foreach (var buffer in _exchange.Values)
                buffer.Dispose();
            _flat.Clear();
            _staging.Clear();
            _exchange.Clear();
        }
    }

    internal static NativeCudaBuffer<float> CopyForwardResident(
        Tensor input)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var inputBuffer = input.EnsureCudaFloat32Buffer();
        var outputBuffer = Tensor.RentCudaFloatBuffer(
            Tensor.CudaDeviceIndex, input.Numel);
        CudaTensorNative.Copy(
            Tensor.CudaDeviceIndex,
            inputBuffer.NativePtr,
            outputBuffer.NativePtr,
            input.Numel);
        return outputBuffer;
    }

    internal static NativeCudaBuffer<float>
        CopyRangeForwardResident(Tensor input, int sourceOffset, int length)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var inputBuffer = input.EnsureCudaFloat32Buffer();
        var outputBuffer = Tensor.RentCudaFloatBuffer(
            Tensor.CudaDeviceIndex, length);
        CudaTensorNative.Copy(
            Tensor.CudaDeviceIndex,
            inputBuffer.NativePtr,
            outputBuffer.NativePtr,
            length,
            sourceOffset: sourceOffset);
        return outputBuffer;
    }

    internal static void AccumulateGradientRangeResident(
        Tensor source,
        Tensor destination,
        int destinationOffset)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var sourceBuffer = source.EnsureCudaGradientBuffer();
        var destinationBuffer = destination.EnsureCudaGradientBuffer();
        CudaTensorNative.Accumulate(
            Tensor.CudaDeviceIndex,
            sourceBuffer.NativePtr,
            destinationBuffer.NativePtr,
            source.Numel,
            destinationOffset: destinationOffset);
        destination.MarkCudaGradientMutated();
    }

    internal static void AccumulateGradientResident(
        Tensor source,
        Tensor destination)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var sourceBuffer = source.EnsureCudaGradientBuffer();
        var destinationBuffer = destination.EnsureCudaGradientBuffer();
        CudaTensorNative.Accumulate(
            Tensor.CudaDeviceIndex,
            sourceBuffer.NativePtr,
            destinationBuffer.NativePtr,
            source.Numel);
        destination.MarkCudaGradientMutated();
    }

    internal static NativeCudaBuffer<float> AddForwardResident(
        Tensor left,
        Tensor right,
        bool bfloat16Compute)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var leftBuffer = left.EnsureCudaFloat32Buffer();
        var rightBuffer = right.EnsureCudaFloat32Buffer();
        var outputBuffer = Tensor.RentCudaFloatBuffer(
            Tensor.CudaDeviceIndex, left.Numel);
        CudaTensorNative.Add(
            Tensor.CudaDeviceIndex,
            leftBuffer.NativePtr,
            rightBuffer.NativePtr,
            outputBuffer.NativePtr,
            left.Numel,
            bfloat16: false);
        return outputBuffer;
    }

    internal static void AddBackwardResident(
        Tensor output,
        Tensor left,
        Tensor right)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var outputGradient = output.EnsureCudaGradientBuffer();
        var leftGradient = left.EnsureCudaGradientBuffer();
        var rightGradient = ReferenceEquals(left, right)
            ? leftGradient
            : right.EnsureCudaGradientBuffer();
        CudaTensorNative.AddBackward(
            Tensor.CudaDeviceIndex,
            outputGradient.NativePtr,
            leftGradient.NativePtr,
            rightGradient.NativePtr,
            output.Numel,
            ReferenceEquals(left, right));
        left.MarkCudaGradientMutated();
        if (!ReferenceEquals(left, right))
            right.MarkCudaGradientMutated();
    }

    internal static NativeCudaBuffer<float>
        EmbeddingForwardResident(
            Tensor table,
            int[] indices,
            int width)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var tableBuffer = table.EnsureCudaFloat32Buffer();
        using var indicesBuffer = accelerator.Allocate1D(indices);
        var outputBuffer = Tensor.RentCudaFloatBuffer(
            Tensor.CudaDeviceIndex, checked(indices.Length * width));
        CudaTensorNative.Embedding(
            Tensor.CudaDeviceIndex,
            tableBuffer.NativePtr,
            indicesBuffer.NativePtr,
            outputBuffer.NativePtr,
            checked((int)outputBuffer.Length),
            width,
            bfloat16: false);
        accelerator.Synchronize();
        return outputBuffer;
    }

    internal static void EmbeddingBackwardResident(
        Tensor output,
        Tensor table,
        int[] indices,
        int width)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var indicesBuffer = accelerator.Allocate1D(indices);
        var outputGradientBuffer = output.EnsureCudaGradientBuffer();
        var tableGradientBuffer = table.EnsureCudaGradientBuffer();
        CudaTensorNative.EmbeddingBackward(
            Tensor.CudaDeviceIndex,
            indicesBuffer.NativePtr,
            outputGradientBuffer.NativePtr,
            tableGradientBuffer.NativePtr,
            output.Numel,
            width);
        accelerator.Synchronize();
        table.MarkCudaGradientMutated();
    }

    internal static EmbeddingPositionsResidentContext
        EmbeddingWithPositionsForwardResident(
            Tensor tokenTable,
            Tensor positionTable,
            int[] indices,
            int sequenceLength,
            int width)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var tokens = tokenTable.EnsureCudaFloat32Buffer();
        var positions = positionTable.EnsureCudaFloat32Buffer();
        var indicesBuffer = Tensor.RentCudaIntBuffer(
            Tensor.CudaDeviceIndex, indices);
        var output = Tensor.RentCudaFloatBuffer(
            Tensor.CudaDeviceIndex, checked(indices.Length * width));
        CudaTensorNative.EmbeddingPositions(
            Tensor.CudaDeviceIndex,
            tokens.NativePtr,
            positions.NativePtr,
            indicesBuffer.NativePtr,
            output.NativePtr,
            checked(indices.Length * width),
            sequenceLength,
            width,
            bfloat16: false);
        return new EmbeddingPositionsResidentContext(
            output, indicesBuffer, accelerator);
    }

    internal static void EmbeddingWithPositionsBackwardResident(
        Tensor output,
        Tensor tokenTable,
        Tensor positionTable,
        EmbeddingPositionsResidentContext context,
        int sequenceLength,
        int width)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var outputGradient = output.EnsureCudaGradientBuffer();
        var tokenGradient = tokenTable.EnsureCudaGradientBuffer();
        var positionGradient = positionTable.EnsureCudaGradientBuffer();
        CudaTensorNative.EmbeddingPositionsBackward(
            Tensor.CudaDeviceIndex,
            context.Indices.NativePtr,
            outputGradient.NativePtr,
            tokenGradient.NativePtr,
            positionGradient.NativePtr,
            output.Numel,
            sequenceLength,
            width);
        tokenTable.MarkCudaGradientMutated();
        positionTable.MarkCudaGradientMutated();
    }

    internal sealed class EmbeddingPositionsResidentContext(
        NativeCudaBuffer<float> output,
        NativeCudaBuffer<int> indices,
        NativeCudaDevice accelerator) : IDisposable
    {
        private int _disposed;
        internal NativeCudaBuffer<float> Output { get; } = output;
        internal NativeCudaBuffer<int> Indices { get; } = indices;
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            Tensor.ReturnCudaIntBuffer(accelerator, Indices);
            GC.SuppressFinalize(this);
        }
    }

    internal static NativeCudaBuffer<float>
        DropoutForwardResident(
            Tensor input,
            uint seed,
            uint dropThreshold,
            float scale)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var inputBuffer = input.EnsureCudaFloat32Buffer();
        var outputBuffer = Tensor.RentCudaFloatBuffer(
            Tensor.CudaDeviceIndex, input.Numel);
        CudaTensorNative.Dropout(
            Tensor.CudaDeviceIndex,
            inputBuffer.NativePtr,
            outputBuffer.NativePtr,
            input.Numel,
            seed,
            dropThreshold,
            scale,
            bfloat16: false);
        return outputBuffer;
    }

    internal static void DropoutBackwardResident(
        Tensor output,
        Tensor input,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var outputGradientBuffer = output.EnsureCudaGradientBuffer();
        var inputGradientBuffer = input.EnsureCudaGradientBuffer();
        CudaTensorNative.DropoutBackward(
            Tensor.CudaDeviceIndex,
            outputGradientBuffer.NativePtr,
            inputGradientBuffer.NativePtr,
            output.Numel,
            seed,
            dropThreshold,
            scale);
        input.MarkCudaGradientMutated();
    }

    internal static NativeCudaBuffer<float>
        AddDropoutForwardResident(
            Tensor residual,
            Tensor branch,
            uint seed,
            uint dropThreshold,
            float scale)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var residualBuffer = residual.EnsureCudaFloat32Buffer();
        var branchBuffer = branch.EnsureCudaFloat32Buffer();
        var outputBuffer = Tensor.RentCudaFloatBuffer(
            Tensor.CudaDeviceIndex, residual.Numel);
        CudaTensorNative.AddDropout(
            Tensor.CudaDeviceIndex,
            residualBuffer.NativePtr,
            branchBuffer.NativePtr,
            outputBuffer.NativePtr,
            residual.Numel,
            seed,
            dropThreshold,
            scale,
            bfloat16: false);
        return outputBuffer;
    }

    internal static void AddDropoutBackwardResident(
        Tensor output,
        Tensor residual,
        Tensor branch,
        bool sameParent,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var outputGradientBuffer = output.EnsureCudaGradientBuffer();
        var residualGradientBuffer = residual.EnsureCudaGradientBuffer();
        NativeCudaBuffer<float> branchGradientBuffer = sameParent
            ? residualGradientBuffer
            : branch.EnsureCudaGradientBuffer();
        CudaTensorNative.AddDropoutBackward(
            Tensor.CudaDeviceIndex,
            outputGradientBuffer.NativePtr,
            residualGradientBuffer.NativePtr,
            branchGradientBuffer.NativePtr,
            output.Numel,
            sameParent,
            seed,
            dropThreshold,
            scale);
        residual.MarkCudaGradientMutated();
        if (!sameParent)
            branch.MarkCudaGradientMutated();
    }

    internal static float[] EmbeddingForward(
        Tensor table,
        int[] indices,
        int width)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var output = new float[checked(indices.Length * width)];
        var tableBuffer = table.EnsureCudaFloat32Buffer();
        using var indicesBuffer = accelerator.Allocate1D(indices);
        using var outputBuffer = accelerator.Allocate1D<float>(output.Length);
        CudaTensorNative.Embedding(
            Tensor.CudaDeviceIndex,
            tableBuffer.NativePtr,
            indicesBuffer.NativePtr,
            outputBuffer.NativePtr,
            output.Length,
            width,
            bfloat16: false);
        accelerator.Synchronize();
        outputBuffer.CopyToCPU(output);
        return output;
    }

    internal static void EmbeddingBackward(
        int[] indices,
        float[] outputGradient,
        float[] tableGradient,
        int width)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var indicesBuffer = accelerator.Allocate1D(indices);
        using var outputGradientBuffer = accelerator.Allocate1D(outputGradient);
        using var tableGradientBuffer = accelerator.Allocate1D(tableGradient);
        CudaTensorNative.EmbeddingBackward(
            Tensor.CudaDeviceIndex,
            indicesBuffer.NativePtr,
            outputGradientBuffer.NativePtr,
            tableGradientBuffer.NativePtr,
            outputGradient.Length,
            width);
        accelerator.Synchronize();
        tableGradientBuffer.CopyToCPU(tableGradient);
    }

    internal static float[] DropoutForward(
        float[] input,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var output = new float[input.Length];
        var inputBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, input);
        using var outputBuffer = accelerator.Allocate1D<float>(output.Length);
        CudaTensorNative.Dropout(
            Tensor.CudaDeviceIndex,
            inputBuffer.NativePtr,
            outputBuffer.NativePtr,
            output.Length,
            seed,
            dropThreshold,
            scale,
            bfloat16: false);
        accelerator.Synchronize();
        outputBuffer.CopyToCPU(output);
        return output;
    }

    internal static void DropoutBackward(
        float[] outputGradient,
        float[] inputGradient,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var outputGradientBuffer = accelerator.Allocate1D(outputGradient);
        using var inputGradientBuffer = accelerator.Allocate1D(inputGradient);
        CudaTensorNative.DropoutBackward(
            Tensor.CudaDeviceIndex,
            outputGradientBuffer.NativePtr,
            inputGradientBuffer.NativePtr,
            outputGradient.Length,
            seed,
            dropThreshold,
            scale);
        accelerator.Synchronize();
        inputGradientBuffer.CopyToCPU(inputGradient);
    }

    internal static float[] AddDropoutForward(
        float[] residual,
        float[] branch,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var output = new float[residual.Length];
        var residualBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, residual);
        var branchBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, branch);
        using var outputBuffer = accelerator.Allocate1D<float>(output.Length);
        CudaTensorNative.AddDropout(
            Tensor.CudaDeviceIndex,
            residualBuffer.NativePtr,
            branchBuffer.NativePtr,
            outputBuffer.NativePtr,
            output.Length,
            seed,
            dropThreshold,
            scale,
            bfloat16: false);
        accelerator.Synchronize();
        outputBuffer.CopyToCPU(output);
        return output;
    }

    internal static void AddDropoutBackward(
        float[] outputGradient,
        float[] residualGradient,
        float[] branchGradient,
        bool sameParent,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        using var outputGradientBuffer = accelerator.Allocate1D(outputGradient);
        using var residualGradientBuffer = accelerator.Allocate1D(residualGradient);
        using var branchGradientBuffer = sameParent
            ? null
            : accelerator.Allocate1D(branchGradient);
        CudaTensorNative.AddDropoutBackward(
            Tensor.CudaDeviceIndex,
            outputGradientBuffer.NativePtr,
            residualGradientBuffer.NativePtr,
            sameParent
                ? residualGradientBuffer.NativePtr
                : branchGradientBuffer!.NativePtr,
            outputGradient.Length,
            sameParent,
            seed,
            dropThreshold,
            scale);
        accelerator.Synchronize();
        residualGradientBuffer.CopyToCPU(residualGradient);
        if (!sameParent)
            branchGradientBuffer!.CopyToCPU(branchGradient);
    }

    internal static NativeCudaBuffer<float>
        LinearForwardResident(
            Tensor input,
            Tensor weight,
            Tensor bias,
            int rows,
            int inputWidth,
            int outputWidth,
            bool applyRelu,
            bool bfloat16Compute)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var inputBuffer = input.EnsureCudaFloat32Buffer(deviceIndex);
        var weightBuffer = weight.EnsureCudaFloat32Buffer(deviceIndex);
        var biasBuffer = bias.EnsureCudaFloat32Buffer(deviceIndex);
        var outputBuffer = Tensor.RentCudaFloatBuffer(
            deviceIndex, checked(rows * outputWidth));
        CudaBlas.LinearForward(
            accelerator,
            deviceIndex,
            inputBuffer,
            weightBuffer,
            outputBuffer,
            rows,
            inputWidth,
            outputWidth,
            bfloat16Compute);
        CudaTensorNative.LinearBias(
            deviceIndex,
            outputBuffer.NativePtr,
            biasBuffer.NativePtr,
            checked(rows * outputWidth),
            outputWidth,
            applyRelu,
            bfloat16: false);
        return outputBuffer;
    }

    internal static void LinearBackwardResident(
        Tensor input,
        Tensor weight,
        Tensor bias,
        Tensor output,
        int rows,
        int inputWidth,
        int outputWidth,
        bool applyRelu,
        bool bfloat16Compute)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var inputBuffer = input.EnsureCudaFloat32Buffer(deviceIndex);
        var weightBuffer = weight.EnsureCudaFloat32Buffer(deviceIndex);
        var outputBuffer = output.EnsureCudaFloat32Buffer(deviceIndex);
        var outputGradientBuffer = output.EnsureCudaGradientBuffer(deviceIndex);
        var inputGradientBuffer = input.EnsureCudaGradientBuffer(deviceIndex);
        var weightGradientBuffer = weight.EnsureCudaGradientBuffer(deviceIndex);
        var biasGradientBuffer = bias.EnsureCudaGradientBuffer(deviceIndex);
        CudaTensorNative.LinearMask(
            deviceIndex,
            outputBuffer.NativePtr,
            outputGradientBuffer.NativePtr,
            output.Numel,
            applyRelu);
        CudaBlas.LinearBackwardInput(
            accelerator,
            deviceIndex,
            outputGradientBuffer,
            weightBuffer,
            inputGradientBuffer,
            rows,
            inputWidth,
            outputWidth,
            bfloat16Compute);
        CudaBlas.LinearBackwardWeight(
            accelerator,
            deviceIndex,
            inputBuffer,
            outputGradientBuffer,
            weightGradientBuffer,
            rows,
            inputWidth,
            outputWidth,
            bfloat16Compute);
        CudaTensorNative.LinearBiasBackward(
            deviceIndex,
            outputGradientBuffer.NativePtr,
            biasGradientBuffer.NativePtr,
            rows,
            outputWidth,
            bfloat16: false);
        input.MarkCudaGradientMutated(deviceIndex);
        weight.MarkCudaGradientMutated(deviceIndex);
        bias.MarkCudaGradientMutated(deviceIndex);
    }

    internal static float[] LinearForward(
        float[] input,
        Tensor weight,
        Tensor bias,
        int rows,
        int inputWidth,
        int outputWidth,
        bool applyRelu,
        bool bfloat16Compute = false)
    {
        int[] devices = Tensor.CudaDeviceIndices
            .Take(Math.Min(rows, Tensor.CudaDeviceIndices.Count))
            .ToArray();
        if (devices.Length == 1)
        {
            return LinearForwardSingle(
                ForgetMemoryV2Cuda.GetAccelerator(devices[0]), devices[0],
                input, weight, bias, rows, inputWidth, outputWidth, applyRelu,
                bfloat16Compute,
                cacheInput: true);
        }

        var output = new float[checked(rows * outputWidth)];
        Parallel.For(0, devices.Length, shard =>
        {
            int start = rows * shard / devices.Length;
            int end = rows * (shard + 1) / devices.Length;
            float[] shardOutput = LinearForwardSingle(
                ForgetMemoryV2Cuda.GetAccelerator(devices[shard]),
                devices[shard],
                input.AsSpan(start * inputWidth, (end - start) * inputWidth)
                    .ToArray(),
                weight, bias, end - start, inputWidth, outputWidth, applyRelu,
                bfloat16Compute,
                cacheInput: false);
            shardOutput.CopyTo(output, start * outputWidth);
        });
        return output;
    }

    private static float[] LinearForwardSingle(
        NativeCudaDevice accelerator,
        int deviceIndex,
        float[] input,
        Tensor weight,
        Tensor bias,
        int rows,
        int inputWidth,
        int outputWidth,
        bool applyRelu,
        bool bfloat16Compute,
        bool cacheInput)
    {
        var output = new float[checked(rows * outputWidth)];
        using NativeCudaBuffer<float>? temporaryInputBuffer =
            cacheInput ? null : accelerator.Allocate1D(input);
        NativeCudaBuffer<float> inputBuffer = cacheInput
            ? CudaResidentArrayCache.GetOrUpload(accelerator, input)
            : temporaryInputBuffer!;
        var weightBuffer = weight.EnsureCudaFloat32Buffer(deviceIndex);
        var biasBuffer = bias.EnsureCudaFloat32Buffer(deviceIndex);
        using var outputBuffer = accelerator.Allocate1D<float>(output.Length);
        CudaBlas.LinearForward(
            accelerator,
            deviceIndex,
            inputBuffer,
            weightBuffer,
            outputBuffer,
            rows,
            inputWidth,
            outputWidth,
            bfloat16Compute);
        CudaTensorNative.LinearBias(
            deviceIndex,
            outputBuffer.NativePtr,
            biasBuffer.NativePtr,
            output.Length,
            outputWidth,
            applyRelu,
            bfloat16: false);
        accelerator.Synchronize();
        outputBuffer.CopyToCPU(output);
        return output;
    }

    internal static void LinearBackward(
        float[] input,
        float[] weight,
        float[] storedOutput,
        float[] outputGradient,
        float[] inputGradient,
        float[] weightGradient,
        float[] biasGradient,
        int rows,
        int inputWidth,
        int outputWidth,
        bool applyRelu,
        bool bfloat16Compute)
    {
        int[] devices = Tensor.CudaDeviceIndices
            .Take(Math.Min(rows, Tensor.CudaDeviceIndices.Count))
            .ToArray();
        if (devices.Length == 1
            || ReferenceEquals(inputGradient, weightGradient))
        {
            LinearBackwardSingle(
                ForgetMemoryV2Cuda.GetAccelerator(devices[0]), devices[0], input, weight,
                storedOutput, outputGradient, inputGradient, weightGradient,
                biasGradient, rows, inputWidth, outputWidth, applyRelu,
                bfloat16Compute);
            return;
        }

        var shardInputGradients = new float[devices.Length][];
        var shardWeightGradients = new float[devices.Length][];
        var shardBiasGradients = new float[devices.Length][];
        Parallel.For(0, devices.Length, shard =>
        {
            int start = rows * shard / devices.Length;
            int end = rows * (shard + 1) / devices.Length;
            int shardRows = end - start;
            var localInputGradient = new float[shardRows * inputWidth];
            var localWeightGradient = new float[weightGradient.Length];
            var localBiasGradient = new float[biasGradient.Length];
            LinearBackwardSingle(
                ForgetMemoryV2Cuda.GetAccelerator(devices[shard]),
                devices[shard],
                input.AsSpan(start * inputWidth, shardRows * inputWidth).ToArray(),
                weight,
                storedOutput.AsSpan(start * outputWidth, shardRows * outputWidth).ToArray(),
                outputGradient.AsSpan(start * outputWidth, shardRows * outputWidth).ToArray(),
                localInputGradient, localWeightGradient, localBiasGradient,
                shardRows, inputWidth, outputWidth, applyRelu,
                bfloat16Compute);
            shardInputGradients[shard] = localInputGradient;
            shardWeightGradients[shard] = localWeightGradient;
            shardBiasGradients[shard] = localBiasGradient;
        });
        for (int shard = 0; shard < devices.Length; shard++)
        {
            int start = rows * shard / devices.Length;
            float[] localInput = shardInputGradients[shard];
            for (int index = 0; index < localInput.Length; index++)
                inputGradient[start * inputWidth + index] += localInput[index];
            float[] localWeight = shardWeightGradients[shard];
            for (int index = 0; index < localWeight.Length; index++)
                weightGradient[index] += localWeight[index];
            float[] localBias = shardBiasGradients[shard];
            for (int index = 0; index < localBias.Length; index++)
                biasGradient[index] += localBias[index];
        }
    }

    private static void LinearBackwardSingle(
        NativeCudaDevice accelerator,
        int deviceIndex,
        float[] input,
        float[] weight,
        float[] storedOutput,
        float[] outputGradient,
        float[] inputGradient,
        float[] weightGradient,
        float[] biasGradient,
        int rows,
        int inputWidth,
        int outputWidth,
        bool applyRelu,
        bool bfloat16Compute)
    {
        var inputBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, input);
        var weightBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, weight);
        var outputBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, storedOutput);
        using var outputGradientBuffer = accelerator.Allocate1D(outputGradient);
        using var inputGradientBuffer = accelerator.Allocate1D(inputGradient);
        using var weightGradientBuffer = accelerator.Allocate1D(weightGradient);
        using var biasGradientBuffer = accelerator.Allocate1D(biasGradient);
        CudaTensorNative.LinearMask(
            deviceIndex,
            outputBuffer.NativePtr,
            outputGradientBuffer.NativePtr,
            outputGradient.Length,
            applyRelu);
        CudaBlas.LinearBackwardInput(
            accelerator,
            deviceIndex,
            outputGradientBuffer,
            weightBuffer,
            inputGradientBuffer,
            rows,
            inputWidth,
            outputWidth,
            bfloat16Compute);
        CudaBlas.LinearBackwardWeight(
            accelerator,
            deviceIndex,
            inputBuffer,
            outputGradientBuffer,
            weightGradientBuffer,
            rows,
            inputWidth,
            outputWidth,
            bfloat16Compute);
        CudaTensorNative.LinearBiasBackward(
            deviceIndex,
            outputGradientBuffer.NativePtr,
            biasGradientBuffer.NativePtr,
            rows,
            outputWidth,
            bfloat16: false);
        accelerator.Synchronize();
        inputGradientBuffer.CopyToCPU(inputGradient);
        weightGradientBuffer.CopyToCPU(weightGradient);
        biasGradientBuffer.CopyToCPU(biasGradient);
    }

    internal static LayerNormResidentContext LayerNormForwardResident(
        Tensor input,
        Tensor gamma,
        Tensor beta,
        int rows,
        int columns,
        float epsilon)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var inputBuffer = input.EnsureCudaFloat32Buffer();
        var gammaBuffer = gamma.EnsureCudaFloat32Buffer();
        var betaBuffer = beta.EnsureCudaFloat32Buffer();
        int deviceIndex = Tensor.CudaDeviceIndex;
        var outputBuffer = Tensor.RentCudaFloatBuffer(deviceIndex, input.Numel);
        var meansBuffer = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        var inverseBuffer = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        bool native = CudaLayerNorm.TryForward(
            accelerator,
            inputBuffer,
            gammaBuffer,
            betaBuffer,
            outputBuffer,
            meansBuffer,
            inverseBuffer,
            rows,
            columns,
            epsilon);
        if (!native)
        {
            Tensor.ReturnCudaFloatBuffer(accelerator, outputBuffer);
            Tensor.ReturnCudaFloatBuffer(accelerator, meansBuffer);
            Tensor.ReturnCudaFloatBuffer(accelerator, inverseBuffer);
            throw new PlatformNotSupportedException(
                "CUDA LayerNorm requires the native reduction kernel.");
        }
        return new LayerNormResidentContext(
            outputBuffer,
            meansBuffer,
            inverseBuffer,
            accelerator,
            native);
    }

    internal static LayerNormResidentContext?
        TryResidualDropoutLayerNormForwardResident(
            Tensor residual,
            Tensor branch,
            Tensor gamma,
            Tensor beta,
            int rows,
            int columns,
            uint seed,
            uint dropThreshold,
            float dropoutScale,
            float epsilon)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        var output = Tensor.RentCudaFloatBuffer(deviceIndex, residual.Numel);
        var means = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        var inverses = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        bool succeeded = CudaLayerNorm.TryFusedForward(
            accelerator,
            residual.EnsureCudaFloat32Buffer(deviceIndex),
            branch.EnsureCudaFloat32Buffer(deviceIndex),
            gamma.EnsureCudaFloat32Buffer(deviceIndex),
            beta.EnsureCudaFloat32Buffer(deviceIndex),
            output,
            means,
            inverses,
            rows,
            columns,
            seed,
            dropThreshold,
            dropoutScale,
            epsilon);
        if (succeeded)
        {
            return new LayerNormResidentContext(
                output, means, inverses, accelerator, native: true);
        }
        Tensor.ReturnCudaFloatBuffer(accelerator, output);
        Tensor.ReturnCudaFloatBuffer(accelerator, means);
        Tensor.ReturnCudaFloatBuffer(accelerator, inverses);
        return null;
    }

    internal static void LayerNormBackwardResident(
        Tensor input,
        Tensor gamma,
        Tensor beta,
        Tensor output,
        LayerNormResidentContext context,
        int rows,
        int columns)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var gammaBuffer = gamma.EnsureCudaFloat32Buffer();
        var outputGradientBuffer = output.EnsureCudaGradientBuffer();
        var inputGradientBuffer = input.EnsureCudaGradientBuffer();
        var gammaGradientBuffer = gamma.EnsureCudaGradientBuffer();
        var betaGradientBuffer = beta.EnsureCudaGradientBuffer();
        if (!context.Native)
            throw new InvalidOperationException(
                "CUDA LayerNorm context was not produced by native CUDA.");
        CudaLayerNorm.Backward(
            accelerator,
            input.EnsureCudaFloat32Buffer(),
            gammaBuffer,
            context.Means,
            context.Inverses,
            outputGradientBuffer,
            inputGradientBuffer,
            gammaGradientBuffer,
            betaGradientBuffer,
            rows,
            columns);
        input.MarkCudaGradientMutated();
        gamma.MarkCudaGradientMutated();
        beta.MarkCudaGradientMutated();
    }

    internal static void ResidualDropoutLayerNormBackwardResident(
        Tensor residual,
        Tensor branch,
        Tensor gamma,
        Tensor beta,
        Tensor output,
        LayerNormResidentContext context,
        int rows,
        int columns,
        bool sameParent,
        uint seed,
        uint dropThreshold,
        float dropoutScale)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        CudaLayerNorm.FusedBackward(
            accelerator,
            residual.EnsureCudaFloat32Buffer(deviceIndex),
            branch.EnsureCudaFloat32Buffer(deviceIndex),
            gamma.EnsureCudaFloat32Buffer(deviceIndex),
            context.Means,
            context.Inverses,
            output.EnsureCudaGradientBuffer(deviceIndex),
            residual.EnsureCudaGradientBuffer(deviceIndex),
            sameParent
                ? residual.EnsureCudaGradientBuffer(deviceIndex)
                : branch.EnsureCudaGradientBuffer(deviceIndex),
            gamma.EnsureCudaGradientBuffer(deviceIndex),
            beta.EnsureCudaGradientBuffer(deviceIndex),
            rows,
            columns,
            sameParent,
            seed,
            dropThreshold,
            dropoutScale);
        residual.MarkCudaGradientMutated(deviceIndex);
        if (!sameParent)
            branch.MarkCudaGradientMutated(deviceIndex);
        gamma.MarkCudaGradientMutated(deviceIndex);
        beta.MarkCudaGradientMutated(deviceIndex);
    }

    internal sealed class LayerNormResidentContext(
        NativeCudaBuffer<float> output,
        NativeCudaBuffer<float> means,
        NativeCudaBuffer<float> inverses,
        NativeCudaDevice accelerator,
        bool native) : IDisposable
    {
        private bool _disposed;
        internal NativeCudaBuffer<float> Output { get; } = output;
        internal NativeCudaBuffer<float> Means { get; } = means;
        internal NativeCudaBuffer<float> Inverses { get; } = inverses;
        internal bool Native { get; } = native;

        internal void Dispose()
        {
            if (_disposed)
                return;
            Tensor.ReturnCudaFloatBuffer(accelerator, Means);
            Tensor.ReturnCudaFloatBuffer(accelerator, Inverses);
            _disposed = true;
            GC.SuppressFinalize(this);
        }

        void IDisposable.Dispose() => Dispose();

        ~LayerNormResidentContext() => Dispose();
    }

    internal static (
        float[] Output,
        float[] Normalized,
        float[] InverseStandardDeviations) LayerNormForward(
        float[] input,
        Tensor gamma,
        Tensor beta,
        int rows,
        int columns,
        float epsilon)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var output = new float[input.Length];
        var normalized = new float[input.Length];
        var inverses = new float[rows];
        var inputBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, input);
        var gammaBuffer = gamma.EnsureCudaFloat32Buffer();
        var betaBuffer = beta.EnsureCudaFloat32Buffer();
        using var outputBuffer = accelerator.Allocate1D<float>(output.Length);
        using var meansBuffer = accelerator.Allocate1D<float>(rows);
        using var inverseBuffer = accelerator.Allocate1D<float>(rows);
        if (!CudaLayerNorm.TryForward(
            accelerator,
            inputBuffer,
            gammaBuffer,
            betaBuffer,
            outputBuffer,
            meansBuffer,
            inverseBuffer,
            rows,
            columns,
            epsilon))
        {
            throw new PlatformNotSupportedException(
                "CUDA LayerNorm requires the native reduction kernel.");
        }
        accelerator.Synchronize();
        outputBuffer.CopyToCPU(output);
        var means = new float[rows];
        meansBuffer.CopyToCPU(means);
        inverseBuffer.CopyToCPU(inverses);
        for (int row = 0; row < rows; row++)
        {
            int offset = row * columns;
            for (int column = 0; column < columns; column++)
            {
                int index = offset + column;
                normalized[index] = (input[index] - means[row]) * inverses[row];
            }
        }
        return (output, normalized, inverses);
    }

    internal static void LayerNormBackward(
        Tensor gamma,
        float[] normalized,
        float[] inverses,
        float[] outputGradient,
        float[] inputGradient,
        float[] gammaGradient,
        float[] betaGradient,
        int rows,
        int columns)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var gammaBuffer = gamma.EnsureCudaFloat32Buffer();
        var reconstructedInput = new float[normalized.Length];
        for (int row = 0; row < rows; row++)
        {
            float inverse = inverses[row];
            int offset = row * columns;
            for (int column = 0; column < columns; column++)
                reconstructedInput[offset + column] =
                    normalized[offset + column] / inverse;
        }
        var inputBuffer = CudaResidentArrayCache.GetOrUpload(
            accelerator, reconstructedInput);
        using var meansBuffer = accelerator.Allocate1D<float>(rows);
        meansBuffer.MemSetToZero();
        var inverseBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, inverses);
        using var outputGradientBuffer = accelerator.Allocate1D(outputGradient);
        using var inputGradientBuffer = accelerator.Allocate1D(inputGradient);
        using var gammaGradientBuffer = accelerator.Allocate1D(gammaGradient);
        using var betaGradientBuffer = accelerator.Allocate1D(betaGradient);
        CudaLayerNorm.Backward(
            accelerator,
            inputBuffer,
            gammaBuffer,
            meansBuffer,
            inverseBuffer,
            outputGradientBuffer,
            inputGradientBuffer,
            gammaGradientBuffer,
            betaGradientBuffer,
            rows,
            columns);
        accelerator.Synchronize();
        inputGradientBuffer.CopyToCPU(inputGradient);
        gammaGradientBuffer.CopyToCPU(gammaGradient);
        betaGradientBuffer.CopyToCPU(betaGradient);
    }

    internal static CrossEntropyResidentContext CrossEntropyForwardResident(
        Tensor logits,
        int[] labels,
        int rows,
        int columns,
        int ignoreIndex,
        int validRows,
        float labelSmoothing)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var logitsBuffer = logits.EnsureCudaFloat32Buffer();
        var labelsBuffer = Tensor.RentCudaIntBuffer(
            Tensor.CudaDeviceIndex, labels);
        int deviceIndex = Tensor.CudaDeviceIndex;
        var maximaBuffer = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        var inverseSumsBuffer = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        var rowLossesBuffer = Tensor.RentCudaFloatBuffer(deviceIndex, rows);
        var lossBuffer = Tensor.RentCudaFloatBuffer(deviceIndex, 1);
        CudaTensorNative.CrossEntropy(
            deviceIndex,
            logitsBuffer.NativePtr,
            labelsBuffer.NativePtr,
            maximaBuffer.NativePtr,
            inverseSumsBuffer.NativePtr,
            rowLossesBuffer.NativePtr,
            lossBuffer.NativePtr,
            rows,
            columns,
            ignoreIndex,
            validRows,
            labelSmoothing,
            bfloat16: false);
        return new CrossEntropyResidentContext(
            lossBuffer,
            maximaBuffer,
            inverseSumsBuffer,
            rowLossesBuffer,
            labelsBuffer,
            accelerator);
    }

    internal static void CrossEntropyBackwardResident(
        Tensor logits,
        Tensor loss,
        CrossEntropyResidentContext context,
        int columns,
        int ignoreIndex,
        int validRows,
        float labelSmoothing)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var lossGradientBuffer = loss.EnsureCudaGradientBuffer();
        var logitsGradientBuffer = logits.EnsureCudaGradientBuffer();
        CudaTensorNative.CrossEntropyBackward(
            Tensor.CudaDeviceIndex,
            logits.EnsureCudaFloat32Buffer().NativePtr,
            context.Maxima.NativePtr,
            context.InverseSums.NativePtr,
            context.Labels.NativePtr,
            logitsGradientBuffer.NativePtr,
            lossGradientBuffer.NativePtr,
            logits.Numel,
            columns,
            ignoreIndex,
            validRows,
            labelSmoothing,
            bfloat16: false);
        logits.MarkCudaGradientMutated();
    }

    internal sealed class CrossEntropyResidentContext(
        NativeCudaBuffer<float> loss,
        NativeCudaBuffer<float> maxima,
        NativeCudaBuffer<float> inverseSums,
        NativeCudaBuffer<float> rowLosses,
        NativeCudaBuffer<int> labels,
        NativeCudaDevice accelerator) : IDisposable
    {
        private bool _disposed;
        internal NativeCudaBuffer<float> Loss { get; } = loss;
        internal NativeCudaBuffer<float> Maxima { get; } = maxima;
        internal NativeCudaBuffer<float> InverseSums { get; } = inverseSums;
        internal NativeCudaBuffer<float> RowLosses { get; } = rowLosses;
        internal NativeCudaBuffer<int> Labels { get; } = labels;

        internal void Dispose()
        {
            if (_disposed)
                return;
            Tensor.ReturnCudaFloatBuffer(accelerator, Maxima);
            Tensor.ReturnCudaFloatBuffer(accelerator, InverseSums);
            Tensor.ReturnCudaFloatBuffer(accelerator, RowLosses);
            Tensor.ReturnCudaIntBuffer(accelerator, Labels);
            _disposed = true;
            GC.SuppressFinalize(this);
        }

        void IDisposable.Dispose() => Dispose();

        ~CrossEntropyResidentContext() => Dispose();
    }

    internal static (float Loss, float[] Probabilities) CrossEntropyForward(
        float[] logits,
        int[] labels,
        int rows,
        int columns,
        int ignoreIndex,
        int validRows,
        float labelSmoothing)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var probabilities = new float[logits.Length];
        var loss = new float[1];
        var logitsBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, logits);
        using var labelsBuffer = accelerator.Allocate1D(labels);
        using var probabilitiesBuffer =
            accelerator.Allocate1D<float>(probabilities.Length);
        using var maximaBuffer = accelerator.Allocate1D<float>(rows);
        using var inverseSumsBuffer = accelerator.Allocate1D<float>(rows);
        using var rowLossesBuffer = accelerator.Allocate1D<float>(rows);
        using var lossBuffer = accelerator.Allocate1D<float>(1);
        CudaTensorNative.CrossEntropy(
            Tensor.CudaDeviceIndex,
            logitsBuffer.NativePtr,
            labelsBuffer.NativePtr,
            maximaBuffer.NativePtr,
            inverseSumsBuffer.NativePtr,
            rowLossesBuffer.NativePtr,
            lossBuffer.NativePtr,
            rows,
            columns,
            ignoreIndex,
            validRows,
            labelSmoothing,
            bfloat16: false);
        CudaTensorNative.SoftmaxProbabilities(
            Tensor.CudaDeviceIndex,
            logitsBuffer.NativePtr,
            maximaBuffer.NativePtr,
            inverseSumsBuffer.NativePtr,
            probabilitiesBuffer.NativePtr,
            probabilities.Length,
            columns);
        accelerator.Synchronize();
        probabilitiesBuffer.CopyToCPU(probabilities);
        lossBuffer.CopyToCPU(loss);
        return (loss[0], probabilities);
    }

    internal static void CrossEntropyBackward(
        float[] probabilities,
        int[] labels,
        float[] logitsGradient,
        int columns,
        int ignoreIndex,
        int validRows,
        float labelSmoothing,
        float upstreamGradient)
    {
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        var probabilitiesBuffer = CudaResidentArrayCache.GetOrUpload(accelerator, probabilities);
        using var labelsBuffer = accelerator.Allocate1D(labels);
        using var gradientBuffer = accelerator.Allocate1D(logitsGradient);
        CudaTensorNative.CrossEntropyProbabilitiesBackward(
            Tensor.CudaDeviceIndex,
            probabilitiesBuffer.NativePtr,
            labelsBuffer.NativePtr,
            gradientBuffer.NativePtr,
            probabilities.Length,
            columns,
            ignoreIndex,
            validRows,
            labelSmoothing,
            upstreamGradient);
        accelerator.Synchronize();
        gradientBuffer.CopyToCPU(logitsGradient);
    }
}
