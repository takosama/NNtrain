
namespace NNtrain;

/// <summary>
/// Packs completed leaf gradients into BF16 buckets and overlaps two-GPU
/// exchange with the remaining backward graph on non-blocking CUDA streams.
/// </summary>
internal sealed class CudaBFloat16GradientAllReducePlan : IDisposable
{
    private const int TargetBucketElements = 4 * 1024 * 1024;
    private const int NonPeerTargetBucketElements = 16 * 1024 * 1024;
    private const string BucketElementsEnvironmentVariable =
        "NNTRAIN_GRADIENT_BUCKET_ELEMENTS";
    private const int DefaultHostPipelineChunkElements = 4 * 1024 * 1024;
    private const string HostPipelineChunkElementsEnvironmentVariable =
        "NNTRAIN_GRADIENT_HOST_CHUNK_ELEMENTS";
    private const string DisableHostPipelineEnvironmentVariable =
        "NNTRAIN_DISABLE_GRADIENT_HOST_PIPELINE";
    private readonly Parameter[] _parameters;
    private readonly int[] _devices;
    private readonly Bucket[] _buckets;
    private readonly DeviceBuffers[] _deviceBuffers;
    private readonly Dictionary<Tensor, SegmentLocation> _locations;
    private readonly int[][] _remaining;
    private readonly int[] _readyDeviceCounts;
    private readonly bool _overlapExchange;
    private readonly bool _useHostPipeline;
    private readonly nint[] _hostPipelines;
    private NativeCudaBuffer<double>? _primarySquaredSum;
    private int _disposed;

    internal CudaBFloat16GradientAllReducePlan(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> devices)
    {
        if (devices.Count != 2)
        {
            throw new ArgumentException(
                "Asynchronous BF16 gradient buckets currently require two GPUs.",
                nameof(devices));
        }
        _parameters = parameters.ToArray();
        _devices = devices.ToArray();
        _overlapExchange = NativeCudaRuntime.CanAccessPeer(
                _devices[0], _devices[1])
            && NativeCudaRuntime.CanAccessPeer(_devices[1], _devices[0]);
        _useHostPipeline = !_overlapExchange
            && !string.Equals(
                Environment.GetEnvironmentVariable(
                    DisableHostPipelineEnvironmentVariable),
                "1",
                StringComparison.Ordinal);
        _buckets = BuildBuckets(
            _parameters,
            ResolveTargetBucketElements(_overlapExchange));
        _locations = new Dictionary<Tensor, SegmentLocation>(
            ReferenceEqualityComparer.Instance);
        for (int bucket = 0; bucket < _buckets.Length; bucket++)
        {
            foreach (Segment segment in _buckets[bucket].Segments)
                _locations.Add(segment.Tensor, new SegmentLocation(bucket, segment));
        }
        _remaining = Enumerable.Range(0, _buckets.Length)
            .Select(bucket => new int[_devices.Length])
            .ToArray();
        _readyDeviceCounts = new int[_buckets.Length];
        _deviceBuffers = new DeviceBuffers[_devices.Length];
        _hostPipelines = new nint[_devices.Length];
        try
        {
            for (int device = 0; device < _devices.Length; device++)
                _deviceBuffers[device] = CreateDeviceBuffers(device);
            _primarySquaredSum = _deviceBuffers[0].Accelerator
                .Allocate1D<double>(1);
            if (_useHostPipeline)
            {
                int chunkElements = ResolveHostPipelineChunkElements();
                for (int destination = 0;
                    destination < _devices.Length;
                    destination++)
                {
                    int source = 1 - destination;
                    _hostPipelines[destination] =
                        CudaGradientBuckets.CreateHostPipeline(
                            _devices[source],
                            _devices[destination],
                            chunkElements);
                }
            }
            BindGradientArenas();
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    internal bool Matches(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> devices)
        => parameters.Count == _parameters.Length
            && devices.SequenceEqual(_devices)
            && parameters.Select((parameter, index) =>
                ReferenceEquals(parameter, _parameters[index])).All(value => value);

    internal bool DefersExchangeUntilBackward => !_overlapExchange;

    internal void BeginStep()
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0, this);
        _primarySquaredSum!.MemSetToZero();
        for (int bucket = 0; bucket < _buckets.Length; bucket++)
        {
            Volatile.Write(ref _readyDeviceCounts[bucket], 0);
            for (int device = 0; device < _devices.Length; device++)
            {
                _deviceBuffers[device].Buckets[bucket]
                    .GradientArena.ClearIfDirty();
                Volatile.Write(
                    ref _remaining[bucket][device],
                    _buckets[bucket].Segments.Length);
            }
        }
    }

    internal void NotifyGradientReady(Tensor tensor, int deviceIndex)
    {
        if (!_locations.TryGetValue(tensor, out SegmentLocation location))
            return;
        int deviceSlot = Array.IndexOf(_devices, deviceIndex);
        if (deviceSlot < 0)
            return;
        DeviceBuffers deviceBuffers = _deviceBuffers[deviceSlot];
        NativeCudaDevice accelerator = deviceBuffers.Accelerator;
        BucketBuffers bucketBuffers = deviceBuffers.Buckets[location.Bucket];

        if (Interlocked.Decrement(
            ref _remaining[location.Bucket][deviceSlot]) != 0)
        {
            return;
        }
        CudaGradientBuckets.Pack(
            deviceIndex,
            accelerator,
            bucketBuffers.GradientArena.Buffer,
            bucketBuffers.Local,
            0,
            _buckets[location.Bucket].TotalElements);
        CudaGradientBuckets.RecordReady(
            deviceIndex,
            accelerator,
            bucketBuffers.ReadyEvent);
        if (Interlocked.Increment(
                ref _readyDeviceCounts[location.Bucket]) == _devices.Length
            && _overlapExchange)
        {
            try
            {
                EnqueueBucketReduction(location.Bucket);
            }
            finally
            {
                accelerator.Bind();
            }
        }
    }

    internal void Complete()
    {
        for (int bucket = 0; bucket < _buckets.Length; bucket++)
        {
            for (int device = 0; device < _devices.Length; device++)
            {
                if (Volatile.Read(ref _remaining[bucket][device]) != 0)
                {
                    throw new InvalidOperationException(
                        $"Gradient bucket {bucket} was not completed on " +
                        $"CUDA device {_devices[device]}.");
                }
            }
        }
        if (_useHostPipeline)
        {
            // Consumer/WDDM pairs commonly have no peer mapping.  Exchange
            // each direction through two persistent pinned-host slots so the
            // next D2H copy overlaps the preceding H2D copy and reduction.
            for (int bucket = 0; bucket < _buckets.Length; ++bucket)
            {
                int capturedBucket = bucket;
                Parallel.For(0, _devices.Length, destination =>
                    EnqueueBucketReduction(capturedBucket, destination));
            }
        }
        else
        {
            if (!_overlapExchange)
            {
                for (int bucket = 0; bucket < _buckets.Length; bucket++)
                    EnqueueBucketReduction(bucket);
            }
            Parallel.For(0, _devices.Length, device =>
            {
                DeviceBuffers buffers = _deviceBuffers[device];
                CudaGradientBuckets.Synchronize(
                    buffers.Accelerator,
                    buffers.DeviceIndex,
                    buffers.CommunicationStream);
            });
        }
        var squaredSum = new double[1];
        _primarySquaredSum!.CopyToCPU(squaredSum);
        foreach (Parameter parameter in _parameters)
            parameter.T.MarkCudaGradientsSynchronized(_devices);
        TensorCudaKernels.PublishGradientSquaredSum(
            _parameters, _devices, squaredSum[0]);
    }

    private void EnqueueBucketReduction(int bucketIndex)
    {
        for (int destination = 0; destination < _devices.Length; destination++)
            EnqueueBucketReduction(bucketIndex, destination);
    }

    private void EnqueueBucketReduction(int bucketIndex, int destination)
    {
        Bucket bucket = _buckets[bucketIndex];
        int source = 1 - destination;
        DeviceBuffers destinationBuffers = _deviceBuffers[destination];
        BucketBuffers destinationBucket =
            destinationBuffers.Buckets[bucketIndex];
        BucketBuffers sourceBucket = _deviceBuffers[source].Buckets[bucketIndex];
        if (_useHostPipeline)
        {
            CudaGradientBuckets.HostPipelineExchange(
                destinationBuffers.Accelerator,
                _hostPipelines[destination],
                destinationBucket.Local,
                sourceBucket.Local,
                destinationBucket.GradientArena.Buffer,
                bucket.TotalElements,
                destination == 0
                    ? _primarySquaredSum!.NativePtr
                    : 0,
                destinationBucket.ReadyEvent,
                sourceBucket.ReadyEvent);
        }
        else
        {
            CudaGradientBuckets.Exchange(
                destinationBuffers.Accelerator,
                destinationBuffers.DeviceIndex,
                _deviceBuffers[source].DeviceIndex,
                destinationBucket.Local,
                sourceBucket.Local,
                destinationBucket.Remote!,
                destinationBucket.GradientArena.Buffer,
                bucket.TotalElements,
                destination == 0
                    ? _primarySquaredSum!.NativePtr
                    : 0,
                destinationBuffers.CommunicationStream,
                destinationBucket.ReadyEvent,
                sourceBucket.ReadyEvent);
        }
        destinationBucket.GradientArena.MarkDirty();
    }

    private DeviceBuffers CreateDeviceBuffers(int deviceSlot)
    {
        int deviceIndex = _devices[deviceSlot];
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        nint stream = CudaGradientBuckets.CreateCommunicationStream(
            accelerator, deviceIndex);
        var buckets = new BucketBuffers[_buckets.Length];
        try
        {
            for (int bucket = 0; bucket < _buckets.Length; bucket++)
            {
                int length = _buckets[bucket].TotalElements;
                buckets[bucket] = CreateBucketBuffers(
                    accelerator, deviceIndex, length);
            }
            return new DeviceBuffers(deviceIndex, accelerator, stream, buckets);
        }
        catch
        {
            foreach (BucketBuffers? bucket in buckets)
                bucket?.Dispose(accelerator, deviceIndex);
            CudaGradientBuckets.DestroyCommunicationStream(
                accelerator, deviceIndex, stream);
            throw;
        }
    }

    private BucketBuffers CreateBucketBuffers(
        NativeCudaDevice accelerator,
        int deviceIndex,
        int length)
    {
        NativeCudaBuffer<ushort>? local = null;
        NativeCudaBuffer<ushort>? remote = null;
        NativeCudaArena<float>? gradientArena = null;
        nint readyEvent = 0;
        try
        {
            local = accelerator.Allocate1D<ushort>(length);
            if (!_useHostPipeline)
                remote = accelerator.Allocate1D<ushort>(length);
            gradientArena = new NativeCudaArena<float>(accelerator, length);
            readyEvent = CudaGradientBuckets.CreateReadyEvent(
                accelerator, deviceIndex);
            return new BucketBuffers(
                local, remote, gradientArena, readyEvent);
        }
        catch
        {
            if (readyEvent != 0)
            {
                CudaGradientBuckets.DestroyEvent(
                    accelerator, deviceIndex, readyEvent);
            }
            gradientArena?.Dispose();
            remote?.Dispose();
            local?.Dispose();
            throw;
        }
    }

    private void BindGradientArenas()
    {
        for (int device = 0; device < _devices.Length; device++)
        {
            int deviceIndex = _devices[device];
            for (int bucket = 0; bucket < _buckets.Length; bucket++)
            {
                NativeCudaArena<float> arena =
                    _deviceBuffers[device].Buckets[bucket].GradientArena;
                foreach (Segment segment in _buckets[bucket].Segments)
                {
                    segment.Tensor.BindCudaGradientArena(
                        deviceIndex,
                        arena.Slice(segment.Offset, segment.Length));
                }
            }
        }
    }

    private void UnbindGradientArenas()
    {
        for (int device = 0; device < _devices.Length; device++)
        {
            DeviceBuffers? buffers = _deviceBuffers[device];
            if (buffers is null)
                continue;
            int deviceIndex = _devices[device];
            for (int bucket = 0; bucket < _buckets.Length; bucket++)
            {
                NativeCudaArena<float> arena =
                    buffers.Buckets[bucket].GradientArena;
                foreach (Segment segment in _buckets[bucket].Segments)
                    segment.Tensor.UnbindCudaGradientArena(deviceIndex, arena);
            }
        }
    }

    private static int ResolveTargetBucketElements(bool peerAccess)
    {
        string? configured = Environment.GetEnvironmentVariable(
            BucketElementsEnvironmentVariable);
        return int.TryParse(configured, out int elements) && elements > 0
            ? elements
            : peerAccess
                ? TargetBucketElements
                : NonPeerTargetBucketElements;
    }

    private static int ResolveHostPipelineChunkElements()
    {
        string? configured = Environment.GetEnvironmentVariable(
            HostPipelineChunkElementsEnvironmentVariable);
        return int.TryParse(configured, out int elements) && elements > 0
            ? elements
            : DefaultHostPipelineChunkElements;
    }

    private static Bucket[] BuildBuckets(
        IReadOnlyList<Parameter> parameters,
        int targetBucketElements)
    {
        var buckets = new List<Bucket>();
        var segments = new List<Segment>();
        int offset = 0;
        void FinishBucket()
        {
            if (segments.Count == 0)
                return;
            buckets.Add(new Bucket(segments.ToArray(), offset));
            segments.Clear();
            offset = 0;
        }

        foreach (Parameter parameter in parameters)
        {
            int length = parameter.T.Numel;
            if (segments.Count > 0
                && offset + (long)length > targetBucketElements)
            {
                FinishBucket();
            }
            segments.Add(new Segment(parameter.T, offset, length));
            offset = checked(offset + length);
            if (offset >= targetBucketElements)
                FinishBucket();
        }
        FinishBucket();
        return buckets.ToArray();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        UnbindGradientArenas();
        for (int destination = 0;
            destination < _hostPipelines.Length;
            destination++)
        {
            DeviceBuffers? buffers = _deviceBuffers[destination];
            NativeCudaDevice accelerator = buffers?.Accelerator
                ?? ForgetMemoryV2Cuda.GetAccelerator(_devices[destination]);
            CudaGradientBuckets.DestroyHostPipeline(
                accelerator, _hostPipelines[destination]);
        }
        foreach (DeviceBuffers? device in _deviceBuffers)
            device?.Dispose();
        _primarySquaredSum?.Dispose();
        _primarySquaredSum = null;
    }

    private sealed record Bucket(Segment[] Segments, int TotalElements);
    private sealed record Segment(Tensor Tensor, int Offset, int Length);
    private readonly record struct SegmentLocation(int Bucket, Segment Segment);

    private sealed class BucketBuffers(
        NativeCudaBuffer<ushort> local,
        NativeCudaBuffer<ushort>? remote,
        NativeCudaArena<float> gradientArena,
        nint readyEvent)
    {
        internal NativeCudaBuffer<ushort> Local { get; } = local;
        internal NativeCudaBuffer<ushort>? Remote { get; } = remote;
        internal NativeCudaArena<float> GradientArena { get; } = gradientArena;
        internal nint ReadyEvent { get; } = readyEvent;

        internal void Dispose(
            NativeCudaDevice accelerator,
            int deviceIndex)
        {
            CudaGradientBuckets.DestroyEvent(
                accelerator, deviceIndex, ReadyEvent);
            Local.Dispose();
            Remote?.Dispose();
            GradientArena.Dispose();
        }
    }

    private sealed class DeviceBuffers(
        int deviceIndex,
        NativeCudaDevice accelerator,
        nint communicationStream,
        BucketBuffers[] buckets) : IDisposable
    {
        internal int DeviceIndex { get; } = deviceIndex;
        internal NativeCudaDevice Accelerator { get; } = accelerator;
        internal nint CommunicationStream { get; } = communicationStream;
        internal BucketBuffers[] Buckets { get; } = buckets;

        public void Dispose()
        {
            CudaGradientBuckets.Synchronize(
                Accelerator, DeviceIndex, CommunicationStream);
            foreach (BucketBuffers bucket in Buckets)
                bucket.Dispose(Accelerator, DeviceIndex);
            CudaGradientBuckets.DestroyCommunicationStream(
                Accelerator, DeviceIndex, CommunicationStream);
        }
    }
}

internal static class CudaGradientReductionContext
{
    private static readonly AsyncLocal<Entry?> Current = new();

    internal static IDisposable Push(
        CudaBFloat16GradientAllReducePlan plan,
        int deviceIndex)
    {
        Entry? previous = Current.Value;
        Current.Value = new Entry(plan, deviceIndex);
        return new Scope(previous);
    }

    internal static void NotifyLeaf(Tensor tensor)
    {
        Entry? current = Current.Value;
        current?.Plan.NotifyGradientReady(tensor, current.DeviceIndex);
    }

    private sealed record Entry(
        CudaBFloat16GradientAllReducePlan Plan,
        int DeviceIndex);

    private sealed class Scope(Entry? previous) : IDisposable
    {
        private int _disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            Current.Value = previous;
        }
    }
}
