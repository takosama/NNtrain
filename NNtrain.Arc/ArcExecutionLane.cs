using System.Runtime.InteropServices;
using System.Text;
using System.Diagnostics;
using NNtrain.Runtime.Execution;

namespace NNtrain.Arc;

/// <summary>
/// Session-owned OpenCL lane for resident training and an explicit staged A/B reference.
/// All native allocations and the in-order queue are owned by this session.
/// </summary>
public sealed partial class ArcExecutionLane : IExecutionLane, IDeviceMemoryManager, IKernelCapabilitySet
{
    private static readonly string[] Qwen35KernelResourceSuffixes =
    [
        ".qwen.cl", ".qwen35_attention.cl", ".qwen35_delta.cl",
        ".qwen35_delta_fused.cl", ".qwen35_linear_fast.cl", ".qwen35_iq.cl", ".qwen35_lora.cl", ".qwen35_prism.cl",
        ".qwen35_projection_pair.cl", ".qwen35_norm_fast.cl", ".qwen35_prefill_linear.cl", ".qwen35_prefill_xmx.cl", ".qwen35_prefill_xmx_tiles.cl", ".qwen35_vision.cl",
        ".qwen35_vision_attention_fast.cl", ".qwen35_vision_attention_xmx.cl", ".qwen35_vision_linear_fast.cl"
    ];

    private static readonly string[] Qwen35TrainingResourceSuffixes =
        [".qwen35_train.cl", ".qwen35_train_attention.cl", ".qwen35_train_attention_packed.cl", ".qwen35_train_attention_streamed.cl", ".qwen35_train_attention_rowfused.cl", ".qwen35_train_attention_streamed_rowfused.cl", ".qwen35_train_delta.cl", ".qwen35_train_linear.cl", ".qwen35_train_embedding.cl", ".qwen35_train_transpose.cl"];

    private static readonly string[] Qwen35VisionResourceSuffixes =
        [".qwen35_vision.cl", ".qwen35_vision_attention_fast.cl", ".qwen35_vision_attention_xmx.cl", ".qwen35_vision_linear_fast.cl"];

    private readonly object _sync = new();
    private readonly Dictionary<string, nint> _kernels = [];
    private readonly HashSet<ArcBuffer> _buffers = [];
    private readonly Dictionary<long, Stack<nint>> _pool = [];
    private readonly Dictionary<long, LinkedList<CachedBuffer>> _lruPool = [];
    private readonly LinkedList<CachedBuffer> _freeLru = [];
    private readonly List<(nint Handle, long Bytes)> _retired = [];
    // Entries also count eventless kernels: retirement and queue-window
    // safety depend on pending commands, not on whether timing is enabled.
    private readonly List<(string Name, nint Event, string? Label)> _pendingEvents = [];
    private readonly List<(string Kind, nint Event, long Bytes)> _pendingCopyEvents = [];
    private bool _queuedCopies;
    private nint _context, _queue, _program;
    private bool _disposed;
    private ArcBuffer? _numericStatus;
    private sealed class CachedBuffer
    {
        internal readonly nint Handle;
        internal readonly long Bytes;
        internal readonly LinkedListNode<CachedBuffer> SizeNode, RecencyNode;
        internal CachedBuffer(nint handle, long bytes)
        {
            Handle = handle; Bytes = bytes;
            SizeNode = new(this); RecencyNode = new(this);
        }
    }
    public ArcDeviceInfo Device { get; }
    public ArcExecutionOptions Options { get; }
    public ArcDetailedProfiler? DetailedProfiler { get; }
    private bool CollectKernelEvents => Options.CollectKernelTimings || DetailedProfiler is not null || Timeline is not null;
    public void ResetDetailedProfile()
    {
        lock (_sync)
        {
            if (_pendingEvents.Count != 0 || _queuedCopies) throw new InvalidOperationException("Synchronize before resetting the Arc profile.");
            DetailedProfiler?.Reset();
        }
    }
    public ExecutionDeviceKind DeviceKind => ExecutionDeviceKind.Arc;
    public int DeviceIndex => Device.Index;
    public IDeviceMemoryManager MemoryManager => this;
    public IKernelCapabilitySet Capabilities => this;
    public IExecutionProfiler Profiler => NullExecutionProfiler.Instance;
    public long AllocationCount { get; private set; }
    /// <summary>Cumulative logical requests; not simultaneous VRAM use.</summary>
    public long RequestedBytes { get; private set; }
    /// <summary>Cumulative successful clCreateBuffer bytes, including initial model upload.</summary>
    public long NativeAllocatedBytes { get; private set; }
    public long NativeReleasedBytes { get; private set; }
    public long NativeReleaseCount { get; private set; }
    public long AllocatedBytes { get; private set; }
    public long KernelLaunchCount { get; private set; }
    public long CachedBytes { get; private set; }
    public long RetiredBytes { get; private set; }
    public long PoolHits { get; private set; }
    public long RetiredReuseCount { get; private set; }
    public long H2DBytes { get; private set; }
    public long D2HBytes { get; private set; }
    public long PeakAllocatedBytes { get; private set; }
    public double KernelMilliseconds { get; private set; }
    /// <summary>Blocking readback wall time (includes any wait for the preceding kernel).</summary>
    public double TransferMilliseconds { get; private set; }
    /// <summary>Native allocation plus initial host upload wall time.</summary>
    public double AllocationMilliseconds { get; private set; }
    public Dictionary<string, double> KernelTimings { get; } = [];
    public ArcBuffer NumericStatus
    {
        get
        {
            lock (_sync)
            {
                if (_numericStatus is null)
                {
                    _numericStatus = Allocate(1);
                    Run(Options.Qwen35VisionKernelsOnly ? "q35v_zero"
                        : Options.Qwen35InferenceKernelsOnly ? "q35a_zero" : "resident_zero",
                        1, 0, _numericStatus, 1);
                }
                return _numericStatus;
            }
        }
    }

    public void CheckNumericStatus()
    {
        if (_numericStatus is null) return;
        var status = new int[1]; ReadRaw(_numericStatus, status);
        if (status[0] != 0) throw new ArithmeticException("Arc BFP8 publication encountered non-finite values. The optimizer must not commit this step.");
    }
    public bool Supports(string feature) => Options.Qwen35VisionKernelsOnly
        ? feature is "qwen35-vision" or "float32" : Options.Qwen35InferenceKernelsOnly
        ? feature is "qwen35-inference" or "float32"
        : feature is "transformer" or "float32" or "mix16_32" or "mix8_32" or "mix8_16";

    public ArcExecutionLane(int deviceIndex = 0, ArcExecutionOptions? options = null)
    {
        Options = options ?? new();
        if (Options.AsrKernelsOnly && (!Options.Qwen35InferenceKernelsOnly || Options.Qwen35VisionKernelsOnly || Options.Qwen35TrainingKernels))
            throw new ArgumentException("An ASR lane requires inference-only kernels without vision/training sources.", nameof(options));
        if (Options.Qwen35VisionKernelsOnly && (!Options.Qwen35InferenceKernelsOnly || Options.Qwen35TrainingKernels))
            throw new ArgumentException("A vision-only lane must select inference kernels without training kernels.", nameof(options));
        DetailedProfiler = Options.DetailedProfiling ? new() : null;
        ArgumentOutOfRangeException.ThrowIfNegative(Options.BufferPoolBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(Options.DeferredReleaseBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(Options.PhysicalBufferBudgetBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(Options.TransformerCheckpointLayers);
        if (Options.MatrixPanelCacheMiB is < 0 or > 1024)
            throw new ArgumentOutOfRangeException(nameof(options), "Matrix panel cache must be 0..1024 MiB.");
        if (Options.AttentionDkvRows is not (2 or 4 or 8 or 16)) throw new ArgumentOutOfRangeException(nameof(options), "DKV rows must be 2, 4, 8 or 16.");
        if (Options.StreamedWeightGradientWorkspaceMiB is < 0 or > 256)
            throw new ArgumentOutOfRangeException(nameof(options), "Streamed weight-gradient workspace must be between 0 and 256 MiB.");
        if (Options.QueuedKernelLimit is < 16 or > 4096) throw new ArgumentOutOfRangeException(nameof(Options.QueuedKernelLimit));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Options.LossChunkRows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Options.LossLogitsWorkspaceMiB);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Options.LossPanelWorkspaceMiB);
        if (Options.Qwen35ProjectionWorkgroupSize is not (32 or 64 or 128)
            || Options.Qwen35PairedProjectionTypes is < 0 or > 7)
            throw new ArgumentOutOfRangeException(nameof(options));
        if (Options.Qwen35LoraReductionSize is not (16 or 128 or 256 or 512 or 1024))
            throw new ArgumentOutOfRangeException(nameof(options));
        if (!Enum.IsDefined(Options.XmxGemmMode)) throw new ArgumentOutOfRangeException(nameof(options));
        if (Options.AttentionWorkspaceMiB is < 8 or > 256)
            throw new ArgumentOutOfRangeException(nameof(options), "Attention workspace must be between 8 and 256 MiB.");
        Device = ArcDevices.Get(deviceIndex);
        try
        {
            _context = OpenClNative.clCreateContext(0, 1, [Device.NativeDevice], 0, 0, out int error);
            OpenClNative.Check(error, "create context");
            _queue = OpenClNative.clCreateCommandQueue(_context, Device.NativeDevice, 2, out error);
            OpenClNative.Check(error, "create queue");
            var sourceText = new StringBuilder();
            var assembly = typeof(ArcExecutionLane).Assembly;
            int resourceCount = 0;
            string[] selectedResources = Options.AsrKernelsOnly ? [".asr_linear.cl"] : Options.Qwen35VisionKernelsOnly
                ? Qwen35VisionResourceSuffixes : Qwen35KernelResourceSuffixes;
            foreach (string resourceName in assembly.GetManifestResourceNames().Where(n => n.EndsWith(".cl", StringComparison.Ordinal)).Order(StringComparer.Ordinal))
            {
                if (!Options.AsrKernelsOnly && resourceName.EndsWith(".asr_linear.cl", StringComparison.Ordinal)) continue;
                if (Options.Qwen35InferenceKernelsOnly
                    && !selectedResources.Any(suffix => resourceName.EndsWith(suffix, StringComparison.Ordinal))
                    && !(Options.Qwen35TrainingKernels && Qwen35TrainingResourceSuffixes.Any(suffix => resourceName.EndsWith(suffix, StringComparison.Ordinal))))
                    continue;
                // Flash has its own compiler policy; never change GEMM/codec
                // register allocation to accommodate a different kernel group.
                if (resourceName.EndsWith(".attention_flash.cl", StringComparison.Ordinal)
                    || resourceName.EndsWith(".attention_xmx_products.cl", StringComparison.Ordinal)) continue;
                using Stream resource = assembly.GetManifestResourceStream(resourceName)!;
                using var reader = new StreamReader(resource);
                sourceText.AppendLine(reader.ReadToEnd());
                resourceCount++;
            }
            if (sourceText.Length == 0) throw new InvalidOperationException("Arc OpenCL kernels are missing.");
            if (Options.Qwen35InferenceKernelsOnly && resourceCount != selectedResources.Length + (Options.Qwen35TrainingKernels ? Qwen35TrainingResourceSuffixes.Length : 0))
                throw new InvalidOperationException("The dedicated Qwen3.5 OpenCL kernel resources are incomplete.");
            byte[] source = Encoding.UTF8.GetBytes(sourceText.ToString());
            string buildOptions = "-cl-std=CL1.2 -cl-fp32-correctly-rounded-divide-sqrt";
            buildOptions += $" -DQ35_PROJECTION_WG={Options.Qwen35ProjectionWorkgroupSize}";
            if (Options.Qwen35UnrollQ4) buildOptions += " -DQ35_Q4_UNROLL=1";
            if (Options.Qwen35NativeHalfScale) buildOptions += " -DQ35_NATIVE_HALF=1";
            if (Options.Qwen35VisionFlashAttention) buildOptions += " -DARC_VISION_FLASH=1";
            if (Options.Qwen35DeltaSubgroupRms) buildOptions += " -DARC_DELTA_SUBGROUP_RMS=1";
            if (Options.Qwen35TrainingKernels || !Options.Qwen35InferenceKernelsOnly) buildOptions += " -DARC_QWEN35_TRAINING=1";
            if (Options.ExperimentalOptimizationKernels) buildOptions += " -DARC_OPTIMIZATION_PROBES=1";
            if (Options.Qwen35ResidentIq2Panels) buildOptions += " -DARC_Q35_RESIDENT_IQ2=1";
            if (Options.Qwen35GgufBslmPrefill) buildOptions += " -DARC_Q35_GGUF_BSLM=1";
            if (Options.Mix8_16Int8Linear) buildOptions += " -DARC_INT8_LINEAR=1";
            if (Device.SupportsXmx && Options.XmxMatrices)
                buildOptions += $" -DARC_XMX=1 -DARC_SG={Device.MinimumSubgroupSize}";
            if (Device.Extensions.Split(' ').Contains("cl_intel_subgroup_local_block_io"))
                buildOptions += " -DARC_SLM_BLOCK_IO=1";
            string? cachePath = ProgramCachePath(source, buildOptions);
            if (!TryLoadProgramCache(cachePath, buildOptions))
            {
                GCHandle pin = GCHandle.Alloc(source, GCHandleType.Pinned);
                try { _program = OpenClNative.clCreateProgramWithSource(_context, 1, [pin.AddrOfPinnedObject()], [(nuint)source.Length], out error); }
                finally { pin.Free(); }
                OpenClNative.Check(error, "create program");
                error = OpenClNative.clBuildProgram(_program, 1, [Device.NativeDevice], buildOptions, 0, 0);
                if (error != 0)
                {
                    OpenClNative.clGetProgramBuildInfo(_program, Device.NativeDevice, 0x1183, 0, null, out nuint length);
                    byte[] log = new byte[checked((int)length)];
                    OpenClNative.clGetProgramBuildInfo(_program, Device.NativeDevice, 0x1183, length, log, out _);
                    throw new InvalidOperationException($"Arc OpenCL kernel build failed ({error}): {Encoding.UTF8.GetString(log)}");
                }
                SaveProgramCache(cachePath);
            }
        }
        catch { Dispose(); throw; }
    }

    private nint ProgramForRequestedKernel(string name)
    {
        if (Options.AsrKernelsOnly)
            return name.StartsWith("asr_", StringComparison.Ordinal) ? _program
                : throw new NotSupportedException("This Arc lane only contains ASR kernels.");
        if (Options.Qwen35VisionKernelsOnly)
            return name.StartsWith("q35v_", StringComparison.Ordinal) ? _program
                : throw new NotSupportedException("This Arc lane only contains Qwen3.5 vision kernels.");
        if (!Options.Qwen35InferenceKernelsOnly) return ProgramForKernel(name);
        // Do not silently compile standalone training programs when a caller
        // requests a kernel outside the explicitly selected inference workload.
        if ((Options.Qwen35TrainingKernels && name.StartsWith("q35t_", StringComparison.Ordinal))
            || name.StartsWith("qwen_", StringComparison.Ordinal)
            || name.StartsWith("q35a_", StringComparison.Ordinal)
            || name.StartsWith("q35d_", StringComparison.Ordinal)
            || name.StartsWith("q35l_", StringComparison.Ordinal)
            || name.StartsWith("q35v_", StringComparison.Ordinal)) return _program;
        throw new NotSupportedException("This Arc lane only contains Qwen3.5 inference kernels.");
    }

    public ArcBuffer Allocate(int elements)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(elements);
        lock (_sync) return AllocateCore(checked((long)Math.Max(1, elements) * 4), null);
    }

    /// <summary>Driver-reported resource usage; unsupported queries remain null, not zero.</summary>
    public ArcKernelResources GetKernelResources(string name)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!_kernels.TryGetValue(name, out nint kernel))
            {
                kernel = OpenClNative.clCreateKernel(ProgramForRequestedKernel(name), name, out int error);
                OpenClNative.Check(error, $"create kernel {name}");
                _kernels.Add(name, kernel);
            }
            ulong? Info(uint field) => OpenClNative.clGetKernelWorkGroupInfo(kernel, Device.NativeDevice,
                field, 8, out ulong value, out _) == 0 ? value : null;
            return new(name, Info(0x11b0), Info(0x11b2), Info(0x11b4), Info(0x4109));
        }
    }

    public ArcBuffer Upload(float[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        lock (_sync) return AllocateCore(Math.Max(4L, values.LongLength * 4), values);
    }

    public ArcBuffer AllocateBytes(int bytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        lock (_sync) return AllocateCore(Math.Max(4, bytes), null);
    }

    public void ReadRaw(ArcBuffer buffer, Array values)
    {
        lock (_sync) Transfer(buffer, values, read: true);
    }

    public void WriteRaw(ArcBuffer buffer, Array values)
    {
        lock (_sync) Transfer(buffer, values, read: false);
    }

    public void ReadFloatRange(ArcBuffer buffer, int elementOffset, float[] values)
    {
        lock (_sync) Transfer(buffer, values, read: true, checked(elementOffset * 4L));
    }

    public void ReadBFloat16Range(ArcBuffer buffer, int elementOffset, ushort[] values)
    {
        lock (_sync) Transfer(buffer, values, read: true, checked(elementOffset * 2L));
    }

    public unsafe void CopyBytes(ArcBuffer source, ArcBuffer target, int sourceOffset, int targetOffset, int bytes)
    {
        lock (_sync)
        {
            if (source.Owner != this || target.Owner != this || !source.IsAlive || !target.IsAlive
                || sourceOffset < 0 || targetOffset < 0 || bytes < 0
                || sourceOffset + (long)bytes > source.Bytes || targetOffset + (long)bytes > target.Bytes)
                throw new ArgumentException("Invalid Arc device copy range.");
            if (bytes == 0) return;
            nint evt = 0;
            using (Timeline?.Host("queue-submit", "D2D"))
                OpenClNative.Check(OpenClNative.clEnqueueCopyBuffer(_queue, source.Handle, target.Handle,
                    (nuint)sourceOffset, (nuint)targetOffset, (nuint)bytes, 0, 0, Timeline is null ? 0 : (nint)(&evt)), "device copy");
            if (evt != 0) _pendingCopyEvents.Add(("D2D", evt, bytes));
            _queuedCopies = true;
            if (!Options.BatchDispatch) Synchronize();
        }
    }

    public ArcBuffer UploadRaw(Array values)
    {
        if (values is not (float[] or int[] or ushort[] or short[] or byte[] or sbyte[]))
            throw new ArgumentException("Arc raw upload requires a primitive numeric array.");
        lock (_sync) return AllocateCore(Math.Max(4, Buffer.ByteLength(values)), values);
    }

    public void Write(ArcBuffer buffer, float[] values)
    {
        lock (_sync) Transfer(buffer, values, read: false);
    }

    public void Read(ArcBuffer buffer, float[] values)
    {
        lock (_sync) Transfer(buffer, values, read: true);
    }

    /// <summary>
    /// Queue a readback on this lane's in-order queue without waiting for the
    /// preceding kernels. The destination stays pinned until the returned
    /// handle is completed or disposed. The source buffer must also remain
    /// undisposed until then. The caller may enqueue later kernels before
    /// waiting, provided it does not inspect the host array early.
    /// </summary>
    public unsafe PendingRead ReadAsync(ArcBuffer buffer, float[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (buffer.Owner != this || !buffer.IsAlive || (long)values.Length * sizeof(float) > buffer.Bytes)
                throw new ArgumentException("Arc transfer buffer is disposed, too small, or belongs to a different lane.", nameof(buffer));
            if (values.Length == 0) throw new ArgumentException("Asynchronous Arc read requires a nonempty destination.", nameof(values));
            GCHandle pin = GCHandle.Alloc(values, GCHandleType.Pinned);
            nint evt = 0;
            bool queued = false;
            try
            {
                long bytes = checked((long)values.Length * sizeof(float));
                using (Timeline?.Host("queue-submit", "D2H-async"))
                    OpenClNative.Check(OpenClNative.clEnqueueReadBuffer(_queue, buffer.Handle, 0,
                        0, (nuint)bytes, pin.AddrOfPinnedObject(), 0, 0, (nint)(&evt)), "queue asynchronous download");
                queued = true;
                if (evt == 0) throw new InvalidOperationException("OpenCL did not return a readback event.");
                _queuedCopies = true;
                D2HBytes += bytes;
                return new PendingRead(this, buffer, values, pin, evt, bytes);
            }
            catch
            {
                bool completed = !queued;
                if (queued)
                {
                    nint pending = evt;
                    if (pending != 0)
                    {
                        OpenClNative.clFlush(_queue);
                        completed = OpenClNative.clWaitForEvents(1, (nint)(&pending)) == 0;
                    }
                    if (!completed) completed = OpenClNative.clFinish(_queue) == 0;
                }
                if (completed)
                {
                    if (evt != 0) OpenClNative.clReleaseEvent(evt);
                    pin.Free();
                }
                throw;
            }
        }
    }

    public sealed class PendingRead : IDisposable
    {
        private readonly ArcExecutionLane _lane;
        private readonly ArcBuffer _source;
        // Retain the managed destination as well as its pin until the event completes.
        private readonly float[] _destination;
        private GCHandle _pin;
        private nint _event;
        private readonly long _bytes;

        internal PendingRead(ArcExecutionLane lane, ArcBuffer source, float[] destination,
            GCHandle pin, nint evt, long bytes)
            => (_lane, _source, _destination, _pin, _event, _bytes) = (lane, source, destination, pin, evt, bytes);

        public unsafe void Wait()
        {
            lock (_lane._sync)
            {
                if (_event == 0) return;
                long start = Stopwatch.GetTimestamp();
                bool completed = false;
                try
                {
                    OpenClNative.Check(OpenClNative.clFlush(_lane._queue), "flush asynchronous download");
                    nint evt = _event;
                    OpenClNative.Check(OpenClNative.clWaitForEvents(1, (nint)(&evt)), "wait asynchronous download");
                    completed = true;
                    _lane.Timeline?.Device("D2H", _lane.DetailedProfiler?.Phase, _event, "D2H", _bytes);
                    _lane.TransferMilliseconds += Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                    _lane.DetailedProfiler?.Add("D2H-async-wait", $"{_lane.DetailedProfiler.Phase}/{_bytes}B",
                        Stopwatch.GetElapsedTime(start).TotalMilliseconds, _bytes);
                    GC.KeepAlive(_source);
                    GC.KeepAlive(_destination);
                }
                finally
                {
                    if (!completed && _lane._queue != 0)
                        completed = OpenClNative.clFinish(_lane._queue) == 0;
                    // If OpenCL cannot establish completion, keep the array
                    // pinned and the source referenced rather than risking a
                    // native DMA into relocated managed memory.
                    if (completed)
                    {
                        OpenClNative.clReleaseEvent(_event);
                        _event = 0;
                        _pin.Free();
                    }
                }
            }
        }

        public void Dispose() => Wait();
    }

    private unsafe void Transfer(ArcBuffer buffer, Array values, bool read, long offset = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (buffer.Owner != this || buffer.Handle == 0 || offset < 0 || Buffer.ByteLength(values) + offset > buffer.Bytes)
            throw new ArgumentException("Arc transfer buffer is disposed, too small, or belongs to a different lane.");
        if (values.Length == 0) return;
        GCHandle pin = GCHandle.Alloc(values, GCHandleType.Pinned);
        long start = Stopwatch.GetTimestamp();
        nint evt = 0;
        try
        {
            nuint bytes = (nuint)Buffer.ByteLength(values);
            int result;
            using (Timeline?.Host("blocking-transfer-wait", read ? "D2H" : "H2D"))
                result = read
                    ? OpenClNative.clEnqueueReadBuffer(_queue, buffer.Handle, 1, (nuint)offset, bytes, pin.AddrOfPinnedObject(), 0, 0, Timeline is null ? 0 : (nint)(&evt))
                    : OpenClNative.clEnqueueWriteBuffer(_queue, buffer.Handle, 1, (nuint)offset, bytes, pin.AddrOfPinnedObject(), 0, 0, Timeline is null ? 0 : (nint)(&evt));
            OpenClNative.Check(result, read ? "download" : "upload");
            Timeline?.Device(read ? "D2H" : "H2D", DetailedProfiler?.Phase, evt, read ? "D2H" : "H2D", (long)bytes);
            // Blocking transfers complete all earlier commands in our in-order queue.
            DrainCompletedEvents();
            if (read) D2HBytes += (long)bytes; else H2DBytes += (long)bytes;
            double elapsed = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            TransferMilliseconds += elapsed;
            DetailedProfiler?.Add(read ? "D2H" : "H2D", $"{DetailedProfiler.Phase}/{bytes}B", elapsed, (long)bytes);
        }
        finally { if (evt != 0) OpenClNative.clReleaseEvent(evt); pin.Free(); }
    }

    private ArcBuffer AllocateCore(long bytes, Array? data)
    {
        using var allocationScope = Timeline?.Host("allocation-pool", DetailedProfiler?.Phase ?? "");
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((ulong)bytes > Device.MaximumAllocationBytes)
            throw new InvalidOperationException($"Arc allocation {bytes:N0} exceeds device maximum {Device.MaximumAllocationBytes:N0} bytes.");
        RequestedBytes += bytes;
        GCHandle pin = default;
        try
        {
            long start = Stopwatch.GetTimestamp();
            bool poolHit = TryTakeCached(bytes, out nint pointer);
            bool retiredHit = false;
            if (poolHit)
            {
                CachedBytes -= bytes;
                PoolHits++;
            }
            else if (Options.ReuseRetiredBuffers && TryTakeRetired(bytes, out pointer))
            {
                retiredHit = true;
                RetiredReuseCount++;
            }
            else
            {
                TrimCacheForAllocation(bytes);
                if (_externalMemoryReservationBytes > 0
                    && AllocatedBytes + CachedBytes + RetiredBytes + bytes > EffectivePhysicalBufferBudgetBytes)
                    throw new InvalidOperationException("Arc allocation exceeds the device budget after external memory reservation.");
                // COPY_HOST_PTR copies the full allocation, not the array length.
                // Tiny packed payloads must not read padding beyond their managed array.
                if (!Options.ExplicitHostUploads && data is { Length: > 0 } && Buffer.ByteLength(data) == bytes)
                    pin = GCHandle.Alloc(data, GCHandleType.Pinned);
                int error;
                using (Timeline?.Host(pin.IsAllocated ? "native-allocation-with-upload" : "native-allocation"))
                    pointer = OpenClNative.clCreateBuffer(_context, pin.IsAllocated ? 33UL : 1UL,
                        (nuint)bytes, pin.IsAllocated ? pin.AddrOfPinnedObject() : 0, out error);
                if (pin.IsAllocated && Timeline is { } timeline) timeline.OpaqueAllocationCopies++;
                OpenClNative.Check(error, $"allocate {bytes:N0} bytes");
                AllocationCount++;
                NativeAllocatedBytes += bytes;
            }
            double elapsed = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            AllocationMilliseconds += elapsed;
            DetailedProfiler?.Add(poolHit ? "pool-hit" : retiredHit ? "retired-reuse" : "native-allocation",
                $"{DetailedProfiler.Phase}/{bytes}B", elapsed, bytes);
            if (pin.IsAllocated)
            {
                H2DBytes += bytes;
                DetailedProfiler?.Add("H2D", $"{DetailedProfiler.Phase}/allocation-copy/{bytes}B", elapsed, bytes);
            }
            var buffer = new ArcBuffer(this, pointer, bytes);
            _buffers.Add(buffer);
            AllocatedBytes += bytes;
            PeakAllocatedBytes = Math.Max(PeakAllocatedBytes, AllocatedBytes + CachedBytes + RetiredBytes);
            if ((poolHit || retiredHit || !pin.IsAllocated) && data is { Length: > 0 })
            {
                try { Transfer(buffer, data, read: false); }
                catch { buffer.Dispose(); throw; }
            }
            return buffer;
        }
        finally { if (pin.IsAllocated) pin.Free(); }
    }

    public static HostArray In(Array values) => new(values, false);
    public static HostArray InOut(float[] values) => new(values, true);
    public static HostArray Out(float[] values) => new(values, true, false);
    public sealed record HostArray(Array Values, bool ReadBack, bool Upload = true);
    public readonly record struct LocalMemory(int Bytes);

    public void Run(string name, long workItems, int localSize, params object[] arguments)
    {
        if (workItems <= 0) return;
        long count = localSize == 0 ? workItems : checked((workItems + localSize - 1) / localSize * localSize);
        RunDimensions(name, [checked((nuint)count)], localSize == 0 ? null : [(nuint)localSize], arguments);
    }

    public void Run2D(string name, long globalX, long globalY, int localX, int localY, params object[] arguments)
        => RunDimensions(name, [checked((nuint)globalX), checked((nuint)globalY)], [(nuint)localX, (nuint)localY], arguments);

    public void Run3D(string name, long globalX, long globalY, long globalZ, int localX, int localY, int localZ, params object[] arguments)
        => RunDimensions(name, [checked((nuint)globalX), checked((nuint)globalY), checked((nuint)globalZ)], [(nuint)localX, (nuint)localY, (nuint)localZ], arguments);

    private unsafe void RunDimensions(string name, nuint[] global, nuint[]? localSize, object[] arguments)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!_kernels.TryGetValue(name, out nint kernel))
            {
                using var compilation = Timeline?.Host("kernel-create-compile", name);
                kernel = OpenClNative.clCreateKernel(ProgramForRequestedKernel(name), name, out int error);
                OpenClNative.Check(error, $"create kernel {name}");
                _kernels.Add(name, kernel);
            }
            List<(ArcBuffer Buffer, HostArray Array)>? temporary = null;
            CachedKernelArguments? argumentCache = GetKernelArgumentCache(kernel, arguments.Length);
            nint kernelEvent = 0;
            long submitStart = Stopwatch.GetTimestamp();
            string? label = DetailedProfiler?.KernelLabel(name, arguments);
            try
            {
                using (Timeline?.Host("queue-submit", label ?? name))
                {
                for (int i = 0; i < arguments.Length; i++)
                {
                    object argument = arguments[i];
                    if (argument is HostArray host)
                    {
                        if (host.Values is not float[] and not int[])
                            throw new ArgumentException("Only float[] and int[] staging arrays are supported.");
                        var allocation = AllocateCore(Math.Max(4, Buffer.ByteLength(host.Values)), host.Upload ? host.Values : null);
                        (temporary ??= []).Add((allocation, host));
                        argument = allocation;
                    }
                    int status;
                    switch (argument)
                    {
                        case ArcBuffer buffer:
                            if (buffer.Owner != this || buffer.Handle == 0) throw new ArgumentException("Arc buffer belongs to a different or disposed lane.");
                            nint handle = buffer.Handle;
                            status = SetKernelArgument(kernel, (uint)i, (nuint)sizeof(nint), (nint)(&handle), argumentCache); break;
                        case int value: status = SetKernelArgument(kernel, (uint)i, 4, (nint)(&value), argumentCache); break;
                        case uint value: status = SetKernelArgument(kernel, (uint)i, 4, (nint)(&value), argumentCache); break;
                        case float value: status = SetKernelArgument(kernel, (uint)i, 4, (nint)(&value), argumentCache); break;
                        case LocalMemory local: status = SetKernelArgument(kernel, (uint)i, checked((nuint)local.Bytes), 0, argumentCache); break;
                        default: throw new ArgumentException($"Unsupported Arc kernel argument {argument.GetType()}.");
                    }
                    if (status != 0) OpenClNative.Check(status, $"set {name} argument {i}");
                }
                // With timing disabled, retain one event at each half-window
                // boundary. The existing partial wait therefore fences the
                // same oldest commands without allocating an event per kernel.
                bool eventRequired = CollectKernelEvents || (Options.BatchDispatch && Options.PipelineEventCollection
                    && (_pendingEvents.Count + 1) % (Options.QueuedKernelLimit / 2) == 0);
                int launchStatus = OpenClNative.clEnqueueNDRangeKernel(_queue, kernel, (uint)global.Length, 0,
                    global, localSize, 0, 0, eventRequired ? (nint)(&kernelEvent) : 0);
                if (launchStatus != 0) OpenClNative.Check(launchStatus, $"launch {name}");
                KernelLaunchCount++;
                }
                if (label is not null) DetailedProfiler!.Add("host-submit", label, Stopwatch.GetElapsedTime(submitStart).TotalMilliseconds);
                if (temporary is not null) foreach (var (buffer, host) in temporary)
                {
                    if (!host.ReadBack || host.Values.Length == 0) continue;
                    GCHandle pin = GCHandle.Alloc(host.Values, GCHandleType.Pinned);
                    long readStart = Stopwatch.GetTimestamp();
                    nint readEvent = 0;
                    try {
                        using (Timeline?.Host("blocking-transfer-wait", "host-result"))
                            OpenClNative.Check(OpenClNative.clEnqueueReadBuffer(_queue, buffer.Handle, 1, 0,
                                (nuint)Buffer.ByteLength(host.Values), pin.AddrOfPinnedObject(), 0, 0,
                                Timeline is null ? 0 : (nint)(&readEvent)), $"read {name}");
                        Timeline?.Device("D2H", label, readEvent, "D2H", Buffer.ByteLength(host.Values));
                        D2HBytes += Buffer.ByteLength(host.Values);
                        double elapsed = Stopwatch.GetElapsedTime(readStart).TotalMilliseconds;
                        TransferMilliseconds += elapsed;
                        DetailedProfiler?.Add("D2H", $"{DetailedProfiler.Phase}/host-result/{Buffer.ByteLength(host.Values)}B", elapsed, Buffer.ByteLength(host.Values));
                    }
                    finally { if (readEvent != 0) OpenClNative.clReleaseEvent(readEvent); pin.Free(); }
                }
                _pendingEvents.Add((name, kernelEvent, label));
                kernelEvent = 0;
                // Bound queued command/event resources. No per-kernel fence is
                // necessary when all uses (including pooled reuse) share this queue.
                if (!Options.BatchDispatch || temporary?.Any(t => t.Array.ReadBack) == true)
                    SynchronizeCore(!Options.BatchDispatch ? "unbatched" : "host-result");
                else if (_pendingEvents.Count >= Options.QueuedKernelLimit)
                {
                    if (Options.PipelineEventCollection) DrainOldestEventHalf();
                    else SynchronizeCore("event-limit");
                }
            }
            catch { if (_queue != 0) OpenClNative.clFinish(_queue); DrainCompletedEvents(); throw; }
            finally {
                if (kernelEvent != 0) OpenClNative.clReleaseEvent(kernelEvent);
                if (temporary is not null) foreach (var entry in temporary) entry.Buffer.Dispose();
            }
        }
    }

    /// <summary>Finish queued work and collect profiling events without transferring tensor data.</summary>
    public void Synchronize() => SynchronizeCore("explicit");

    // Keep the newer half executing while CPU queries/releases older events.
    // This is NOT a retirement fence: retired allocations and D2D events stay
    // owned until the existing full-queue synchronization / memory budget fence.
    private unsafe void DrainOldestEventHalf()
    {
        int count = _pendingEvents.Count / 2;
        nint last = _pendingEvents[count - 1].Event;
        if (last == 0) throw new InvalidOperationException("The queued kernel window has no completion event.");
        long start = Stopwatch.GetTimestamp();
        using (Timeline?.Host("queue-partial-wait", "event-half"))
        {
            OpenClNative.Check(OpenClNative.clFlush(_queue), "flush event window");
            OpenClNative.Check(OpenClNative.clWaitForEvents(1, (nint)(&last)), "wait oldest event window");
        }
        DetailedProfiler?.Add("partial-wait", DetailedProfiler.Phase, Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        using (Timeline?.Host("profiler-event-collection", "completed-half"))
        {
            // Remove each owner even if profiling throws; the caller's failure
            // cleanup must never release the same native event twice.
            int removed = 0;
            try
            {
                for (; removed < count;)
                {
                    var entry = _pendingEvents[removed++];
                    try { RecordKernelTime(entry.Name, entry.Event, entry.Label); }
                    finally { if (entry.Event != 0) OpenClNative.clReleaseEvent(entry.Event); }
                }
            }
            finally { _pendingEvents.RemoveRange(0, removed); }
        }
    }

    private void SynchronizeCore(string reason)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            long start = Stopwatch.GetTimestamp();
            try { using var wait = Timeline?.Host("queue-sync-wait", reason); OpenClNative.Check(OpenClNative.clFinish(_queue), "synchronize"); }
            finally {
                DetailedProfiler?.Add("synchronize", $"{DetailedProfiler.Phase}/{reason}", Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                DrainCompletedEvents();
            }
        }
    }

    private void DrainCompletedEvents()
    {
        using var collection = Timeline?.Host("profiler-event-collection");
        foreach (var entry in _pendingEvents)
        {
            try { RecordKernelTime(entry.Name, entry.Event, entry.Label); }
            finally { if (entry.Event != 0) OpenClNative.clReleaseEvent(entry.Event); }
        }
        _pendingEvents.Clear();
        foreach (var entry in _pendingCopyEvents)
        {
            try { Timeline?.Device(entry.Kind, entry.Kind, entry.Event, entry.Kind, entry.Bytes); }
            finally { OpenClNative.clReleaseEvent(entry.Event); }
        }
        _pendingCopyEvents.Clear();
        _queuedCopies = false;
        foreach (var entry in _retired) ReleaseNative(entry.Handle, entry.Bytes);
        _retired.Clear(); RetiredBytes = 0;
    }

    private void RecordKernelTime(string name, nint evt, string? label)
    {
        if (evt == 0 || !CollectKernelEvents) return;
        Timeline?.Device(name, label, evt, "kernel", 0);
        if (OpenClNative.clGetEventProfilingInfo(evt, 0x1282, 8, out ulong start, out _) != 0
            || OpenClNative.clGetEventProfilingInfo(evt, 0x1283, 8, out ulong end, out _) != 0) return;
        double ms = (end - start) * 1e-6;
        KernelMilliseconds += ms;
        KernelTimings[name] = KernelTimings.GetValueOrDefault(name) + ms;
        if (label is not null)
        {
            DetailedProfiler!.Add("gpu-kernel", label, ms);
            if (OpenClNative.clGetEventProfilingInfo(evt, 0x1281, 8, out ulong submitted, out _) == 0 && start >= submitted)
                DetailedProfiler.Add("queue-delay-overlapping", label, (start - submitted) * 1e-6);
        }
    }

    private void Release(ArcBuffer buffer)
    {
        lock (_sync)
        {
            using var releaseScope = Timeline?.Host("release-pool");
            nint handle = buffer.Handle;
            if (handle == 0) return;
            buffer.Handle = 0;
            _buffers.Remove(buffer);
            AllocatedBytes -= buffer.Bytes;
            if (!_disposed && buffer.Bytes <= Options.BufferPoolBytes
                && (Options.LruBufferPool || buffer.Bytes <= Options.BufferPoolBytes - CachedBytes))
            {
                if (Options.LruBufferPool)
                {
                    // Prepare managed metadata before changing native ownership.
                    // A cache miss may evict only free entries, never live buffers.
                    CachedBuffer incoming;
                    try
                    {
                        incoming = new(handle, buffer.Bytes);
                        while (buffer.Bytes > Options.BufferPoolBytes - CachedBytes)
                        {
                            CachedBuffer victim = _freeLru.First?.Value
                                ?? throw new InvalidOperationException("Arc LRU cache accounting is inconsistent.");
                            RemoveLruEntry(victim);
                            CachedBytes -= victim.Bytes;
                            RetireOrRelease(victim.Handle, victim.Bytes, "pool-lru-eviction");
                            DetailedProfiler?.Add("cache-eviction", $"{DetailedProfiler.Phase}/{victim.Bytes}B", bytes: victim.Bytes);
                        }
                        if (!_lruPool.TryGetValue(buffer.Bytes, out LinkedList<CachedBuffer>? free))
                            _lruPool.Add(buffer.Bytes, free = new());
                        free.AddLast(incoming.SizeNode);
                        _freeLru.AddLast(incoming.RecencyNode);
                        CachedBytes += buffer.Bytes;
                    }
                    catch (Exception cacheFailure)
                    {
                        // Eviction can encounter a queue error. The incoming
                        // native handle must still be retired/freed exactly once.
                        try { RetireOrRelease(handle, buffer.Bytes, "pool-lru-failure"); }
                        catch (Exception releaseFailure) { throw new AggregateException(cacheFailure, releaseFailure); }
                        throw;
                    }
                }
                else
                {
                    if (!_pool.TryGetValue(buffer.Bytes, out Stack<nint>? free)) _pool.Add(buffer.Bytes, free = new());
                    free.Push(handle);
                    CachedBytes += buffer.Bytes;
                }
            }
            else RetireOrRelease(handle, buffer.Bytes, "pool-budget-release");
        }
    }

    // Called under _sync. Taking a free buffer keeps existing in-order queue
    // reuse semantics; no fence or host transfer is added on a pool hit.
    private bool TryTakeCached(long bytes, out nint handle)
    {
        if (Options.LruBufferPool)
        {
            if (_lruPool.TryGetValue(bytes, out var lruFree) && lruFree.Last is { } node)
            {
                CachedBuffer entry = node.Value;
                RemoveLruEntry(entry);
                handle = entry.Handle;
                return true;
            }
        }
        else if (_pool.TryGetValue(bytes, out Stack<nint>? legacyFree) && legacyFree.Count > 0)
        {
            handle = legacyFree.Pop();
            return true;
        }
        handle = 0;
        return false;
    }

    private void RemoveLruEntry(CachedBuffer entry)
    {
        LinkedList<CachedBuffer> free = _lruPool[entry.Bytes];
        free.Remove(entry.SizeNode);
        _freeLru.Remove(entry.RecencyNode);
        if (free.Count == 0) _lruPool.Remove(entry.Bytes);
    }

    private void TrimCacheForAllocation(long bytes)
    {
        if (!Options.LruBufferPool && _externalMemoryReservationBytes == 0) return;
        // Retired bytes can be reclaimed at the single fence below. Do not
        // evict additional useful cache entries merely because that fence has
        // not yet completed.
        TrimCacheToPhysicalBudget(bytes, EffectivePhysicalBufferBudgetBytes, "physical-budget");
    }

    // Called only under _sync after an exact-size pool miss. Retired entries
    // still own their native reference; their original ArcBuffer and all its
    // borrowed views are already invalid. Earlier commands and every new use
    // share this lane's in-order queue, so reuse needs no completion fence.
    // Move, rather than duplicate, the handle and its physical-byte accounting.
    private bool TryTakeRetired(long bytes, out nint handle)
    {
        for (int i = _retired.Count - 1; i >= 0; i--)
        {
            var entry = _retired[i];
            if (entry.Bytes != bytes) continue;
            _retired.RemoveAt(i);
            RetiredBytes -= bytes;
            handle = entry.Handle;
            return true;
        }
        handle = 0;
        return false;
    }

    private void RetireOrRelease(nint handle, long bytes, string fenceReason)
    {
        // Retiring an evicted cache entry preserves its physical-byte accounting
        // until the same queue-complete boundary used for ordinary releases.
        DetailedProfiler?.Add("uncached-release", $"{DetailedProfiler.Phase}/{bytes}B", bytes: bytes);
        if (!_disposed && (_pendingEvents.Count != 0 || _queuedCopies)
            && bytes <= Options.DeferredReleaseBytes - RetiredBytes)
        {
            _retired.Add((handle, bytes)); RetiredBytes += bytes;
            DetailedProfiler?.Add("deferred-release", DetailedProfiler.Phase, bytes: bytes);
            return;
        }
        try { if (!_disposed && (_pendingEvents.Count != 0 || _queuedCopies)) SynchronizeCore(fenceReason); }
        finally { ReleaseNative(handle, bytes); }
    }

    private void ReleaseNative(nint handle, long bytes)
    {
        InvalidateKernelArgumentCaches();
        using var releaseScope = Timeline?.Host("native-free");
        int status = OpenClNative.clReleaseMemObject(handle);
        if (status == 0)
        {
            NativeReleasedBytes += bytes;
            NativeReleaseCount++;
            DetailedProfiler?.Add("native-release", $"{DetailedProfiler.Phase}/{bytes}B", bytes: bytes);
        }
    }

    public void Dispose()
    {
        lock (_sync)
        {
            if (_disposed) return;
            _disposed = true;
            if (_queue != 0) OpenClNative.clFinish(_queue);
            DrainCompletedEvents();
            foreach (ArcBuffer buffer in _buffers.ToArray()) Release(buffer);
            foreach (var pair in _pool)
                foreach (nint pointer in pair.Value) ReleaseNative(pointer, pair.Key);
            foreach (CachedBuffer entry in _freeLru) ReleaseNative(entry.Handle, entry.Bytes);
            _pool.Clear(); _lruPool.Clear(); _freeLru.Clear(); CachedBytes = 0;
            foreach (nint kernel in _kernels.Values) OpenClNative.clReleaseKernel(kernel);
            _kernels.Clear();
            if (_program != 0) OpenClNative.clReleaseProgram(_program);
            foreach (nint program in _flashPrograms.Values) OpenClNative.clReleaseProgram(program);
            _flashPrograms.Clear();
            if (_attentionProductsProgram != 0) OpenClNative.clReleaseProgram(_attentionProductsProgram);
            _attentionProductsProgram = 0;
            if (_largeEpilogueProgram != 0) OpenClNative.clReleaseProgram(_largeEpilogueProgram);
            _largeEpilogueProgram = 0;
            if (_queue != 0) OpenClNative.clReleaseCommandQueue(_queue);
            if (_context != 0) OpenClNative.clReleaseContext(_context);
            _program = _queue = _context = 0;
        }
    }

    public sealed class ArcBuffer : IDisposable
    {
        private readonly ArcBuffer? _borrowedFrom;
        private bool _borrowDisposed;
        private nint _handle;
        internal ArcExecutionLane Owner { get; }
        internal nint Handle { get => _borrowedFrom is null ? _handle : _borrowDisposed ? 0 : _borrowedFrom.Handle; set => _handle = value; }
        internal long Bytes { get; }
        public long ByteLength => Bytes;
        public bool IsAlive => Handle != 0;
        internal ArcBuffer(ArcExecutionLane owner, nint handle, long bytes) => (Owner, Handle, Bytes) = (owner, handle, bytes);
        private ArcBuffer(ArcBuffer source) { _borrowedFrom = source; Owner = source.Owner; Bytes = source.Bytes; }
        public ArcBuffer Borrow() => IsAlive ? new(this) : throw new ObjectDisposedException(nameof(ArcBuffer));
        public void Dispose() { if (_borrowedFrom is null) Owner.Release(this); else _borrowDisposed = true; }
    }
}
