using NNtrain.Runtime.Execution;

namespace NNtrain.Arc;

/// <summary>
/// Session-owned OpenCL lane for resident training and an explicit staged A/B reference.
/// All native allocations and the in-order queue are owned by this session.
/// </summary>
public sealed partial class ArcExecutionLane : IExecutionLane, IDeviceMemoryManager, IKernelCapabilitySet
{

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
    public ArcDeviceInfo Device { get; }
    public ArcExecutionOptions Options { get; }
    public ArcDetailedProfiler? DetailedProfiler { get; }
    private bool CollectKernelEvents => Options.CollectKernelTimings || DetailedProfiler is not null || Timeline is not null;
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
            InitializeProgram();
        }
        catch { Dispose(); throw; }
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
}
