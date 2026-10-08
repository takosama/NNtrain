using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>
/// Text-only Qwen3.5 inference. Weights, activations, attention caches and
/// recurrent state stay on their owning Arc device. Host staging is used only
/// between different OpenCL contexts. Instances own one sequence and are not thread safe.
/// </summary>
public sealed partial class Qwen35QuantizedModel : IDisposable
{
    private const long WorkspaceReserveBytes = 64L * 1024 * 1024;
    private readonly List<ArcExecutionLane> _lanes = [];
    private readonly Dictionary<string, Matrix> _matrices = new(StringComparer.Ordinal);
    private readonly Dictionary<string, ArcBuffer> _dense = new(StringComparer.Ordinal);
    private readonly Dictionary<ArcExecutionLane, ArcBuffer> _zeroBias = [];
    private readonly Dictionary<ArcExecutionLane, long> _auxiliaryBytes = [];
    private readonly List<LayerState> _states = [];
    private readonly List<LayerBindings> _layerBindings = [];
    private Matrix _embedding = null!;
    private ArcBuffer _outputNorm = null!;
    private int _position;
    private int[]? _cachedPromptTokens;
    private int _lastReusedPromptTokens;
    private bool _disposed, _faulted;
    private readonly Qwen35ExecutionOptions _options;
    private Qwen35PrismMetadata? _prism;
    private readonly Dictionary<(ArcExecutionLane Lane, int Width), ArcBuffer> _prismSigns = [];

    private Qwen35QuantizedModel(Qwen35GgufDescriptor descriptor, Qwen35ExecutionOptions options)
        => (Descriptor, _options) = (descriptor, options);

    public Qwen35GgufDescriptor Descriptor { get; }
    public IReadOnlyList<long> ResidentWeightBytes => _lanes.Select(lane =>
        _matrices.Values.Where(matrix => ReferenceEquals(matrix.Lane, lane))
            .Sum(matrix => (long)matrix.StorageBytes)
            + (_splitOutputHead is { } split && ReferenceEquals(split.Peer, lane) ? split.WeightBytes : 0)).ToArray();
    public IReadOnlyList<long> ResidentAuxiliaryWeightBytes => _lanes.Select(lane => _auxiliaryBytes.GetValueOrDefault(lane)).ToArray();
    public IReadOnlyList<long> ResidentStateBytes => _lanes.Select(lane =>
        _states.Where(state => ReferenceEquals(state.Lane, lane)).Sum(state => state.StorageBytes)).ToArray();
    public IReadOnlyList<long> UploadedBytes => _lanes.Select(lane => lane.H2DBytes).ToArray();
    public IReadOnlyList<long> DownloadedBytes => _lanes.Select(lane => lane.D2HBytes).ToArray();
    public IReadOnlyList<long> LiveDeviceBytes => _lanes.Select(lane => lane.AllocatedBytes).ToArray();
    /// <summary>Native memory still owned by each lane, including idle and retired buffers.</summary>
    public IReadOnlyList<long> TotalNativeDeviceBytes => _lanes.Select(lane =>
        checked(lane.AllocatedBytes + lane.CachedBytes + lane.RetiredBytes)).ToArray();
    public IReadOnlyList<long> PeakDeviceBytes => _lanes.Select(lane => lane.PeakAllocatedBytes).ToArray();
    /// <summary>
    /// Account for another allocator on a model device. Call between model
    /// operations, under the same owner serialization used for generation.
    /// The slot is the index in the model's selected device list.
    /// </summary>
    public void SetExternalDeviceMemoryReservation(int slot, long bytes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)slot >= (uint)_lanes.Count) throw new ArgumentOutOfRangeException(nameof(slot));
        ArcExecutionLane lane = _lanes[slot];
        ReleaseSplitOutputHeadForReservation(lane, bytes);
        lane.SetExternalMemoryReservation(bytes);
    }
    /// <summary>Prompt tokens skipped by the most recent prefix-reuse generation.</summary>
    public int LastReusedPromptTokens => _lastReusedPromptTokens;
    public IReadOnlyDictionary<string, double> KernelMilliseconds => _lanes
        .SelectMany(lane => lane.KernelTimings).GroupBy(pair => pair.Key)
        .ToDictionary(group => group.Key, group => group.Sum(pair => pair.Value));

    public static Qwen35QuantizedModel Load(
        string path, IReadOnlyList<int>? devices = null, Action<string>? progress = null,
        Qwen35ExecutionOptions? options = null)
    {
        using var reader = new GgufReader(path);
        return Load(reader, devices, progress, options);
    }

    /// <summary>Loads from an already parsed GGUF. The caller retains ownership of the reader.</summary>
    public static Qwen35QuantizedModel Load(
        GgufReader gguf, IReadOnlyList<int>? devices = null, Action<string>? progress = null,
        Qwen35ExecutionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        string path = gguf.FilePath;
        if (progress is not null)
        {
            Action<string> callback = progress;
            var callbackLock = new object();
            progress = message => { lock (callbackLock) callback(message); };
        }
        options ??= new Qwen35ExecutionOptions();
        if (options.TrainingIQ2Bf16XmxForward && options.TrainingIQ2Fp16XmxForward)
            throw new ArgumentException("Choose either BF16 or FP16 IQ2_S/XMX forward, not both.", nameof(options));
        if ((options.TrainingIQ2Bf16XmxForward || options.TrainingIQ2Fp16XmxForward)
            && !options.LoraTraining)
            throw new ArgumentException("IQ2_S XMX forward is only available for LoRA training.", nameof(options));
        if (options.TrainingIQ2ProjectionCacheMiB is < 0 or > 16384)
            throw new ArgumentOutOfRangeException(nameof(options),
                "The experimental IQ2_S host projection cache cap must be 0..16384 MiB.");
        if (options.TrainingIQ2GpuProjectionCacheMiB is < 0 or > 1024)
            throw new ArgumentOutOfRangeException(nameof(options),
                "The experimental IQ2_S GPU projection cache cap per Arc must be 0..1024 MiB.");
        if (options.TrainingIQ2ProjectionCacheMiB > 0 && options.TrainingIQ2GpuProjectionCacheMiB > 0)
            throw new ArgumentException("Choose the IQ2_S host projection cache or GPU projection cache, not both.", nameof(options));
        if (options.TrainingIQ2GpuProjectionCacheMiB > 0 && !options.TrainingGpuCheckpoints)
            throw new ArgumentException("The IQ2_S GPU projection cache requires GPU training checkpoints.", nameof(options));
        if (options.TrainingFusedAttentionRows && !options.TrainingPackedAttentionScores)
            throw new ArgumentException("Fused attention rows require packed scores.", nameof(options));
        if (options.TrainingFusedAttentionOutput && !options.TrainingFusedAttentionRows)
            throw new ArgumentException("Fused attention output requires fused attention rows.", nameof(options));
        if (!Enum.IsDefined(options.QuantizedKernel) || options.QueuedKernelLimit is < 16 or > 4096
            || options.ProjectionWorkgroupSize is not (32 or 64 or 128)
            || options.InferencePairedProjectionTypes is < 0 or > 7
            || options.InferencePrefillChunkTokens is < 0 or > 1024
            || options.InferenceProjectionRows is not (1 or 2 or 4 or 8 or 16)
            || options.InferenceBufferPoolMiB is < 0 or > 2048
            || options.InferenceDeferredReleaseMiB is < 0 or > 1024
            || options.LoraReductionSize is not (16 or 128 or 256 or 512 or 1024)
            || options.TrainingTransposeRows is not (1 or 4 or 8 or 16 or 32)
            || options.TrainingTransposeOctetRows is not (0 or 4 or 8 or 16)
            || options.TrainingQ4TransposeOctetRows is not (0 or 4 or 8 or 16)
            || options.TrainingIQ3TransposeOctetRows is not (0 or 4 or 8 or 16)
            || options.TrainingQ5TransposeOctetRows is not (0 or 4 or 8 or 16)
            || options.TrainingNormSplits is < 1 or > 32
            || options.TrainingForwardRows is not (1 or 2 or 4 or 8 or 16)
            || options.TrainingQ5ForwardRows is not (1 or 4 or 8)
            || options.TrainingStreamedAttentionTileRows is not (0 or 64 or 128 or 256 or 512 or 1024)
            || options.TrainingHybridCheckpointMiBPerDevice is < 0 or > 1024
            || options.TrainingBufferPoolMiB is < 0 or > 2048)
            throw new ArgumentException("Invalid Qwen3.5 execution options.", nameof(options));
        if (options.TrainingHybridCheckpointMiBPerDevice > 0
            && options.TrainingHostCheckpointForwardCopyHandoff)
            throw new ArgumentException("Hybrid checkpoints cannot overwrite retained layer inputs through forward-copy handoff.", nameof(options));
        Qwen35GgufDescriptor d = Qwen35Gguf.Inspect(gguf);
        Qwen35PrismMetadata? prism = Qwen35PrismMetadata.Read(gguf);
        if (options.LoraTraining && (prism is not null || d.Tensors.Any(t => t.Type == Qwen2Gguf.BF16Type && IsQuantized(t))))
            throw new NotSupportedException("PQ2_0/PTQ1_0 and BF16 matrix backpropagation are not implemented; load this model for generation.");
        ArcDeviceInfo[] available = ArcDevices.Enumerate().ToArray();
        if (available.Length == 0)
            throw new NotSupportedException("Qwen3.5 quantized inference requires Intel Arc.");
        int[] selected = devices?.ToArray() ?? SelectDevices(d, available, options);
        if (selected.Length == 0 || selected.Length > d.LayerCount
            || selected.Distinct().Count() != selected.Length
            || selected.Any(index => index < 0 || index >= available.Length))
            throw new ArgumentException("Specify distinct, available Arc device indices.", nameof(devices));
        if ((options.TrainingIQ2Bf16XmxForward || options.TrainingIQ2Fp16XmxForward) && selected.Any(index =>
            !available[index].SupportsXmx || available[index].MinimumSubgroupSize != 16
            || !available[index].Extensions.Split(' ').Contains("cl_intel_subgroups")
            || (options.TrainingIQ2Fp16XmxForward && !available[index].Extensions.Split(' ').Contains("cl_khr_fp16"))))
            throw new NotSupportedException("IQ2_S XMX training forward requires Arc XMX, subgroup size 16, and FP16 support when selected.");
        bool residentIq2Panels = UseResidentIq2Panels(options, selected.Select(index => available[index]));
        long[] planned = PlanPersistentBytes(d, selected.Length, residentIq2Panels);
        for (int slot = 0; slot < selected.Length; slot++)
        {
            ArcDeviceInfo device = available[selected[slot]];
            if (planned[slot] + WorkspaceReserveBytes > DeviceBudget(device))
                throw new NotSupportedException(
                    $"Arc {device.Index} needs {planned[slot] / 1073741824.0:F2} GiB for weights and initial GPU state " +
                    "plus workspace. Select more GPUs with --devices 0,1.");
            foreach (GgufTensorInfo tensor in d.Tensors)
                if (DeviceSlot(tensor.Name, d.LayerCount, selected.Length) == slot
                    && (ulong)(IsQuantized(tensor) ? MaximumMatrixBufferBytes(tensor, residentIq2Panels) : DenseBytes(tensor)) > device.MaximumAllocationBytes)
                    throw new NotSupportedException($"Tensor '{tensor.Name}' exceeds Arc {device.Index}'s allocation limit.");
        }

        var model = new Qwen35QuantizedModel(d, options)
        {
            _prism = prism,
            _modelPath = Path.GetFullPath(path),
            // Keep the same base snapshot protected from writes/deletion while
            // resident weights and adapter identity refer to it.
            _modelSource = File.OpenRead(path)
        };
        Task<string>? fingerprint = options.ComputeModelFingerprintOnLoad
            ? Task.Run(model.ModelFingerprint) : null;
        try
        {
            ArcExecutionLane CreateLane(int index)
            {
                progress?.Invoke($"Preparing Arc {index}: {available[index].Name}");
                return new ArcExecutionLane(index, new ArcExecutionOptions
                {
                    Qwen35InferenceKernelsOnly = true,
                    Qwen35ResidentIq2Panels = residentIq2Panels,
                    Qwen35GgufBslmPrefill = !options.LoraTraining && options.InferenceXmxGgufBslmPrefill,
                    Qwen35TrainingKernels = options.LoraTraining,
                    Qwen35CooperativeLora = options.LoraTraining ? options.TrainingCooperativeLora : options.InferenceCooperativeLora,
                    Qwen35ProjectionWorkgroupSize = options.LoraTraining ? 32 : options.ProjectionWorkgroupSize,
                    Qwen35PairedProjection = !options.LoraTraining && options.InferencePairedProjection,
                    Qwen35PairedProjectionTypes = options.LoraTraining ? 0 : options.InferencePairedProjectionTypes,
                    Qwen35PairedLoraProjection = !options.LoraTraining && options.InferencePairedLoraProjection,
                    CacheKernelArguments = !options.LoraTraining && options.CacheKernelArguments,
                    CacheProgramBinary = options.CacheProgramBinary,
                    Qwen35FastRmsNorm = !options.LoraTraining && options.InferenceFastRmsNorm,
                    Qwen35ParallelArgmax = !options.LoraTraining && options.ParallelArgmax,
                    Qwen35ParallelDeltaNorm = !options.LoraTraining && options.ParallelDeltaNorm,
                    Qwen35DeltaSubgroupRms = !options.LoraTraining && options.InferenceBatchRecurrent
                        && options.InferenceSubgroupRecurrentRms && available[index].MinimumSubgroupSize == 16
                        && available[index].Extensions.Split(' ').Contains("cl_intel_subgroups"),
                    Qwen35LoraReductionSize = options.LoraTraining ? 128 : options.LoraReductionSize,
                    Qwen35UnrollQ4 = !options.LoraTraining && options.UnrollQ4,
                    Qwen35NativeHalfScale = options.NativeHalfScale,
                    CollectKernelTimings = options.CollectKernelTimings || options.DetailedProfiling,
                    Qwen35CooperativeDelta = options.TrainingCooperativeDelta,
                    DetailedProfiling = options.DetailedProfiling,
                    BufferPoolBytes = (long)(options.LoraTraining ? options.TrainingBufferPoolMiB : options.InferenceBufferPoolMiB) * 1024 * 1024,
                    DeferredReleaseBytes = options.LoraTraining ? 0 : (long)options.InferenceDeferredReleaseMiB * 1024 * 1024,
                    QueuedKernelLimit = options.QueuedKernelLimit,
                    PhysicalBufferBudgetBytes = DeviceBudget(available[index])
                });
            }
            if (options.ParallelModelLoad && selected.Length > 1)
            {
                Task<ArcExecutionLane>[] creating = selected.Select(index => Task.Run(() => CreateLane(index))).ToArray();
                try { Task.WhenAll(creating).GetAwaiter().GetResult(); }
                finally
                {
                    // Even if another device fails, completed lanes must be owned
                    // by the model so the outer failure path disposes them.
                    foreach (var task in creating)
                        if (task.IsCompletedSuccessfully) model._lanes.Add(task.Result);
                }
            }
            else foreach (int index in selected) model._lanes.Add(CreateLane(index));
            if (model._prism is not null)
            {
                foreach (ArcExecutionLane lane in model._lanes)
                foreach (var pair in model._prism.Signs)
                {
                    model._prismSigns.Add((lane, pair.Key), lane.Upload(pair.Value));
                    model._auxiliaryBytes[lane] = model._auxiliaryBytes.GetValueOrDefault(lane) + 4L * pair.Value.Length;
                }
                progress?.Invoke($"Prism: GPU Hadamard 1024, {model._prism.ForwardWeights.Count} projections, {model._prism.InverseWeights.Count} inverse embeddings, grouped GDN={model._prism.GroupedValueHeads}");
            }
            int loaded = 0;
            void LoadTensor(GgufTensorInfo tensor)
            {
                var lane = model._lanes[DeviceSlot(tensor.Name, d.LayerCount, selected.Length)];
                if (IsQuantized(tensor))
                {
                    // UploadRaw is blocking; no model-sized host payload is retained.
                    var matrix = new Matrix(lane, tensor,
                        gguf.ReadTensorBytes(tensor, EncodedBytes(tensor)), options.QuantizedKernel,
                        options.TrainingTransposeRows, options.LoraTraining ? options.TrainingForwardRows : options.InferenceProjectionRows,
                        options.LoraTraining ? options.TrainingQ5ForwardRows : 1,
                        options.TrainingIQ2Bf16XmxForward,
                        options.TrainingIQ2Fp16XmxForward,
                        options.TrainingIQ2RollingTranspose,
                        tensor.Type switch {
                            Qwen35Gguf.IQ2SType => options.TrainingTransposeOctetRows,
                            Qwen2Gguf.Q4KType => options.TrainingQ4TransposeOctetRows,
                            Qwen35Gguf.IQ3SType => options.TrainingIQ3TransposeOctetRows,
                            Qwen35Gguf.Q5KType => options.TrainingQ5TransposeOctetRows,
                            _ => 0 }, inferenceXmxPrefill: !options.LoraTraining && options.InferenceXmxPrefill,
                        inferenceXmxPackedPrefill: !options.LoraTraining && options.InferenceXmxPackedPrefill,
                        inferenceXmxFactoredPrefill: !options.LoraTraining && options.InferenceXmxFactoredPrefill,
                        inferenceResidentIq2Panels: residentIq2Panels,
                        inferenceXmxGgufBslmPrefill: !options.LoraTraining && options.InferenceXmxGgufBslmPrefill,
                        iq2TiledForward: options.IQ2TiledForward, iq2TiledBackward: options.IQ2TiledBackward,
                        inferenceIq2DecodePair4: !options.LoraTraining && options.InferenceIq2DecodePair4,
                        inferenceIq2DecodeLevel3: !options.LoraTraining && options.InferenceIq2DecodeLevel3);
                    lock (model._matrices) model._matrices.Add(tensor.Name, matrix);
                }
                else
                {
                    model._dense.Add(tensor.Name, lane.Upload(Qwen2Gguf.ReadTensor(gguf, tensor)));
                    model._auxiliaryBytes[lane] = model._auxiliaryBytes.GetValueOrDefault(lane) + DenseBytes(tensor);
                }
                int count = Interlocked.Increment(ref loaded);
                if (count % 100 == 0) progress?.Invoke($"Loading tensors: {count}/{d.Tensors.Count}");
            }
            if (options.ParallelModelLoad && selected.Length > 1)
            {
                // Dense conversion uses the reader's sequential stream. Packed
                // reads are positional and each worker owns one device queue.
                foreach (var tensor in d.Tensors.Where(t => !IsQuantized(t))) LoadTensor(tensor);
                Parallel.For(0, selected.Length, slot =>
                {
                    foreach (var tensor in d.Tensors.Where(t => IsQuantized(t)
                        && DeviceSlot(t.Name, d.LayerCount, selected.Length) == slot).OrderBy(t => t.Offset))
                        LoadTensor(tensor);
                });
            }
            else foreach (var tensor in d.Tensors) LoadTensor(tensor);
            foreach (ArcExecutionLane lane in model._lanes)
            {
                int width = model._matrices.Values.Where(matrix => ReferenceEquals(matrix.Lane, lane))
                    .Max(matrix => matrix.OutputWidth);
                ArcBuffer zeroBias = lane.Allocate(width);
                model._zeroBias.Add(lane, zeroBias);
                lane.Run("q35a_zero", width, 0, zeroBias, width);
            }
            model._embedding = model._matrices["token_embd.weight"];
            model._outputNorm = model._dense["output_norm.weight"];
            for (int layer = 0; layer < d.LayerCount; layer++)
            {
                var lane = model._lanes[DeviceSlot($"blk.{layer}.", d.LayerCount, selected.Length)];
                var state = new LayerState(lane, d, d.IsRecurrent(layer));
                model._states.Add(state);
                state.Initialize();
                model._layerBindings.Add(new LayerBindings(model, layer, state));
            }
            model.InitializeSplitOutputHead(gguf, progress);
            for (int slot = 0; slot < selected.Length; slot++)
            {
                model._lanes[slot].Synchronize();
                progress?.Invoke($"Arc {selected[slot]}: quantized weights = {model.ResidentWeightBytes[slot]} bytes " +
                    $"({model.ResidentWeightBytes[slot] / 1073741824.0:F2} GiB), " +
                    $"GPU state = {model.ResidentStateBytes[slot]} bytes, " +
                    $"live device allocation = {model._lanes[slot].AllocatedBytes} bytes");
            }
            if (fingerprint is not null) fingerprint.GetAwaiter().GetResult();
            return model;
        }
        catch
        {
            // The fingerprint reads the protected source handle: join it before
            // disposing that handle, without masking the original load error.
            try { fingerprint?.GetAwaiter().GetResult(); } catch { }
            model.Dispose(); throw;
        }
    }

    internal static long[] PlanWeightBytes(Qwen35GgufDescriptor d, int deviceCount, bool inferenceResidentIq2Panels = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(deviceCount);
        var bytes = new long[deviceCount];
        foreach (GgufTensorInfo tensor in d.Tensors.Where(IsQuantized))
            bytes[DeviceSlot(tensor.Name, d.LayerCount, deviceCount)] += PlannedMatrixBytes(tensor, inferenceResidentIq2Panels);
        return bytes;
    }

    private static long[] PlanPersistentBytes(Qwen35GgufDescriptor d, int devices, bool inferenceResidentIq2Panels = false)
    {
        long[] bytes = PlanWeightBytes(d, devices, inferenceResidentIq2Panels);
        var maxOutput = new int[devices];
        foreach (GgufTensorInfo tensor in d.Tensors)
        {
            int slot = DeviceSlot(tensor.Name, d.LayerCount, devices);
            if (IsQuantized(tensor)) maxOutput[slot] = Math.Max(maxOutput[slot], checked((int)tensor.Shape[1]));
            else bytes[slot] = checked(bytes[slot] + DenseBytes(tensor));
        }
        for (int i = 0; i < devices; i++) bytes[i] += 4L * maxOutput[i];
        for (int layer = 0; layer < d.LayerCount; layer++)
        {
            int slot = DeviceSlot($"blk.{layer}.", d.LayerCount, devices);
            long stateBytes = d.IsRecurrent(layer)
                ? checked(4L * d.LinearValueHeads * d.LinearHeadWidth * d.LinearHeadWidth
                    + Math.Max(4L, 4L * (2 * d.LinearKeyHeads + d.LinearValueHeads) * d.LinearHeadWidth * (d.ConvKernel - 1)))
                : checked(8L * d.KvHeadCount * d.HeadWidth * Math.Min(16, d.ContextLength));
            bytes[slot] = checked(bytes[slot] + stateBytes);
        }
        return bytes;
    }

    private static int[] SelectDevices(Qwen35GgufDescriptor d, ArcDeviceInfo[] available, Qwen35ExecutionOptions options)
    {
        for (int count = 1; count <= Math.Min(available.Length, d.LayerCount); count++)
        {
            long[] bytes = PlanPersistentBytes(d, count, UseResidentIq2Panels(options, available.Take(count)));
            if (bytes.Select((value, slot) => value + WorkspaceReserveBytes <= DeviceBudget(available[slot])).All(fits => fits))
                return Enumerable.Range(0, count).ToArray();
        }
        throw new NotSupportedException("The Qwen3.5 weights and initial GPU state exceed the available Arc VRAM budget.");
    }

    private static long DeviceBudget(ArcDeviceInfo device) => checked((long)(device.GlobalMemoryBytes / 10 * 9));

    internal static int AlignedTransposeChunkRows(int rows, long bytesPerRow,
        long scratchBudgetBytes, int rowTile)
    {
        if (rows < 1 || bytesPerRow < 1 || rowTile < 1)
            throw new ArgumentOutOfRangeException(nameof(rows));
        int capacity = checked((int)Math.Max(1L, Math.Min(rows, scratchBudgetBytes / bytesPerRow)));
        // A short tail launches another complete quantized-weight traversal.
        // Align the main chunks to the kernel's row tile when the budget permits.
        return capacity >= rowTile ? capacity / rowTile * rowTile : capacity;
    }

    private static int DeviceSlot(string name, int layers, int devices)
    {
        if (name.StartsWith("blk.", StringComparison.Ordinal))
        {
            int end = name.IndexOf('.', 4);
            int layer = int.Parse(name.AsSpan(4, end - 4));
            return Math.Min(devices - 1, layer * devices / layers);
        }
        return name == "token_embd.weight" ? 0 : devices - 1;
    }

    private static bool IsQuantized(GgufTensorInfo tensor)
        => tensor.Shape.Count == 2 && !tensor.Name.EndsWith(".ssm_conv1d.weight", StringComparison.Ordinal)
            && Qwen35Gguf.IsSupportedMatrixStorage(tensor.Type);
    private static int EncodedBytes(GgufTensorInfo tensor)
        => checked((int)(tensor.Shape[0] / (ulong)Qwen35Gguf.QuantizedBlockElements(tensor.Type) * tensor.Shape[1]
            * (ulong)Qwen35Gguf.QuantizedBlockBytes(tensor.Type)));
    private static bool CanUseResidentIq2Panels(GgufTensorInfo tensor)
        => tensor.Type == Qwen35Gguf.IQ2SType && tensor.Name.StartsWith("blk.", StringComparison.Ordinal)
            && tensor.Shape[0] % 256 == 0;
    private static bool UseResidentIq2Panels(Qwen35ExecutionOptions options, IEnumerable<ArcDeviceInfo> devices)
        => !options.LoraTraining && options.InferenceResidentIq2Panels && !options.InferenceXmxGgufBslmPrefill
            && options.QuantizedKernel is Qwen35QuantizedKernel.Auto or Qwen35QuantizedKernel.Subgroup
            && devices.All(device => device.SupportsXmx && device.MinimumSubgroupSize == 16
                && device.Extensions.Split(' ').Contains("cl_intel_subgroups")
                && device.Extensions.Split(' ').Contains("cl_khr_fp16"));
    private static long PlannedMatrixBytes(GgufTensorInfo tensor, bool residentIq2Panels)
        => residentIq2Panels && CanUseResidentIq2Panels(tensor)
            ? checked((long)tensor.Shape[0] * (((long)tensor.Shape[1] + 15) / 16 * 16) / 256 * 106)
            : EncodedBytes(tensor);
    private static long MaximumMatrixBufferBytes(GgufTensorInfo tensor, bool residentIq2Panels)
        => residentIq2Panels && CanUseResidentIq2Panels(tensor)
            ? checked((long)tensor.Shape[0] * (((long)tensor.Shape[1] + 15) / 16 * 16) / 256 * 96)
            : EncodedBytes(tensor);
    private static long DenseBytes(GgufTensorInfo tensor)
        => checked(tensor.Shape.Aggregate(1L, (count, dimension) => checked(count * (long)dimension)) * 4);

    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _faulted = true;
        _cachedPromptTokens = null;
        _cachedMixedPromptTokens = null;
        _lastReusedPromptTokens = 0;
        foreach (LayerState state in _states) state.Reset();
        _position = 0;
        _faulted = false;
    }

    /// <summary>Advances GPU state by one token. Only requested final logits are downloaded.</summary>
    public float[] ForwardToken(int tokenId, bool returnLogits = true)
    {
        InvalidatePromptCheckpoint();
        using ArcBuffer? logits = ForwardTokenDevice(tokenId, returnLogits);
        if (logits is null) return [];
        try
        {
            var values = new float[Descriptor.VocabularySize];
            OutputMatrix.Lane.Read(logits, values);
            if (values.Any(value => !float.IsFinite(value)))
                throw new ArithmeticException("Qwen3.5 produced non-finite logits.");
            return values;
        }
        catch { _faulted = true; throw; }
    }

    private Matrix OutputMatrix => _matrices.TryGetValue("output.weight", out Matrix? head)
        ? head : _matrices["token_embd.weight"];

    private ArcBuffer? ForwardTokenDevice(int tokenId, bool returnLogits)
        => ForwardTokenDevice(tokenId, returnLogits, null, null);

    private ArcBuffer? ForwardTokenDevice(int tokenId, bool returnLogits,
        float[]? overrideEmbedding, Qwen35Position? ropePosition)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_faulted) throw new InvalidOperationException("Qwen3.5 GPU state is invalid after a failed step; call Reset before continuing.");
        Qwen35GgufDescriptor d = Descriptor;
        if (overrideEmbedding is null && (uint)tokenId >= (uint)d.VocabularySize)
            throw new ArgumentOutOfRangeException(nameof(tokenId));
        if (overrideEmbedding is not null && (overrideEmbedding.Length != d.EmbeddingLength
            || overrideEmbedding.Any(value => !float.IsFinite(value))))
            throw new ArgumentException("Image embedding width or values are invalid.", nameof(overrideEmbedding));
        if (_position >= d.ContextLength) throw new InvalidOperationException("Qwen3.5 context length exceeded.");
        // Grow caches before mutating any layer's sequence state. Replacement
        // uses D2D copies and keeps old buffers alive until the copy succeeds.
        foreach (LayerState state in _states) state.EnsureCapacity(_position + 1);
        ArcExecutionLane lane = _embedding.Lane;
        ArcBuffer? hidden = null;
        bool fusedResidualRms = _options.InferenceFusedResidualRms && !_options.LoraTraining;
        try
        {
            hidden = overrideEmbedding is null ? _embedding.Embedding(tokenId) : lane.Upload(overrideEmbedding);
            if (overrideEmbedding is null && _prism?.InverseWeights.Contains("token_embd.weight") == true)
            {
                ArcBuffer restored = PrismTransform("token_embd.weight", _embedding, hidden, inverse: true);
                hidden.Dispose(); hidden = restored;
            }
            for (int layer = 0; layer < d.LayerCount; layer++)
            {
                LayerBindings bindings = _layerBindings[layer];
                LayerState state = bindings.State;
                MoveToLane(ref hidden, ref lane, state.Lane, d.EmbeddingLength);
                using ArcBuffer normalized = Qwen35Gpu.RmsNorm(lane, hidden!, bindings.AttentionNorm, 1, d.EmbeddingLength, d.RmsEpsilon);
                using ArcBuffer attention = bindings.Recurrent
                    ? RecurrentAttention(normalized, bindings)
                    : FullAttention(normalized, bindings, ropePosition);
                using ArcBuffer postNorm = AddResidualAndRmsNorm(lane, hidden!, attention,
                    bindings.PostAttentionNorm, d.EmbeddingLength, d.RmsEpsilon, fusedResidualRms);
                using ArcBuffer gate = Project(bindings.FfnGate, postNorm);
                using ArcBuffer up = Project(bindings.FfnUp, postNorm);
                using ArcBuffer activated = Qwen35Gpu.SiluMultiply(lane, gate, up, d.FeedForwardLength);
                using ArcBuffer down = Project(bindings.FfnDown, activated);
                if (fusedResidualRms && returnLogits && layer == d.LayerCount - 1)
                {
                    ArcBuffer finalNorm = AddResidualAndRmsNorm(lane, hidden!, down,
                        _outputNorm, d.EmbeddingLength, d.RmsEpsilon, fused: true);
                    hidden!.Dispose(); hidden = finalNorm;
                }
                else Qwen35Gpu.AddInPlace(lane, hidden!, down, d.EmbeddingLength);
            }
            if (!returnLogits) { _position++; return null; }
            if (!fusedResidualRms)
            {
                ArcBuffer finalNorm = Qwen35Gpu.RmsNorm(lane, hidden!, _outputNorm, 1, d.EmbeddingLength, d.RmsEpsilon);
                hidden!.Dispose(); hidden = finalNorm;
            }
            MoveToLane(ref hidden, ref lane, OutputMatrix.Lane, d.EmbeddingLength);
            ArcBuffer logits = TrySplitOutputHead(hidden!) ?? ProjectMatrix("output.weight", OutputMatrix, hidden!);
            _position++;
            return logits;
        }
        catch { _faulted = true; throw; }
        finally { hidden?.Dispose(); }
    }

    private static void MoveToLane(ref ArcBuffer? buffer, ref ArcExecutionLane source,
        ArcExecutionLane destination, int width)
    {
        if (ReferenceEquals(source, destination)) return;
        // Separate OpenCL contexts cannot share buffers. This is transport,
        // not host arithmetic; all layer computation remains on its GPU.
        var staging = new float[width];
        source.Read(buffer!, staging);
        ArcBuffer moved = destination.Upload(staging);
        buffer!.Dispose(); buffer = moved; source = destination;
    }

    private static ArcBuffer AddResidualAndRmsNorm(ArcExecutionLane lane, ArcBuffer hidden,
        ArcBuffer residual, ArcBuffer weight, int width, float epsilon, bool fused)
    {
        if (!fused)
        {
            Qwen35Gpu.AddInPlace(lane, hidden, residual, width);
            return Qwen35Gpu.RmsNorm(lane, hidden, weight, 1, width, epsilon);
        }
        ArcBuffer normalized = lane.Allocate(width);
        try
        {
            bool fast = lane.Options.Qwen35FastRmsNorm && lane.Options.XmxMatrices
                && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16
                && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups");
            lane.Run(fast ? "q35a_add_rms_norm_sg16_exact" : "q35a_add_rms_norm",
                128, 128, hidden, residual, weight, normalized, width, width, epsilon);
            return normalized;
        }
        catch { normalized.Dispose(); throw; }
    }

    private ArcBuffer Project(string name, ArcBuffer input)
        => ProjectMatrix(name, _matrices[name], input);

    private ArcBuffer Project(Matrix matrix, ArcBuffer input)
        => ProjectMatrix(matrix.Name, matrix, input);

    private ArcBuffer ProjectMatrix(string name, Matrix matrix, ArcBuffer input)
    {
        if (_loraFaulted) throw new InvalidOperationException("LoRA optimizer state is invalid; reload the last saved checkpoint in a new model.");
        if (_prism?.ForwardWeights.Contains(name) == true)
        {
            using ArcBuffer transformed = PrismTransform(name, matrix, input, inverse: false);
            return matrix.Forward(transformed, _zeroBias[matrix.Lane]);
        }
        if (_options.InferenceFusedLora && !_options.LoraTraining && matrix.SupportsFusedLora
            && _lora.TryGetValue(name, out var adapter))
        {
            using ArcBuffer z = adapter.ProjectA(input, 1);
            return matrix.ForwardFusedLora(input, _zeroBias[matrix.Lane], z, adapter);
        }
        ArcBuffer output = matrix.Forward(input, _zeroBias[matrix.Lane]);
        try { ApplyLora(name, input, output, 1); return output; }
        catch { output.Dispose(); throw; }
    }

    private ArcBuffer PrismTransform(string name, Matrix matrix, ArcBuffer input, bool inverse)
    {
        ArcBuffer output = matrix.Lane.Allocate(matrix._inputWidth);
        try
        {
            bool grouped = !inverse && _prism!.GroupedValueHeads && name.EndsWith(".ssm_out.weight", StringComparison.Ordinal);
            matrix.Lane.Run("q35l_prism_hadamard", (long)matrix._inputWidth / 1024 * 256, 256,
                input, _prismSigns[(matrix.Lane, matrix._inputWidth)], output, matrix._inputWidth, 1, inverse ? 1 : 0,
                grouped ? Descriptor.LinearKeyHeads : 0, grouped ? Descriptor.LinearValueHeads : 0, Descriptor.LinearHeadWidth);
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    private ArcBuffer RecurrentAttention(ArcBuffer input, LayerBindings bindings)
    {
        Qwen35GgufDescriptor d = Descriptor;
        LayerState state = bindings.State;
        using ArcBuffer qkv = Project(bindings.Qkv!, input);
        using ArcBuffer gate = Project(bindings.RecurrentGate!, input);
        using ArcBuffer alpha = Project(bindings.Alpha!, input);
        using ArcBuffer beta = Project(bindings.Beta!, input);
        using ArcBuffer delta = _options.FusedDelta ? Qwen35Gpu.DeltaStepFused(state.Lane, qkv, gate, alpha, beta,
            bindings.Convolution!, bindings.DtBias!, bindings.A!,
            bindings.RecurrentNorm!, state.Convolution!, state.Recurrent!,
            d.LinearKeyHeads, d.LinearValueHeads, d.LinearHeadWidth, d.ConvKernel, d.RmsEpsilon,
            cacheState: _options.InferenceCachedDeltaDecode && !_options.LoraTraining)
            : Qwen35Gpu.DeltaStep(state.Lane, qkv, gate, alpha, beta,
            bindings.Convolution!, bindings.DtBias!, bindings.A!,
            bindings.RecurrentNorm!, state.Convolution!, state.Recurrent!,
            d.LinearKeyHeads, d.LinearValueHeads, d.LinearHeadWidth, d.ConvKernel, d.RmsEpsilon);
        return Project(bindings.AttentionOutput, delta);
    }

    private ArcBuffer FullAttention(ArcBuffer input, LayerBindings bindings,
        Qwen35Position? ropePosition = null)
    {
        Qwen35GgufDescriptor d = Descriptor;
        LayerState state = bindings.State;
        using ArcBuffer qAndGate = Project(bindings.Query!, input);
        using ArcBuffer key = Project(bindings.Key!, input);
        using ArcBuffer value = Project(bindings.Value!, input);
        using ArcBuffer attention = Qwen35Gpu.AttentionStep(state.Lane, qAndGate, key, value,
            bindings.QueryNorm!, bindings.KeyNorm!, state.Keys!, state.Values!,
            _position, d.HeadCount, d.KvHeadCount, d.HeadWidth, d.RopeDimensionCount, d.RopeTheta, d.RmsEpsilon,
            ropePosition, d.RopeDimensionSections);
        return Project(bindings.AttentionOutput, attention);
    }

    /// <summary>
    /// Starts a new sequence. Greedy selection stays on the GPU; stochastic
    /// sampling downloads one logits row for top-k and top-p filtering.
    /// The final emitted token has not yet been forwarded.
    /// </summary>
    public int[] GenerateTokenIds(IReadOnlyList<int> prompt, int maxNewTokens,
        int? eosTokenId = null, Action<int>? onToken = null,
        float temperature = 0f, float topP = 1f, int topK = 1, Random? random = null)
        => GenerateTokenIdsCore(prompt, maxNewTokens, eosTokenId, onToken, reusePromptPrefix: false,
            temperature, topP, topK, random, CancellationToken.None);

    public int[] GenerateTokenIds(IReadOnlyList<int> prompt, int maxNewTokens,
        CancellationToken cancellationToken, int? eosTokenId = null, Action<int>? onToken = null,
        float temperature = 0f, float topP = 1f, int topK = 1, Random? random = null)
        => GenerateTokenIdsCore(prompt, maxNewTokens, eosTokenId, onToken, reusePromptPrefix: false,
            temperature, topP, topK, random, cancellationToken);

    /// <summary>
    /// Reuses a previous prompt's GPU state only when its token IDs are an exact
    /// prefix of this prompt. Full-attention K/V rows are retained in place;
    /// recurrent state is restored from a GPU-side copy after each response.
    /// A changed or shorter prompt falls back to a full prefill.
    /// </summary>
    public int[] GenerateTokenIdsWithPrefixReuse(IReadOnlyList<int> prompt, int maxNewTokens,
        int? eosTokenId = null, Action<int>? onToken = null,
        float temperature = 0f, float topP = 1f, int topK = 1, Random? random = null)
        => GenerateTokenIdsCore(prompt, maxNewTokens, eosTokenId, onToken, reusePromptPrefix: true,
            temperature, topP, topK, random, CancellationToken.None);

    public int[] GenerateTokenIdsWithPrefixReuse(IReadOnlyList<int> prompt, int maxNewTokens,
        CancellationToken cancellationToken, int? eosTokenId = null, Action<int>? onToken = null,
        float temperature = 0f, float topP = 1f, int topK = 1, Random? random = null)
        => GenerateTokenIdsCore(prompt, maxNewTokens, eosTokenId, onToken, reusePromptPrefix: true,
            temperature, topP, topK, random, cancellationToken);

    /// <summary>
    /// Advances the resident state to an exact conversation prefix without
    /// sampling. This lets the next user turn reuse a sanitized assistant
    /// answer while leaving earlier thinking text out of the context.
    /// </summary>
    public (int ReusedTokens, bool Cached) PrimePromptPrefix(IReadOnlyList<int> prompt,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _cachedMixedPromptTokens = null;
        ArgumentNullException.ThrowIfNull(prompt);
        if (prompt.Count == 0 || prompt.Count >= Descriptor.ContextLength)
            throw new ArgumentException("Prefix must leave room for a following turn.", nameof(prompt));
        if (prompt.Any(token => (uint)token >= (uint)Descriptor.VocabularySize))
            throw new ArgumentOutOfRangeException(nameof(prompt));
        cancellationToken.ThrowIfCancellationRequested();
        if (_cachedPromptTokens is { } cached && _position == cached.Length
            && _states.All(state => state.HasPromptState) && cached.SequenceEqual(prompt))
        {
            _lastReusedPromptTokens = cached.Length;
            return (cached.Length, true);
        }
        int reused = CanReusePromptPrefix(prompt) ? _cachedPromptTokens!.Length : 0;
        if (reused == 0) Reset();
        _lastReusedPromptTokens = reused;
        try
        {
            foreach (LayerState state in _states) state.EnsureCapacity(prompt.Count);
            int i = reused;
            for (; CanPrefillChunk(prompt.Count - i); i += Math.Min(_options.InferencePrefillChunkTokens, prompt.Count - i))
            {
                cancellationToken.ThrowIfCancellationRequested();
                ForwardPromptChunkDevice(prompt, i, Math.Min(_options.InferencePrefillChunkTokens, prompt.Count - i),
                    cancellationToken);
            }
            for (; i < prompt.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                _ = ForwardTokenDevice(prompt[i], returnLogits: false);
            }
            cancellationToken.ThrowIfCancellationRequested();
            if (!TryCapturePromptCheckpoint()) return (reused, false);
            _cachedPromptTokens = prompt.ToArray();
            return (reused, true);
        }
        catch { _faulted = true; InvalidatePromptCheckpoint(); throw; }
    }

    private int[] GenerateTokenIdsCore(IReadOnlyList<int> prompt, int maxNewTokens,
        int? eosTokenId, Action<int>? onToken, bool reusePromptPrefix,
        float temperature, float topP, int topK, Random? random,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentOutOfRangeException.ThrowIfNegative(maxNewTokens);
        ValidateSampling(temperature, topP, topK);
        if (prompt.Count == 0 || prompt.Count > Descriptor.ContextLength)
            throw new ArgumentException("Prompt must contain 1..context-length tokens.", nameof(prompt));
        if (prompt.Any(token => (uint)token >= (uint)Descriptor.VocabularySize))
            throw new ArgumentOutOfRangeException(nameof(prompt));
        cancellationToken.ThrowIfCancellationRequested();
        int reused = reusePromptPrefix && CanReusePromptPrefix(prompt)
            ? _cachedPromptTokens!.Length : 0;
        if (reused == 0) Reset();
        _lastReusedPromptTokens = reused;
        var result = prompt.ToList();
        if (maxNewTokens == 0 || result.Count == Descriptor.ContextLength)
        {
            // No final logits are computed in this path, so it cannot seed a
            // later continuation even when an earlier prefix was reusable.
            InvalidatePromptCheckpoint();
            return result.ToArray();
        }
        ArcBuffer? logits = null;
        bool checkpointCaptured = false;
        bool greedy = temperature == 0f || topK == 1;
        float[]? hostLogits = greedy ? null : new float[Descriptor.VocabularySize];
        random ??= Random.Shared;
        try
        {
            foreach (LayerState state in _states) state.EnsureCapacity(prompt.Count);
            int i = reused;
            for (; CanPrefillChunk(prompt.Count - 1 - i); i += Math.Min(_options.InferencePrefillChunkTokens, prompt.Count - 1 - i))
            {
                cancellationToken.ThrowIfCancellationRequested();
                ForwardPromptChunkDevice(prompt, i, Math.Min(_options.InferencePrefillChunkTokens, prompt.Count - 1 - i),
                    cancellationToken);
            }
            for (; i < prompt.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                logits = ForwardTokenDevice(prompt[i], i == prompt.Count - 1);
            }
            cancellationToken.ThrowIfCancellationRequested();
            if (reusePromptPrefix)
                checkpointCaptured = TryCapturePromptCheckpoint();
            for (int generated = 0; generated < maxNewTokens && result.Count < Descriptor.ContextLength; generated++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int next;
                if (greedy)
                    next = Qwen35Gpu.ArgMax(OutputMatrix.Lane, logits!, Descriptor.VocabularySize);
                else
                {
                    OutputMatrix.Lane.Read(logits!, hostLogits!);
                    next = SampleLogits(hostLogits, temperature, topP, topK, random);
                }
                logits!.Dispose(); logits = null;
                result.Add(next);
                onToken?.Invoke(next);
                cancellationToken.ThrowIfCancellationRequested();
                if (next == eosTokenId || generated + 1 == maxNewTokens || result.Count == Descriptor.ContextLength) break;
                logits = ForwardTokenDevice(next, true);
            }
            if (checkpointCaptured)
            {
                foreach (LayerState state in _states) state.RestorePromptState();
                foreach (ArcExecutionLane lane in _lanes) lane.Synchronize();
                _position = prompt.Count;
                _cachedPromptTokens = prompt.ToArray();
            }
            return result.ToArray();
        }
        catch { _faulted = true; InvalidatePromptCheckpoint(); throw; }
        finally { logits?.Dispose(); }
    }

    private static void ValidateSampling(float temperature, float topP, int topK)
    {
        if (!float.IsFinite(temperature) || temperature < 0f)
            throw new ArgumentOutOfRangeException(nameof(temperature));
        if (!float.IsFinite(topP) || topP <= 0f || topP > 1f)
            throw new ArgumentOutOfRangeException(nameof(topP));
        if (topK < 1)
            throw new ArgumentOutOfRangeException(nameof(topK));
    }

    /// <summary>
    /// Selects top-k before nucleus filtering, as required by Qwen sampling.
    /// The returned ID is reproducible when the caller supplies a seeded Random.
    /// </summary>
    internal static int SampleLogits(ReadOnlySpan<float> logits,
        float temperature, float topP, int topK, Random random)
    {
        ValidateSampling(temperature, topP, topK);
        ArgumentNullException.ThrowIfNull(random);
        if (logits.IsEmpty) throw new ArgumentException("Logits cannot be empty.", nameof(logits));

        int bestId = 0;
        float bestValue = float.NegativeInfinity;
        int count = Math.Min(topK, logits.Length);
        bool greedy = temperature == 0f || count == 1;
        var candidates = greedy ? null :
            new PriorityQueue<(int Id, float Value), (float Value, int NegativeId)>(count);
        for (int id = 0; id < logits.Length; id++)
        {
            float value = logits[id];
            if (!float.IsFinite(value))
                throw new ArithmeticException("Qwen3.5 produced non-finite logits.");
            if (value > bestValue) { bestValue = value; bestId = id; }
            if (greedy) continue;

            (float Value, int NegativeId) priority = (value, -id);
            if (candidates!.Count < count)
                candidates.Enqueue((id, value), priority);
            else if (candidates.TryPeek(out _, out var worst) && priority.CompareTo(worst) > 0)
            {
                candidates.Dequeue();
                candidates.Enqueue((id, value), priority);
            }
        }
        if (greedy) return bestId;

        (int Id, float Value)[] ordered = candidates!.UnorderedItems
            .Select(item => item.Element).ToArray();
        Array.Sort(ordered, (left, right) =>
        {
            int byValue = right.Value.CompareTo(left.Value);
            return byValue != 0 ? byValue : left.Id.CompareTo(right.Id);
        });
        var weights = new double[ordered.Length];
        double sum = 0;
        double maximum = ordered[0].Value;
        for (int i = 0; i < ordered.Length; i++)
        {
            double weight = Math.Exp(((double)ordered[i].Value - maximum) / temperature);
            weights[i] = weight;
            sum += weight;
        }
        double nucleusMass = 0;
        int nucleusCount = 0;
        do
        {
            nucleusMass += weights[nucleusCount++];
        } while (nucleusCount < ordered.Length && nucleusMass / sum < topP);

        double draw = random.NextDouble() * nucleusMass;
        double cumulative = 0;
        for (int i = 0; i < nucleusCount; i++)
        {
            cumulative += weights[i];
            if (draw < cumulative) return ordered[i].Id;
        }
        return ordered[nucleusCount - 1].Id;
    }

    private bool CanReusePromptPrefix(IReadOnlyList<int> prompt)
    {
        int[]? cached = _cachedPromptTokens;
        if (_faulted || cached is null || cached.Length >= prompt.Count || _position != cached.Length
            || _states.Any(state => !state.HasPromptState)) return false;
        for (int i = 0; i < cached.Length; i++)
            if (cached[i] != prompt[i]) return false;
        return true;
    }

    private bool TryCapturePromptCheckpoint()
    {
        // The previous snapshot is no longer a valid prefix once the prompt
        // advances, but its GPU buffers can hold the new recurrent state.
        _cachedPromptTokens = null;
        _cachedMixedPromptTokens = null;
        foreach (ArcExecutionLane lane in _lanes)
        {
            long bytes = _states.Where(state => ReferenceEquals(state.Lane, lane))
                .Sum(state => state.AdditionalPromptSnapshotBytes);
            if (lane.AllocatedBytes + bytes + WorkspaceReserveBytes > lane.EffectivePhysicalBufferBudgetBytes)
            {
                InvalidatePromptCheckpoint();
                return false; // Generate normally when a snapshot will not fit.
            }
        }
        try
        {
            foreach (LayerState state in _states) state.CapturePromptState();
            foreach (ArcExecutionLane lane in _lanes) lane.Synchronize();
            return true;
        }
        catch
        {
            InvalidatePromptCheckpoint();
            throw;
        }
    }

    private void InvalidatePromptCheckpoint()
    {
        _cachedPromptTokens = null;
        _cachedMixedPromptTokens = null;
        foreach (LayerState state in _states) state.ClearPromptState();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var adapter in _lora.Values) adapter.Dispose();
        _lora.Clear();
        _splitOutputHead?.Dispose(); _splitOutputHead = null;
        foreach (LayerState state in _states) state.Dispose();
        foreach (Matrix matrix in _matrices.Values) matrix.Dispose();
        foreach (ArcBuffer buffer in _dense.Values) buffer.Dispose();
        foreach (ArcBuffer buffer in _zeroBias.Values) buffer.Dispose();
        foreach (ArcBuffer buffer in _prismSigns.Values) buffer.Dispose();
        foreach (ArcExecutionLane lane in _lanes) lane.Dispose();
        _modelSource?.Dispose();
        _states.Clear(); _matrices.Clear(); _dense.Clear(); _zeroBias.Clear(); _prismSigns.Clear();
        _layerBindings.Clear(); _embedding = null!; _outputNorm = null!;
    }

    // Non-owning references resolved once after the validated tensor directory
    // is loaded. LoRA attachment remains dynamic, keyed by each matrix's name.
    private sealed class LayerBindings
    {
        internal readonly LayerState State;
        internal readonly bool Recurrent;
        internal readonly ArcBuffer AttentionNorm, PostAttentionNorm;
        internal readonly Matrix FfnGate, FfnUp, FfnDown, AttentionOutput;
        internal readonly Matrix? Qkv, RecurrentGate, Alpha, Beta, Query, Key, Value;
        internal readonly ArcBuffer? Convolution, DtBias, A, RecurrentNorm, QueryNorm, KeyNorm;

        internal LayerBindings(Qwen35QuantizedModel model, int layer, LayerState state)
        {
            State = state;
            Recurrent = model.Descriptor.IsRecurrent(layer);
            string prefix = $"blk.{layer}.";
            Matrix GetMatrix(string suffix) => model._matrices[prefix + suffix];
            ArcBuffer GetDense(string suffix) => model._dense[prefix + suffix];
            AttentionNorm = GetDense("attn_norm.weight");
            PostAttentionNorm = GetDense("post_attention_norm.weight");
            FfnGate = GetMatrix("ffn_gate.weight");
            FfnUp = GetMatrix("ffn_up.weight");
            FfnDown = GetMatrix("ffn_down.weight");
            if (Recurrent)
            {
                Qkv = GetMatrix("attn_qkv.weight");
                RecurrentGate = GetMatrix("attn_gate.weight");
                Alpha = GetMatrix("ssm_alpha.weight");
                Beta = GetMatrix("ssm_beta.weight");
                AttentionOutput = GetMatrix("ssm_out.weight");
                Convolution = GetDense("ssm_conv1d.weight");
                DtBias = GetDense("ssm_dt.bias");
                A = GetDense("ssm_a");
                RecurrentNorm = GetDense("ssm_norm.weight");
            }
            else
            {
                Query = GetMatrix("attn_q.weight");
                Key = GetMatrix("attn_k.weight");
                Value = GetMatrix("attn_v.weight");
                AttentionOutput = GetMatrix("attn_output.weight");
                QueryNorm = GetDense("attn_q_norm.weight");
                KeyNorm = GetDense("attn_k_norm.weight");
            }
        }
    }

    private sealed class LayerState(ArcExecutionLane lane, Qwen35GgufDescriptor d, bool recurrent) : IDisposable
    {
        internal ArcExecutionLane Lane { get; } = lane;
        internal ArcBuffer? Convolution, Recurrent, Keys, Values;
        private ArcBuffer? _promptConvolution, _promptRecurrent;
        private int _capacity;
        internal long StorageBytes => (Convolution?.ByteLength ?? 0) + (Recurrent?.ByteLength ?? 0)
            + (Keys?.ByteLength ?? 0) + (Values?.ByteLength ?? 0)
            + (_promptConvolution?.ByteLength ?? 0) + (_promptRecurrent?.ByteLength ?? 0);
        internal long AdditionalPromptSnapshotBytes => recurrent
            ? (_promptConvolution is null ? Convolution?.ByteLength ?? 0 : 0)
                + (_promptRecurrent is null ? Recurrent?.ByteLength ?? 0 : 0) : 0;
        internal bool HasPromptState => !recurrent || (_promptConvolution is not null && _promptRecurrent is not null);
        internal void Initialize()
        {
            if (recurrent)
            {
                Convolution = Lane.Allocate(checked((2 * d.LinearKeyHeads + d.LinearValueHeads) * d.LinearHeadWidth * (d.ConvKernel - 1)));
                Recurrent = Lane.Allocate(checked(d.LinearValueHeads * d.LinearHeadWidth * d.LinearHeadWidth));
                Reset();
            }
            else EnsureCapacity(Math.Min(16, d.ContextLength));
        }
        internal void Reset()
        {
            ClearPromptState();
            if (Convolution is not null) Zero(Convolution);
            if (Recurrent is not null) Zero(Recurrent);
            // Cached rows beyond the current position are never read. Resetting
            // the model position is enough; capacity remains reusable on device.
        }
        internal void CapturePromptState()
        {
            if (!recurrent) return;
            ArcBuffer convolutionBuffer = Convolution!;
            ArcBuffer recurrentBuffer = Recurrent!;
            try
            {
                _promptConvolution ??= Lane.AllocateBytes(checked((int)convolutionBuffer.ByteLength));
                _promptRecurrent ??= Lane.AllocateBytes(checked((int)recurrentBuffer.ByteLength));
                Lane.CopyBytes(convolutionBuffer, _promptConvolution, 0, 0, checked((int)convolutionBuffer.ByteLength));
                Lane.CopyBytes(recurrentBuffer, _promptRecurrent, 0, 0, checked((int)recurrentBuffer.ByteLength));
            }
            catch { ClearPromptState(); throw; }
        }
        internal void RestorePromptState()
        {
            if (!recurrent) return;
            if (!HasPromptState) throw new InvalidOperationException("Qwen3.5 prompt state is unavailable.");
            Lane.CopyBytes(_promptConvolution!, Convolution!, 0, 0, checked((int)Convolution!.ByteLength));
            Lane.CopyBytes(_promptRecurrent!, Recurrent!, 0, 0, checked((int)Recurrent!.ByteLength));
        }
        internal void ClearPromptState()
        {
            _promptConvolution?.Dispose();
            _promptRecurrent?.Dispose();
            _promptConvolution = _promptRecurrent = null;
        }
        private void Zero(ArcBuffer buffer)
        {
            int length = checked((int)(buffer.ByteLength / 4));
            Lane.Run("q35a_zero", length, 0, buffer, length);
        }
        internal void EnsureCapacity(int required)
        {
            if (recurrent || required <= _capacity) return;
            int capacity = (int)Math.Min(d.ContextLength, Math.Max(required, Math.Max(16L, 2L * _capacity)));
            long bytes = checked((long)capacity * d.KvHeadCount * d.HeadWidth * 4);
            if (bytes > int.MaxValue || (ulong)bytes > Lane.Device.MaximumAllocationBytes)
                throw new NotSupportedException("Qwen3.5 KV cache exceeds the device allocation limit.");
            if (Lane.AllocatedBytes + 2 * bytes + WorkspaceReserveBytes > Lane.EffectivePhysicalBufferBudgetBytes)
                throw new NotSupportedException("Qwen3.5 KV cache growth exceeds the GPU memory budget; use a shorter context or more devices.");
            ArcBuffer? keys = null, values = null;
            try
            {
                keys = Lane.AllocateBytes((int)bytes); values = Lane.AllocateBytes((int)bytes);
                if (Keys is not null)
                {
                    int oldBytes = checked((int)Keys.ByteLength);
                    Lane.CopyBytes(Keys, keys, 0, 0, oldBytes);
                    Lane.CopyBytes(Values!, values, 0, 0, oldBytes);
                    Lane.Synchronize();
                }
                Keys?.Dispose(); Values?.Dispose();
                Keys = keys; Values = values; keys = values = null; _capacity = capacity;
            }
            finally { keys?.Dispose(); values?.Dispose(); }
        }
        public void Dispose()
        {
            ClearPromptState();
            Convolution?.Dispose(); Recurrent?.Dispose(); Keys?.Dispose(); Values?.Dispose();
        }
    }

    private sealed class Matrix : IDisposable
    {
        internal readonly ArcExecutionLane Lane;
        internal readonly string Name;
        internal readonly int StorageBytes, OutputWidth;
        private readonly ArcBuffer _encoded;
        private readonly ArcBuffer? _residentGrid, _residentScales, _residentBlockD;
        private readonly uint _type;
        private readonly string _quantization;
        internal readonly int _inputWidth;
        private readonly Qwen35QuantizedKernel _kernel;
        private readonly int _transposeRows;
        private readonly int _transposeOctetRows;
        private readonly int _forwardRows;
        private readonly int _q5ForwardRows;
        private readonly bool _trainingIq2Bf16XmxForward;
        private readonly bool _trainingIq2Fp16XmxForward;
        private readonly bool _trainingIq2RollingTranspose;
        private readonly string _projectionKernel, _fusedProjectionKernel;
        private readonly string _rowProjectionKernelPrefix;
        private readonly bool _inferenceXmxPrefill;
        private readonly bool _inferenceXmxPackedPrefill;
        private readonly bool _inferenceXmxFactoredPrefill;
        private readonly bool _inferenceXmxGgufBslmPrefill;
        private readonly bool _iq2TiledForward, _iq2TiledBackward;
        private readonly bool PairedProjection;
        private readonly bool _iq2DecodePair4;
        private readonly bool _iq2DecodeLevel3;
        private readonly bool PairedLoraProjection;
        internal bool SupportsFusedLora => _kernel == Qwen35QuantizedKernel.Subgroup
            && _type is not (Qwen35Gguf.PQ20Type or Qwen35Gguf.PTQ10Type or Qwen2Gguf.BF16Type);
        internal bool SupportsSharedPrefillProjection => !Lane.Options.Qwen35TrainingKernels
            && (_forwardRows > 1 || _inferenceXmxPrefill || _iq2TiledForward) && _kernel == Qwen35QuantizedKernel.Subgroup
            && _type is Qwen2Gguf.Q4KType or Qwen35Gguf.IQ2SType or Qwen35Gguf.IQ3SType;
        internal bool IsIq2S => _type == Qwen35Gguf.IQ2SType;
        internal uint StorageType => _type;
        internal bool SupportsSplitOutputHead => _type == Qwen35Gguf.Q5KType
            && _kernel == Qwen35QuantizedKernel.Subgroup && !Lane.Options.Qwen35TrainingKernels;
        internal ArcBuffer EncodedOutputHead => _encoded;
        internal Matrix(ArcExecutionLane lane, GgufTensorInfo info, byte[] payload, Qwen35QuantizedKernel kernel, int transposeRows, int forwardRows, int q5ForwardRows, bool trainingIq2Bf16XmxForward, bool trainingIq2Fp16XmxForward, bool trainingIq2RollingTranspose, int transposeOctetRows, bool inferenceXmxPrefill = false, bool inferenceXmxPackedPrefill = false, bool inferenceXmxFactoredPrefill = false, bool inferenceResidentIq2Panels = false, bool inferenceXmxGgufBslmPrefill = false, bool iq2TiledForward = false, bool iq2TiledBackward = false, bool inferenceIq2DecodePair4 = false, bool inferenceIq2DecodeLevel3 = false)
        {
            _transposeRows = transposeRows;
            _forwardRows = forwardRows;
            _rowProjectionKernelPrefix = lane.Options.Qwen35TrainingKernels ? "q35t_linear_" : "q35l_prefill_linear_";
            _q5ForwardRows = q5ForwardRows;
            _trainingIq2Bf16XmxForward = trainingIq2Bf16XmxForward;
            _trainingIq2Fp16XmxForward = trainingIq2Fp16XmxForward;
            _trainingIq2RollingTranspose = trainingIq2RollingTranspose;
            Lane = lane; Name = info.Name; StorageBytes = payload.Length; _type = info.Type;
            _quantization = _type switch
            {
                Qwen2Gguf.Q4KType => "q4_k", Qwen35Gguf.Q5KType => "q5_k",
                Qwen2Gguf.Q6KType => "q6_k", Qwen35Gguf.IQ2SType => "iq2_s",
                Qwen35Gguf.IQ3SType => "iq3_s",
                Qwen35Gguf.PQ20Type => "pq2_0", Qwen35Gguf.PTQ10Type => "ptq1_0",
                Qwen2Gguf.BF16Type => "bf16",
                _ => throw new NotSupportedException($"Unsupported matrix storage type {_type}.")
            };
            _inputWidth = checked((int)info.Shape[0]); OutputWidth = checked((int)info.Shape[1]);
            bool subgroup = lane.Options.XmxMatrices && lane.Device.SupportsXmx
                && lane.Device.MinimumSubgroupSize == 16
                && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups");
            _transposeOctetRows = subgroup ? transposeOctetRows : 0;
            _kernel = kernel == Qwen35QuantizedKernel.Auto
                // PQ2's cooperative reduction is faster on the measured Arc
                // workload; PTQ1 benefits from its dedicated SG16 block loop.
                ? (subgroup && _type != Qwen35Gguf.PQ20Type
                    ? Qwen35QuantizedKernel.Subgroup : Qwen35QuantizedKernel.Cooperative) : kernel;
            _inferenceXmxPrefill = (inferenceXmxPrefill || inferenceResidentIq2Panels
                || (inferenceXmxGgufBslmPrefill && _type == Qwen35Gguf.IQ2SType)) && !lane.Options.Qwen35TrainingKernels
                && _kernel == Qwen35QuantizedKernel.Subgroup
                && lane.Device.Extensions.Split(' ').Contains("cl_khr_fp16")
                && _type is Qwen2Gguf.Q4KType or Qwen35Gguf.IQ2SType or Qwen35Gguf.IQ3SType
                && HasFiniteF16DecodedWeightRange(payload, _type);
            _inferenceXmxPackedPrefill = inferenceXmxPackedPrefill && _inferenceXmxPrefill;
            _inferenceXmxFactoredPrefill = inferenceXmxFactoredPrefill && _inferenceXmxPrefill
                && _type is Qwen35Gguf.IQ2SType or Qwen35Gguf.IQ3SType;
            _inferenceXmxGgufBslmPrefill = inferenceXmxGgufBslmPrefill && _inferenceXmxPrefill
                && _type == Qwen35Gguf.IQ2SType;
            _iq2TiledForward = iq2TiledForward && IsIq2S && lane.SupportsIq2TiledProjection
                && _kernel == Qwen35QuantizedKernel.Subgroup && !trainingIq2Bf16XmxForward && !trainingIq2Fp16XmxForward;
            _iq2TiledBackward = iq2TiledBackward && IsIq2S && lane.SupportsIq2TiledProjection
                && _kernel == Qwen35QuantizedKernel.Subgroup;
            if (_kernel == Qwen35QuantizedKernel.Subgroup && !subgroup)
                throw new NotSupportedException("The Qwen3.5 subgroup kernel requires Intel SG16 support.");
            PairedProjection = (lane.Options.Qwen35PairedProjection
                || (lane.Options.Qwen35PairedProjectionTypes & (_type switch {
                    Qwen35Gguf.IQ2SType => 1, Qwen2Gguf.Q4KType => 2, Qwen35Gguf.IQ3SType => 4, _ => 0 })) != 0)
                && _type is Qwen2Gguf.Q4KType or Qwen35Gguf.IQ2SType or Qwen35Gguf.IQ3SType;
            _projectionKernel = $"q35l_{_quantization}_sg16" + (PairedProjection ? "_pair" : "");
            _iq2DecodePair4 = inferenceIq2DecodePair4 && !lane.Options.Qwen35TrainingKernels
                && _kernel == Qwen35QuantizedKernel.Subgroup && _type == Qwen35Gguf.IQ2SType;
            _iq2DecodeLevel3 = inferenceIq2DecodeLevel3 && !lane.Options.Qwen35TrainingKernels
                && _kernel == Qwen35QuantizedKernel.Subgroup && _type == Qwen35Gguf.IQ2SType;
            PairedLoraProjection = PairedProjection && lane.Options.Qwen35PairedLoraProjection;
            _fusedProjectionKernel = $"q35l_{_quantization}_sg16" + (PairedLoraProjection ? "_pair" : "") + "_lora";
            if (inferenceResidentIq2Panels && _inferenceXmxPrefill && CanUseResidentIq2Panels(info))
            {
                int columns = checked((OutputWidth + 15) / 16 * 16);
                int gridBytes = checked(_inputWidth / 32 * columns * 12);
                int scaleBytes = checked(_inputWidth / 32 * columns);
                int coefficientBytes = checked(_inputWidth / 256 * columns * sizeof(ushort));
                try
                {
                    _residentGrid = lane.AllocateBytes(gridBytes);
                    _residentScales = lane.AllocateBytes(scaleBytes);
                    _residentBlockD = lane.AllocateBytes(coefficientBytes);
                    using ArcBuffer encoded = lane.UploadRaw(payload);
                    lane.Run("q35l_prefill_xmx_pack_iq2_s_grid3", (long)_inputWidth / 32 * OutputWidth, 0,
                        encoded, _residentGrid, _residentScales, _residentBlockD, _inputWidth, OutputWidth);
                    // Publish the panels before releasing their temporary GGUF input.
                    lane.Synchronize();
                    StorageBytes = checked(gridBytes + scaleBytes + coefficientBytes);
                    _encoded = null!;
                }
                catch
                {
                    _residentGrid?.Dispose(); _residentScales?.Dispose(); _residentBlockD?.Dispose();
                    throw;
                }
            }
            else _encoded = lane.UploadRaw(payload);
        }
        internal ArcBuffer Forward(ArcBuffer input, ArcBuffer zeroBias, int rows = 1)
        {
            if (input.ByteLength < 4L * rows * _inputWidth) throw new ArgumentException("Qwen3.5 projection width mismatch.");
            ArcBuffer output = Lane.Allocate(checked(rows * OutputWidth));
            try
            {
                if (_residentGrid is not null)
                {
                    if (rows >= 16)
                    {
                        int elements = checked((rows + 7) / 8 * 8 * _inputWidth);
                        using ArcBuffer high = Lane.AllocateBytes(checked(elements * sizeof(ushort)));
                        using ArcBuffer low = Lane.AllocateBytes(checked(elements * sizeof(ushort)));
                        using ArcBuffer rangeStatus = Lane.Allocate(1);
                        Lane.Run("q35a_zero", 1, 0, rangeStatus, 1);
                        Lane.Run("q35l_prefill_xmx_pack_input_f16x2", elements, 0, input, high, low, rangeStatus, rows, _inputWidth);
                        Lane.Run2D("q35l_prefill_xmx_iq2_s_grid3_bslm", ((long)OutputWidth + 63) / 64 * 16,
                            ((long)rows + 127) / 128 * 16, 16, 16, high, low, _residentGrid, _residentScales!, _residentBlockD!,
                            zeroBias, output, rows, _inputWidth, OutputWidth, input, rangeStatus);
                    }
                    else
                    {
                        int group = PairedProjection ? 32 : Lane.Options.Qwen35ProjectionWorkgroupSize;
                        long global = PairedProjection
                            ? ((long)rows * ((OutputWidth + 1) / 2) + 1) / 2 * 32
                            : ((long)rows * OutputWidth + group / 16 - 1) / (group / 16) * group;
                        Lane.Run(PairedProjection ? "q35l_iq2_s_grid3_sg16_pair" : "q35l_iq2_s_grid3_sg16_fast",
                            global, group, input, _residentGrid, _residentScales!, _residentBlockD!,
                            zeroBias, output, rows, _inputWidth, OutputWidth);
                    }
                }
                // Training uses block-factored integer panels. Inference keeps
                // the original K16 coefficient order to preserve model logits.
                else if (Lane.Options.Qwen35TrainingKernels && _iq2TiledForward && !_inferenceXmxPrefill && rows >= 128
                    && Lane.AllocatedBytes + Lane.Iq2TiledForwardWorkspaceBytes(rows, _inputWidth, OutputWidth)
                        + 16L * 1024 * 1024 < Lane.EffectivePhysicalBufferBudgetBytes)
                {
                    Lane.Iq2TiledForward(input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                }
                else if (_inferenceXmxPrefill && rows >= 16)
                {
                    if (_inferenceXmxGgufBslmPrefill)
                    {
                        // Row-major A removes the long stride in each XMX
                        // reduction while retaining the old K16 arithmetic.
                        bool rowMajor = _iq2TiledForward && rows >= 512;
                        int elements = checked((rows + 7) / 8 * 8 * _inputWidth);
                        using ArcBuffer high = Lane.AllocateBytes(checked(elements * sizeof(ushort)));
                        using ArcBuffer low = Lane.AllocateBytes(checked(elements * sizeof(ushort)));
                        using ArcBuffer rangeStatus = Lane.Allocate(1);
                        Lane.Run("q35a_zero", 1, 0, rangeStatus, 1);
                        Lane.Run(rowMajor ? "q35l_prefill_xmx_pack_input_f16x2_rowmajor" : "q35l_prefill_xmx_pack_input_f16x2",
                            elements, 0, input, high, low, rangeStatus, rows, _inputWidth);
                        Lane.Run2D(rowMajor ? "q35l_prefill_xmx_iq2_s_gguf_bslm_k32r" : "q35l_prefill_xmx_iq2_s_gguf_bslm", ((long)OutputWidth + 63) / 64 * 16,
                            ((long)rows + 127) / 128 * 16, 16, 16, high, low, _encoded, zeroBias, output,
                            rows, _inputWidth, OutputWidth, input, rangeStatus);
                    }
                    else if (_inferenceXmxFactoredPrefill && rows >= 384
                        && checked((long)_inputWidth * ((OutputWidth + 15L) / 16 * 16) * 9 / 4) <= 512L * 1024 * 1024)
                    {
                        int inputElements = checked((rows + 7) / 8 * 8 * _inputWidth);
                        int weightElements = checked((OutputWidth + 15) / 16 * 16 * _inputWidth);
                        using ArcBuffer high = Lane.AllocateBytes(checked(inputElements * sizeof(ushort)));
                        using ArcBuffer low = Lane.AllocateBytes(checked(inputElements * sizeof(ushort)));
                        using ArcBuffer grid = Lane.AllocateBytes(checked(weightElements * sizeof(ushort)));
                        using ArcBuffer scales = Lane.Allocate(weightElements / 16);
                        using ArcBuffer rangeStatus = Lane.Allocate(1);
                        Lane.Run("q35a_zero", 1, 0, rangeStatus, 1);
                        Lane.Run("q35l_prefill_xmx_pack_input_f16x2", inputElements, 0, input, high, low, rangeStatus, rows, _inputWidth);
                        Lane.Run($"q35l_prefill_xmx_pack_{_quantization}_factored", (long)_inputWidth * OutputWidth / 8, 0,
                            _encoded, grid, scales, _inputWidth, OutputWidth);
                        Lane.Run2D("q35l_prefill_xmx_packed_factored", ((long)OutputWidth + 63) / 64 * 16,
                            ((long)rows + 127) / 128 * 16, 16, 16, high, low, grid, scales, zeroBias, output,
                            rows, _inputWidth, OutputWidth, input, _encoded, rangeStatus,
                            _type == Qwen35Gguf.IQ2SType ? 0 : 1);
                    }
                    else if (_inferenceXmxPackedPrefill && rows >= 128
                        && checked((long)_inputWidth * ((OutputWidth + 15L) / 16 * 16) * 4) <= 512L * 1024 * 1024)
                    {
                        int inputElements = checked((rows + 7) / 8 * 8 * _inputWidth);
                        int weightElements = checked((OutputWidth + 15) / 16 * 16 * _inputWidth);
                        using ArcBuffer high = Lane.AllocateBytes(checked(inputElements * sizeof(ushort)));
                        using ArcBuffer low = Lane.AllocateBytes(checked(inputElements * sizeof(ushort)));
                        using ArcBuffer weightHigh = Lane.AllocateBytes(checked(weightElements * sizeof(ushort)));
                        using ArcBuffer weightLow = Lane.AllocateBytes(checked(weightElements * sizeof(ushort)));
                        using ArcBuffer rangeStatus = Lane.Allocate(1);
                        Lane.Run("q35a_zero", 1, 0, rangeStatus, 1);
                        Lane.Run("q35l_prefill_xmx_pack_input_f16x2", inputElements, 0, input, high, low, rangeStatus, rows, _inputWidth);
                        Lane.Run($"q35l_prefill_xmx_pack_{_quantization}_f16x2", (long)_inputWidth * OutputWidth / 8, 0,
                            _encoded, weightHigh, weightLow, _inputWidth, OutputWidth);
                        Lane.Run2D("q35l_prefill_xmx_packed_f16x2", ((long)OutputWidth + 63) / 64 * 16,
                            ((long)rows + 127) / 128 * 16, 16, 16, high, low, weightHigh, weightLow, zeroBias, output,
                            rows, _inputWidth, OutputWidth, input, _encoded, rangeStatus,
                            _type == Qwen35Gguf.IQ2SType ? 0 : _type == Qwen35Gguf.IQ3SType ? 1 : 2);
                    }
                    else
                    {
                        int elements = checked(rows * _inputWidth);
                        using ArcBuffer high = Lane.AllocateBytes(checked(elements * sizeof(ushort)));
                        using ArcBuffer low = Lane.AllocateBytes(checked(elements * sizeof(ushort)));
                        using ArcBuffer rangeStatus = Lane.Allocate(1);
                        Lane.Run("q35a_zero", 1, 0, rangeStatus, 1);
                        Lane.Run("q35l_prefill_xmx_input_f16x2", elements, 0, input, high, low, rangeStatus, elements);
                        Lane.Run2D($"q35l_prefill_xmx_{_quantization}_" + (_inferenceXmxFactoredPrefill ? "factored" : "f16x2"),
                            ((long)OutputWidth + 63) / 64 * 16, ((long)rows + 127) / 128 * 16,
                            16, 16, high, low, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth, input, rangeStatus);
                    }
                }
                else if ((_trainingIq2Bf16XmxForward || _trainingIq2Fp16XmxForward)
                    && _kernel == Qwen35QuantizedKernel.Subgroup
                    && IsIq2S && rows > 1)
                {
                    using ArcBuffer input16 = Lane.AllocateBytes(checked(rows * _inputWidth * sizeof(ushort)));
                    Lane.Run(_trainingIq2Fp16XmxForward ? "q35t_iq2_input_f16" : "q35t_iq2_input_bf16",
                        (long)rows * _inputWidth, 0, input, input16, checked(rows * _inputWidth));
                    if (_trainingIq2Fp16XmxForward && rows >= 128)
                        Lane.Run2D("q35t_linear_iq2_s_xmx_f16_r128_n64",
                            ((long)OutputWidth + 63) / 64 * 16, ((long)rows + 127) / 128 * 16,
                            16, 16, input16, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                    else
                        Lane.Run2D(_trainingIq2Fp16XmxForward
                                ? "q35t_linear_iq2_s_xmx_f16" : "q35t_linear_iq2_s_xmx_bf16",
                            ((long)OutputWidth + 31) / 32 * 16, ((long)rows + 63) / 64 * 8,
                            16, 8, input16, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                }
                else if (_kernel == Qwen35QuantizedKernel.Subgroup
                    && _type == Qwen35Gguf.Q5KType && _q5ForwardRows > 1 && rows > 1)
                    Lane.Run($"q35t_linear_q5_k_rows{_q5ForwardRows}",
                        ((((long)rows + _q5ForwardRows - 1) / _q5ForwardRows * OutputWidth + 1) / 2) * 32, 32,
                        input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                else if (_kernel == Qwen35QuantizedKernel.Subgroup && _forwardRows > 1 && rows > 1
                    && _type is Qwen2Gguf.Q4KType or Qwen35Gguf.IQ2SType or Qwen35Gguf.IQ3SType)
                    Lane.Run($"{_rowProjectionKernelPrefix}{_quantization}_rows{_forwardRows}",
                        ((((long)rows + _forwardRows - 1) / _forwardRows * OutputWidth + 1) / 2) * 32, 32,
                        input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                else if (_iq2DecodeLevel3 && rows == 1)
                    Lane.Run("q35l_iq2_s_sg16_levels3", ((long)((OutputWidth + 1) / 2) + 1) / 2 * 32,
                        32, input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                else if (_iq2DecodePair4 && rows == 1)
                    Lane.Run("q35l_iq2_s_sg16_pair4", ((long)((OutputWidth + 3) / 4) + 1) / 2 * 32,
                        32, input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                else if (_kernel == Qwen35QuantizedKernel.Subgroup && PairedProjection)
                    Lane.Run(_projectionKernel, ((long)rows * ((OutputWidth + 1) / 2) + 1) / 2 * 32,
                        32, input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                else if (_kernel == Qwen35QuantizedKernel.Subgroup)
                    Lane.Run(_projectionKernel,
                        ((long)rows * OutputWidth + Lane.Options.Qwen35ProjectionWorkgroupSize / 16 - 1)
                            / (Lane.Options.Qwen35ProjectionWorkgroupSize / 16) * Lane.Options.Qwen35ProjectionWorkgroupSize,
                        Lane.Options.Qwen35ProjectionWorkgroupSize, input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                else if (_kernel == Qwen35QuantizedKernel.Cooperative)
                    Lane.Run2D($"q35l_{_quantization}_coop64",
                        (long)OutputWidth * 64, rows, 64, 1, input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                else
                    Lane.Run(_type is Qwen2Gguf.Q4KType or Qwen2Gguf.Q6KType
                        ? $"qwen_linear_{_quantization}" : $"q35l_{_quantization}_reference",
                        (long)rows * OutputWidth, 0, input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
                return output;
            }
            catch { output.Dispose(); throw; }
        }
        private static bool HasFiniteF16DecodedWeightRange(byte[] payload, uint type)
        {
            int blockBytes = type == Qwen35Gguf.IQ2SType ? 82 : type == Qwen35Gguf.IQ3SType ? 110 : 144;
            for (int offset = 0; offset < payload.Length; offset += blockBytes)
            {
                float d = MathF.Abs((float)BitConverter.UInt16BitsToHalf((ushort)(payload[offset] | payload[offset + 1] << 8)));
                // Bounds cover every scale, grid entry and sign in the block.
                float bound = type == Qwen35Gguf.IQ2SType ? d * 166.625f : d * 1024f;
                if (type == Qwen2Gguf.Q4KType)
                {
                    float minimum = MathF.Abs((float)BitConverter.UInt16BitsToHalf((ushort)(payload[offset + 2] | payload[offset + 3] << 8)));
                    bound = d * 945f + minimum * 63f;
                }
                if (!float.IsFinite(bound) || bound > 65504f) return false;
            }
            return true;
        }
        internal ArcBuffer ForwardFusedLora(ArcBuffer input, ArcBuffer zeroBias, ArcBuffer z, Qwen35LoraMatrix adapter, int rows = 1)
        {
            ArcBuffer output = Lane.Allocate(checked(rows * OutputWidth));
            try
            {
                if (_residentGrid is not null)
                {
                    // Resident pairing retains bitwise original base and LoRA sums.
                    int residentGroup = PairedProjection ? 32 : Lane.Options.Qwen35ProjectionWorkgroupSize;
                    long residentGlobal = PairedProjection
                        ? ((long)rows * ((OutputWidth + 1) / 2) + 1) / 2 * 32
                        : ((long)rows * OutputWidth + residentGroup / 16 - 1) / (residentGroup / 16) * residentGroup;
                    Lane.Run(PairedProjection ? "q35l_iq2_s_grid3_sg16_pair_lora" : "q35l_iq2_s_grid3_sg16_fast_lora", residentGlobal,
                        residentGroup, input, _residentGrid, _residentScales!, _residentBlockD!, zeroBias, output,
                        rows, _inputWidth, OutputWidth, z, adapter.B, adapter.Rank, adapter.Scale);
                    return output;
                }
                int group = PairedLoraProjection ? 32 : Lane.Options.Qwen35ProjectionWorkgroupSize;
                long work = PairedLoraProjection ? ((long)rows * ((OutputWidth + 1) / 2) + 1) / 2 * 32
                    : ((long)rows * OutputWidth + group / 16 - 1) / (group / 16) * group;
                Lane.Run(_fusedProjectionKernel, work,
                    group, input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth,
                    z, adapter.B, adapter.Rank, adapter.Scale);
                return output;
            }
            catch { output.Dispose(); throw; }
        }
        internal void BackwardInput(ArcBuffer dy, ArcBuffer dx, int rows, long scratchBudgetBytes)
        {
            if (_iq2TiledBackward && rows >= 128 && OutputWidth % 16 == 0 && scratchBudgetBytes >= 128L * 1024 * 1024
                && Lane.AllocatedBytes + Math.Min(scratchBudgetBytes, Lane.Iq2TiledBackwardWorkspaceBytes(rows, _inputWidth, OutputWidth))
                    + 16L * 1024 * 1024 < Lane.EffectivePhysicalBufferBudgetBytes)
            {
                Lane.Iq2TiledBackward(dy, _encoded, dx, rows, _inputWidth, OutputWidth, workspaceBudgetBytes: scratchBudgetBytes);
                return;
            }
            const int tile = 1024;
            int splits = (OutputWidth + tile - 1) / tile;
            if (scratchBudgetBytes < sizeof(float)) throw new ArgumentOutOfRangeException(nameof(scratchBudgetBytes));
            long bytesPerRow = checked((long)splits * _inputWidth * sizeof(float));
            if (_trainingIq2RollingTranspose && IsIq2S && _transposeOctetRows == 8
                && splits > 1 && checked((long)rows * bytesPerRow) > scratchBudgetBytes
                && scratchBudgetBytes >= (long)_inputWidth * sizeof(float))
            {
                int rollingChunkRows = AlignedTransposeChunkRows(rows,
                    checked((long)_inputWidth * sizeof(float)), scratchBudgetBytes, 8);
                for (int first = 0; first < rows; first += rollingChunkRows)
                {
                    int count = Math.Min(rollingChunkRows, rows - first);
                    int elements = checked(count * _inputWidth);
                    using ArcBuffer rolling = Lane.Allocate(elements);
                    Lane.Run("q35a_zero", elements, 0, rolling, elements);
                    for (int split = 0; split < splits; split++)
                        Lane.Run("q35t_xpose_iq2_s_vec8_rows8_rolling",
                            ((long)count + 7) / 8 * (_inputWidth / 8), 32,
                            dy, _encoded, rolling, count, first, _inputWidth,
                            OutputWidth, split, tile);
                    Lane.Run("q35t_add_offset", elements, 0, rolling, dx,
                        elements, checked(first * _inputWidth));
                }
                return;
            }
            if (checked((long)rows * bytesPerRow) <= scratchBudgetBytes)
            {
                // Keep the original fast path and its accumulation order.
                using ArcBuffer fullPartial = Lane.Allocate(checked(rows * splits * _inputWidth));
                RunTranspose(dy, fullPartial, rows);
                Lane.Run("q35t_xpose_reduce", (long)rows * _inputWidth, 0,
                    fullPartial, dx, rows, _inputWidth, splits);
                return;
            }
            int rowTile = _transposeOctetRows != 0 ? _transposeOctetRows : _transposeRows;
            int chunkRows = AlignedTransposeChunkRows(rows, bytesPerRow, scratchBudgetBytes, rowTile);
            // Row tiles leave every token's quantized transpose and split
            // reduction unchanged, while bounding the vocabulary-head scratch.
            for (int row = 0; row < rows; row += chunkRows)
            {
                int count = Math.Min(chunkRows, rows - row);
                int dyElements = checked(count * OutputWidth);
                int dxElements = checked(count * _inputWidth);
                using ArcBuffer dyTile = Lane.Allocate(dyElements);
                Lane.CopyBytes(dy, dyTile, checked(row * OutputWidth * sizeof(float)), 0,
                    checked(dyElements * sizeof(float)));
                using ArcBuffer partial = Lane.Allocate(checked(count * splits * _inputWidth));
                using ArcBuffer dxTile = Lane.Allocate(dxElements);
                Lane.Run("q35a_zero", dxElements, 0, dxTile, dxElements);
                RunTranspose(dyTile, partial, count);
                Lane.Run("q35t_xpose_reduce", (long)count * _inputWidth, 0,
                    partial, dxTile, count, _inputWidth, splits);
                Lane.Run("q35t_add_offset", dxElements, 0, dxTile, dx, dxElements,
                    checked(row * _inputWidth));
            }

            void RunTranspose(ArcBuffer source, ArcBuffer partial, int count)
            {
                if (_transposeOctetRows != 0)
                    Lane.Run("q35t_xpose_" + _quantization + "_vec8_rows" + _transposeOctetRows,
                        ((long)count + _transposeOctetRows - 1) / _transposeOctetRows * splits * (_inputWidth / 8), 32,
                        source, _encoded, partial, count, _inputWidth, OutputWidth, splits, tile);
                else
                {
                    string suffix = _transposeRows == 1 ? "" : "_rows" + _transposeRows;
                    Lane.Run("q35t_xpose_" + _quantization + suffix,
                        ((long)count + _transposeRows - 1) / _transposeRows * splits * _inputWidth, 128,
                        source, _encoded, partial, count, _inputWidth, OutputWidth, splits, tile);
                }
            }
        }
        internal ArcBuffer Embedding(int tokenId)
        {
            ArcBuffer output = Lane.Allocate(_inputWidth);
            try
            {
                using ArcBuffer id = Lane.UploadRaw(new[] { tokenId });
                Lane.Run(_type is Qwen2Gguf.Q4KType or Qwen2Gguf.Q6KType
                    ? $"qwen_embedding_{_quantization}" : $"q35l_{_quantization}_embedding",
                    _inputWidth, 0, _encoded, id, output, _inputWidth);
                return output;
            }
            catch { output.Dispose(); throw; }
        }
        internal ArcBuffer Embedding(IReadOnlyList<int> tokens, int count)
        {
            ArcBuffer output = Lane.Allocate(checked(count * _inputWidth));
            try
            {
                using ArcBuffer ids = Lane.UploadRaw(tokens.Take(count).ToArray());
                Lane.Run("q35t_embedding_" + _quantization, (long)count * _inputWidth, 128,
                    _encoded, ids, output, count, _inputWidth);
                return output;
            }
            catch { output.Dispose(); throw; }
        }
        public void Dispose()
        {
            _encoded?.Dispose();
            _residentGrid?.Dispose(); _residentScales?.Dispose(); _residentBlockD?.Dispose();
        }
    }
}
