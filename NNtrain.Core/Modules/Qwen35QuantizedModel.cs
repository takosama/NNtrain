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
            .Sum(matrix => (long)matrix.StorageBytes)).ToArray();
    public IReadOnlyList<long> ResidentAuxiliaryWeightBytes => _lanes.Select(lane => _auxiliaryBytes.GetValueOrDefault(lane)).ToArray();
    public IReadOnlyList<long> ResidentStateBytes => _lanes.Select(lane =>
        _states.Where(state => ReferenceEquals(state.Lane, lane)).Sum(state => state.StorageBytes)).ToArray();
    public IReadOnlyList<long> UploadedBytes => _lanes.Select(lane => lane.H2DBytes).ToArray();
    public IReadOnlyList<long> DownloadedBytes => _lanes.Select(lane => lane.D2HBytes).ToArray();
    public IReadOnlyList<long> LiveDeviceBytes => _lanes.Select(lane => lane.AllocatedBytes).ToArray();
    public IReadOnlyList<long> PeakDeviceBytes => _lanes.Select(lane => lane.PeakAllocatedBytes).ToArray();
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
        if (!Enum.IsDefined(options.QuantizedKernel) || options.QueuedKernelLimit is < 16 or > 4096
            || options.ProjectionWorkgroupSize is not (32 or 64 or 128)
            || options.InferencePairedProjectionTypes is < 0 or > 7
            || options.LoraReductionSize is not (16 or 128 or 256 or 512 or 1024)
            || options.TrainingTransposeRows is not (1 or 4 or 8 or 16 or 32)
            || options.TrainingTransposeOctetRows is not (0 or 4 or 8 or 16)
            || options.TrainingQ4TransposeOctetRows is not (0 or 4 or 8 or 16)
            || options.TrainingIQ3TransposeOctetRows is not (0 or 4 or 8 or 16)
            || options.TrainingNormSplits is < 1 or > 32
            || options.TrainingForwardRows is not (1 or 2 or 4 or 8 or 16)
            || options.TrainingBufferPoolMiB is < 0 or > 2048)
            throw new ArgumentException("Invalid Qwen3.5 execution options.", nameof(options));
        Qwen35GgufDescriptor d = Qwen35Gguf.Inspect(gguf);
        Qwen35PrismMetadata? prism = Qwen35PrismMetadata.Read(gguf);
        if (options.LoraTraining && (prism is not null || d.Tensors.Any(t => t.Type == Qwen2Gguf.BF16Type && IsQuantized(t))))
            throw new NotSupportedException("PQ2_0/PTQ1_0 and BF16 matrix backpropagation are not implemented; load this model for generation.");
        ArcDeviceInfo[] available = ArcDevices.Enumerate().ToArray();
        if (available.Length == 0)
            throw new NotSupportedException("Qwen3.5 quantized inference requires Intel Arc.");
        int[] selected = devices?.ToArray() ?? SelectDevices(d, available);
        if (selected.Length == 0 || selected.Length > d.LayerCount
            || selected.Distinct().Count() != selected.Length
            || selected.Any(index => index < 0 || index >= available.Length))
            throw new ArgumentException("Specify distinct, available Arc device indices.", nameof(devices));
        long[] planned = PlanPersistentBytes(d, selected.Length);
        for (int slot = 0; slot < selected.Length; slot++)
        {
            ArcDeviceInfo device = available[selected[slot]];
            if (planned[slot] + WorkspaceReserveBytes > DeviceBudget(device))
                throw new NotSupportedException(
                    $"Arc {device.Index} needs {planned[slot] / 1073741824.0:F2} GiB for weights and initial GPU state " +
                    "plus workspace. Select more GPUs with --devices 0,1.");
            foreach (GgufTensorInfo tensor in d.Tensors)
                if (DeviceSlot(tensor.Name, d.LayerCount, selected.Length) == slot
                    && (ulong)(IsQuantized(tensor) ? EncodedBytes(tensor) : DenseBytes(tensor)) > device.MaximumAllocationBytes)
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
                    Qwen35LoraReductionSize = options.LoraTraining ? 128 : options.LoraReductionSize,
                    Qwen35UnrollQ4 = !options.LoraTraining && options.UnrollQ4,
                    Qwen35NativeHalfScale = options.NativeHalfScale,
                    CollectKernelTimings = options.CollectKernelTimings || options.DetailedProfiling,
                    Qwen35CooperativeDelta = options.TrainingCooperativeDelta,
                    DetailedProfiling = options.DetailedProfiling,
                    BufferPoolBytes = options.LoraTraining ? (long)options.TrainingBufferPoolMiB * 1024 * 1024 : WorkspaceReserveBytes,
                    DeferredReleaseBytes = 0,
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
                        options.TrainingTransposeRows, options.LoraTraining ? options.TrainingForwardRows : 1,
                        tensor.Type switch {
                            Qwen35Gguf.IQ2SType => options.TrainingTransposeOctetRows,
                            Qwen2Gguf.Q4KType => options.TrainingQ4TransposeOctetRows,
                            Qwen35Gguf.IQ3SType => options.TrainingIQ3TransposeOctetRows,
                            _ => 0 });
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

    internal static long[] PlanWeightBytes(Qwen35GgufDescriptor d, int deviceCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(deviceCount);
        var bytes = new long[deviceCount];
        foreach (GgufTensorInfo tensor in d.Tensors.Where(IsQuantized))
            bytes[DeviceSlot(tensor.Name, d.LayerCount, deviceCount)] += EncodedBytes(tensor);
        return bytes;
    }

    private static long[] PlanPersistentBytes(Qwen35GgufDescriptor d, int devices)
    {
        long[] bytes = PlanWeightBytes(d, devices);
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

    private static int[] SelectDevices(Qwen35GgufDescriptor d, ArcDeviceInfo[] available)
    {
        for (int count = 1; count <= Math.Min(available.Length, d.LayerCount); count++)
        {
            long[] bytes = PlanPersistentBytes(d, count);
            if (bytes.Select((value, slot) => value + WorkspaceReserveBytes <= DeviceBudget(available[slot])).All(fits => fits))
                return Enumerable.Range(0, count).ToArray();
        }
        throw new NotSupportedException("The Qwen3.5 weights and initial GPU state exceed the available Arc VRAM budget.");
    }

    private static long DeviceBudget(ArcDeviceInfo device) => checked((long)(device.GlobalMemoryBytes / 10 * 9));

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
    private static long DenseBytes(GgufTensorInfo tensor)
        => checked(tensor.Shape.Aggregate(1L, (count, dimension) => checked(count * (long)dimension)) * 4);

    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _faulted = true;
        _cachedPromptTokens = null;
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
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_faulted) throw new InvalidOperationException("Qwen3.5 GPU state is invalid after a failed step; call Reset before continuing.");
        Qwen35GgufDescriptor d = Descriptor;
        if ((uint)tokenId >= (uint)d.VocabularySize) throw new ArgumentOutOfRangeException(nameof(tokenId));
        if (_position >= d.ContextLength) throw new InvalidOperationException("Qwen3.5 context length exceeded.");
        // Grow caches before mutating any layer's sequence state. Replacement
        // uses D2D copies and keeps old buffers alive until the copy succeeds.
        foreach (LayerState state in _states) state.EnsureCapacity(_position + 1);
        ArcExecutionLane lane = _embedding.Lane;
        ArcBuffer? hidden = null;
        try
        {
            hidden = _embedding.Embedding(tokenId);
            if (_prism?.InverseWeights.Contains("token_embd.weight") == true)
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
                    : FullAttention(normalized, bindings);
                Qwen35Gpu.AddInPlace(lane, hidden!, attention, d.EmbeddingLength);
                using ArcBuffer postNorm = Qwen35Gpu.RmsNorm(lane, hidden!, bindings.PostAttentionNorm,
                    1, d.EmbeddingLength, d.RmsEpsilon);
                using ArcBuffer gate = Project(bindings.FfnGate, postNorm);
                using ArcBuffer up = Project(bindings.FfnUp, postNorm);
                using ArcBuffer activated = Qwen35Gpu.SiluMultiply(lane, gate, up, d.FeedForwardLength);
                using ArcBuffer down = Project(bindings.FfnDown, activated);
                Qwen35Gpu.AddInPlace(lane, hidden!, down, d.EmbeddingLength);
            }
            if (!returnLogits) { _position++; return null; }
            ArcBuffer finalNorm = Qwen35Gpu.RmsNorm(lane, hidden!, _outputNorm, 1, d.EmbeddingLength, d.RmsEpsilon);
            hidden!.Dispose(); hidden = finalNorm;
            MoveToLane(ref hidden, ref lane, OutputMatrix.Lane, d.EmbeddingLength);
            ArcBuffer logits = ProjectMatrix("output.weight", OutputMatrix, hidden!);
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
            d.LinearKeyHeads, d.LinearValueHeads, d.LinearHeadWidth, d.ConvKernel, d.RmsEpsilon)
            : Qwen35Gpu.DeltaStep(state.Lane, qkv, gate, alpha, beta,
            bindings.Convolution!, bindings.DtBias!, bindings.A!,
            bindings.RecurrentNorm!, state.Convolution!, state.Recurrent!,
            d.LinearKeyHeads, d.LinearValueHeads, d.LinearHeadWidth, d.ConvKernel, d.RmsEpsilon);
        return Project(bindings.AttentionOutput, delta);
    }

    private ArcBuffer FullAttention(ArcBuffer input, LayerBindings bindings)
    {
        Qwen35GgufDescriptor d = Descriptor;
        LayerState state = bindings.State;
        using ArcBuffer qAndGate = Project(bindings.Query!, input);
        using ArcBuffer key = Project(bindings.Key!, input);
        using ArcBuffer value = Project(bindings.Value!, input);
        using ArcBuffer attention = Qwen35Gpu.AttentionStep(state.Lane, qAndGate, key, value,
            bindings.QueryNorm!, bindings.KeyNorm!, state.Keys!, state.Values!,
            _position, d.HeadCount, d.KvHeadCount, d.HeadWidth, d.RopeDimensionCount, d.RopeTheta, d.RmsEpsilon);
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
            temperature, topP, topK, random);

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
            temperature, topP, topK, random);

    private int[] GenerateTokenIdsCore(IReadOnlyList<int> prompt, int maxNewTokens,
        int? eosTokenId, Action<int>? onToken, bool reusePromptPrefix,
        float temperature, float topP, int topK, Random? random)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentOutOfRangeException.ThrowIfNegative(maxNewTokens);
        ValidateSampling(temperature, topP, topK);
        if (prompt.Count == 0 || prompt.Count > Descriptor.ContextLength)
            throw new ArgumentException("Prompt must contain 1..context-length tokens.", nameof(prompt));
        if (prompt.Any(token => (uint)token >= (uint)Descriptor.VocabularySize))
            throw new ArgumentOutOfRangeException(nameof(prompt));
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
            for (int i = reused; i < prompt.Count; i++)
                logits = ForwardTokenDevice(prompt[i], i == prompt.Count - 1);
            if (reusePromptPrefix)
                checkpointCaptured = TryCapturePromptCheckpoint();
            for (int generated = 0; generated < maxNewTokens && result.Count < Descriptor.ContextLength; generated++)
            {
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
        foreach (ArcExecutionLane lane in _lanes)
        {
            long bytes = _states.Where(state => ReferenceEquals(state.Lane, lane))
                .Sum(state => state.AdditionalPromptSnapshotBytes);
            if (lane.AllocatedBytes + bytes + WorkspaceReserveBytes > DeviceBudget(lane.Device))
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
        foreach (LayerState state in _states) state.ClearPromptState();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var adapter in _lora.Values) adapter.Dispose();
        _lora.Clear();
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
            if (Lane.AllocatedBytes + 2 * bytes + WorkspaceReserveBytes > DeviceBudget(Lane.Device))
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
        private readonly uint _type;
        private readonly string _quantization;
        internal readonly int _inputWidth;
        private readonly Qwen35QuantizedKernel _kernel;
        private readonly int _transposeRows;
        private readonly int _transposeOctetRows;
        private readonly int _forwardRows;
        private readonly string _projectionKernel, _fusedProjectionKernel;
        private readonly bool PairedProjection;
        private readonly bool PairedLoraProjection;
        internal bool SupportsFusedLora => _kernel == Qwen35QuantizedKernel.Subgroup
            && _type is not (Qwen35Gguf.PQ20Type or Qwen35Gguf.PTQ10Type or Qwen2Gguf.BF16Type);
        internal Matrix(ArcExecutionLane lane, GgufTensorInfo info, byte[] payload, Qwen35QuantizedKernel kernel, int transposeRows, int forwardRows, int transposeOctetRows)
        {
            _transposeRows = transposeRows;
            _forwardRows = forwardRows;
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
            if (_kernel == Qwen35QuantizedKernel.Subgroup && !subgroup)
                throw new NotSupportedException("The Qwen3.5 subgroup kernel requires Intel SG16 support.");
            PairedProjection = (lane.Options.Qwen35PairedProjection
                || (lane.Options.Qwen35PairedProjectionTypes & (_type switch {
                    Qwen35Gguf.IQ2SType => 1, Qwen2Gguf.Q4KType => 2, Qwen35Gguf.IQ3SType => 4, _ => 0 })) != 0)
                && _type is Qwen2Gguf.Q4KType or Qwen35Gguf.IQ2SType or Qwen35Gguf.IQ3SType;
            _projectionKernel = $"q35l_{_quantization}_sg16" + (PairedProjection ? "_pair" : "");
            PairedLoraProjection = PairedProjection && lane.Options.Qwen35PairedLoraProjection;
            _fusedProjectionKernel = $"q35l_{_quantization}_sg16" + (PairedLoraProjection ? "_pair" : "") + "_lora";
            _encoded = lane.UploadRaw(payload);
        }
        internal ArcBuffer Forward(ArcBuffer input, ArcBuffer zeroBias, int rows = 1)
        {
            if (input.ByteLength < 4L * rows * _inputWidth) throw new ArgumentException("Qwen3.5 projection width mismatch.");
            ArcBuffer output = Lane.Allocate(checked(rows * OutputWidth));
            try
            {
                if (_kernel == Qwen35QuantizedKernel.Subgroup && _forwardRows > 1 && rows > 1
                    && _type is Qwen2Gguf.Q4KType or Qwen35Gguf.IQ2SType or Qwen35Gguf.IQ3SType)
                    Lane.Run($"q35t_linear_{_quantization}_rows{_forwardRows}",
                        ((((long)rows + _forwardRows - 1) / _forwardRows * OutputWidth + 1) / 2) * 32, 32,
                        input, _encoded, zeroBias, output, rows, _inputWidth, OutputWidth);
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
        internal ArcBuffer ForwardFusedLora(ArcBuffer input, ArcBuffer zeroBias, ArcBuffer z, Qwen35LoraMatrix adapter)
        {
            ArcBuffer output = Lane.Allocate(OutputWidth);
            try
            {
                int group = PairedLoraProjection ? 32 : Lane.Options.Qwen35ProjectionWorkgroupSize;
                long work = PairedLoraProjection ? ((long)(OutputWidth + 1) / 2 + 1) / 2 * 32
                    : ((long)OutputWidth + group / 16 - 1) / (group / 16) * group;
                Lane.Run(_fusedProjectionKernel, work,
                    group, input, _encoded, zeroBias, output, 1, _inputWidth, OutputWidth,
                    z, adapter.B, adapter.Rank, adapter.Scale);
                return output;
            }
            catch { output.Dispose(); throw; }
        }
        internal void BackwardInput(ArcBuffer dy, ArcBuffer dx, int rows, long scratchBudgetBytes)
        {
            const int tile = 1024;
            int splits = (OutputWidth + tile - 1) / tile;
            if (scratchBudgetBytes < sizeof(float)) throw new ArgumentOutOfRangeException(nameof(scratchBudgetBytes));
            long bytesPerRow = checked((long)splits * _inputWidth * sizeof(float));
            if (checked((long)rows * bytesPerRow) <= scratchBudgetBytes)
            {
                // Keep the original fast path and its accumulation order.
                using ArcBuffer fullPartial = Lane.Allocate(checked(rows * splits * _inputWidth));
                RunTranspose(dy, fullPartial, rows);
                Lane.Run("q35t_xpose_reduce", (long)rows * _inputWidth, 0,
                    fullPartial, dx, rows, _inputWidth, splits);
                return;
            }
            int chunkRows = checked((int)Math.Max(1L, Math.Min(rows, scratchBudgetBytes / bytesPerRow)));
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
        public void Dispose() => _encoded.Dispose();
    }
}
