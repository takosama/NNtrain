using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;
using V = NNtrain.Qwen35TrainingTape.Value;

namespace NNtrain;

public sealed partial class Qwen35QuantizedModel
{
    private string _modelPath = "";
    private FileStream? _modelSource;
    private string? _modelFingerprint;
    private bool _loraFaulted;
    // Internal parity-test hook; the production threshold stays at 64 rows.
    internal int LoraCheckpointThresholdRows { get; set; } = 64;
    internal long LoraTransposeScratchBudgetBytes { get; set; } = 256L * 1024 * 1024;
    internal (int Captured, int Reused, long PeakBytes) LastIq2ProjectionCacheStats { get; private set; }
    internal (int Captured, int Reused, long PeakBytes) LastIq2GpuProjectionCacheStats { get; private set; }
    internal IReadOnlyList<long> LastIq2GpuProjectionCachePeakBytesByDevice { get; private set; } = [];
    internal IReadOnlyList<long> LastIq2GpuProjectionCacheBudgetBytesByDevice { get; private set; } = [];
    // Lets tiny GGUF tests exercise a full cache without a million-byte activation.
    internal long? LoraIq2ProjectionCacheBudgetBytesOverride { get; set; }
    private readonly Dictionary<string, Qwen35LoraMatrix> _lora = new(StringComparer.Ordinal);
    private Qwen35LoraOptions? _loraOptions;
    public int LoraStep { get; private set; }
    public long LoraParameterCount => _lora.Values.Sum(adapter => adapter.ParameterCount);
    public IReadOnlyList<string> LoraTargets => _lora.Keys.ToArray();
    internal IReadOnlyDictionary<string, Qwen35LoraMatrix> LoraMatrices => _lora;

    public void AttachLora(Qwen35LoraOptions options)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(options); options.Validate();
        if (_prism is not null)
            throw new NotSupportedException("LoRA adapters for Prism Hadamard-folded models are not supported yet.");
        if (_lora.Count != 0) throw new InvalidOperationException("A LoRA adapter is already attached.");
        if (options.Layers?.Any(layer => layer >= Descriptor.LayerCount) == true)
            throw new ArgumentException("A selected LoRA layer is outside this model.");
        options = options with { Layers = options.Layers?.ToArray(), Targets = options.Targets.ToArray() };
        var selected = new List<(string Name, Matrix Matrix)>();
        foreach (var pair in _matrices.OrderBy(pair => pair.Key, StringComparer.Ordinal))
        {
            if (!pair.Key.StartsWith("blk.", StringComparison.Ordinal)) continue;
            string[] parts = pair.Key.Split('.'); int layer = int.Parse(parts[1]);
            if ((options.Layers is null || options.Layers.Contains(layer)) && options.Targets.Contains(parts[2]))
                selected.Add((pair.Key, pair.Value));
        }
        if (options.IncludeOutput) selected.Add(("output.weight", OutputMatrix));
        if (selected.Count == 0) throw new ArgumentException("No LoRA targets match this model.");
        foreach (var group in selected.GroupBy(item => item.Matrix.Lane))
        {
            long parameters = group.Sum(item => (long)options.Rank * (item.Matrix._inputWidth + item.Matrix.OutputWidth));
            long bytes = parameters * (_options.LoraTraining ? 16L : 4L);
            if (group.Key.AllocatedBytes + bytes + WorkspaceReserveBytes > DeviceBudget(group.Key.Device))
                throw new NotSupportedException("LoRA parameters and optimizer exceed GPU memory; reduce rank or select more devices.");
        }
        var created = new Dictionary<string, Qwen35LoraMatrix>();
        try
        {
            var random = new Random(options.Seed);
            foreach (var (name, matrix) in selected)
                created.Add(name, new Qwen35LoraMatrix(matrix.Lane, matrix._inputWidth, matrix.OutputWidth, options, random));
            foreach (var pair in created) _lora.Add(pair.Key, pair.Value);
            _loraOptions = options; LoraStep = 0; Reset();
        }
        catch { foreach (var adapter in created.Values) adapter.Dispose(); _lora.Clear(); _loraOptions = null; throw; }
    }

    private void ApplyLora(string name, ArcBuffer input, ArcBuffer output, int rows)
    {
        if (_loraFaulted) throw new InvalidOperationException("LoRA optimizer state is invalid; reload the last saved checkpoint in a new model.");
        if (_lora.TryGetValue(name, out var adapter)) { using ArcBuffer z = adapter.Forward(input, output, rows); }
    }

    /// <summary>Full-sequence, response-masked CE. Base matrices remain encoded and frozen.</summary>
    public Qwen35LoraStepResult TrainLora(IReadOnlyList<int> tokens, int responseStartIndex)
    {
        if (LoraStep == int.MaxValue) throw new InvalidOperationException("LoRA step counter overflow.");
        var result = ComputeLoraGradients(tokens, responseStartIndex);
        float clip = (float)Math.Min(1.0, _loraOptions!.GradientClip / Math.Max(result.GradientNorm, 1e-30));
        try
        {
            foreach (var adapter in _lora.Values) adapter.Update(_loraOptions, LoraStep + 1, clip);
            foreach (var lane in _lanes) lane.Synchronize();
            // LoraLoss reset inference state before its independent sequence
            // tape. Training never mutates those caches, so they remain empty.
            LoraStep++;
            return result with { Step = LoraStep };
        }
        catch { _faulted = true; _loraFaulted = true; throw; }
    }

    internal Qwen35LoraStepResult ComputeLoraGradients(IReadOnlyList<int> tokens, int responseStartIndex)
    {
        ValidateLoraSequence(tokens, responseStartIndex);
        foreach (var adapter in _lora.Values) adapter.ZeroGrad();
        double loss = LoraLoss(tokens, responseStartIndex, backward: true);
        double squared = _options.TrainingBatchGradientNorm ? BatchedGradientSquaredNorm()
            : _lora.Values.Sum(adapter => adapter.GradientSquaredNorm());
        if (!double.IsFinite(loss) || !double.IsFinite(squared) || squared < 0)
            throw new ArithmeticException("Non-finite LoRA loss/gradient; optimizer update was not committed.");
        return new(LoraStep, loss, Math.Sqrt(squared), tokens.Count - responseStartIndex);
    }

    private double BatchedGradientSquaredNorm()
    {
        var sums = new Dictionary<Qwen35LoraMatrix, double>();
        int splits = _options.TrainingNormSplits;
        var batches = new List<(ArcExecutionLane Lane, Qwen35LoraMatrix[] Adapters, ArcBuffer Results)>();
        try
        {
            // Queue every device's independent reductions before a blocking
            // readback. Summation below retains the previous adapter order.
            foreach (var group in _lora.Values.GroupBy(adapter => adapter.Lane))
            {
                var adapters = group.ToArray();
                ArcBuffer buffer = group.Key.Allocate(checked(adapters.Length * 2 * splits));
                batches.Add((group.Key, adapters, buffer));
                for (int i = 0; i < adapters.Length; i++)
                    adapters[i].EnqueueGradientSquaredNorm(buffer, i * 2 * splits, splits);
            }
            foreach (var (lane, adapters, buffer) in batches)
            {
                var values = new float[adapters.Length * 2 * splits];
                lane.Read(buffer, values);
                for (int i = 0; i < adapters.Length; i++)
                {
                    double sum = 0;
                    for (int j = 0; j < 2 * splits; j++) sum += values[i * 2 * splits + j];
                    sums[adapters[i]] = sum;
                }
            }
            // Splits=1 also preserves the original GPU norm reduction.
            return _lora.Values.Sum(adapter => sums[adapter]);
        }
        finally { foreach (var batch in batches) batch.Results.Dispose(); }
    }

    public double EvaluateLoraLoss(IReadOnlyList<int> tokens, int responseStartIndex)
    {
        ValidateLoraSequence(tokens, responseStartIndex);
        return LoraLoss(tokens, responseStartIndex, backward: false);
    }

    private void ValidateLoraSequence(IReadOnlyList<int> tokens, int start)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_loraFaulted) throw new InvalidOperationException("LoRA optimizer state is invalid; reload the last saved checkpoint in a new model.");
        if (!_options.LoraTraining || _loraOptions is null)
            throw new InvalidOperationException("Load with LoraTraining=true and attach/load an adapter before training.");
        ArgumentNullException.ThrowIfNull(tokens);
        if (tokens.Count < 2 || tokens.Count - 1 > Descriptor.ContextLength || start < 1 || start >= tokens.Count)
            throw new ArgumentException("Training requires 2..context+1 tokens and a nonempty response after the prompt.");
        if (tokens.Any(token => (uint)token >= (uint)Descriptor.VocabularySize))
            throw new ArgumentOutOfRangeException(nameof(tokens));
        // Per-layer recurrent tapes are recomputed only during that layer's
        // backward pass. Reject predictable oversized workspaces before forward.
        int rows = tokens.Count - 1;
        long recurrentElements = checked((long)(rows + 1) * Descriptor.LinearValueHeads
            * Descriptor.LinearHeadWidth * Descriptor.LinearHeadWidth);
        long logitsElements = checked((long)rows * Descriptor.VocabularySize);
        if (recurrentElements > int.MaxValue || logitsElements > int.MaxValue)
            throw new NotSupportedException("Training sequence exceeds the current GPU buffer limit.");
        long recurrentBytes = checked(recurrentElements * sizeof(float));
        long logitsBytes = checked(logitsElements * sizeof(float));
        // Tiled head backprop uses ArcExecutionLane.CopyBytes with Int32 byte offsets.
        if (logitsBytes > int.MaxValue)
            throw new NotSupportedException("Training logits exceed the current GPU copy-offset limit.");
        foreach (var lane in _lanes)
            if ((ulong)Math.Max(recurrentBytes, logitsBytes) > lane.Device.MaximumAllocationBytes)
                throw new NotSupportedException("Training workspace exceeds the device allocation limit; shorten context.");
    }

    private double LoraLoss(IReadOnlyList<int> tokens, int start, bool backward)
    {
        Reset();
        LastIq2ProjectionCacheStats = default;
        LastIq2GpuProjectionCacheStats = default;
        LastIq2GpuProjectionCachePeakBytesByDevice = [];
        LastIq2GpuProjectionCacheBudgetBytesByDevice = [];
        var d = Descriptor; int rows = tokens.Count - 1, valid = tokens.Count - start;
        if (rows > LoraCheckpointThresholdRows)
            return CanKeepLoraCheckpointsOnGpu(rows)
                ? LoraLossGpuCheckpointed(tokens, start, backward)
                : LoraLossCheckpointed(tokens, start, backward);
        using var tape = new Qwen35TrainingTape();
        try
        {
            Matrix embedding = _matrices["token_embd.weight"];
            V hidden = tape.Add(embedding.Lane, embedding.Embedding(tokens, rows), rows, d.EmbeddingLength, false);
            for (int layer = 0; layer < d.LayerCount; layer++)
                hidden = ForwardLoraLayer(tape, layer, hidden, rows);
            V final = tape.Norm(hidden, _dense["output_norm.weight"], d.RmsEpsilon);
            final = tape.Move(final, OutputMatrix.Lane);
            V head = _options.TrainingResponseOnlyHead ? tape.SliceRows(final, start - 1, valid) : final;
            V logits = ProjectTrain(tape, "output.weight", head);
            int[] targets = _options.TrainingResponseOnlyHead ? tokens.Skip(start).ToArray()
                : Enumerable.Range(0, rows).Select(t => t + 1 >= start ? tokens[t + 1] : -1).ToArray();
            int logitRows = logits.Rows;
            using ArcBuffer labels = logits.Lane.UploadRaw(targets), stats = logits.Lane.Allocate(checked(logitRows * 3));
            logits.Lane.Run("q35t_ce_stats", (long)logitRows * 128, 128, logits.Data, labels, stats, d.VocabularySize, valid);
            float[] numbers = new float[logitRows * 3]; logits.Lane.Read(stats, numbers);
            double loss = Enumerable.Range(0, logitRows).Sum(t => (double)numbers[t * 3]);
            if (!double.IsFinite(loss)) throw new ArithmeticException("Non-finite LoRA loss; optimizer update was not committed.");
            if (backward)
            {
                logits.Lane.Run("q35t_ce_grad", (long)logitRows * d.VocabularySize, 0, logits.Data, labels, stats, logits.Grad(), logitRows, d.VocabularySize, valid);
                tape.Backward();
            }
            return loss;
        }
        catch { _faulted = true; throw; }
    }

    // Long sequences keep only one layer's graph on the GPU at a time. Saved
    // layer inputs are immutable host checkpoints, so recomputation uses the
    // same full causal sequence and the same adapter weights before Update.
    private double LoraLossCheckpointed(IReadOnlyList<int> tokens, int start, bool backward)
    {
        Qwen35GgufDescriptor d = Descriptor;
        int rows = tokens.Count - 1, valid = tokens.Count - start;
        int hiddenElements = checked(rows * d.EmbeddingLength);
        var checkpoints = new float[d.LayerCount + 1][];
        try
        {
            Matrix embedding = _matrices["token_embd.weight"];
            using (ArcBuffer embedded = embedding.Embedding(tokens, rows))
            {
                checkpoints[0] = new float[hiddenElements];
                embedding.Lane.Read(embedded, checkpoints[0]);
            }

            for (int layer = 0; layer < d.LayerCount; layer++)
            {
                ArcExecutionLane lane = _states[layer].Lane;
                using var tape = new Qwen35TrainingTape();
                using ArcBuffer source = lane.Upload(checkpoints[layer]);
                V input = tape.Add(lane, source, rows, d.EmbeddingLength, false, ownsData: false);
                V output = ForwardLoraLayer(tape, layer, input, rows,
                    captureDeltaCheckpoints: false);
                checkpoints[layer + 1] = new float[hiddenElements];
                lane.Read(output.Data, checkpoints[layer + 1]);
                if (!backward) checkpoints[layer] = null!;
            }

            float[]? upstream = null;
            double loss;
            using (var tape = new Qwen35TrainingTape())
            {
                ArcExecutionLane lastLane = _states[d.LayerCount - 1].Lane;
                using ArcBuffer source = lastLane.Upload(checkpoints[d.LayerCount]);
                V hidden = tape.Add(lastLane, source, rows, d.EmbeddingLength, backward, ownsData: false);
                V normalized = tape.Norm(hidden, _dense["output_norm.weight"], d.RmsEpsilon);
                V final = tape.Move(normalized, OutputMatrix.Lane);
                V head = _options.TrainingResponseOnlyHead ? tape.SliceRows(final, start - 1, valid) : final;
                V logits = ProjectTrain(tape, "output.weight", head);
                int[] targets = _options.TrainingResponseOnlyHead ? tokens.Skip(start).ToArray()
                    : Enumerable.Range(0, rows).Select(t => t + 1 >= start ? tokens[t + 1] : -1).ToArray();
                int logitRows = logits.Rows;
                using ArcBuffer labels = logits.Lane.UploadRaw(targets), stats = logits.Lane.Allocate(checked(logitRows * 3));
                logits.Lane.Run("q35t_ce_stats", (long)logitRows * 128, 128,
                    logits.Data, labels, stats, d.VocabularySize, valid);
                float[] numbers = new float[checked(logitRows * 3)];
                logits.Lane.Read(stats, numbers);
                loss = Enumerable.Range(0, logitRows).Sum(t => (double)numbers[t * 3]);
                if (!double.IsFinite(loss))
                    throw new ArithmeticException("Non-finite LoRA loss; optimizer update was not committed.");
                if (backward)
                {
                    logits.Lane.Run("q35t_ce_grad", (long)logitRows * d.VocabularySize, 0,
                        logits.Data, labels, stats, logits.Grad(), logitRows, d.VocabularySize, valid);
                    tape.Backward();
                    if (hidden.Gradient is null)
                        throw new InvalidOperationException("The output head did not propagate its input gradient.");
                    upstream = new float[hiddenElements];
                    lastLane.Read(hidden.Gradient, upstream);
                }
            }

            if (backward)
            {
                for (int layer = d.LayerCount - 1; layer >= 0; layer--)
                {
                    ArcExecutionLane lane = _states[layer].Lane;
                    using var tape = new Qwen35TrainingTape();
                    using ArcBuffer source = lane.Upload(checkpoints[layer]);
                    V input = tape.Add(lane, source, rows, d.EmbeddingLength, true, ownsData: false);
                    V output = ForwardLoraLayer(tape, layer, input, rows);
                    lane.Write(output.Grad(), upstream!);
                    tape.Backward();
                    if (layer > 0)
                    {
                        if (input.Gradient is null)
                            throw new InvalidOperationException($"LoRA layer {layer} did not propagate its input gradient.");
                        upstream = new float[hiddenElements];
                        lane.Read(input.Gradient, upstream);
                    }
                    checkpoints[layer + 1] = null!;
                }
            }
            return loss;
        }
        catch { _faulted = true; throw; }
    }

    private bool CanKeepLoraCheckpointsOnGpu(int rows)
    {
        if (!_options.TrainingGpuCheckpoints) return false;
        long hiddenBytes = checked((long)rows * Descriptor.EmbeddingLength * sizeof(float));
        if (hiddenBytes > int.MaxValue) return false; // ArcExecutionLane.CopyBytes uses Int32 offsets.
        // Leave room for the active layer's recurrent tape, vocabulary head,
        // transpose scratch and reusable buffers. Cached buffers are reclaimable;
        // AllocatedBytes counts only live allocations that cannot be evicted.
        long workingReserve = LoraTrainingWorkingReserveBytes();
        foreach (ArcExecutionLane lane in _lanes)
        {
            long budget = DeviceBudget(lane.Device);
            if ((ulong)hiddenBytes > lane.Device.MaximumAllocationBytes
                || ProjectedLoraCheckpointBytes(lane, hiddenBytes) >
                    budget - lane.AllocatedBytes - workingReserve)
                return false;
        }
        return true;
    }

    private long LoraTrainingWorkingReserveBytes()
    {
        const long mebibyte = 1024L * 1024;
        return checked((long)_options.TrainingBufferPoolMiB * mebibyte
            + Math.Max(1024L * mebibyte, LoraTransposeScratchBudgetBytes));
    }

    private long ProjectedLoraCheckpointBytes(ArcExecutionLane lane, long hiddenBytes)
    {
        long count = 0;
        for (int layer = 0; layer <= Descriptor.LayerCount; layer++)
        {
            ArcExecutionLane owner = _states[Math.Min(layer, Descriptor.LayerCount - 1)].Lane;
            if (ReferenceEquals(owner, lane)) count++;
        }
        return checked(count * hiddenBytes);
    }

    // Copy within an Arc lane without a host fence. Only the boundary between
    // model-parallel devices needs a blocking host staging transfer.
    private static void CopyLoraHidden(ArcExecutionLane sourceLane, ArcBuffer source,
        ArcExecutionLane targetLane, ArcBuffer target, int elements, int bytes)
    {
        if (ReferenceEquals(sourceLane, targetLane))
        {
            sourceLane.CopyBytes(source, target, 0, 0, bytes);
            return;
        }
        var staging = new float[elements];
        sourceLane.Read(source, staging);
        targetLane.Write(target, staging);
    }

    // One IQ2_S forward performs work proportional to inputWidth * outputWidth,
    // while retaining its FP32 result costs rows * outputWidth * four bytes.
    // Prefer the widest inputs to save the most forward work per host byte.
    internal static HashSet<string> SelectIq2ProjectionCacheTargets(
        IEnumerable<(string Name, int InputWidth, int OutputWidth)> projections,
        int rows, long byteLimit)
    {
        var selected = new HashSet<string>(StringComparer.Ordinal);
        long usedBytes = 0;
        foreach (var candidate in projections
            .OrderByDescending(projection => projection.InputWidth)
            .ThenByDescending(projection => projection.OutputWidth)
            .ThenBy(projection => projection.Name, StringComparer.Ordinal))
        {
            long bytes = checked((long)rows * candidate.OutputWidth * sizeof(float));
            if (bytes > byteLimit - usedBytes) continue;
            if (selected.Add(candidate.Name)) usedBytes += bytes;
        }
        return selected;
    }

    private sealed class Iq2BaseOutputCache(long byteLimit, HashSet<string> targets)
    {
        private readonly Dictionary<string, float[]> _outputs = new(StringComparer.Ordinal);
        private long _usedBytes;
        internal int Captured { get; private set; }
        internal int Reused { get; private set; }
        internal long PeakBytes { get; private set; }

        internal bool CanCapture(string name, int elements)
            => targets.Contains(name) && !_outputs.ContainsKey(name)
                && checked((long)elements * sizeof(float)) <= byteLimit - _usedBytes;

        internal void Capture(string name, float[] values)
        {
            if (!CanCapture(name, values.Length))
                throw new InvalidOperationException("IQ2_S projection exceeds the configured host cache cap.");
            _outputs.Add(name, values);
            _usedBytes += checked((long)values.Length * sizeof(float));
            PeakBytes = Math.Max(PeakBytes, _usedBytes);
            Captured++;
        }

        internal bool TryTake(string name, out float[]? values)
        {
            if (!_outputs.Remove(name, out values)) return false;
            _usedBytes -= checked((long)values.Length * sizeof(float));
            Reused++;
            return true;
        }

        internal void ReleaseLayer(int layer)
        {
            string prefix = $"blk.{layer}.";
            foreach (string name in _outputs.Keys.Where(name => name.StartsWith(prefix, StringComparison.Ordinal)).ToArray())
            {
                float[] values = _outputs[name];
                _outputs.Remove(name);
                _usedBytes -= checked((long)values.Length * sizeof(float));
            }
        }

        internal void Clear() { _outputs.Clear(); _usedBytes = 0; }
    }

    private sealed class Iq2GpuBaseOutputCache(
        IReadOnlyDictionary<ArcExecutionLane, long> byteLimits, HashSet<string> targets) : IDisposable
    {
        private readonly Dictionary<string, (ArcExecutionLane Lane, ArcBuffer Buffer)> _outputs =
            new(StringComparer.Ordinal);
        private readonly Dictionary<ArcExecutionLane, long> _usedBytes =
            byteLimits.Keys.ToDictionary(lane => lane, _ => 0L);
        private readonly Dictionary<ArcExecutionLane, long> _peakBytes =
            byteLimits.Keys.ToDictionary(lane => lane, _ => 0L);
        internal int Captured { get; private set; }
        internal int Reused { get; private set; }
        internal long PeakBytes => _peakBytes.Values.Sum();

        internal long PeakBytesOn(ArcExecutionLane lane) => _peakBytes.GetValueOrDefault(lane);
        internal long BudgetBytesOn(ArcExecutionLane lane) => byteLimits.GetValueOrDefault(lane);

        internal bool CanCapture(string name, ArcExecutionLane lane, int elements)
            => targets.Contains(name) && !_outputs.ContainsKey(name)
                && byteLimits.TryGetValue(lane, out long limit)
                && checked((long)elements * sizeof(float)) <= limit - _usedBytes.GetValueOrDefault(lane);

        internal void Capture(string name, ArcExecutionLane lane, ArcBuffer buffer, int elements)
        {
            if (!CanCapture(name, lane, elements))
                throw new InvalidOperationException("IQ2_S projection exceeds the configured Arc cache cap.");
            _outputs.Add(name, (lane, buffer));
            long used = _usedBytes.GetValueOrDefault(lane) + checked((long)elements * sizeof(float));
            _usedBytes[lane] = used;
            _peakBytes[lane] = Math.Max(used, _peakBytes.GetValueOrDefault(lane));
            Captured++;
        }

        internal bool TryTake(string name, out ArcBuffer? buffer)
        {
            if (!_outputs.Remove(name, out var entry))
            {
                buffer = null;
                return false;
            }
            buffer = entry.Buffer;
            _usedBytes[entry.Lane] -= buffer.ByteLength;
            Reused++;
            return true;
        }

        internal void ReleaseLayer(int layer)
        {
            string prefix = $"blk.{layer}.";
            foreach (string name in _outputs.Keys.Where(name => name.StartsWith(prefix, StringComparison.Ordinal)).ToArray())
            {
                var entry = _outputs[name];
                _outputs.Remove(name);
                _usedBytes[entry.Lane] -= entry.Buffer.ByteLength;
                entry.Buffer.Dispose();
            }
        }

        public void Dispose()
        {
            foreach (var entry in _outputs.Values) entry.Buffer.Dispose();
            _outputs.Clear();
            _usedBytes.Clear();
        }
    }

    private Iq2GpuBaseOutputCache CreateIq2GpuProjectionCache(int rows, long hiddenBytes)
    {
        const long mebibyte = 1024L * 1024;
        long requestedBytes = checked((long)_options.TrainingIQ2GpuProjectionCacheMiB * mebibyte);
        long workingReserve = LoraTrainingWorkingReserveBytes();
        var limits = new Dictionary<ArcExecutionLane, long>();
        var targets = new HashSet<string>(StringComparer.Ordinal);
        foreach (ArcExecutionLane lane in _lanes)
        {
            // Reserve all layer checkpoints, the regular training workspace,
            // and one extra buffer margin below the 90% physical VRAM budget.
            long spare = DeviceBudget(lane.Device) - lane.AllocatedBytes
                - ProjectedLoraCheckpointBytes(lane, hiddenBytes)
                - workingReserve - WorkspaceReserveBytes;
            long limit = Math.Min(requestedBytes, Math.Max(0, spare));
            limits.Add(lane, limit);
            targets.UnionWith(SelectIq2ProjectionCacheTargets(_matrices.Values
                .Where(matrix => matrix.IsIq2S && ReferenceEquals(matrix.Lane, lane)
                    && matrix.Name.StartsWith("blk.", StringComparison.Ordinal))
                .Select(matrix => (matrix.Name, matrix._inputWidth, matrix.OutputWidth)),
                rows, limit));
        }
        return new Iq2GpuBaseOutputCache(limits, targets);
    }

    private double LoraLossGpuCheckpointed(IReadOnlyList<int> tokens, int start, bool backward)
    {
        Qwen35GgufDescriptor d = Descriptor;
        int rows = tokens.Count - 1, valid = tokens.Count - start;
        int hiddenElements = checked(rows * d.EmbeddingLength);
        int hiddenBytes = checked(hiddenElements * sizeof(float));
        var checkpoints = new ArcBuffer?[d.LayerCount + 1];
        long iq2ByteLimit = LoraIq2ProjectionCacheBudgetBytesOverride
            ?? checked((long)_options.TrainingIQ2ProjectionCacheMiB * 1024 * 1024);
        Iq2BaseOutputCache? iq2Cache = null;
        if (backward && _options.TrainingIQ2ProjectionCacheMiB > 0)
        {
            var projections = _matrices.Values
                .Where(matrix => matrix.IsIq2S && matrix.Name.StartsWith("blk.", StringComparison.Ordinal))
                .Select(matrix => (matrix.Name, matrix._inputWidth, matrix.OutputWidth)).ToArray();
            HashSet<string> targets = _options.TrainingIQ2ProjectionCachePrioritize
                ? SelectIq2ProjectionCacheTargets(projections, rows, iq2ByteLimit)
                : projections.Select(projection => projection.Name).ToHashSet(StringComparer.Ordinal);
            iq2Cache = new Iq2BaseOutputCache(iq2ByteLimit, targets);
        }
        Iq2GpuBaseOutputCache? iq2GpuCache = backward && _options.TrainingIQ2GpuProjectionCacheMiB > 0
            ? CreateIq2GpuProjectionCache(rows, hiddenBytes) : null;
        ArcBuffer? upstream = null;
        ArcExecutionLane? upstreamLane = null;
        try
        {
            Matrix embedding = _matrices["token_embd.weight"];
            checkpoints[0] = embedding.Embedding(tokens, rows);
            for (int layer = 0; layer < d.LayerCount; layer++)
            {
                ArcExecutionLane lane = _states[layer].Lane;
                using var tape = new Qwen35TrainingTape();
                V input = tape.Add(lane, checkpoints[layer]!, rows, d.EmbeddingLength,
                    false, ownsData: false);
                V output = ForwardLoraLayer(tape, layer, input, rows, iq2Cache, iq2GpuCache,
                    captureIq2Base: true, captureDeltaCheckpoints: false);
                ArcExecutionLane nextLane = _states[Math.Min(layer + 1, d.LayerCount - 1)].Lane;
                checkpoints[layer + 1] = nextLane.Allocate(hiddenElements);
                CopyLoraHidden(lane, output.Data, nextLane, checkpoints[layer + 1]!,
                    hiddenElements, hiddenBytes);
                if (!backward)
                {
                    checkpoints[layer]!.Dispose();
                    checkpoints[layer] = null;
                }
            }

            double loss;
            using (var tape = new Qwen35TrainingTape())
            {
                ArcExecutionLane lastLane = _states[d.LayerCount - 1].Lane;
                V hidden = tape.Add(lastLane, checkpoints[d.LayerCount]!, rows,
                    d.EmbeddingLength, backward, ownsData: false);
                V normalized = tape.Norm(hidden, _dense["output_norm.weight"], d.RmsEpsilon);
                V final = tape.Move(normalized, OutputMatrix.Lane);
                V head = _options.TrainingResponseOnlyHead ? tape.SliceRows(final, start - 1, valid) : final;
                V logits = ProjectTrain(tape, "output.weight", head);
                int[] targets = _options.TrainingResponseOnlyHead ? tokens.Skip(start).ToArray()
                    : Enumerable.Range(0, rows).Select(t => t + 1 >= start ? tokens[t + 1] : -1).ToArray();
                int logitRows = logits.Rows;
                using ArcBuffer labels = logits.Lane.UploadRaw(targets), stats = logits.Lane.Allocate(checked(logitRows * 3));
                logits.Lane.Run("q35t_ce_stats", (long)logitRows * 128, 128,
                    logits.Data, labels, stats, d.VocabularySize, valid);
                float[] numbers = new float[checked(logitRows * 3)];
                logits.Lane.Read(stats, numbers);
                loss = Enumerable.Range(0, logitRows).Sum(t => (double)numbers[t * 3]);
                if (!double.IsFinite(loss))
                    throw new ArithmeticException("Non-finite LoRA loss; optimizer update was not committed.");
                if (backward)
                {
                    logits.Lane.Run("q35t_ce_grad", (long)logitRows * d.VocabularySize, 0,
                        logits.Data, labels, stats, logits.Grad(), logitRows, d.VocabularySize, valid);
                    tape.Backward();
                    if (hidden.Gradient is null)
                        throw new InvalidOperationException("The output head did not propagate its input gradient.");
                    upstream = lastLane.Allocate(hiddenElements);
                    CopyLoraHidden(lastLane, hidden.Gradient, lastLane, upstream,
                        hiddenElements, hiddenBytes);
                    upstreamLane = lastLane;
                }
            }

            if (backward)
            {
                for (int layer = d.LayerCount - 1; layer >= 0; layer--)
                {
                    ArcExecutionLane lane = _states[layer].Lane;
                    using var tape = new Qwen35TrainingTape();
                    V input = tape.Add(lane, checkpoints[layer]!, rows,
                        d.EmbeddingLength, true, ownsData: false);
                    V output = ForwardLoraLayer(tape, layer, input, rows, iq2Cache, iq2GpuCache,
                        captureIq2Base: false);
                    CopyLoraHidden(upstreamLane!, upstream!, lane, output.Grad(),
                        hiddenElements, hiddenBytes);
                    tape.Backward();
                    iq2Cache?.ReleaseLayer(layer);
                    iq2GpuCache?.ReleaseLayer(layer);
                    ArcBuffer? nextUpstream = null;
                    if (layer > 0)
                    {
                        if (input.Gradient is null)
                            throw new InvalidOperationException($"LoRA layer {layer} did not propagate its input gradient.");
                        nextUpstream = lane.Allocate(hiddenElements);
                        CopyLoraHidden(lane, input.Gradient, lane, nextUpstream,
                            hiddenElements, hiddenBytes);
                    }
                    upstream!.Dispose();
                    upstream = nextUpstream;
                    upstreamLane = lane;
                    checkpoints[layer + 1]!.Dispose();
                    checkpoints[layer + 1] = null;
                }
            }
            return loss;
        }
        catch { _faulted = true; throw; }
        finally
        {
            upstream?.Dispose();
            foreach (ArcBuffer? checkpoint in checkpoints) checkpoint?.Dispose();
            LastIq2ProjectionCacheStats = (iq2Cache?.Captured ?? 0,
                iq2Cache?.Reused ?? 0, iq2Cache?.PeakBytes ?? 0);
            iq2Cache?.Clear();
            LastIq2GpuProjectionCacheStats = (iq2GpuCache?.Captured ?? 0,
                iq2GpuCache?.Reused ?? 0, iq2GpuCache?.PeakBytes ?? 0);
            LastIq2GpuProjectionCachePeakBytesByDevice = _lanes
                .Select(lane => iq2GpuCache?.PeakBytesOn(lane) ?? 0).ToArray();
            LastIq2GpuProjectionCacheBudgetBytesByDevice = _lanes
                .Select(lane => iq2GpuCache?.BudgetBytesOn(lane) ?? 0).ToArray();
            iq2GpuCache?.Dispose();
        }
    }

    private V ForwardLoraLayer(Qwen35TrainingTape tape, int layer, V hidden, int rows,
        Iq2BaseOutputCache? iq2Cache = null, Iq2GpuBaseOutputCache? iq2GpuCache = null,
        bool captureIq2Base = false,
        bool captureDeltaCheckpoints = true)
    {
        V Project(string name, V input) => ProjectTrain(tape, name, input,
            iq2Cache, iq2GpuCache, captureIq2Base);
        Qwen35GgufDescriptor d = Descriptor;
        string p = $"blk.{layer}.";
        ArcExecutionLane lane = _states[layer].Lane;
        hidden = tape.Move(hidden, lane);
        V norm = tape.Norm(hidden, _dense[p + "attn_norm.weight"], d.RmsEpsilon);
        V attended;
        if (d.IsRecurrent(layer))
        {
            V qkv = Project(p + "attn_qkv.weight", norm);
            V gate = Project(p + "attn_gate.weight", norm);
            V alpha = Project(p + "ssm_alpha.weight", norm);
            V beta = Project(p + "ssm_beta.weight", norm);
            var op = tape.Own(new Qwen35TrainingDelta(lane, qkv.Data, gate.Data, alpha.Data, beta.Data,
                _dense[p + "ssm_conv1d.weight"], _dense[p + "ssm_dt.bias"], _dense[p + "ssm_a"],
                _dense[p + "ssm_norm.weight"], d, rows, captureDeltaCheckpoints));
            V delta = tape.Add(lane, op.Output, rows, d.LinearValueHeads * d.LinearHeadWidth,
                qkv.Differentiable || gate.Differentiable || alpha.Differentiable || beta.Differentiable, ownsData: false);
            tape.Record(() => { if (delta.Gradient is not null) op.Backward(delta.Gradient, qkv.Grad(), gate.Grad(), alpha.Grad(), beta.Grad()); });
            attended = Project(p + "ssm_out.weight", delta);
        }
        else
        {
            V q = Project(p + "attn_q.weight", norm);
            V k = Project(p + "attn_k.weight", norm);
            V v = Project(p + "attn_v.weight", norm);
            var op = tape.Own(new Qwen35TrainingAttention(lane, q.Data, k.Data, v.Data,
                _dense[p + "attn_q_norm.weight"], _dense[p + "attn_k_norm.weight"], d, rows));
            V attention = tape.Add(lane, op.Output, rows, d.HeadCount * d.HeadWidth,
                q.Differentiable || k.Differentiable || v.Differentiable, ownsData: false);
            tape.Record(() => { if (attention.Gradient is not null) op.Backward(attention.Gradient, q.Grad(), k.Grad(), v.Grad()); });
            attended = Project(p + "attn_output.weight", attention);
        }
        hidden = tape.Sum(hidden, attended);
        V post = tape.Norm(hidden, _dense[p + "post_attention_norm.weight"], d.RmsEpsilon);
        V activated = tape.Silu(Project(p + "ffn_gate.weight", post),
            Project(p + "ffn_up.weight", post));
        return tape.Sum(hidden, Project(p + "ffn_down.weight", activated));
    }

    private V ProjectTrain(Qwen35TrainingTape tape, string name, V input,
        Iq2BaseOutputCache? iq2Cache = null, Iq2GpuBaseOutputCache? iq2GpuCache = null,
        bool captureIq2Base = false)
    {
        Matrix matrix = name == "output.weight" ? OutputMatrix : _matrices[name];
        _lora.TryGetValue(name, out var adapter);
        int projectionRows = input.Rows;
        ArcBuffer baseOutput;
        if (!captureIq2Base && iq2GpuCache is not null
            && iq2GpuCache.TryTake(name, out ArcBuffer? deviceCached))
            baseOutput = deviceCached!;
        else if (!captureIq2Base && iq2Cache is not null && iq2Cache.TryTake(name, out float[]? cached))
            baseOutput = matrix.Lane.Upload(cached!);
        else
            baseOutput = matrix.Forward(input.Data, _zeroBias[matrix.Lane], projectionRows);
        V output = tape.Add(matrix.Lane,
            baseOutput,
            projectionRows, matrix.OutputWidth, input.Differentiable || adapter is not null);
        // Retain the frozen base result before adapter.Forward adds LoRA in place.
        // Backward still runs the ordinary quantized transpose and LoRA gradients.
        if (captureIq2Base && matrix.IsIq2S)
        {
            int elements = checked(projectionRows * matrix.OutputWidth);
            if (iq2GpuCache is not null && iq2GpuCache.CanCapture(name, matrix.Lane, elements))
            {
                ArcBuffer retained = matrix.Lane.Allocate(elements);
                try
                {
                    matrix.Lane.CopyBytes(output.Data, retained, 0, 0,
                        checked(elements * sizeof(float)));
                    iq2GpuCache.Capture(name, matrix.Lane, retained, elements);
                }
                catch { retained.Dispose(); throw; }
            }
            else if (iq2Cache is not null && iq2Cache.CanCapture(name, elements))
            {
                var values = new float[elements];
                matrix.Lane.Read(output.Data, values);
                iq2Cache.Capture(name, values);
            }
        }
        ArcBuffer? z = adapter is null ? null : tape.Own(adapter.Forward(input.Data, output.Data, projectionRows));
        tape.Record(() =>
        {
            if (output.Gradient is null) return;
            ArcBuffer? dx = input.Differentiable ? input.Grad() : null;
            if (dx is not null) matrix.BackwardInput(output.Gradient, dx, projectionRows,
                LoraTransposeScratchBudgetBytes);
            adapter?.Backward(input.Data, z!, output.Gradient, dx, projectionRows);
        });
        return output;
    }
}
