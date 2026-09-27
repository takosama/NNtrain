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
        long recurrentBytes = checked(4L * (rows + 1) * Descriptor.LinearValueHeads
            * Descriptor.LinearHeadWidth * Descriptor.LinearHeadWidth);
        long logitsBytes = checked(4L * rows * Descriptor.VocabularySize);
        if (recurrentBytes > int.MaxValue || logitsBytes > int.MaxValue)
            throw new NotSupportedException("Training sequence exceeds the current GPU buffer limit.");
        foreach (var lane in _lanes)
            if ((ulong)Math.Max(recurrentBytes, logitsBytes) > lane.Device.MaximumAllocationBytes)
                throw new NotSupportedException("Training workspace exceeds the device allocation limit; shorten context.");
    }

    private double LoraLoss(IReadOnlyList<int> tokens, int start, bool backward)
    {
        Reset();
        var d = Descriptor; int rows = tokens.Count - 1, valid = tokens.Count - start;
        using var tape = new Qwen35TrainingTape();
        try
        {
            Matrix embedding = _matrices["token_embd.weight"];
            V hidden = tape.Add(embedding.Lane, embedding.Embedding(tokens, rows), rows, d.EmbeddingLength, false);
            for (int layer = 0; layer < d.LayerCount; layer++)
            {
                string p = $"blk.{layer}."; var lane = _states[layer].Lane;
                hidden = tape.Move(hidden, lane);
                V norm = tape.Norm(hidden, _dense[p + "attn_norm.weight"], d.RmsEpsilon);
                V attended;
                if (d.IsRecurrent(layer))
                {
                    V qkv = ProjectTrain(p + "attn_qkv.weight", norm);
                    V gate = ProjectTrain(p + "attn_gate.weight", norm);
                    V alpha = ProjectTrain(p + "ssm_alpha.weight", norm);
                    V beta = ProjectTrain(p + "ssm_beta.weight", norm);
                    var op = tape.Own(new Qwen35TrainingDelta(lane, qkv.Data, gate.Data, alpha.Data, beta.Data,
                        _dense[p + "ssm_conv1d.weight"], _dense[p + "ssm_dt.bias"], _dense[p + "ssm_a"],
                        _dense[p + "ssm_norm.weight"], d, rows));
                    V delta = tape.Add(lane, op.Output, rows, d.LinearValueHeads * d.LinearHeadWidth,
                        qkv.Differentiable || gate.Differentiable || alpha.Differentiable || beta.Differentiable, ownsData: false);
                    tape.Record(() => { if (delta.Gradient is not null) op.Backward(delta.Gradient, qkv.Grad(), gate.Grad(), alpha.Grad(), beta.Grad()); });
                    attended = ProjectTrain(p + "ssm_out.weight", delta);
                }
                else
                {
                    V q = ProjectTrain(p + "attn_q.weight", norm), k = ProjectTrain(p + "attn_k.weight", norm), v = ProjectTrain(p + "attn_v.weight", norm);
                    var op = tape.Own(new Qwen35TrainingAttention(lane, q.Data, k.Data, v.Data,
                        _dense[p + "attn_q_norm.weight"], _dense[p + "attn_k_norm.weight"], d, rows));
                    V attention = tape.Add(lane, op.Output, rows, d.HeadCount * d.HeadWidth,
                        q.Differentiable || k.Differentiable || v.Differentiable, ownsData: false);
                    tape.Record(() => { if (attention.Gradient is not null) op.Backward(attention.Gradient, q.Grad(), k.Grad(), v.Grad()); });
                    attended = ProjectTrain(p + "attn_output.weight", attention);
                }
                hidden = tape.Sum(hidden, attended);
                V post = tape.Norm(hidden, _dense[p + "post_attention_norm.weight"], d.RmsEpsilon);
                V activated = tape.Silu(ProjectTrain(p + "ffn_gate.weight", post), ProjectTrain(p + "ffn_up.weight", post));
                hidden = tape.Sum(hidden, ProjectTrain(p + "ffn_down.weight", activated));
            }
            V final = tape.Norm(hidden, _dense["output_norm.weight"], d.RmsEpsilon);
            final = tape.Move(final, OutputMatrix.Lane);
            V head = _options.TrainingResponseOnlyHead ? tape.SliceRows(final, start - 1, valid) : final;
            V logits = ProjectTrain("output.weight", head);
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

        V ProjectTrain(string name, V input)
        {
            Matrix matrix = name == "output.weight" ? OutputMatrix : _matrices[name];
            _lora.TryGetValue(name, out var adapter);
            int projectionRows = input.Rows;
            V output = tape.Add(matrix.Lane, matrix.Forward(input.Data, _zeroBias[matrix.Lane], projectionRows), projectionRows, matrix.OutputWidth,
                input.Differentiable || adapter is not null);
            ArcBuffer? z = adapter is null ? null : tape.Own(adapter.Forward(input.Data, output.Data, projectionRows));
            tape.Record(() =>
            {
                if (output.Gradient is null) return;
                ArcBuffer? dx = input.Differentiable ? input.Grad() : null;
                if (dx is not null) matrix.BackwardInput(output.Gradient, dx, projectionRows);
                adapter?.Backward(input.Data, z!, output.Gradient, dx, projectionRows);
            });
            return output;
        }
    }
}
