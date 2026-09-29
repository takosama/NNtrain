using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

public sealed partial class Qwen35QuantizedModel
{
    private bool CanPrefillChunk(int remaining)
        => remaining > 1 && _options.InferencePrefillChunkTokens > 1
            && !_options.LoraTraining && _prism is null;

    /// <summary>
    /// Processes a prompt chunk one layer at a time. Projections, norms, and
    /// feedforward work span all rows; cached attention and recurrent state
    /// still advance in token order with the existing single-token kernels.
    /// </summary>
    private void ForwardPromptChunkDevice(IReadOnlyList<int> prompt, int start, int count)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_faulted) throw new InvalidOperationException("Qwen3.5 GPU state is invalid after a failed step; call Reset before continuing.");
        if (start != _position || count < 2 || count > _options.InferencePrefillChunkTokens
            || (long)_position + count > Descriptor.ContextLength)
            throw new ArgumentOutOfRangeException(nameof(count));
        Qwen35GgufDescriptor d = Descriptor;
        foreach (LayerState state in _states) state.EnsureCapacity(_position + count);
        ArcExecutionLane lane = _embedding.Lane;
        ArcBuffer? hidden = null;
        try
        {
            int hiddenBytes = checked(d.EmbeddingLength * sizeof(float));
            hidden = lane.Allocate(checked(count * d.EmbeddingLength));
            for (int row = 0; row < count; row++)
            {
                using ArcBuffer embedding = _embedding.Embedding(prompt[start + row]);
                lane.CopyBytes(embedding, hidden, 0, checked(row * hiddenBytes), hiddenBytes);
            }
            for (int layer = 0; layer < d.LayerCount; layer++)
            {
                LayerBindings bindings = _layerBindings[layer];
                LayerState state = bindings.State;
                MoveToLane(ref hidden, ref lane, state.Lane, checked(count * d.EmbeddingLength));
                using ArcBuffer normalized = Qwen35Gpu.RmsNorm(lane, hidden!, bindings.AttentionNorm,
                    count, d.EmbeddingLength, d.RmsEpsilon);
                using ArcBuffer attention = bindings.Recurrent
                    ? RecurrentAttentionRows(normalized, bindings, count)
                    : FullAttentionRows(normalized, bindings, count, _position);
                Qwen35Gpu.AddInPlace(lane, hidden!, attention, checked(count * d.EmbeddingLength));
                using ArcBuffer postNorm = Qwen35Gpu.RmsNorm(lane, hidden!, bindings.PostAttentionNorm,
                    count, d.EmbeddingLength, d.RmsEpsilon);
                using ArcBuffer gate = ProjectRows(bindings.FfnGate, postNorm, count);
                using ArcBuffer up = ProjectRows(bindings.FfnUp, postNorm, count);
                using ArcBuffer activated = Qwen35Gpu.SiluMultiply(lane, gate, up,
                    checked(count * d.FeedForwardLength));
                using ArcBuffer down = ProjectRows(bindings.FfnDown, activated, count);
                Qwen35Gpu.AddInPlace(lane, hidden!, down, checked(count * d.EmbeddingLength));
            }
            _position += count;
        }
        catch { _faulted = true; throw; }
        finally { hidden?.Dispose(); }
    }

    private ArcBuffer ProjectRows(string name, ArcBuffer input, int rows)
        => ProjectRows(name, _matrices[name], input, rows);

    private ArcBuffer ProjectRows(Matrix matrix, ArcBuffer input, int rows)
        => ProjectRows(matrix.Name, matrix, input, rows);

    private ArcBuffer ProjectRows(string name, Matrix matrix, ArcBuffer input, int rows)
    {
        if (_loraFaulted) throw new InvalidOperationException("LoRA optimizer state is invalid; reload the last saved checkpoint in a new model.");
        if (_options.InferenceFusedLora && matrix.SupportsFusedLora
            && _lora.TryGetValue(name, out Qwen35LoraMatrix? adapter))
        {
            using ArcBuffer z = adapter.ProjectA(input, rows);
            return matrix.ForwardFusedLora(input, _zeroBias[matrix.Lane], z, adapter, rows);
        }
        ArcBuffer output = matrix.Forward(input, _zeroBias[matrix.Lane], rows);
        try { ApplyLora(name, input, output, rows); return output; }
        catch { output.Dispose(); throw; }
    }

    private ArcBuffer RecurrentAttentionRows(ArcBuffer input, LayerBindings bindings, int rows)
    {
        Qwen35GgufDescriptor d = Descriptor;
        ArcExecutionLane lane = bindings.State.Lane;
        using ArcBuffer qkv = ProjectRows(bindings.Qkv!, input, rows);
        using ArcBuffer gate = ProjectRows(bindings.RecurrentGate!, input, rows);
        using ArcBuffer alpha = ProjectRows(bindings.Alpha!, input, rows);
        using ArcBuffer beta = ProjectRows(bindings.Beta!, input, rows);
        int qkvWidth = bindings.Qkv!.OutputWidth;
        int gateWidth = bindings.RecurrentGate!.OutputWidth;
        int alphaWidth = bindings.Alpha!.OutputWidth;
        int betaWidth = bindings.Beta!.OutputWidth;
        int outputWidth = checked(d.LinearValueHeads * d.LinearHeadWidth);
        using ArcBuffer rowQkv = lane.Allocate(qkvWidth);
        using ArcBuffer rowGate = lane.Allocate(gateWidth);
        using ArcBuffer rowAlpha = lane.Allocate(alphaWidth);
        using ArcBuffer rowBeta = lane.Allocate(betaWidth);
        using ArcBuffer output = lane.Allocate(checked(rows * outputWidth));
        for (int row = 0; row < rows; row++)
        {
            CopyRow(qkv, rowQkv, row, qkvWidth);
            CopyRow(gate, rowGate, row, gateWidth);
            CopyRow(alpha, rowAlpha, row, alphaWidth);
            CopyRow(beta, rowBeta, row, betaWidth);
            using ArcBuffer delta = _options.FusedDelta
                ? Qwen35Gpu.DeltaStepFused(lane, rowQkv, rowGate, rowAlpha, rowBeta,
                    bindings.Convolution!, bindings.DtBias!, bindings.A!, bindings.RecurrentNorm!,
                    bindings.State.Convolution!, bindings.State.Recurrent!, d.LinearKeyHeads,
                    d.LinearValueHeads, d.LinearHeadWidth, d.ConvKernel, d.RmsEpsilon)
                : Qwen35Gpu.DeltaStep(lane, rowQkv, rowGate, rowAlpha, rowBeta,
                    bindings.Convolution!, bindings.DtBias!, bindings.A!, bindings.RecurrentNorm!,
                    bindings.State.Convolution!, bindings.State.Recurrent!, d.LinearKeyHeads,
                    d.LinearValueHeads, d.LinearHeadWidth, d.ConvKernel, d.RmsEpsilon);
            lane.CopyBytes(delta, output, 0, checked(row * outputWidth * sizeof(float)),
                checked(outputWidth * sizeof(float)));
        }
        return ProjectRows(bindings.AttentionOutput, output, rows);

        void CopyRow(ArcBuffer source, ArcBuffer target, int row, int width)
            => lane.CopyBytes(source, target, checked(row * width * sizeof(float)), 0,
                checked(width * sizeof(float)));
    }

    private ArcBuffer FullAttentionRows(ArcBuffer input, LayerBindings bindings, int rows, int startPosition)
    {
        Qwen35GgufDescriptor d = Descriptor;
        ArcExecutionLane lane = bindings.State.Lane;
        using ArcBuffer qAndGate = ProjectRows(bindings.Query!, input, rows);
        using ArcBuffer key = ProjectRows(bindings.Key!, input, rows);
        using ArcBuffer value = ProjectRows(bindings.Value!, input, rows);
        int queryWidth = checked(2 * d.HeadCount * d.HeadWidth);
        int kvWidth = checked(d.KvHeadCount * d.HeadWidth);
        int outputWidth = checked(d.HeadCount * d.HeadWidth);
        using ArcBuffer rowQuery = lane.Allocate(queryWidth);
        using ArcBuffer rowKey = lane.Allocate(kvWidth);
        using ArcBuffer rowValue = lane.Allocate(kvWidth);
        using ArcBuffer output = lane.Allocate(checked(rows * outputWidth));
        for (int row = 0; row < rows; row++)
        {
            CopyRow(qAndGate, rowQuery, row, queryWidth);
            CopyRow(key, rowKey, row, kvWidth);
            CopyRow(value, rowValue, row, kvWidth);
            using ArcBuffer attention = Qwen35Gpu.AttentionStep(lane, rowQuery, rowKey, rowValue,
                bindings.QueryNorm!, bindings.KeyNorm!, bindings.State.Keys!, bindings.State.Values!,
                startPosition + row, d.HeadCount, d.KvHeadCount, d.HeadWidth,
                d.RopeDimensionCount, d.RopeTheta, d.RmsEpsilon);
            lane.CopyBytes(attention, output, 0, checked(row * outputWidth * sizeof(float)),
                checked(outputWidth * sizeof(float)));
        }
        return ProjectRows(bindings.AttentionOutput, output, rows);

        void CopyRow(ArcBuffer source, ArcBuffer target, int row, int width)
            => lane.CopyBytes(source, target, checked(row * width * sizeof(float)), 0,
                checked(width * sizeof(float)));
    }
}
