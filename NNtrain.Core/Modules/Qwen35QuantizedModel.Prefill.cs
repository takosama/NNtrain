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
    private void ForwardPromptChunkDevice(IReadOnlyList<int> prompt, int start, int count,
        CancellationToken cancellationToken)
        => ForwardPromptChunkDevice(prompt, null, start, count, cancellationToken);

    private void ForwardPromptChunkDevice(IReadOnlyList<Qwen35PromptToken> prompt,
        int start, int count, CancellationToken cancellationToken)
        => ForwardPromptChunkDevice(null, prompt, start, count, cancellationToken);

    private void ForwardPromptChunkDevice(IReadOnlyList<int>? textPrompt,
        IReadOnlyList<Qwen35PromptToken>? mixedPrompt, int start, int count,
        CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        cancellationToken.ThrowIfCancellationRequested();
        if (_faulted) throw new InvalidOperationException("Qwen3.5 GPU state is invalid after a failed step; call Reset before continuing.");
        if (start != _position || count < 2 || count > _options.InferencePrefillChunkTokens
            || (long)_position + count > Descriptor.ContextLength)
            throw new ArgumentOutOfRangeException(nameof(count));
        Qwen35GgufDescriptor d = Descriptor;
        bool shareLoraPrefill = mixedPrompt is not null;
        foreach (LayerState state in _states) state.EnsureCapacity(_position + count);
        ArcExecutionLane lane = _embedding.Lane;
        ArcBuffer? hidden = null;
        try
        {
            int hiddenBytes = checked(d.EmbeddingLength * sizeof(float));
            // Upload the entire projected image chunk once. Uploading each
            // image row separately blocks the OpenCL queue for every token.
            // Text rows retain the existing device embedding lookup.
            if (mixedPrompt is not null)
            {
                var imageRows = new float[checked(count * d.EmbeddingLength)];
                for (int row = 0; row < count; row++)
                    mixedPrompt[start + row].Embedding?.CopyTo(imageRows,
                        checked(row * d.EmbeddingLength));
                hidden = lane.Upload(imageRows);
            }
            else hidden = lane.Allocate(checked(count * d.EmbeddingLength));
            for (int row = 0; row < count; row++)
            {
                if (mixedPrompt?[start + row].Embedding is not null) continue;
                int tokenId = mixedPrompt is null ? textPrompt![start + row] : mixedPrompt[start + row].TokenId;
                using ArcBuffer embedding = _embedding.Embedding(tokenId);
                lane.CopyBytes(embedding, hidden, 0, checked(row * hiddenBytes), hiddenBytes);
            }
            for (int layer = 0; layer < d.LayerCount; layer++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                LayerBindings bindings = _layerBindings[layer];
                LayerState state = bindings.State;
                MoveToLane(ref hidden, ref lane, state.Lane, checked(count * d.EmbeddingLength));
                using ArcBuffer normalized = Qwen35Gpu.RmsNorm(lane, hidden!, bindings.AttentionNorm,
                    count, d.EmbeddingLength, d.RmsEpsilon);
                using ArcBuffer attention = bindings.Recurrent
                    ? RecurrentAttentionRows(normalized, bindings, count, shareLoraPrefill, cancellationToken)
                    : FullAttentionRows(normalized, bindings, count, _position, mixedPrompt);
                Qwen35Gpu.AddInPlace(lane, hidden!, attention, checked(count * d.EmbeddingLength));
                using ArcBuffer postNorm = Qwen35Gpu.RmsNorm(lane, hidden!, bindings.PostAttentionNorm,
                    count, d.EmbeddingLength, d.RmsEpsilon);
                using ArcBuffer gate = ProjectRows(bindings.FfnGate, postNorm, count, shareLoraPrefill);
                using ArcBuffer up = ProjectRows(bindings.FfnUp, postNorm, count, shareLoraPrefill);
                using ArcBuffer activated = Qwen35Gpu.SiluMultiply(lane, gate, up,
                    checked(count * d.FeedForwardLength));
                using ArcBuffer down = ProjectRows(bindings.FfnDown, activated, count, shareLoraPrefill);
                Qwen35Gpu.AddInPlace(lane, hidden!, down, checked(count * d.EmbeddingLength));
            }
            _position += count;
        }
        catch { _faulted = true; throw; }
        finally { hidden?.Dispose(); }
    }

    private ArcBuffer ProjectRows(string name, ArcBuffer input, int rows, bool shareLoraPrefill = false)
        => ProjectRows(name, _matrices[name], input, rows, shareLoraPrefill);

    private ArcBuffer ProjectRows(Matrix matrix, ArcBuffer input, int rows, bool shareLoraPrefill = false)
        => ProjectRows(matrix.Name, matrix, input, rows, shareLoraPrefill);

    private ArcBuffer ProjectRows(string name, Matrix matrix, ArcBuffer input, int rows, bool shareLoraPrefill)
    {
        if (_loraFaulted) throw new InvalidOperationException("LoRA optimizer state is invalid; reload the last saved checkpoint in a new model.");
        if (_options.InferenceFusedLora && matrix.SupportsFusedLora
            && !((shareLoraPrefill || _options.InferenceXmxPrefill && rows >= 16)
                && rows > 1 && matrix.SupportsSharedPrefillProjection)
            && _lora.TryGetValue(name, out Qwen35LoraMatrix? adapter))
        {
            using ArcBuffer z = adapter.ProjectA(input, rows);
            return matrix.ForwardFusedLora(input, _zeroBias[matrix.Lane], z, adapter, rows);
        }
        ArcBuffer output = matrix.Forward(input, _zeroBias[matrix.Lane], rows);
        try { ApplyLora(name, input, output, rows); return output; }
        catch { output.Dispose(); throw; }
    }

    private ArcBuffer RecurrentAttentionRows(ArcBuffer input, LayerBindings bindings, int rows,
        bool shareLoraPrefill, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Qwen35GgufDescriptor d = Descriptor;
        ArcExecutionLane lane = bindings.State.Lane;
        using ArcBuffer qkv = ProjectRows(bindings.Qkv!, input, rows, shareLoraPrefill);
        using ArcBuffer gate = ProjectRows(bindings.RecurrentGate!, input, rows, shareLoraPrefill);
        using ArcBuffer alpha = ProjectRows(bindings.Alpha!, input, rows, shareLoraPrefill);
        using ArcBuffer beta = ProjectRows(bindings.Beta!, input, rows, shareLoraPrefill);
        int qkvWidth = bindings.Qkv!.OutputWidth;
        int gateWidth = bindings.RecurrentGate!.OutputWidth;
        int alphaWidth = bindings.Alpha!.OutputWidth;
        int betaWidth = bindings.Beta!.OutputWidth;
        int outputWidth = checked(d.LinearValueHeads * d.LinearHeadWidth);
        if (_options.InferenceBatchRecurrent && _options.FusedDelta && d.LinearHeadWidth == 128)
        {
            using ArcBuffer batched = Qwen35Gpu.DeltaRowsFused(lane, qkv, gate, alpha, beta,
                bindings.Convolution!, bindings.DtBias!, bindings.A!, bindings.RecurrentNorm!,
                bindings.State.Convolution!, bindings.State.Recurrent!, d.LinearKeyHeads,
                d.LinearValueHeads, d.LinearHeadWidth, d.ConvKernel, d.RmsEpsilon, rows,
                cancellationToken: cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            return ProjectRows(bindings.AttentionOutput, batched, rows, shareLoraPrefill);
        }
        using ArcBuffer rowQkv = lane.Allocate(qkvWidth);
        using ArcBuffer rowGate = lane.Allocate(gateWidth);
        using ArcBuffer rowAlpha = lane.Allocate(alphaWidth);
        using ArcBuffer rowBeta = lane.Allocate(betaWidth);
        using ArcBuffer output = lane.Allocate(checked(rows * outputWidth));
        for (int row = 0; row < rows; row++)
        {
            cancellationToken.ThrowIfCancellationRequested();
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
        cancellationToken.ThrowIfCancellationRequested();
        return ProjectRows(bindings.AttentionOutput, output, rows, shareLoraPrefill);

        void CopyRow(ArcBuffer source, ArcBuffer target, int row, int width)
            => lane.CopyBytes(source, target, checked(row * width * sizeof(float)), 0,
                checked(width * sizeof(float)));
    }

    private ArcBuffer FullAttentionRows(ArcBuffer input, LayerBindings bindings, int rows, int startPosition,
        IReadOnlyList<Qwen35PromptToken>? mixedPrompt = null)
    {
        Qwen35GgufDescriptor d = Descriptor;
        ArcExecutionLane lane = bindings.State.Lane;
        bool shareLoraPrefill = mixedPrompt is not null;
        using ArcBuffer qAndGate = ProjectRows(bindings.Query!, input, rows, shareLoraPrefill);
        using ArcBuffer key = ProjectRows(bindings.Key!, input, rows, shareLoraPrefill);
        using ArcBuffer value = ProjectRows(bindings.Value!, input, rows, shareLoraPrefill);
        int queryWidth = checked(2 * d.HeadCount * d.HeadWidth);
        int kvWidth = checked(d.KvHeadCount * d.HeadWidth);
        int outputWidth = checked(d.HeadCount * d.HeadWidth);
        bool batchAttention = mixedPrompt is null
            ? _options.InferenceBatchTextAttention : _options.InferenceBatchMixedAttention;
        int textTileLimit = _options.InferenceTextAttentionTileRows;
        if (mixedPrompt is null && batchAttention && (textTileLimit < 0 || textTileLimit == 1))
            throw new ArgumentOutOfRangeException(nameof(Qwen35ExecutionOptions.InferenceTextAttentionTileRows),
                "Text attention tiles must be zero (automatic) or at least two rows.");
        if (batchAttention
            && (mixedPrompt is not null || textTileLimit == 0 || rows <= textTileLimit)
            && Qwen35Gpu.CanAttentionRows(lane, startPosition, rows, d.HeadCount, d.KvHeadCount, d.HeadWidth))
        {
            Qwen35Position[] positions = Enumerable.Range(startPosition, rows)
                .Select(position => mixedPrompt is null
                    ? Qwen35Position.Scalar(position) : mixedPrompt[position].Position).ToArray();
            using ArcBuffer attention = Qwen35Gpu.AttentionRows(lane, qAndGate, key, value,
                bindings.QueryNorm!, bindings.KeyNorm!, bindings.State.Keys!, bindings.State.Values!,
                startPosition, rows, d.HeadCount, d.KvHeadCount, d.HeadWidth,
                d.RopeDimensionCount, d.RopeTheta, d.RmsEpsilon, positions, d.RopeDimensionSections);
            return ProjectRows(bindings.AttentionOutput, attention, rows, shareLoraPrefill);
        }
        // The score matrix grows with the cached prefix as well as the chunk.
        // Tile only text attention when a full chunk exceeds the same bounded
        // scratch guard; Q/K/V and the output projection still span all rows.
        // Plan with all staging buffers included before any cache mutation.
        // If even a two-row tile cannot fit, retain the scalar fallback.
        int[] attentionTiles = mixedPrompt is null && batchAttention && rows > 1
            ? Qwen35Gpu.PlanAttentionRowsTiles(startPosition, rows,
                Math.Min(rows, textTileLimit > 0 ? textTileLimit : 256),
                d.HeadCount, d.KvHeadCount, d.HeadWidth, lane.Device.MaximumAllocationBytes,
                lane.EffectivePhysicalBufferBudgetBytes - lane.AllocatedBytes)
            : [];
        if (attentionTiles.Length > 0)
        {
            int tileRows = attentionTiles[0];
            using ArcBuffer batchedOutput = lane.Allocate(checked(rows * outputWidth));
            // Release Q/K/V staging before the output projection allocates its result.
            using (ArcBuffer tileQuery = lane.Allocate(checked(tileRows * queryWidth)))
            using (ArcBuffer tileKey = lane.Allocate(checked(tileRows * kvWidth)))
            using (ArcBuffer tileValue = lane.Allocate(checked(tileRows * kvWidth)))
            {
                int first = 0;
                foreach (int count in attentionTiles)
                {
                    int position = checked(startPosition + first);
                    lane.CopyBytes(qAndGate, tileQuery, checked(first * queryWidth * sizeof(float)), 0,
                        checked(count * queryWidth * sizeof(float)));
                    lane.CopyBytes(key, tileKey, checked(first * kvWidth * sizeof(float)), 0,
                        checked(count * kvWidth * sizeof(float)));
                    lane.CopyBytes(value, tileValue, checked(first * kvWidth * sizeof(float)), 0,
                        checked(count * kvWidth * sizeof(float)));
                    Qwen35Position[] positions = Enumerable.Range(position, count)
                        .Select(Qwen35Position.Scalar).ToArray();
                    using ArcBuffer attention = Qwen35Gpu.AttentionRows(lane, tileQuery, tileKey, tileValue,
                        bindings.QueryNorm!, bindings.KeyNorm!, bindings.State.Keys!, bindings.State.Values!,
                        position, count, d.HeadCount, d.KvHeadCount, d.HeadWidth,
                        d.RopeDimensionCount, d.RopeTheta, d.RmsEpsilon, positions, d.RopeDimensionSections);
                    lane.CopyBytes(attention, batchedOutput, 0, checked(first * outputWidth * sizeof(float)),
                        checked(count * outputWidth * sizeof(float)));
                    first += count;
                }
            }
            return ProjectRows(bindings.AttentionOutput, batchedOutput, rows, shareLoraPrefill);
        }
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
                d.RopeDimensionCount, d.RopeTheta, d.RmsEpsilon,
                mixedPrompt?[startPosition + row].Position, d.RopeDimensionSections);
            lane.CopyBytes(attention, output, 0, checked(row * outputWidth * sizeof(float)),
                checked(outputWidth * sizeof(float)));
        }
        return ProjectRows(bindings.AttentionOutput, output, rows, shareLoraPrefill);

        void CopyRow(ArcBuffer source, ArcBuffer target, int row, int width)
            => lane.CopyBytes(source, target, checked(row * width * sizeof(float)), 0,
                checked(width * sizeof(float)));

    }
}
