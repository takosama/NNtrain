namespace NNtrain.Audio;

public sealed partial class NemotronAsrModel
{
    internal float[][] Encode(float[][] mel, NemotronAsrStream state, CancellationToken ct)
    {
        float[][] rows = mel;
        int channels = 1, frequency = NemotronMel.Bands;
        for (int stage = 0; stage < 3; stage++)
        {
            ct.ThrowIfCancellationRequested();
            string prefix = stage == 0 ? "encoder.subsampling.conv_in" : $"encoder.subsampling.layers.{stage - 1}.depthwise_conv";
            rows = Conv2d(rows, channels, frequency, prefix, stage != 0, state, out int newFrequency);
            frequency = newFrequency;
            channels = ConvChannels;
            if (stage > 0)
            {
                string pointwise = $"encoder.subsampling.layers.{stage - 1}.pointwise_conv";
                var columns = new List<float[]>();
                foreach (float[] row in rows)
                    for (int f = 0; f < frequency; f++)
                        columns.Add(Enumerable.Range(0, channels).Select(c => row[c * frequency + f]).ToArray());
                float[][] projected = Linear(columns.ToArray(), pointwise, ct);
                for (int t = 0; t < rows.Length; t++)
                    for (int f = 0; f < frequency; f++)
                        for (int c = 0; c < channels; c++) rows[t][c * frequency + f] = projected[t * frequency + f][c];
            }
            foreach (float[] row in rows) AsrCpuMath.Relu(row);
        }
        rows = Linear(rows, "encoder.subsampling.linear", ct);
        for (int layer = 0; layer < Layers; layer++)
        {
            ct.ThrowIfCancellationRequested();
            string prefix = $"encoder.layers.{layer}";
            FeedForward(rows, prefix + ".feed_forward1", prefix + ".norm_feed_forward1", ct);
            float[][] normalized = rows.Select(row => Norm(row, prefix + ".norm_self_att")).ToArray();
            float[][] attention = Attention(normalized, prefix + ".self_attn", layer, state, ct);
            for (int t = 0; t < rows.Length; t++) AsrCpuMath.Add(rows[t], attention[t]);
            float[][] convolution = ConformerConv(rows.Select(row => Norm(row, prefix + ".norm_conv")).ToArray(), prefix + ".conv", state, ct);
            for (int t = 0; t < rows.Length; t++) AsrCpuMath.Add(rows[t], convolution[t]);
            FeedForward(rows, prefix + ".feed_forward2", prefix + ".norm_feed_forward2", ct);
            for (int t = 0; t < rows.Length; t++) rows[t] = Norm(rows[t], prefix + ".norm_out");
        }
        state.EncoderFrames += rows.Length;
        return rows;
    }

    private void FeedForward(float[][] rows, string name, string norm, CancellationToken ct)
    {
        float[][] projected = Linear(rows.Select(row => Norm(row, norm)).ToArray(), name + ".linear1", ct);
        foreach (float[] row in projected) AsrCpuMath.Silu(row);
        projected = Linear(projected, name + ".linear2", ct);
        for (int t = 0; t < rows.Length; t++)
        {
            ct.ThrowIfCancellationRequested();
            AsrCpuMath.Add(rows[t], projected[t], 0.5f);
        }
    }

    private float[][] Conv2d(float[][] input, int channels, int frequency, string name, bool depthwise,
        NemotronAsrStream state, out int newFrequency)
    {
        var weight = Weight(name + ".weight"); var bias = Bias(name);
        if (!weight.Shape.SequenceEqual(new[] { ConvChannels, depthwise ? 1 : channels, 3, 3 }))
            throw new InvalidDataException($"Subsampling shape mismatch: {name}");
        bool first = !state.ConvCache.TryGetValue(name, out float[][]? previous);
        previous ??= [new float[channels * frequency]];
        int left = first ? 2 : 1;
        newFrequency = frequency / 2 + 1;
        int outputFrequency = newFrequency;
        int frames = (input.Length + left - 3) / 2 + 1;
        var output = Enumerable.Range(0, frames).Select(_ => new float[ConvChannels * outputFrequency]).ToArray();
        for (int t = 0; t < frames; t++)
            for (int c = 0; c < ConvChannels; c++)
                for (int f = 0; f < newFrequency; f++)
                {
                    float sum = bias is null ? 0 : (float)bias.Values[c];
                    int inputChannels = depthwise ? 1 : channels;
                    for (int ic = 0; ic < inputChannels; ic++)
                        for (int kt = 0; kt < 3; kt++)
                            for (int kf = 0; kf < 3; kf++)
                            {
                                int time = t * 2 + kt - left, freq = f * 2 + kf - 2;
                                if (freq < 0 || freq >= frequency || time < -1 || time >= input.Length) continue;
                                float[] row = time < 0 ? previous[^1] : input[time];
                                int sourceChannel = depthwise ? c : ic;
                                sum += row[sourceChannel * frequency + freq] * (float)weight.Values[((c * inputChannels + ic) * 3 + kt) * 3 + kf];
                            }
                    output[t][c * newFrequency + f] = sum;
                }
        state.ConvCache[name] = [input[^1]];
        return output;
    }

    private float[][] ConformerConv(float[][] rows, string name, NemotronAsrStream state, CancellationToken ct)
    {
        float[][] projected = Linear(rows, name + ".pointwise_conv1", ct);
        var glu = new float[rows.Length][];
        for (int t = 0; t < rows.Length; t++)
        {
            glu[t] = new float[HiddenSize];
            for (int c = 0; c < HiddenSize; c++) glu[t][c] = projected[t][c] * AsrCpuMath.Sigmoid(projected[t][c + HiddenSize]);
        }
        int left = Kernel - 1;
        if (!state.ConvCache.TryGetValue(name, out float[][]? previous)) previous = Enumerable.Range(0, left).Select(_ => new float[HiddenSize]).ToArray();
        var combined = previous.Concat(glu).ToArray();
        var weight = Weight(name + ".depthwise_conv.weight"); var bias = Bias(name + ".depthwise_conv");
        if (!weight.Shape.SequenceEqual(new[] { HiddenSize, 1, Kernel })) throw new InvalidDataException("Conformer depthwise shape mismatch.");
        var output = new float[rows.Length][];
        for (int t = 0; t < rows.Length; t++)
        {
            ct.ThrowIfCancellationRequested();
            var row = new float[HiddenSize];
            for (int c = 0; c < HiddenSize; c++)
            {
                float sum = bias is null ? 0 : (float)bias.Values[c];
                for (int k = 0; k < Kernel; k++) sum += combined[t + k][c] * (float)weight.Values[c * Kernel + k];
                row[c] = sum;
            }
            row = Norm(row, name + ".norm"); AsrCpuMath.Silu(row);
            output[t] = row;
        }
        state.ConvCache[name] = combined.TakeLast(left).ToArray();
        return Linear(output, name + ".pointwise_conv2", ct);
    }

    private float[][] Attention(float[][] rows, string name, int layer, NemotronAsrStream state, CancellationToken ct)
    {
        var query = Linear(rows, name + ".q_proj", ct);
        var key = Linear(rows, name + ".k_proj", ct);
        var value = Linear(rows, name + ".v_proj", ct);
        int past = 0;
        if (state.AttentionCache.TryGetValue(layer, out var cached))
        {
            past = cached.Key.Length;
            key = cached.Key.Concat(key).ToArray(); value = cached.Value.Concat(value).ToArray();
        }
        int width = HiddenSize / Heads, length = key.Length;
        var positions = new float[2 * length - 1][];
        if (!state.RelativePositionCache.TryGetValue(layer, out var relativeCache))
            state.RelativePositionCache[layer] = relativeCache = [];
        var missingDistances = new List<int>();
        var missingVectors = new List<float[]>();
        for (int p = 0; p < positions.Length; p++)
        {
            int distance = length - 1 - p;
            if (relativeCache.TryGetValue(distance, out var cachedPosition))
            { positions[p] = cachedPosition; continue; }
            var vector = new float[HiddenSize];
            for (int d = 0; d < HiddenSize; d += 2)
            {
                float angle = distance * MathF.Pow(10000, -(float)d / HiddenSize);
                vector[d] = MathF.Sin(angle); vector[d + 1] = MathF.Cos(angle);
            }
            missingDistances.Add(distance); missingVectors.Add(vector);
        }
        // Relative sinusoidal projections depend only on distance and layer.
        // Cache them within the bounded sliding-window utterance, and batch
        // only newly encountered distances instead of recomputing every chunk.
        float[][] projectedPositions = Linear(missingVectors.ToArray(), name + ".relative_k_proj", ct);
        for (int p = 0; p < missingDistances.Count; p++)
        {
            relativeCache[missingDistances[p]] = projectedPositions[p];
            positions[length - 1 - missingDistances[p]] = projectedPositions[p];
        }
        var u = Weight(name + ".bias_u").Values; var v = Weight(name + ".bias_v").Values;
        var output = new float[rows.Length][];
        float scale = 1 / MathF.Sqrt(width);
        for (int q = 0; q < rows.Length; q++)
        {
            ct.ThrowIfCancellationRequested();
            var row = new float[HiddenSize];
            for (int head = 0; head < Heads; head++)
            {
                var scores = new float[length]; float maximum = float.NegativeInfinity;
                for (int k = 0; k < length; k++)
                {
                    long queryPosition = state.EncoderFrames + q, keyPosition = state.EncoderFrames - past + k;
                    int chunkSize = state.Lookahead + 1;
                    long chunkDifference = queryPosition / chunkSize - keyPosition / chunkSize;
                    if (chunkDifference < 0 || chunkDifference > LeftContext / chunkSize) { scores[k] = float.NegativeInfinity; continue; }
                    float score = 0; float[] position = positions[length - 1 - (past + q - k)];
                    for (int d = head * width; d < (head + 1) * width; d++)
                        score += (query[q][d] + (float)u[d]) * key[k][d] + (query[q][d] + (float)v[d]) * position[d];
                    scores[k] = score * scale; maximum = Math.Max(maximum, scores[k]);
                }
                float denominator = 0;
                for (int k = 0; k < length; k++) { scores[k] = MathF.Exp(scores[k] - maximum); denominator += scores[k]; }
                for (int k = 0; k < length; k++)
                    for (int d = head * width; d < (head + 1) * width; d++) row[d] += scores[k] / denominator * value[k][d];
            }
            output[q] = row;
        }
        state.AttentionCache[layer] = (key.TakeLast(LeftContext).ToArray(), value.TakeLast(LeftContext).ToArray());
        return Linear(output, name + ".o_proj", ct);
    }
}
