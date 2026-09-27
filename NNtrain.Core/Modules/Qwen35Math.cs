namespace NNtrain;

/// <summary>Float32 reference operations for the text path of GGUF qwen35 models.</summary>
internal static class Qwen35Math
{
    internal static float Sigmoid(float x)
    {
        if (x >= 0f) return 1f / (1f + MathF.Exp(-x));
        float e = MathF.Exp(x);
        return e / (1f + e);
    }

    internal static float Silu(float x) => x * Sigmoid(x);

    internal static float Softplus(float x) => x > 20f ? x
        : x < -20f ? MathF.Exp(x) : MathF.Log(1f + MathF.Exp(x));

    /// <summary>GGUF norm weights already include any zero-centered RMSNorm offset.</summary>
    internal static float[] RmsNorm(float[] x, float[] weights, float eps, int headWidth = 0)
    {
        int width = headWidth == 0 ? x.Length : headWidth;
        if (width <= 0 || x.Length % width != 0 || weights.Length != width)
            throw new ArgumentException("RMSNorm input and weight widths do not match.");
        if (!float.IsFinite(eps) || eps < 0f)
            throw new ArgumentOutOfRangeException(nameof(eps));
        float[] result = new float[x.Length];
        for (int offset = 0; offset < x.Length; offset += width)
        {
            float sum = 0f;
            for (int i = 0; i < width; ++i) sum += x[offset + i] * x[offset + i];
            float inverse = 1f / MathF.Sqrt(sum / width + eps);
            for (int i = 0; i < width; ++i) result[offset + i] = x[offset + i] * inverse * weights[i];
        }
        return result;
    }

    /// <summary>For text, all MRoPE position axes coincide; only the first ropeDims rotate.</summary>
    internal static void Rope(float[] x, int heads, int headWidth, int ropeDims, int position, float theta)
    {
        if (heads <= 0 || headWidth <= 0 || x.Length != checked(heads * headWidth))
            throw new ArgumentException("RoPE input dimensions do not match.");
        if (ropeDims < 0 || ropeDims > headWidth || (ropeDims & 1) != 0)
            throw new ArgumentOutOfRangeException(nameof(ropeDims));
        if (position < 0) throw new ArgumentOutOfRangeException(nameof(position));
        if (!float.IsFinite(theta) || theta <= 0f) throw new ArgumentOutOfRangeException(nameof(theta));
        int half = ropeDims / 2;
        for (int h = 0; h < heads; ++h)
        for (int i = 0; i < half; ++i)
        {
            float angle = position * MathF.Pow(theta, -2f * i / ropeDims);
            float cos = MathF.Cos(angle), sin = MathF.Sin(angle);
            int first = h * headWidth + i, second = first + half;
            float a = x[first], b = x[second];
            x[first] = a * cos - b * sin;
            x[second] = a * sin + b * cos;
        }
    }

    /// <summary>
    /// One causal Gated DeltaNet step. Convolution storage is [channel, oldest..newest],
    /// and recurrent storage is [value head, key component, value component].
    /// GGUF stores tiled value heads, so value head h uses key head h % keyHeads.
    /// </summary>
    internal static float[] DeltaStep(
        float[] qkv, float[] gate, float[] alpha, float[] beta,
        float[] convWeights, float[] dt, float[] a, float[] norm,
        float[] convState, float[] recurrentState,
        int keyHeads, int valueHeads, int headWidth, int convKernel, float eps)
    {
        if (keyHeads <= 0 || valueHeads <= 0 || valueHeads % keyHeads != 0 || headWidth <= 0 || convKernel <= 0)
            throw new ArgumentException("Invalid Gated DeltaNet dimensions.");
        if (!float.IsFinite(eps) || eps <= 0f) throw new ArgumentOutOfRangeException(nameof(eps));
        int keySize = checked(keyHeads * headWidth);
        int valueSize = checked(valueHeads * headWidth);
        int channels = checked(2 * keySize + valueSize);
        int history = convKernel - 1;
        if (qkv.Length != channels || gate.Length != valueSize || alpha.Length != valueHeads
            || beta.Length != valueHeads || dt.Length != valueHeads || a.Length != valueHeads
            || norm.Length != headWidth || convWeights.Length != checked(channels * convKernel)
            || convState.Length != checked(channels * history)
            || recurrentState.Length != checked(valueSize * headWidth))
            throw new ArgumentException("Gated DeltaNet buffers do not match the declared dimensions.");

        float[] mixed = new float[channels];
        for (int c = 0; c < channels; ++c)
        {
            int stateOffset = c * history, weightOffset = c * convKernel;
            float sum = 0f;
            for (int tap = 0; tap < history; ++tap)
                sum += convState[stateOffset + tap] * convWeights[weightOffset + tap];
            sum += qkv[c] * convWeights[weightOffset + history];
            mixed[c] = Silu(sum);
            for (int tap = 0; tap < history - 1; ++tap)
                convState[stateOffset + tap] = convState[stateOffset + tap + 1];
            if (history > 0) convState[stateOffset + history - 1] = qkv[c];
        }

        // Q/K use L2 normalization (sum of squares), unlike the later RMSNorm.
        float queryScale = 1f / MathF.Sqrt(headWidth);
        for (int h = 0; h < keyHeads; ++h)
        {
            int qOffset = h * headWidth, kOffset = keySize + qOffset;
            float qSum = 0f, kSum = 0f;
            for (int i = 0; i < headWidth; ++i)
            {
                qSum += mixed[qOffset + i] * mixed[qOffset + i];
                kSum += mixed[kOffset + i] * mixed[kOffset + i];
            }
            float qInverse = queryScale / MathF.Sqrt(qSum + eps), kInverse = 1f / MathF.Sqrt(kSum + eps);
            for (int i = 0; i < headWidth; ++i)
            {
                mixed[qOffset + i] *= qInverse;
                mixed[kOffset + i] *= kInverse;
            }
        }

        float[] output = new float[valueSize];
        for (int h = 0; h < valueHeads; ++h)
        {
            int qOffset = (h % keyHeads) * headWidth, kOffset = keySize + qOffset;
            int valueOffset = h * headWidth, stateOffset = valueOffset * headWidth;
            // The GGUF converter has already replaced A_log with -exp(A_log).
            float decay = MathF.Exp(a[h] * Softplus(alpha[h] + dt[h]));
            float updateRate = Sigmoid(beta[h]);
            for (int i = 0; i < headWidth * headWidth; ++i) recurrentState[stateOffset + i] *= decay;
            for (int v = 0; v < headWidth; ++v)
            {
                float predicted = 0f;
                for (int k = 0; k < headWidth; ++k)
                    predicted += mixed[kOffset + k] * recurrentState[stateOffset + k * headWidth + v];
                float delta = (mixed[2 * keySize + valueOffset + v] - predicted) * updateRate;
                float value = 0f;
                for (int k = 0; k < headWidth; ++k)
                {
                    int index = stateOffset + k * headWidth + v;
                    recurrentState[index] += mixed[kOffset + k] * delta;
                    value += mixed[qOffset + k] * recurrentState[index];
                }
                output[valueOffset + v] = value;
            }
        }

        float[] normalized = RmsNorm(output, norm, eps, headWidth);
        for (int i = 0; i < normalized.Length; ++i) normalized[i] *= Silu(gate[i]);
        return normalized;
    }
}
