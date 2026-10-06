using System.Numerics;
namespace NNtrain.Audio;
public sealed partial class ParakeetCtcModel
{
    internal Action<string, float[][]>? EncoderObserver { get; set; }
    internal float[][] Encode(float[][] mel, CancellationToken ct, int? validMelFrames = null)
    {
        long timer = System.Diagnostics.Stopwatch.GetTimestamp();
        void Mark(string name)
        {
            TimingObserver?.Invoke(name, System.Diagnostics.Stopwatch.GetElapsedTime(timer).TotalMilliseconds);
            timer = System.Diagnostics.Stopwatch.GetTimestamp();
        }
        float[][] rows = mel; int channels = 1, frequency = 80;
        int validFrames = validMelFrames ?? mel.Length - 1;
        if (validFrames < 1 || validFrames > mel.Length) throw new ArgumentOutOfRangeException(nameof(validMelFrames));
        for (int stage = 0; stage < 3; stage++)
        {
            int index = stage == 0 ? 0 : stage == 1 ? 2 : 5;
            rows = Conv2d(rows, channels, frequency, $"encoder.pre_encode.conv.{index}", stage > 0, out frequency, ct); channels = ConvChannels;
            if (stage > 0)
            {
                var columns = new float[rows.Length * frequency][];
                for (int t = 0; t < rows.Length; t++) for (int f = 0; f < frequency; f++)
                { columns[t * frequency + f] = new float[channels]; for (int c = 0; c < channels; c++) columns[t * frequency + f][c] = rows[t][c * frequency + f]; }
                float[][] projected = Linear(columns, $"encoder.pre_encode.conv.{index + 1}", ct);
                for (int t = 0; t < rows.Length; t++) for (int f = 0; f < frequency; f++) for (int c = 0; c < channels; c++) rows[t][c * frequency + f] = projected[t * frequency + f][c];
            }
            foreach (float[] row in rows) AsrCpuMath.Relu(row);
            validFrames = (validFrames + 1) / 2;
            // NeMo 3's length-aware convolution masks invalid intermediate
            // time rows before the next convolution, including nonzero biases.
            for (int t = validFrames; t < rows.Length; t++) Array.Clear(rows[t]);
        }
        // The frontend preserves one zero terminal frame for convolution geometry.
        // NeMo's valid sequence length is floor(samples / hop), not STFT output length.
        rows = rows[..validFrames];
        rows = Linear(rows, "encoder.pre_encode.out", ct);
        EncoderObserver?.Invoke("pre_encode", rows);
        float scale = MathF.Sqrt(Hidden); foreach (float[] row in rows) for (int i = 0; i < Hidden; i++) row[i] *= scale;
        Mark("encoder.subsampling");
        float[][] relativePositions = CreateRelativePositions(rows.Length);
        for (int layer = 0; layer < Layers; layer++)
        {
            ct.ThrowIfCancellationRequested(); string name = $"encoder.layers.{layer}";
            FeedForward(rows, name + ".feed_forward1", name + ".norm_feed_forward1", ct);
            Mark("encoder.feedForward1");
            float[][] attention = Attention(rows.Select(row => Norm(row, name + ".norm_self_att")).ToArray(), relativePositions, name + ".self_attn", ct);
            for (int t = 0; t < rows.Length; t++) AsrCpuMath.Add(rows[t], attention[t]);
            Mark("encoder.attention");
            float[][] convolution = ConformerConv(rows.Select(row => Norm(row, name + ".norm_conv")).ToArray(), name + ".conv", ct);
            for (int t = 0; t < rows.Length; t++) AsrCpuMath.Add(rows[t], convolution[t]);
            Mark("encoder.convolution");
            FeedForward(rows, name + ".feed_forward2", name + ".norm_feed_forward2", ct);
            for (int t = 0; t < rows.Length; t++) rows[t] = Norm(rows[t], name + ".norm_out");
            EncoderObserver?.Invoke($"layer-{layer}", rows);
            Mark("encoder.feedForward2Norm");
        }
        return rows;
    }
    private void FeedForward(float[][] rows, string name, string norm, CancellationToken ct)
    {
        float[][] normalized = rows.Select(row => Norm(row, norm)).ToArray();
        float[][] result;
        if (_arc is not null)
        { _arc.TimingObserver = TimingObserver; result = _arc.FeedForward(normalized, name, ct); }
        else
        {
            result = Linear(normalized, name + ".linear1", ct);
            foreach (float[] row in result) AsrCpuMath.Silu(row);
            result = Linear(result, name + ".linear2", ct);
        }
        for (int t = 0; t < rows.Length; t++) AsrCpuMath.Add(rows[t], result[t], .5f);
    }
    private float[][] Conv2d(float[][] rows, int channels, int frequency, string name, bool depthwise, out int outputFrequency, CancellationToken ct)
    {
        var weight = Weight(name + ".weight"); var bias = Bias(name);
        if (!weight.Shape.SequenceEqual(new[] { ConvChannels, depthwise ? 1 : channels, 3, 3 })) throw new InvalidDataException("Parakeet subsampling shape mismatch.");
        outputFrequency = (frequency + 1) / 2;
        int bins = outputFrequency;
        int frames = (rows.Length + 1) / 2;
        var result = Enumerable.Range(0, frames).Select(_ => new float[ConvChannels * bins]).ToArray();
        Parallel.For(0, frames, new ParallelOptions { CancellationToken = ct, MaxDegreeOfParallelism = Math.Min(4, Environment.ProcessorCount) }, t =>
        {
            ct.ThrowIfCancellationRequested();
            for (int c = 0; c < ConvChannels; c++) for (int f = 0; f < bins; f++)
            {
                float sum = bias is null ? 0 : (float)bias.Values[c]; int inputs = depthwise ? 1 : channels;
                for (int ic = 0; ic < inputs; ic++) for (int kt = 0; kt < 3; kt++) for (int kf = 0; kf < 3; kf++)
                {
                    int time = t * 2 + kt - 1, freq = f * 2 + kf - 1;
                    if (time < 0 || time >= rows.Length || freq < 0 || freq >= frequency) continue;
                    sum += rows[time][(depthwise ? c : ic) * frequency + freq] * (float)weight.Values[((c * inputs + ic) * 3 + kt) * 3 + kf];
                }
                result[t][c * bins + f] = sum;
            }
        });
        return result;
    }
    private float[][] ConformerConv(float[][] rows, string name, CancellationToken ct)
    {
        float[][] projected = Linear(rows, name + ".pointwise_conv1", ct); var glu = new float[rows.Length][];
        for (int t = 0; t < rows.Length; t++)
        { glu[t] = new float[Hidden]; for (int c = 0; c < Hidden; c++) glu[t][c] = projected[t][c] * AsrCpuMath.Sigmoid(projected[t][c + Hidden]); }
        var weights = Weight(name + ".depthwise_conv.weight"); var bias = Bias(name + ".depthwise_conv");
        if (!weights.Shape.SequenceEqual(new[] { Hidden, 1, Kernel })) throw new InvalidDataException("Parakeet convolution shape mismatch.");
        var mean = Weight(name + ".batch_norm.running_mean").Values; var variance = Weight(name + ".batch_norm.running_var").Values;
        var gamma = Weight(name + ".batch_norm.weight").Values; var beta = Weight(name + ".batch_norm.bias").Values;
        // Small per-operation coefficients, not a second FP32 model copy.
        float[] convolutionWeights = weights.Values.Select(x => (float)x).ToArray();
        float[] denominator = variance.Select(x => MathF.Sqrt((float)x + 1e-5f)).ToArray();
        var result = new float[rows.Length][];
        Parallel.For(0, rows.Length, new ParallelOptions { CancellationToken = ct, MaxDegreeOfParallelism = Math.Min(4, Environment.ProcessorCount) }, t =>
        {
            ct.ThrowIfCancellationRequested(); var row = new float[Hidden];
            for (int c = 0; c < Hidden; c++)
            {
                float sum = bias is null ? 0 : (float)bias.Values[c];
                for (int k = 0; k < Kernel; k++) { int pos = t + k - Kernel / 2; if (pos >= 0 && pos < rows.Length) sum += glu[pos][c] * convolutionWeights[c * Kernel + k]; }
                row[c] = (sum - (float)mean[c]) / denominator[c] * (float)gamma[c] + (float)beta[c];
            }
            AsrCpuMath.Silu(row); result[t] = row;
        });
        return Linear(result, name + ".pointwise_conv2", ct);
    }
    private float[][] CreateRelativePositions(int length)
    {
        var positions = new float[2 * length - 1][];
        double[] divisors = Enumerable.Range(0, Hidden / 2).Select(i => Math.Pow(10000, (double)(2 * i) / Hidden)).ToArray();
        for (int p = 0; p < positions.Length; p++)
        {
            positions[p] = new float[Hidden]; int distance = length - 1 - p;
            for (int d = 0; d < Hidden; d += 2)
            { double angle = distance / divisors[d / 2]; positions[p][d] = (float)Math.Sin(angle); positions[p][d + 1] = (float)Math.Cos(angle); }
        }
        return positions;
    }
    private float[][] Attention(float[][] rows, float[][] relativePositions, string name, CancellationToken ct)
    {
        if (_arc is not null)
        {
            _arc.TimingObserver = TimingObserver;
            return _arc.RelativeAttention(rows, relativePositions, name, Heads, ct);
        }
        float[][] query = Linear(rows, name + ".linear_q", ct), key = Linear(rows, name + ".linear_k", ct), value = Linear(rows, name + ".linear_v", ct);
        int length = rows.Length, width = Hidden / Heads;
        float[][] positions = Linear(relativePositions, name + ".linear_pos", ct);
        var u = Weight(name + ".pos_bias_u").Values; var v = Weight(name + ".pos_bias_v").Values;
        var output = new float[length][];
        float divisor = MathF.Sqrt(width);
        Parallel.For(0, length, new ParallelOptions { CancellationToken = ct, MaxDegreeOfParallelism = Math.Min(4, Environment.ProcessorCount) }, q =>
        {
            ct.ThrowIfCancellationRequested(); output[q] = new float[Hidden];
            var queryU = new float[Hidden]; var queryV = new float[Hidden];
            for (int i = 0; i < Hidden; i++) { queryU[i] = query[q][i] + (float)u[i]; queryV[i] = query[q][i] + (float)v[i]; }
            for (int head = 0; head < Heads; head++)
            {
                var scores = new float[length]; float maximum = float.NegativeInfinity; int start = head * width;
                for (int k = 0; k < length; k++)
                {
                    float score = 0; float[] relative = positions[length - 1 - q + k];
                    int d = 0; var accumulator = Vector<float>.Zero;
                    for (; d <= width - Vector<float>.Count; d += Vector<float>.Count)
                    {
                        int i = start + d;
                        accumulator += new Vector<float>(queryU, i) * new Vector<float>(key[k], i)
                            + new Vector<float>(queryV, i) * new Vector<float>(relative, i);
                    }
                    for (int lane = 0; lane < Vector<float>.Count; lane++) score += accumulator[lane];
                    for (; d < width; d++) { int i = start + d; score += queryU[i] * key[k][i] + queryV[i] * relative[i]; }
                    scores[k] = score / divisor; maximum = Math.Max(maximum, scores[k]);
                }
                float sum = 0; for (int k = 0; k < length; k++) { scores[k] = MathF.Exp(scores[k] - maximum); sum += scores[k]; }
                for (int k = 0; k < length; k++)
                {
                    float probability = scores[k] / sum; int d = 0;
                    var probabilityVector = new Vector<float>(probability);
                    for (; d <= width - Vector<float>.Count; d += Vector<float>.Count)
                    {
                        int i = start + d;
                        (new Vector<float>(output[q], i) + probabilityVector * new Vector<float>(value[k], i)).CopyTo(output[q], i);
                    }
                    for (; d < width; d++) output[q][start + d] += probability * value[k][start + d];
                }
            }
        });
        return Linear(output, name + ".linear_out", ct);
    }
}
