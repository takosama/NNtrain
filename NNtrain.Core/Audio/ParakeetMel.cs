using System.Numerics;
namespace NNtrain.Audio;
/// <summary>NeMo 3 evaluation frontend: zero-centered STFT and per-feature valid-frame normalization.</summary>
internal static class ParakeetMel
{
    private static readonly float[,] Filters = NemotronMel.CreateFilters(80);
    private static readonly float[] Window = Enumerable.Range(0, 400).Select(i => (float)(.5 - .5 * Math.Cos(2 * Math.PI * i / 399))).ToArray();
    internal static float[][] Extract(ReadOnlySpan<float> samples, CancellationToken ct)
    {
        if (samples.Length <= 256) return [];
        var signal = new float[samples.Length];
        for (int i = 0; i < signal.Length; i++)
        { if (!float.IsFinite(samples[i])) throw new InvalidDataException("Nonfinite PCM input."); signal[i] = i == 0 ? samples[i] : samples[i] - .97f * samples[i - 1]; }
        int frames = signal.Length / 160 + 1;
        int validFrames = frames - 1;
        var mel = new float[frames][];
        for (int t = 0; t < frames; t++)
        {
            ct.ThrowIfCancellationRequested(); var fft = new Complex[512];
            for (int i = 0; i < 400; i++)
            {
                int position = t * 160 - 200 + i;
                fft[56 + i] = (position < 0 || position >= signal.Length ? 0 : signal[position]) * Window[i];
            }
            NemotronMel.Transform(fft); var power = new float[257];
            for (int i = 0; i < power.Length; i++) power[i] = (float)(fft[i].Real * fft[i].Real + fft[i].Imaginary * fft[i].Imaginary);
            mel[t] = new float[80];
            for (int b = 0; b < 80; b++)
            { float sum = 0; for (int k = 0; k < 257; k++) sum += Filters[b, k] * power[k]; mel[t][b] = MathF.Log(sum + 1f / 16777216); }
        }
        for (int b = 0; b < 80; b++)
        {
            double sum = 0, variance = 0;
            for (int t = 0; t < validFrames; t++) sum += mel[t][b];
            float mean = (float)(sum / validFrames);
            for (int t = 0; t < validFrames; t++) variance += (double)(mel[t][b] - mean) * (mel[t][b] - mean);
            float deviation = (float)Math.Sqrt(variance / Math.Max(1, validFrames - 1)) + 1e-5f;
            for (int t = 0; t < validFrames; t++) mel[t][b] = (mel[t][b] - mean) / deviation;
            mel[^1][b] = 0; // NeMo masks the centered terminal frame after normalization.
        }
        return mel;
    }
}
