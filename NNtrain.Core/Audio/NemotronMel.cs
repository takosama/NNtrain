using System.Numerics;

namespace NNtrain.Audio;

/// <summary>Nemotron's unnormalized Slaney log-mel frontend (16 kHz, symmetric Hann, constant STFT padding).</summary>
/// <remarks>Algorithm follows Hugging Face NemotronAsrStreamingFeatureExtractor, Apache-2.0.
/// https://github.com/huggingface/transformers/tree/main/src/transformers/models/nemotron_asr_streaming</remarks>
public sealed class NemotronMel
{
    public const int Bands = 128, FftSize = 512, Hop = 160, Window = 400;
    private static readonly float[] WindowValues = Enumerable.Range(0, Window)
        .Select(i => (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * i / (Window - 1)))).ToArray();
    private static readonly float[,] Filters = CreateFilters();
    private readonly List<float> _samples = [];
    private long _sampleOffset, _sampleCount, _frameIndex;
    private float _previous;
    private bool _finished;

    public long SampleCount => _sampleCount;
    public long FramesProduced => _frameIndex;
    public long BufferedBytes => _samples.Count * 4L;

    /// <summary>Accepts new mono samples only. Carries preemphasis history and overlapping STFT samples across calls.</summary>
    public float[][] Append(ReadOnlySpan<float> samples, bool final = false, CancellationToken ct = default)
    {
        if (_finished) throw new InvalidOperationException("Frontend already finalized.");
        foreach (float sample in samples)
        {
            if (!float.IsFinite(sample)) throw new ArgumentException("Audio must be finite.");
            _samples.Add(_sampleCount == 0 ? sample : sample - 0.97f * _previous);
            _previous = sample;
            _sampleCount++;
        }
        var result = new List<float[]>();
        // The official centered frontend masks the terminal frame: valid count is floor(L / hop).
        long validFrames = _sampleCount / Hop;
        while (_frameIndex < validFrames && (final || _frameIndex * Hop + FftSize / 2 <= _sampleCount))
        {
            ct.ThrowIfCancellationRequested();
            var fft = new Complex[FftSize];
            long begin = _frameIndex * Hop - Window / 2;
            for (int i = 0; i < Window; i++)
            {
                long index = begin + i;
                float sample = index < 0 || index >= _sampleCount ? 0 : _samples[checked((int)(index - _sampleOffset))];
                fft[(FftSize - Window) / 2 + i] = sample * WindowValues[i];
            }
            Transform(fft);
            var power = new float[FftSize / 2 + 1];
            for (int i = 0; i < power.Length; i++)
                power[i] = (float)(fft[i].Real * fft[i].Real + fft[i].Imaginary * fft[i].Imaginary);
            var frame = new float[Bands];
            for (int band = 0; band < Bands; band++)
            {
                float sum = 0;
                for (int bin = 0; bin < power.Length; bin++) sum += Filters[band, bin] * power[bin];
                frame[band] = MathF.Log(sum + 1f / 16777216);
            }
            result.Add(frame);
            _frameIndex++;
        }
        long retainFrom = Math.Max(0, _frameIndex * Hop - FftSize / 2);
        int remove = (int)Math.Min(_samples.Count, Math.Max(0, retainFrom - _sampleOffset));
        if (remove > 0) { _samples.RemoveRange(0, remove); _sampleOffset += remove; }
        if (final) { _finished = true; _samples.Clear(); }
        return result.ToArray();
    }

    internal static void Transform(Complex[] values)
    {
        for (int i = 1, j = 0; i < values.Length; i++)
        {
            int bit = values.Length >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) (values[i], values[j]) = (values[j], values[i]);
        }
        for (int size = 2; size <= values.Length; size <<= 1)
        {
            Complex step = Complex.FromPolarCoordinates(1, -2 * Math.PI / size);
            for (int start = 0; start < values.Length; start += size)
            {
                Complex phase = Complex.One;
                for (int j = 0; j < size / 2; j++)
                {
                    Complex a = values[start + j], b = values[start + j + size / 2] * phase;
                    values[start + j] = a + b;
                    values[start + j + size / 2] = a - b;
                    phase *= step;
                }
            }
        }
    }

    internal static float[,] CreateFilters(int bands = Bands)
    {
        static double ToMel(double hz) => hz < 1000 ? hz / (200d / 3) : 15 + Math.Log(hz / 1000) / (Math.Log(6.4) / 27);
        static double ToHz(double mel) => mel < 15 ? mel * (200d / 3) : 1000 * Math.Exp((mel - 15) * Math.Log(6.4) / 27);
        double[] edges = Enumerable.Range(0, bands + 2).Select(i => ToHz(i * ToMel(8000) / (bands + 1))).ToArray();
        var filters = new float[bands, FftSize / 2 + 1];
        for (int band = 0; band < bands; band++)
            for (int bin = 0; bin <= FftSize / 2; bin++)
            {
                double hz = bin * 16000d / FftSize;
                float triangle = (float)Math.Max(0, Math.Min((hz - edges[band]) / (edges[band + 1] - edges[band]),
                    (edges[band + 2] - hz) / (edges[band + 2] - edges[band + 1])));
                filters[band, bin] = (float)(triangle * (2 / (edges[band + 2] - edges[band])));
            }
        return filters;
    }
}
