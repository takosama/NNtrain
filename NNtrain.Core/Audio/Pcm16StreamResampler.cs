namespace NNtrain.Audio;

/// <summary>Continuous PCM16 channel mixing and phase-preserving sinc conversion to 16 kHz.</summary>
public sealed class Pcm16StreamResampler
{
    private readonly int _rate, _channels, _radius;
    private readonly double _cutoff;
    private readonly List<float> _samples = [];
    private long _base, _received, _output;
    private int _channelPosition, _channelSum;
    private bool _final;
    public Pcm16StreamResampler(int sampleRate, int channels)
    {
        if (sampleRate is < 8000 or > 192000 || channels is < 1 or > 2)
            throw new ArgumentOutOfRangeException(nameof(sampleRate));
        _rate = sampleRate; _channels = channels;
        _cutoff = Math.Min(1, 16000d / sampleRate) * .95;
        _radius = (int)Math.Ceiling(24 / _cutoff);
    }
    public float[] Append(ReadOnlySpan<short> pcm, bool final = false, CancellationToken ct = default)
    {
        if (_final) throw new InvalidOperationException("Audio stream is already finalized.");
        ct.ThrowIfCancellationRequested();
        foreach (short value in pcm)
        {
            _channelSum += value;
            if (++_channelPosition == _channels)
            {
                _samples.Add(_channelSum / (32768f * _channels)); _received++;
                _channelSum = _channelPosition = 0;
            }
        }
        if (final && _channelPosition != 0) throw new InvalidDataException("Incomplete PCM channel frame.");
        _final = final;
        var result = new List<float>();
        long limit = (_received * 16000 + _rate - 1) / _rate;
        while (_output < limit)
        {
            ct.ThrowIfCancellationRequested();
            double position = _output * (double)_rate / 16000;
            long center = (long)position;
            if (!final && _rate != 16000 && center + _radius >= _received) break;
            if (_rate == 16000) result.Add(_samples[checked((int)(_output - _base))]);
            else
            {
                double sum = 0, norm = 0;
                for (long j = Math.Max(0, center - _radius); j <= Math.Min(_received - 1, center + _radius); j++)
                {
                    double distance = j - position;
                    if (Math.Abs(distance) >= _radius) continue;
                    double x = Math.PI * distance * _cutoff;
                    double weight = _cutoff * (Math.Abs(x) < 1e-12 ? 1 : Math.Sin(x) / x)
                        * (.5 + .5 * Math.Cos(Math.PI * distance / _radius));
                    sum += _samples[checked((int)(j - _base))] * weight; norm += weight;
                }
                result.Add(norm == 0 ? 0 : (float)(sum / norm));
            }
            _output++;
        }
        long keep = Math.Min(_received, Math.Max(_base, (long)(_output * (double)_rate / 16000) - _radius));
        int remove = checked((int)(keep - _base));
        if (remove > 0) { _samples.RemoveRange(0, remove); _base = keep; }
        return result.ToArray();
    }
}
