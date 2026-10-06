using System.Buffers.Binary;
using System.Text;

namespace NNtrain.Audio;

/// <summary>Strict, bounded file input. Does not access a microphone.</summary>
public sealed record Pcm16Wave(int SampleRate, float[] MonoSamples)
{
    public static Pcm16Wave Read(Stream stream, int maximumSeconds = 300)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maximumSeconds);
        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
        if (new string(reader.ReadChars(4)) != "RIFF") throw new InvalidDataException("Expected RIFF WAV.");
        uint length = reader.ReadUInt32();
        if (new string(reader.ReadChars(4)) != "WAVE" || length < 4) throw new InvalidDataException("Expected WAVE.");
        long remaining = length - 4;
        int rate = 0, channels = 0;
        float[]? samples = null;
        while (remaining >= 8)
        {
            string id = new(reader.ReadChars(4));
            uint size = reader.ReadUInt32();
            long padded = size + (size & 1L);
            remaining -= 8;
            if (padded > remaining) throw new InvalidDataException("Truncated WAV chunk.");
            if (id == "fmt ")
            {
                if (rate != 0 || size < 16 || size > 4096) throw new InvalidDataException("Invalid WAV format chunk.");
                byte[] format = reader.ReadBytes((int)size);
                if (format.Length != size) throw new EndOfStreamException();
                channels = BinaryPrimitives.ReadUInt16LittleEndian(format.AsSpan(2));
                rate = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(format.AsSpan(4)));
                if (BinaryPrimitives.ReadUInt16LittleEndian(format) != 1 || channels is < 1 or > 2 || rate is < 8000 or > 192000
                    || BinaryPrimitives.ReadUInt16LittleEndian(format.AsSpan(14)) != 16
                    || BinaryPrimitives.ReadUInt16LittleEndian(format.AsSpan(12)) != channels * 2
                    || BinaryPrimitives.ReadUInt32LittleEndian(format.AsSpan(8)) != rate * channels * 2)
                    throw new InvalidDataException("Expected 16-bit PCM WAV, mono or stereo (8–192 kHz).");
            }
            else if (id == "data")
            {
                if (rate == 0 || samples is not null || size % (channels * 2) != 0)
                    throw new InvalidDataException("Invalid WAV data chunk.");
                long frames = size / (channels * 2);
                if (frames > (long)rate * maximumSeconds) throw new InvalidDataException("Audio exceeds duration limit.");
                samples = new float[checked((int)frames)];
                for (int i = 0; i < samples.Length; i++)
                {
                    int sum = reader.ReadInt16();
                    if (channels == 2) sum += reader.ReadInt16();
                    samples[i] = sum / (32768f * channels);
                }
            }
            else
            {
                // Skip without allocating attacker-controlled chunk sizes; works on nonseekable streams.
                byte[] buffer = new byte[4096];
                long left = size;
                while (left > 0)
                {
                    int read = stream.Read(buffer, 0, (int)Math.Min(buffer.Length, left));
                    if (read == 0) throw new EndOfStreamException();
                    left -= read;
                }
            }
            if ((size & 1) != 0) _ = reader.ReadByte();
            remaining -= padded;
        }
        if (remaining != 0 || samples is null || samples.Length == 0) throw new InvalidDataException("Missing or invalid WAV audio.");
        return new(rate, samples);
    }

    /// <summary>Windowed-sinc resampling with antialias filtering to the model's 16 kHz input.</summary>
    public float[] To16Khz(CancellationToken cancellationToken = default)
    {
        if (SampleRate is < 8000 or > 192000) throw new InvalidDataException("Invalid sample rate.");
        cancellationToken.ThrowIfCancellationRequested();
        if (SampleRate == 16000) return (float[])MonoSamples.Clone();
        int length = checked((int)Math.Ceiling(MonoSamples.Length * 16000d / SampleRate));
        var output = new float[length];
        double cutoff = Math.Min(1, 16000d / SampleRate) * 0.95;
        int radius = (int)Math.Ceiling(24 / cutoff);
        for (int i = 0; i < length; i++)
        {
            if ((i & 255) == 0) cancellationToken.ThrowIfCancellationRequested();
            double position = i * (double)SampleRate / 16000, sum = 0, normalization = 0;
            int center = (int)position;
            for (int j = Math.Max(0, center - radius); j <= Math.Min(MonoSamples.Length - 1, center + radius); j++)
            {
                double distance = j - position;
                if (Math.Abs(distance) >= radius) continue;
                double x = Math.PI * distance * cutoff;
                double weight = cutoff * (Math.Abs(x) < 1e-12 ? 1 : Math.Sin(x) / x)
                    * (0.5 + 0.5 * Math.Cos(Math.PI * distance / radius));
                sum += MonoSamples[j] * weight;
                normalization += weight;
            }
            output[i] = normalization == 0 ? 0 : (float)(sum / normalization);
        }
        return output;
    }
}
