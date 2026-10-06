using System.Text;
using System.Text.Json;
using NNtrain.Audio;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class AsrInputTests
{
    [Fact]
    public void FloatCheckpointIsHeldAsHalfWithExactByteAccounting()
    {
        string path = Checkpoint("F32", [1f, -2f, 0.5f]);
        try
        {
            var loaded = AsrHalfCheckpoint.Load(path, 6, TestContext.Current.CancellationToken);
            Assert.Equal(6, loaded.WeightBytes);
            Assert.Equal(new Half[] { (Half)1, (Half)(-2), (Half)0.5 }, loaded.Weights["test"].Values);
            Assert.Throws<InvalidDataException>(() => AsrHalfCheckpoint.Load(path, 5, TestContext.Current.CancellationToken));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void HalfCheckpointAndCancellationAreSupported()
    {
        string path = Checkpoint("F16", [0.25f, -0.125f]);
        try
        {
            Assert.Equal((Half)0.25, AsrHalfCheckpoint.Load(path, 4, TestContext.Current.CancellationToken).Weights["test"].Values[0]);
            Assert.Throws<OperationCanceledException>(() => AsrHalfCheckpoint.Load(path, 4, new CancellationToken(true)));
        }
        finally { File.Delete(path); }
    }

    [Theory]
    [InlineData(float.PositiveInfinity)]
    [InlineData(100000f)]
    public void RejectsWeightsThatCannotBeRepresentedInHalf(float value)
    {
        string path = Checkpoint("F32", [value]);
        try { Assert.Throws<InvalidDataException>(() => AsrHalfCheckpoint.Load(path, 2, TestContext.Current.CancellationToken)); }
        finally { File.Delete(path); }
    }

    [Fact]
    public void ReadsStereoPcm16AndNormalizesToMono()
    {
        using var stream = Wave(16000, 2, [32767, -32768, 16384, 16384]);
        var audio = Pcm16Wave.Read(stream);
        Assert.Equal(16000, audio.SampleRate);
        Assert.Equal(-1f / 65536, audio.MonoSamples[0]);
        Assert.Equal(0.5f, audio.MonoSamples[1]);
        Assert.Equal(audio.MonoSamples, audio.To16Khz(TestContext.Current.CancellationToken));
        Assert.True(stream.CanRead);
    }

    [Fact]
    public void DownsamplingSuppressesAboveNyquistTone()
    {
        float[] tone = Enumerable.Range(0, 48000).Select(i => (float)Math.Sin(2 * Math.PI * 12000 * i / 48000)).ToArray();
        float[] output = new Pcm16Wave(48000, tone).To16Khz(TestContext.Current.CancellationToken);
        Assert.Equal(16000, output.Length);
        double rms = Math.Sqrt(output.Skip(100).Take(15800).Average(x => (double)x * x));
        Assert.True(rms < 0.005, $"Aliased RMS: {rms}");
        Assert.Throws<OperationCanceledException>(() => new Pcm16Wave(48000, tone).To16Khz(new CancellationToken(true)));
    }

    [Fact]
    public void RejectsTruncatedAudioAndExcessDuration()
    {
        using var stream = Wave(8000, 1, new short[8001]);
        Assert.Throws<InvalidDataException>(() => Pcm16Wave.Read(stream, 1));
        using var source = Wave(16000, 1, [1, 2, 3]);
        using var truncated = new MemoryStream(source.ToArray()[..^1]);
        Assert.Throws<EndOfStreamException>(() => Pcm16Wave.Read(truncated));
    }

    private static string Checkpoint(string dtype, float[] values)
    {
        string path = Path.Combine(Path.GetTempPath(), $"nntrain-asr-{Guid.NewGuid():N}.safetensors");
        int width = dtype == "F32" ? 4 : 2;
        byte[] header = JsonSerializer.SerializeToUtf8Bytes(new Dictionary<string, object>
        {
            ["test"] = new { dtype, shape = new[] { values.Length }, data_offsets = new[] { 0, values.Length * width } }
        });
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write((ulong)header.Length);
        writer.Write(header);
        foreach (float value in values)
            if (width == 4) writer.Write(value); else writer.Write(BitConverter.HalfToUInt16Bits((Half)value));
        return path;
    }

    private static MemoryStream Wave(int rate, short channels, short[] values)
    {
        var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, Encoding.ASCII, true))
        {
            writer.Write(Encoding.ASCII.GetBytes("RIFF")); writer.Write(36 + values.Length * 2);
            writer.Write(Encoding.ASCII.GetBytes("WAVEfmt ")); writer.Write(16);
            writer.Write((short)1); writer.Write(channels); writer.Write(rate);
            writer.Write(rate * channels * 2); writer.Write((short)(channels * 2)); writer.Write((short)16);
            writer.Write(Encoding.ASCII.GetBytes("data")); writer.Write(values.Length * 2);
            foreach (short value in values) writer.Write(value);
        }
        stream.Position = 0;
        return stream;
    }
}
