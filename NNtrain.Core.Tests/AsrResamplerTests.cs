using NNtrain.Audio;
using Xunit;
namespace NNtrain.Core.Tests;
public sealed class AsrResamplerTests
{
    [Theory]
    [InlineData(48000, 2)]
    [InlineData(48000, 1)]
    [InlineData(44100, 2)]
    [InlineData(16000, 1)]
    [InlineData(8000, 2)]
    public void ArbitraryChunksMatchWholeFileAndFinalLength(int rate, int channels)
    {
        int frames = rate * 2 + 17;
        var pcm = new short[frames * channels]; var mono = new float[frames];
        for (int i = 0; i < frames; i++)
        {
            int sum = 0;
            for (int c = 0; c < channels; c++)
            { short value = (short)(12000 * Math.Sin(2 * Math.PI * (437 + c * 13) * i / rate)); pcm[i * channels + c] = value; sum += value; }
            mono[i] = sum / (32768f * channels);
        }
        float[] expected = new Pcm16Wave(rate, mono).To16Khz(TestContext.Current.CancellationToken);
        var stream = new Pcm16StreamResampler(rate, channels); var actual = new List<float>();
        int[] sizes = [1, 7, 511, 3, 2048, 19, 15361];
        for (int offset = 0, index = 0; offset < pcm.Length; index++)
        {
            int count = Math.Min(sizes[index % sizes.Length], pcm.Length - offset);
            actual.AddRange(stream.Append(pcm.AsSpan(offset, count), ct: TestContext.Current.CancellationToken)); offset += count;
        }
        actual.AddRange(stream.Append([], final: true, ct: TestContext.Current.CancellationToken));
        Assert.Equal(expected.Length, actual.Count);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i]);
        Assert.Throws<InvalidOperationException>(() => stream.Append([], ct: TestContext.Current.CancellationToken));
    }
    [Fact]
    public void IncompleteStereoFrameIsRejectedAtFinalization()
    {
        var stream = new Pcm16StreamResampler(48000, 2);
        Assert.Empty(stream.Append(new short[] { 123 }, ct: TestContext.Current.CancellationToken));
        Assert.Throws<InvalidDataException>(() => stream.Append([], true, TestContext.Current.CancellationToken));
    }
    [Fact]
    public void HighFrequencyIsFilteredAndCancellationIsObserved()
    {
        var stream = new Pcm16StreamResampler(48000, 1);
        short[] pcm = Enumerable.Range(0, 48000).Select(i => (short)(20000 * Math.Sin(2 * Math.PI * 12000 * i / 48000))).ToArray();
        float[] samples = stream.Append(pcm, true, TestContext.Current.CancellationToken);
        double rms = Math.Sqrt(samples.Skip(100).Take(samples.Length - 200).Average(x => (double)x * x));
        Assert.True(rms < .005, $"Aliased tone RMS={rms}");
        using var cancellation = new CancellationTokenSource(); cancellation.Cancel();
        Assert.Throws<OperationCanceledException>(() => new Pcm16StreamResampler(48000, 2).Append(pcm, ct: cancellation.Token));
    }
}
