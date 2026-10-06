using NNtrain.Audio;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class NemotronMelTests
{
    [Fact]
    public void SilenceMatchesOfficialLogGuardAndTerminalMask()
    {
        var frames = new NemotronMel().Append(new float[1600], true, TestContext.Current.CancellationToken);
        Assert.Equal(10, frames.Length);
        foreach (var frame in frames)
            foreach (float value in frame) Assert.Equal(MathF.Log(1f / 16777216), value);
    }

    [Fact]
    public void ArbitraryChunkBoundariesMatchSinglePassFeatures()
    {
        float[] samples = Enumerable.Range(0, 12345).Select(i => (float)(0.25 * Math.Sin(i * 0.13) + 0.1 * Math.Cos(i * 0.027))).ToArray();
        var expected = new NemotronMel().Append(samples, true, TestContext.Current.CancellationToken);
        var stream = new NemotronMel();
        var actual = new List<float[]>();
        for (int offset = 0; offset < samples.Length; offset += 137)
            actual.AddRange(stream.Append(samples.AsSpan(offset, Math.Min(137, samples.Length - offset)), ct: TestContext.Current.CancellationToken));
        actual.AddRange(stream.Append([], true, TestContext.Current.CancellationToken));
        Assert.Equal(expected.Length, actual.Count);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i]);
        Assert.Equal(0, stream.BufferedBytes);
        Assert.Throws<InvalidOperationException>(() => stream.Append([], ct: TestContext.Current.CancellationToken));
    }

    [Fact]
    public void FftFrontendMatchesIndependentDirectDft()
    {
        float[] samples = Enumerable.Range(0, 800).Select(i => (float)(0.2 * Math.Sin(i * 0.09))).ToArray();
        float[] actual = new NemotronMel().Append(samples, true, TestContext.Current.CancellationToken)[2];
        // Independent O(N^2) DFT using the documented centered, symmetric-Hann STFT.
        double[] power = new double[257];
        for (int bin = 0; bin < power.Length; bin++)
        {
            double real = 0, imaginary = 0;
            for (int j = 0; j < 400; j++)
            {
                int index = 320 - 200 + j;
                float emphasized = samples[index] - 0.97f * samples[index - 1];
                float window = (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * j / 399));
                double value = emphasized * window;
                double phase = -2 * Math.PI * bin * (56 + j) / 512;
                real += value * Math.Cos(phase); imaginary += value * Math.Sin(phase);
            }
            power[bin] = real * real + imaginary * imaginary;
        }
        float[,] filters = NemotronMel.CreateFilters();
        for (int band = 0; band < 128; band++)
        {
            double energy = 0;
            for (int bin = 0; bin < power.Length; bin++) energy += filters[band, bin] * power[bin];
            double expected = Math.Log(energy + 1d / 16777216);
            Assert.True(Math.Abs(actual[band] - expected) < 1e-4, $"band {band}: {actual[band]} vs {expected}");
        }
    }

    [Fact]
    public void LstmUsesPytorchGateOrderingAndCarriesCellState()
    {
        static AsrHalfCheckpoint.Weight Weight(float[] values) => new([4, 1], values.Select(x => (Half)x).ToArray());
        var hidden = new float[1]; var cell = new float[1];
        var ih = Weight([0, 0, 1, 0]); var hh = Weight([0, 0, 0, 0]); var bias = Weight([0, 0, 0, 0]);
        _ = AsrCpuMath.Lstm([1], hidden, cell, ih, hh, bias, bias);
        float expectedCell = 0.5f * MathF.Tanh(1);
        Assert.Equal(expectedCell, cell[0], 6);
        Assert.Equal(0.5f * MathF.Tanh(expectedCell), hidden[0], 6);
        _ = AsrCpuMath.Lstm([0], hidden, cell, ih, hh, bias, bias);
        Assert.Equal(0.5f * expectedCell, cell[0], 6);
    }
}
