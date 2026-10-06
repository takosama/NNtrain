using NNtrain.Audio;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class OfficialAsrParityTests
{
    [Fact]
    public void ExportsParakeetFrontendAndEncoderForOfficialComparison()
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_ASR_EXPORT_PARAKEET") != "1", "Explicit offline official comparison only.");
        var ct = TestContext.Current.CancellationToken;
        using var file = File.OpenRead(Environment.GetEnvironmentVariable("NNTRAIN_ASR_TEST_WAV")!);
        float[] samples = Pcm16Wave.Read(file).To16Khz(ct);
        int length = Environment.GetEnvironmentVariable("NNTRAIN_ASR_EXPORT_FULL") == "1" ? samples.Length : 9120;
        float[][] mel = ParakeetMel.Extract(samples.AsSpan(0, length), ct);
        using (var features = new BinaryWriter(File.Create(Environment.GetEnvironmentVariable("NNTRAIN_ASR_FEATURE_OUTPUT")!)))
            for (int band = 0; band < 80; band++) foreach (float[] row in mel[..^1]) features.Write(row[band]);
        using var model = ParakeetCtcModel.Load(Environment.GetEnvironmentVariable("NNTRAIN_ASR_MODEL")!, ct);
        if (Environment.GetEnvironmentVariable("NNTRAIN_ASR_EXPORT_DEVICE") == "Arc") model.EnableArc(0, ct);
        if (Environment.GetEnvironmentVariable("NNTRAIN_ASR_EXPORT_TRACE") is { } traceDirectory)
        {
            Directory.CreateDirectory(traceDirectory);
            model.EncoderObserver = (name, rows) =>
            {
                using var trace = new BinaryWriter(File.Create(Path.Combine(traceDirectory, name + ".f32")));
                foreach (var row in rows) foreach (float value in row) trace.Write(value);
            };
        }
        float[][] encoded = model.Encode(mel, ct);
        using var output = new BinaryWriter(File.Create(Environment.GetEnvironmentVariable("NNTRAIN_ASR_ENCODER_OUTPUT")!));
        foreach (var row in encoded) foreach (float value in row) output.Write(value);
        Assert.Equal((length / 160 + 7) / 8, encoded.Length);
    }

    [Fact]
    public void ExportsCpuOrArcEncoderForOfficialComparison()
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_ASR_EXPORT_ENCODER") != "1", "Explicit offline official comparison only.");
        var ct = TestContext.Current.CancellationToken;
        using var file = File.OpenRead(Environment.GetEnvironmentVariable("NNTRAIN_ASR_TEST_WAV")!);
        float[] samples = Pcm16Wave.Read(file).To16Khz(ct);
        float[][] frames = new NemotronMel().Append(samples.AsSpan(0, 9120), true, ct);
        using var model = NemotronAsrModel.Load(Environment.GetEnvironmentVariable("NNTRAIN_ASR_MODEL")!, ct);
        if (Environment.GetEnvironmentVariable("NNTRAIN_ASR_EXPORT_DEVICE") == "Arc") model.EnableArc(0, ct);
        var stream = model.CreateStream();
        float[][] first = model.Encode(frames[..25], stream, ct);
        float[][] second = model.Encode(frames[25..57], stream, ct);
        using var output = new BinaryWriter(File.Create(Environment.GetEnvironmentVariable("NNTRAIN_ASR_ENCODER_OUTPUT")!));
        foreach (var row in first.Concat(second)) foreach (float value in row) output.Write(value);
        Assert.Equal(8, first.Length + second.Length);
    }

    [Fact]
    public void ExportsApprovedFileFrontendForOfficialComparison()
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_ASR_EXPORT_FEATURES") != "1", "Explicit offline official comparison only.");
        using var file = File.OpenRead(Environment.GetEnvironmentVariable("NNTRAIN_ASR_TEST_WAV")!);
        float[] samples = Pcm16Wave.Read(file).To16Khz(TestContext.Current.CancellationToken);
        float[][] frames = new NemotronMel().Append(samples, true, TestContext.Current.CancellationToken);
        using var output = new BinaryWriter(File.Create(Environment.GetEnvironmentVariable("NNTRAIN_ASR_FEATURE_OUTPUT")!));
        // Official processor uses [batch, mel band, time].
        for (int band = 0; band < NemotronMel.Bands; band++)
            foreach (float[] frame in frames) output.Write(frame[band]);
        Assert.Equal(samples.Length / NemotronMel.Hop, frames.Length);
    }
}
