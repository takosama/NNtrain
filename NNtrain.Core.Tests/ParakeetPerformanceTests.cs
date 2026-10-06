using System.Diagnostics;
using System.Text.Json;
using NNtrain.Audio;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class ParakeetPerformanceTests
{
    [Fact]
    public void ProfileApprovedFileOnArc()
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_PARAKEET_PROFILE") != "1", "Explicit ASR-only file profiling.");
        var ct = TestContext.Current.CancellationToken;
        using var model = ParakeetCtcModel.Load(Environment.GetEnvironmentVariable("NNTRAIN_ASR_MODEL")!, ct);
        using var memory = new AsrGpuMemoryProbe();
        model.EnableArc(0, ct);
        var timings = new Dictionary<string, double>();
        model.TimingObserver = (name, milliseconds) => timings[name] = timings.GetValueOrDefault(name) + milliseconds;
        string firstPath = Environment.GetEnvironmentVariable("NNTRAIN_ASR_TEST_WAV")!;
        string report = Environment.GetEnvironmentVariable("NNTRAIN_ASR_REPORT")!;
        bool all = Environment.GetEnvironmentVariable("NNTRAIN_PARAKEET_PROFILE_ALL") == "1";
        var results = new List<object>();
        for (int index = 0; index < (all ? 3 : 1); index++)
        {
            string path = all ? Path.Combine(Path.GetDirectoryName(firstPath)!, $"fleurs-ja-validation-{index}-pcm16.wav") : firstPath;
            using var file = File.OpenRead(path);
            float[] samples = Pcm16Wave.Read(file).To16Khz(ct); timings.Clear();
            if (all) model.EncoderObserver = (name, rows) =>
            {
                if (name != $"layer-{model.Layers - 1}") return;
                using var binary = new BinaryWriter(File.Create(report + $".{index}.encoder.f32"));
                foreach (float[] row in rows) foreach (float value in row) binary.Write(value);
            };
            var clock = Stopwatch.StartNew();
            string text = model.Transcribe(samples, ct);
            double seconds = clock.Elapsed.TotalSeconds;
            var result = new { index, text, seconds, audioSeconds = samples.Length / 16000d,
                timings = new Dictionary<string, double>(timings), osDedicatedPeakBytes = memory.DedicatedPeakBytes, osSharedPeakBytes = memory.SharedPeakBytes,
                model.ResidentDeviceBytes, model.PeakDeviceBufferBytes, memory.Error, microphoneStarted = false };
            results.Add(result); Assert.NotEmpty(text);
        }
        File.WriteAllText(report, JsonSerializer.Serialize(all ? (object)results : results[0], new JsonSerializerOptions { WriteIndented = true }));
    }
}
