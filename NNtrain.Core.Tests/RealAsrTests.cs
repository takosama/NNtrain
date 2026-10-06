using System.Diagnostics;
using System.Text.Json;
using NNtrain.Audio;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class RealAsrTests(ITestOutputHelper output)
{
    [Fact]
    public async Task RunsLocalCheckpointOnArcWithRecordedMetrics()
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_ASR_REAL_TEST") != "1", "Explicit real-model test only.");
        string directory = Environment.GetEnvironmentVariable("NNTRAIN_ASR_MODEL")!;
        string wave = Environment.GetEnvironmentVariable("NNTRAIN_ASR_TEST_WAV")!;
        string report = Environment.GetEnvironmentVariable("NNTRAIN_ASR_REPORT")!;
        var timer = Stopwatch.StartNew();
        using var model = NemotronAsrModel.Load(directory, TestContext.Current.CancellationToken);
        double loadSeconds = timer.Elapsed.TotalSeconds;
        using var memory = new AsrGpuMemoryProbe();
        model.EnableArc(0, TestContext.Current.CancellationToken);
        File.WriteAllText(report, JsonSerializer.Serialize(new { phase = "loaded", pid = Environment.ProcessId,
            weightBytes = model.HostWeightBytes, residentDeviceBytes = model.ResidentDeviceBytes, loadSeconds }));
        using var file = File.OpenRead(wave);
        float[] samples = Pcm16Wave.Read(file).To16Khz(TestContext.Current.CancellationToken);
        var stream = model.CreateStream();
        timer.Restart();
        var partials = new List<string>();
        for (int offset = 0; offset < samples.Length; offset += 5120)
            stream.Append(samples.AsSpan(offset, Math.Min(5120, samples.Length - offset)),
                partial: text => partials.Add(text), ct: TestContext.Current.CancellationToken);
        string transcript = stream.Append([], true, ct: TestContext.Current.CancellationToken);
        double recognitionSeconds = timer.Elapsed.TotalSeconds;
        string? referencePath = Environment.GetEnvironmentVariable("NNTRAIN_ASR_REFERENCE");
        string? reference = referencePath is null ? null : File.ReadAllText(referencePath);
        var result = new { phase = "complete", pid = Environment.ProcessId, transcript, partials,
            reference, characterErrorRate = reference is null ? (double?)null : CharacterErrorRate(reference, transcript),
            audioSeconds = samples.Length / 16000d, recognitionSeconds, loadSeconds,
            hostWeightBytes = model.HostWeightBytes, residentDeviceBytes = model.ResidentDeviceBytes,
            peakDeviceBufferBytes = model.PeakDeviceBufferBytes, cpuCacheBytes = stream.CacheBytes,
            decoderEvaluations = stream.DecoderEvaluations, device = model.ExecutionDevice,
            osDedicatedGpuPeakBytes = memory.DedicatedPeakBytes, osSharedGpuPeakBytes = memory.SharedPeakBytes,
            osMemorySamples = memory.Samples, osMemoryError = memory.Error };
        string json = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(report, json); output.WriteLine(json);
        Assert.NotNull(transcript);
        Assert.True(partials.Count > 0);
        Assert.True(model.PeakDeviceBufferBytes > model.ResidentDeviceBytes);
        Assert.InRange(model.PeakDeviceBufferBytes, 1, 2L * 1024 * 1024 * 1024);
        if (int.TryParse(Environment.GetEnvironmentVariable("NNTRAIN_ASR_METRICS_HOLD_MS"), out int hold))
            await Task.Delay(Math.Clamp(hold, 0, 30000), TestContext.Current.CancellationToken);
    }

    internal static double CharacterErrorRate(string reference, string actual)
    {
        static System.Text.Rune[] Normalize(string value) => value.EnumerateRunes().Where(System.Text.Rune.IsLetterOrDigit).ToArray();
        var expected = Normalize(reference); var observed = Normalize(actual);
        var previous = Enumerable.Range(0, observed.Length + 1).ToArray();
        for (int row = 1; row <= expected.Length; row++)
        {
            var next = new int[observed.Length + 1]; next[0] = row;
            for (int col = 1; col <= observed.Length; col++)
                next[col] = Math.Min(Math.Min(next[col - 1] + 1, previous[col] + 1), previous[col - 1] + (expected[row - 1] == observed[col - 1] ? 0 : 1));
            previous = next;
        }
        return previous[^1] / (double)Math.Max(1, expected.Length);
    }
}
