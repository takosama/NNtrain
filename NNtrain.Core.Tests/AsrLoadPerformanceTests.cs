using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using NNtrain.Audio;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class AsrLoadPerformanceTests(ITestOutputHelper output)
{
    [Fact]
    public void ProfileApprovedModelLoads()
    {
        string? mode = Environment.GetEnvironmentVariable("NNTRAIN_ASR_LOAD_PROFILE");
        Assert.SkipWhen(mode is not ("cpu" or "arc"), "Explicit ASR-only loading profile.");
        var ct = TestContext.Current.CancellationToken;
        string report = Environment.GetEnvironmentVariable("NNTRAIN_ASR_LOAD_REPORT")!;
        var results = new List<object>();
        foreach (string name in new[] { "nemotron", "parakeet" })
        {
            string directory = Environment.GetEnvironmentVariable($"NNTRAIN_ASR_{name.ToUpperInvariant()}_MODEL")!;
            for (int run = 0; run < 2; run++)
            {
                GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
                var result = Run(name, directory, run, mode == "arc", ct);
                results.Add(result);
                File.WriteAllText(report, JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true }));
                output.WriteLine(JsonSerializer.Serialize(result));
            }
        }
    }

    private object Run(string name, string directory, int run, bool arc, CancellationToken ct)
    {
        var phases = new Dictionary<string, double>();
        void Observe(string phase, double ms) => phases[phase] = phases.GetValueOrDefault(phase) + ms;
        long allocatedBefore = GC.GetTotalAllocatedBytes(true);
        var timer = Stopwatch.StartNew();
        using ILocalAsrModel model = name == "nemotron"
            ? NemotronAsrModel.Load(directory, ct, Observe)
            : ParakeetCtcModel.Load(directory, ct, Observe);
        double cpuLoadSeconds = timer.Elapsed.TotalSeconds;
        long managedAllocatedBytes = GC.GetTotalAllocatedBytes(true) - allocatedBefore;
        long managedLiveBytes = GC.GetTotalMemory(false);
        using var process = Process.GetCurrentProcess();
        long workingSetBytes = process.WorkingSet64;
        var checkpoint = (AsrHalfCheckpoint)model.GetType().GetField("_checkpoint", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (var pair in checkpoint.Weights.OrderBy(x => x.Key, StringComparer.Ordinal))
            hash.AppendData(MemoryMarshal.AsBytes(pair.Value.Values.AsSpan()));
        string weightSha256 = Convert.ToHexString(hash.GetHashAndReset());
        double gpuLoadSeconds = 0, recognitionSeconds = 0;
        string? transcript = null;
        using var memory = arc ? new AsrGpuMemoryProbe() : null;
        if (arc)
        {
            timer.Restart();
            if (model is NemotronAsrModel nemotron) nemotron.EnableArc(0, ct, Observe);
            else ((ParakeetCtcModel)model).EnableArc(0, ct, Observe);
            gpuLoadSeconds = timer.Elapsed.TotalSeconds;
            using var file = File.OpenRead(Environment.GetEnvironmentVariable("NNTRAIN_ASR_TEST_WAV")!);
            float[] samples = Pcm16Wave.Read(file).To16Khz(ct);
            timer.Restart();
            var stream = model.CreateStream();
            using var streamLifetime = stream as IDisposable;
            if (model is NemotronAsrModel)
            {
                for (int offset = 0; offset < samples.Length; offset += 5120)
                    stream.Append(samples.AsSpan(offset, Math.Min(5120, samples.Length - offset)), ct: ct);
                transcript = stream.Append([], final: true, ct: ct);
            }
            else transcript = stream.Append(samples, final: true, ct: ct);
            recognitionSeconds = timer.Elapsed.TotalSeconds;
            Assert.NotEmpty(transcript);
            string expectedPath = Environment.GetEnvironmentVariable($"NNTRAIN_ASR_{name.ToUpperInvariant()}_EXPECTED_REPORT")!;
            using var expected = JsonDocument.Parse(File.ReadAllText(expectedPath));
            string expectedText = expected.RootElement.GetProperty("transcript").GetString()!;
            Assert.Equal(expectedText, transcript);
            Assert.InRange(model.PeakDeviceBufferBytes, 1, 2L * 1024 * 1024 * 1024);
        }
        return new { model = name, run, processFirstLoad = run == 0,
            osCacheState = "not flushed; process first load is not a cold-disk claim",
            cpuLoadSeconds, gpuLoadSeconds, recognitionSeconds, phases,
            managedAllocatedBytes, managedLiveBytes, workingSetBytes,
            hostWeightBytes = model.HostWeightBytes, weightSha256, transcript,
            model.ResidentDeviceBytes, model.PeakDeviceBufferBytes,
            osDedicatedGpuPeakBytes = memory?.DedicatedPeakBytes,
            osSharedGpuPeakBytes = memory?.SharedPeakBytes,
            gpuOpened = arc, microphoneOpened = false, llmStarted = false };
    }
}
