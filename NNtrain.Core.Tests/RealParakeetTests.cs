using System.Diagnostics;
using System.Text.Json;
using NNtrain.Audio;
using Xunit;
namespace NNtrain.Core.Tests;
public sealed class RealParakeetTests(ITestOutputHelper output)
{
    [Fact]
    public void ConvertsAndLoadsActualWeightsOnCpu()
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_ASR_PARAKEET_CONVERT") != "1", "Explicit official-checkpoint conversion only.");
        string directory = Environment.GetEnvironmentVariable("NNTRAIN_ASR_MODEL")!;
        string nemo = Path.Combine(directory, "parakeet-tdt_ctc-0.6b-ja.nemo"), safetensors = Path.Combine(directory, "model.safetensors");
        var clock = Stopwatch.StartNew();
        NemoCheckpointConverter.ExtractModelAssets(nemo, directory, TestContext.Current.CancellationToken);
        long? converted = File.Exists(safetensors) ? null : NemoCheckpointConverter.ConvertToFp16(nemo, safetensors, TestContext.Current.CancellationToken);
        double conversionSeconds = clock.Elapsed.TotalSeconds; clock.Restart();
        using var model = ParakeetCtcModel.Load(directory, TestContext.Current.CancellationToken);
        var result = new { convertedWeightBytes = converted, conversionSeconds, cpuLoadSeconds = clock.Elapsed.TotalSeconds,
            hostWeightBytes = model.HostWeightBytes, model.ExecutionDevice, gpuOpened = false, microphoneOpened = false };
        string json = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Environment.GetEnvironmentVariable("NNTRAIN_ASR_REPORT")!, json); output.WriteLine(json);
        Assert.InRange(model.HostWeightBytes, 1, 2L * 1024 * 1024 * 1024);
    }
    [Fact]
    public void RecognizesApprovedJapaneseFileOnArcWithoutStartingLlmOrMicrophone()
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_ASR_PARAKEET_REAL_TEST") != "1", "Explicit Arc Japanese ASR test only.");
        CancellationToken ct = TestContext.Current.CancellationToken;
        var clock = Stopwatch.StartNew();
        using var model = ParakeetCtcModel.Load(Environment.GetEnvironmentVariable("NNTRAIN_ASR_MODEL")!, ct);
        double cpuLoadSeconds = clock.Elapsed.TotalSeconds;
        using var memory = new AsrGpuMemoryProbe(); clock.Restart(); model.EnableArc(0, ct);
        double gpuLoadSeconds = clock.Elapsed.TotalSeconds;
        using var file = File.OpenRead(Environment.GetEnvironmentVariable("NNTRAIN_ASR_TEST_WAV")!);
        float[] samples = Pcm16Wave.Read(file).To16Khz(ct);
        clock.Restart(); string transcript = model.Transcribe(samples, ct); double recognitionSeconds = clock.Elapsed.TotalSeconds;
        string reference = File.ReadAllText(Environment.GetEnvironmentVariable("NNTRAIN_ASR_REFERENCE")!);
        var result = new { transcript, reference, characterErrorRate = RealAsrTests.CharacterErrorRate(reference, transcript),
            audioSeconds = samples.Length / 16000d, recognitionSeconds, cpuLoadSeconds, gpuLoadSeconds,
            hostWeightBytes = model.HostWeightBytes, residentDeviceBytes = model.ResidentDeviceBytes, peakDeviceBufferBytes = model.PeakDeviceBufferBytes,
            osDedicatedGpuPeakBytes = memory.DedicatedPeakBytes, osSharedGpuPeakBytes = memory.SharedPeakBytes, osMemorySamples = memory.Samples,
            osMemoryError = memory.Error, device = model.ExecutionDevice, decoder = "CTC greedy", inputMode = "whole approved file", microphoneOpened = false, llmStarted = false };
        string json = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(Environment.GetEnvironmentVariable("NNTRAIN_ASR_REPORT")!, json); output.WriteLine(json);
        Assert.NotEmpty(transcript);
        Assert.InRange(model.PeakDeviceBufferBytes, 1, 2L * 1024 * 1024 * 1024);
    }
}
