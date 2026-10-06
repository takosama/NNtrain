using NNtrain.Arc;
using NNtrain.Audio;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class AsrArcTests(ITestOutputHelper output)
{
    [Fact]
    public void MiniConformerMatchesCpuWithResidentArcProjections()
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_ASR_ARC_TEST") != "1", "Explicit Arc test only.");
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc required.");
        using var fixture = new NemotronModelTests.MiniAsrFixture();
        using var cpu = NemotronAsrModel.Load(fixture.Directory, TestContext.Current.CancellationToken);
        using var gpu = NemotronAsrModel.Load(fixture.Directory, TestContext.Current.CancellationToken);
        gpu.EnableArc(0, TestContext.Current.CancellationToken);
        var cpuStream = cpu.CreateStream(); var gpuStream = gpu.CreateStream();
        float maximumError = 0;
        for (int chunk = 0; chunk < 3; chunk++)
        {
            float[][] mel = Enumerable.Range(0, chunk == 0 ? 25 : 32).Select(t => Enumerable.Range(0, 128).Select(f => MathF.Sin(t * .1f + f * .03f)).ToArray()).ToArray();
            float[][] expected = cpu.Encode(mel, cpuStream, TestContext.Current.CancellationToken);
            float[][] actual = gpu.Encode(mel, gpuStream, TestContext.Current.CancellationToken);
            for (int t = 0; t < actual.Length; t++)
                for (int d = 0; d < actual[t].Length; d++) maximumError = Math.Max(maximumError, Math.Abs(actual[t][d] - expected[t][d]));
        }
        output.WriteLine($"Mini Conformer/cache CPU vs Arc max error: {maximumError:G9}");
        Assert.InRange(maximumError, 0, 2e-5f);
        Assert.Equal("", gpu.CreateStream().Append(new float[16000], true, ct: TestContext.Current.CancellationToken));
    }

    [Fact]
    public void ResidentHalfLinearMatchesCpuOnEveryAvailableArc()
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_ASR_ARC_TEST") != "1", "Set NNTRAIN_ASR_ARC_TEST=1 for explicit Arc test.");
        var devices = ArcDevices.Enumerate();
        Assert.SkipWhen(devices.Count == 0, "Intel Arc required.");
        var random = new Random(61006);
        var weight = new AsrHalfCheckpoint.Weight([65, 257], Enumerable.Range(0, 65 * 257).Select(_ => (Half)((random.NextSingle() - 0.5f) / 10)).ToArray());
        var bias = new AsrHalfCheckpoint.Weight([65], Enumerable.Range(0, 65).Select(_ => (Half)(random.NextSingle() / 10)).ToArray());
        var weights = new Dictionary<string, AsrHalfCheckpoint.Weight> { ["projection.weight"] = weight, ["projection.bias"] = bias };
        float[][] rows = Enumerable.Range(0, 4).Select(_ => Enumerable.Range(0, 257).Select(_ => random.NextSingle() - 0.5f).ToArray()).ToArray();
        foreach (var device in devices)
        {
            using var arc = new AsrArcLinear(weights, device.Index, TestContext.Current.CancellationToken);
            float[][] actual = arc.Linear(rows, "projection");
            float maximumError = 0;
            for (int row = 0; row < rows.Length; row++)
            {
                float[] expected = AsrCpuMath.Linear(rows[row], weight, bias);
                for (int i = 0; i < expected.Length; i++) maximumError = Math.Max(maximumError, Math.Abs(expected[i] - actual[row][i]));
            }
            output.WriteLine($"{device.Index} {device.Name}: maximum error {maximumError:G9}, device buffer peak {arc.PeakDeviceBufferBytes} bytes");
            Assert.InRange(maximumError, 0, 1e-5f);
        }
    }
}
