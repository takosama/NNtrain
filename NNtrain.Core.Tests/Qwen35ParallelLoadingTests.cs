using System.Reflection;
using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35ParallelLoadingTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ParallelAndSequentialLoadsPreserveDeviceOrderResidentBytesAndLogitBits(
        bool reverseDevices, bool tiedOutput)
    {
        RequireTwoDevices();
        int[] devices = reverseDevices ? [1, 0] : [0, 1];
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(tiedOutput, contextLength: 40);
        Snapshot expected = Capture(file.Path, devices, parallel: false);
        Snapshot actual = Capture(file.Path, devices, parallel: true);

        Assert.Equal(devices, expected.Devices);
        Assert.Equal(devices, actual.Devices);
        Assert.Equal(expected.Weights, actual.Weights);
        Assert.Equal(expected.AuxiliaryWeights, actual.AuxiliaryWeights);
        Assert.Equal(expected.InitialState, actual.InitialState);
        Assert.Equal(expected.UploadedAtLoad, actual.UploadedAtLoad);
        Assert.Equal(expected.DownloadedAtLoad, actual.DownloadedAtLoad);
        Assert.All(actual.DownloadedAtLoad, bytes => Assert.Equal(0, bytes));
        Assert.Equal(expected.FinalState, actual.FinalState);
        Assert.Equal(expected.Logits.Length, actual.Logits.Length);
        for (int position = 0; position < expected.Logits.Length; position++)
            Assert.Equal(expected.Logits[position], actual.Logits[position]);
        Assert.Equal(expected.Generated, actual.Generated);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PreparingCallbackFailureReleasesSourceAndAllowsAnotherParallelLoad(bool reverseDevices)
    {
        RequireTwoDevices();
        int[] devices = reverseDevices ? [1, 0] : [0, 1];
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(tiedOutput: false);
        var preparing = new List<int>();
        const string failure = "Injected parallel preparation failure.";
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            Qwen35QuantizedModel.Load(file.Path, devices, message =>
            {
                // Progress callbacks are serialized by the loader. The other
                // worker still creates its lane, which the failing load owns
                // and must dispose after joining all preparation workers.
                foreach (int device in devices)
                {
                    if (!message.StartsWith($"Preparing Arc {device}:", StringComparison.Ordinal)) continue;
                    preparing.Add(device);
                    if (device == devices[1]) throw new InvalidOperationException(failure);
                }
            }, new() { ParallelModelLoad = true, ComputeModelFingerprintOnLoad = true }));
        Assert.Equal(failure, error.Message);
        Assert.Equal(devices.Order(), preparing.Order());

        // Both the shared GGUF reader and fingerprint handle must have closed.
        // Merely opening another read handle would not detect a leaked lock.
        using (var exclusive = new FileStream(file.Path, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
            Assert.True(exclusive.Length > 0);

        using (Qwen35QuantizedModel recovered = Qwen35QuantizedModel.Load(file.Path, devices,
            options: new() { ParallelModelLoad = true, ComputeModelFingerprintOnLoad = true }))
        {
            Assert.Equal(devices, DeviceIndices(recovered));
            float[] logits = recovered.ForwardToken(1);
            AssertMeaningful(logits);
            recovered.Reset();
            Assert.Equal(logits.Select(BitConverter.SingleToInt32Bits),
                recovered.ForwardToken(1).Select(BitConverter.SingleToInt32Bits));
        }
        using var afterDispose = new FileStream(file.Path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        Assert.True(afterDispose.Length > 0);
    }

    private static Snapshot Capture(string path, int[] devices, bool parallel)
    {
        using Qwen35QuantizedModel model = Qwen35QuantizedModel.Load(path, devices,
            options: new() { ParallelModelLoad = parallel });
        int[] loadedDevices = DeviceIndices(model);
        long[] weights = model.ResidentWeightBytes.ToArray();
        long[] auxiliary = model.ResidentAuxiliaryWeightBytes.ToArray();
        long[] initialState = model.ResidentStateBytes.ToArray();
        long[] uploaded = model.UploadedBytes.ToArray(), downloaded = model.DownloadedBytes.ToArray();
        var logits = new List<int[]>();
        // Advance both recurrent and full-attention state through the first
        // KV capacity growth, comparing exact bits on identical device maps.
        for (int position = 0; position < 20; position++)
        {
            float[] values = model.ForwardToken((position * 3 + 1) % 4);
            AssertMeaningful(values);
            logits.Add(values.Select(BitConverter.SingleToInt32Bits).ToArray());
        }
        long[] finalState = model.ResidentStateBytes.ToArray();
        int[] generated = model.GenerateTokenIds([1, 2, 0], 5);
        Assert.Equal(generated, model.GenerateTokenIds([1, 2, 0], 5));
        Assert.Equal(weights, model.ResidentWeightBytes);
        return new(loadedDevices, weights, auxiliary, initialState, finalState,
            uploaded, downloaded, logits.ToArray(), generated);
    }

    private static int[] DeviceIndices(Qwen35QuantizedModel model)
    {
        var lanes = (IReadOnlyList<ArcExecutionLane>)typeof(Qwen35QuantizedModel)
            .GetField("_lanes", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(model)!;
        return lanes.Select(lane => lane.DeviceIndex).ToArray();
    }

    private static void RequireTwoDevices()
        => Assert.SkipWhen(ArcDevices.Enumerate().Count < 2, "Two Intel Arc GPUs are required.");

    private static void AssertMeaningful(float[] logits)
    {
        Assert.Equal(4, logits.Length);
        Assert.All(logits, value => Assert.True(float.IsFinite(value)));
        Assert.True(logits.Max() - logits.Min() > 1e-5f);
    }

    private sealed record Snapshot(int[] Devices, long[] Weights, long[] AuxiliaryWeights,
        long[] InitialState, long[] FinalState, long[] UploadedAtLoad, long[] DownloadedAtLoad,
        int[][] Logits, int[] Generated);
}
