using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35TrainingGradientStateTests
{
    [Fact]
    public void LogicalClearOverwritesFirstBackwardAndPreservesLaterAccumulation()
    {
        using var lane = CreateLane();
        const int input = 513, output = 259, rank = 3, rows = 2;
        using var adapter = new Qwen35LoraMatrix(lane, input, output,
            new() { Rank = rank, Alpha = 6 }, new Random(11));
        adapter.RestoreState(State(input, output, rank), training: true);
        using ArcBuffer x = lane.Upload(Values(rows * input, 31));
        using ArcBuffer y = lane.Upload(new float[rows * output]);
        using ArcBuffer dy = lane.Upload(Values(rows * output, 37));
        using ArcBuffer z = adapter.Forward(x, y, rows);
        adapter.Backward(x, z, dy, null, rows);
        float[][] first = adapter.ReadGradients();
        Assert.Contains(first.SelectMany(values => values), value => value != 0f);
        adapter.Backward(x, z, dy, null, rows);
        float[][] twice = adapter.ReadGradients();
        for (int field = 0; field < 2; field++)
        for (int i = 0; i < first[field].Length; i++) Close(first[field][i] * 2f, twice[field][i]);

        long launches = lane.KernelLaunchCount, downloads = lane.D2HBytes;
        adapter.ZeroGrad();
        Assert.Equal(launches, lane.KernelLaunchCount);
        Assert.All(adapter.ReadGradients().SelectMany(values => values), value => Assert.Equal(0f, value));
        Assert.Equal(0d, adapter.GradientSquaredNorm());
        Assert.Equal(downloads, lane.D2HBytes);
        Assert.Equal(launches, lane.KernelLaunchCount);

        // An unused adapter must overwrite only its own range in a shared norm buffer.
        using ArcBuffer norms = lane.Upload(Enumerable.Repeat(41f, 8).ToArray());
        adapter.EnqueueGradientSquaredNorm(norms, 2, 2);
        var actualNorms = new float[8];
        lane.Read(norms, actualNorms);
        Assert.Equal(new float[] { 41, 41, 0, 0, 0, 0, 41, 41 }, actualNorms);

        adapter.Backward(x, z, dy, null, rows);
        float[][] fresh = adapter.ReadGradients();
        Assert.Equal(first[0], fresh[0]);
        Assert.Equal(first[1], fresh[1]);
    }

    [Fact]
    public void UnusedAdapterUpdateUsesZeroGradientAndStillAdvancesAdamMoments()
    {
        using var lane = CreateLane();
        const int input = 13, output = 7, rank = 3, rows = 2;
        var options = new Qwen35LoraOptions
        {
            Rank = rank, Alpha = 6, LearningRate = 0.001f, WeightDecay = 0.1f
        };
        using var adapter = new Qwen35LoraMatrix(lane, input, output, options, new Random(19));
        float[][] initial = State(input, output, rank);
        adapter.RestoreState(initial, training: true);
        using ArcBuffer x = lane.Upload(Values(rows * input, 43));
        using ArcBuffer y = lane.Upload(new float[rows * output]);
        using ArcBuffer dy = lane.Upload(Values(rows * output, 47));
        using ArcBuffer z = adapter.Forward(x, y, rows);
        adapter.Backward(x, z, dy, null, rows);
        Assert.Contains(adapter.ReadGradients().SelectMany(values => values), value => value != 0f);
        adapter.ZeroGrad();
        const int step = 5;
        adapter.Update(options, step, 0.7f);
        float[][] actual = adapter.ReadState();
        float c1 = (float)(1 - Math.Pow(.9, step)), c2 = (float)(1 - Math.Pow(.999, step));
        for (int field = 0; field < 2; field++)
        for (int i = 0; i < initial[field].Length; i++)
        {
            float moment = .9f * initial[field + 2][i];
            float variance = .999f * initial[field + 4][i];
            float expected = initial[field][i] * (1 - options.LearningRate * options.WeightDecay)
                - options.LearningRate * (moment / c1) / (MathF.Sqrt(variance / c2) + 1e-8f);
            Close(expected, actual[field][i]);
            Close(moment, actual[field + 2][i]);
            Close(variance, actual[field + 4][i]);
        }
        Assert.All(adapter.ReadGradients().SelectMany(values => values), value => Assert.Equal(0f, value));
    }

    [Fact]
    public void TrainingLeavesInferenceCachesEmptyWithoutAnAdditionalReset()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        using var fixture = Qwen35ResidentModelTests.CreateFixture(tiedOutput: false);
        using var model = Qwen35QuantizedModel.Load(fixture.Path, [0], options: new() { LoraTraining = true });
        model.AttachLora(new() { Rank = 2, Alpha = 4, IncludeOutput = true });
        model.ForwardToken(1);
        model.ForwardToken(2);
        model.TrainLora([1, 2, 3, 0], 2);
        float[] afterTraining = model.ForwardToken(1);
        model.Reset();
        Assert.Equal(afterTraining, model.ForwardToken(1));
    }

    private static ArcExecutionLane CreateLane()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        return new(0, new()
        {
            Qwen35InferenceKernelsOnly = true, Qwen35TrainingKernels = true,
            Qwen35CooperativeLora = true
        });
    }

    private static float[] Values(int length, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length).Select(_ => (float)(random.NextDouble() * .4 - .2)).ToArray();
    }

    private static float[][] State(int input, int output, int rank)
        => [Values(input * rank, 3), Values(output * rank, 5),
            Values(input * rank, 7), Values(output * rank, 11),
            Values(input * rank, 13).Select(value => MathF.Abs(value) + .01f).ToArray(),
            Values(output * rank, 17).Select(value => MathF.Abs(value) + .01f).ToArray()];

    private static void Close(float expected, float actual)
        => Assert.True(float.IsFinite(actual) && Math.Abs(expected - actual) <= 5e-7 * (1 + Math.Abs(expected)),
            $"Expected {expected:R}, actual {actual:R}.");
}
