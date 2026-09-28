using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35LoraCheckpointParityTests
{
    [Theory]
    [InlineData(1, true, 1)]
    [InlineData(3, true, 1)]
    [InlineData(3, false, 1)]
    [InlineData(3, true, 2)]
    public void RecomputedLayersMatchFullTapeLossGradientsAndUpdate(
        int responseStart, bool responseOnlyHead, int deviceCount)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count < deviceCount, "Required Intel Arc GPUs are unavailable.");
        using TemporaryQwenGguf fixture = Qwen35ResidentModelTests.CreateFixture(tiedOutput: false);
        int[] devices = Enumerable.Range(0, deviceCount).ToArray();
        var execution = new Qwen35ExecutionOptions
        {
            LoraTraining = true,
            TrainingResponseOnlyHead = responseOnlyHead
        };
        using Qwen35QuantizedModel reference = Qwen35QuantizedModel.Load(fixture.Path, devices, options: execution);
        using Qwen35QuantizedModel recomputed = Qwen35QuantizedModel.Load(fixture.Path, devices, options: execution);
        var adapterOptions = new Qwen35LoraOptions
        {
            Rank = 2, Alpha = 4, IncludeOutput = true, Seed = 94, LearningRate = .001f
        };
        reference.AttachLora(adapterOptions);
        recomputed.AttachLora(adapterOptions);
        recomputed.LoraCheckpointThresholdRows = 0;
        recomputed.LoraTransposeScratchBudgetBytes = 64;

        var random = new Random(821);
        foreach (var pair in reference.LoraMatrices)
        {
            float[][] state = pair.Value.ReadState();
            foreach (float[] values in state.Take(2))
                for (int i = 0; i < values.Length; i++)
                    values[i] = (float)(random.NextDouble() * .04 - .02);
            pair.Value.RestoreState(state, training: true);
            recomputed.LoraMatrices[pair.Key].RestoreState(state, training: true);
        }

        int[] tokens = [1, 2, 3, 0, 1, 2];
        Qwen35LoraStepResult baseline = reference.ComputeLoraGradients(tokens, responseStart);
        Qwen35LoraStepResult checkpointed = recomputed.ComputeLoraGradients(tokens, responseStart);
        Close(baseline.Loss, checkpointed.Loss);
        Close(baseline.GradientNorm, checkpointed.GradientNorm);
        Assert.Equal(baseline.SupervisedTokens, checkpointed.SupervisedTokens);
        foreach (var pair in reference.LoraMatrices)
        {
            float[][] a = pair.Value.ReadGradients();
            float[][] b = recomputed.LoraMatrices[pair.Key].ReadGradients();
            for (int field = 0; field < a.Length; field++)
                for (int i = 0; i < a[field].Length; i++) Close(a[field][i], b[field][i]);
        }

        Qwen35LoraStepResult baselineStep = reference.TrainLora(tokens, responseStart);
        Qwen35LoraStepResult checkpointedStep = recomputed.TrainLora(tokens, responseStart);
        Assert.Equal(1, baselineStep.Step);
        Assert.Equal(baselineStep.Step, checkpointedStep.Step);
        Close(baselineStep.Loss, checkpointedStep.Loss);
        Close(baselineStep.GradientNorm, checkpointedStep.GradientNorm);
        foreach (var pair in reference.LoraMatrices)
        {
            float[][] a = pair.Value.ReadState();
            float[][] b = recomputed.LoraMatrices[pair.Key].ReadState();
            for (int field = 0; field < a.Length; field++)
                for (int i = 0; i < a[field].Length; i++) Close(a[field][i], b[field][i]);
        }
    }

    private static void Close(double expected, double actual)
        => Assert.True(double.IsFinite(actual)
            && Math.Abs(actual - expected) <= 2e-5 * (1 + Math.Abs(expected)),
            $"Expected {expected:R}, actual {actual:R}.");
}
