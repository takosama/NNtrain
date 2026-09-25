using NNtrain;
using NNtrain.Arc;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class ArcDataParallelEngineTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Float32, 0.0001f, false)]
    [InlineData(TensorPrecisionMode.Mix16_32, 0.0001f, false)]
    [InlineData(TensorPrecisionMode.Mix8_32, 0.0001f, false)]
    // Each replica rounds its completed BF16 gradient before the cross-GPU sum.
    [InlineData(TensorPrecisionMode.Mix8_16, 0.005f, false)]
    [InlineData(TensorPrecisionMode.Mix8_16, 0.005f, true)]
    public void TwoGpuReducedGradientsMatchOneGpu(
        TensorPrecisionMode precision, float relativeRmsTolerance, bool pipeline)
    {
        RequireTwoArcDevices();
        using var execution = Tensor.BeginArcInferenceExecution([0, 1], precision,
            new ArcExecutionOptions { Mix8_16PipelinedGradientReduction = pipeline });
        GptRinWikiJp reference = NewModel(precision, 0, seed: 71);
        GptRinWikiJp primary = NewModel(precision, 0, seed: 71);
        GptRinWikiJp secondary = NewModel(precision, 1, seed: 97);
        using var parallel = new ArcDataParallelEngine(primary, secondary, [0, 1]);
        ArcLanguageModelMicroBatch[] batches =
        [
            Batch(batchSize: 4, sequenceLength: 4, offset: 0),
            Batch(batchSize: 4, sequenceLength: 4, offset: 17),
        ];

        float expectedLoss = RunSingle(reference, batches);
        long secondaryKernelsBefore = Lane(1).KernelLaunchCount;
        float actualLoss = parallel.ForwardBackwardAccumulated(
            batches, Tensor.DefaultCrossEntropyIgnoreIndex, globalStep: 0);
        Assert.Equal([2, 2], parallel.LastShardBatchSizes.ToArray());
        Assert.True(Lane(1).KernelLaunchCount > secondaryKernelsBefore);
        Assert.InRange(MathF.Abs(actualLoss - expectedLoss), 0f, 0.0001f);
        AssertReducedGradients(reference, primary, relativeRmsTolerance);
    }

    [Fact]
    public void UnevenAndIgnoredTargetsKeepGlobalLossAndGradientWeights()
    {
        RequireTwoArcDevices();
        using var execution = Tensor.BeginArcInferenceExecution([0, 1]);
        GptRinWikiJp reference = NewModel(TensorPrecisionMode.Float32, 0, seed: 83);
        GptRinWikiJp primary = NewModel(TensorPrecisionMode.Float32, 0, seed: 83);
        GptRinWikiJp secondary = NewModel(TensorPrecisionMode.Float32, 1, seed: 101);
        using var parallel = new ArcDataParallelEngine(primary, secondary, [0, 1]);
        ArcLanguageModelMicroBatch first = Batch(3, 3, 0);
        first.Target[0] = -1;
        first.Target[3] = -1;
        first.Target[4] = -1;
        ArcLanguageModelMicroBatch second = Batch(1, 3, 13);
        second.Target[1] = -1;
        ArcLanguageModelMicroBatch[] batches = [first, second];

        float expectedLoss = RunSingle(reference, batches);
        long secondaryKernelsBefore = Lane(1).KernelLaunchCount;
        float actualLoss = parallel.ForwardBackwardAccumulated(
            batches, Tensor.DefaultCrossEntropyIgnoreIndex, globalStep: 0);

        Assert.Equal([1, 0], parallel.LastShardBatchSizes.ToArray());
        Assert.True(Lane(1).KernelLaunchCount > secondaryKernelsBefore);
        Assert.InRange(MathF.Abs(actualLoss - expectedLoss), 0f, 0.0001f);
        AssertReducedGradients(reference, primary, relativeRmsTolerance: 0.0001f);
    }

    [Fact]
    public void OptimizerUpdateSynchronizesSecondaryReplicaIncludingMixed8Master()
    {
        RequireTwoArcDevices();
        using var execution = Tensor.BeginArcInferenceExecution(
            [0, 1], TensorPrecisionMode.Mix8_32);
        GptRinWikiJp primary = NewModel(TensorPrecisionMode.Mix8_32, 0, seed: 109);
        GptRinWikiJp secondary = NewModel(TensorPrecisionMode.Mix8_32, 1, seed: 137);
        using var parallel = new ArcDataParallelEngine(primary, secondary, [0, 1]);
        var optimizer = new AdamW(primary.parameters(), new AdamWOptions
        {
            LearningRate = 0.001f,
            WeightDecay = 0f,
        });
        optimizer.zero_grad();
        _ = parallel.ForwardBackwardAccumulated(
            [Batch(4, 4, 5)], Tensor.DefaultCrossEntropyIgnoreIndex, globalStep: 0);
        optimizer.step();

        ModuleState updated = primary.state_dict();
        ModuleState stale = secondary.state_dict();
        Assert.True(AnyWeightDifferent(updated, stale));
        parallel.SynchronizeSecondaryModel();
        ModuleState synchronized = secondary.state_dict();
        Assert.False(AnyWeightDifferent(updated, synchronized));
        Parameter[] primaryParameters = primary.parameters().ToArray();
        Parameter[] secondaryParameters = secondary.parameters().ToArray();
        for (int index = 0; index < primaryParameters.Length; index++)
            Assert.Equal(primaryParameters[index].T.Data.ToArray(),
                secondaryParameters[index].T.Data.ToArray());
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    [InlineData(TensorPrecisionMode.Mix8_16)]
    public void PackedForwardSynchronizationMatchesFullStateOnNextUpdate(
        TensorPrecisionMode precision)
    {
        RequireTwoArcDevices();
        using var execution = Tensor.BeginArcInferenceExecution([0, 1], precision);
        GptRinWikiJp primary = NewModel(precision, 0, seed: 239);
        GptRinWikiJp packedSecondary = NewModel(precision, 1, seed: 241);
        GptRinWikiJp fullSecondary = NewModel(precision, 1, seed: 251);
        using var packed = new ArcDataParallelEngine(primary, packedSecondary, [0, 1]);
        using var full = new ArcDataParallelEngine(primary, fullSecondary, [0, 1]);
        var optimizer = new AdamW(primary.parameters(), new AdamWOptions
        {
            LearningRate = 0.001f,
            WeightDecay = 0f,
        });
        optimizer.zero_grad();
        _ = packed.ForwardBackwardAccumulated(
            [Batch(4, 4, 7)], Tensor.DefaultCrossEntropyIgnoreIndex, globalStep: 0);
        optimizer.step();

        packed.SynchronizeSecondaryForwardReplica();
        full.SynchronizeSecondaryModel();
        for (int step = 1; step <= 2; step++)
        {
            ArcLanguageModelMicroBatch[] batch = [Batch(4, 4, 23 + step * 7)];
            optimizer.zero_grad();
            float packedLoss = packed.ForwardBackwardAccumulated(
                batch, Tensor.DefaultCrossEntropyIgnoreIndex, globalStep: step);
            float[][] packedGradients = primary.parameters()
                .Select(parameter => parameter.T.Grad.ToArray()).ToArray();
            optimizer.zero_grad();
            float fullLoss = full.ForwardBackwardAccumulated(
                batch, Tensor.DefaultCrossEntropyIgnoreIndex, globalStep: step);
            float[][] fullGradients = primary.parameters()
                .Select(parameter => parameter.T.Grad.ToArray()).ToArray();

            Assert.InRange(MathF.Abs(packedLoss - fullLoss), 0f, 1e-5f);
            Assert.Equal(fullGradients.Length, packedGradients.Length);
            for (int parameter = 0; parameter < fullGradients.Length; parameter++)
            {
                Assert.Equal(fullGradients[parameter].Length, packedGradients[parameter].Length);
                for (int index = 0; index < fullGradients[parameter].Length; index++)
                    Assert.InRange(MathF.Abs(packedGradients[parameter][index]
                        - fullGradients[parameter][index]), 0f, 1e-5f);
            }
            if (step == 2) continue;
            optimizer.step();
            packed.SynchronizeSecondaryForwardReplica();
            full.SynchronizeSecondaryModel();
        }
    }

    [Fact]
    public void DropoutStepAfterCheckpointMatchesUninterruptedTwoGpuTraining()
    {
        RequireTwoArcDevices();
        using var execution = Tensor.BeginArcInferenceExecution(
            [0, 1], TensorPrecisionMode.Mix8_32);
        GptRinWikiJp uninterruptedPrimary = NewModel(
            TensorPrecisionMode.Mix8_32, 0, seed: 173,
            dropout: 0.1f, checkpointableRandom: true);
        GptRinWikiJp uninterruptedSecondary = NewModel(
            TensorPrecisionMode.Mix8_32, 1, seed: 197,
            dropout: 0.1f, checkpointableRandom: true);
        ArcLanguageModelMicroBatch[] firstBatch = [Batch(4, 4, 3)];
        ArcLanguageModelMicroBatch[] secondBatch = [Batch(4, 4, 19)];
        ModuleState checkpointState;
        TrainingRandomState checkpointRandom;
        float expectedLoss;
        float[][] expectedGradients;

        using (var parallel = new ArcDataParallelEngine(
            uninterruptedPrimary, uninterruptedSecondary, [0, 1]))
        {
            var optimizer = new AdamW(
                uninterruptedPrimary.parameters(),
                new AdamWOptions { LearningRate = 0.001f, WeightDecay = 0f });
            optimizer.zero_grad();
            _ = parallel.ForwardBackwardAccumulated(
                firstBatch, Tensor.DefaultCrossEntropyIgnoreIndex, globalStep: 0);
            optimizer.step();
            parallel.SynchronizeSecondaryModel();
            checkpointState = uninterruptedPrimary.state_dict();
            checkpointRandom = Assert.IsType<TrainingRandomState>(
                uninterruptedPrimary.CaptureTrainingRandomState());

            optimizer.zero_grad();
            expectedLoss = parallel.ForwardBackwardAccumulated(
                secondBatch, Tensor.DefaultCrossEntropyIgnoreIndex, globalStep: 1);
            expectedGradients = uninterruptedPrimary.parameters()
                .Select(parameter => parameter.T.Grad.ToArray()).ToArray();
        }

        GptRinWikiJp resumedPrimary = NewModel(
            TensorPrecisionMode.Mix8_32, 0, seed: 173,
            dropout: 0.1f, checkpointableRandom: true);
        GptRinWikiJp resumedSecondary = NewModel(
            TensorPrecisionMode.Mix8_32, 1, seed: 211,
            dropout: 0.1f, checkpointableRandom: true);
        resumedPrimary.load_state_dict(checkpointState);
        resumedPrimary.RestoreTrainingRandomState(checkpointRandom);
        using var resumed = new ArcDataParallelEngine(
            resumedPrimary, resumedSecondary, [0, 1]);
        foreach (Parameter parameter in resumedPrimary.parameters())
            parameter.ZeroGrad();
        float actualLoss = resumed.ForwardBackwardAccumulated(
            secondBatch, Tensor.DefaultCrossEntropyIgnoreIndex, globalStep: 1);
        float[][] actualGradients = resumedPrimary.parameters()
            .Select(parameter => parameter.T.Grad.ToArray()).ToArray();

        Assert.InRange(MathF.Abs(actualLoss - expectedLoss), 0f, 1e-6f);
        Assert.Equal(expectedGradients.Length, actualGradients.Length);
        for (int parameter = 0; parameter < expectedGradients.Length; parameter++)
        {
            Assert.Equal(expectedGradients[parameter].Length, actualGradients[parameter].Length);
            for (int index = 0; index < expectedGradients[parameter].Length; index++)
            {
                Assert.InRange(
                    MathF.Abs(actualGradients[parameter][index]
                        - expectedGradients[parameter][index]),
                    0f, 1e-5f);
            }
        }
    }

    private static GptRinWikiJp NewModel(
        TensorPrecisionMode precision, int device, int seed,
        float dropout = 0f, bool checkpointableRandom = false)
    {
        using var selected = TensorExecutionContext.Push(
            new TorchDevice(TensorDevice.Arc, device));
        Random random = checkpointableRandom
            ? new CheckpointableRandom(seed)
            : new Random(seed);
        var model = new GptRinWikiJp(32, 4, 8, 2, 16, 1,
            random, dropout: dropout, tieWordEmbeddings: true);
        if (random is CheckpointableRandom trainingRandom)
        {
            trainingRandom.BeginRuntime();
            model.AttachTrainingRandom(trainingRandom);
        }
        model.to(precision, 32);
        model.to(new TorchDevice(TensorDevice.Arc, device));
        model.train();
        return model;
    }

    private static ArcLanguageModelMicroBatch Batch(
        int batchSize, int sequenceLength, int offset)
    {
        int[] tokens = Enumerable.Range(0, batchSize * sequenceLength)
            .Select(index => 2 + (index * 7 + offset) % 29).ToArray();
        int[] targets = tokens.Select(token => (token + 1) % 32).ToArray();
        return new(tokens, targets, batchSize, sequenceLength);
    }

    private static float RunSingle(
        GptRinWikiJp model, IReadOnlyList<ArcLanguageModelMicroBatch> batches)
    {
        using var selected = TensorExecutionContext.Push(
            new TorchDevice(TensorDevice.Arc, 0));
        Parameter[] parameters = model.parameters().ToArray();
        foreach (Parameter parameter in parameters) parameter.ZeroGrad();
        int validCount = batches.Sum(batch => batch.Target.Count(
            token => token != Tensor.DefaultCrossEntropyIgnoreIndex));
        double weightedLoss = 0d;
        foreach (ArcLanguageModelMicroBatch batch in batches)
        {
            int localValid = batch.Target.Count(
                token => token != Tensor.DefaultCrossEntropyIgnoreIndex);
            Tensor loss = model.forward_loss(batch.Input, batch.Target,
                batch.BatchSize, batch.SequenceLength);
            float value = loss.item();
            loss.BackwardAndRelease([(float)localValid / validCount]);
            weightedLoss += value * localValid;
        }
        return (float)(weightedLoss / validCount);
    }

    private static void AssertReducedGradients(
        GptRinWikiJp expected, GptRinWikiJp actual, float relativeRmsTolerance)
    {
        Parameter[] expectedParameters = expected.parameters().ToArray();
        Parameter[] actualParameters = actual.parameters().ToArray();
        Assert.Equal(expectedParameters.Length, actualParameters.Length);
        double squaredError = 0d, squaredReference = 0d;
        int count = 0;
        for (int parameter = 0; parameter < expectedParameters.Length; parameter++)
        {
            float[] reference = expectedParameters[parameter].T.Grad.ToArray();
            float[] reduced = actualParameters[parameter].T.Grad.ToArray();
            Assert.Equal(reference.Length, reduced.Length);
            for (int index = 0; index < reference.Length; index++)
            {
                Assert.True(float.IsFinite(reference[index]));
                Assert.True(float.IsFinite(reduced[index]));
                double difference = reference[index] - reduced[index];
                squaredError += difference * difference;
                squaredReference += reference[index] * reference[index];
                count++;
            }
        }
        Assert.True(count > 0 && squaredReference > 0d);
        double relativeRms = Math.Sqrt(squaredError / squaredReference);
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"Arc data-parallel gradient relative RMS = {relativeRms:G6}");
        Assert.InRange(relativeRms, 0d, relativeRmsTolerance);
    }

    private static bool AnyWeightDifferent(ModuleState first, ModuleState second)
    {
        Assert.Equal(first.Parameters.Length, second.Parameters.Length);
        return first.Parameters.Zip(second.Parameters)
            .Any(pair => !pair.First.Values.SequenceEqual(pair.Second.Values));
    }

    private static ArcExecutionLane Lane(int index)
        => (ArcExecutionLane)ExecutionSession.Current!.GetRequiredLane(
            ExecutionDeviceKind.Arc, index);

    private static void RequireTwoArcDevices()
        => Assert.SkipWhen(!Tensor.IsArcAvailable(0) || !Tensor.IsArcAvailable(1),
            "Two Intel Arc OpenCL GPUs are required.");
}
