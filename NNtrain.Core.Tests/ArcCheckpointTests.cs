using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcCheckpointTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Float32, 0f)]
    [InlineData(TensorPrecisionMode.Float32, .1f)]
    [InlineData(TensorPrecisionMode.Mix16_32, 0f)]
    [InlineData(TensorPrecisionMode.Mix16_32, .1f)]
    [InlineData(TensorPrecisionMode.Mix8_32, 0f)]
    [InlineData(TensorPrecisionMode.Mix8_32, .1f)]
    public void CheckpointPreservesAccumulationUpdateRngAndResidentTransfers(TensorPrecisionMode precision, float dropout)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        var expected = RunTraining(precision, dropout, false);
        var actual = RunTraining(precision, dropout, true);
        Assert.Equal(expected.NextRandom, actual.NextRandom);
        Close(expected.Losses, actual.Losses);
        for (int index = 0; index < expected.Gradients.Length; index++)
        {
            Close(expected.Gradients[index], actual.Gradients[index]);
            Close(expected.Weights[index], actual.Weights[index]);
        }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void ModelEvalBeforeBackwardPreservesTrainingMasksAndLeavesModelInEval(TensorPrecisionMode precision)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        var expected = RunTraining(precision, .1f, false, evalBeforeBackward: true);
        var actual = RunTraining(precision, .1f, true, evalBeforeBackward: true);
        Assert.Equal(expected.NextRandom, actual.NextRandom);
        Close(expected.Losses, actual.Losses);
        for (int index = 0; index < expected.Gradients.Length; index++)
        {
            Close(expected.Gradients[index], actual.Gradients[index]);
            Close(expected.Weights[index], actual.Weights[index]);
        }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void SwappingDropoutModesWithoutChangingSeedCountCannotChangeReplay(TensorPrecisionMode precision)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        (float[][] Gradients, long NextRandom) Run(bool checkpoint)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision);
            var random = new Random(7);
            var block = new TransformerBlock(8, 2, 13, rng: random, dropout: .1f);
            block.to(precision, 32);
            block.AttnDropout.eval();
            Tensor input = Input();
            input.ConvertStorageInPlace(precision.ToStorageDType(),
                precision == TensorPrecisionMode.Mix8_32 ? Bfp8QuantizationDescriptor.Block(32) : null);
            Tensor[] parameters = block.Parameters().Select(parameter => parameter.T).ToArray();
            Tensor output = checkpoint
                ? input.ArcCheckpoint(block.CaptureArcCheckpointForward(false), parameters)
                : block.Forward(input);
            // Both configurations use one seed. Checking only transcript
            // length would not detect that the mask moved to another branch.
            block.AttnDropout.train();
            block.FfnDropout.eval();
            output.BackwardAndRelease(Enumerable.Range(0, 64).Select(i => MathF.Sin(i * .17f) * .1f).ToArray());
            Assert.True(block.AttnDropout.IsTraining);
            Assert.False(block.FfnDropout.IsTraining);
            return (parameters.Append(input).Select(parameter => parameter.Grad.ToArray()).ToArray(), random.NextInt64());
        }
        var expected = Run(false); var actual = Run(true);
        Assert.Equal(expected.NextRandom, actual.NextRandom);
        for (int parameter = 0; parameter < expected.Gradients.Length; parameter++)
            Assert.Equal(expected.Gradients[parameter], actual.Gradients[parameter]);
    }

    [Fact]
    public void CheckpointCutsRetainedForwardMemoryAndReleasesEveryBlock()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        long Run(bool checkpoint)
        {
            using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_32,
                options: new() { TransformerCheckpointing = checkpoint, BufferPoolBytes = 0 });
            var model = new GptRinWikiJp(129, 128, 128, 4, 384, 6, new Random(57), dropout: .1f, tieWordEmbeddings: true);
            model.to(TensorPrecisionMode.Mix8_32, 32);
            int[] tokens = Enumerable.Range(0, 512).Select(i => i % 129).ToArray();
            int[] targets = tokens.Select(i => (i + 1) % 129).ToArray();
            Tensor loss = model.forward_loss(tokens, targets, 4, 128);
            long parameters = model.parameters().Sum(p => (long)p.T.StorageByteLength);
            long savedForward = Tensor.ArcLane.AllocatedBytes - parameters;
            loss.BackwardAndRelease();
            Assert.Equal(4 + model.parameters().Sum(p => (long)p.T.StorageByteLength + p.T.Numel * 4L),
                Tensor.ArcLane.AllocatedBytes);
            return savedForward;
        }
        long baseline = Run(false), checkpointed = Run(true);
        Assert.True(checkpointed < baseline / 2,
            $"Checkpoint retained {checkpointed:N0} bytes versus {baseline:N0} bytes without checkpointing.");
    }

    [Fact]
    public void ParametersRemainVersionCheckedBeforeRecomputation()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var execution = Tensor.BeginArcExecution();
        var block = new TransformerBlock(8, 2, 13, rng: new Random(7));
        Tensor input = Input();
        int calls = 0;
        Tensor output = input.ArcCheckpoint(x => { calls++; return block.Forward(x); },
            block.Parameters().Select(p => p.T).ToArray());
        block.Ffn.Parameters().First().CompleteUpdate();
        Assert.Throws<InvalidOperationException>(() => output.BackwardAndRelease(new float[64]));
        Assert.Equal(1, calls);
        Assert.False(output.Node.HasLeases);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PartialForwardAndRecomputeFailuresReleaseEveryTemporary(bool duringBackward)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_32);
        var block = new TransformerBlock(8, 2, 13, rng: new Random(7), dropout: .1f);
        block.to(TensorPrecisionMode.Mix8_32, 32);
        Tensor input = Input();
        input.ConvertStorageInPlace(TensorDType.Bfp8, Bfp8QuantizationDescriptor.Block(32));
        float[] seed = Enumerable.Repeat(.125f, 64).ToArray();
        block.Forward(input).BackwardAndRelease(seed);
        long baseline = Tensor.ArcLane.AllocatedBytes;
        int calls = 0;
        Tensor? output = null;
        void Operation()
        {
            output = input.ArcCheckpoint(x =>
            {
                Tensor value = block.Forward(x);
                if (++calls == (duringBackward ? 2 : 1)) throw new InvalidOperationException("injected checkpoint failure");
                return value;
            }, block.Parameters().Select(p => p.T).ToArray());
            output.BackwardAndRelease(seed);
        }
        Assert.Contains("injected checkpoint failure", Assert.Throws<InvalidOperationException>(Operation).Message);
        Assert.Equal(baseline, Tensor.ArcLane.AllocatedBytes);
        if (output is not null) Assert.False(output.Node.HasLeases);
        // Both thread-local scopes must also recover after an exception.
        Tensor recovery = input.ArcCheckpoint(block.Forward, block.Parameters().Select(p => p.T).ToArray());
        recovery.BackwardAndRelease(seed);
        Assert.Equal(baseline, Tensor.ArcLane.AllocatedBytes);
    }

    [Fact]
    public void BackwardInsideNoGradRestoresCallerRecordingState()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var execution = Tensor.BeginArcExecution();
        var block = new TransformerBlock(8, 2, 13, rng: new Random(7), dropout: .1f);
        Tensor output = Input().ArcCheckpoint(block.Forward, block.Parameters().Select(p => p.T).ToArray());
        using (AutogradContext.NoGrad())
        {
            output.BackwardAndRelease(Enumerable.Repeat(.125f, 64).ToArray());
            Assert.False(AutogradContext.IsRecordingEnabled);
        }
        Assert.True(AutogradContext.IsRecordingEnabled);
        Assert.Contains(block.Parameters(), p => p.T.Grad.Any(value => value != 0));
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void PrefixCheckpointPreservesMixedGraphTraining(TensorPrecisionMode precision)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        var expected = RunTraining(precision, .1f, false);
        var actual = RunTraining(precision, .1f, true, checkpointLayers: 1);
        Assert.Equal(expected.NextRandom, actual.NextRandom);
        Close(expected.Losses, actual.Losses);
        for (int index = 0; index < expected.Gradients.Length; index++)
        {
            Close(expected.Gradients[index], actual.Gradients[index]);
            Close(expected.Weights[index], actual.Weights[index]);
        }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32, false)]
    [InlineData(TensorPrecisionMode.Float32, true)]
    [InlineData(TensorPrecisionMode.Mix16_32, false)]
    [InlineData(TensorPrecisionMode.Mix16_32, true)]
    [InlineData(TensorPrecisionMode.Mix8_32, false)]
    [InlineData(TensorPrecisionMode.Mix8_32, true)]
    public void FfnOnlyAndHybridCheckpointPreserveDropoutAccumulationAndUpdates(TensorPrecisionMode precision, bool hybrid)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        var expected = RunTraining(precision, .1f, false);
        // Hybrid: first block is fully checkpointed, second block checkpoints
        // only its FFN. Both forward and replay must suppress nested FFN nodes.
        var actual = RunTraining(precision, .1f, hybrid, checkpointLayers: 1, checkpointFfn: true);
        Assert.Equal(expected.NextRandom, actual.NextRandom);
        Close(expected.Losses, actual.Losses);
        for (int index = 0; index < expected.Gradients.Length; index++)
        {
            Close(expected.Gradients[index], actual.Gradients[index]);
            Close(expected.Weights[index], actual.Weights[index]);
        }
    }

    [Fact]
    public void FfnOnlyCheckpointDropsExpandedActivationWithoutDroppingResidualInput()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        (long Saved, float[] Value, float[][] Gradients) Run(bool checkpoint)
        {
            using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_32,
                options: new() { TransformerFfnCheckpointing = checkpoint });
            var block = new TransformerBlock(128, 4, 384, rng: new Random(7), dropout: .1f);
            block.to(TensorPrecisionMode.Mix8_32, 32);
            var input = new Tensor(Enumerable.Range(0, 512 * 128).Select(i => MathF.Sin(i * .013f)).ToArray(), [4, 128, 128]);
            input.ConvertStorageInPlace(TensorDType.Bfp8, Bfp8QuantizationDescriptor.Block(32));
            Tensor result = block.Forward(input);
            long saved = Tensor.ArcLane.AllocatedBytes;
            float[] values = result.Data.ToArray();
            result.BackwardAndRelease(Enumerable.Repeat(.01f, result.Numel).ToArray());
            return (saved, values, block.Parameters().Select(p => p.T.Grad.ToArray()).Append(input.Grad.ToArray()).ToArray());
        }
        var expected = Run(false); var actual = Run(true);
        // The only discarded saved value is FC1/ReLU's [512,384] BFP8+scale
        // storage. FC2's output and LN1's residual input retain identical bytes.
        const long expandedBytes = 512L * 384 * (32 + 4) / 32;
        Assert.Equal(expandedBytes, expected.Saved - actual.Saved);
        Assert.Equal(expected.Value, actual.Value);
        for (int parameter = 0; parameter < expected.Gradients.Length; parameter++)
            Assert.Equal(expected.Gradients[parameter], actual.Gradients[parameter]);
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32, false)]
    [InlineData(TensorPrecisionMode.Float32, true)]
    [InlineData(TensorPrecisionMode.Mix16_32, false)]
    [InlineData(TensorPrecisionMode.Mix16_32, true)]
    [InlineData(TensorPrecisionMode.Mix8_32, false)]
    [InlineData(TensorPrecisionMode.Mix8_32, true)]
    public void CheckpointAtFirstOrLastBlockPreservesExactGradientsAndMasks(TensorPrecisionMode precision, bool last)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        (float[] Values, float[][] Gradients, long NextRandom) Run(bool checkpoint)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision);
            var random = new Random(7);
            var first = new TransformerBlock(8, 2, 13, rng: random, dropout: .1f);
            var second = new TransformerBlock(8, 2, 13, rng: random, dropout: .1f);
            first.to(precision, 32); second.to(precision, 32);
            Tensor input = Input();
            input.ConvertStorageInPlace(precision.ToStorageDType(),
                precision == TensorPrecisionMode.Mix8_32 ? Bfp8QuantizationDescriptor.Block(32) : null);
            Tensor[] firstParameters = first.Parameters().Select(p => p.T).ToArray();
            Tensor[] lastParameters = second.Parameters().Select(p => p.T).ToArray();
            Tensor middle = checkpoint && !last ? input.ArcCheckpoint(first.Forward, firstParameters) : first.Forward(input);
            Tensor output = checkpoint && last ? middle.ArcCheckpoint(second.Forward, lastParameters) : second.Forward(middle);
            float[] values = output.Data.ToArray();
            output.BackwardAndRelease(Enumerable.Range(0, 64).Select(i => MathF.Sin(i * .17f) * .1f).ToArray());
            return (values, firstParameters.Concat(lastParameters).Append(input).Select(p => p.Grad.ToArray()).ToArray(), random.NextInt64());
        }
        var expected = Run(false); var actual = Run(true);
        Assert.Equal(expected.NextRandom, actual.NextRandom);
        Assert.Equal(expected.Values, actual.Values);
        for (int parameter = 0; parameter < expected.Gradients.Length; parameter++)
            Assert.Equal(expected.Gradients[parameter], actual.Gradients[parameter]);
    }

    [Fact]
    public void NestedCheckpointIsRejectedWithoutLeakingItsPartialGraph()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var execution = Tensor.BeginArcExecution();
        var block = new TransformerBlock(8, 2, 13, rng: new Random(7));
        Tensor input = Input();
        block.Forward(input).BackwardAndRelease(new float[64]);
        long retained = Tensor.ArcLane.AllocatedBytes;
        Tensor[] parameters = block.Parameters().Select(p => p.T).ToArray();
        Assert.Contains("Nested Arc checkpoints", Assert.Throws<InvalidOperationException>(() =>
            input.ArcCheckpoint(x => block.Forward(x).ArcCheckpoint(block.Forward, parameters), parameters)).Message);
        Assert.Equal(retained, Tensor.ArcLane.AllocatedBytes);
    }

    [Fact]
    public void NegativeCheckpointPrefixIsRejectedBeforeDeviceCreation()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ArcExecutionLane(
            options: new() { TransformerCheckpointLayers = -1 }));
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void PackedOutputOwnershipSurvivesTemporaryCleanupAndDiesWithCheckpoint(TensorPrecisionMode precision)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var execution = Tensor.BeginArcExecution(precision: precision);
        var block = new TransformerBlock(8, 2, 13, rng: new Random(7));
        block.to(precision, 32);
        Tensor input = Input();
        input.ConvertStorageInPlace(precision.ToStorageDType(),
            precision == TensorPrecisionMode.Mix8_32 ? Bfp8QuantizationDescriptor.Block(32) : null);
        Tensor normal = block.Forward(input);
        float[] expected = normal.Data.ToArray();
        normal.BackwardAndRelease(new float[64]);
        long baseline = Tensor.ArcLane.AllocatedBytes;
        Tensor? temporary = null;
        ArcXmxStorageOperand? borrowed = null;
        try
        {
            Tensor output = input.ArcCheckpoint(x =>
            {
                temporary = block.Forward(x);
                borrowed = temporary.ArcMatrixOperand();
                return temporary;
            }, block.Parameters().Select(p => p.T).ToArray());
            Assert.NotNull(temporary);
            Assert.NotNull(borrowed);
            // The borrow originates at the temporary's original owned buffer:
            // it would be dead here if we copied and disposed that buffer.
            Assert.True(borrowed.Value.IsAlive);
            if (borrowed.Scales is not null) Assert.True(borrowed.Scales.IsAlive);
            Assert.Throws<InvalidOperationException>(() => temporary.Data.ToArray());
            temporary.ReleaseArcGraph(); temporary.Node.ReleaseGraph();
            Assert.True(borrowed.Value.IsAlive);
            Assert.Equal(expected, output.Data.ToArray());
            output.BackwardAndRelease(new float[64]);
            Assert.False(borrowed.Value.IsAlive);
            if (borrowed.Scales is not null) Assert.False(borrowed.Scales.IsAlive);
            Assert.Equal(baseline, Tensor.ArcLane.AllocatedBytes);
        }
        finally { borrowed?.Dispose(); }
    }

    [Fact]
    public void FailureAfterOutputMoveReleasesBothTransferredBuffersAndAllOtherTemporaries()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_32);
        var block = new TransformerBlock(8, 2, 13, rng: new Random(7));
        block.to(TensorPrecisionMode.Mix8_32, 32);
        Tensor input = Input();
        input.ConvertStorageInPlace(TensorDType.Bfp8, Bfp8QuantizationDescriptor.Block(32));
        block.Forward(input).BackwardAndRelease(new float[64]);
        long baseline = Tensor.ArcLane.AllocatedBytes, download = Tensor.ArcLane.D2HBytes;
        ArcXmxStorageOperand? borrowed = null;
        try
        {
            AggregateException failure = Assert.Throws<AggregateException>(() => input.ArcCheckpoint(x =>
            {
                Tensor temporary = block.Forward(x);
                borrowed = temporary.ArcMatrixOperand();
                // This resource fails during frame cleanup, after its values
                // have moved into the already-created checkpoint result.
                temporary.Node.RegisterResource(new ThrowDuringDispose());
                return temporary;
            }, block.Parameters().Select(p => p.T).ToArray()));
            Assert.Contains(failure.Flatten().InnerExceptions, exception => exception.Message == "injected release failure");
            Assert.NotNull(borrowed);
            Assert.False(borrowed.Value.IsAlive);
            Assert.False(borrowed.Scales!.IsAlive);
            Assert.Equal(baseline, Tensor.ArcLane.AllocatedBytes);
            Assert.Equal(download, Tensor.ArcLane.D2HBytes);
            Tensor recovery = input.ArcCheckpoint(block.Forward, block.Parameters().Select(p => p.T).ToArray());
            recovery.BackwardAndRelease(new float[64]);
            Assert.Equal(baseline, Tensor.ArcLane.AllocatedBytes);
        }
        finally { borrowed?.Dispose(); }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void RepeatedBackwardReplaysMasksAndRetainsOnlyItsOuterGraph(TensorPrecisionMode precision)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        (float[][] Gradients, long NextRandom) Run(bool checkpoint)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision);
            var random = new Random(7);
            var block = new TransformerBlock(8, 2, 13, rng: random, dropout: .1f);
            block.to(precision, 32);
            Tensor input = Input();
            input.ConvertStorageInPlace(precision.ToStorageDType(),
                precision == TensorPrecisionMode.Mix8_32 ? Bfp8QuantizationDescriptor.Block(32) : null);
            Tensor[] parameters = block.Parameters().Select(p => p.T).ToArray();
            Tensor output = checkpoint ? input.ArcCheckpoint(block.Forward, parameters) : block.Forward(input);
            float[] seed = Enumerable.Range(0, 64).Select(i => MathF.Sin(i * .17f) * .1f).ToArray();
            output.Backward(seed);
            long retained = Tensor.ArcLane.AllocatedBytes;
            output.Backward(seed);
            Assert.Equal(retained, Tensor.ArcLane.AllocatedBytes);
            Assert.True(output.Node.HasLeases);
            output.BackwardAndRelease(seed);
            Assert.False(output.Node.HasLeases);
            Assert.True(Tensor.ArcLane.AllocatedBytes < retained);
            return (parameters.Append(input).Select(p => p.Grad.ToArray()).ToArray(), random.NextInt64());
        }
        var expected = Run(false); var actual = Run(true);
        Assert.Equal(expected.NextRandom, actual.NextRandom);
        for (int parameter = 0; parameter < expected.Gradients.Length; parameter++)
            Assert.Equal(expected.Gradients[parameter], actual.Gradients[parameter]);
    }

    private static Snapshot RunTraining(TensorPrecisionMode precision, float dropout, bool checkpoint,
        int checkpointLayers = int.MaxValue, bool checkpointFfn = false, bool evalBeforeBackward = false)
    {
        const int batch = 4, sequence = 128, vocabulary = 129, steps = 2, accumulation = 2;
        using var execution = Tensor.BeginArcExecution(precision: precision,
            options: new() { TransformerCheckpointing = checkpoint, TransformerCheckpointLayers = checkpointLayers,
                TransformerFfnCheckpointing = checkpointFfn, BufferPoolBytes = 64L * 1024 * 1024 });
        ArcExecutionLane lane = Tensor.ArcLane;
        var random = new Random(57);
        var model = new GptRinWikiJp(vocabulary, sequence, 128, 4, 384, 2, random,
            dropout: dropout, tieWordEmbeddings: true);
        model.to(precision, 32);
        Parameter[] parameters = model.parameters().ToArray();
        var optimizer = new AdamW(parameters, new AdamWOptions { LearningRate = .0003f, WeightDecay = .01f });
        var losses = new float[steps * accumulation];
        long retained = -1;
        for (int step = 0; step < steps; step++)
        {
            long upload = lane.H2DBytes, download = lane.D2HBytes;
            optimizer.zero_grad();
            for (int micro = 0; micro < accumulation; micro++)
            {
                int[] tokens = Enumerable.Range(0, batch * sequence).Select(i => (i * 7 + step * 11 + micro * 3) % vocabulary).ToArray();
                int[] targets = tokens.Select((token, i) => i % 17 == 0 ? -1 : (token + 1) % vocabulary).ToArray();
                Tensor loss = model.forward_loss(tokens, targets, batch, sequence);
                losses[step * accumulation + micro] = loss.item();
                if (evalBeforeBackward) model.eval();
                loss.BackwardAndRelease([1f / accumulation]);
                if (evalBeforeBackward)
                {
                    Assert.False(model.IsTraining);
                    model.train();
                }
            }
            optimizer.step();
            lane.Synchronize();
            if (step == 0) retained = lane.AllocatedBytes;
            else
            {
                Assert.Equal(retained, lane.AllocatedBytes);
                Assert.Equal(accumulation * 2L * batch * sequence * sizeof(int), lane.H2DBytes - upload);
                // One scalar loss per microbatch; device copies for checkpoint
                // outputs and backward seeds must never count as host traffic.
                Assert.Equal(accumulation * sizeof(float), lane.D2HBytes - download);
            }
        }
        return new(losses, parameters.Select(p => p.T.Grad.ToArray()).ToArray(),
            parameters.Select(p => p.T.CaptureData(true)).ToArray(), random.NextInt64());
    }

    private static Tensor Input() => new(Enumerable.Range(0, 64).Select(i => MathF.Sin(i * .1f)).ToArray(), [1, 8, 8]);

    private static void Close(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(MathF.Abs(expected[i] - actual[i]), 0, 1e-4f);
    }

    private sealed record Snapshot(float[] Losses, float[][] Gradients, float[][] Weights, long NextRandom);
    private sealed class ThrowDuringDispose : IDisposable
    {
        public void Dispose() => throw new InvalidOperationException("injected release failure");
    }
}
