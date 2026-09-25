using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMix8_16BackwardStorageTests
{
    [Fact]
    public void FusedAccumulationPreservesCheckpointRecomputationGradients()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");

        (float[] Losses, float[][] Gradients) Run(bool checkpoint)
        {
            using var execution = Tensor.BeginArcExecution(
                precision: TensorPrecisionMode.Mix8_16,
                options: new ArcExecutionOptions
                {
                    TransformerCheckpointing = checkpoint,
                    TransformerCheckpointLayers = 1,
                    AutomaticTransformerMemoryPlan = false,
                    Mix8_16FusedGradientAccumulation = true,
                });
            var model = new GptRinWikiJp(32, 8, 8, 2, 16, 2, new Random(43),
                dropout: 0f, tieWordEmbeddings: true);
            model.to(TensorPrecisionMode.Mix8_16, 32);
            model.train();
            var losses = new float[2];
            for (int micro = 0; micro < 2; micro++)
            {
                int[] tokens = Enumerable.Range(0, 8)
                    .Select(i => 2 + (i * 5 + micro * 7) % 29).ToArray();
                int[] targets = tokens.Select(value => (value + 1) % 32).ToArray();
                Tensor loss = model.forward_loss(tokens, targets, 1, 8);
                losses[micro] = loss.item();
                loss.BackwardAndRelease();
                Assert.All(model.parameters().Where(parameter => parameter.T.HasGradientBuffer),
                    parameter => Assert.Equal(0, parameter.T.ArcGradientResidentBytes.FloatBytes));
            }
            return (losses, model.parameters()
                .Select(parameter => parameter.T.Grad.ToArray()).ToArray());
        }

        var baseline = Run(false);
        var checkpointed = Run(true);
        Assert.Equal(baseline.Losses, checkpointed.Losses);
        double errorSquared = 0, referenceSquared = 0;
        for (int parameter = 0; parameter < baseline.Gradients.Length; parameter++)
        {
            Assert.Equal(baseline.Gradients[parameter].Length,
                checkpointed.Gradients[parameter].Length);
            for (int i = 0; i < baseline.Gradients[parameter].Length; i++)
            {
                double reference = baseline.Gradients[parameter][i];
                double error = checkpointed.Gradients[parameter][i] - reference;
                errorSquared += error * error;
                referenceSquared += reference * reference;
            }
        }
        Assert.InRange(Math.Sqrt(errorSquared / referenceSquared), 0d, 0.002d);
    }

    [Fact]
    public void FusedMicrobatchAccumulationMatchesUnpackPathAndUsesFusedKernel()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");

        (float[][] Gradients, HashSet<string> Kernels) Run(bool fused)
        {
            using var execution = Tensor.BeginArcExecution(
                precision: TensorPrecisionMode.Mix8_16,
                options: new ArcExecutionOptions
                {
                    Mix8_16FusedGradientAccumulation = fused,
                    DetailedProfiling = true,
                });
            var model = new GptRinWikiJp(32, 4, 8, 2, 16, 1, new Random(17),
                dropout: 0f, tieWordEmbeddings: true);
            model.to(TensorPrecisionMode.Mix8_16, 32);
            model.train();
            Backward(3);
            Backward(11);
            Tensor.ArcLane.Synchronize();
            return (model.parameters().Select(parameter => parameter.T.Grad.ToArray()).ToArray(),
                Tensor.ArcLane.KernelTimings.Keys.ToHashSet());

            void Backward(int offset)
            {
                int[] tokens = Enumerable.Range(0, 8)
                    .Select(i => 2 + (i * 3 + offset) % 29).ToArray();
                int[] targets = tokens.Select(value => (value + 1) % 32).ToArray();
                Tensor loss = model.forward_loss(tokens, targets, 2, 4);
                loss.BackwardAndRelease();
            }
        }

        var oldPath = Run(false);
        var fusedPath = Run(true);
        Assert.Equal(oldPath.Gradients.Length, fusedPath.Gradients.Length);
        double errorSquared = 0, referenceSquared = 0;
        for (int parameter = 0; parameter < oldPath.Gradients.Length; parameter++)
        {
            Assert.Equal(oldPath.Gradients[parameter].Length, fusedPath.Gradients[parameter].Length);
            for (int i = 0; i < oldPath.Gradients[parameter].Length; i++)
            {
                double reference = oldPath.Gradients[parameter][i];
                double error = fusedPath.Gradients[parameter][i] - reference;
                errorSquared += error * error;
                referenceSquared += reference * reference;
            }
        }
        double relativeRms = Math.Sqrt(errorSquared / referenceSquared);
        Assert.InRange(relativeRms, 0d, 0.002d);
        Assert.DoesNotContain("add_float_to_bf16_state", oldPath.Kernels);
        Assert.Contains("add_float_to_bf16_state", fusedPath.Kernels);
    }

    [Fact]
    public void TransformerRetainsLeafGradientsAsPhysicalBFloat16AcrossMicrobatches()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc GPU is required.");
        using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_16);
        var model = new GptRinWikiJp(32, 4, 8, 2, 16, 1, new Random(17),
            dropout: 0f, tieWordEmbeddings: true);
        model.to(TensorPrecisionMode.Mix8_16, 32);
        model.train();
        Parameter[] parameters = model.parameters().ToArray();

        RunBackward(3);
        float[][] first = parameters.Select(parameter => parameter.T.Grad.ToArray()).ToArray();
        AssertPacked();

        RunBackward(11);
        AssertPacked();
        Assert.Contains(parameters.Select((parameter, index) =>
            parameter.T.Grad.ToArray().Zip(first[index]).Any(pair => pair.First != pair.Second)),
            changed => changed);

        void RunBackward(int offset)
        {
            int[] tokens = Enumerable.Range(0, 8).Select(i => 2 + (i * 3 + offset) % 29).ToArray();
            int[] targets = tokens.Select(value => (value + 1) % 32).ToArray();
            Tensor loss = model.forward_loss(tokens, targets, 2, 4);
            loss.BackwardAndRelease();
        }

        void AssertPacked()
        {
            Tensor[] withGradients = parameters.Select(parameter => parameter.T)
                .Where(tensor => tensor.HasGradientBuffer).ToArray();
            Assert.NotEmpty(withGradients);
            foreach (Tensor tensor in withGradients)
            {
                var (floatBytes, bfloat16Bytes) = tensor.ArcGradientResidentBytes;
                Assert.Equal(0, floatBytes);
                Assert.Equal(tensor.Numel * sizeof(ushort), bfloat16Bytes);
            }
        }
    }
}
