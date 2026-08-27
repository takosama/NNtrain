using NNtrain;
using Xunit;

public sealed class CheckpointOptimizerStateStreamTests
{
    [Fact]
    public void LegacyCompositeMissingLeafFailsAfterPartialRestore()
    {
        string checkpointPath = Path.Combine(
            Path.GetTempPath(),
            $"nntrain-partial-optimizer-{Guid.NewGuid():N}.json");
        try
        {
            Parameter sourceFirstParameter = CreateParameter("first");
            var sourceFirst = new AdamW(
                [sourceFirstParameter],
                new AdamWOptions { WeightDecay = 0f });
            var partialState = new OptimizerStateDictionary(
                "CompositeOptimizer",
                StateJson: null,
                [sourceFirst.state_dict()]);
            var checkpoint = new WikiLanguageModelCommand.WikiModelCheckpoint(
                FormatVersion: 6,
                Epoch: 1,
                ValidationLoss: 1f,
                VocabularySize: 2,
                ContextLength: 1,
                ModelWidth: 1,
                Heads: 1,
                HiddenSize: 1,
                Layers: 1,
                Dropout: 0f,
                InitializationScale: 1f,
                Model: new ModuleState(ModuleState.CurrentFormatVersion, []),
                Optimizer: partialState);
            torch.save(checkpoint, checkpointPath);

            var target = new CompositeOptimizer(
                new AdamW([CreateParameter("first")]),
                new AdamW([CreateParameter("second")]));

            InvalidDataException exception = Assert.Throws<InvalidDataException>(
                () => CheckpointOptimizerStateStream.TryLoad(
                    checkpointPath,
                    checkpoint,
                    target,
                    TextWriter.Null));

            Assert.Contains("partially restored", exception.Message);
            Assert.Contains("1 of 2", exception.Message);
        }
        finally
        {
            if (File.Exists(checkpointPath))
                File.Delete(checkpointPath);
        }
    }

    private static Parameter CreateParameter(string name)
        => new(
            [1f],
            [1],
            name,
            WeightDecayPolicy.Exclude);
}
