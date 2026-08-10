using NNtrain;
using Xunit;

public sealed class WikiLanguageModelCommandTests
{
    [Fact]
    public void RunPrintsTheSelectedConfigurationAndEffectiveModelSettings()
    {
        string configurationPath = Path.Combine(
            Path.GetTempPath(),
            $"NNtrain-wiki-settings-{Guid.NewGuid():N}.json");
        File.WriteAllText(
            configurationPath,
            """
            {
              "task": "gpt_rin_wiki_jp",
              "dataPath": "missing-wiki-data",
              "batchSize": 7,
              "contextLength": 12,
              "modelWidth": 12,
              "heads": 3,
              "hiddenSize": 20,
              "layers": 3
            }
            """);
        try
        {
            using var output = new StringWriter();
            using var error = new StringWriter();

            int exitCode = WikiLanguageModelCommand.Run(
                configurationPath,
                generatePrompt: null,
                output,
                error);

            Assert.Equal(2, exitCode);
            Assert.Contains(
                $"configuration = {Path.GetFullPath(configurationPath)}",
                output.ToString());
            Assert.Contains("simd = enabled", output.ToString());
            Assert.Contains(
                "thread parallelism = Parallel.For",
                output.ToString());
            Assert.Contains(
                "effective training = epochs 5, batch 7, context 12",
                output.ToString());
            Assert.Contains(
                "effective model = vocabulary 2048, width 12, heads 3, " +
                "hidden 20, layers 3",
                output.ToString());
            Assert.Contains(
                "Wikipedia data directory was not found",
                error.ToString());
        }
        finally
        {
            File.Delete(configurationPath);
        }
    }

    [Fact]
    public void CreatesNekoMuonWithAuxiliaryAdamWForGpt()
    {
        var model = new GptRinWikiJp(
            BpeTokenizer.BaseVocabularySize,
            contextLength: 4,
            dModel: 8,
            numHeads: 2,
            dHidden: 16,
            numLayers: 1,
            rng: new Random(2));
        var config = new WikiTrainingConfiguration
        {
            Optimizer = "nekomuon",
            LearningRate = 4e-4f,
            AuxiliaryLearningRate = 2e-4f,
        };

        IOptimizer optimizer = WikiLanguageModelCommand.CreateOptimizer(
            model,
            config);

        CompositeOptimizer composite = Assert.IsType<CompositeOptimizer>(
            optimizer);
        Assert.Collection(
            composite.Optimizers,
            child => Assert.IsType<NekoMuon>(child),
            child => Assert.IsType<AdamW>(child));
        Assert.Equal(
            config.LearningRate,
            ((NekoMuon)composite.Optimizers[0]).LearningRate);
    }

    [Fact]
    public void DatasetContinuationSplitsDocumentAndGeneratesFromFirstHalf()
    {
        const string document = "日本の歴史を説明します。ここからが文章の後半です。";
        BpeTokenizer tokenizer = BpeTokenizer.Train(
            Enumerable.Repeat(document, 4),
            vocabularySize: 280);
        var model = new GptRinWikiJp(
            tokenizer.VocabularySize,
            contextLength: 8,
            dModel: 4,
            numHeads: 1,
            dHidden: 8,
            numLayers: 1,
            rng: new Random(3));
        var config = new WikiTrainingConfiguration
        {
            ContextLength = 8,
            ModelWidth = 4,
            Heads = 1,
            HiddenSize = 8,
            Layers = 1,
            MaxNewTokens = 2,
            Temperature = 0f,
            TopK = 1,
        };

        WikiLanguageModelCommand.DatasetContinuation result =
            WikiLanguageModelCommand.CreateDatasetContinuation(
                model,
                tokenizer,
                [document],
                config,
                new Random(5));

        Assert.Equal(document.Length, result.DocumentLength);
        Assert.Equal(document.Length / 2, result.SplitIndex);
        Assert.Equal(document[..result.SplitIndex], result.PromptTail);
        Assert.StartsWith(
            document[result.SplitIndex..],
            result.ExpectedContinuation);
        Assert.NotNull(result.GeneratedContinuation);
        Assert.True(model.IsTraining);
    }
}
