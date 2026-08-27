using NNtrain;
using NNtrain.Training.Metrics;
using Parquet;
using Parquet.Schema;
using Xunit;

public sealed class WikiEpochBoundaryTests
{
    [Fact]
    public async Task MaterializedTrainingPersistsMetricsWhenHtmlIsDisabled()
    {
        string directory = Path.Combine(
            Path.GetTempPath(),
            $"NNtrain.WikiMaterializedMetricTests-{Guid.NewGuid():N}");
        string dataDirectory = Path.Combine(directory, "wiki");
        Directory.CreateDirectory(dataDirectory);
        string configurationPath = Path.Combine(directory, "training.json");
        try
        {
            await WriteShard(
                Path.Combine(dataDirectory, "train-00000.parquet"),
                ["日本語の小さな学習文書です。メトリクス永続化を確認します。"]);
            File.WriteAllText(
                configurationPath,
                """
                {
                  "task": "gpt_rin_wiki_jp",
                  "dataPath": "wiki",
                  "textColumn": "text",
                  "tokenizerPath": "tokenizer.json",
                  "checkpointPath": "checkpoint.json",
                  "vocabularySize": 300,
                  "tokenizerTrainingDocuments": 1,
                  "tokenizerTrainingBytes": 10000,
                  "maxTrainingDocuments": 1,
                  "maxTrainingTokens": 20,
                  "validationFraction": 0.0,
                  "epochs": 1,
                  "batchSize": 1,
                  "contextLength": 4,
                  "modelWidth": 4,
                  "heads": 1,
                  "hiddenSize": 8,
                  "layers": 1,
                  "modelArchitecture": "transformer",
                  "device": "cpu",
                  "precisionMode": "float32",
                  "dropout": 0.0,
                  "optimizer": "adamw",
                  "learningRate": 0.001,
                  "auxiliaryLearningRate": 0.001,
                  "warmupPercent": 0,
                  "seed": 1234,
                  "logEveryBatches": 100,
                  "showLossGraph": false,
                  "graphUpdateSteps": 100,
                  "datasetSampleEverySteps": 0,
                  "datasetSamplePoolSize": 1,
                  "maxNewTokens": 1,
                  "temperature": 0.0,
                  "topK": 1
                }
                """);
            using var output = new StringWriter();
            using var error = new StringWriter();

            int exitCode = WikiLanguageModelCommand.Run(
                configurationPath,
                generatePrompt: null,
                output,
                error);

            Assert.True(
                exitCode == 0,
                $"Wiki training failed with exit code {exitCode}: " +
                error + Environment.NewLine + output);
            string graphPath = Path.ChangeExtension(
                configurationPath,
                ".loss.html");
            Assert.False(File.Exists(graphPath));
            MetricJournalLoadResult metrics =
                new MetricJournalJsonlRepository(
                    TrainingMetricReporter.GetSidecarPath(graphPath)).Load();
            Assert.Equal(2, metrics.Journal.Count);
            Assert.Equal(
                [MetricKinds.TrainLoss, MetricKinds.EvaluationLoss],
                metrics.Journal.Entries
                    .Select(entry => entry.Kind)
                    .ToArray());
            Assert.All(
                metrics.Journal.Entries,
                entry => Assert.Equal(1d, entry.Epoch));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task StreamingLossWindowCrossesEpochWithoutFakeEvaluation()
    {
        string directory = Path.Combine(
            Path.GetTempPath(),
            $"NNtrain.WikiEpochBoundaryTests-{Guid.NewGuid():N}");
        string dataDirectory = Path.Combine(directory, "wiki");
        Directory.CreateDirectory(dataDirectory);
        string configurationPath = Path.Combine(directory, "training.json");
        try
        {
            await WriteShard(
                Path.Combine(dataDirectory, "train-00000.parquet"),
                ["日本語の小さな学習文書です。境界テストを行います。"]);
            File.WriteAllText(
                configurationPath,
                """
                {
                  "task": "gpt_rin_wiki_jp",
                  "dataPath": "wiki",
                  "textColumn": "text",
                  "tokenizerPath": "tokenizer.json",
                  "checkpointPath": "checkpoint.json",
                  "vocabularySize": 300,
                  "tokenizerTrainingDocuments": 1,
                  "tokenizerTrainingBytes": 10000,
                  "maxTrainingDocuments": 0,
                  "maxTrainingTokens": 0,
                  "maxDocumentTokens": 8,
                  "shuffleBufferSize": 4,
                  "validationFraction": 0.0,
                  "epochs": 2,
                  "batchSize": 1,
                  "contextLength": 4,
                  "modelWidth": 4,
                  "heads": 1,
                  "hiddenSize": 8,
                  "layers": 1,
                  "modelArchitecture": "transformer",
                  "device": "cpu",
                  "precisionMode": "float32",
                  "dropout": 0.0,
                  "optimizer": "adamw",
                  "learningRate": 0.001,
                  "auxiliaryLearningRate": 0.001,
                  "warmupPercent": 0,
                  "seed": 1234,
                  "logEveryBatches": 100,
                  "showLossGraph": true,
                  "graphUpdateSteps": 100,
                  "datasetSampleEverySteps": 0,
                  "datasetSamplePoolSize": 1,
                  "maxNewTokens": 1,
                  "temperature": 0.0,
                  "topK": 1
                }
                """);
            using var output = new StringWriter();
            using var error = new StringWriter();

            int exitCode = WikiLanguageModelCommand.Run(
                configurationPath,
                generatePrompt: null,
                output,
                error);

            Assert.True(
                exitCode == 0,
                $"Wiki training failed with exit code {exitCode}: " +
                error + Environment.NewLine + output);
            string graphPath = Path.ChangeExtension(
                configurationPath,
                ".loss.html");
            string html = File.ReadAllText(graphPath);
            Assert.Contains("train points 1", html);
            Assert.Contains("eval points 0", html);
            Assert.DoesNotContain("eval loss", html);
            string metricsPath = TrainingMetricReporter.GetSidecarPath(
                graphPath);
            MetricJournalLoadResult metrics =
                new MetricJournalJsonlRepository(metricsPath).Load();
            MetricJournalEntry entry = Assert.Single(
                metrics.Journal.Entries);
            Assert.Equal(MetricKinds.TrainLoss, entry.Kind);
            Assert.Equal(2d, entry.Epoch);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static async Task WriteShard(string path, string[] values)
    {
        var field = new DataField<string>("text");
        var schema = new ParquetSchema(field);
        await using Stream stream = File.Create(path);
        await using ParquetWriter writer =
            await ParquetWriter.CreateAsync(schema, stream);
        using ParquetRowGroupWriter rowGroup = writer.CreateRowGroup();
        await rowGroup.WriteAsync(field, values);
    }
}
