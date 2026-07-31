namespace NNtrain;

sealed record TrainingComponents(
    Trainer Trainer,
    TransformerClassifier Model,
    AdamW Optimizer);

static class TrainingComposition
{
    public static TrainingComponents Create(
        TrainingConfiguration configuration)
    {
        ArgumentNullException.ThrowIfNull(configuration);
        configuration.Validate();

        EnsureDataFile(
            configuration.TrainingData.ImagePath,
            "Training image");
        EnsureDataFile(
            configuration.TrainingData.LabelPath,
            "Training label");
        EnsureDataFile(
            configuration.EvaluationData.ImagePath,
            "Evaluation image");
        EnsureDataFile(
            configuration.EvaluationData.LabelPath,
            "Evaluation label");

        IImageClassificationDataset trainingDataset = new Mnist(
            configuration.TrainingData.ImagePath,
            configuration.TrainingData.LabelPath);
        IImageClassificationDataset evaluationDataset = new Mnist(
            configuration.EvaluationData.ImagePath,
            configuration.EvaluationData.LabelPath);
        var model = new TransformerClassifier(
            seqLen: trainingDataset.Rows,
            dModel: trainingDataset.Columns,
            numHeads: configuration.Model.Heads,
            dHidden: configuration.Model.HiddenSize,
            numLayers: configuration.Model.Layers,
            numClasses: trainingDataset.ClassCount,
            rng: new Random(configuration.Model.Seed),
            initScale: configuration.Model.InitializationScale);
        var optimizer = new AdamW(
            model.Parameters(),
            new AdamWOptions
            {
                LearningRate = configuration.LearningRate,
            });
        var trainer = new Trainer(
            model,
            trainingDataset,
            evaluationDataset,
            optimizer,
            new TrainerOptions
            {
                Epochs = configuration.Epochs,
                StepsPerEpoch = configuration.StepsPerEpoch,
                RandomSeed = configuration.Seed,
            });

        return new TrainingComponents(trainer, model, optimizer);
    }

    private static void EnsureDataFile(string path, string role)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                $"{role} data file was not found at '{path}'. " +
                "Check the corresponding path in the training " +
                "configuration.",
                path);
        }
    }
}
