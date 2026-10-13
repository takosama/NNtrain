namespace NNtrain;

internal static class ClassificationTrainingConsole
{
    internal static void WriteModel(
        TransformerClassifier model,
        CanonicalClassificationTrainingSpec classification,
        TextWriter output)
    {
        output.WriteLine(
            $"model = {model.GetType().Name} (image classifier)");
        output.WriteLine(
            $"precision = " +
            $"{TensorPrecisionModeNames.Format(classification.PrecisionMode)}" +
            (classification.PrecisionMode is TensorPrecisionMode.Bfp8
                ? " (tensor scale)"
                : classification.PrecisionMode == TensorPrecisionMode.Mix8_32
                    ? $" (block {classification.Bfp8BlockSize})"
                    : string.Empty));
    }

    internal static void WriteSettings(
        TrainingConfiguration config,
        IImageClassificationDataset trainData,
        TextWriter output)
    {
        output.WriteLine(
            $"workers = {Environment.ProcessorCount}");
        output.WriteLine(
            $"simd = {GetSimdStatus()}");
        output.WriteLine(
            $"micro batch = {config.ResolvedMicroBatchSize} samples x " +
            $"{config.MicroBatchCount} accumulation(s), effective " +
            $"batch {config.EffectiveBatchSize}");
        output.WriteLine(
            $"optimizer = {config.Optimizer.ToLowerInvariant()}");
        output.WriteLine(
            $"learning rate = {config.LearningRate:F6}");
        if (config.IsOptimizer(TrainingConfiguration.MuonOptimizer))
        {
            output.WriteLine(
                "muon = momentum 0.95, Nesterov, fixed NS5 " +
                "every step");
        }
        if (config.IsOptimizer(
                TrainingConfiguration.MuonOptimizer)
            || config.IsOptimizer(
                TrainingConfiguration.NekoMuonOptimizer))
        {
            output.WriteLine(
                "auxiliary optimizer = adamw");
            output.WriteLine(
                "auxiliary learning rate = " +
                $"{config.AuxiliaryLearningRate:F6}");
        }
        output.WriteLine(
            $"label smoothing = {config.LabelSmoothing:F3}");
        output.WriteLine(
            $"weight decay = {config.WeightDecay:F6}");
        if (config.IsOptimizer(
            TrainingConfiguration.GainShareAdamWOptimizer))
        {
            output.WriteLine(
                $"gainshare = block depth " +
                $"{config.GainShareBlockDepth}, " +
                $"betas ({config.GainShareBeta1:F3}, " +
                $"{config.GainShareBeta2:F3}), " +
                $"eps {config.GainShareEpsilon:E1}, " +
                $"rho {config.GainShareRho:F3}, " +
                $"gamma {config.GainShareGamma:F3}, " +
                $"scale [{config.GainShareMinScale:F3}, " +
                $"{config.GainShareMaxScale:F3}]");
        }
        output.WriteLine(
            $"dropout = {config.Model.Dropout:F3}");
        output.WriteLine(
            $"learning-rate schedule = warmup {config.WarmupEpochs} " +
            $"epoch(s), cosine to " +
            $"{config.MinimumLearningRateRatio:P1}");
        output.WriteLine(
            $"checkpoint = {config.CheckpointPath}, " +
            $"resume {(config.ResumeFromCheckpoint ? "enabled" : "disabled")}, " +
            $"auto-resume {(config.AutoResume ? "enabled" : "disabled")}");
        output.WriteLine(
            config.EarlyStoppingPatience > 0
                ? $"early stopping = patience " +
                    $"{config.EarlyStoppingPatience} epoch(s)"
                : "early stopping = disabled");
        if (trainData is Cifar100)
        {
            var cifar100 = (Cifar100)trainData;
            Cifar100Options augmentation = cifar100.Options;
            output.WriteLine(
                $"tokenization = {augmentation.PatchSize}x" +
                $"{augmentation.PatchSize} patches, " +
                $"{trainData.Rows} tokens x " +
                $"{trainData.Columns} features");
            output.WriteLine(
                $"normalization = " +
                (augmentation.Normalize ? "cifar100" : "disabled"));
            output.WriteLine(
                ClassificationDatasets.DescribeAugmentation(augmentation));
        }

    }

    private static string GetSimdStatus()
    {
        if (!Tensor.SimdEnabled)
            return "disabled (scalar)";

        return Tensor.IsSimdHardwareAccelerated
            ? "enabled (Vector256)"
            : "enabled, hardware unavailable (scalar)";
    }

}
