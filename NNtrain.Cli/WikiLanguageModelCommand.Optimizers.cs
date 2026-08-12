namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    internal static IOptimizer CreateOptimizer(
        IWikiLanguageModel model,
        WikiTrainingConfiguration config)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(config);

        if (config.IsOptimizer(WikiTrainingConfiguration.NekoMuonOptimizer))
        {
            IOptimizer nekoMuon = optim.NekoMuon(
                model.HiddenWeightParameters,
                lr: config.LearningRate,
                newton_schulz_interval:
                    config.NekoMuonNewtonSchulzInterval,
                weight_decay: config.WeightDecay);
            IOptimizer auxiliaryAdamW = optim.AdamW(
                model.AuxiliaryParameters,
                lr: config.AuxiliaryLearningRate,
                beta1: 0.9f,
                beta2: 0.95f,
                eps: 1e-8f,
                weight_decay: config.WeightDecay,
                bf16_first_moment: config.AdamWUseBFloat16FirstMoment,
                bf16_second_moment: config.AdamWUseBFloat16SecondMoment);
            return optim.Composite(nekoMuon, auxiliaryAdamW);
        }

        return optim.AdamW(
            model.parameters(),
            lr: config.LearningRate,
            weight_decay: config.WeightDecay,
            bf16_first_moment: config.AdamWUseBFloat16FirstMoment,
            bf16_second_moment: config.AdamWUseBFloat16SecondMoment);
    }

    internal static float CalculateLearningRateFactor(
        double overallProgress,
        float warmupPercent)
    {
        return WarmupCosineProgressLRScheduler.CalculateFactor(
            overallProgress,
            warmupPercent);
    }

    internal static float SetScheduledLearningRates(
        IOptimizer optimizer,
        WikiTrainingConfiguration config,
        double overallProgress)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(config);
        float factor = CalculateLearningRateFactor(
            overallProgress,
            config.WarmupPercent);

        if (config.IsOptimizer(WikiTrainingConfiguration.NekoMuonOptimizer))
        {
            if (optimizer is not CompositeOptimizer composite
                || composite.Optimizers.Count != 2
                || composite.Optimizers[0]
                    is not ILearningRateAdjustable primary
                || composite.Optimizers[1]
                    is not ILearningRateAdjustable auxiliary)
            {
                throw new InvalidOperationException(
                    "NekoMuon scheduling requires adjustable primary and " +
                    "auxiliary optimizers.");
            }

            primary.SetLearningRate(
                MathF.Max(float.Epsilon, config.LearningRate * factor));
            auxiliary.SetLearningRate(
                MathF.Max(
                    float.Epsilon,
                    config.AuxiliaryLearningRate * factor));
            return factor;
        }

        if (optimizer is not ILearningRateAdjustable adjustable)
        {
            throw new InvalidOperationException(
                "Learning-rate scheduling requires an adjustable optimizer.");
        }
        adjustable.SetLearningRate(
            MathF.Max(float.Epsilon, config.LearningRate * factor));
        return factor;
    }

    private static void WriteOptimizerSummary(
        IWikiLanguageModel model,
        WikiTrainingConfiguration config,
        TextWriter output)
    {
        if (config.IsOptimizer(WikiTrainingConfiguration.NekoMuonOptimizer))
        {
            output.WriteLine(
                $"optimizer = NekoMuon " +
                $"({model.HiddenWeightParameters.Count} matrix parameters, " +
                $"lr {config.LearningRate:G}, Newton-Schulz every " +
                $"{config.NekoMuonNewtonSchulzInterval} steps) + AdamW " +
                $"({model.AuxiliaryParameters.Count} auxiliary parameters, " +
                $"lr {config.AuxiliaryLearningRate:G}, moments " +
                $"{GetAdamWMomentStorage(config)})");
        }
        else
        {
            output.WriteLine(
                $"optimizer = AdamW ({model.parameters().Count()} " +
                $"parameters, lr {config.LearningRate:G}, moments " +
                $"{GetAdamWMomentStorage(config)})");
        }
        output.WriteLine(
            $"learning-rate schedule = linear warmup " +
            $"{config.WarmupPercent:G}% of total training, then cosine " +
            "decay");
    }

    private static string GetAdamWMomentStorage(
        WikiTrainingConfiguration config)
        => $"{(config.AdamWUseBFloat16FirstMoment ? "bf16" : "f32")}/" +
            $"{(config.AdamWUseBFloat16SecondMoment ? "bf16" : "f32")}";
}
