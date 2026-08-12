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
            var nekoMuon = new NekoMuon(
                model.HiddenWeightParameters,
                new NekoMuonOptions
                {
                    LearningRate = config.LearningRate,
                    NewtonSchulzInterval =
                        config.NekoMuonNewtonSchulzInterval,
                    WeightDecay = config.WeightDecay,
                });
            var auxiliaryAdamW = new AdamW(
                model.AuxiliaryParameters,
                new AdamWOptions
                {
                    LearningRate = config.AuxiliaryLearningRate,
                    Beta1 = 0.9f,
                    Beta2 = 0.95f,
                    Epsilon = 1e-8f,
                    WeightDecay = config.WeightDecay,
                    UseBFloat16FirstMoment =
                        config.AdamWUseBFloat16FirstMoment,
                    UseBFloat16SecondMoment =
                        config.AdamWUseBFloat16SecondMoment,
                });
            return new CompositeOptimizer(nekoMuon, auxiliaryAdamW);
        }

        return new AdamW(
            model.Parameters(),
            new AdamWOptions
            {
                LearningRate = config.LearningRate,
                WeightDecay = config.WeightDecay,
                UseBFloat16FirstMoment =
                    config.AdamWUseBFloat16FirstMoment,
                UseBFloat16SecondMoment =
                    config.AdamWUseBFloat16SecondMoment,
            });
    }

    internal static float CalculateLearningRateFactor(
        double overallProgress,
        float warmupPercent)
    {
        const float MinimumFactor = 1e-6f;
        if (!double.IsFinite(overallProgress)
            || overallProgress < 0d
            || overallProgress > 1d)
        {
            throw new ArgumentOutOfRangeException(nameof(overallProgress));
        }
        if (!float.IsFinite(warmupPercent)
            || warmupPercent < 0f
            || warmupPercent >= 100f)
        {
            throw new ArgumentOutOfRangeException(nameof(warmupPercent));
        }

        double warmupFraction = warmupPercent / 100d;
        if (warmupFraction > 0d && overallProgress <= warmupFraction)
        {
            return MathF.Max(
                MinimumFactor,
                (float)(overallProgress / warmupFraction));
        }

        double decayProgress = warmupFraction == 1d
            ? 1d
            : (overallProgress - warmupFraction)
                / (1d - warmupFraction);
        decayProgress = Math.Clamp(decayProgress, 0d, 1d);
        float cosine = 0.5f
            * (1f + MathF.Cos(MathF.PI * (float)decayProgress));
        return MathF.Max(MinimumFactor, cosine);
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
                $"optimizer = AdamW ({model.Parameters().Count()} " +
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
