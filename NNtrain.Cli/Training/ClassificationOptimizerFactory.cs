using NNtrain.Training.Optimization;

namespace NNtrain;

internal static class ClassificationOptimizerFactory
{
    internal static OptimizerBundle CreateBundle(
        TransformerClassifier model,
        TrainingConfiguration configuration)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(configuration);

        if (configuration.IsOptimizer(
            TrainingConfiguration.GainShareAdamWOptimizer))
        {
            IOptimizer optimizer = optim.GainShareAdamW(
                model.make_gainshare_parameter_groups(
                    configuration.GainShareBlockDepth),
                lr: configuration.LearningRate,
                beta1: configuration.GainShareBeta1,
                beta2: configuration.GainShareBeta2,
                eps: configuration.GainShareEpsilon,
                rho: configuration.GainShareRho,
                gamma: configuration.GainShareGamma,
                min_scale: configuration.GainShareMinScale,
                max_scale: configuration.GainShareMaxScale,
                weight_decay: configuration.WeightDecay);
            return OptimizerBundle.Wrap(optimizer, ["all"]);
        }

        if (configuration.IsOptimizer(TrainingConfiguration.MuonOptimizer))
        {
            var muon = (NekoMuon)optim.Muon(
                model.HiddenWeightParameters,
                lr: configuration.LearningRate,
                momentum: 0.95f,
                weight_decay: configuration.WeightDecay);
            IOptimizer auxiliaryAdamW = CreateAuxiliaryAdamW(
                model,
                configuration);
            return new OptimizerBundle(
            [
                new OptimizerGroup("hidden", muon),
                new OptimizerGroup("auxiliary", auxiliaryAdamW),
            ]);
        }

        if (configuration.IsOptimizer(
            TrainingConfiguration.NekoMuonOptimizer))
        {
            IOptimizer nekoMuon = optim.NekoMuon(
                model.HiddenWeightParameters,
                lr: configuration.LearningRate,
                weight_decay: configuration.WeightDecay);
            IOptimizer auxiliaryAdamW = CreateAuxiliaryAdamW(
                model,
                configuration);
            return new OptimizerBundle(
            [
                new OptimizerGroup("hidden", nekoMuon),
                new OptimizerGroup("auxiliary", auxiliaryAdamW),
            ]);
        }

        if (configuration.IsOptimizer(TrainingConfiguration.LionOptimizer))
        {
            IOptimizer optimizer = optim.Lion(
                model.parameters(),
                lr: configuration.LearningRate,
                weight_decay: configuration.WeightDecay);
            return OptimizerBundle.Wrap(optimizer, ["all"]);
        }

        IOptimizer adamW = optim.AdamW(
            model.parameters(),
            lr: configuration.LearningRate,
            weight_decay: configuration.WeightDecay);
        return OptimizerBundle.Wrap(adamW, ["all"]);
    }

    private static IOptimizer CreateAuxiliaryAdamW(
        TransformerClassifier model,
        TrainingConfiguration configuration)
        => optim.AdamW(
            model.AuxiliaryParameters,
            lr: configuration.AuxiliaryLearningRate,
            beta1: 0.9f,
            beta2: 0.95f,
            eps: 1e-8f,
            weight_decay: configuration.WeightDecay);

    internal static void ApplyMuonPolicyAfterResume(
        TrainingConfiguration configuration,
        IOptimizer optimizer,
        TextWriter output)
    {
        ArgumentNullException.ThrowIfNull(configuration);
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(output);
        if (!configuration.IsOptimizer(TrainingConfiguration.MuonOptimizer))
            return;

        int applied = 0;
        foreach (NekoMuon muon in OptimizerBundle
            .GetCheckpointLeafOptimizers(optimizer)
            .OfType<NekoMuon>())
        {
            muon.SetOrdinaryMuonPolicy();
            applied++;
        }
        if (applied == 0)
        {
            throw new InvalidDataException(
                "Muon was configured, but the restored optimizer has no " +
                "compatible matrix-optimizer state.");
        }
        output.WriteLine(
            "resume Muon policy = momentum 0.95, Nesterov, " +
            "fixed NS5 every step (runtime override)");
    }
}
