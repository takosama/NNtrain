using NNtrain.Cuda.Execution;
using NNtrain.Training.Optimization;

namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    internal static void PreflightCudaOptimizer(
        WikiTrainingConfiguration config,
        TensorPrecisionMode precisionMode,
        Func<int, CudaKernelCapabilities>? capabilityProvider = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.GetExecutionDevice() != TensorDevice.Cuda)
            return;

        CudaOptimizerCapabilityPreflight.EnsureBeforeAllocation(
            GetCudaOptimizerKind(config),
            precisionMode,
            config.DeviceIndices ?? [config.DeviceIndex],
            capabilityProvider);
    }

    private static CudaOptimizerKind GetCudaOptimizerKind(
        WikiTrainingConfiguration config)
    {
        if (config.IsOptimizer(WikiTrainingConfiguration.NekoMuonOptimizer))
            return CudaOptimizerKind.NekoMuon;
        if (config.IsOptimizer(WikiTrainingConfiguration.LionOptimizer))
            return CudaOptimizerKind.Lion;
        if (config.IsOptimizer(
            WikiTrainingConfiguration.GainShareAdamWOptimizer))
        {
            return CudaOptimizerKind.GainShareAdamW;
        }
        return CudaOptimizerKind.AdamW;
    }

    internal static IOptimizer CreateOptimizer(
        LanguageModel model,
        WikiTrainingConfiguration config)
        => CreateOptimizerBundle(model, config).RootOptimizer;

    internal static OptimizerBundle CreateOptimizerBundle(
        LanguageModel model,
        WikiTrainingConfiguration config)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(config);
        bool useBFloat16Moments =
            model.PrecisionMode == TensorPrecisionMode.BFloat16;

        if (config.IsOptimizer(WikiTrainingConfiguration.NekoMuonOptimizer))
        {
            IOptimizer nekoMuon = optim.NekoMuon(
                model.HiddenWeightParameters,
                lr: config.LearningRate,
                newton_schulz_interval:
                    config.NekoMuonNewtonSchulzInterval,
                weight_decay: config.WeightDecay,
                newton_schulz_depth_mode:
                    config.GetNekoMuonNewtonSchulzDepthMode(),
                newton_schulz_depth:
                    config.GetNekoMuonNewtonSchulzDepth());
            IOptimizer auxiliaryAdamW = optim.AdamW(
                model.AuxiliaryParameters,
                lr: config.AuxiliaryLearningRate,
                beta1: 0.9f,
                beta2: 0.95f,
                eps: 1e-8f,
                weight_decay: config.WeightDecay,
                bf16_first_moment: useBFloat16Moments,
                bf16_second_moment: useBFloat16Moments);
            return new OptimizerBundle(
            [
                new OptimizerGroup("hidden", nekoMuon),
                new OptimizerGroup("auxiliary", auxiliaryAdamW),
            ]);
        }

        if (config.IsOptimizer(
            WikiTrainingConfiguration.GainShareAdamWOptimizer))
        {
            IOptimizer optimizer = optim.GainShareAdamW(
                model.make_gainshare_parameter_groups(
                    config.GainShareBlockDepth),
                lr: config.LearningRate,
                beta1: config.GainShareBeta1,
                beta2: config.GainShareBeta2,
                eps: config.GainShareEpsilon,
                rho: config.GainShareRho,
                gamma: config.GainShareGamma,
                min_scale: config.GainShareMinScale,
                max_scale: config.GainShareMaxScale,
                weight_decay: config.WeightDecay);
            return OptimizerBundle.Wrap(optimizer, ["all"]);
        }

        if (config.IsOptimizer(WikiTrainingConfiguration.LionOptimizer))
        {
            IOptimizer optimizer = optim.Lion(
                model.parameters(),
                lr: config.LearningRate,
                weight_decay: config.WeightDecay);
            return OptimizerBundle.Wrap(optimizer, ["all"]);
        }

        IOptimizer adamW = optim.AdamW(
            model.parameters(),
            lr: config.LearningRate,
            weight_decay: config.WeightDecay,
            bf16_first_moment: useBFloat16Moments,
            bf16_second_moment: useBFloat16Moments);
        return OptimizerBundle.Wrap(adamW, ["all"]);
    }

    private static void WriteOptimizerSummary(
        LanguageModel model,
        WikiTrainingConfiguration config,
        TextWriter output)
    {
        if (config.IsOptimizer(WikiTrainingConfiguration.NekoMuonOptimizer))
        {
            output.WriteLine(
                $"optimizer = NekoMuon " +
                $"({model.HiddenWeightParameters.Count} matrix parameters, " +
                $"lr {config.LearningRate:G}, Newton-Schulz every " +
                $"{config.NekoMuonNewtonSchulzInterval} steps, " +
                $"{FormatNekoMuonNewtonSchulzDepthPolicy(config)}) + " +
                "AdamW " +
                $"({model.AuxiliaryParameters.Count} auxiliary parameters, " +
                $"lr {config.AuxiliaryLearningRate:G}, moments " +
                $"{GetAdamWMomentStorage(model)})");
        }
        else if (config.IsOptimizer(
            WikiTrainingConfiguration.GainShareAdamWOptimizer))
        {
            IReadOnlyList<IReadOnlyList<Parameter>> groups =
                model.make_gainshare_parameter_groups(
                    config.GainShareBlockDepth);
            output.WriteLine(
                $"optimizer = GainShareAdamW " +
                $"({groups.Sum(group => group.Count)} parameters in " +
                $"{groups.Count} blocks at depth " +
                $"{config.GainShareBlockDepth}, lr {config.LearningRate:G}, " +
                $"rho {config.GainShareRho:G}, gamma " +
                $"{config.GainShareGamma:G}, scale " +
                $"{config.GainShareMinScale:G}-{config.GainShareMaxScale:G})");
        }
        else if (config.IsOptimizer(WikiTrainingConfiguration.LionOptimizer))
        {
            output.WriteLine(
                $"optimizer = Lion ({model.parameters().Count()} " +
                $"parameters, lr {config.LearningRate:G}, weight decay " +
                $"{config.WeightDecay:G})");
        }
        else
        {
            output.WriteLine(
                $"optimizer = AdamW ({model.parameters().Count()} " +
                $"parameters, lr {config.LearningRate:G}, moments " +
                $"{GetAdamWMomentStorage(model)})");
        }
        output.WriteLine(config.WarmupPercent == 0f
            ? "learning-rate schedule = no warmup; cosine decay from the " +
                "first update"
            : $"learning-rate schedule = linear warmup " +
                $"{config.WarmupPercent:G}% of total training, then " +
                "cosine decay");
    }

    private static string FormatNekoMuonNewtonSchulzDepthPolicy(
        WikiTrainingConfiguration config)
    {
        NekoMuonNewtonSchulzDepthMode mode =
            config.GetNekoMuonNewtonSchulzDepthMode();
        return mode == NekoMuonNewtonSchulzDepthMode.Adaptive
            ? "adaptive depth"
            : $"{mode.ToString().ToLowerInvariant()} depth " +
                $"{config.GetNekoMuonNewtonSchulzDepth():G}";
    }

    private static string GetAdamWMomentStorage(
        LanguageModel model)
        => model.PrecisionMode == TensorPrecisionMode.BFloat16
            ? "bf16/bf16"
            : "f32/f32";

    private static string FormatOptimizerDiagnostics(IOptimizer optimizer)
    {
        NekoMuon? nekoMuon = OptimizerBundle
            .GetCheckpointLeafOptimizers(optimizer)
            .OfType<NekoMuon>()
            .FirstOrDefault();
        if (nekoMuon is null)
            return string.Empty;

        NekoMuonDiagnostics diagnostics = nekoMuon.GetDiagnostics();
        return $", neko confidence = {diagnostics.MeanConfidence:G4} " +
            $"[{diagnostics.MinimumConfidence:G4}-" +
            $"{diagnostics.MaximumConfidence:G4}], NS depth = " +
            $"{diagnostics.MeanNewtonSchulzDepth:G4}/" +
            $"{diagnostics.MaximumNewtonSchulzDepth}";
    }
}
