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
        if (config.IsOptimizer(WikiTrainingConfiguration.MuonOptimizer)
            || config.IsOptimizer(
                WikiTrainingConfiguration.NekoMuonOptimizer))
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

        if (config.IsOptimizer(WikiTrainingConfiguration.MuonOptimizer))
        {
            var muon = (NekoMuon)optim.Muon(
                model.HiddenWeightParameters,
                lr: config.LearningRate,
                momentum: 0.95f,
                weight_decay: config.WeightDecay);
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
                new OptimizerGroup("hidden", muon),
                new OptimizerGroup("auxiliary", auxiliaryAdamW),
            ]);
        }

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
        if (config.IsOptimizer(WikiTrainingConfiguration.MuonOptimizer))
        {
            output.WriteLine(
                $"optimizer = Muon " +
                $"({model.HiddenWeightParameters.Count} matrix parameters, " +
                $"lr {config.LearningRate:G}, momentum 0.95, " +
                "Nesterov, fixed NS5 every step) + AdamW " +
                $"({model.AuxiliaryParameters.Count} auxiliary parameters, " +
                $"lr {config.AuxiliaryLearningRate:G}, moments " +
                $"{GetAdamWMomentStorage(model)})");
        }
        else if (config.IsOptimizer(
            WikiTrainingConfiguration.NekoMuonOptimizer))
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

    internal static string FormatOptimizerDiagnostics(
        IOptimizer optimizer,
        WikiTrainingConfiguration config)
    {
        IReadOnlyList<IOptimizer> leaves =
            GetOptimizerDiagnosticLeaves(optimizer);
        string optimizerDiagnostics;
        if (config.IsOptimizer(WikiTrainingConfiguration.MuonOptimizer))
        {
            optimizerDiagnostics = ", muon NS depth = 5";
        }
        else
        {
            NekoMuon? nekoMuon = leaves
                .OfType<NekoMuon>()
                .FirstOrDefault();
            if (nekoMuon is null)
            {
                optimizerDiagnostics = string.Empty;
            }
            else
            {
                NekoMuonDiagnostics diagnostics = nekoMuon.GetDiagnostics();
                optimizerDiagnostics =
                    $", neko confidence = {diagnostics.MeanConfidence:G4} " +
                    $"[{diagnostics.MinimumConfidence:G4}-" +
                    $"{diagnostics.MaximumConfidence:G4}], NS depth = " +
                    $"{diagnostics.MeanNewtonSchulzDepth:G4}/" +
                    $"{diagnostics.MaximumNewtonSchulzDepth}";
            }
        }

        return optimizerDiagnostics +
            FormatMix8QuantizationDiagnostics(leaves);
    }

    private static string FormatMix8QuantizationDiagnostics(
        IReadOnlyList<IOptimizer> leaves)
    {
        var values = new List<Mix8QuantizationDiagnostics>();
        foreach (IOptimizer leaf in leaves)
        {
            if (leaf is IMix8QuantizationDiagnosticsProvider provider
                && provider.TryGetMix8QuantizationDiagnostics(
                    out Mix8QuantizationDiagnostics value))
            {
                values.Add(value);
            }
        }

        Mix8QuantizationDiagnostics diagnostics =
            Mix8QuantizationDiagnostics.Combine(values);
        if (!diagnostics.HasValues)
            return string.Empty;

        return $", quantized_weight_change_rate = " +
            $"{diagnostics.QuantizedWeightChangeRate:G6}, " +
            $"residual_rms / quant_step = " +
            $"{diagnostics.ResidualRmsPerQuantStep:G6}, " +
            $"update_rms / quant_step = " +
            $"{diagnostics.UpdateRmsPerQuantStep:G6}";
    }

    private static IReadOnlyList<IOptimizer> GetOptimizerDiagnosticLeaves(
        IOptimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        var leaves = new List<IOptimizer>();
        AddLeaves(optimizer, leaves);
        return leaves;

        static void AddLeaves(
            IOptimizer current,
            List<IOptimizer> destination)
        {
            if (current is not IOptimizerContainer container)
            {
                destination.Add(current);
                return;
            }

            foreach (IOptimizer child in container.Optimizers)
                AddLeaves(child, destination);
        }
    }
}
