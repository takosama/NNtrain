#pragma warning disable CS8981

namespace NNtrain;

/// <summary>PyTorch-style optimizer factories.</summary>
public static class optim
{
    public static IOptimizer AdamW(
        IEnumerable<Parameter> parameters,
        float lr = 1e-3f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float weight_decay = 5e-2f,
        bool decay_1d = false,
        bool bf16_first_moment = false,
        bool bf16_second_moment = false)
        => new NNtrain.AdamW(
            parameters,
            new AdamWOptions
            {
                LearningRate = lr,
                Beta1 = beta1,
                Beta2 = beta2,
                Epsilon = eps,
                WeightDecay = weight_decay,
                Decay1D = decay_1d,
                UseBFloat16FirstMoment = bf16_first_moment,
                UseBFloat16SecondMoment = bf16_second_moment,
            });

    public static IOptimizer Lion(
        IEnumerable<Parameter> parameters,
        float lr = 3e-4f,
        float beta1 = 0.9f,
        float beta2 = 0.99f,
        float weight_decay = 1e-2f,
        bool decay_1d = false)
        => new NNtrain.Lion(
            parameters,
            new LionOptions
            {
                LearningRate = lr,
                Beta1 = beta1,
                Beta2 = beta2,
                WeightDecay = weight_decay,
                Decay1D = decay_1d,
            });

    public static IOptimizer NekoMuon(
        IEnumerable<Parameter> parameters,
        float lr = 3e-4f,
        float beta_fast = 0.9f,
        float beta_slow = 0.99f,
        float rho = 0.9f,
        float eps = 1e-7f,
        int newton_schulz_steps = 5,
        int newton_schulz_interval = 5,
        float weight_decay = 1e-2f,
        bool decay_1d = false)
        => new NNtrain.NekoMuon(
            parameters,
            new NekoMuonOptions
            {
                LearningRate = lr,
                BetaFast = beta_fast,
                BetaSlow = beta_slow,
                Rho = rho,
                Epsilon = eps,
                MaxNewtonSchulzSteps = newton_schulz_steps,
                NewtonSchulzInterval = newton_schulz_interval,
                WeightDecay = weight_decay,
                Decay1D = decay_1d,
            });

    public static IOptimizer GainShareAdamW(
        IReadOnlyList<IReadOnlyList<Parameter>> parameter_groups,
        float lr = 3e-4f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float eps = 1e-8f,
        float rho = 0.95f,
        float gamma = 1f,
        float min_scale = 0.5f,
        float max_scale = 2f,
        float weight_decay = 5e-4f)
        => new NNtrain.GainShareAdamW(
            parameter_groups,
            new GainShareAdamWOptions
            {
                LearningRate = lr,
                Beta1 = beta1,
                Beta2 = beta2,
                Epsilon = eps,
                Rho = rho,
                Gamma = gamma,
                MinScale = min_scale,
                MaxScale = max_scale,
                WeightDecay = weight_decay,
            });

    public static IOptimizer Composite(params IOptimizer[] optimizers)
        => new CompositeOptimizer(optimizers);
}
