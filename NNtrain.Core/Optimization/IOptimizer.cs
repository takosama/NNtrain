using System.Text.Json;

namespace NNtrain;

public interface IOptimizer
{
    void ZeroGrad();

    void Step();
}

public static class OptimizerTorchExtensions
{
    private static readonly JsonSerializerOptions StateJsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    public static void zero_grad(this IOptimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        optimizer.ZeroGrad();
    }

    public static void step(this IOptimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        optimizer.Step();
    }

    public static OptimizerStateDictionary state_dict(
        this IOptimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        return optimizer switch
        {
            AdamW value => Create("AdamW", value.CaptureState()),
            NekoMuon value => Create("NekoMuon", value.CaptureState()),
            Lion value => Create("Lion", value.CaptureState()),
            GainShareAdamW value =>
                Create("GainShareAdamW", value.CaptureState()),
            CompositeOptimizer value => new OptimizerStateDictionary(
                "CompositeOptimizer",
                StateJson: null,
                value.Optimizers.Select(state_dict).ToArray()),
            _ => throw new NotSupportedException(
                $"Optimizer '{optimizer.GetType().Name}' does not expose " +
                "a serializable state dictionary."),
        };
    }

    public static void load_state_dict(
        this IOptimizer optimizer,
        OptimizerStateDictionary state)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(state);

        switch (optimizer, state.OptimizerType)
        {
            case (AdamW value, "AdamW"):
                value.RestoreState(Read<AdamWState>(state));
                break;
            case (NekoMuon value, "NekoMuon"):
                value.RestoreState(Read<NekoMuonState>(state));
                break;
            case (Lion value, "Lion"):
                value.RestoreState(Read<LionState>(state));
                break;
            case (GainShareAdamW value, "GainShareAdamW"):
                value.RestoreState(Read<GainShareAdamWState>(state));
                break;
            case (CompositeOptimizer value, "CompositeOptimizer"):
                RestoreComposite(value, state);
                break;
            default:
                throw new ArgumentException(
                    $"State for '{state.OptimizerType}' cannot be loaded " +
                    $"into optimizer '{optimizer.GetType().Name}'.",
                    nameof(state));
        }
    }

    private static OptimizerStateDictionary Create<T>(
        string optimizerType,
        T state)
        => new(
            optimizerType,
            JsonSerializer.Serialize(state, StateJsonOptions),
            []);

    private static T Read<T>(OptimizerStateDictionary state)
    {
        if (string.IsNullOrWhiteSpace(state.StateJson))
        {
            throw new ArgumentException(
                "Optimizer state dictionary has no serialized state.",
                nameof(state));
        }
        return JsonSerializer.Deserialize<T>(
            state.StateJson,
            StateJsonOptions)
            ?? throw new InvalidDataException(
                "Optimizer state dictionary was JSON null.");
    }

    private static void RestoreComposite(
        CompositeOptimizer optimizer,
        OptimizerStateDictionary state)
    {
        if (state.Children is null
            || state.Children.Length != optimizer.Optimizers.Count)
        {
            throw new ArgumentException(
                "Composite optimizer state group count does not match.",
                nameof(state));
        }

        for (int index = 0; index < state.Children.Length; index++)
        {
            optimizer.Optimizers[index].load_state_dict(
                state.Children[index]);
        }
    }
}

public sealed record OptimizerStateDictionary(
    string OptimizerType,
    string? StateJson,
    OptimizerStateDictionary[] Children);

public interface ILearningRateAdjustable
{
    float LearningRate { get; }

    void SetLearningRate(float learningRate);
}
