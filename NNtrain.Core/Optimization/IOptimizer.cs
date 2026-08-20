using System.Text.Json;
using System.Text.Json.Serialization;

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
            JsonSerializer.SerializeToElement(state, StateJsonOptions),
            []);

    private static T Read<T>(OptimizerStateDictionary state)
    {
        if (state.StateJson is not JsonElement element
            || element.ValueKind is JsonValueKind.Undefined
                or JsonValueKind.Null)
        {
            throw new ArgumentException(
                "Optimizer state dictionary has no serialized state.",
                nameof(state));
        }

        // Checkpoints written before the state was embedded as real JSON hold
        // an escaped string here. Both layouts stay readable.
        if (element.ValueKind == JsonValueKind.String)
        {
            string? legacy = element.GetString();
            if (string.IsNullOrWhiteSpace(legacy))
            {
                throw new ArgumentException(
                    "Optimizer state dictionary has no serialized state.",
                    nameof(state));
            }

            return JsonSerializer.Deserialize<T>(legacy, StateJsonOptions)
                ?? throw new InvalidDataException(
                    "Optimizer state dictionary was JSON null.");
        }

        return element.Deserialize<T>(StateJsonOptions)
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

/// <summary>
/// Serializable optimizer state.
/// </summary>
/// <remarks>
/// <see cref="StateJson"/> holds the optimizer's own state as embedded JSON.
/// It used to be a pre-serialized <see cref="string"/>, which made every
/// checkpoint carry one enormous JSON string value; System.Text.Json refuses
/// to write a string longer than 166,666,666 characters, so models above
/// roughly six million parameters could not be saved at all. Storing a
/// <see cref="JsonElement"/> keeps the numbers as a JSON array, which has no
/// such limit. Older checkpoints that still hold a string remain loadable.
/// </remarks>
public sealed record OptimizerStateDictionary(
    string OptimizerType,
    JsonElement? StateJson,
    OptimizerStateDictionary[] Children)
{
    /// <summary>
    /// The embedded state as raw JSON text. <see cref="JsonElement"/> has no
    /// value equality, so comparisons and assertions must use this instead of
    /// <see cref="StateJson"/> directly.
    /// </summary>
    [JsonIgnore]
    public string? StateJsonText => StateJson?.GetRawText();

    public bool Equals(OptimizerStateDictionary? other)
        => other is not null
            && string.Equals(
                OptimizerType,
                other.OptimizerType,
                StringComparison.Ordinal)
            && string.Equals(
                StateJsonText,
                other.StateJsonText,
                StringComparison.Ordinal)
            && EqualityComparer<OptimizerStateDictionary[]>.Default.Equals(
                Children,
                other.Children);

    public override int GetHashCode()
        => HashCode.Combine(OptimizerType, StateJsonText);
}

public interface ILearningRateAdjustable
{
    float LearningRate { get; }

    void SetLearningRate(float learningRate);
}
