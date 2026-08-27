using System.Text.Json;
using System.Text.Json.Serialization;

namespace NNtrain;

public interface IOptimizer
{
    void zero_grad();

    void step();

    OptimizerStateDictionary state_dict()
        => throw new NotSupportedException(
            $"Optimizer '{GetType().Name}' does not expose serializable state.");

    void load_state_dict(OptimizerStateDictionary state)
        => throw new NotSupportedException(
            $"Optimizer '{GetType().Name}' does not expose serializable state.");
}

/// <summary>
/// An optimizer that owns an ordered collection of child optimizers.
/// Streaming checkpoints flatten this collection depth-first, preserving the
/// declared order without depending on concrete optimizer types.
/// </summary>
public interface IOptimizerContainer : IOptimizer
{
    IReadOnlyList<IOptimizer> Optimizers { get; }
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
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    internal static OptimizerStateDictionary Create<T>(
        string optimizerType,
        T state)
        => new(
            optimizerType,
            JsonSerializer.SerializeToElement(state, JsonOptions),
            []);

    internal T Read<T>(string expectedOptimizerType)
    {
        if (!string.Equals(
            OptimizerType,
            expectedOptimizerType,
            StringComparison.Ordinal))
        {
            throw new ArgumentException(
                $"State for '{OptimizerType}' cannot be loaded into " +
                $"optimizer '{expectedOptimizerType}'.",
                nameof(expectedOptimizerType));
        }

        if (StateJson is not JsonElement element
            || element.ValueKind is JsonValueKind.Undefined
                or JsonValueKind.Null)
        {
            throw new ArgumentException(
                "Optimizer state dictionary has no serialized state.");
        }

        if (element.ValueKind == JsonValueKind.String)
        {
            string? legacy = element.GetString();
            if (string.IsNullOrWhiteSpace(legacy))
            {
                throw new ArgumentException(
                    "Optimizer state dictionary has no serialized state.");
            }

            return JsonSerializer.Deserialize<T>(legacy, JsonOptions)
                ?? throw new InvalidDataException(
                    "Optimizer state dictionary was JSON null.");
        }

        return element.Deserialize<T>(JsonOptions)
            ?? throw new InvalidDataException(
                "Optimizer state dictionary was JSON null.");
    }

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
