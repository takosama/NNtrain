namespace NNtrain;

public sealed class Dropout : Module
{
    private readonly Random _random;

    public Dropout(float probability = 0.5f, Random? random = null)
    {
        if (!float.IsFinite(probability)
            || probability < 0f
            || probability >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(probability),
                probability,
                "Dropout probability must be finite and in [0, 1).");
        }

        Probability = probability;
        _random = random ?? Random.Shared;
    }

    public float Probability { get; }

    public Tensor Forward(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);
        return IsTraining ? input.Dropout(Probability, _random) : input;
    }
}
