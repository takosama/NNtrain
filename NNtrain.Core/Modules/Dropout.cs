namespace NNtrain;

public sealed class Dropout : Module
{
    private readonly Random _random;

    public Dropout(
        float probability = 0.5f,
        Random? random = null,
        TensorDType dtype = TensorDType.Float32)
        : base(dtype)
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

    public Tensor AddResidual(Tensor residual, Tensor branch)
    {
        ArgumentNullException.ThrowIfNull(residual);
        ArgumentNullException.ThrowIfNull(branch);
        return IsTraining
            ? residual.AddDropout(branch, Probability, _random)
            : residual + branch;
    }
}
