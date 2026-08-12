namespace NNtrain;

/// <summary>
/// Generates a channel-wise long-convolution filter from continuous position
/// features using the implicit MLP parameterization from Hyena.
/// </summary>
class HyenaFilter : Module
{
    private readonly int _contextLength;
    private readonly int _modelWidth;
    private readonly Tensor _positionFeatures;
    private readonly Tensor _modulation;
    private readonly Linear _input;
    private readonly Linear _hidden;
    private readonly Linear _output;

    public HyenaFilter(
        int contextLength,
        int modelWidth,
        int filterWidth,
        Random random,
        float initializationScale)
    {
        if (contextLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(contextLength));
        if (modelWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(modelWidth));
        if (filterWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(filterWidth));
        ArgumentNullException.ThrowIfNull(random);

        _contextLength = contextLength;
        _modelWidth = modelWidth;
        _positionFeatures = Tensor.FromOwnedData(
            CreatePositionFeatures(contextLength),
            [contextLength, 3]);
        _modulation = Tensor.FromOwnedData(
            CreateExponentialModulation(contextLength, modelWidth),
            [contextLength, modelWidth]);
        _input = RegisterModule(
            new Linear(3, filterWidth, random, initializationScale));
        _hidden = RegisterModule(
            new Linear(
                filterWidth,
                filterWidth,
                random,
                initializationScale));
        _output = RegisterModule(
            new Linear(
                filterWidth,
                modelWidth,
                random,
                initializationScale));
    }

    public Tensor Forward(int sequenceLength)
    {
        if (sequenceLength <= 0 || sequenceLength > _contextLength)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sequenceLength),
                sequenceLength,
                $"Filter length must be between 1 and {_contextLength}.");
        }

        Tensor positions = sequenceLength == _contextLength
            ? _positionFeatures
            : _positionFeatures.Slice(0, 0, sequenceLength);
        Tensor modulation = sequenceLength == _contextLength
            ? _modulation
            : _modulation.Slice(0, 0, sequenceLength);
        Tensor hidden = _input.ForwardBatch(positions).Sin();
        hidden = _hidden.ForwardBatch(hidden).Sin();
        Tensor filter = _output.ForwardBatch(hidden);
        return filter * modulation;
    }

    private static float[] CreatePositionFeatures(int contextLength)
    {
        var values = new float[checked(contextLength * 3)];
        float denominator = Math.Max(1, contextLength - 1);
        for (int position = 0; position < contextLength; position++)
        {
            float time = position / denominator;
            float angle = 2f * MathF.PI * position / contextLength;
            int offset = position * 3;
            values[offset] = time;
            values[offset + 1] = MathF.Cos(angle);
            values[offset + 2] = MathF.Sin(angle);
        }
        return values;
    }

    private static float[] CreateExponentialModulation(
        int contextLength,
        int modelWidth)
    {
        const float target = 1e-2f;
        const float fastDecayFraction = 0.3f;
        const float slowDecayFraction = 1.5f;
        const float shift = 0.05f;
        float slowDecay = -MathF.Log(target) / slowDecayFraction;
        float fastDecay = -MathF.Log(target) / fastDecayFraction;
        float timeDenominator = Math.Max(1, contextLength - 1);
        float channelDenominator = Math.Max(1, modelWidth - 1);
        var values = new float[checked(contextLength * modelWidth)];
        for (int position = 0; position < contextLength; position++)
        {
            float time = position / timeDenominator;
            for (int channel = 0; channel < modelWidth; channel++)
            {
                float channelFraction = channel / channelDenominator;
                float decay = slowDecay
                    + channelFraction * (fastDecay - slowDecay);
                values[position * modelWidth + channel] =
                    MathF.Exp(-time * decay) + shift;
            }
        }
        return values;
    }
}
