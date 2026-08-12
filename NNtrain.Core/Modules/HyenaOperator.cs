namespace NNtrain;

/// <summary>
/// Order-2 causal Hyena sequence mixer.
/// </summary>
class HyenaOperator : Module
{
    private readonly int _modelWidth;
    private readonly HyenaConvolutionAlgorithm _convolutionAlgorithm;
    private readonly Linear _inputProjection;
    private readonly Parameter _shortFilter;
    private readonly HyenaFilter _longFilter;
    private readonly Parameter _diagonalBias;
    private readonly Linear _outputProjection;

    public HyenaOperator(
        int modelWidth,
        int contextLength,
        int filterWidth,
        Random? random = null,
        float initializationScale = 0.02f,
        HyenaConvolutionAlgorithm convolutionAlgorithm =
            HyenaConvolutionAlgorithm.Auto)
    {
        if (modelWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(modelWidth));
        if (contextLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(contextLength));
        if (filterWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(filterWidth));
        if (!float.IsFinite(initializationScale)
            || initializationScale <= 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(initializationScale));
        }

        random ??= new Random(1);
        _modelWidth = modelWidth;
        _convolutionAlgorithm = convolutionAlgorithm;
        int channels = checked(3 * modelWidth);
        _inputProjection = RegisterModule(
            new Linear(
                modelWidth,
                channels,
                random,
                initializationScale));

        var shortValues = new float[checked(3 * channels)];
        for (int channel = 0; channel < channels; channel++)
        {
            shortValues[channel] = 1f;
            shortValues[channels + channel] =
                ((float)random.NextDouble() * 2f - 1f)
                * initializationScale;
            shortValues[2 * channels + channel] =
                ((float)random.NextDouble() * 2f - 1f)
                * initializationScale;
        }
        _shortFilter = RegisterParameter(
            new Parameter(
                shortValues,
                [3, channels],
                "ShortFilter",
                WeightDecayPolicy.Apply));
        _longFilter = RegisterModule(
            new HyenaFilter(
                contextLength,
                modelWidth,
                filterWidth,
                random,
                initializationScale));
        _diagonalBias = RegisterParameter(
            new Parameter(
                Enumerable.Repeat(1f, modelWidth).ToArray(),
                [modelWidth],
                "DiagonalBias",
                WeightDecayPolicy.Exclude));
        _outputProjection = RegisterModule(
            new Linear(
                modelWidth,
                modelWidth,
                random,
                initializationScale));
    }

    public Tensor Forward(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.Rank is not (2 or 3))
        {
            throw new InvalidOperationException(
                "Hyena input must have shape [sequence, width] or " +
                "[batch, sequence, width].");
        }
        if (input.Shape[^1] != _modelWidth)
        {
            throw new ArgumentException(
                $"Hyena input width must be {_modelWidth}.",
                nameof(input));
        }

        bool unbatched = input.Rank == 2;
        int batch = unbatched ? 1 : input.Shape[0];
        int sequence = unbatched ? input.Shape[0] : input.Shape[1];
        Tensor batched = unbatched
            ? input.Reshape(1, sequence, _modelWidth)
            : input;
        Tensor projected = _inputProjection.ForwardBatch(batched);
        Tensor filter = _longFilter.Forward(sequence);
        Tensor mixed = projected.FusedCausalHyenaOrder2(
            _shortFilter.T,
            filter,
            _diagonalBias.T,
            _convolutionAlgorithm);
        Tensor output = _outputProjection.ForwardBatch(mixed);
        return unbatched
            ? output.Reshape(sequence, _modelWidth)
            : output;
    }
}
