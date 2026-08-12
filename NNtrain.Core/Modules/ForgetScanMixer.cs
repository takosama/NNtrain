namespace NNtrain;

/// <summary>
/// Content-dependent fixed-state sequence mixer using an associative affine
/// prefix scan.
/// </summary>
class ForgetScanMixer : Module
{
    private readonly int _modelWidth;
    private readonly Linear _gateProjection;
    private readonly Linear _outputProjection;

    public ForgetScanMixer(
        int modelWidth,
        Random? random = null,
        float initializationScale = 0.02f)
    {
        if (modelWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(modelWidth));
        if (!float.IsFinite(initializationScale)
            || initializationScale <= 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(initializationScale));
        }

        random ??= new Random(1);
        _modelWidth = modelWidth;
        _gateProjection = RegisterModule(
            new Linear(
                modelWidth,
                checked(3 * modelWidth),
                random,
                initializationScale));
        _outputProjection = RegisterModule(
            new Linear(
                modelWidth,
                modelWidth,
                random,
                initializationScale));
    }

    public Tensor Forward(Tensor normalizedInput)
    {
        ArgumentNullException.ThrowIfNull(normalizedInput);
        if (normalizedInput.Rank is not (2 or 3))
        {
            throw new InvalidOperationException(
                "ForgetScan input must have shape [sequence, width] or " +
                "[batch, sequence, width].");
        }
        if (normalizedInput.Shape[^1] != _modelWidth)
        {
            throw new ArgumentException(
                $"ForgetScan input width must be {_modelWidth}.",
                nameof(normalizedInput));
        }

        bool unbatched = normalizedInput.Rank == 2;
        int batch = unbatched ? 1 : normalizedInput.Shape[0];
        int sequence = unbatched
            ? normalizedInput.Shape[0]
            : normalizedInput.Shape[1];
        Tensor batched = unbatched
            ? normalizedInput.Reshape(1, sequence, _modelWidth)
            : normalizedInput;
        Tensor projected = _gateProjection.ForwardBatch(batched);
        Tensor memory = projected.FusedForgetScan();
        Tensor output = _outputProjection.ForwardBatch(memory);
        return unbatched
            ? output.Reshape(sequence, _modelWidth)
            : output;
    }
}
