namespace NNtrain;

/// <summary>
/// Pre-normalized GPT layer backed by a stable, matrix-valued delta-rule
/// memory.
/// </summary>
public sealed class FrogetMemoryV2Layer : Module
{
    private readonly Linear _memoryProjection;
    private readonly Linear _outputProjection;

    public FrogetMemoryV2Layer(
        int modelWidth,
        int hiddenWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        Random? random = null,
        float initializationScale = 0.02f,
        float dropout = 0f)
    {
        if (modelWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(modelWidth));
        if (hiddenWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenWidth));
        if (keyWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(keyWidth));
        if (valueWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(valueWidth));
        if (!float.IsFinite(retentionFloor)
            || retentionFloor < 0f
            || retentionFloor >= 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(retentionFloor));
        }
        if (!float.IsFinite(initializationScale)
            || initializationScale <= 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(initializationScale));
        }
        if (!float.IsFinite(dropout) || dropout < 0f || dropout >= 1f)
            throw new ArgumentOutOfRangeException(nameof(dropout));

        ModelWidth = modelWidth;
        KeyWidth = keyWidth;
        ValueWidth = valueWidth;
        RetentionFloor = retentionFloor;
        random ??= new Random(1);

        Ln1 = RegisterModule(new LayerNorm(modelWidth));
        _memoryProjection = RegisterModule(
            new Linear(
                modelWidth,
                checked(2 * keyWidth + 3 * valueWidth),
                random,
                initializationScale));
        _outputProjection = RegisterModule(
            new Linear(
                valueWidth,
                modelWidth,
                random,
                initializationScale));
        MemoryDropout = RegisterModule(new Dropout(dropout, random));
        Ln2 = RegisterModule(new LayerNorm(modelWidth));
        Ffn = RegisterModule(
            new FeedForward(
                modelWidth,
                hiddenWidth,
                random,
                initializationScale));
        FfnDropout = RegisterModule(new Dropout(dropout, random));
    }

    public int ModelWidth { get; }

    public int KeyWidth { get; }

    public int ValueWidth { get; }

    public float RetentionFloor { get; }

    internal LayerNorm Ln1 { get; }

    public Dropout MemoryDropout { get; }

    internal LayerNorm Ln2 { get; }

    internal FeedForward Ffn { get; }

    public Dropout FfnDropout { get; }

    public Tensor Forward(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.Rank is not (2 or 3))
        {
            throw new InvalidOperationException(
                "FrogetMemoryV2 input must have shape [sequence, width] or " +
                "[batch, sequence, width].");
        }
        if (input.Shape[^1] != ModelWidth)
        {
            throw new ArgumentException(
                $"FrogetMemoryV2 input width must be {ModelWidth}.",
                nameof(input));
        }

        bool unbatched = input.Rank == 2;
        int batch = unbatched ? 1 : input.Shape[0];
        int sequence = unbatched ? input.Shape[0] : input.Shape[1];
        Tensor batchedInput = unbatched
            ? input.Reshape(1, sequence, ModelWidth)
            : input;
        Tensor projected = _memoryProjection.ForwardBatch(
            Ln1.Forward(batchedInput));
        Tensor recalled = projected.FrogetMemoryV2(
            KeyWidth,
            ValueWidth,
            RetentionFloor);
        Tensor memoryOutput = _outputProjection.ForwardBatch(recalled);
        Tensor mixed = MemoryDropout.AddResidual(
            batchedInput,
            memoryOutput);
        Tensor output = FfnDropout.AddResidual(
            mixed,
            Ffn.Forward(Ln2.Forward(mixed)));
        return unbatched
            ? output.Reshape(sequence, ModelWidth)
            : output;
    }
}
