namespace NNtrain;

/// <summary>
/// Pre-normalized GPT layer backed by a stable, matrix-valued delta-rule
/// memory.
/// </summary>
public sealed class ForgetMemoryV2Layer : Module
{
    private readonly Linear _memoryProjection;
    private readonly Linear _outputProjection;

    public ForgetMemoryV2Layer(
        int modelWidth,
        int hiddenWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        Random? random = null,
        float initializationScale = 0.02f,
        float dropout = 0f,
        TensorDType dtype = TensorDType.Float16)
        : this(
            modelWidth,
            hiddenWidth,
            keyWidth,
            valueWidth,
            retentionFloor,
            random,
            initializationScale,
            dropout,
            dtype,
            useV3: false,
            useDrn: false)
    {
    }

    internal ForgetMemoryV2Layer(
        int modelWidth,
        int hiddenWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        Random? random,
        float initializationScale,
        float dropout,
        TensorDType dtype,
        bool useV3,
        bool useDrn)
        : base(dtype)
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
        UseV3 = useV3;
        UseDrn = useDrn;
        random ??= new Random(1);

        Ln1 = RegisterModule(new LayerNorm(modelWidth, dtype: dtype));
        _memoryProjection = RegisterModule(
            new Linear(
                modelWidth,
                checked(2 * keyWidth + 3 * valueWidth),
                random,
                initializationScale,
                dtype));
        _outputProjection = RegisterModule(
            new Linear(
                valueWidth,
                modelWidth,
                random,
                initializationScale,
                dtype));
        MemoryDropout = RegisterModule(new Dropout(dropout, random, dtype));
        Ln2 = RegisterModule(new LayerNorm(modelWidth, dtype: dtype));
        Ffn = RegisterModule(
            new FeedForward(
                modelWidth,
                hiddenWidth,
                random,
                initializationScale,
                dtype));
        FfnDropout = RegisterModule(new Dropout(dropout, random, dtype));
    }

    public int ModelWidth { get; }

    public int KeyWidth { get; }

    public int ValueWidth { get; }

    public float RetentionFloor { get; }

    public bool UseV3 { get; }

    public bool UseDrn { get; }

    internal LayerNorm Ln1 { get; }

    public Dropout MemoryDropout { get; }

    internal LayerNorm Ln2 { get; }

    internal FeedForward Ffn { get; }

    public Dropout FfnDropout { get; }

    /// <summary>Elements of recurrent memory this layer carries.</summary>
    internal int StateSize => checked(ValueWidth * KeyWidth);

    /// <summary>
    /// Applies the layer to <paramref name="input"/> of shape
    /// [1, sequence, width], continuing from and updating
    /// <paramref name="state"/>. Everything except the memory recurrence is
    /// identical to <see cref="Forward"/>.
    /// </summary>
    internal Tensor Continue(Tensor input, float[] state)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(state);
        if (input.Rank != 3 || input.Shape[0] != 1)
        {
            throw new InvalidOperationException(
                "Recurrent stepping requires input of shape "
                + "[1, sequence, width].");
        }
        if (input.Shape[^1] != ModelWidth)
        {
            throw new ArgumentException(
                $"ForgetMemoryV2 input width must be {ModelWidth}.",
                nameof(input));
        }

        Tensor projected = _memoryProjection.ForwardBatch(Ln1.Forward(input));
        Tensor recalled = UseDrn
            ? projected.ForgetMemoryDRNContinue(
                KeyWidth,
                ValueWidth,
                RetentionFloor,
                state)
            : UseV3
                ? projected.ForgetMemoryV3Continue(
                    KeyWidth,
                    ValueWidth,
                    RetentionFloor,
                    state)
                : projected.ForgetMemoryV2Continue(
                    KeyWidth,
                    ValueWidth,
                    RetentionFloor,
                    state);
        Tensor memoryOutput = _outputProjection.ForwardBatch(recalled);
        Tensor mixed = MemoryDropout.AddResidual(input, memoryOutput);
        return FfnDropout.AddResidual(
            mixed,
            Ffn.Forward(Ln2.Forward(mixed)));
    }

    public Tensor Forward(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.Rank is not (2 or 3))
        {
            throw new InvalidOperationException(
                "ForgetMemoryV2 input must have shape [sequence, width] or " +
                "[batch, sequence, width].");
        }
        if (input.Shape[^1] != ModelWidth)
        {
            throw new ArgumentException(
                $"ForgetMemoryV2 input width must be {ModelWidth}.",
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
        Tensor recalled = UseDrn
            ? projected.ForgetMemoryDRN(
                KeyWidth,
                ValueWidth,
                RetentionFloor)
            : UseV3
                ? projected.ForgetMemoryV3(
                    KeyWidth,
                    ValueWidth,
                    RetentionFloor)
                : projected.ForgetMemoryV2(
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
