namespace NNtrain.Runtime.Execution;

/// <summary>The supported model-level precision contracts.</summary>
public enum PrecisionMode
{
    Float32 = 0,
    BFloat16 = 1,
    Mix16_32 = 2,
    Bfp8 = 3,
    Mix8_32 = 4,
}

/// <summary>Numeric formats used by a precision policy.</summary>
public enum NumericFormat
{
    Float32 = 0,
    BFloat16 = 1,
    Bfp8 = 2,
}

/// <summary>
/// Central numeric contract for storage, kernels, stable reductions and
/// optimizer state. Backends consume this policy instead of inferring
/// arithmetic from physical tensor storage.
/// </summary>
public sealed record PrecisionPolicy
{
    private PrecisionPolicy(
        PrecisionMode mode,
        NumericFormat parameterStorage,
        NumericFormat activationStorage,
        NumericFormat elementwiseCompute,
        NumericFormat matrixOperand,
        NumericFormat accumulation,
        NumericFormat reduction,
        NumericFormat normalization,
        NumericFormat loss,
        NumericFormat gradient,
        NumericFormat optimizerState,
        NumericFormat? masterWeight)
    {
        Mode = mode;
        ParameterStorage = parameterStorage;
        ActivationStorage = activationStorage;
        ElementwiseCompute = elementwiseCompute;
        MatrixOperand = matrixOperand;
        Accumulation = accumulation;
        Reduction = reduction;
        Normalization = normalization;
        Loss = loss;
        Gradient = gradient;
        OptimizerState = optimizerState;
        MasterWeight = masterWeight;
    }

    public PrecisionMode Mode { get; }
    public NumericFormat ParameterStorage { get; }
    public NumericFormat ActivationStorage { get; }
    public NumericFormat ElementwiseCompute { get; }
    public NumericFormat MatrixOperand { get; }
    public NumericFormat Accumulation { get; }
    public NumericFormat Reduction { get; }
    public NumericFormat Normalization { get; }
    public NumericFormat Loss { get; }
    public NumericFormat Gradient { get; }
    public NumericFormat OptimizerState { get; }
    public NumericFormat? MasterWeight { get; }
    public bool UsesMasterWeights => MasterWeight.HasValue;

    public static PrecisionPolicy Float32 { get; } = new(
        PrecisionMode.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        masterWeight: null);

    public static PrecisionPolicy BFloat16 { get; } = new(
        PrecisionMode.BFloat16,
        NumericFormat.BFloat16,
        NumericFormat.BFloat16,
        NumericFormat.BFloat16,
        NumericFormat.BFloat16,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.BFloat16,
        NumericFormat.BFloat16,
        masterWeight: null);

    public static PrecisionPolicy Mix16_32 { get; } = new(
        PrecisionMode.Mix16_32,
        NumericFormat.BFloat16,
        NumericFormat.BFloat16,
        NumericFormat.BFloat16,
        NumericFormat.BFloat16,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32);

    public static PrecisionPolicy Bfp8 { get; } = new(
        PrecisionMode.Bfp8,
        NumericFormat.Bfp8,
        NumericFormat.Bfp8,
        NumericFormat.Bfp8,
        NumericFormat.Bfp8,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Bfp8,
        NumericFormat.Bfp8,
        masterWeight: null);

    public static PrecisionPolicy Mix8_32 { get; } = new(
        PrecisionMode.Mix8_32,
        NumericFormat.Bfp8,
        NumericFormat.Bfp8,
        NumericFormat.Bfp8,
        NumericFormat.Bfp8,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32);

    public static PrecisionPolicy For(PrecisionMode mode)
        => mode switch
        {
            PrecisionMode.Float32 => Float32,
            PrecisionMode.BFloat16 => BFloat16,
            PrecisionMode.Mix16_32 => Mix16_32,
            PrecisionMode.Bfp8 => Bfp8,
            PrecisionMode.Mix8_32 => Mix8_32,
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };

    public static PrecisionPolicy Parse(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        if (string.Equals(value, "float32", StringComparison.OrdinalIgnoreCase))
            return Float32;
        if (string.Equals(value, "bfloat16", StringComparison.OrdinalIgnoreCase))
            return BFloat16;
        if (string.Equals(value, "mix16_32", StringComparison.OrdinalIgnoreCase))
            return Mix16_32;
        if (string.Equals(value, "bfp8", StringComparison.OrdinalIgnoreCase))
            return Bfp8;
        if (string.Equals(value, "mix8_32", StringComparison.OrdinalIgnoreCase))
            return Mix8_32;
        throw new ArgumentException(
            $"Unsupported precision policy '{value}'.",
            nameof(value));
    }

    public override string ToString()
        => Mode switch
        {
            PrecisionMode.Float32 => "float32",
            PrecisionMode.BFloat16 => "bfloat16",
            PrecisionMode.Mix16_32 => "mix16_32",
            PrecisionMode.Bfp8 => "bfp8",
            PrecisionMode.Mix8_32 => "mix8_32",
            _ => throw new InvalidOperationException("Unknown precision mode."),
        };
}
