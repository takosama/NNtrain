namespace NNtrain.Runtime.Execution;

/// <summary>The supported model-level precision contracts.</summary>
public enum PrecisionMode
{
    Float32 = 0,
    BFloat16 = 1,
    Mix16_32 = 2,
    Bfp8 = 3,
    Mix8_32 = 4,
    Mix8_16 = 5,
}

/// <summary>Numeric formats used by a precision policy.</summary>
public enum NumericFormat
{
    Float32 = 0,
    BFloat16 = 1,
    Bfp8 = 2,
}

/// <summary>Formats permitted at a selected execution or publication boundary.</summary>
[Flags]
public enum NumericFormatSet
{
    Float32 = 1 << 0,
    BFloat16 = 1 << 1,
    Bfp8 = 1 << 2,
}

/// <summary>How a backend chooses among permitted non-weight paths.</summary>
public enum NonWeightSelectionPolicy
{
    Fixed = 0,
    FastestAvailable = 1,
}

/// <summary>
/// Physical operand encodings a GEMM dispatcher may select after any
/// storage decode or packing. This is deliberately separate from
/// <see cref="PrecisionPolicy.MatrixOperand"/>, which is the logical numeric
/// contract presented to the dispatcher.
/// </summary>
[Flags]
public enum GemmExecutionFormat
{
    Float32 = 1 << 0,
    BFloat16 = 1 << 1,
    Int8 = 1 << 2,
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
        GemmExecutionFormat gemmExecutionFormats,
        NumericFormat accumulation,
        NumericFormat reduction,
        NumericFormat normalization,
        NumericFormat loss,
        NumericFormat gradient,
        NumericFormat optimizerState,
        NumericFormat? masterWeight,
        NumericFormatSet? allowedActivationStorageFormats = null,
        NumericFormatSet? allowedNonWeightComputeFormats = null,
        NonWeightSelectionPolicy nonWeightSelection =
            NonWeightSelectionPolicy.Fixed,
        bool preferInt8WeightGemm = false,
        bool allowNonWeightReassociation = false)
    {
        Mode = mode;
        ParameterStorage = parameterStorage;
        ActivationStorage = activationStorage;
        ElementwiseCompute = elementwiseCompute;
        MatrixOperand = matrixOperand;
        GemmExecutionFormats = gemmExecutionFormats;
        Accumulation = accumulation;
        Reduction = reduction;
        Normalization = normalization;
        Loss = loss;
        Gradient = gradient;
        OptimizerState = optimizerState;
        MasterWeight = masterWeight;
        AllowedActivationStorageFormats = allowedActivationStorageFormats
            ?? FormatSetFor(activationStorage);
        AllowedNonWeightComputeFormats = allowedNonWeightComputeFormats
            ?? (FormatSetFor(elementwiseCompute) | FormatSetFor(accumulation));
        NonWeightSelection = nonWeightSelection;
        PreferInt8WeightGemm = preferInt8WeightGemm;
        AllowNonWeightReassociation = allowNonWeightReassociation;
    }

    public PrecisionMode Mode { get; }
    public NumericFormat ParameterStorage { get; }
    public NumericFormat ActivationStorage { get; }
    public NumericFormat ElementwiseCompute { get; }

    /// <summary>
    /// Logical matrix operand contract. BFP8 policies remain BFP8 here even
    /// when a backend decodes them to BF16 or packs them as INT8.
    /// </summary>
    public NumericFormat MatrixOperand { get; }

    /// <summary>Physical GEMM operand encodings allowed by this policy.</summary>
    public GemmExecutionFormat GemmExecutionFormats { get; }
    public NumericFormat Accumulation { get; }
    public NumericFormat Reduction { get; }
    public NumericFormat Normalization { get; }
    public NumericFormat Loss { get; }
    public NumericFormat Gradient { get; }
    public NumericFormat OptimizerState { get; }
    public NumericFormat? MasterWeight { get; }
    public bool UsesMasterWeights => MasterWeight.HasValue;

    /// <summary>
    /// Allowed activation publication formats. <see cref="ActivationStorage"/>
    /// is the default when a backend has not selected a faster route.
    /// </summary>
    public NumericFormatSet AllowedActivationStorageFormats { get; }

    /// <summary>Allowed formats for non-weight kernels and their outputs.</summary>
    public NumericFormatSet AllowedNonWeightComputeFormats { get; }

    /// <summary>Whether the backend may select the fastest allowed path.</summary>
    public NonWeightSelectionPolicy NonWeightSelection { get; }

    /// <summary>
    /// Prefer signed Int8 weight GEMM when available; a measured faster BF16
    /// route is permitted by <see cref="GemmExecutionFormats"/>.
    /// </summary>
    public bool PreferInt8WeightGemm { get; }

    /// <summary>
    /// Allows non-weight algorithms to change the FP32 summation order when
    /// choosing a faster permitted format.
    /// </summary>
    public bool AllowNonWeightReassociation { get; }

    private static NumericFormatSet FormatSetFor(NumericFormat format)
        => format switch
        {
            NumericFormat.Float32 => NumericFormatSet.Float32,
            NumericFormat.BFloat16 => NumericFormatSet.BFloat16,
            NumericFormat.Bfp8 => NumericFormatSet.Bfp8,
            _ => throw new ArgumentOutOfRangeException(nameof(format)),
        };

    public static PrecisionPolicy Float32 { get; } = new(
        PrecisionMode.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        GemmExecutionFormat.Float32,
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
        GemmExecutionFormat.BFloat16,
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
        GemmExecutionFormat.BFloat16,
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
        GemmExecutionFormat.Int8 | GemmExecutionFormat.BFloat16,
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
        GemmExecutionFormat.BFloat16,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32);

    /// <summary>
    /// Block-scaled BFP8 default storage with performance-selected Int8 or
    /// BF16 weight GEMM and physically BF16 retained gradients, master
    /// weights, and optimizer state. Non-weight kernels may use BFP8, BF16,
    /// or FP32 and publish BFP8 or BF16 activations. FP32/Int32 internal
    /// accumulators are permitted; FP32 operation order is not required.
    /// </summary>
    public static PrecisionPolicy Mix8_16 { get; } = new(
        PrecisionMode.Mix8_16,
        NumericFormat.Bfp8,
        NumericFormat.Bfp8,
        NumericFormat.Bfp8,
        NumericFormat.Bfp8,
        GemmExecutionFormat.Int8 | GemmExecutionFormat.BFloat16,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.Float32,
        NumericFormat.BFloat16,
        NumericFormat.BFloat16,
        NumericFormat.BFloat16,
        allowedActivationStorageFormats: NumericFormatSet.Bfp8
            | NumericFormatSet.BFloat16,
        allowedNonWeightComputeFormats: NumericFormatSet.Bfp8
            | NumericFormatSet.BFloat16 | NumericFormatSet.Float32,
        nonWeightSelection: NonWeightSelectionPolicy.FastestAvailable,
        preferInt8WeightGemm: true,
        allowNonWeightReassociation: true);

    public static PrecisionPolicy For(PrecisionMode mode)
        => mode switch
        {
            PrecisionMode.Float32 => Float32,
            PrecisionMode.BFloat16 => BFloat16,
            PrecisionMode.Mix16_32 => Mix16_32,
            PrecisionMode.Bfp8 => Bfp8,
            PrecisionMode.Mix8_32 => Mix8_32,
            PrecisionMode.Mix8_16 => Mix8_16,
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };

    public static PrecisionPolicy Parse(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        if (string.Equals(value, "float32", StringComparison.OrdinalIgnoreCase))
            return Float32;
        if (string.Equals(value, "bfloat16", StringComparison.OrdinalIgnoreCase))
            return BFloat16;
        if (string.Equals(value, "mix16_32", StringComparison.OrdinalIgnoreCase)
            || string.Equals(value, "fp16_32", StringComparison.OrdinalIgnoreCase))
            return Mix16_32;
        if (string.Equals(value, "bfp8", StringComparison.OrdinalIgnoreCase))
            return Bfp8;
        if (string.Equals(value, "mix8_32", StringComparison.OrdinalIgnoreCase))
            return Mix8_32;
        if (string.Equals(value, "mix8_16", StringComparison.OrdinalIgnoreCase))
            return Mix8_16;
        throw new ArgumentException(
            $"Unsupported precision policy '{value}'. Supported values are " +
            "'float32', 'bfloat16', 'mix16_32' (alias 'fp16_32'), " +
            "'bfp8', 'mix8_32', and 'mix8_16'.",
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
            PrecisionMode.Mix8_16 => "mix8_16",
            _ => throw new InvalidOperationException("Unknown precision mode."),
        };
}
