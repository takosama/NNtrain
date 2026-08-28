namespace NNtrain;

/// <summary>
/// Selects the numeric contract used by a model independently of device.
/// </summary>
public enum TensorPrecisionMode
{
    /// <summary>Float32 storage and Float32 arithmetic.</summary>
    Float32 = 0,

    /// <summary>BFloat16 storage and BFloat16 tensor outputs.</summary>
    BFloat16 = 1,

    /// <summary>
    /// 16-bit storage with Float32 accumulation for reductions, gradients,
    /// master weights, losses, normalization, and optimizer state.
    /// </summary>
    Mix16_32 = 2,

    /// <summary>
    /// Signed Int8 tensor-wide storage. The tensor owns one Float32 scale.
    /// </summary>
    Bfp8 = 3,

    /// <summary>
    /// Signed Int8 block storage with Float32 accumulation, reductions,
    /// normalization, loss, gradients, master weights, and optimizer state.
    /// </summary>
    Mix8_32 = 4,
}

/// <summary>Canonical configuration names for precision modes.</summary>
public static class TensorPrecisionModeNames
{
    public const string Float32 = "float32";
    public const string BFloat16 = "bfloat16";
    public const string Mix16_32 = "mix16_32";
    public const string Fp16_32Alias = "fp16_32";
    public const string Bfp8 = "bfp8";
    public const string Mix8_32 = "mix8_32";
    public const string SupportedValuesDescription =
        $"'{Float32}', '{BFloat16}', '{Mix16_32}' (alias " +
        $"'{Fp16_32Alias}'), '{Bfp8}', and '{Mix8_32}'";

    public static TensorPrecisionMode Parse(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);
        if (string.Equals(value, Float32, StringComparison.OrdinalIgnoreCase))
            return TensorPrecisionMode.Float32;
        if (string.Equals(value, BFloat16, StringComparison.OrdinalIgnoreCase))
            return TensorPrecisionMode.BFloat16;
        if (string.Equals(value, Mix16_32, StringComparison.OrdinalIgnoreCase)
            || string.Equals(
                value,
                Fp16_32Alias,
                StringComparison.OrdinalIgnoreCase))
            return TensorPrecisionMode.Mix16_32;
        if (string.Equals(value, Bfp8, StringComparison.OrdinalIgnoreCase))
            return TensorPrecisionMode.Bfp8;
        if (string.Equals(value, Mix8_32, StringComparison.OrdinalIgnoreCase))
            return TensorPrecisionMode.Mix8_32;
        throw new ArgumentException(
            $"Unsupported precision mode '{value}'. Supported values are " +
            $"{SupportedValuesDescription}.",
            nameof(value));
    }

    public static string Format(TensorPrecisionMode mode)
        => mode switch
        {
            TensorPrecisionMode.Float32 => Float32,
            TensorPrecisionMode.BFloat16 => BFloat16,
            TensorPrecisionMode.Mix16_32 => Mix16_32,
            TensorPrecisionMode.Bfp8 => Bfp8,
            TensorPrecisionMode.Mix8_32 => Mix8_32,
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };
}

public static class TensorPrecisionModeExtensions
{
    public static TensorDType ToStorageDType(this TensorPrecisionMode mode)
        => mode switch
        {
            TensorPrecisionMode.Float32 => TensorDType.Float32,
            TensorPrecisionMode.BFloat16 => TensorDType.BFloat16,
            TensorPrecisionMode.Mix16_32 => TensorDType.BFloat16,
            TensorPrecisionMode.Bfp8 => TensorDType.Bfp8,
            TensorPrecisionMode.Mix8_32 => TensorDType.Bfp8,
            _ => throw new ArgumentOutOfRangeException(
                nameof(mode), mode, "Unknown tensor precision mode."),
        };

    public static TensorPrecisionMode ToPrecisionMode(this TensorDType dtype)
        => dtype switch
        {
            TensorDType.Float32 => TensorPrecisionMode.Float32,
            TensorDType.BFloat16 => TensorPrecisionMode.BFloat16,
            // Raw Float16 remains a supported low-level storage format. Its
            // training contract is the legacy form of mixed precision.
            TensorDType.Float16 => TensorPrecisionMode.Mix16_32,
            TensorDType.Bfp8 => TensorPrecisionMode.Bfp8,
            _ => throw new NotSupportedException(
                $"Tensor dtype '{dtype}' has no training precision mode."),
        };
}
