namespace NNtrain;

/// <summary>
/// Describes how logical tensor values are represented in a physical payload.
/// </summary>
/// <remarks>
/// <para>
/// This is a description and validation boundary, not a codec. The current
/// runtime implements only raw <see cref="TensorDType.Float32"/> and
/// <see cref="TensorDType.Float16"/> and <see cref="TensorDType.BFloat16"/>
/// storage. The remaining dtypes can be
/// described here before their codecs, kernels, and serialization are added.
/// </para>
/// <para>
/// Payload byte counts exclude quantization sidecars. Use
/// <see cref="GetAuxiliaryByteLength"/> or <see cref="GetTotalByteLength"/>
/// when accounting for per-block scales and zero points.
/// </para>
/// </remarks>
public sealed record TensorStorageDescriptor(
    TensorDType DType,
    TensorStorageMetadata? Metadata = null)
{
    /// <summary>Gets the effective metadata, using raw native storage by default.</summary>
    public TensorStorageMetadata EffectiveMetadata
        => Metadata ?? TensorStorageMetadata.Raw;

    /// <summary>
    /// Gets whether the descriptor is consumable by the current Tensor runtime.
    /// A false value does not make the descriptor invalid; it means that a
    /// future codec/kernel is still required.
    /// </summary>
    public bool IsSupportedByCurrentRuntime
        => TensorDTypeContract.IsImplemented(DType)
            && EffectiveMetadata.IsRaw;

    /// <summary>Validates this descriptor for a logical element count.</summary>
    public void Validate(int elementCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(elementCount);
        if (!Enum.IsDefined(DType))
        {
            throw new ArgumentOutOfRangeException(
                nameof(DType),
                DType,
                "Unknown tensor dtype.");
        }

        TensorStorageMetadata metadata = EffectiveMetadata;
        metadata.Validate(elementCount);

        switch (DType)
        {
            case TensorDType.Float32:
            case TensorDType.Float16:
            case TensorDType.BFloat16:
                RequireNative(DType, metadata);
                return;

            case TensorDType.Float8E4M3Fn:
            case TensorDType.Float8E5M2:
                RequireFloat8Layout(DType, metadata);
                return;

            case TensorDType.Float4:
                RequirePackedBits(DType, metadata, 4);
                return;

            case TensorDType.Float2:
                RequirePackedBits(DType, metadata, 2);
                return;

            case TensorDType.Ternary1Bit58:
                RequireTernaryLayout(metadata);
                return;

            default:
                throw new ArgumentOutOfRangeException(
                    nameof(DType),
                    DType,
                    "Unknown tensor dtype.");
        }
    }

    /// <summary>
    /// Gets the bytes occupied by encoded values, excluding scale and zero-point
    /// sidecars.
    /// </summary>
    public long GetPayloadByteLength(int elementCount)
    {
        Validate(elementCount);
        TensorPackingMetadata? packing = EffectiveMetadata.Packing;
        if (packing is not null)
            return packing.GetPayloadByteLength(elementCount);

        int bytesPerValue = DType switch
        {
            TensorDType.Float32 => sizeof(float),
            TensorDType.Float16 => sizeof(ushort),
            TensorDType.BFloat16 => sizeof(ushort),
            TensorDType.Float8E4M3Fn or TensorDType.Float8E5M2 => sizeof(byte),
            _ => throw new InvalidOperationException(
                $"Tensor dtype '{DType}' requires packing metadata."),
        };
        return checked((long)elementCount * bytesPerValue);
    }

    /// <summary>
    /// Gets bytes occupied by block scales and zero points, excluding the value
    /// payload itself.
    /// </summary>
    public long GetAuxiliaryByteLength(int elementCount)
    {
        Validate(elementCount);
        TensorQuantizationMetadata? quantization =
            EffectiveMetadata.Quantization;
        if (quantization is null)
            return 0;

        return checked(
            (long)quantization.Scales.Length * sizeof(float)
            + (long)(quantization.ZeroPoints?.Length ?? 0) * sizeof(int));
    }

    /// <summary>Gets value-payload and quantization-sidecar bytes together.</summary>
    public long GetTotalByteLength(int elementCount)
        => checked(
            GetPayloadByteLength(elementCount)
            + GetAuxiliaryByteLength(elementCount));

    private static void RequireNative(
        TensorDType dtype,
        TensorStorageMetadata metadata)
    {
        if (!metadata.IsRaw)
        {
            throw new ArgumentException(
                $"Tensor dtype '{dtype}' only supports raw native storage.",
                nameof(metadata));
        }
    }

    private static void RequireFloat8Layout(
        TensorDType dtype,
        TensorStorageMetadata metadata)
    {
        if (metadata.Encoding is TensorStorageEncoding.Native
            or TensorStorageEncoding.BlockQuantized)
        {
            return;
        }

        if (metadata.Packing?.BitsPerValue == 8)
            return;

        throw new ArgumentException(
            $"Tensor dtype '{dtype}' requires native bytes or 8-bit packing.",
            nameof(metadata));
    }

    private static void RequirePackedBits(
        TensorDType dtype,
        TensorStorageMetadata metadata,
        int expectedBits)
    {
        if (metadata.Encoding is not TensorStorageEncoding.Packed
            and not TensorStorageEncoding.PackedBlockQuantized
            || metadata.Packing?.BitsPerValue != expectedBits)
        {
            throw new ArgumentException(
                $"Tensor dtype '{dtype}' requires {expectedBits}-bit packing metadata.",
                nameof(metadata));
        }
    }

    private static void RequireTernaryLayout(TensorStorageMetadata metadata)
    {
        RequirePackedBits(TensorDType.Ternary1Bit58, metadata, 2);
        double expectedBits = Math.Log2(3d);
        if (Math.Abs(
            metadata.Packing!.LogicalBitsPerValue - expectedBits) > 1e-12)
        {
            throw new ArgumentException(
                "Ternary storage uses two-bit codes with an effective " +
                "log2(3) bits per value.",
                nameof(metadata));
        }

        if (metadata.Quantization is not null
            && metadata.Quantization.Scheme
                != TensorQuantizationScheme.Ternary)
        {
            throw new ArgumentException(
                "Ternary storage requires ternary quantization metadata.",
                nameof(metadata));
        }
    }
}

/// <summary>Physical encoding family for a tensor value payload.</summary>
public enum TensorStorageEncoding
{
    /// <summary>One native storage element per logical value.</summary>
    Native = 0,

    /// <summary>Several fixed-width encoded values are packed into each unit.</summary>
    Packed = 1,

    /// <summary>Native elements plus per-block quantization sidecars.</summary>
    BlockQuantized = 2,

    /// <summary>Packed values plus per-block quantization sidecars.</summary>
    PackedBlockQuantized = 3,
}

/// <summary>Bit ordering for encoded values inside a storage unit.</summary>
public enum TensorPackingOrder
{
    /// <summary>The first logical value occupies the least-significant bits.</summary>
    LeastSignificantBitFirst = 0,

    /// <summary>The first logical value occupies the most-significant bits.</summary>
    MostSignificantBitFirst = 1,
}

/// <summary>Quantization rule used for every logical block.</summary>
public enum TensorQuantizationScheme
{
    /// <summary>Values are scaled around zero and do not use zero points.</summary>
    Symmetric = 0,

    /// <summary>Values use an affine scale and integer zero point.</summary>
    Affine = 1,

    /// <summary>Values use the ternary codebook {-1, 0, +1} and a scale.</summary>
    Ternary = 2,
}

/// <summary>
/// Defines the packed representation of fixed-width value codes.
/// </summary>
/// <param name="BitsPerValue">Physical bits used by each stored code.</param>
/// <param name="StorageUnitBits">Width of each independently packed unit.</param>
/// <param name="BitOrder">Order of values inside each unit.</param>
/// <param name="EffectiveBitsPerValue">
/// Optional information-rate target. Ternary values, for example, use two-bit
/// codes today while reporting log2(3) effective bits.
/// </param>
public sealed record TensorPackingMetadata(
    int BitsPerValue,
    int StorageUnitBits = 8,
    TensorPackingOrder BitOrder = TensorPackingOrder.LeastSignificantBitFirst,
    double? EffectiveBitsPerValue = null)
{
    /// <summary>Gets the information-rate target, or physical code width.</summary>
    public double LogicalBitsPerValue
        => EffectiveBitsPerValue ?? BitsPerValue;

    /// <summary>Gets how many logical codes fit into one packed unit.</summary>
    public int ValuesPerStorageUnit => StorageUnitBits / BitsPerValue;

    internal void Validate()
    {
        if (BitsPerValue <= 0 || BitsPerValue > StorageUnitBits)
        {
            throw new ArgumentOutOfRangeException(
                nameof(BitsPerValue),
                "Packed values must fit inside their storage unit.");
        }
        if (StorageUnitBits is not 8 and not 16 and not 32 and not 64)
        {
            throw new ArgumentOutOfRangeException(
                nameof(StorageUnitBits),
                "Storage units must be 8, 16, 32, or 64 bits.");
        }
        if (StorageUnitBits % BitsPerValue != 0)
        {
            throw new ArgumentException(
                "BitsPerValue must divide StorageUnitBits.",
                nameof(BitsPerValue));
        }
        if (!Enum.IsDefined(BitOrder))
        {
            throw new ArgumentOutOfRangeException(
                nameof(BitOrder),
                "Unknown tensor packing order.");
        }
        if (EffectiveBitsPerValue is { } effective
            && (double.IsNaN(effective)
                || double.IsInfinity(effective)
                || effective <= 0d
                || effective > BitsPerValue))
        {
            throw new ArgumentOutOfRangeException(
                nameof(EffectiveBitsPerValue),
                "Effective bits must be finite, positive, and no larger " +
                "than the physical code width.");
        }
    }

    internal long GetPayloadByteLength(int elementCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(elementCount);
        Validate();
        long units = (elementCount + (long)ValuesPerStorageUnit - 1)
            / ValuesPerStorageUnit;
        return checked(units * (StorageUnitBits / 8));
    }
}

/// <summary>
/// Holds block scales and optional zero points for a quantized payload.
/// </summary>
public sealed record TensorQuantizationMetadata(
    TensorQuantizationScheme Scheme,
    int BlockSize,
    float[] Scales,
    int[]? ZeroPoints = null)
{
    internal void Validate(int elementCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(elementCount);
        if (!Enum.IsDefined(Scheme))
        {
            throw new ArgumentOutOfRangeException(
                nameof(Scheme),
                "Unknown tensor quantization scheme.");
        }
        if (BlockSize <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(BlockSize),
                "Block size must be positive.");
        }
        if (Scales is null)
            throw new ArgumentNullException(nameof(Scales));

        int expectedBlockCount = checked(
            (int)((elementCount + (long)BlockSize - 1) / BlockSize));
        if (Scales.Length != expectedBlockCount)
        {
            throw new ArgumentException(
                "Scale count must match the number of logical blocks.",
                nameof(Scales));
        }
        for (int index = 0; index < Scales.Length; index++)
        {
            if (!float.IsFinite(Scales[index]) || Scales[index] <= 0f)
            {
                throw new ArgumentException(
                    "Every quantization scale must be finite and positive.",
                    nameof(Scales));
            }
        }

        if (Scheme == TensorQuantizationScheme.Affine)
        {
            if (ZeroPoints is null || ZeroPoints.Length != expectedBlockCount)
            {
                throw new ArgumentException(
                    "Affine quantization requires one zero point per block.",
                    nameof(ZeroPoints));
            }
            return;
        }

        if (ZeroPoints is not null)
        {
            throw new ArgumentException(
                "Symmetric and ternary quantization do not use zero points.",
                nameof(ZeroPoints));
        }
    }
}

/// <summary>
/// Optional physical-layout metadata for a tensor payload.
/// </summary>
/// <remarks>
/// <see cref="Raw"/> is implicit when this value is omitted from a
/// <see cref="ModuleParameterState"/>. Keeping that default implicit preserves
/// existing JSON and SafeTensors payloads byte-for-byte.
/// </remarks>
public sealed record TensorStorageMetadata(
    TensorStorageEncoding Encoding,
    TensorPackingMetadata? Packing = null,
    TensorQuantizationMetadata? Quantization = null)
{
    /// <summary>Native unquantized storage used by Float32 and Float16 today.</summary>
    public static TensorStorageMetadata Raw { get; } = new(
        TensorStorageEncoding.Native);

    /// <summary>Gets whether this metadata carries no layout sidecars.</summary>
    public bool IsRaw
        => Encoding == TensorStorageEncoding.Native
            && Packing is null
            && Quantization is null;

    internal void Validate(int elementCount)
    {
        if (!Enum.IsDefined(Encoding))
        {
            throw new ArgumentOutOfRangeException(
                nameof(Encoding),
                "Unknown tensor storage encoding.");
        }

        Packing?.Validate();
        Quantization?.Validate(elementCount);

        bool hasPacking = Packing is not null;
        bool hasQuantization = Quantization is not null;
        switch (Encoding)
        {
            case TensorStorageEncoding.Native when hasPacking || hasQuantization:
                throw new ArgumentException(
                    "Native storage cannot include packing or quantization " +
                    "metadata.",
                    nameof(Encoding));

            case TensorStorageEncoding.Packed when !hasPacking || hasQuantization:
                throw new ArgumentException(
                    "Packed storage requires packing metadata and cannot " +
                    "include quantization sidecars.",
                    nameof(Encoding));

            case TensorStorageEncoding.BlockQuantized
                when hasPacking || !hasQuantization:
                throw new ArgumentException(
                    "Block-quantized storage requires quantization metadata " +
                    "and cannot include packing metadata.",
                    nameof(Encoding));

            case TensorStorageEncoding.PackedBlockQuantized
                when !hasPacking || !hasQuantization:
                throw new ArgumentException(
                    "Packed block-quantized storage requires both packing " +
                    "and quantization metadata.",
                    nameof(Encoding));
        }
    }
}
