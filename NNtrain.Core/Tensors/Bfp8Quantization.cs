namespace NNtrain;

/// <summary>Determines how many values share one BFP8 scale.</summary>
public enum Bfp8ScaleGranularity
{
    /// <summary>All values in the tensor share one scale.</summary>
    Tensor = 0,

    /// <summary>Every fixed-size contiguous block owns one scale.</summary>
    Block = 1,
}

/// <summary>
/// Stable, extensible description of a signed Int8 BFP payload. Pure
/// <c>bfp8</c> uses <see cref="TensorWide"/>; <c>mix8_32</c> uses
/// <see cref="Mix8_32"/> (128 values per block).
/// </summary>
public sealed record Bfp8QuantizationDescriptor
{
    public const int DefaultBlockSize = 128;

    public Bfp8QuantizationDescriptor(
        Bfp8ScaleGranularity granularity,
        int blockSize = 0)
    {
        if (!Enum.IsDefined(granularity))
        {
            throw new ArgumentOutOfRangeException(
                nameof(granularity), granularity, "Unknown BFP8 granularity.");
        }
        if (granularity == Bfp8ScaleGranularity.Tensor && blockSize != 0)
        {
            throw new ArgumentException(
                "Tensor-wide BFP8 does not accept a fixed block size.",
                nameof(blockSize));
        }
        if (granularity == Bfp8ScaleGranularity.Block && blockSize <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(blockSize), "Block BFP8 requires a positive block size.");
        }

        Granularity = granularity;
        BlockSize = blockSize;
    }

    public Bfp8ScaleGranularity Granularity { get; }

    /// <summary>
    /// Fixed block size, or zero when the logical block is the whole tensor.
    /// </summary>
    public int BlockSize { get; }

    public static Bfp8QuantizationDescriptor TensorWide { get; } = new(
        Bfp8ScaleGranularity.Tensor);

    public static Bfp8QuantizationDescriptor Mix8_32 { get; } = new(
        Bfp8ScaleGranularity.Block,
        DefaultBlockSize);

    public static Bfp8QuantizationDescriptor Block(int blockSize)
        => new(Bfp8ScaleGranularity.Block, blockSize);

    public int GetEffectiveBlockSize(int elementCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(elementCount);
        return Granularity == Bfp8ScaleGranularity.Tensor
            ? Math.Max(1, elementCount)
            : BlockSize;
    }

    public int GetScaleCount(int elementCount)
    {
        int effectiveBlockSize = GetEffectiveBlockSize(elementCount);
        return checked((int)(
            (elementCount + (long)effectiveBlockSize - 1)
            / effectiveBlockSize));
    }

    internal int GetScaleIndex(int elementIndex, int elementCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(elementIndex);
        if (elementIndex >= elementCount)
            throw new ArgumentOutOfRangeException(nameof(elementIndex));
        return elementIndex / GetEffectiveBlockSize(elementCount);
    }
}

/// <summary>Immutable signed Int8 payload and its Float32 scale sidecar.</summary>
public sealed class Bfp8EncodedStorage
{
    private readonly sbyte[] _payload;
    private readonly float[] _scales;

    public Bfp8EncodedStorage(
        ReadOnlySpan<sbyte> payload,
        ReadOnlySpan<float> scales,
        Bfp8QuantizationDescriptor descriptor)
        : this(payload.ToArray(), scales.ToArray(), descriptor, takeOwnership: true)
    {
    }

    internal Bfp8EncodedStorage(
        sbyte[] payload,
        float[] scales,
        Bfp8QuantizationDescriptor descriptor,
        bool takeOwnership)
    {
        ArgumentNullException.ThrowIfNull(payload);
        ArgumentNullException.ThrowIfNull(scales);
        ArgumentNullException.ThrowIfNull(descriptor);
        int expectedScales = descriptor.GetScaleCount(payload.Length);
        if (scales.Length != expectedScales)
        {
            throw new ArgumentException(
                $"BFP8 scale count must be {expectedScales} for " +
                $"{payload.Length} values.",
                nameof(scales));
        }
        for (int index = 0; index < scales.Length; index++)
        {
            if (!float.IsFinite(scales[index]) || scales[index] <= 0f)
            {
                throw new ArgumentException(
                    "Every BFP8 scale must be finite and positive.",
                    nameof(scales));
            }
        }

        _payload = takeOwnership ? payload : (sbyte[])payload.Clone();
        _scales = takeOwnership ? scales : (float[])scales.Clone();
        Descriptor = descriptor;
    }

    public ReadOnlyMemory<sbyte> Payload => _payload;
    public ReadOnlyMemory<float> Scales => _scales;
    public Bfp8QuantizationDescriptor Descriptor { get; }
    public int Count => _payload.Length;

    public TensorStorageDescriptor StorageDescriptor => new(
        TensorDType.Bfp8,
        new TensorStorageMetadata(
            TensorStorageEncoding.BlockQuantized,
            Quantization: new TensorQuantizationMetadata(
                TensorQuantizationScheme.Symmetric,
                Descriptor.GetEffectiveBlockSize(Count),
                (float[])_scales.Clone())));

    internal sbyte[] PayloadArray => _payload;
    internal float[] ScaleArray => _scales;

    internal Bfp8EncodedStorage Clone()
        => new(
            (sbyte[])_payload.Clone(),
            (float[])_scales.Clone(),
            Descriptor,
            takeOwnership: true);
}

/// <summary>Codec contract so a later scale/codebook policy can be injected.</summary>
public interface IBfp8QuantizationCodec
{
    Bfp8EncodedStorage Encode(
        ReadOnlySpan<float> source,
        Bfp8QuantizationDescriptor descriptor);

    void Decode(
        ReadOnlySpan<sbyte> payload,
        ReadOnlySpan<float> scales,
        Bfp8QuantizationDescriptor descriptor,
        Span<float> destination);
}

/// <summary>
/// CPU reference for symmetric signed Int8 BFP8. It deliberately uses the
/// same round-to-nearest-even rule as the CUDA implementation.
/// </summary>
public sealed class SymmetricBfp8QuantizationCodec : IBfp8QuantizationCodec
{
    private const float MaximumMagnitude = 127f;

    public static SymmetricBfp8QuantizationCodec Instance { get; } = new();

    private SymmetricBfp8QuantizationCodec()
    {
    }

    public Bfp8EncodedStorage Encode(
        ReadOnlySpan<float> source,
        Bfp8QuantizationDescriptor descriptor)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        int blockSize = descriptor.GetEffectiveBlockSize(source.Length);
        int scaleCount = descriptor.GetScaleCount(source.Length);
        var payload = new sbyte[source.Length];
        var scales = new float[scaleCount];

        for (int block = 0; block < scaleCount; block++)
        {
            int start = checked(block * blockSize);
            int end = Math.Min(source.Length, checked(start + blockSize));
            float maximum = 0f;
            for (int index = start; index < end; index++)
            {
                float value = source[index];
                if (!float.IsFinite(value))
                {
                    throw new ArgumentException(
                        "BFP8 encoding requires finite input values.",
                        nameof(source));
                }
                maximum = MathF.Max(maximum, MathF.Abs(value));
            }

            // A unit scale gives an all-zero block a canonical, positive
            // sidecar and avoids zero/zero in every backend.
            float scale = maximum == 0f ? 1f : maximum / MaximumMagnitude;
            scales[block] = scale;
            for (int index = start; index < end; index++)
            {
                float rounded = MathF.Round(
                    source[index] / scale,
                    MidpointRounding.ToEven);
                payload[index] = (sbyte)Math.Clamp(
                    (int)rounded,
                    -127,
                    127);
            }
        }

        return new Bfp8EncodedStorage(
            payload,
            scales,
            descriptor,
            takeOwnership: true);
    }

    public void Decode(
        ReadOnlySpan<sbyte> payload,
        ReadOnlySpan<float> scales,
        Bfp8QuantizationDescriptor descriptor,
        Span<float> destination)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        if (destination.Length != payload.Length)
        {
            throw new ArgumentException(
                "BFP8 payload and destination lengths must match.",
                nameof(destination));
        }
        int scaleCount = descriptor.GetScaleCount(payload.Length);
        if (scales.Length != scaleCount)
        {
            throw new ArgumentException(
                "BFP8 scale count does not match the descriptor.",
                nameof(scales));
        }

        int blockSize = descriptor.GetEffectiveBlockSize(payload.Length);
        for (int block = 0; block < scaleCount; block++)
        {
            float scale = scales[block];
            if (!float.IsFinite(scale) || scale <= 0f)
            {
                throw new ArgumentException(
                    "Every BFP8 scale must be finite and positive.",
                    nameof(scales));
            }
            int start = checked(block * blockSize);
            int end = Math.Min(payload.Length, checked(start + blockSize));
            for (int index = start; index < end; index++)
                destination[index] = payload[index] * scale;
        }
    }
}

/// <summary>Canonical BFP8 codec selection point.</summary>
public static class Bfp8QuantizationCodec
{
    public static IBfp8QuantizationCodec Default { get; }
        = SymmetricBfp8QuantizationCodec.Instance;
}
