using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace NNtrain;

/// <summary>
/// Owns the physical bytes of a tensor independently from the dtype used for
/// computation and accumulation.
/// </summary>
/// <remarks>
/// Only one backing array is retained. Float16 tensors therefore do not keep
/// a hidden Float32 mirror. Trainable parameters keep their optional Float32
/// master weight separately in <see cref="Tensor"/>.
/// </remarks>
internal sealed class TensorStorage : IList<float>, IReadOnlyList<float>
{
    private readonly float[]? _float32;
    private readonly Half[]? _float16;
    private readonly ushort[]? _bfloat16;

    private TensorStorage(float[] values)
    {
        _float32 = values;
        DType = TensorDType.Float32;
        Count = values.Length;
    }

    private TensorStorage(Half[] values)
    {
        _float16 = values;
        DType = TensorDType.Float16;
        Count = values.Length;
    }

    private TensorStorage(ushort[] values)
    {
        _bfloat16 = values;
        DType = TensorDType.BFloat16;
        Count = values.Length;
    }

    internal TensorDType DType { get; }

    public int Count { get; }

    public bool IsReadOnly => true;

    internal int Length => Count;

    internal int ByteLength => DType switch
    {
        TensorDType.Float32 => checked(Count * sizeof(float)),
        TensorDType.Float16 => checked(Count * sizeof(ushort)),
        TensorDType.BFloat16 => checked(Count * sizeof(ushort)),
        _ => throw UnsupportedDType(),
    };

    public float this[int index]
    {
        get
        {
            ArgumentOutOfRangeException.ThrowIfNegative(index);
            if (index >= Count)
                throw new ArgumentOutOfRangeException(nameof(index));

            return DType switch
            {
                TensorDType.Float32 => _float32![index],
                TensorDType.Float16 => (float)_float16![index],
                TensorDType.BFloat16 =>
                    TensorStorageCodec.DecodeBFloat16(_bfloat16![index]),
                _ => throw UnsupportedDType(),
            };
        }
        set => throw new NotSupportedException(
            "Tensor data views are read-only.");
    }

    internal static TensorStorage Create(
        ReadOnlySpan<float> values,
        TensorDType dtype)
    {
        TensorDTypeContract.ValidateImplemented(dtype, nameof(dtype));
        if (dtype == TensorDType.Float32)
            return new TensorStorage(values.ToArray());

        if (dtype == TensorDType.Float16)
        {
            var half = new Half[values.Length];
            TensorStorageCodec.EncodeFloat16(values, half);
            return new TensorStorage(half);
        }

        var bfloat16 = new ushort[values.Length];
        TensorStorageCodec.EncodeBFloat16(values, bfloat16);
        return new TensorStorage(bfloat16);
    }

    internal static TensorStorage FromOwnedFloat32(float[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        return new TensorStorage(values);
    }

    internal static TensorStorage FromOwnedFloat16(Half[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        return new TensorStorage(values);
    }

    internal static TensorStorage CreateUninitialized(
        int length,
        TensorDType dtype)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        TensorDTypeContract.ValidateImplemented(dtype, nameof(dtype));
        return dtype switch
        {
            TensorDType.Float32 => new TensorStorage(new float[length]),
            TensorDType.Float16 => new TensorStorage(new Half[length]),
            TensorDType.BFloat16 => new TensorStorage(new ushort[length]),
            _ => throw new NotSupportedException(
                $"Tensor storage dtype '{dtype}' is not implemented."),
        };
    }

    internal TensorStorage Clone()
        => DType switch
        {
            TensorDType.Float32 => new TensorStorage(
                (float[])_float32!.Clone()),
            TensorDType.Float16 => new TensorStorage(
                (Half[])_float16!.Clone()),
            TensorDType.BFloat16 => new TensorStorage(
                (ushort[])_bfloat16!.Clone()),
            _ => throw UnsupportedDType(),
        };

    internal float[] ToFloat32Array()
    {
        if (_float32 is not null)
            return (float[])_float32.Clone();

        var result = new float[Count];
        if (_float16 is not null)
            TensorStorageCodec.DecodeFloat16(_float16, result);
        else
            TensorStorageCodec.DecodeBFloat16(_bfloat16!, result);
        return result;
    }

    internal void CopyTo(Span<float> destination)
    {
        if (destination.Length < Count)
        {
            throw new ArgumentException(
                "Destination is shorter than the tensor storage.",
                nameof(destination));
        }

        if (_float32 is not null)
        {
            _float32.AsSpan().CopyTo(destination);
            return;
        }

        if (_float16 is not null)
            TensorStorageCodec.DecodeFloat16(_float16, destination);
        else
            TensorStorageCodec.DecodeBFloat16(_bfloat16!, destination);
    }

    internal void CopyRangeTo(
        int sourceOffset,
        Span<float> destination)
    {
        if ((uint)sourceOffset > (uint)Count
            || destination.Length > Count - sourceOffset)
        {
            throw new ArgumentOutOfRangeException(nameof(sourceOffset));
        }

        if (_float32 is not null)
        {
            _float32.AsSpan(sourceOffset, destination.Length)
                .CopyTo(destination);
            return;
        }

        if (_float16 is not null)
        {
            TensorStorageCodec.DecodeFloat16(
                _float16.AsSpan(sourceOffset, destination.Length),
                destination);
        }
        else
        {
            TensorStorageCodec.DecodeBFloat16(
                _bfloat16!.AsSpan(sourceOffset, destination.Length),
                destination);
        }
    }

    internal void CopyRangeTo(
        int sourceOffset,
        TensorStorage destination,
        int destinationOffset,
        int length)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if ((uint)sourceOffset > (uint)Count
            || length < 0
            || length > Count - sourceOffset)
        {
            throw new ArgumentOutOfRangeException(nameof(sourceOffset));
        }
        if ((uint)destinationOffset > (uint)destination.Count
            || length > destination.Count - destinationOffset)
        {
            throw new ArgumentOutOfRangeException(nameof(destinationOffset));
        }

        if (_float32 is not null && destination._float32 is not null)
        {
            _float32.AsSpan(sourceOffset, length).CopyTo(
                destination._float32.AsSpan(destinationOffset, length));
            return;
        }
        if (_float16 is not null && destination._float16 is not null)
        {
            _float16.AsSpan(sourceOffset, length).CopyTo(
                destination._float16.AsSpan(destinationOffset, length));
            return;
        }
        if (_float16 is not null && destination._float32 is not null)
        {
            TensorStorageCodec.DecodeFloat16(
                _float16.AsSpan(sourceOffset, length),
                destination._float32.AsSpan(destinationOffset, length));
            return;
        }
        if (_float32 is not null && destination._float16 is not null)
        {
            TensorStorageCodec.EncodeFloat16(
                _float32.AsSpan(sourceOffset, length),
                destination._float16.AsSpan(destinationOffset, length));
            return;
        }

        if (_bfloat16 is not null && destination._bfloat16 is not null)
        {
            _bfloat16.AsSpan(sourceOffset, length).CopyTo(
                destination._bfloat16.AsSpan(destinationOffset, length));
            return;
        }

        var temporary = new float[length];
        CopyRangeTo(sourceOffset, temporary);
        destination.CopyRangeFromFloat32(
            temporary,
            destinationOffset);
        return;
    }

    internal void Transpose2DTo(
        TensorStorage destination,
        int rows,
        int columns)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (rows <= 0 || columns <= 0 || checked(rows * columns) != Count)
            throw new ArgumentException("Transpose dimensions are invalid.");
        if (destination.Count != Count || destination.DType != DType)
        {
            throw new ArgumentException(
                "Transpose destination must have the same length and dtype.",
                nameof(destination));
        }

        const int BlockSize = 32;
        if (_float32 is not null)
        {
            float[] output = destination._float32!;
            for (int rowBlock = 0; rowBlock < rows; rowBlock += BlockSize)
            {
                int rowEnd = Math.Min(rowBlock + BlockSize, rows);
                for (int columnBlock = 0;
                    columnBlock < columns;
                    columnBlock += BlockSize)
                {
                    int columnEnd = Math.Min(
                        columnBlock + BlockSize,
                        columns);
                    for (int row = rowBlock; row < rowEnd; row++)
                    {
                        int sourceRow = row * columns;
                        for (int column = columnBlock;
                            column < columnEnd;
                            column++)
                        {
                            output[column * rows + row] =
                                _float32[sourceRow + column];
                        }
                    }
                }
            }
            return;
        }

        if (_float16 is null)
        {
            ushort[] bfloatOutput = destination._bfloat16!;
            for (int row = 0; row < rows; row++)
            {
                int sourceRow = row * columns;
                for (int column = 0; column < columns; column++)
                    bfloatOutput[column * rows + row] =
                        _bfloat16![sourceRow + column];
            }
            return;
        }

        Half[] halfOutput = destination._float16!;
        for (int rowBlock = 0; rowBlock < rows; rowBlock += BlockSize)
        {
            int rowEnd = Math.Min(rowBlock + BlockSize, rows);
            for (int columnBlock = 0;
                columnBlock < columns;
                columnBlock += BlockSize)
            {
                int columnEnd = Math.Min(columnBlock + BlockSize, columns);
                for (int row = rowBlock; row < rowEnd; row++)
                {
                    int sourceRow = row * columns;
                    for (int column = columnBlock;
                        column < columnEnd;
                        column++)
                    {
                        halfOutput[column * rows + row] =
                            _float16![sourceRow + column];
                    }
                }
            }
        }
    }

    internal void CopyFrom(ReadOnlySpan<float> source)
    {
        if (source.Length != Count)
        {
            throw new ArgumentException(
                "Source length does not match the tensor storage.",
                nameof(source));
        }

        if (_float32 is not null)
        {
            source.CopyTo(_float32);
            return;
        }

        if (_float16 is not null)
            TensorStorageCodec.EncodeFloat16(source, _float16);
        else
            TensorStorageCodec.EncodeBFloat16(source, _bfloat16!);
    }

    private void CopyRangeFromFloat32(
        ReadOnlySpan<float> source,
        int destinationOffset)
    {
        if (_float32 is not null)
            source.CopyTo(_float32.AsSpan(destinationOffset));
        else if (_float16 is not null)
            TensorStorageCodec.EncodeFloat16(
                source,
                _float16.AsSpan(destinationOffset));
        else
            TensorStorageCodec.EncodeBFloat16(
                source,
                _bfloat16!.AsSpan(destinationOffset));
    }

    internal float[] GetMutableFloat32Buffer()
        => _float32
            ?? throw new InvalidOperationException(
                "Only Float32 storage exposes a mutable Float32 buffer.");

    internal bool TryGetFloat32Buffer(out float[] values)
    {
        values = _float32!;
        return _float32 is not null;
    }

    internal bool TryGetFloat16Buffer(out Half[] values)
    {
        values = _float16!;
        return _float16 is not null;
    }

    internal Vector256<float> LoadVector256(int offset)
    {
        if ((uint)offset > (uint)(Count - Vector256<float>.Count))
            throw new ArgumentOutOfRangeException(nameof(offset));

        if (_float32 is not null)
            return Vector256.LoadUnsafe(ref _float32[offset]);
        if (_float16 is not null)
            return TensorStorageCodec.LoadFloat16Vector256(_float16, offset);
        return TensorStorageCodec.LoadBFloat16Vector256(_bfloat16!, offset);
    }

    internal Vector128<float> LoadVector128(int offset)
    {
        if ((uint)offset > (uint)(Count - Vector128<float>.Count))
            throw new ArgumentOutOfRangeException(nameof(offset));

        if (_float32 is not null)
            return Vector128.LoadUnsafe(ref _float32[offset]);
        if (_float16 is not null)
            return TensorStorageCodec.LoadFloat16Vector128(_float16, offset);
        return TensorStorageCodec.LoadBFloat16Vector128(_bfloat16!, offset);
    }

    public IEnumerator<float> GetEnumerator()
    {
        for (int index = 0; index < Count; index++)
            yield return this[index];
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    public int IndexOf(float item)
    {
        for (int index = 0; index < Count; index++)
        {
            if (this[index].Equals(item))
                return index;
        }
        return -1;
    }

    public bool Contains(float item) => IndexOf(item) >= 0;

    public void CopyTo(float[] array, int arrayIndex)
    {
        ArgumentNullException.ThrowIfNull(array);
        CopyTo(array.AsSpan(arrayIndex));
    }

    public void Add(float item) => throw ReadOnlyMutation();

    public void Clear() => throw ReadOnlyMutation();

    public void Insert(int index, float item) => throw ReadOnlyMutation();

    public bool Remove(float item) => throw ReadOnlyMutation();

    public void RemoveAt(int index) => throw ReadOnlyMutation();

    private static NotSupportedException ReadOnlyMutation()
        => new("Tensor data views are read-only.");

    private NotSupportedException UnsupportedDType()
        => new($"Tensor storage dtype '{DType}' is not implemented.");
}

/// <summary>
/// Converts physical storage blocks to the Float32 vectors used by current
/// kernels. Future FP8 and packed low-bit codecs plug into this boundary.
/// </summary>
internal static class TensorStorageCodec
{
    internal static void EncodeBFloat16(
        ReadOnlySpan<float> source,
        Span<ushort> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("BFloat16 codec lengths must match.");

        for (int index = 0; index < source.Length; index++)
            destination[index] = EncodeBFloat16(source[index]);
    }

    internal static void DecodeBFloat16(
        ReadOnlySpan<ushort> source,
        Span<float> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("BFloat16 codec lengths must match.");

        for (int index = 0; index < source.Length; index++)
            destination[index] = DecodeBFloat16(source[index]);
    }

    internal static ushort EncodeBFloat16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint absolute = bits & 0x7FFFFFFFu;
        if (absolute > 0x7F800000u)
            return (ushort)((bits >> 16) | 0x0040u);

        uint rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
        return (ushort)(rounded >> 16);
    }

    internal static float DecodeBFloat16(ushort value)
        => BitConverter.UInt32BitsToSingle((uint)value << 16);

    internal static Vector256<float> LoadBFloat16Vector256(
        ushort[] source,
        int offset)
        => Vector256.Create(
            DecodeBFloat16(source[offset]),
            DecodeBFloat16(source[offset + 1]),
            DecodeBFloat16(source[offset + 2]),
            DecodeBFloat16(source[offset + 3]),
            DecodeBFloat16(source[offset + 4]),
            DecodeBFloat16(source[offset + 5]),
            DecodeBFloat16(source[offset + 6]),
            DecodeBFloat16(source[offset + 7]));

    internal static Vector128<float> LoadBFloat16Vector128(
        ushort[] source,
        int offset)
        => Vector128.Create(
            DecodeBFloat16(source[offset]),
            DecodeBFloat16(source[offset + 1]),
            DecodeBFloat16(source[offset + 2]),
            DecodeBFloat16(source[offset + 3]));

    internal static void EncodeFloat16(
        ReadOnlySpan<float> source,
        Span<Half> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Float16 codec lengths must match.");

        int index = 0;
        if (Avx2.IsSupported
            && Sse41.IsSupported
            && Vector256.IsHardwareAccelerated)
        {
            int vectorizedLength = source.Length
                - source.Length % Vector256<float>.Count;
            ref float sourceReference = ref MemoryMarshal.GetReference(source);
            ref Half destinationReference =
                ref MemoryMarshal.GetReference(destination);
            for (; index < vectorizedLength;
                index += Vector256<float>.Count)
            {
                Vector256<float> values = Vector256.LoadUnsafe(
                    ref Unsafe.Add(ref sourceReference, index));
                if (!TryStoreFloat16Vector256(
                    values,
                    ref Unsafe.Add(ref destinationReference, index)))
                {
                    for (int lane = 0;
                        lane < Vector256<float>.Count;
                        lane++)
                    {
                        Unsafe.Add(ref destinationReference, index + lane) =
                            checked((Half)values.GetElement(lane));
                    }
                }
            }
        }

        for (; index < source.Length; index++)
            destination[index] = checked((Half)source[index]);
    }

    internal static void DecodeFloat16(
        ReadOnlySpan<Half> source,
        Span<float> destination)
    {
        if (source.Length != destination.Length)
            throw new ArgumentException("Float16 codec lengths must match.");

        int index = 0;
        if (Avx2.IsSupported && Vector256.IsHardwareAccelerated)
        {
            int vectorizedLength = source.Length
                - source.Length % Vector256<float>.Count;
            ref Half sourceReference = ref MemoryMarshal.GetReference(source);
            ref float destinationReference =
                ref MemoryMarshal.GetReference(destination);
            for (; index < vectorizedLength;
                index += Vector256<float>.Count)
            {
                LoadFloat16Vector256(
                    ref Unsafe.Add(ref sourceReference, index))
                    .StoreUnsafe(
                        ref Unsafe.Add(ref destinationReference, index));
            }
        }

        for (; index < source.Length; index++)
            destination[index] = (float)source[index];
    }

    internal static Vector256<float> LoadFloat16Vector256(
        Half[] source,
        int offset)
    {
        if (!Avx2.IsSupported)
        {
            return Vector256.Create(
                (float)source[offset],
                (float)source[offset + 1],
                (float)source[offset + 2],
                (float)source[offset + 3],
                (float)source[offset + 4],
                (float)source[offset + 5],
                (float)source[offset + 6],
                (float)source[offset + 7]);
        }

        return LoadFloat16Vector256(ref source[offset]);
    }

    internal static Vector256<float> LoadFloat16Vector256Avx2(
        Half[] source,
        int offset)
        => LoadFloat16Vector256(ref source[offset]);

    internal static Vector128<float> LoadFloat16Vector128(
        Half[] source,
        int offset)
        => Vector128.Create(
            (float)source[offset],
            (float)source[offset + 1],
            (float)source[offset + 2],
            (float)source[offset + 3]);

    private static bool TryStoreFloat16Vector256(
        Vector256<float> values,
        ref Half destination)
    {
        Vector256<int> bits = values.AsInt32();
        Vector256<int> absoluteBits =
            bits & Vector256.Create(0x7FFFFFFF);

        // This fast formula is exact for finite values in the binary16 normal
        // range. Zero, subnormal, overflow, infinity, and NaN lanes use the
        // runtime scalar conversion so their IEEE edge semantics stay exact.
        Vector256<int> atLeastMinimumNormal = Avx2.CompareGreaterThan(
            absoluteBits,
            Vector256.Create(0x387FFFFF));
        Vector256<int> belowOverflowBoundary = Avx2.CompareGreaterThan(
            Vector256.Create(0x47800000),
            absoluteBits);
        Vector256<int> eligible =
            atLeastMinimumNormal & belowOverflowBoundary;
        if (eligible.ExtractMostSignificantBits() != 0xFF)
            return false;

        Vector256<int> sign = Avx2.ShiftRightLogical(
            (bits & Vector256.Create(unchecked((int)0x80000000))).AsUInt32(),
            16).AsInt32();
        Vector256<int> tieToEven =
            Avx2.ShiftRightLogical(absoluteBits.AsUInt32(), 13).AsInt32()
            & Vector256.Create(1);
        Vector256<int> rounded = absoluteBits
            + Vector256.Create(0x00000FFF)
            + tieToEven;
        Vector256<int> halfBits = Avx2.ShiftRightLogical(
            (rounded - Vector256.Create(0x38000000)).AsUInt32(),
            13).AsInt32() | sign;
        Vector128<ushort> packed = Sse41.PackUnsignedSaturate(
            halfBits.GetLower(),
            halfBits.GetUpper());
        ref ushort destinationBits =
            ref Unsafe.As<Half, ushort>(ref destination);
        packed.StoreUnsafe(ref destinationBits);
        return true;
    }

    private static Vector256<float> LoadFloat16Vector256(ref Half source)
    {
        // .NET 10 does not expose F16C or native Vector256<Half>
        // arithmetic. Convert the packed IEEE-754 binary16 bits with AVX2,
        // then reuse the mature Float32 SIMD kernels. Subnormal lanes use the
        // scalar runtime conversion to preserve exact IEEE behavior.
        ref ushort bitsReference = ref Unsafe.As<Half, ushort>(ref source);
        Vector128<ushort> packed = Vector128.LoadUnsafe(ref bitsReference);
        Vector256<int> bits = Avx2.ConvertToVector256Int32(packed);
        Vector256<int> exponent = bits & Vector256.Create(0x7C00);
        Vector256<int> mantissa = bits & Vector256.Create(0x03FF);
        Vector256<int> sign = Avx2.ShiftLeftLogical(
            (bits & Vector256.Create(0x8000)).AsUInt32(),
            16).AsInt32();

        Vector256<int> zeroExponent = Vector256.Equals(
            exponent,
            Vector256<int>.Zero);
        Vector256<int> nonZeroMantissa = ~Vector256.Equals(
            mantissa,
            Vector256<int>.Zero);
        if ((zeroExponent & nonZeroMantissa).ExtractMostSignificantBits()
            != 0)
        {
            return Vector256.Create(
                (float)source,
                (float)Unsafe.Add(ref source, 1),
                (float)Unsafe.Add(ref source, 2),
                (float)Unsafe.Add(ref source, 3),
                (float)Unsafe.Add(ref source, 4),
                (float)Unsafe.Add(ref source, 5),
                (float)Unsafe.Add(ref source, 6),
                (float)Unsafe.Add(ref source, 7));
        }

        Vector256<int> normal = sign
            | (Avx2.ShiftLeftLogical(exponent.AsUInt32(), 13).AsInt32()
                + Vector256.Create((127 - 15) << 23))
            | Avx2.ShiftLeftLogical(mantissa.AsUInt32(), 13).AsInt32();
        Vector256<int> quietNaN = Vector256.ConditionalSelect(
            nonZeroMantissa,
            Vector256.Create(0x00400000),
            Vector256<int>.Zero);
        Vector256<int> infinityOrNaN = sign
            | Vector256.Create(unchecked((int)0x7F800000))
            | Avx2.ShiftLeftLogical(mantissa.AsUInt32(), 13).AsInt32()
            | quietNaN;
        Vector256<int> zero = sign;
        Vector256<int> specialExponent = Vector256.Equals(
            exponent,
            Vector256.Create(0x7C00));
        Vector256<int> result = Vector256.ConditionalSelect(
            zeroExponent,
            zero,
            Vector256.ConditionalSelect(
                specialExponent,
                infinityOrNaN,
                normal));
        return result.AsSingle();
    }
}
