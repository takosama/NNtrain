using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace NNtrain;

/// <summary>
/// Physical SafeTensors scalar codec used by the streaming checkpoint path.
/// Keeping descriptor names, encoded widths, and conversion here allows a
/// future quantized codec (including companion scale tensors) to be added
/// without changing the public Tensor or Module APIs.
/// </summary>
internal abstract class SafeTensorDTypeCodec
{
    protected SafeTensorDTypeCodec(
        TensorDType dtype,
        string descriptorName,
        int encodedElementSize)
    {
        DType = dtype;
        DescriptorName = descriptorName;
        EncodedElementSize = encodedElementSize;
    }

    internal TensorDType DType { get; }
    internal string DescriptorName { get; }
    internal int EncodedElementSize { get; }

    internal abstract void Write(Stream stream, ReadOnlySpan<float> values);
    internal abstract void Read(Stream stream, Span<float> values);

    protected static void ReadExactly(Stream stream, Span<byte> destination)
    {
        while (!destination.IsEmpty)
        {
            int read = stream.Read(destination);
            if (read == 0)
                throw new EndOfStreamException();
            destination = destination[read..];
        }
    }
}

internal static class SafeTensorDTypeCodecs
{
    private static readonly SafeTensorDTypeCodec[] Registered =
    [
        new Float32Codec(),
        new Float16Codec(),
        new BFloat16Codec(),
    ];

    internal static SafeTensorDTypeCodec Get(TensorDType dtype)
        => Registered.FirstOrDefault(codec => codec.DType == dtype)
            ?? throw new NotSupportedException(
                $"SafeTensors dtype '{dtype}' has no registered codec.");

    internal static SafeTensorDTypeCodec Parse(string? descriptorName)
        => Registered.FirstOrDefault(codec => string.Equals(
                codec.DescriptorName,
                descriptorName,
                StringComparison.Ordinal))
            ?? throw new NotSupportedException(
                $"SafeTensors dtype '{descriptorName}' has no registered codec.");

    private sealed class Float32Codec()
        : SafeTensorDTypeCodec(
            TensorDType.Float32,
            "F32",
            sizeof(float))
    {
        internal override void Write(
            Stream stream,
            ReadOnlySpan<float> values)
        {
            if (BitConverter.IsLittleEndian)
            {
                stream.Write(MemoryMarshal.AsBytes(values));
                return;
            }
            Span<byte> scalar = stackalloc byte[sizeof(float)];
            foreach (float value in values)
            {
                BinaryPrimitives.WriteSingleLittleEndian(scalar, value);
                stream.Write(scalar);
            }
        }

        internal override void Read(Stream stream, Span<float> values)
        {
            Span<byte> bytes = MemoryMarshal.AsBytes(values);
            ReadExactly(stream, bytes);
            if (BitConverter.IsLittleEndian)
                return;
            for (int index = 0; index < values.Length; index++)
            {
                values[index] = BinaryPrimitives.ReadSingleLittleEndian(
                    bytes.Slice(index * sizeof(float), sizeof(float)));
            }
        }
    }

    private abstract class Float16BaseCodec(
        TensorDType dtype,
        string descriptorName)
        : SafeTensorDTypeCodec(dtype, descriptorName, sizeof(ushort))
    {
        protected abstract ushort Encode(float value);
        protected abstract float Decode(ushort bits);

        internal override void Write(
            Stream stream,
            ReadOnlySpan<float> values)
        {
            const int ValuesPerChunk = 4096;
            Span<byte> encoded =
                stackalloc byte[ValuesPerChunk * sizeof(ushort)];
            int offset = 0;
            while (offset < values.Length)
            {
                int count = Math.Min(
                    ValuesPerChunk,
                    values.Length - offset);
                for (int index = 0; index < count; index++)
                {
                    BinaryPrimitives.WriteUInt16LittleEndian(
                        encoded.Slice(
                            index * sizeof(ushort),
                            sizeof(ushort)),
                        Encode(values[offset + index]));
                }
                stream.Write(encoded[..checked(count * sizeof(ushort))]);
                offset += count;
            }
        }

        internal override void Read(Stream stream, Span<float> values)
        {
            Span<byte> bytes = MemoryMarshal.AsBytes(values);
            Span<byte> encoded =
                bytes[..checked(values.Length * sizeof(ushort))];
            ReadExactly(stream, encoded);
            for (int index = values.Length - 1; index >= 0; index--)
            {
                ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(
                    encoded.Slice(
                        index * sizeof(ushort),
                        sizeof(ushort)));
                values[index] = Decode(bits);
            }
        }
    }

    private sealed class Float16Codec()
        : Float16BaseCodec(TensorDType.Float16, "F16")
    {
        protected override ushort Encode(float value)
            => BitConverter.HalfToUInt16Bits((Half)value);

        protected override float Decode(ushort bits)
            => (float)BitConverter.UInt16BitsToHalf(bits);
    }

    private sealed class BFloat16Codec()
        : Float16BaseCodec(TensorDType.BFloat16, "BF16")
    {
        protected override ushort Encode(float value)
            => TensorStorageCodec.EncodeBFloat16(value);

        protected override float Decode(ushort bits)
            => TensorStorageCodec.DecodeBFloat16(bits);
    }
}
