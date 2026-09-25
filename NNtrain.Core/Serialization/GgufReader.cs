using System.Buffers.Binary;
using System.Text;

namespace NNtrain;

/// <summary>
/// Minimal, dependency-free GGUF v2/v3 reader.
///
/// Phase one deliberately separates container parsing from model mapping and
/// quantization.  It can inspect real llama.cpp/Hugging Face GGUF files and
/// exposes tensor offsets without allocating the tensor payloads.
/// </summary>
public sealed class GgufReader : IDisposable
{
    private const uint Magic = 0x46554747; // "GGUF" little-endian
    private readonly FileStream _stream;
    private readonly BinaryReader _reader;
    private readonly Dictionary<string, object> _metadata = new(StringComparer.Ordinal);
    private readonly List<GgufTensorInfo> _tensors = [];
    private readonly long _dataOffset;

    public GgufReader(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        _stream = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        _reader = new BinaryReader(_stream, Encoding.UTF8, leaveOpen: true);

        try
        {
            if (_reader.ReadUInt32() != Magic)
                throw new InvalidDataException("Not a GGUF file.");
            Version = _reader.ReadUInt32();
            if (Version is < 2 or > 3)
                throw new NotSupportedException($"GGUF version {Version} is not supported.");

            ulong tensorCount = _reader.ReadUInt64();
            ulong metadataCount = _reader.ReadUInt64();
            if (tensorCount > int.MaxValue || metadataCount > int.MaxValue)
                throw new InvalidDataException("GGUF table is too large.");

            for (ulong i = 0; i < metadataCount; i++)
            {
                string key = ReadString();
                var type = (GgufValueType)_reader.ReadUInt32();
                _metadata.Add(key, ReadValue(type));
            }

            for (ulong i = 0; i < tensorCount; i++)
            {
                string name = ReadString();
                uint dimensions = _reader.ReadUInt32();
                if (dimensions is 0 or > 4)
                    throw new InvalidDataException($"Tensor '{name}' has unsupported rank {dimensions}.");
                var shape = new ulong[dimensions];
                for (int d = 0; d < shape.Length; d++) shape[d] = _reader.ReadUInt64();
                uint type = _reader.ReadUInt32();
                ulong offset = _reader.ReadUInt64();
                _tensors.Add(new GgufTensorInfo(name, shape, type, offset));
            }

            int alignment = 32;
            if (_metadata.TryGetValue("general.alignment", out object? value))
                alignment = checked((int)Convert.ToUInt64(value));
            if (alignment <= 0 || (alignment & (alignment - 1)) != 0)
                throw new InvalidDataException($"Invalid GGUF alignment {alignment}.");

            _dataOffset = Align(_stream.Position, alignment);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public uint Version { get; }
    public IReadOnlyDictionary<string, object> Metadata => _metadata;
    public IReadOnlyList<GgufTensorInfo> Tensors => _tensors;

    public Stream OpenTensorData(GgufTensorInfo tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        _stream.Position = checked(_dataOffset + (long)tensor.Offset);
        return new NonOwningStream(_stream);
    }

    public byte[] ReadTensorBytes(GgufTensorInfo tensor, int byteCount)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentOutOfRangeException.ThrowIfNegative(byteCount);
        _stream.Position = checked(_dataOffset + (long)tensor.Offset);
        byte[] payload = _reader.ReadBytes(byteCount);
        if (payload.Length != byteCount) throw new EndOfStreamException();
        return payload;
    }

    public GgufTensorInfo GetTensor(string name)
        => _tensors.FirstOrDefault(t => t.Name == name)
           ?? throw new KeyNotFoundException($"GGUF tensor '{name}' was not found.");

    private object ReadValue(GgufValueType type) => type switch
    {
        GgufValueType.UInt8 => _reader.ReadByte(),
        GgufValueType.Int8 => _reader.ReadSByte(),
        GgufValueType.UInt16 => _reader.ReadUInt16(),
        GgufValueType.Int16 => _reader.ReadInt16(),
        GgufValueType.UInt32 => _reader.ReadUInt32(),
        GgufValueType.Int32 => _reader.ReadInt32(),
        GgufValueType.Float32 => _reader.ReadSingle(),
        GgufValueType.Bool => ReadBool(),
        GgufValueType.String => ReadString(),
        GgufValueType.Array => ReadArray(),
        GgufValueType.UInt64 => _reader.ReadUInt64(),
        GgufValueType.Int64 => _reader.ReadInt64(),
        GgufValueType.Float64 => _reader.ReadDouble(),
        _ => throw new NotSupportedException($"GGUF metadata type {(uint)type} is unsupported.")
    };

    private bool ReadBool()
    {
        byte value = _reader.ReadByte();
        return value switch { 0 => false, 1 => true, _ => throw new InvalidDataException("Invalid GGUF bool.") };
    }

    private object[] ReadArray()
    {
        var elementType = (GgufValueType)_reader.ReadUInt32();
        ulong count = _reader.ReadUInt64();
        if (count > int.MaxValue) throw new InvalidDataException("GGUF metadata array is too large.");
        var values = new object[(int)count];
        for (int i = 0; i < values.Length; i++) values[i] = ReadValue(elementType);
        return values;
    }

    private string ReadString()
    {
        ulong length = _reader.ReadUInt64();
        if (length > int.MaxValue) throw new InvalidDataException("GGUF string is too large.");
        byte[] bytes = _reader.ReadBytes((int)length);
        if ((ulong)bytes.Length != length) throw new EndOfStreamException();
        return Encoding.UTF8.GetString(bytes);
    }

    private static long Align(long value, int alignment)
        => checked((value + alignment - 1) / alignment * alignment);

    public void Dispose()
    {
        _reader.Dispose();
        _stream.Dispose();
    }

    private sealed class NonOwningStream(Stream inner) : Stream
    {
        public override bool CanRead => inner.CanRead;
        public override bool CanSeek => inner.CanSeek;
        public override bool CanWrite => false;
        public override long Length => inner.Length;
        public override long Position { get => inner.Position; set => inner.Position = value; }
        public override void Flush() => inner.Flush();
        public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, count);
        public override int Read(Span<byte> buffer) => inner.Read(buffer);
        public override long Seek(long offset, SeekOrigin origin) => inner.Seek(offset, origin);
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        protected override void Dispose(bool disposing) { }
    }
}

public sealed record GgufTensorInfo(string Name, IReadOnlyList<ulong> Shape, uint Type, ulong Offset);

public enum GgufValueType : uint
{
    UInt8 = 0, Int8 = 1, UInt16 = 2, Int16 = 3, UInt32 = 4, Int32 = 5,
    Float32 = 6, Bool = 7, String = 8, Array = 9, UInt64 = 10,
    Int64 = 11, Float64 = 12
}
