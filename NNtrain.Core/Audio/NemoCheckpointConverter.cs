using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using System.Text.Json;

namespace NNtrain.Audio;

/// <summary>Converts the official uncompressed NeMo/PyTorch archive to FP16 safetensors without executing pickle.</summary>
public static class NemoCheckpointConverter
{
    public static void ExtractModelAssets(string nemoPath, string directory, CancellationToken ct = default)
    {
        using var file = File.Open(nemoPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        using var tar = new System.Formats.Tar.TarReader(file);
        while (tar.GetNextEntry() is { } entry)
        {
            ct.ThrowIfCancellationRequested();
            string name = entry.Name.StartsWith("./", StringComparison.Ordinal) ? entry.Name[2..] : entry.Name;
            if (name == "model_weights.ckpt") break;
            string? target = name == "model_config.yaml" ? name
                : System.Text.RegularExpressions.Regex.IsMatch(name, "^[a-f0-9]{32}_tokenizer\\.vocab$") ? "tokenizer.vocab"
                : System.Text.RegularExpressions.Regex.IsMatch(name, "^[a-f0-9]{32}_tokenizer\\.model$") ? "tokenizer.model"
                : System.Text.RegularExpressions.Regex.IsMatch(name, "^[a-f0-9]{32}_vocab\\.txt$") ? "vocab.txt" : null;
            if (target is null) continue;
            if (entry.EntryType is not (System.Formats.Tar.TarEntryType.RegularFile or System.Formats.Tar.TarEntryType.V7RegularFile)
                || entry.Length is <= 0 or > 2 * 1024 * 1024 || entry.DataStream is null)
                throw new InvalidDataException("Invalid NeMo tokenizer/config member.");
            string destination = Path.Combine(directory, target);
            if (File.Exists(destination)) continue;
            byte[] data = new byte[checked((int)entry.Length)]; entry.DataStream.ReadExactly(data);
            using var output = new FileStream(destination, FileMode.CreateNew, FileAccess.Write, FileShare.None);
            output.Write(data);
        }
    }
    public static long ConvertToFp16(string nemoPath, string outputPath, CancellationToken ct = default,
        Action<string>? progress = null)
    {
        if (File.Exists(outputPath)) throw new IOException("The output checkpoint already exists.");
        using var file = File.OpenRead(nemoPath);
        var (start, length) = FindWeights(file);
        if (length <= 0 || start + length > file.Length) throw new InvalidDataException("NeMo download is incomplete.");
        using var slice = new Slice(file, start, length);
        using var zip = new ZipArchive(slice, ZipArchiveMode.Read, leaveOpen: true);
        if (zip.Entries.Count > 10000 || zip.Entries.Select(x => x.FullName).Distinct(StringComparer.Ordinal).Count() != zip.Entries.Count)
            throw new InvalidDataException("Invalid or duplicate ZIP members.");
        ZipArchiveEntry[] pickles = zip.Entries.Where(x => x.FullName.EndsWith("/data.pkl", StringComparison.Ordinal)).ToArray();
        if (pickles.Length != 1 || pickles[0].Length is <= 0 or > 8 * 1024 * 1024) throw new InvalidDataException("Invalid PyTorch metadata member.");
        string prefix = pickles[0].FullName[..^8];
        if (zip.GetEntry(prefix + "byteorder") is { } order)
        {
            if (order.Length > 16) throw new InvalidDataException("Invalid checkpoint byte order.");
            using var reader = new StreamReader(order.Open());
            if (reader.ReadToEnd().Trim() != "little") throw new InvalidDataException("Only little-endian PyTorch checkpoints are supported.");
        }
        IReadOnlyDictionary<string, NemoTensorMetadata.Tensor> metadata;
        using (Stream pkl = pickles[0].Open()) metadata = NemoTensorMetadata.ReadPickle(pkl, ct);
        var tensors = new List<(string Name, NemoTensorMetadata.Tensor Tensor, long Count, ZipArchiveEntry Entry, int Width)>();
        long total = 0;
        var header = new Dictionary<string, object>(StringComparer.Ordinal)
        { ["__metadata__"] = new Dictionary<string, string> { ["format"] = "pt", ["source"] = "nvidia/parakeet-tdt_ctc-0.6b-ja", ["license"] = "CC-BY-4.0", ["conversion"] = "NNtrain data-only reader; F32/F16/BF16 to F16; integer batch counters omitted" } };
        foreach (var (name, tensor) in metadata)
        {
            ct.ThrowIfCancellationRequested();
            if (tensor.DType == "LongStorage" && name.EndsWith(".num_batches_tracked", StringComparison.Ordinal)) continue;
            int width = tensor.DType switch { "FloatStorage" => 4, "HalfStorage" or "BFloat16Storage" => 2, _ => throw new InvalidDataException($"Unsupported weight dtype: {tensor.DType}") };
            long count = 1, stride = 1;
            for (int i = tensor.Shape.Length - 1; i >= 0; i--)
            {
                if (tensor.Shape[i] <= 0 || tensor.Shape[i] > int.MaxValue || tensor.Stride[i] != stride)
                    throw new InvalidDataException("Only nonempty contiguous tensors are supported.");
                stride = checked(stride * tensor.Shape[i]); count = stride;
            }
            if (count > int.MaxValue || tensor.Offset > tensor.StorageCount - count)
                throw new InvalidDataException("Tensor exceeds its storage.");
            if (tensor.Storage.Contains('/') || tensor.Storage.Contains('\\') || tensor.Storage.Contains("..", StringComparison.Ordinal))
                throw new InvalidDataException("Invalid storage identifier.");
            var entry = zip.GetEntry(prefix + "data/" + tensor.Storage) ?? throw new InvalidDataException("Missing tensor storage.");
            if (entry.Length != checked(tensor.StorageCount * width)) throw new InvalidDataException("Tensor storage length mismatch.");
            long end = checked(total + count * 2);
            if (end > 2L * 1024 * 1024 * 1024) throw new InvalidDataException("Converted ASR checkpoint exceeds 2 GiB weight limit.");
            header.Add(name, new { dtype = "F16", shape = tensor.Shape, data_offsets = new[] { total, end } });
            total = end; tensors.Add((name, tensor, count, entry, width));
        }
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
        int padded = checked((json.Length + 7) / 8 * 8);
        if (padded > 16 * 1024 * 1024) throw new InvalidDataException("Safetensors header limit.");
        string temporary = Path.GetFullPath(outputPath) + "." + Guid.NewGuid().ToString("N") + ".part";
        try
        {
            using (var output = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                byte[] size = new byte[8]; BinaryPrimitives.WriteUInt64LittleEndian(size, (ulong)padded);
                output.Write(size); output.Write(json); for (int i = json.Length; i < padded; i++) output.WriteByte(32);
                byte[] staging = new byte[64 * 1024], half = new byte[64 * 1024];
                foreach (var item in tensors)
                {
                    ct.ThrowIfCancellationRequested(); progress?.Invoke(item.Name);
                    using Stream source = item.Entry.Open();
                    long skip = checked(item.Tensor.Offset * item.Width);
                    while (skip > 0) { ct.ThrowIfCancellationRequested(); int n = (int)Math.Min(skip, staging.Length); source.ReadExactly(staging.AsSpan(0, n)); skip -= n; }
                    long left = item.Count;
                    while (left > 0)
                    {
                        ct.ThrowIfCancellationRequested(); int n = (int)Math.Min(left, staging.Length / item.Width);
                        source.ReadExactly(staging.AsSpan(0, n * item.Width));
                        for (int i = 0; i < n; i++)
                        {
                            ushort bits;
                            if (item.Tensor.DType == "HalfStorage") bits = BinaryPrimitives.ReadUInt16LittleEndian(staging.AsSpan(i * 2, 2));
                            else
                            {
                                float value = item.Width == 4 ? BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(staging.AsSpan(i * 4, 4)))
                                    : BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(staging.AsSpan(i * 2, 2)) << 16);
                                if (!float.IsFinite(value) || Math.Abs(value) > 65504) throw new InvalidDataException("Weight cannot be represented as finite FP16.");
                                bits = BitConverter.HalfToUInt16Bits((Half)value);
                            }
                            if ((bits & 0x7c00) == 0x7c00) throw new InvalidDataException("Nonfinite FP16 weight.");
                            BinaryPrimitives.WriteUInt16LittleEndian(half.AsSpan(i * 2, 2), bits);
                        }
                        output.Write(half, 0, n * 2); left -= n;
                    }
                }
                output.Flush(flushToDisk: true);
            }
            File.Move(temporary, outputPath, overwrite: false); return total;
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }

    private static (long Start, long Length) FindWeights(Stream file)
    {
        var header = new byte[512];
        while (file.Position + 512 <= file.Length)
        {
            file.ReadExactly(header);
            if (header.All(x => x == 0)) break;
            string name = Encoding.UTF8.GetString(header, 0, 100).TrimEnd('\0');
            string sizeText = Encoding.ASCII.GetString(header, 124, 12).Trim('\0', ' ');
            if (sizeText.Length == 0 || sizeText.Any(x => x is < '0' or > '7')) throw new InvalidDataException("Invalid tar size.");
            long size = System.Convert.ToInt64(sizeText, 8);
            string checksumText = Encoding.ASCII.GetString(header, 148, 8).Trim('\0', ' ');
            long checksum = System.Convert.ToInt64(checksumText, 8);
            long actual = 0; for (int i = 0; i < 512; i++) actual += i is >= 148 and < 156 ? 32 : header[i];
            if (actual != checksum || size < 0 || size > 4L * 1024 * 1024 * 1024) throw new InvalidDataException("Invalid tar header.");
            if (name is "model_weights.ckpt" or "./model_weights.ckpt")
            {
                if (header[156] is not (0 or (byte)'0')) throw new InvalidDataException("Weight member must be a regular file.");
                return (file.Position, size);
            }
            long next = checked(file.Position + ((size + 511) / 512 * 512));
            if (next > file.Length) throw new InvalidDataException("Incomplete NeMo archive.");
            file.Position = next;
        }
        throw new InvalidDataException("Missing model_weights.ckpt.");
    }

    private sealed class Slice(Stream source, long begin, long length) : Stream
    {
        private long _position;
        public override bool CanRead => true; public override bool CanSeek => true; public override bool CanWrite => false;
        public override long Length => length;
        public override long Position { get => _position; set => Seek(value, SeekOrigin.Begin); }
        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));
        public override int Read(Span<byte> buffer)
        {
            source.Position = begin + _position;
            int count = source.Read(buffer[..(int)Math.Min(buffer.Length, length - _position)]); _position += count; return count;
        }
        public override long Seek(long offset, SeekOrigin origin)
        {
            long position = checked(offset + (origin == SeekOrigin.Begin ? 0 : origin == SeekOrigin.Current ? _position : length));
            if (position < 0 || position > length) throw new IOException("Checkpoint slice seek out of bounds.");
            return _position = position;
        }
        public override void Flush() { }
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
