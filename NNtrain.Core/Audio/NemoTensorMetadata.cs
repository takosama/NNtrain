using System.Buffers.Binary;
using System.Formats.Tar;
using System.Text;

namespace NNtrain.Audio;

/// <summary>Data-only reader for PyTorch tensor descriptors. Never imports or executes pickle globals.</summary>
public static class NemoTensorMetadata
{
    public sealed record Tensor(string Storage, string DType, long StorageCount, long Offset, long[] Shape, long[] Stride);
    private sealed record Global(string Module, string Name);
    private sealed record Storage(string Key, string DType, long Count);
    private static readonly object Mark = new();

    public static IReadOnlyDictionary<string, Tensor> ReadNemo(string path, CancellationToken ct = default)
    {
        using var file = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        // This model is an uncompressed tar; incomplete downloads may still expose complete data.pkl metadata.
        using var tar = new TarReader(file);
        while (tar.GetNextEntry() is { } entry)
        {
            ct.ThrowIfCancellationRequested();
            if (entry.Name is not ("model_weights.ckpt" or "./model_weights.ckpt")) continue;
            if (entry.Length is <= 0 or > 4L * 1024 * 1024 * 1024 || entry.DataStream is null)
                throw new InvalidDataException("Invalid NeMo weight member.");
            using var reader = new BinaryReader(entry.DataStream, Encoding.UTF8, leaveOpen: true);
            if (reader.ReadUInt32() != 0x04034b50) throw new InvalidDataException("Expected PyTorch ZIP checkpoint.");
            reader.ReadUInt16(); ushort flags = reader.ReadUInt16(), method = reader.ReadUInt16();
            reader.ReadUInt32(); reader.ReadUInt32(); reader.ReadUInt32(); reader.ReadUInt32();
            ushort nameBytes = reader.ReadUInt16(), extraBytes = reader.ReadUInt16();
            string name = Encoding.UTF8.GetString(reader.ReadBytes(nameBytes));
            if (method != 0 || (flags & 1) != 0 || !name.EndsWith("/data.pkl", StringComparison.Ordinal))
                throw new InvalidDataException("Expected uncompressed data.pkl as the first ZIP member.");
            if (reader.ReadBytes(extraBytes).Length != extraBytes) throw new EndOfStreamException();
            return ReadPickle(entry.DataStream, ct);
        }
        throw new InvalidDataException("Missing model_weights.ckpt.");
    }

    public static IReadOnlyDictionary<string, Tensor> ReadPickle(Stream stream, CancellationToken ct = default)
    {
        using var input = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        var stack = new List<object?>(); var memo = new Dictionary<int, object?>(); long bytes = 0;
        byte Byte() { if (++bytes > 8 * 1024 * 1024) throw new InvalidDataException("Pickle metadata limit."); return input.ReadByte(); }
        byte[] Bytes(int n)
        {
            if (n < 0 || n > 1024 * 1024 || (bytes += n) > 8 * 1024 * 1024) throw new InvalidDataException("Pickle field limit.");
            byte[] data = input.ReadBytes(n); if (data.Length != n) throw new EndOfStreamException(); return data;
        }
        int Int() => BinaryPrimitives.ReadInt32LittleEndian(Bytes(4));
        string Line()
        {
            var data = new List<byte>(); byte value;
            while ((value = Byte()) != 10) { if (data.Count >= 4096) throw new InvalidDataException("Pickle line limit."); data.Add(value); }
            return Encoding.UTF8.GetString(data.ToArray());
        }
        object? Pop() { if (stack.Count == 0) throw new InvalidDataException("Invalid pickle stack."); var value = stack[^1]; stack.RemoveAt(stack.Count - 1); return value; }
        object?[] Tuple()
        {
            int start = stack.LastIndexOf(Mark); if (start < 0) throw new InvalidDataException("Missing pickle mark.");
            var result = stack.Skip(start + 1).ToArray(); stack.RemoveRange(start, stack.Count - start); return result;
        }
        long Number(object? value) => value is long l ? l : throw new InvalidDataException("Expected pickle integer.");
        void AddPairs(Dictionary<string, object?> dict, object?[] pairs)
        {
            if (pairs.Length % 2 != 0) throw new InvalidDataException("Invalid dictionary pairs.");
            for (int i = 0; i < pairs.Length; i += 2)
            { if (pairs[i] is not string key || !dict.TryAdd(key, pairs[i + 1])) throw new InvalidDataException("Invalid or duplicate tensor key."); }
        }
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            if (stack.Count > 10000 || memo.Count > 100000) throw new InvalidDataException("Pickle object limit.");
            byte opcode = Byte();
            switch (opcode)
            {
                case 0x80: if (Byte() is not (2 or 3)) throw new InvalidDataException("Only PyTorch pickle protocols 2/3 are supported."); break;
                case (byte)'(': stack.Add(Mark); break;
                case (byte)'}': stack.Add(new Dictionary<string, object?>()); break;
                case (byte)')': stack.Add(Array.Empty<object?>()); break;
                case (byte)'N': stack.Add(null); break;
                case 0x88: stack.Add(true); break;
                case 0x89: stack.Add(false); break;
                case (byte)'X': stack.Add(Encoding.UTF8.GetString(Bytes(Int()))); break;
                case (byte)'K': stack.Add((long)Byte()); break;
                case (byte)'M': stack.Add((long)BinaryPrimitives.ReadUInt16LittleEndian(Bytes(2))); break;
                case (byte)'J': stack.Add((long)Int()); break;
                case 0x8a:
                    byte[] number = Bytes(Byte());
                    if (number.Length is < 1 or > 8) throw new InvalidDataException("Integer overflow.");
                    long integer = (number[^1] & 128) == 0 ? 0 : -1;
                    for (int i = number.Length - 1; i >= 0; i--) integer = (integer << 8) | number[i];
                    stack.Add(integer); break;
                case (byte)'q': memo.Add(Byte(), stack[^1]); break;
                case (byte)'r': memo.Add(Int(), stack[^1]); break;
                case (byte)'h': stack.Add(memo[Byte()]); break;
                case (byte)'j': stack.Add(memo[Int()]); break;
                case (byte)'t': stack.Add(Tuple()); break;
                case 0x85: stack.Add(new[] { Pop() }); break;
                case 0x86: { var b = Pop(); var a = Pop(); stack.Add(new[] { a, b }); break; }
                case 0x87: { var c = Pop(); var b = Pop(); var a = Pop(); stack.Add(new[] { a, b, c }); break; }
                case (byte)'c':
                    var global = new Global(Line(), Line());
                    if (!((global.Module == "collections" && global.Name == "OrderedDict")
                        || (global.Module == "torch._utils" && global.Name == "_rebuild_tensor_v2")
                        || (global.Module == "torch" && global.Name is "FloatStorage" or "HalfStorage" or "BFloat16Storage" or "LongStorage")))
                        throw new InvalidDataException($"Unsupported pickle global {global.Module}.{global.Name}; no code was executed.");
                    stack.Add(global); break;
                case (byte)'Q':
                    if (Pop() is not object?[] persistent || persistent.Length != 5 || persistent[0] as string != "storage"
                        || persistent[1] is not Global type || type.Module != "torch" || persistent[2] is not string storageKey
                        || persistent[3] is not string location || !(location == "cpu" || location.StartsWith("cuda:", StringComparison.Ordinal)))
                        throw new InvalidDataException("Unsupported tensor storage descriptor.");
                    long count = Number(persistent[4]);
                    if (count < 0 || count > 2_000_000_000) throw new InvalidDataException("Storage size limit.");
                    stack.Add(new Storage(storageKey, type.Name, count)); break;
                case (byte)'R':
                    if (Pop() is not object?[] args || Pop() is not Global function) throw new InvalidDataException("Unsupported pickle reduce.");
                    if (function.Module == "collections" && function.Name == "OrderedDict" && args.Length == 0)
                        stack.Add(new Dictionary<string, object?>());
                    else if (function.Module == "torch._utils" && function.Name == "_rebuild_tensor_v2" && args.Length == 6
                        && args[0] is Storage storage && args[2] is object?[] shape && args[3] is object?[] stride
                        && args[4] is bool && args[5] is Dictionary<string, object?> hooks && hooks.Count == 0)
                    {
                        long offset = Number(args[1]); long[] dimensions = shape.Select(Number).ToArray(), strides = stride.Select(Number).ToArray();
                        if (offset < 0 || dimensions.Length > 8 || dimensions.Length != strides.Length || dimensions.Any(x => x < 0) || strides.Any(x => x < 0))
                            throw new InvalidDataException("Invalid tensor dimensions.");
                        stack.Add(new Tensor(storage.Key, storage.DType, storage.Count, offset, dimensions, strides));
                    }
                    else throw new InvalidDataException("Unsupported data-only tensor reconstruction.");
                    break;
                case (byte)'u': { object?[] pairs = Tuple(); if (stack[^1] is not Dictionary<string, object?> dict) throw new InvalidDataException("Invalid dictionary."); AddPairs(dict, pairs); break; }
                case (byte)'s': { object? value = Pop(), key = Pop(); if (stack[^1] is not Dictionary<string, object?> dict) throw new InvalidDataException("Invalid dictionary."); AddPairs(dict, [key, value]); break; }
                case (byte)'b':
                    // state_dict's optional OrderedDict attribute dictionary contains only version metadata.
                    if (Pop() is not Dictionary<string, object?> state || stack[^1] is not Dictionary<string, object?> || state.Keys.Any(x => x != "_metadata"))
                        throw new InvalidDataException("Unsupported pickle object state.");
                    break;
                case (byte)'.':
                    if (stack.Count != 1 || stack[0] is not Dictionary<string, object?> root) throw new InvalidDataException("Expected tensor state dictionary.");
                    if (root.TryGetValue("state_dict", out object? nested) && nested is Dictionary<string, object?> nestedDict) root = nestedDict;
                    if (root.Count == 0 || root.Values.Any(x => x is not Tensor)) throw new InvalidDataException("Checkpoint is not a pure tensor state dictionary.");
                    return root.ToDictionary(x => x.Key, x => (Tensor)x.Value!, StringComparer.Ordinal);
                default: throw new InvalidDataException($"Unsupported pickle opcode 0x{opcode:X2}; no code was executed.");
            }
        }
    }
}
