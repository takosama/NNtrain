using System.Text;

namespace NNtrain.Core.Tests;

public sealed class GgufReaderTests
{
    [Fact]
    public void ReadsMinimalVersion3Container()
    {
        string path = Path.GetTempFileName();
        try
        {
            using (var stream = File.Create(path))
            using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: false))
            {
                writer.Write(0x46554747u);
                writer.Write(3u);
                writer.Write(0ul); // tensors
                writer.Write(2ul); // metadata
                WriteString(writer, "general.architecture");
                writer.Write((uint)GgufValueType.String);
                WriteString(writer, "qwen2");
                WriteString(writer, "general.alignment");
                writer.Write((uint)GgufValueType.UInt32);
                writer.Write(32u);
            }

            using var reader = new GgufReader(path);
            Assert.Equal(3u, reader.Version);
            Assert.Equal("qwen2", reader.Metadata["general.architecture"]);
            Assert.Empty(reader.Tensors);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void RejectsWrongMagic()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(path, [1, 2, 3, 4]);
            Assert.Throws<InvalidDataException>(() => new GgufReader(path));
        }
        finally { File.Delete(path); }
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }
}
