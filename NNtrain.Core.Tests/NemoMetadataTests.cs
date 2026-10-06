using System.Text;
using System.Text.Json;
using NNtrain.Audio;
using Xunit;
namespace NNtrain.Core.Tests;
public sealed class NemoMetadataTests
{
    [Fact]
    public void NemoConversionProducesLoadableFp16WithoutExtractingArchivePaths()
    {
        string folder = Path.Combine(Path.GetTempPath(), "nntrain-nemo-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        string nemo = Path.Combine(folder, "fixture.nemo"), output = Path.Combine(folder, "model.safetensors");
        try
        {
            using var checkpoint = new MemoryStream();
            using (var zip = new System.IO.Compression.ZipArchive(checkpoint, System.IO.Compression.ZipArchiveMode.Create, leaveOpen: true))
            {
                using (var pkl = zip.CreateEntry("archive/data.pkl", System.IO.Compression.CompressionLevel.NoCompression).Open()) pkl.Write(TensorPickle());
                using (var data = new BinaryWriter(zip.CreateEntry("archive/data/0", System.IO.Compression.CompressionLevel.NoCompression).Open()))
                    foreach (float value in new[] { 1f, -2f, .5f, 3f }) data.Write(value);
            }
            checkpoint.Position = 0;
            using (var file = File.Create(nemo))
            using (var tar = new System.Formats.Tar.TarWriter(file))
                tar.WriteEntry(new System.Formats.Tar.UstarTarEntry(System.Formats.Tar.TarEntryType.RegularFile, "./model_weights.ckpt") { DataStream = checkpoint });
            Assert.Equal(8, NemoCheckpointConverter.ConvertToFp16(nemo, output, TestContext.Current.CancellationToken));
            var loaded = AsrHalfCheckpoint.Load(output, 8, TestContext.Current.CancellationToken);
            Assert.Equal(new Half[] { (Half)1, (Half)(-2), (Half).5, (Half)3 }, loaded.Weights["weight"].Values);
            Assert.Throws<IOException>(() => NemoCheckpointConverter.ConvertToFp16(nemo, output, TestContext.Current.CancellationToken));
        }
        finally { File.Delete(nemo); File.Delete(output); Directory.Delete(folder); }
    }
    [Fact]
    public void TensorDescriptorsAreReadWithoutExecutingGlobals()
    {
        using var stream = new MemoryStream(TensorPickle());
        var metadata = NemoTensorMetadata.ReadPickle(stream, TestContext.Current.CancellationToken);
        Assert.Single(metadata); Assert.Equal(new long[] { 2, 2 }, metadata["weight"].Shape);
        Assert.Equal("FloatStorage", metadata["weight"].DType); Assert.Equal(4, metadata["weight"].StorageCount);
    }
    private static byte[] TensorPickle()
    {
        using var stream = new MemoryStream(); using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        void Op(byte op) => writer.Write(op);
        void Text(string text) { Op((byte)'X'); byte[] bytes = Encoding.UTF8.GetBytes(text); writer.Write(bytes.Length); writer.Write(bytes); }
        void Global(string module, string name) { Op((byte)'c'); writer.Write(Encoding.UTF8.GetBytes(module + "\n" + name + "\n")); }
        void Number(int number) { Op((byte)'J'); writer.Write(number); }
        Op(0x80); Op(2); Op((byte)'}'); Op((byte)'('); Text("weight"); Global("torch._utils", "_rebuild_tensor_v2");
        Op((byte)'('); Op((byte)'('); Text("storage"); Global("torch", "FloatStorage"); Text("0"); Text("cpu"); Number(4); Op((byte)'t'); Op((byte)'Q');
        Number(0); Op((byte)'('); Number(2); Number(2); Op((byte)'t'); Op((byte)'('); Number(2); Number(1); Op((byte)'t');
        Op(0x89); Op((byte)'}'); Op((byte)'t'); Op((byte)'R'); Op((byte)'u'); Op((byte)'.');
        return stream.ToArray();
    }
    [Fact]
    public void ExecutableGlobalsAreRejectedBeforeReduce()
    {
        using var stream = new MemoryStream(new byte[] { 128, 2 }.Concat(Encoding.ASCII.GetBytes("cos\nsystem\n")).ToArray());
        Assert.Throws<InvalidDataException>(() => NemoTensorMetadata.ReadPickle(stream, TestContext.Current.CancellationToken));
    }
    [Fact]
    public void OfficialNemoMetadataCanBeInspectedWithoutWeightsOrPython()
    {
        string? path = Environment.GetEnvironmentVariable("NNTRAIN_ASR_NEMO_METADATA");
        if (string.IsNullOrEmpty(path)) return;
        var metadata = NemoTensorMetadata.ReadNemo(path, TestContext.Current.CancellationToken);
        string? report = Environment.GetEnvironmentVariable("NNTRAIN_ASR_NEMO_METADATA_REPORT");
        if (report is not null) File.WriteAllText(report, JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }));
        Assert.True(metadata.Count > 100);
        Assert.Contains(metadata.Keys, x => x.StartsWith("encoder.", StringComparison.Ordinal));
    }
}
