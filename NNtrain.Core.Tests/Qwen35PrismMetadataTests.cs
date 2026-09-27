using System.Text;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35PrismMetadataTests
{
    private const string Prefix = "prism.hadamard.";

    [Fact]
    public void OrdinaryQuantizedModelWithoutPrismMetadataReturnsNull()
        => Assert.Null(Read(new() { ["general.architecture"] = "qwen35" }, Tensors(12)));

    [Theory]
    [InlineData(142u)]
    [InlineData(143u)]
    public void PrismStorageRequiresTransformMetadata(uint type)
        => Assert.Throws<InvalidDataException>(() => Read(new() { ["general.architecture"] = "qwen35" }, Tensors(type)));

    [Theory]
    [InlineData(142u, false)]
    [InlineData(143u, true)]
    public void ReadsExplicitSignsForEachWidthAndSeparatesBothTransformDirections(uint type, bool grouped)
    {
        var metadata = Metadata();
        metadata[Prefix + "gdn_v_grouped"] = grouped;
        metadata[Prefix + "tied_output"] = false;
        Qwen35PrismMetadata result = Assert.IsType<Qwen35PrismMetadata>(Read(metadata, Tensors(type)));
        Assert.Equal(1024, Qwen35PrismMetadata.BlockSize);
        Assert.Equal(grouped, result.GroupedValueHeads);
        Assert.True(result.ForwardWeights.SetEquals(new[]
            { "output.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight" }));
        Assert.True(result.InverseWeights.SetEquals(new[] { "token_embd.weight" }));
        Assert.Equal(new[] { 1024, 2048 }, result.Signs.Keys.Order());
        Assert.Equal(Enumerable.Range(0, 1024).Select(i => i % 2 == 0 ? -1f : 1f), result.Signs[1024]);
        Assert.Equal(Enumerable.Range(0, 2048).Select(i => i % 3 == 0 ? 1f : -1f), result.Signs[2048]);
    }

    [Theory]
    [InlineData("version")]
    [InlineData("block_size")]
    [InlineData("transform")]
    [InlineData("axis")]
    [InlineData("sign_mode")]
    [InlineData("weight_names")]
    [InlineData("sign_widths")]
    [InlineData("sign_values")]
    public void MissingRequiredMetadataFailsClosed(string suffix)
    {
        var metadata = Metadata();
        metadata.Remove(Prefix + suffix);
        Assert.Throws<InvalidDataException>(() => Read(metadata));
    }

    [Theory]
    [InlineData("version", 2u)]
    [InlineData("block_size", 512u)]
    [InlineData("transform", "unnormalized-sylvester-walsh-hadamard")]
    [InlineData("axis", "output-first-dimension")]
    [InlineData("sign_mode", "identity")]
    [InlineData("future_transform", 1u)]
    public void UnsupportedMetadataFailsClosed(string suffix, object value)
    {
        var metadata = Metadata();
        metadata[Prefix + suffix] = value;
        Assert.Throws<NotSupportedException>(() => Read(metadata));
    }

    [Fact]
    public void RejectsTiedOutputForVersionOne()
    {
        var metadata = Metadata();
        metadata[Prefix + "tied_output"] = true;
        Assert.Throws<NotSupportedException>(() => Read(metadata));
        Assert.Throws<NotSupportedException>(() => Read(Metadata(),
            Tensors().Where(t => t.Name != "output.weight").ToArray()));
    }

    [Theory]
    [InlineData("gdn_v_grouped", 1u)]
    [InlineData("tied_output", 0u)]
    [InlineData("version", "1")]
    [InlineData("sign_widths", "1024,2048")]
    public void WrongMetadataScalarOrArrayTypeIsRejected(string suffix, object value)
    {
        var metadata = Metadata();
        metadata[Prefix + suffix] = value;
        Assert.Throws<InvalidDataException>(() => Read(metadata));
    }

    [Fact]
    public void RejectsUnverifiedArchitectureAndTensorKinds()
    {
        var metadata = Metadata();
        metadata["general.architecture"] = "llama";
        Assert.Throws<NotSupportedException>(() => Read(metadata));
        metadata = Metadata();
        metadata[Prefix + "weight_names"] = new[] { "blk.0.ssm_alpha.weight" };
        Assert.Throws<NotSupportedException>(() => Read(metadata));
        metadata = Metadata();
        metadata[Prefix + "inverse_weight_names"] = new[] { "output.weight" };
        Assert.Throws<NotSupportedException>(() => Read(metadata));
    }

    [Theory]
    [InlineData("weight_names", "output.weight")]
    [InlineData("inverse_weight_names", "token_embd.weight")]
    public void DuplicateTransformNamesAreRejected(string suffix, string name)
    {
        var metadata = Metadata();
        metadata[Prefix + suffix] = new[] { name, name };
        Assert.Throws<InvalidDataException>(() => Read(metadata));
    }

    [Fact]
    public void EmptyForwardListAndMissingTensorAreRejected()
    {
        var metadata = Metadata();
        metadata[Prefix + "weight_names"] = Array.Empty<string>();
        Assert.Throws<InvalidDataException>(() => Read(metadata));
        metadata = Metadata();
        metadata[Prefix + "weight_names"] = new[] { "blk.99.attn_q.weight" };
        Assert.Throws<InvalidDataException>(() => Read(metadata));
    }

    [Theory]
    [InlineData("output.weight")]
    [InlineData("token_embd.weight")]
    [InlineData("blk.0.ffn_down.weight")]
    public void EveryPrismMatrixMustDeclareItsTransform(string omitted)
    {
        var metadata = Metadata();
        string key = Prefix + (omitted == "token_embd.weight" ? "inverse_weight_names" : "weight_names");
        metadata[key] = ((string[])metadata[key]).Where(name => name != omitted).ToArray();
        InvalidDataException error = Assert.Throws<InvalidDataException>(() => Read(metadata));
        Assert.Contains(omitted, error.Message);
    }

    [Fact]
    public void DuplicateTensorDirectoryAndMalformedMatrixShapeAreRejected()
    {
        var tensors = Tensors();
        Assert.Throws<InvalidDataException>(() => Read(Metadata(), [.. tensors, tensors[0]]));
        foreach (ulong[] shape in new ulong[][] { [1024], [0, 7], [1024, 0], [513, 7], [1024, 7, 1] })
            Assert.Throws<InvalidDataException>(() => Read(Metadata(), tensors
                .Select(t => t.Name == "output.weight" ? t with { Shape = shape } : t).ToArray()));
    }

    [Theory]
    [InlineData("empty")]
    [InlineData("duplicate")]
    [InlineData("zero")]
    [InlineData("negative")]
    [InlineData("unaligned")]
    [InlineData("short-values")]
    [InlineData("long-values")]
    [InlineData("missing-width")]
    [InlineData("bad-sign")]
    public void InvalidSignVectorsAreRejected(string variant)
    {
        var metadata = Metadata();
        switch (variant)
        {
            case "empty": metadata[Prefix + "sign_widths"] = Array.Empty<int>(); break;
            case "duplicate": metadata[Prefix + "sign_widths"] = new[] { 1024, 1024 }; break;
            case "zero": metadata[Prefix + "sign_widths"] = new[] { 0, 2048 }; break;
            case "negative": metadata[Prefix + "sign_widths"] = new[] { -1024, 2048 }; break;
            case "unaligned": metadata[Prefix + "sign_widths"] = new[] { 1025, 2048 }; break;
            case "short-values": metadata[Prefix + "sign_values"] = new int[3071]; break;
            case "long-values": metadata[Prefix + "sign_values"] = new int[3073]; break;
            case "missing-width":
                metadata[Prefix + "sign_widths"] = new[] { 1024 };
                metadata[Prefix + "sign_values"] = Enumerable.Repeat(1, 1024).ToArray();
                break;
            case "bad-sign": ((int[])metadata[Prefix + "sign_values"])[1025] = 0; break;
        }
        Assert.Throws<InvalidDataException>(() => Read(metadata));
    }

    [Fact]
    public void UnusedSignVectorIsOutsideTheSupportedInventory()
    {
        var metadata = Metadata();
        metadata[Prefix + "sign_widths"] = new[] { 1024, 2048, 3072 };
        metadata[Prefix + "sign_values"] = Enumerable.Repeat(1, 6144).ToArray();
        Assert.Throws<NotSupportedException>(() => Read(metadata));
    }

    private static Dictionary<string, object> Metadata() => new()
    {
        ["general.architecture"] = "qwen35",
        [Prefix + "version"] = 1u,
        [Prefix + "block_size"] = 1024u,
        [Prefix + "transform"] = "normalized-sylvester-walsh-hadamard",
        [Prefix + "axis"] = "input-last-dimension",
        [Prefix + "sign_mode"] = "explicit",
        [Prefix + "weight_names"] = new[] { "output.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight" },
        [Prefix + "inverse_weight_names"] = new[] { "token_embd.weight" },
        [Prefix + "sign_widths"] = new[] { 1024, 2048 },
        [Prefix + "sign_values"] = Enumerable.Range(0, 1024).Select(i => i % 2 == 0 ? -1 : 1)
            .Concat(Enumerable.Range(0, 2048).Select(i => i % 3 == 0 ? 1 : -1)).ToArray()
    };

    private static GgufTensorInfo[] Tensors(uint type = 142) =>
    [
        new("token_embd.weight", [1024, 7], type, 0),
        new("output.weight", [1024, 7], type, 0),
        new("blk.0.ffn_gate.weight", [1024, 2048], type, 0),
        new("blk.0.ffn_down.weight", [2048, 1024], type, 0),
        new("blk.0.ssm_alpha.weight", [1024, 2], 30, 0)
    ];

    private static Qwen35PrismMetadata? Read(Dictionary<string, object> metadata, GgufTensorInfo[]? tensors = null)
    {
        var booleans = metadata.Where(pair => pair.Value is bool).ToArray();
        using var file = new TemporaryQwenGguf(metadata.Where(pair => pair.Value is not bool)
            .ToDictionary(pair => pair.Key, pair => pair.Value), tensors ?? Tensors());
        if (booleans.Length != 0)
        {
            // The shared fixture supports strings and integer metadata. Insert
            // real GGUF BOOL entries so these cases also exercise reader typing.
            byte[] original = File.ReadAllBytes(file.Path);
            using var stream = new FileStream(file.Path, FileMode.Create, FileAccess.Write);
            using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: false);
            writer.Write(original, 0, 16);
            writer.Write((ulong)metadata.Count);
            foreach (var pair in booleans)
            {
                byte[] key = Encoding.UTF8.GetBytes(pair.Key);
                writer.Write((ulong)key.Length);
                writer.Write(key);
                writer.Write((uint)GgufValueType.Bool);
                writer.Write((bool)pair.Value);
            }
            writer.Write(original, 24, original.Length - 24);
            while (stream.Position % 32 != 0) writer.Write((byte)0);
        }
        using var reader = new GgufReader(file.Path);
        return Qwen35PrismMetadata.Read(reader);
    }
}
