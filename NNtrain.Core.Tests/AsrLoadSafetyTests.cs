using System.Text;
using System.Text.Json;
using NNtrain.Audio;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class AsrLoadSafetyTests
{
    [Fact]
    public void DirectHalfReadsPreserveEveryFiniteBitPatternAcrossBlocks()
    {
        ushort[] bits = Enumerable.Range(0, 65536).Select(x => (ushort)x)
            .Where(x => (x & 0x7c00) != 0x7c00).ToArray();
        using var fixture = new CheckpointFile("F16", bits.Length, writer =>
        {
            foreach (ushort value in bits) writer.Write(value);
        });
        var loaded = AsrHalfCheckpoint.Load(fixture.Path, bits.Length * 2L, TestContext.Current.CancellationToken);
        Assert.Equal(bits, loaded.Weights["test"].Values.Select(BitConverter.HalfToUInt16Bits));
    }

    [Fact]
    public void FloatConversionMatchesScalarRoundingAcrossVectorsAndBlocks()
    {
        var values = new List<float> { 0f, -0f, float.Epsilon, -float.Epsilon, 65504f, -65504f };
        foreach (float value in new[] { 0.00006103515625f, 0.000000059604645f, 1f, 1.00048828125f, 2f, 65519f })
        {
            values.AddRange([MathF.BitDecrement(value), value, MathF.BitIncrement(value), -value]);
        }
        var random = new Random(60106);
        while (values.Count < 65541)
        {
            float value = BitConverter.Int32BitsToSingle((int)random.NextInt64(int.MinValue, (long)int.MaxValue + 1));
            if (Half.IsFinite((Half)value)) values.Add(value);
        }
        using var fixture = new CheckpointFile("F32", values.Count, writer =>
        {
            foreach (float value in values) writer.Write(value);
        });
        var loaded = AsrHalfCheckpoint.Load(fixture.Path, values.Count * 2L, TestContext.Current.CancellationToken);
        Assert.Equal(values.Select(x => BitConverter.HalfToUInt16Bits((Half)x)),
            loaded.Weights["test"].Values.Select(BitConverter.HalfToUInt16Bits));
    }

    [Theory]
    [InlineData("F16")]
    [InlineData("F32")]
    public void RejectsNonfiniteAndOverflowAtVectorAndReadBoundaries(string dtype)
    {
        int block = AsrHalfCheckpoint.StagingBytes / (dtype == "F16" ? 2 : 4);
        foreach (int index in new[] { 0, 15, 16, block - 1, block, block + 16 })
            foreach (float bad in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 65520f })
            {
                using var fixture = new CheckpointFile(dtype, block + 17, writer =>
                {
                    for (int i = 0; i < block + 17; i++)
                        if (dtype == "F16") writer.Write(BitConverter.HalfToUInt16Bits((Half)(i == index ? bad : 1f)));
                        else writer.Write(i == index ? bad : 1f);
                });
                Assert.Throws<InvalidDataException>(() => AsrHalfCheckpoint.Load(fixture.Path, (block + 17) * 2L, TestContext.Current.CancellationToken));
            }
    }

    [Theory]
    [InlineData("overlap")]
    [InlineData("gap")]
    [InlineData("trailing")]
    [InlineData("duplicate")]
    [InlineData("negative")]
    [InlineData("dimension")]
    [InlineData("dtype")]
    public void RejectsMalformedLayoutsBeforeReadingPayload(string kind)
    {
        const string tensor = "{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,8]}";
        string header = kind switch
        {
            "overlap" => "{\"a\":" + tensor + ",\"b\":" + tensor + "}",
            "gap" => "{\"a\":" + tensor + ",\"b\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[12,16]}}",
            "duplicate" => "{\"a\":" + tensor + ",\"a\":" + tensor + "}",
            "negative" => "{\"a\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[-1,7]}}",
            "dimension" => "{\"a\":{\"dtype\":\"F16\",\"shape\":[0],\"data_offsets\":[0,0]}}",
            "dtype" => "{\"a\":{\"dtype\":\"I16\",\"shape\":[8],\"data_offsets\":[0,16]}}",
            _ => "{\"a\":" + tensor + "}"
        };
        using var fixture = new CheckpointFile(header, writer => writer.Write(new byte[16]));
        bool payloadRead = false;
        Assert.Throws<InvalidDataException>(() => AsrHalfCheckpoint.Load(fixture.Path, 1024,
            TestContext.Current.CancellationToken, (name, _) => { if (name == "checkpoint.read") payloadRead = true; }));
        Assert.False(payloadRead);
    }

    [Fact]
    public void RejectsTruncationAndBudgetBeforeExposingWeights()
    {
        using var fixture = new CheckpointFile("F16", 4, writer => writer.Write(new byte[7]));
        Assert.Throws<InvalidDataException>(() => AsrHalfCheckpoint.Load(fixture.Path, 8, TestContext.Current.CancellationToken));
        fixture.Rewrite("F16", 4, writer => writer.Write(new byte[8]));
        Assert.Throws<InvalidDataException>(() => AsrHalfCheckpoint.Load(fixture.Path, 7, TestContext.Current.CancellationToken));
    }

    [Fact]
    public void CancellationAfterHeaderAndAfterPayloadDoesNotPublishCheckpoint()
    {
        using var fixture = new CheckpointFile("F16", 32769, writer => writer.Write(new byte[32769 * 2]));
        foreach (string phase in new[] { "checkpoint.headerValidation", "checkpoint.convertValidate" })
        {
            using var cancelled = new CancellationTokenSource();
            Assert.Throws<OperationCanceledException>(() => AsrHalfCheckpoint.Load(fixture.Path, 32769 * 2L,
                cancelled.Token, (name, _) => { if (name == phase) cancelled.Cancel(); }));
        }
        Assert.Equal(32769 * 2L, AsrHalfCheckpoint.Load(fixture.Path, 32769 * 2L, TestContext.Current.CancellationToken).WeightBytes);
    }

    [Fact]
    public void ReloadReadsChangedSourceAndRejectsNewCorruption()
    {
        using var fixture = new CheckpointFile("F16", 1, writer => writer.Write((ushort)0x3c00));
        Assert.Equal((Half)1, AsrHalfCheckpoint.Load(fixture.Path, 2, TestContext.Current.CancellationToken).Weights["test"].Values[0]);
        fixture.Rewrite("F16", 1, writer => writer.Write((ushort)0x4000));
        Assert.Equal((Half)2, AsrHalfCheckpoint.Load(fixture.Path, 2, TestContext.Current.CancellationToken).Weights["test"].Values[0]);
        fixture.Rewrite("F16", 1, writer => writer.Write((ushort)0x7c00));
        Assert.Throws<InvalidDataException>(() => AsrHalfCheckpoint.Load(fixture.Path, 2, TestContext.Current.CancellationToken));
    }

    [Fact]
    public void WindowsLoadDeniesConcurrentSourceWrites()
    {
        if (!OperatingSystem.IsWindows()) return;
        using var fixture = new CheckpointFile("F16", 1, writer => writer.Write((ushort)0x3c00));
        bool checkedSharing = false;
        var result = AsrHalfCheckpoint.Load(fixture.Path, 2, TestContext.Current.CancellationToken, (phase, _) =>
        {
            if (phase != "checkpoint.headerValidation") return;
            Assert.Throws<IOException>(() => File.Open(fixture.Path, FileMode.Open, FileAccess.Write, FileShare.ReadWrite));
            checkedSharing = true;
        });
        Assert.True(checkedSharing);
        Assert.Equal((Half)1, result.Weights["test"].Values[0]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ModelLoadCancellationAllowsAValidReload(bool parakeet)
    {
        using IDisposable fixture = parakeet ? new ParakeetCtcTests.Fixture() : new NemotronModelTests.MiniAsrFixture();
        string directory = parakeet ? ((ParakeetCtcTests.Fixture)fixture).Directory : ((NemotronModelTests.MiniAsrFixture)fixture).Directory;
        using var cancelled = new CancellationTokenSource();
        void Observe(string phase, double _) { if (phase == "checkpoint.headerValidation") cancelled.Cancel(); }
        Assert.Throws<OperationCanceledException>(() =>
        {
            using ILocalAsrModel model = parakeet ? ParakeetCtcModel.Load(directory, cancelled.Token, Observe)
                : NemotronAsrModel.Load(directory, cancelled.Token, Observe);
        });
        using ILocalAsrModel reloaded = parakeet ? ParakeetCtcModel.Load(directory, TestContext.Current.CancellationToken) : NemotronAsrModel.Load(directory, TestContext.Current.CancellationToken);
        Assert.True(reloaded.HostWeightBytes > 0);
        Assert.Throws<OperationCanceledException>(() => reloaded.EnableArc(0, cancelled.Token));
    }

    private sealed class CheckpointFile : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "nntrain-asr-load-" + Guid.NewGuid().ToString("N") + ".safetensors");
        public CheckpointFile(string dtype, int count, Action<BinaryWriter> payload) => Rewrite(dtype, count, payload);
        public CheckpointFile(string header, Action<BinaryWriter> payload) => Write(header, payload);
        public void Rewrite(string dtype, int count, Action<BinaryWriter> payload)
            => Write(JsonSerializer.Serialize(new Dictionary<string, object> { ["test"] = new { dtype, shape = new[] { count },
                data_offsets = new[] { 0, count * (dtype == "F16" ? 2 : 4) } } }), payload);
        private void Write(string header, Action<BinaryWriter> payload)
        {
            using var writer = new BinaryWriter(File.Create(Path));
            byte[] bytes = Encoding.UTF8.GetBytes(header); writer.Write((ulong)bytes.Length); writer.Write(bytes); payload(writer);
        }
        public void Dispose() => File.Delete(Path);
    }
}
