using System.Text;
using NNtrain;
using Xunit;

public sealed class SafeTensorDescriptorValidationTests
{
    [Theory]
    [InlineData(0, 8, 0, 8)]
    [InlineData(0, 8, 4, 12)]
    [InlineData(0, 12, 4, 8)]
    [InlineData(4, 8, 0, 12)]
    public void OverlapRejectedBeforePayloadReads(int a, int b, int c, int d)
    {
        using var stream = Fixture(Descriptor("a", "F32", (b-a)/4, a, b) + "," + Descriptor("b", "F32", (d-c)/4, c, d), 12);
        Assert.Throws<InvalidDataException>(() => SafeTensorFile.ReadKeys(stream));
        Assert.Equal(0, stream.PayloadReads);
        stream.Position = 0;
        Assert.Throws<InvalidDataException>(() => SafeTensorFile.Load(stream));
        Assert.Equal(0, stream.PayloadReads);
        WithPath(stream, path => Assert.Throws<InvalidDataException>(() => safetensors.torch.load_file(path)));
    }

    [Theory]
    [InlineData("F32", 4)]
    [InlineData("F16", 2)]
    [InlineData("BF16", 2)]
    public void AdjacentAndEmptyRangesAreValidAndKeysAreHeaderOnly(string dtype, int width)
    {
        using var stream = Fixture(Descriptor("0:a", dtype, 1, 0, width) + "," + Descriptor("1:b", dtype, 1, width, width*2) + "," + Descriptor("2:empty", dtype, 0, 0, 0), width*2);
        Assert.Equal(new[] { "0:a", "1:b", "2:empty" }, SafeTensorFile.ReadKeys(stream));
        Assert.Equal(0, stream.PayloadReads);
        WithPath(stream, path => Assert.Empty(safetensors.torch.load_file(path).Parameters[2].Values));
    }

    [Fact]
    public void CumulativeExpansionAndCountLimitsUseTinyFixtures()
    {
        using var stream = Fixture(Descriptor("a", "F16", 1, 0, 2) + "," + Descriptor("b", "BF16", 1, 2, 4), 4);
        Assert.Throws<InvalidDataException>(() => SafeTensorFile.ReadKeys(stream, new() { MaximumDecodedBytes = 7 }));
        Assert.Equal(0, stream.PayloadReads);
        stream.Position = 0;
        Assert.Throws<InvalidDataException>(() => SafeTensorFile.ReadKeys(stream, new() { MaximumTensors = 1 }));
    }

    [Theory]
    [InlineData("[2147483647,2]", "F32", 0, 4)]
    [InlineData("[1]", "unknown", 0, 4)]
    [InlineData("[1]", "F32", 0, 5)]
    [InlineData("[1]", "F32", -1, 3)]
    public void LateInvalidDescriptorDoesNotDecodeEarlierPayload(string shape, string dtype, int start, int end)
    {
        using var stream = Fixture(Descriptor("a", "F32", 1, 0, 4) + $",\"b\":{{\"dtype\":\"{dtype}\",\"shape\":{shape},\"data_offsets\":[{start},{end}]}}", 8);
        Assert.Throws<InvalidDataException>(() => SafeTensorFile.ReadKeys(stream));
        Assert.Equal(0, stream.PayloadReads);
        stream.Position = 0;
        Assert.Throws<InvalidDataException>(() => SafeTensorFile.Load(stream));
        Assert.Equal(0, stream.PayloadReads);
    }

    [Fact]
    public void HeaderRankAndExpandedLimitsAcceptBoundaryRejectPlusOne()
    {
        using var stream = Fixture(Descriptor("0:a", "F32", 1, 0, 4), 4);
        int headerLength = (int)stream.Length - 12;
        Assert.Single(SafeTensorFile.ReadKeys(stream, new() { MaximumHeaderBytes = headerLength, MaximumRank = 1, MaximumDecodedBytes = 4 }));
        stream.Position = 0;
        Assert.Throws<InvalidDataException>(() => SafeTensorFile.ReadKeys(stream, new() { MaximumHeaderBytes = headerLength - 1 }));
        stream.Position = 0;
        Assert.Throws<InvalidDataException>(() => SafeTensorFile.ReadKeys(stream, new() { MaximumRank = 0 }));
        stream.Position = 0;
        Assert.Throws<InvalidDataException>(() => SafeTensorFile.Load(stream, new() { MaximumDecodedBytes = 3 }));
        Assert.Equal(0, stream.PayloadReads);
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("{\"dtype\":\"F32\",\"shape\":\"bad\",\"data_offsets\":[0,4]}")]
    [InlineData("{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,9223372036854775808]}")]
    public void MalformedLaterDescriptorIsRejectedWithoutModelWrites(string invalid)
    {
        using var stream = Fixture(Descriptor("0:first", "F32", 1, 0, 4) + ",\"1:second\":" + invalid, 8);
        var model = new Pair();
        WithPath(stream, path => Assert.Throws<InvalidDataException>(() => SafeTensorFile.LoadModel(path, model)));
        Assert.Equal(new[] { 1f }, model.state_dict().Parameters[0].Values);
        Assert.Equal(new[] { 2f }, model.state_dict().Parameters[1].Values);
    }

    [Fact]
    public void LateInvalidPayloadDoesNotUpdateModel()
    {
        using var stream = Fixture(Descriptor("0:first", "F32", 1, 0, 4) + "," + Descriptor("1:second", "F32", 1, 4, 8), 8);
        var bytes = stream.ToArray();
        BitConverter.GetBytes(9f).CopyTo(bytes, bytes.Length-8);
        BitConverter.GetBytes(float.NaN).CopyTo(bytes, bytes.Length-4);
        var model = new Pair();
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(path, bytes);
            Assert.Throws<InvalidDataException>(() => SafeTensorFile.LoadModel(path, model));
            Assert.Equal(new[] { 1f }, model.state_dict().Parameters[0].Values);
        }
        finally { File.Delete(path); }
    }

    private sealed class Pair : Module
    {
        public Pair() : base(TensorDType.Float32)
        {
            RegisterParameter(new Parameter(new[] { 1f }, new[] { 1 }, "first", WeightDecayPolicy.Apply, TensorDType.Float32));
            RegisterParameter(new Parameter(new[] { 2f }, new[] { 1 }, "second", WeightDecayPolicy.Apply, TensorDType.Float32));
        }
    }
    private static string Descriptor(string key, string dtype, int count, int start, int end)
        => $"\"{key}\":{{\"dtype\":\"{dtype}\",\"shape\":[{count}],\"data_offsets\":[{start},{end}]}}";
    private static TrackingStream Fixture(string descriptors, int payload)
    {
        byte[] header = Encoding.UTF8.GetBytes("{" + descriptors + "}");
        var bytes = new byte[8+header.Length+payload];
        BitConverter.GetBytes((ulong)header.Length).CopyTo(bytes,0);
        header.CopyTo(bytes,8);
        return new TrackingStream(bytes, 8+header.Length);
    }
    private static void WithPath(TrackingStream stream, Action<string> test)
    {
        string path = Path.GetTempFileName();
        try { File.WriteAllBytes(path,stream.ToArray()); test(path); }
        finally { File.Delete(path); }
    }
    private sealed class TrackingStream(byte[] bytes, int dataStart) : MemoryStream(bytes)
    {
        public int PayloadReads { get; private set; }
        public override int Read(Span<byte> buffer)
        {
            if (Position >= dataStart && buffer.Length > 0) PayloadReads++;
            return base.Read(buffer);
        }
    }
}
