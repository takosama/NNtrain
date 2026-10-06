using System.Text;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class GgufReaderLimitTests
{
    [Fact]
    public void NormalEmptyAndNestedArraysRemainSupported()
    {
        WithFile(w =>
        {
            w.Write((uint)GgufValueType.Array); w.Write(2ul);
            Array(w, GgufValueType.UInt32, 0);
            Array(w, GgufValueType.UInt32, 2); w.Write(3u); w.Write(4u);
        }, path =>
        {
            using var reader = new GgufReader(path, new() { MaximumNestingDepth = 2 });
            var outer = Assert.IsType<object[]>(reader.Metadata["k"]);
            Assert.Empty(Assert.IsType<object[]>(outer[0]));
            Assert.Equal(new object[] { 3u, 4u }, Assert.IsType<object[]>(outer[1]));
        });
    }

    [Theory]
    [InlineData(3ul)]
    [InlineData(ulong.MaxValue)]
    public void OversizeCountRejectedWithTinyLimit(ulong count)
        => WithFile(w => Array(w, GgufValueType.UInt8, count), path =>
            Assert.Throws<InvalidDataException>(() => new GgufReader(path, new() { MaximumArrayElements = 2 })));

    [Fact]
    public void ExactArrayAndStringLimitsAreAccepted()
        => WithFile(w =>
        {
            Array(w, GgufValueType.String, 2); String(w, "ab"); String(w, "cd");
        }, path =>
        {
            using var reader = new GgufReader(path, new() { MaximumArrayElements = 2, MaximumStringBytes = 2 });
            Assert.Equal(new object[] { "ab", "cd" }, Assert.IsType<object[]>(reader.Metadata["k"]));
        });

    [Fact]
    public void ImpossibleUnsignedTableCountsAreRejectedBeforeArithmetic()
    {
        string path = Path.GetTempFileName();
        try
        {
            using (var w = new BinaryWriter(File.Create(path)))
            { w.Write(0x46554747u); w.Write(3u); w.Write(ulong.MaxValue); w.Write(ulong.MaxValue); }
            Assert.Throws<InvalidDataException>(() => new GgufReader(path));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void MissingPayloadRejectedBeforeAnArrayCanBeAllocated()
        => WithFile(w => Array(w, GgufValueType.UInt64, 2), path =>
            Assert.Throws<EndOfStreamException>(() => new GgufReader(path)));

    [Fact]
    public void UnknownElementTypeRejectedEvenForEmptyArray()
        => WithFile(w => Array(w, (GgufValueType)123, 0), path =>
            Assert.Throws<NotSupportedException>(() => new GgufReader(path)));

    [Fact]
    public void ExcessDepthIsRejectedWhileNestedFormatIsRetained()
        => WithFile(w =>
        {
            Array(w, GgufValueType.Array, 1);
            Array(w, GgufValueType.UInt8, 0);
        }, path => Assert.Throws<InvalidDataException>(() => new GgufReader(path, new() { MaximumNestingDepth = 1 })));

    [Fact]
    public void ImmediateValueBudgetIsCheckedBeforeReadingElements()
        => WithFile(w => { Array(w, GgufValueType.UInt8, 4); w.Write(new byte[4]); }, path =>
            Assert.Throws<InvalidDataException>(() => new GgufReader(path, new() { MaximumDecodedBytes = 300 })));

    [Theory]
    [InlineData(3ul, false)]
    [InlineData(ulong.MaxValue, false)]
    [InlineData(3ul, true)]
    public void StringLimitsAndMissingBytesAreRejected(ulong length, bool missing)
        => WithFile(w => { w.Write(length); if (!missing && length == 3) w.Write(new byte[3]); }, path =>
        {
            if (missing) Assert.Throws<EndOfStreamException>(() => new GgufReader(path));
            else Assert.Throws<InvalidDataException>(() => new GgufReader(path, new() { MaximumStringBytes = 2 }));
        }, GgufValueType.String);

    [Fact]
    public void CumulativeStringsShareOneBudget()
        => WithFile(w =>
        {
            Array(w, GgufValueType.String, 2);
            String(w, "abcd"); String(w, "efgh");
        }, path => Assert.Throws<InvalidDataException>(() => new GgufReader(path, new() { MaximumDecodedBytes = 360 })));

    [Fact]
    public void ImpossibleTablesRejectedWithoutWalkingOrAllocatingThem()
    {
        string path = Path.GetTempFileName();
        try
        {
            using (var w = new BinaryWriter(File.Create(path)))
            { w.Write(0x46554747u); w.Write(3u); w.Write(0ul); w.Write(100ul); }
            Assert.Throws<EndOfStreamException>(() => new GgufReader(path));
        }
        finally { File.Delete(path); }
    }

    private static void WithFile(Action<BinaryWriter> value, Action<string> test, GgufValueType type = GgufValueType.Array)
    {
        string path = Path.GetTempFileName();
        try
        {
            using (var w = new BinaryWriter(File.Create(path)))
            {
                w.Write(0x46554747u); w.Write(3u); w.Write(0ul); w.Write(1ul);
                String(w, "k"); w.Write((uint)type); value(w);
            }
            test(path);
        }
        finally { File.Delete(path); }
    }
    private static void Array(BinaryWriter w, GgufValueType type, ulong count)
    { w.Write((uint)type); w.Write(count); }
    private static void String(BinaryWriter w, string value)
    { byte[] bytes = Encoding.UTF8.GetBytes(value); w.Write((ulong)bytes.Length); w.Write(bytes); }
}
