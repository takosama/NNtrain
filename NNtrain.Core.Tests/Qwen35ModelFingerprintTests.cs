using System.Security.Cryptography;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35ModelFingerprintTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(4 * 1024 * 1024 - 1)]
    [InlineData(4 * 1024 * 1024)]
    [InlineData(4 * 1024 * 1024 + 1)]
    [InlineData(12 * 1024 * 1024 + 17)]
    public void HashesTheEntireStreamFromItsBeginning(int length)
    {
        byte[] bytes = CreateBytes(length);
        using var stream = new MemoryStream(bytes, writable: false);
        stream.Position = length / 2;
        string actual = Qwen35QuantizedModel.ComputeModelFingerprint(stream);
        Assert.Equal(Convert.ToHexString(SHA256.HashData(bytes)), actual);
        Assert.Equal(length, stream.Position);
        Assert.True(stream.CanRead);
        // A second call must again read the whole snapshot, even at EOF.
        Assert.Equal(actual, Qwen35QuantizedModel.ComputeModelFingerprint(stream));
    }

    [Fact]
    public void HashesEveryByteExactlyOnceWithShortReads()
    {
        byte[] bytes = CreateBytes(8 * 1024 * 1024 + 53);
        using var stream = new ShortReadStream(bytes, maximumRead: 1031);
        stream.Position = 97;
        Assert.Equal(Convert.ToHexString(SHA256.HashData(bytes)),
            Qwen35QuantizedModel.ComputeModelFingerprint(stream));
        Assert.Equal(bytes.Length, stream.BytesRead);
        Assert.Equal(bytes.Length, stream.Position);
        Assert.True(stream.CanRead);
        Assert.Equal(4 * 1024 * 1024, stream.MaximumRequestedRead);
    }

    [Fact]
    public void EveryPartOfTheSnapshotAffectsItsIdentity()
    {
        byte[] bytes = CreateBytes(8 * 1024 * 1024 + 53);
        string original = Convert.ToHexString(SHA256.HashData(bytes));
        foreach (int offset in new[] { 0, 4 * 1024 * 1024 - 1, 4 * 1024 * 1024, bytes.Length - 1 })
        {
            bytes[offset] ^= 0x80;
            using var stream = new MemoryStream(bytes, writable: false);
            string actual = Qwen35QuantizedModel.ComputeModelFingerprint(stream);
            Assert.NotEqual(original, actual);
            Assert.Equal(Convert.ToHexString(SHA256.HashData(bytes)), actual);
            bytes[offset] ^= 0x80;
        }
    }

    [Fact]
    public void ReadFailurePropagatesWithoutClosingTheProtectedSource()
    {
        using var stream = new ShortReadStream(CreateBytes(1024), maximumRead: 97, failAfter: 400);
        Assert.Throws<IOException>(() => Qwen35QuantizedModel.ComputeModelFingerprint(stream));
        Assert.True(stream.CanRead);
    }

    private static byte[] CreateBytes(int count)
    {
        byte[] bytes = new byte[count];
        new Random(913).NextBytes(bytes);
        return bytes;
    }

    private sealed class ShortReadStream(byte[] bytes, int maximumRead, int? failAfter = null) : Stream
    {
        private readonly MemoryStream _inner = new(bytes, writable: false);
        public long BytesRead { get; private set; }
        public int MaximumRequestedRead { get; private set; }
        public override bool CanRead => _inner.CanRead;
        public override bool CanSeek => true;
        public override bool CanWrite => false;
        public override long Length => _inner.Length;
        public override long Position { get => _inner.Position; set => _inner.Position = value; }
        public override int Read(Span<byte> buffer)
        {
            if (failAfter is not null && BytesRead >= failAfter.Value)
                throw new IOException("Simulated protected source read failure.");
            MaximumRequestedRead = Math.Max(MaximumRequestedRead, buffer.Length);
            int count = _inner.Read(buffer[..Math.Min(buffer.Length, maximumRead)]);
            BytesRead += count;
            return count;
        }
        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));
        public override long Seek(long offset, SeekOrigin origin) => _inner.Seek(offset, origin);
        public override void Flush() => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        protected override void Dispose(bool disposing)
        {
            if (disposing) _inner.Dispose();
            base.Dispose(disposing);
        }
    }
}
