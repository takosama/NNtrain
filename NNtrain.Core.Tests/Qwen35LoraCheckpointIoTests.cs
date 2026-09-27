using System.Security.Cryptography;
using System.Text;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35LoraCheckpointIoTests
{
    private static readonly byte[] Header = Encoding.UTF8.GetBytes(
        "{\"Version\":1,\"ModelSha256\":\"base\",\"TrainingIdentity\":\"つくよみ\",\"Step\":1536}");

    [Fact]
    public void SinglePassWriterPreservesVersionOneBytesAndDigestExactly()
    {
        float[][] states =
        [
            [0f, -0f, float.Epsilon, -float.Epsilon],
            [1f, -1f, float.MaxValue, float.MinValue],
            [.125f, -2.5f, 1e-30f, -1e30f],
            [.33f, -.33f, 65536f, -65536f],
            [0f, float.Epsilon, .75f, float.MaxValue],
            [2f, 4f, 8f, 16f],
        ];
        using var stream = new MemoryStream();
        Qwen35QuantizedModel.WriteLoraCheckpoint(stream, Header, states);
        byte[] actual = stream.ToArray();
        Assert.Equal(LegacyCheckpoint(Header, states), actual);
        Assert.Equal(SHA256.HashData(actual.AsSpan(0, actual.Length - 32)), actual[^32..]);
    }

    [Fact]
    public void LargeCheckpointUsesOneWritePassAndEnumeratesStateOnce()
    {
        // Cross the old 1 MiB reread buffer boundary. A stream without read,
        // position or seek support enforces the I/O reduction without a timing
        // threshold that would depend on disk cache or machine load.
        float[] values = Enumerable.Range(0, 300_001).Select(i => (i - 150_000) * .125f).ToArray();
        int enumerations = 0;
        IEnumerable<float[]> States()
        {
            Assert.Equal(1, ++enumerations);
            yield return values;
            yield return values;
            yield return [];
        }
        using var stream = new WriteOnlyStream();
        Qwen35QuantizedModel.WriteLoraCheckpoint(stream, Header, States());
        byte[] actual = stream.Snapshot();
        Assert.Equal(1, enumerations);
        Assert.Equal(8L + sizeof(int) + Header.Length + 2L * values.Length * sizeof(float) + 32,
            stream.BytesWritten);
        Assert.Equal(LegacyCheckpoint(Header, [values, values, []]), actual);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void NonFiniteStateFailsBeforeWritingInvalidArrayOrChecksum(float invalid)
    {
        using var stream = new WriteOnlyStream();
        float[][] states = [[1f, 2f], [3f, invalid, 4f]];
        Assert.Throws<ArithmeticException>(() =>
            Qwen35QuantizedModel.WriteLoraCheckpoint(stream, Header, states));
        Assert.Equal(8L + sizeof(int) + Header.Length + 2 * sizeof(float), stream.BytesWritten);
    }

    private static byte[] LegacyCheckpoint(byte[] header, IEnumerable<float[]> states)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write("NNQ35LR1"u8);
        writer.Write(header.Length);
        writer.Write(header);
        foreach (float[] values in states)
            foreach (float value in values) writer.Write(value);
        writer.Flush();
        byte[] digest = SHA256.HashData(stream.ToArray());
        writer.Write(digest);
        writer.Flush();
        return stream.ToArray();
    }

    private sealed class WriteOnlyStream : Stream
    {
        private readonly MemoryStream _sink = new();
        internal long BytesWritten { get; private set; }
        internal byte[] Snapshot() => _sink.ToArray();
        public override bool CanRead => false;
        public override bool CanSeek => false;
        public override bool CanWrite => true;
        public override long Length => throw new NotSupportedException();
        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }
        public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));
        public override void Write(ReadOnlySpan<byte> buffer)
        {
            _sink.Write(buffer);
            BytesWritten += buffer.Length;
        }
        public override void Flush() => _sink.Flush();
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        protected override void Dispose(bool disposing)
        {
            if (disposing) _sink.Dispose();
            base.Dispose(disposing);
        }
    }
}
