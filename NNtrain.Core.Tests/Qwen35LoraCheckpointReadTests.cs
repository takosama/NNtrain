using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35LoraCheckpointReadTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReadsEveryByteOnceAndRetainsOnlyTheRequestedState(bool training)
    {
        var fixture = CreateCheckpoint();
        using var stream = new ShortReadStream(fixture.Bytes, 127);
        var loaded = Qwen35QuantizedModel.ReadLoraCheckpoint(stream, training, header =>
        {
            Assert.Equal(fixture.Header, header);
            return fixture.Shapes;
        });
        Assert.Equal(fixture.Bytes.Length, stream.BytesRead);
        Assert.Equal(0, stream.Seeks);
        Assert.Equal(fixture.Bytes.Length, stream.Position);
        Assert.Equal(fixture.Header, loaded.Header);
        Assert.Equal(fixture.States.Length, loaded.States.Length);
        for (int entry = 0; entry < fixture.States.Length; entry++)
        {
            Assert.Equal(6, loaded.States[entry].Length);
            for (int field = 0; field < 6; field++)
                if (training || field < 2)
                    Assert.Equal(fixture.States[entry][field].Select(BitConverter.SingleToInt32Bits),
                        loaded.States[entry][field].Select(BitConverter.SingleToInt32Bits));
                else Assert.Empty(loaded.States[entry][field]);
        }
    }

    [Fact]
    public void ChecksumCoversHeaderParametersDiscardedMomentsAndTrailer()
    {
        var fixture = CreateCheckpoint();
        foreach (int offset in new[] { 12 + 13, fixture.Offsets[0][0], fixture.Offsets[0][2], fixture.Bytes.Length - 1 })
        {
            byte[] corrupted = (byte[])fixture.Bytes.Clone();
            corrupted[offset] ^= 1;
            using var stream = new ShortReadStream(corrupted, 193);
            Assert.Throws<InvalidDataException>(() => Qwen35QuantizedModel.ReadLoraCheckpoint(
                stream, false, _ => fixture.Shapes));
            Assert.Equal(0, stream.Seeks);
        }
    }

    [Fact]
    public void DiscardedOptimizerArraysAreFullyValidatedIncludingChunkTails()
    {
        var fixture = CreateCheckpoint();
        for (int field = 0; field < 6; field++)
        foreach (float invalid in field >= 4
            ? new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, -0.25f }
            : new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        foreach (int index in new[] { 0, fixture.States[0][field].Length - 1 })
        {
            byte[] corrupted = (byte[])fixture.Bytes.Clone();
            BinaryPrimitives.WriteSingleLittleEndian(corrupted.AsSpan(fixture.Offsets[0][field] + index * 4, 4), invalid);
            Rehash(corrupted);
            using var stream = new MemoryStream(corrupted, writable: false);
            Assert.Throws<InvalidDataException>(() => Qwen35QuantizedModel.ReadLoraCheckpoint(
                stream, false, _ => fixture.Shapes));
        }
    }

    [Fact]
    public void RejectsTruncationAndRehashedTrailingTensorBytes()
    {
        var fixture = CreateCheckpoint();
        foreach (int length in new[] { 0, 11, 43, fixture.Bytes.Length - 1, fixture.Bytes.Length - 65 })
        {
            using var stream = new MemoryStream(fixture.Bytes[..length], writable: false);
            Assert.Throws<InvalidDataException>(() => Qwen35QuantizedModel.ReadLoraCheckpoint(
                stream, false, _ => fixture.Shapes));
        }
        var trailing = new byte[fixture.Bytes.Length + 4];
        fixture.Bytes.AsSpan(0, fixture.Bytes.Length - 32).CopyTo(trailing);
        Rehash(trailing);
        using var extra = new MemoryStream(trailing, writable: false);
        Assert.Throws<InvalidDataException>(() => Qwen35QuantizedModel.ReadLoraCheckpoint(
            extra, false, _ => fixture.Shapes));
    }

    [Fact]
    public void HeaderValidationRejectsBeforeReturningAnyState()
    {
        var fixture = CreateCheckpoint();
        using var stream = new ShortReadStream(fixture.Bytes, 127);
        Assert.Throws<InvalidDataException>(() => Qwen35QuantizedModel.ReadLoraCheckpoint(stream, false,
            _ => throw new InvalidDataException("Model, directory or configuration mismatch.")));
        Assert.Equal(12 + fixture.Header.Length, stream.BytesRead);
    }

    private sealed record Fixture(byte[] Bytes, byte[] Header,
        (int Input, int Output, int Rank)[] Shapes, float[][][] States, int[][] Offsets);

    private static Fixture CreateCheckpoint()
    {
        byte[] header = Encoding.UTF8.GetBytes("{\"fixture\":\"load-v1\"}");
        (int Input, int Output, int Rank)[] shapes = [(17013, 29, 2), (7, 11, 3)];
        var states = new float[shapes.Length][][];
        var offsets = new int[shapes.Length][];
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
        {
            // Independent v1 writer, retaining the established BinaryWriter layout.
            writer.Write(Encoding.ASCII.GetBytes("NNQ35LR1"));
            writer.Write(header.Length);
            writer.Write(header);
            for (int entry = 0; entry < shapes.Length; entry++)
            {
                states[entry] = new float[6][];
                offsets[entry] = new int[6];
                for (int field = 0; field < 6; field++)
                {
                    var shape = shapes[entry];
                    int count = (field % 2 == 0 ? shape.Input : shape.Output) * shape.Rank;
                    states[entry][field] = Enumerable.Range(0, count)
                        .Select(i => field >= 4 ? (i % 31 + 1) * .003f : (i % 31 - 15) * .003f).ToArray();
                    // Signed zero is valid for both parameters and second moments.
                    states[entry][field][0] = BitConverter.Int32BitsToSingle(unchecked((int)0x80000000));
                    offsets[entry][field] = checked((int)stream.Position);
                    foreach (float value in states[entry][field]) writer.Write(value);
                }
            }
        }
        byte[] payload = stream.ToArray();
        return new(payload.Concat(SHA256.HashData(payload)).ToArray(), header, shapes, states, offsets);
    }

    private static void Rehash(byte[] bytes)
        => SHA256.HashData(bytes.AsSpan(0, bytes.Length - 32)).CopyTo(bytes.AsSpan(bytes.Length - 32));

    private sealed class ShortReadStream(byte[] bytes, int maxRead) : Stream
    {
        private readonly MemoryStream _inner = new(bytes, writable: false);
        internal long BytesRead { get; private set; }
        internal int Seeks { get; private set; }
        public override bool CanRead => true;
        public override bool CanSeek => true;
        public override bool CanWrite => false;
        public override long Length => _inner.Length;
        public override int Read(Span<byte> buffer)
        {
            int count = _inner.Read(buffer[..Math.Min(buffer.Length, maxRead)]);
            BytesRead += count;
            return count;
        }
        public override int Read(byte[] buffer, int offset, int count)
        {
            int read = _inner.Read(buffer, offset, Math.Min(count, maxRead));
            BytesRead += read;
            return read;
        }
        public override long Seek(long offset, SeekOrigin loc)
        {
            Seeks++;
            throw new InvalidOperationException("The reader must not seek or reread the checkpoint.");
        }
        public override long Position
        {
            get => _inner.Position;
            set
            {
                Seeks++;
                throw new InvalidOperationException("The reader must not rewind the checkpoint.");
            }
        }
        public override void Flush() { }
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        protected override void Dispose(bool disposing)
        {
            if (disposing) _inner.Dispose();
            base.Dispose(disposing);
        }
    }
}
