using System.Text;
using NNtrain.Training.Optimization;

namespace NNtrain;

/// <summary>
/// Reads legacy optimizer StateJson values directly from Wiki or
/// classification checkpoints. It deliberately never creates an aggregate
/// OptimizerStateDictionary or JsonElement for the payload; each leaf state is
/// exposed as a bounded stream and deserialized into its owned moment arrays.
/// </summary>
internal static class CheckpointOptimizerStateStream
{
    private static readonly byte[] OptimizerTypeProperty =
        Encoding.UTF8.GetBytes("\"OptimizerType\"");
    private static readonly byte[] StateJsonProperty =
        Encoding.UTF8.GetBytes("\"StateJson\"");

    internal static bool TryLoad(
        string checkpointPath,
        WikiLanguageModelCommand.WikiModelCheckpoint checkpoint,
        IOptimizer optimizer,
        TextWriter output)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(output);

        IReadOnlyList<IOptimizer> leaves =
            OptimizerBundle.GetCheckpointLeafOptimizers(optimizer);
        if (checkpoint.FormatVersion >= 7)
        {
            return TryLoadArtifacts(
                checkpointPath,
                checkpoint,
                leaves,
                output);
        }

        return TryLoadLegacyJson(checkpointPath, leaves, output);
    }

    internal static bool TryLoadLegacyJson(
        string checkpointPath,
        IOptimizer optimizer,
        TextWriter output)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(output);
        return TryLoadLegacyJson(
            checkpointPath,
            OptimizerBundle.GetCheckpointLeafOptimizers(optimizer),
            output);
    }

    private static bool TryLoadLegacyJson(
        string checkpointPath,
        IReadOnlyList<IOptimizer> leaves,
        TextWriter output)
    {
        using var stream = new FileStream(
            Path.GetFullPath(checkpointPath),
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 4 * 1024 * 1024,
            FileOptions.SequentialScan);
        var reader = new BufferedByteReader(stream);
        int leafIndex = 0;
        while (leafIndex < leaves.Count)
        {
            if (!Find(reader, OptimizerTypeProperty))
            {
                if (leafIndex == 0)
                    return false;
                throw new InvalidDataException(
                    $"Checkpoint '{Path.GetFullPath(checkpointPath)}' ended " +
                    $"after restoring {leafIndex} of {leaves.Count} optimizer " +
                    "leaves. Refusing to continue with a partially restored " +
                    "composite optimizer.");
            }
            string serializedType = ReadPropertyString(reader);
            if (!Find(reader, StateJsonProperty))
                throw InvalidOptimizerPayload(checkpointPath);
            int first = ReadPropertyValueStart(reader);
            if (first == 'n')
            {
                RequireBytes(reader, "ull"u8);
                continue;
            }
            if (first != '{')
            {
                throw new InvalidDataException(
                    "Streaming resume requires object-valued optimizer " +
                    "StateJson. This checkpoint uses an unsupported legacy " +
                    "string representation.");
            }

            IOptimizer leaf = leaves[leafIndex];
            string expectedType = OptimizerStateStream.GetStateType(leaf);
            if (!string.Equals(
                    serializedType,
                    expectedType,
                    StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    $"Checkpoint optimizer leaf {leafIndex} is " +
                    $"'{serializedType}', but the configured optimizer " +
                    $"expects '{expectedType}'.");
            }

            using var valueStream = new JsonValueReadStream(reader, (byte)first);
            OptimizerStateStream.LoadStateJson(leaf, valueStream);
            if (!valueStream.Completed)
            {
                throw new InvalidDataException(
                    $"Checkpoint optimizer state '{serializedType}' was " +
                    "not consumed completely.");
            }
            output.WriteLine(
                $"streamed optimizer state = {serializedType} " +
                $"({leafIndex + 1}/{leaves.Count})");
            leafIndex++;

            // The optimizer constructor allocated an initial state which has
            // just been replaced. Reclaim it before reading the next large
            // child state rather than stacking both peaks.
            GC.Collect(
                GC.MaxGeneration,
                GCCollectionMode.Aggressive,
                blocking: true,
                compacting: true);
        }
        return true;
    }

    private static bool TryLoadArtifacts(
        string checkpointPath,
        WikiLanguageModelCommand.WikiModelCheckpoint checkpoint,
        IReadOnlyList<IOptimizer> leaves,
        TextWriter output)
    {
        string[]? serializedTypes = checkpoint.OptimizerStateTypes;
        if (serializedTypes is null || serializedTypes.Length == 0)
            return false;
        if (checkpoint.ArtifactSlot is < 0 or > 1
            || serializedTypes.Length != leaves.Count)
        {
            throw new InvalidDataException(
                "Checkpoint optimizer artifact metadata is invalid.");
        }

        for (int leafIndex = 0; leafIndex < leaves.Count; leafIndex++)
        {
            IOptimizer leaf = leaves[leafIndex];
            string expectedType = OptimizerStateStream.GetStateType(leaf);
            if (!string.Equals(
                    serializedTypes[leafIndex],
                    expectedType,
                    StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    $"Checkpoint optimizer leaf {leafIndex} is " +
                    $"'{serializedTypes[leafIndex]}', but the configured " +
                    $"optimizer expects '{expectedType}'.");
            }

            string artifactPath = checkpoint.FormatVersion >= 8
                ? WikiLanguageModelCommand.GetOptimizerBinaryArtifactPath(
                    checkpointPath,
                    checkpoint.ArtifactSlot,
                    leafIndex)
                : WikiLanguageModelCommand.GetOptimizerArtifactPath(
                    checkpointPath,
                    checkpoint.ArtifactSlot,
                    leafIndex);
            if (!File.Exists(artifactPath))
            {
                throw new FileNotFoundException(
                    "Checkpoint optimizer artifact was not found.",
                    artifactPath);
            }
            using var stream = new FileStream(
                artifactPath,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                bufferSize: 4 * 1024 * 1024,
                FileOptions.SequentialScan);
            if (checkpoint.FormatVersion >= 8)
                OptimizerStateStream.LoadStateBinary(leaf, stream);
            else
                OptimizerStateStream.LoadStateJson(leaf, stream);
            output.WriteLine(
                $"streamed optimizer state = {expectedType} " +
                $"({leafIndex + 1}/{leaves.Count})");
            GC.Collect(
                GC.MaxGeneration,
                GCCollectionMode.Aggressive,
                blocking: true,
                compacting: true);
        }
        return true;
    }

    private static bool Find(
        BufferedByteReader reader,
        ReadOnlySpan<byte> pattern)
    {
        int matched = 0;
        while (reader.TryRead(out byte value))
        {
            if (value == pattern[matched])
            {
                matched++;
                if (matched == pattern.Length)
                    return true;
                continue;
            }
            matched = value == pattern[0] ? 1 : 0;
        }
        return false;
    }

    private static string ReadPropertyString(BufferedByteReader reader)
    {
        int first = ReadPropertyValueStart(reader);
        if (first != '"')
            throw InvalidOptimizerPayload(reader.Path);
        var bytes = new List<byte>(32);
        bool escaped = false;
        while (reader.TryRead(out byte value))
        {
            if (escaped)
            {
                // Optimizer type names are ASCII and never escaped. Rejecting
                // escapes here keeps corrupted metadata unambiguous.
                throw InvalidOptimizerPayload(reader.Path);
            }
            if (value == '\\')
            {
                escaped = true;
                continue;
            }
            if (value == '"')
                return Encoding.UTF8.GetString(bytes.ToArray());
            bytes.Add(value);
        }
        throw InvalidOptimizerPayload(reader.Path);
    }

    private static int ReadPropertyValueStart(BufferedByteReader reader)
    {
        int value = ReadNonWhitespace(reader);
        if (value != ':')
            throw InvalidOptimizerPayload(reader.Path);
        return ReadNonWhitespace(reader);
    }

    private static int ReadNonWhitespace(BufferedByteReader reader)
    {
        while (reader.TryRead(out byte value))
        {
            if (value is not (byte)' ' and not (byte)'\t'
                and not (byte)'\r' and not (byte)'\n')
            {
                return value;
            }
        }
        throw InvalidOptimizerPayload(reader.Path);
    }

    private static void RequireBytes(
        BufferedByteReader reader,
        ReadOnlySpan<byte> expected)
    {
        foreach (byte expectedByte in expected)
        {
            if (!reader.TryRead(out byte actual) || actual != expectedByte)
                throw InvalidOptimizerPayload(reader.Path);
        }
    }

    private static InvalidDataException InvalidOptimizerPayload(string path)
        => new(
            $"Checkpoint '{Path.GetFullPath(path)}' contains an invalid " +
            "optimizer payload.");

    private sealed class BufferedByteReader
    {
        private readonly Stream _stream;
        private readonly byte[] _buffer = new byte[4 * 1024 * 1024];
        private int _offset;
        private int _count;

        internal BufferedByteReader(FileStream stream)
        {
            _stream = stream;
            Path = stream.Name;
        }

        internal string Path { get; }

        internal bool TryRead(out byte value)
        {
            if (_offset == _count)
            {
                _count = _stream.Read(_buffer, 0, _buffer.Length);
                _offset = 0;
                if (_count == 0)
                {
                    value = 0;
                    return false;
                }
            }
            value = _buffer[_offset++];
            return true;
        }
    }

    private sealed class JsonValueReadStream : Stream
    {
        private readonly BufferedByteReader _reader;
        private byte _first;
        private bool _hasFirst = true;
        private int _depth = 1;
        private bool _inString;
        private bool _escaped;

        internal JsonValueReadStream(
            BufferedByteReader reader,
            byte first)
        {
            _reader = reader;
            _first = first;
        }

        internal bool Completed { get; private set; }

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }

        public override int Read(byte[] buffer, int offset, int count)
            => Read(buffer.AsSpan(offset, count));

        public override int Read(Span<byte> buffer)
        {
            if (Completed || buffer.Length == 0)
                return 0;
            int written = 0;
            if (_hasFirst)
            {
                buffer[written++] = _first;
                _hasFirst = false;
            }
            while (written < buffer.Length && _depth > 0)
            {
                if (!_reader.TryRead(out byte value))
                {
                    throw new InvalidDataException(
                        "Optimizer StateJson ended before its closing token.");
                }
                buffer[written++] = value;
                Advance(value);
            }
            if (_depth == 0)
                Completed = true;
            return written;
        }

        private void Advance(byte value)
        {
            if (_inString)
            {
                if (_escaped)
                {
                    _escaped = false;
                    return;
                }
                if (value == '\\')
                    _escaped = true;
                else if (value == '"')
                    _inString = false;
                return;
            }
            if (value == '"')
                _inString = true;
            else if (value is (byte)'{' or (byte)'[')
                _depth++;
            else if (value is (byte)'}' or (byte)']')
                _depth--;
        }

        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin)
            => throw new NotSupportedException();
        public override void SetLength(long value)
            => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count)
            => throw new NotSupportedException();
    }
}
