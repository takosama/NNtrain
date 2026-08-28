using NNtrain.Training.Execution;

namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    internal readonly record struct WikiTrainingBatch(
        LanguageBatch Values,
        int BatchIndex,
        int BatchSize,
        int SequenceLength,
        long DocumentsProcessed);

    internal sealed class FixedWikiTrainingDataCursor
        : ITrainingDataCursor<WikiTrainingBatch>
    {
        private readonly IReadOnlyList<int> _tokens;
        private readonly IReadOnlyList<int> _sequenceOrder;
        private readonly int _batchSize;
        private readonly int _sequenceLength;
        private long _position;

        internal FixedWikiTrainingDataCursor(
            IReadOnlyList<int> tokens,
            IReadOnlyList<int> sequenceOrder,
            int batchSize,
            int sequenceLength)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            ArgumentNullException.ThrowIfNull(sequenceOrder);
            if (batchSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(batchSize));
            if (sequenceLength <= 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(sequenceLength));
            }
            _tokens = tokens;
            _sequenceOrder = sequenceOrder;
            _batchSize = batchSize;
            _sequenceLength = sequenceLength;
        }

        public long Position => _position;

        internal int BatchTotal => TrainingRunner.DivideRoundUp(
            _sequenceOrder.Count,
            _batchSize);

        internal int NextBatchSize
        {
            get
            {
                long orderStart = checked(_position * _batchSize);
                if (orderStart >= _sequenceOrder.Count)
                    return 0;
                return (int)Math.Min(
                    _batchSize,
                    _sequenceOrder.Count - orderStart);
            }
        }

        internal void StartEpoch(int completedBatches)
        {
            if ((uint)completedBatches > (uint)BatchTotal)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(completedBatches));
            }
            _position = completedBatches;
        }

        public WikiTrainingBatch AcquireNext()
        {
            int count = NextBatchSize;
            if (count == 0)
            {
                throw new InvalidOperationException(
                    "The fixed Wiki data cursor is at the end of its epoch.");
            }
            int batchIndex = checked((int)_position);
            int orderStart = checked(batchIndex * _batchSize);
            LanguageBatch values = CreateBatch(
                _tokens,
                _sequenceOrder,
                orderStart,
                count,
                _sequenceLength);
            _position++;
            return new WikiTrainingBatch(
                values,
                batchIndex,
                count,
                _sequenceLength,
                DocumentsProcessed: 0);
        }
    }

    internal sealed class StreamingWikiTrainingDataCursor
        : ITrainingDataCursor<WikiTrainingBatch>
    {
        private readonly List<int> _buffer;
        private long _position;
        private int _batchSize;
        private int _sequenceLength;
        private long _documentsProcessed;
        private bool _configured;

        internal StreamingWikiTrainingDataCursor(List<int> buffer)
        {
            ArgumentNullException.ThrowIfNull(buffer);
            _buffer = buffer;
        }

        public long Position => _position;

        internal IReadOnlyList<int> BufferedTokens => _buffer;

        internal void StartEpoch(int completedBatches)
        {
            if (completedBatches < 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(completedBatches));
            }
            _position = completedBatches;
            _configured = false;
        }

        internal void ConfigureNext(
            int batchSize,
            int sequenceLength,
            long documentsProcessed)
        {
            if (batchSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(batchSize));
            if (sequenceLength <= 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(sequenceLength));
            }
            if (documentsProcessed < 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(documentsProcessed));
            }
            _batchSize = batchSize;
            _sequenceLength = sequenceLength;
            _documentsProcessed = documentsProcessed;
            _configured = true;
        }

        public WikiTrainingBatch AcquireNext()
        {
            if (!_configured)
            {
                throw new InvalidOperationException(
                    "The streaming Wiki data cursor has no configured batch.");
            }
            int batchIndex = checked((int)_position);
            LanguageBatch values = CreateStreamingBatch(
                _buffer,
                _batchSize,
                _sequenceLength);
            _position++;
            _configured = false;
            return new WikiTrainingBatch(
                values,
                batchIndex,
                _batchSize,
                _sequenceLength,
                _documentsProcessed);
        }
    }
}
