using NNtrain.Training.Execution;

namespace NNtrain;

internal static partial class Program
{
    internal readonly record struct ClassificationUpdateBatch
    {
        private readonly ClassificationTrainingDataCursor _cursor;
        private readonly int _generation;

        internal ClassificationUpdateBatch(
            ClassificationTrainingDataCursor cursor,
            int generation,
            int update,
            int updateTotal,
            int firstMicroBatch,
            int microBatchCount,
            int microBatchTotal,
            int sampleCount)
        {
            _cursor = cursor;
            _generation = generation;
            Update = update;
            UpdateTotal = updateTotal;
            FirstMicroBatch = firstMicroBatch;
            MicroBatchCount = microBatchCount;
            MicroBatchTotal = microBatchTotal;
            SampleCount = sampleCount;
        }

        internal int Update { get; }

        internal int UpdateTotal { get; }

        internal int FirstMicroBatch { get; }

        internal int MicroBatchCount { get; }

        internal int MicroBatchTotal { get; }

        internal int SampleCount { get; }

        internal DataBatch AcquireMicroBatch(int accumulationIndex)
            => _cursor.AcquireMicroBatch(
                _generation,
                accumulationIndex);
    }

    /// <summary>
    /// Owns classification DataLoader advancement, including deterministic
    /// checkpoint replay. One task-facing update reads its first microbatch in
    /// <see cref="AcquireNext"/> and streams the remaining microbatches through
    /// the same cursor during gradient accumulation.
    /// </summary>
    internal sealed class ClassificationTrainingDataCursor
        : ITrainingDataCursor<ClassificationUpdateBatch>
    {
        private IEnumerator<DataBatch>? _batches;
        private DataBatch? _firstBatch;
        private long _position;
        private int _generation;
        private int _activeGeneration;
        private int _nextAccumulationIndex;
        private int _update;
        private int _updateTotal;
        private int _firstMicroBatch;
        private int _microBatchCount;
        private int _microBatchTotal;
        private int _sampleCount;
        private bool _configured;

        public long Position => _position;

        internal void StartEpoch(
            IEnumerator<DataBatch> batches,
            int microBatchesToSkip)
        {
            ArgumentNullException.ThrowIfNull(batches);
            if (microBatchesToSkip < 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(microBatchesToSkip));
            }
            if (_activeGeneration != 0)
            {
                throw new InvalidOperationException(
                    "The previous classification update was not fully consumed.");
            }

            _batches = batches;
            _position = 0;
            _configured = false;
            _firstBatch = null;
            for (int skipped = 0; skipped < microBatchesToSkip; skipped++)
                MoveNextOrThrow("restoring checkpoint position");
        }

        internal void ConfigureNext(
            int update,
            int updateTotal,
            int firstMicroBatch,
            int microBatchCount,
            int microBatchTotal,
            int sampleCount)
        {
            if (_batches is null)
            {
                throw new InvalidOperationException(
                    "The classification cursor has no active epoch.");
            }
            if (_activeGeneration != 0)
            {
                throw new InvalidOperationException(
                    "The previous classification update was not fully consumed.");
            }
            if ((uint)update >= (uint)updateTotal)
                throw new ArgumentOutOfRangeException(nameof(update));
            if (firstMicroBatch < 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(firstMicroBatch));
            }
            if (microBatchCount <= 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(microBatchCount));
            }
            if (microBatchTotal <= 0
                || firstMicroBatch > microBatchTotal - microBatchCount)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(microBatchTotal));
            }
            if (sampleCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(sampleCount));
            if (_position != firstMicroBatch)
            {
                throw new InvalidOperationException(
                    $"Classification cursor position {_position} does not " +
                    $"match update start {firstMicroBatch}.");
            }

            _update = update;
            _updateTotal = updateTotal;
            _firstMicroBatch = firstMicroBatch;
            _microBatchCount = microBatchCount;
            _microBatchTotal = microBatchTotal;
            _sampleCount = sampleCount;
            _configured = true;
        }

        public ClassificationUpdateBatch AcquireNext()
        {
            if (!_configured)
            {
                throw new InvalidOperationException(
                    "The classification cursor has no configured update.");
            }
            _firstBatch = MoveNextOrThrow("acquiring an update");
            int generation = unchecked(++_generation);
            if (generation == 0)
                generation = unchecked(++_generation);
            _activeGeneration = generation;
            _nextAccumulationIndex = 0;
            _configured = false;
            return new ClassificationUpdateBatch(
                this,
                generation,
                _update,
                _updateTotal,
                _firstMicroBatch,
                _microBatchCount,
                _microBatchTotal,
                _sampleCount);
        }

        internal DataBatch AcquireMicroBatch(
            int generation,
            int accumulationIndex)
        {
            if (generation != _activeGeneration)
            {
                throw new InvalidOperationException(
                    "The classification update batch is stale.");
            }
            if (accumulationIndex != _nextAccumulationIndex
                || accumulationIndex >= _microBatchCount)
            {
                throw new InvalidOperationException(
                    "Classification microbatches must be consumed once in order.");
            }

            DataBatch batch;
            if (accumulationIndex == 0)
            {
                batch = _firstBatch
                    ?? throw new InvalidOperationException(
                        "The first classification microbatch was not retained.");
            }
            else
            {
                batch = MoveNextOrThrow("accumulating an update");
            }

            _nextAccumulationIndex++;
            if (_nextAccumulationIndex == _microBatchCount)
            {
                _firstBatch = null;
                _activeGeneration = 0;
            }
            return batch;
        }

        private DataBatch MoveNextOrThrow(string operation)
        {
            IEnumerator<DataBatch> batches = _batches
                ?? throw new InvalidOperationException(
                    "The classification cursor has no active epoch.");
            if (!batches.MoveNext())
            {
                throw new InvalidDataException(
                    $"DataLoader ended while {operation}.");
            }
            _position++;
            return batches.Current;
        }
    }
}
