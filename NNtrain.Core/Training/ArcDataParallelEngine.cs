using NNtrain.Runtime.Execution;

namespace NNtrain;

internal readonly record struct ArcLanguageModelMicroBatch(
    int[] Input,
    int[] Target,
    int BatchSize,
    int SequenceLength);

/// <summary>
/// Runs independent model replicas on two Arc lanes. The primary model owns
/// optimizer and checkpoint state; gradients are reduced to that model
/// before clipping, and its updated weights are copied to the second lane.
/// </summary>
internal sealed class ArcDataParallelEngine : IDisposable
{
    private readonly LanguageModel _primary;
    private readonly LanguageModel _secondary;
    private readonly Parameter[] _primaryParameters;
    private readonly Parameter[] _secondaryParameters;
    private readonly Array?[] _forwardPayloads;
    private readonly float[]?[] _forwardScales;
    private readonly int _primaryDevice;
    private readonly int _secondaryDevice;
    private int[] _lastShardBatchSizes = [0, 0];
    private bool _disposed;

    internal ArcDataParallelEngine(
        LanguageModel primary,
        LanguageModel secondary,
        IReadOnlyList<int> deviceIndices)
    {
        ArgumentNullException.ThrowIfNull(primary);
        ArgumentNullException.ThrowIfNull(secondary);
        ArgumentNullException.ThrowIfNull(deviceIndices);
        if (ReferenceEquals(primary, secondary) || deviceIndices.Count != 2
            || deviceIndices[0] == deviceIndices[1])
            throw new ArgumentException("Arc data parallelism requires two model replicas and two distinct GPUs.");
        ExecutionSession session = ExecutionSession.Current
            ?? throw new InvalidOperationException("Arc data parallelism requires an active execution session.");
        if (session.Options.Device != ExecutionDeviceKind.Arc
            || !session.TryGetLane(ExecutionDeviceKind.Arc, deviceIndices[0], out _)
            || !session.TryGetLane(ExecutionDeviceKind.Arc, deviceIndices[1], out _))
            throw new InvalidOperationException("Both Arc lanes must be attached before data parallel training.");

        _primary = primary;
        _secondary = secondary;
        _primaryDevice = deviceIndices[0];
        _secondaryDevice = deviceIndices[1];
        _primaryParameters = primary.parameters().ToArray();
        _secondaryParameters = secondary.parameters().ToArray();
        _forwardPayloads = new Array?[_primaryParameters.Length];
        _forwardScales = new float[]?[_primaryParameters.Length];
        if (_primaryParameters.Length != _secondaryParameters.Length
            || _primaryParameters.Where((parameter, index) =>
                parameter.Name != _secondaryParameters[index].Name
                || parameter.T.DType != _secondaryParameters[index].T.DType
                || !parameter.T.Shape.SequenceEqual(_secondaryParameters[index].T.Shape)).Any())
            throw new ArgumentException("Arc model replicas must have identical parameters.");

        SynchronizeSecondaryModel();
    }

    internal IReadOnlyList<int> LastShardBatchSizes => _lastShardBatchSizes;

    internal float ForwardBackwardAccumulated(
        IReadOnlyList<ArcLanguageModelMicroBatch> microBatches,
        int ignoreIndex,
        long globalStep)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(microBatches);
        if (microBatches.Count == 0 || globalStep < 0)
            throw new ArgumentException("Arc data parallelism requires a nonempty update and a valid step.");

        var first = new List<Shard>(microBatches.Count);
        var second = new List<Shard>(microBatches.Count);
        int totalValidTargets = 0;
        foreach (ArcLanguageModelMicroBatch microBatch in microBatches)
        {
            if (microBatch.BatchSize < 1 || microBatch.SequenceLength < 1
                || microBatch.Input.Length != checked(microBatch.BatchSize * microBatch.SequenceLength)
                || microBatch.Target.Length != microBatch.Input.Length)
                throw new ArgumentException("Arc microbatch shape does not match its token arrays.");
            int primaryRows = (microBatch.BatchSize + 1) / 2;
            int secondaryRows = microBatch.BatchSize - primaryRows;
            first.Add(Slice(microBatch, 0, primaryRows, ignoreIndex));
            second.Add(Slice(microBatch, primaryRows, secondaryRows, ignoreIndex));
            totalValidTargets = checked(totalValidTargets
                + first[^1].ValidTargets + second[^1].ValidTargets);
            _lastShardBatchSizes = [primaryRows, secondaryRows];
        }
        if (totalValidTargets == 0)
            throw new InvalidOperationException("The Arc update has no valid targets.");

        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, _secondaryDevice)))
        {
            foreach (Parameter parameter in _secondaryParameters)
                parameter.ZeroGrad();
            _secondary.train();
            ResetSecondaryRandom(globalStep);
        }

        Task<double> secondaryTask = Task.Run(() => TrainLocal(
            _secondary, _secondaryDevice, second, totalValidTargets));
        double firstLoss = 0d, secondLoss = 0d;
        Exception? firstFailure = null, secondFailure = null;
        try { firstLoss = TrainLocal(_primary, _primaryDevice, first, totalValidTargets); }
        catch (Exception exception) { firstFailure = exception; }
        try { secondLoss = secondaryTask.GetAwaiter().GetResult(); }
        catch (Exception exception) { secondFailure = exception; }
        if (firstFailure is not null && secondFailure is not null)
            throw new AggregateException("Both Arc workers failed.", firstFailure, secondFailure);
        if (firstFailure is not null)
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(firstFailure).Throw();
        if (secondFailure is not null)
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(secondFailure).Throw();

        if (second.Any(shard => shard.ValidTargets > 0))
        {
            bool packedBFloat16 = TensorExecutionContext.ActivePrecisionPolicy?.Mode == PrecisionMode.Mix8_16;
            if (packedBFloat16 && Tensor.ArcLane.Options.Mix8_16PipelinedGradientReduction)
                ReducePackedGradientsPipelined();
            else
            {
                for (int index = 0; index < _primaryParameters.Length; index++)
                {
                    Tensor gradient = _secondaryParameters[index].T;
                    if (!gradient.HasGradientBuffer) continue;
                    if (packedBFloat16)
                    {
                        ushort[] packed;
                        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, _secondaryDevice)))
                            packed = gradient.CaptureArcBFloat16Gradient();
                        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, _primaryDevice)))
                            _primaryParameters[index].T.AccumulateArcBFloat16Gradient(packed);
                    }
                    else
                        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, _primaryDevice)))
                            _primaryParameters[index].T.AccumulateArcGradient(gradient.GradientBuffer);
                }
            }
        }

        return (float)((firstLoss + secondLoss) / totalValidTargets);
    }

    /// <summary>
    /// Read the next BF16 gradient on GPU 1 while the previous gradient is
    /// uploaded and combined on GPU 0. Only one read is prefetched, keeping
    /// host staging bounded to two parameter gradients.
    /// </summary>
    private void ReducePackedGradientsPipelined()
    {
        Task<(int Index, ushort[] Values)>? pending = null;
        try
        {
            for (int index = 0; index < _secondaryParameters.Length; index++)
            {
                if (!_secondaryParameters[index].T.HasGradientBuffer) continue;
                int nextIndex = index;
                Task<(int Index, ushort[] Values)> next = Task.Run(() =>
                {
                    using var secondary = TensorExecutionContext.Push(
                        new TorchDevice(TensorDevice.Arc, _secondaryDevice));
                    return (nextIndex, _secondaryParameters[nextIndex].T.CaptureArcBFloat16Gradient());
                });
                Task<(int Index, ushort[] Values)>? previous = pending;
                pending = next;
                if (previous is not null) Consume(previous.GetAwaiter().GetResult());
            }
            if (pending is not null)
            {
                Task<(int Index, ushort[] Values)> last = pending;
                pending = null;
                Consume(last.GetAwaiter().GetResult());
            }
        }
        finally
        {
            // A primary-lane failure must not leave an in-flight read using a
            // lane that the execution session is about to dispose.
            if (pending is not null)
                try { pending.GetAwaiter().GetResult(); }
                catch { /* Preserve the failure from the primary reduction. */ }
        }

        void Consume((int Index, ushort[] Values) gradient)
        {
            using var primary = TensorExecutionContext.Push(
                new TorchDevice(TensorDevice.Arc, _primaryDevice));
            _primaryParameters[gradient.Index].T.AccumulateArcBFloat16Gradient(gradient.Values);
        }
    }

    internal void SynchronizeSecondaryModel()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ModuleState state;
        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, _primaryDevice)))
            state = _primary.state_dict();
        using (TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, _secondaryDevice)))
            _secondary.load_state_dict(state);
    }

    /// <summary>
    /// Transfers the exact packed forward weights after an optimizer update.
    /// The secondary model only computes forward/backward, so its master
    /// and optimizer state do not need to be synchronized each update.
    /// </summary>
    internal void SynchronizeSecondaryForwardReplica()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        for (int index = 0; index < _primaryParameters.Length; index++)
            _secondaryParameters[index].T.CopyArcForwardReplicaFrom(
                _primaryParameters[index].T, _primaryDevice, _secondaryDevice,
                ref _forwardPayloads[index], ref _forwardScales[index]);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Array.Clear(_forwardPayloads);
        Array.Clear(_forwardScales);
    }

    private void ResetSecondaryRandom(long globalStep)
    {
        if (!_secondary.HasCheckpointableTrainingRandom)
            return;
        ulong root = _primary.TrainingRandomRootSeed;
        ulong seed = Mix(root ^ unchecked((ulong)globalStep * 0x9E3779B97F4A7C15UL)
            ^ unchecked((ulong)(uint)_secondaryDevice * 0xD1B54A32D192ED03UL));
        _secondary.RestoreTrainingRandomState(new TrainingRandomState(
            TrainingRandomState.CurrentFormatVersion, root, seed, []));
    }

    private static ulong Mix(ulong value)
    {
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9UL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBUL;
        return value ^ (value >> 31);
    }

    private static double TrainLocal(
        LanguageModel model,
        int deviceIndex,
        IReadOnlyList<Shard> shards,
        int totalValidTargets)
    {
        using var device = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, deviceIndex));
        double weightedLoss = 0d;
        foreach (Shard shard in shards)
        {
            if (shard.ValidTargets == 0) continue;
            Tensor loss = model.forward_loss(shard.Input, shard.Target,
                shard.BatchSize, shard.SequenceLength);
            float value = loss.item();
            loss.BackwardAndRelease([(float)shard.ValidTargets / totalValidTargets]);
            weightedLoss += value * shard.ValidTargets;
        }
        return weightedLoss;
    }

    private static Shard Slice(
        ArcLanguageModelMicroBatch microBatch,
        int firstRow,
        int rowCount,
        int ignoreIndex)
    {
        if (rowCount == 0) return new([], [], 0, microBatch.SequenceLength, 0);
        int count = checked(rowCount * microBatch.SequenceLength);
        int offset = checked(firstRow * microBatch.SequenceLength);
        var input = new int[count];
        var target = new int[count];
        Array.Copy(microBatch.Input, offset, input, 0, count);
        Array.Copy(microBatch.Target, offset, target, 0, count);
        int valid = target.Count(token => token != ignoreIndex);
        return new(input, target, rowCount, microBatch.SequenceLength, valid);
    }

    private readonly record struct Shard(
        int[] Input,
        int[] Target,
        int BatchSize,
        int SequenceLength,
        int ValidTargets);
}
