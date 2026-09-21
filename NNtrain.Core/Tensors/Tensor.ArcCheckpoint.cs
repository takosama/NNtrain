using System.Runtime.ExceptionServices;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    internal static bool IsArcCheckpointActive => ArcCheckpointFrame.Current is not null;
    internal static bool IsArcCheckpointReplay => ArcCheckpointRandomScope.Current?.IsReplaying == true;

    /// <summary>
    /// Retains only the packed input/output of an Arc transformer block. Its
    /// local graph is reconstructed during backward with the original masks.
    /// </summary>
    internal Tensor ArcCheckpoint(Func<Tensor, Tensor> forward, IReadOnlyList<Tensor> parameters)
    {
        ArgumentNullException.ThrowIfNull(forward);
        ArgumentNullException.ThrowIfNull(parameters);
        if (!ArcResident)
            return forward(this);
        // Nested checkpoints require a nested RNG transcript and ownership of
        // the inner checkpoint result. Reject them explicitly rather than
        // silently consuming different masks or leaking a partial inner graph.
        if (ArcCheckpointFrame.Current is not null)
            throw new InvalidOperationException("Nested Arc checkpoints are not supported.");
        if (!AutogradContext.IsRecordingEnabled) return forward(this);

        Tensor? result = null;
        try
        {
            return ArcCheckpointFrame.Run(() =>
            {
                using var random = new ArcCheckpointRandomScope();
                Tensor temporary;
                using (AutogradContext.NoGrad()) temporary = forward(this);
                if (ReferenceEquals(temporary, this))
                    throw new InvalidOperationException("Arc checkpoint blocks must produce a new tensor.");

                // Parameters are graph parents even though their individual
                // forward operations were discarded. This preserves the normal
                // mutation/version checks before any recomputation is launched.
                var parents = new Tensor[parameters.Count + 1];
                parents[0] = this;
                for (int index = 0; index < parameters.Count; index++) parents[index + 1] = parameters[index];
                result = FromStorageResult(temporary.ArcCheckpointPlaceholder(), temporary._shape, parents);
                temporary.MoveArcCheckpointValues(result);
                var context = new ArcCheckpointContext(this, result, forward, random.RecordedSeeds());
                var lease = AutogradLease<ArcCheckpointContext>.Own(context,
                    new AutogradLeaseMetadata(TensorDevice.Arc, ArcLane.DeviceIndex, result.DType,
                        generation: 0, AutogradLeaseOwnership.Owned), static owned => owned.Dispose());
                result.Node.SetBackward(lease, static saved => saved.Backward());
                return result;
            });
        }
        catch
        {
            if (result is not null)
            {
                try { result.ReleaseArcGraph(); }
                finally { result.Node.ReleaseGraph(); }
            }
            throw;
        }
    }

    private TensorStorage ArcCheckpointPlaceholder() => DType == TensorDType.Bfp8
        ? TensorStorage.CreateDeviceBfp8Placeholder(Numel, Bfp8Quantization!)
        : TensorStorage.CreateDevicePlaceholder(Numel, DType);

    private void MoveArcCheckpointValues(Tensor target)
    {
        if (ArcCheckpointFrame.Current?.Owns(this) != true)
            throw new InvalidOperationException("Arc checkpoint outputs must be owned by the current checkpoint frame.");
        EnsureArcPacked();
        ArcReplica source = _arcReplica!, destination = target.ArcOwner();
        // ArcDeviceResult publishes independently owned buffers. Transfer those
        // handles instead of allocating/copying the final packed activation.
        // All potentially throwing preparation is above this point. Neither
        // the old tensor's cleanup nor its finalizer may retain these handles.
        destination.Value = source.Value;
        destination.Scales = source.Scales;
        destination.DataDirty = true;
        target._device = TensorDevice.Arc;
        target._arcDeviceIndex = destination.Lane.DeviceIndex;
        _arcValuesReleased |= source.DataDirty;
        source.Value = source.Scales = null;
        source.DataDirty = false;
    }

    private Tensor BorrowArcCheckpointInput()
    {
        EnsureArcPacked();
        Tensor detached = FromStorageResult(ArcCheckpointPlaceholder(), _shape, []);
        try
        {
            ArcReplica source = _arcReplica!, destination = detached.ArcOwner();
            destination.Value = source.Value!.Borrow();
            destination.Scales = source.Scales?.Borrow();
            // Accumulate directly into the original input gradient, in the
            // original operation order. Computing a separate sum and adding it
            // afterwards would introduce a new rounding boundary.
            destination.Gradient = ArcGradient().Borrow();
            destination.DataDirty = destination.GradientDirty = true;
            detached._device = TensorDevice.Arc;
            detached._arcDeviceIndex = source.Lane.DeviceIndex;
            return detached;
        }
        catch { detached.ReleaseArcReplica(preserve: false); throw; }
    }

    internal void AccumulateArcCheckpointSeed(ArcBuffer seed)
    {
        // A recomputed block's output is a fresh non-leaf and therefore has
        // zero gradient after ClearIntermediateGradients. Copy its FP32 seed
        // bit-for-bit; CopyBytes validates the lane and source/target ranges.
        if (Node.IsLeaf)
            throw new InvalidOperationException("An Arc checkpoint must recompute a differentiable output.");
        ArcLane.CopyBytes(seed, ArcGradient(), 0, 0, checked(Numel * sizeof(float)));
    }

    private sealed class ArcCheckpointContext(
        Tensor input, Tensor output, Func<Tensor, Tensor> forward, uint[] seeds) : IDisposable
    {
        private Tensor? _input = input, _output = output;
        private Func<Tensor, Tensor>? _forward = forward;
        private uint[] _seeds = seeds;

        internal void Backward()
        {
            using var recording = AutogradContext.EnableRecording();
            Tensor localInput = _input!.BorrowArcCheckpointInput();
            try
            {
                ArcCheckpointFrame.Run(() =>
                {
                    using var random = new ArcCheckpointRandomScope(_seeds);
                    Tensor localOutput = _forward!(localInput);
                    random.ValidateReplayConsumed();
                    AutogradEngine.BackwardArcSeed(localOutput, _output!.ArcGradient());
                    return true;
                });
            }
            finally { localInput.ReleaseArcReplica(preserve: false); }
        }

        public void Dispose()
        {
            _input = _output = null;
            _forward = null;
            _seeds = [];
        }
    }

    // A failed recompute may not have produced a final graph root yet. Track
    // every newly published Arc value, so both partial and complete graphs are
    // cleaned up without touching the borrowed input or parameter leaves.
    private sealed class ArcCheckpointFrame
    {
        [ThreadStatic] internal static ArcCheckpointFrame? Current;
        private readonly ArcCheckpointFrame? _previous = Current;
        private readonly List<Tensor> _values = [];
        private ArcCheckpointFrame() => Current = this;
        internal void Add(Tensor value) => _values.Add(value);
        internal bool Owns(Tensor value) => _values.Contains(value);

        internal static T Run<T>(Func<T> action)
        {
            var frame = new ArcCheckpointFrame();
            T result = default!;
            Exception? failure = null;
            try { result = action(); }
            catch (Exception exception) { failure = exception; }
            var cleanupFailures = new List<Exception>();
            try
            {
                foreach (Tensor value in frame._values)
                {
                    try { value.ReleaseArcGraph(); }
                    catch (Exception exception) { cleanupFailures.Add(exception); }
                    try { value.Node.ReleaseGraph(); }
                    catch (Exception exception) { cleanupFailures.Add(exception); }
                }
            }
            finally { frame._values.Clear(); Current = frame._previous; }
            if (cleanupFailures.Count > 0)
            {
                if (failure is not null) cleanupFailures.Insert(0, failure);
                throw new AggregateException("Arc checkpoint execution/cleanup failed.", cleanupFailures);
            }
            if (failure is not null) ExceptionDispatchInfo.Capture(failure).Throw();
            return result;
        }
    }

    private sealed class ArcCheckpointRandomScope : IDisposable
    {
        [ThreadStatic] internal static ArcCheckpointRandomScope? Current;
        private readonly ArcCheckpointRandomScope? _previous = Current;
        private readonly List<uint> _recorded = [];
        private readonly uint[]? _replay;
        private int _position;
        private bool _disposed;
        internal bool IsReplaying => _replay is not null;
        internal ArcCheckpointRandomScope(uint[]? replay = null) { _replay = replay; Current = this; }
        internal bool TryReplay(out uint seed)
        {
            seed = 0;
            if (_replay is null) return false;
            if (_position >= _replay.Length)
                throw new InvalidOperationException("Arc checkpoint dropout sequence changed before backward.");
            seed = _replay[_position++];
            return true;
        }
        internal void Record(uint seed) => _recorded.Add(seed);
        internal uint[] RecordedSeeds() => _recorded.ToArray();
        internal void ValidateReplayConsumed()
        {
            if (_replay is not null && _position != _replay.Length)
                throw new InvalidOperationException("Arc checkpoint dropout sequence changed before backward.");
        }
        public void Dispose() { if (_disposed) return; _disposed = true; Current = _previous; }
    }
}
