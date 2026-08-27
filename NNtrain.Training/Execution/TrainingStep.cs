namespace NNtrain.Training.Execution;

/// <summary>The ordered commit phases of one training step.</summary>
public enum TrainingStepPhase
{
    Initialized = 0,
    BatchAcquired = 1,
    GradientsCleared = 2,
    ForwardCompleted = 3,
    BackwardCompleted = 4,
    GradientsReduced = 5,
    GradientsClipped = 6,
    ScheduleApplied = 7,
    OptimizerCommitted = 8,
    MetricsCommitted = 9,
    Faulted = 10,
}

/// <summary>Immutable diagnostic snapshot of a training step.</summary>
public sealed record TrainingStepState(
    long GlobalStep,
    TrainingStepPhase Phase,
    DateTimeOffset StartedAt,
    DateTimeOffset UpdatedAt,
    Exception? Failure)
{
    public bool IsTerminal
        => Phase is TrainingStepPhase.MetricsCommitted
            or TrainingStepPhase.Faulted;

    public bool CanPublishCheckpoint
        => Phase == TrainingStepPhase.MetricsCommitted;
}

/// <summary>
/// Enforces the batch-to-metrics commit order. Reduction, clipping and
/// scheduling may be no-ops for a task, but their phase must still be
/// acknowledged before an optimizer commit can become visible.
/// </summary>
public sealed class TrainingStep : IDisposable
{
    private readonly object _sync = new();
    private readonly TimeProvider _timeProvider;
    private readonly Action<TrainingStep> _onTerminal;
    private readonly DateTimeOffset _startedAt;
    private TrainingStepPhase _phase = TrainingStepPhase.Initialized;
    private DateTimeOffset _updatedAt;
    private Exception? _failure;
    private int _terminalNotificationSent;

    internal TrainingStep(
        long globalStep,
        TimeProvider timeProvider,
        Action<TrainingStep> onTerminal)
    {
        if (globalStep < 0)
            throw new ArgumentOutOfRangeException(nameof(globalStep));
        GlobalStep = globalStep;
        _timeProvider = timeProvider;
        _onTerminal = onTerminal;
        _startedAt = timeProvider.GetUtcNow();
        _updatedAt = _startedAt;
    }

    public long GlobalStep { get; }

    public TrainingStepState State
    {
        get
        {
            lock (_sync)
            {
                return new TrainingStepState(
                    GlobalStep,
                    _phase,
                    _startedAt,
                    _updatedAt,
                    _failure);
            }
        }
    }

    public bool CanPublishCheckpoint
    {
        get
        {
            lock (_sync)
                return _phase == TrainingStepPhase.MetricsCommitted;
        }
    }

    public void Advance(TrainingStepPhase phase)
    {
        bool becameTerminal;
        lock (_sync)
        {
            if (_phase is TrainingStepPhase.MetricsCommitted
                or TrainingStepPhase.Faulted)
            {
                throw new InvalidOperationException(
                    $"Training step {GlobalStep} is already terminal ({_phase}).");
            }
            TrainingStepPhase expected = (TrainingStepPhase)((int)_phase + 1);
            if (phase != expected || phase == TrainingStepPhase.Faulted)
            {
                throw new InvalidOperationException(
                    $"Training step {GlobalStep} expected phase {expected}, not {phase}.");
            }
            _phase = phase;
            _updatedAt = _timeProvider.GetUtcNow();
            becameTerminal = phase == TrainingStepPhase.MetricsCommitted;
        }
        if (becameTerminal)
            NotifyTerminalOnce();
    }

    public void Fault(Exception failure)
    {
        ArgumentNullException.ThrowIfNull(failure);
        lock (_sync)
        {
            if (_phase is TrainingStepPhase.MetricsCommitted
                or TrainingStepPhase.Faulted)
            {
                return;
            }
            _phase = TrainingStepPhase.Faulted;
            _failure = failure;
            _updatedAt = _timeProvider.GetUtcNow();
        }
        NotifyTerminalOnce();
    }

    public void Dispose()
        => Fault(new OperationCanceledException(
            $"Training step {GlobalStep} was disposed before metrics commit."));

    private void NotifyTerminalOnce()
    {
        if (Interlocked.Exchange(ref _terminalNotificationSent, 1) == 0)
            _onTerminal(this);
    }
}
