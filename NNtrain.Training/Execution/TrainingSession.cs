using NNtrain.Runtime.Execution;

namespace NNtrain.Training.Execution;

/// <summary>
/// Owns the step transaction boundary around an execution session. Only one
/// step may be in flight, and checkpoints become publishable only after a
/// metrics commit.
/// </summary>
public sealed class TrainingSession : IDisposable
{
    private readonly object _sync = new();
    private readonly bool _ownsExecutionSession;
    private readonly TimeProvider _timeProvider;
    private readonly List<IDisposable> _ownedResources = [];
    private TrainingStep? _activeStep;
    private Exception? _failure;
    private long _lastCommittedStep;
    private int _disposed;

    public TrainingSession(
        ExecutionSession executionSession,
        bool ownsExecutionSession = false,
        long lastCommittedStep = -1,
        TimeProvider? timeProvider = null)
    {
        if (lastCommittedStep < -1)
            throw new ArgumentOutOfRangeException(nameof(lastCommittedStep));
        ExecutionSession = executionSession
            ?? throw new ArgumentNullException(nameof(executionSession));
        _ownsExecutionSession = ownsExecutionSession;
        _lastCommittedStep = lastCommittedStep;
        _timeProvider = timeProvider ?? TimeProvider.System;
    }

    public ExecutionSession ExecutionSession { get; }

    public long LastCommittedStep
    {
        get
        {
            lock (_sync)
                return _lastCommittedStep;
        }
    }

    public bool CanPublishCheckpoint
    {
        get
        {
            lock (_sync)
                return _failure is null
                    && _activeStep is null
                    && _lastCommittedStep >= 0;
        }
    }

    public bool IsFaulted
    {
        get
        {
            lock (_sync)
                return _failure is not null;
        }
    }

    public Exception? Failure
    {
        get
        {
            lock (_sync)
                return _failure;
        }
    }

    /// <summary>
    /// Creates and owns the data-parallel engine for this training session.
    /// The engine is released even when the session faults mid-step.
    /// </summary>
    public CudaDataParallelEngine OwnCudaDataParallel(
        LanguageModel model,
        CudaAdaptiveShardingOptions? adaptiveShardingOptions = null)
        => OwnCudaDataParallel(
            model,
            Tensor.CudaDeviceIndices,
            adaptiveShardingOptions);

    /// <summary>
    /// Creates and owns an engine whose CUDA device set is fixed for the
    /// lifetime of this session rather than following ambient static state.
    /// </summary>
    public CudaDataParallelEngine OwnCudaDataParallel(
        LanguageModel model,
        IReadOnlyList<int> cudaDeviceIndices,
        CudaAdaptiveShardingOptions? adaptiveShardingOptions = null)
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
        var engine = new CudaDataParallelEngine(
            model,
            cudaDeviceIndices,
            adaptiveShardingOptions);
        try
        {
            lock (_sync)
            {
                ObjectDisposedException.ThrowIf(
                    Volatile.Read(ref _disposed) != 0,
                    this);
                _ownedResources.Add(engine);
            }
            return engine;
        }
        catch
        {
            engine.Dispose();
            throw;
        }
    }

    public TrainingStep BeginStep(long globalStep)
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _disposed) != 0,
                this);
            if (_failure is not null)
            {
                throw new InvalidOperationException(
                    "The training session is faulted and cannot be reused.",
                    _failure);
            }
            if (_activeStep is not null)
                throw new InvalidOperationException(
                    $"Training step {_activeStep.GlobalStep} is still active.");
            if (globalStep <= _lastCommittedStep)
                throw new ArgumentOutOfRangeException(
                    nameof(globalStep),
                    $"Step {globalStep} must be greater than the last committed step {_lastCommittedStep}.");
            _activeStep = new TrainingStep(
                globalStep,
                _timeProvider,
                OnStepTerminal);
            return _activeStep;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        TrainingStep? active;
        IDisposable[] resources;
        lock (_sync)
        {
            active = _activeStep;
            resources = _ownedResources.ToArray();
            _ownedResources.Clear();
        }
        active?.Fault(new OperationCanceledException(
            "The training session ended before its active step committed."));

        List<Exception>? failures = null;
        for (int index = resources.Length - 1; index >= 0; index--)
        {
            try
            {
                resources[index].Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        if (_ownsExecutionSession)
        {
            try
            {
                ExecutionSession.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        if (failures is not null)
        {
            throw new AggregateException(
                "One or more training-session resources failed to dispose.",
                failures);
        }
    }

    private void OnStepTerminal(TrainingStep step)
    {
        TrainingStepState state = step.State;
        lock (_sync)
        {
            if (!ReferenceEquals(_activeStep, step))
                return;
            if (state.Phase == TrainingStepPhase.MetricsCommitted)
                _lastCommittedStep = step.GlobalStep;
            else if (state.Phase == TrainingStepPhase.Faulted)
            {
                _failure = state.Failure
                    ?? new InvalidOperationException(
                        $"Training step {step.GlobalStep} faulted.");
            }
            _activeStep = null;
        }
    }
}
