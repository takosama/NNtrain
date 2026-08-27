namespace NNtrain.Training.Execution;

/// <summary>
/// Describes whether forward, backward, and gradient reduction are separate
/// operations or one task/data-parallel operation that completes all three.
/// </summary>
public enum TrainingGradientExecutionMode
{
    Separate = 0,
    FusedForwardBackwardReduced = 1,
}

/// <summary>
/// Reusable typed operations for a production training step. Implementations
/// may retain task state between calls, avoiding per-step delegate and closure
/// allocations on the hot path.
/// </summary>
public interface ITrainingStepOperations
{
    TrainingGradientExecutionMode GradientExecutionMode { get; }

    void AcquireBatch();

    void ClearGradients();

    void Forward();

    void Backward();

    void ReduceGradients();

    void ForwardBackwardReduced();

    void ClipGradients();

    void ApplySchedule();

    void CommitOptimizer();

    void CommitMetrics();
}

/// <summary>
/// Task-specific actions for one ordered training-step transaction. A phase
/// may be a no-op, but it remains explicit so optimizer and metric commits
/// cannot move ahead of reduction, clipping, or scheduling.
/// </summary>
public sealed record TrainingStepOperations(
    Action AcquireBatch,
    Action ClearGradients,
    Action Forward,
    Action Backward,
    Action ReduceGradients,
    Action ClipGradients,
    Action ApplySchedule,
    Action CommitOptimizer,
    Action CommitMetrics)
    : ITrainingStepOperations
{
    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(AcquireBatch);
        ArgumentNullException.ThrowIfNull(ClearGradients);
        ArgumentNullException.ThrowIfNull(Forward);
        ArgumentNullException.ThrowIfNull(Backward);
        ArgumentNullException.ThrowIfNull(ReduceGradients);
        ArgumentNullException.ThrowIfNull(ClipGradients);
        ArgumentNullException.ThrowIfNull(ApplySchedule);
        ArgumentNullException.ThrowIfNull(CommitOptimizer);
        ArgumentNullException.ThrowIfNull(CommitMetrics);
    }

    TrainingGradientExecutionMode
        ITrainingStepOperations.GradientExecutionMode
        => TrainingGradientExecutionMode.Separate;

    void ITrainingStepOperations.AcquireBatch() => AcquireBatch();

    void ITrainingStepOperations.ClearGradients() => ClearGradients();

    void ITrainingStepOperations.Forward() => Forward();

    void ITrainingStepOperations.Backward() => Backward();

    void ITrainingStepOperations.ReduceGradients() => ReduceGradients();

    void ITrainingStepOperations.ForwardBackwardReduced()
        => throw new InvalidOperationException(
            "Action-based training-step operations use separate gradient phases.");

    void ITrainingStepOperations.ClipGradients() => ClipGradients();

    void ITrainingStepOperations.ApplySchedule() => ApplySchedule();

    void ITrainingStepOperations.CommitOptimizer() => CommitOptimizer();

    void ITrainingStepOperations.CommitMetrics() => CommitMetrics();
}

/// <summary>
/// Executes action-based compatibility code or reusable typed operations
/// against the authoritative
/// <see cref="TrainingSession"/> and <see cref="TrainingStep"/> contracts.
/// </summary>
public sealed class TrainingStepExecutor
{
    private readonly TrainingSession _session;
    private TrainingStep? _activeStep;
    private TrainingStepState? _lastState;
    private int _executing;

    public TrainingStepExecutor(TrainingSession session)
        => _session = session
            ?? throw new ArgumentNullException(nameof(session));

    public TrainingSession Session => _session;

    /// <summary>
    /// Returns the active step snapshot, or the most recently terminal step
    /// when no step is active.
    /// </summary>
    public TrainingStepState? State
    {
        get
        {
            TrainingStep? active = Volatile.Read(ref _activeStep);
            return active?.State ?? Volatile.Read(ref _lastState);
        }
    }

    public TrainingStepState Execute(TrainingStepOperations operations)
        => Execute(checked(_session.LastCommittedStep + 1), operations);

    public TrainingStepState Execute(
        long globalStep,
        TrainingStepOperations operations)
    {
        ArgumentNullException.ThrowIfNull(operations);
        operations.Validate();
        return ExecuteCore(globalStep, operations);
    }

    public TrainingStepState Execute<TOperations>(TOperations operations)
        where TOperations : class, ITrainingStepOperations
        => Execute(
            checked(_session.LastCommittedStep + 1),
            operations);

    public TrainingStepState Execute<TOperations>(
        long globalStep,
        TOperations operations)
        where TOperations : class, ITrainingStepOperations
    {
        ArgumentNullException.ThrowIfNull(operations);
        if (!Enum.IsDefined(operations.GradientExecutionMode))
        {
            throw new ArgumentOutOfRangeException(
                nameof(operations),
                operations.GradientExecutionMode,
                "Unknown gradient execution mode.");
        }
        return ExecuteCore(globalStep, operations);
    }

    private TrainingStepState ExecuteCore<TOperations>(
        long globalStep,
        TOperations operations)
        where TOperations : class, ITrainingStepOperations
    {
        if (Interlocked.CompareExchange(ref _executing, 1, 0) != 0)
        {
            throw new InvalidOperationException(
                "A training step is already executing.");
        }

        TrainingStep? step = null;
        try
        {
            step = _session.BeginStep(globalStep);
            Volatile.Write(ref _activeStep, step);
            operations.AcquireBatch();
            step.Advance(TrainingStepPhase.BatchAcquired);
            operations.ClearGradients();
            step.Advance(TrainingStepPhase.GradientsCleared);
            if (operations.GradientExecutionMode
                == TrainingGradientExecutionMode.FusedForwardBackwardReduced)
            {
                operations.ForwardBackwardReduced();
                step.Advance(TrainingStepPhase.ForwardCompleted);
                step.Advance(TrainingStepPhase.BackwardCompleted);
                step.Advance(TrainingStepPhase.GradientsReduced);
            }
            else
            {
                operations.Forward();
                step.Advance(TrainingStepPhase.ForwardCompleted);
                operations.Backward();
                step.Advance(TrainingStepPhase.BackwardCompleted);
                operations.ReduceGradients();
                step.Advance(TrainingStepPhase.GradientsReduced);
            }
            operations.ClipGradients();
            step.Advance(TrainingStepPhase.GradientsClipped);
            operations.ApplySchedule();
            step.Advance(TrainingStepPhase.ScheduleApplied);
            operations.CommitOptimizer();
            step.Advance(TrainingStepPhase.OptimizerCommitted);
            operations.CommitMetrics();
            step.Advance(TrainingStepPhase.MetricsCommitted);
            TrainingStepState committed = step.State;
            Volatile.Write(ref _lastState, committed);
            return committed;
        }
        catch (Exception failure)
        {
            if (step is not null)
            {
                step.Fault(failure);
                Volatile.Write(ref _lastState, step.State);
            }
            throw;
        }
        finally
        {
            step?.Dispose();
            Volatile.Write(ref _activeStep, null);
            Volatile.Write(ref _executing, 0);
        }
    }
}
