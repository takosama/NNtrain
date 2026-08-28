namespace NNtrain.Training.Execution;

/// <summary>
/// Owns the deterministic position and acquisition policy for a training
/// data stream. The cursor, rather than the model task, is the only component
/// that advances the underlying dataset iterator or token buffer.
/// </summary>
/// <typeparam name="TBatch">The reusable task-facing batch description.</typeparam>
public interface ITrainingDataCursor<out TBatch>
{
    /// <summary>
    /// Gets the number of source acquisition units consumed in the current
    /// cursor epoch. A cursor may define an acquisition unit as a batch or a
    /// microbatch, but it must keep that definition stable across resume.
    /// </summary>
    long Position { get; }

    /// <summary>Acquires the next task-facing batch and advances the cursor.</summary>
    TBatch AcquireNext();
}

/// <summary>
/// Performs task/model work for a batch supplied by an independent data
/// cursor. Implementations retain their mutable step state and are reusable;
/// no per-step delegates or closures are required on the hot path.
/// </summary>
/// <typeparam name="TBatch">The task-facing batch description.</typeparam>
public interface ITrainingTaskAdapter<in TBatch>
{
    TrainingGradientExecutionMode GradientExecutionMode { get; }

    /// <summary>
    /// Prepares reusable model and optimizer resources before guarded training
    /// begins. It is invoked once per composed operations object.
    /// </summary>
    void Prepare()
    {
    }

    /// <summary>Accepts the batch acquired for the current transaction.</summary>
    void AcceptBatch(TBatch batch);

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
/// Allocation-free bridge from a data cursor and task adapter to the common
/// ordered <see cref="TrainingStepExecutor"/> contract.
/// </summary>
public sealed class CursorTrainingStepOperations<TBatch>
    : ITrainingStepOperations
{
    private readonly ITrainingDataCursor<TBatch> _cursor;
    private readonly ITrainingTaskAdapter<TBatch> _adapter;

    public CursorTrainingStepOperations(
        ITrainingDataCursor<TBatch> cursor,
        ITrainingTaskAdapter<TBatch> adapter)
    {
        _cursor = cursor
            ?? throw new ArgumentNullException(nameof(cursor));
        _adapter = adapter
            ?? throw new ArgumentNullException(nameof(adapter));
    }

    public ITrainingDataCursor<TBatch> Cursor => _cursor;

    public ITrainingTaskAdapter<TBatch> Adapter => _adapter;

    public TrainingGradientExecutionMode GradientExecutionMode
        => _adapter.GradientExecutionMode;

    public void Prepare() => _adapter.Prepare();

    public void AcquireBatch()
        => _adapter.AcceptBatch(_cursor.AcquireNext());

    public void ClearGradients() => _adapter.ClearGradients();

    public void Forward() => _adapter.Forward();

    public void Backward() => _adapter.Backward();

    public void ReduceGradients() => _adapter.ReduceGradients();

    public void ForwardBackwardReduced()
        => _adapter.ForwardBackwardReduced();

    public void ClipGradients() => _adapter.ClipGradients();

    public void ApplySchedule() => _adapter.ApplySchedule();

    public void CommitOptimizer() => _adapter.CommitOptimizer();

    public void CommitMetrics() => _adapter.CommitMetrics();
}
