namespace NNtrain;

/// <summary>
/// Serializable PyTorch-style state required to continue a training run.
/// Application-specific checkpoints may add metrics and early-stopping state.
/// </summary>
public sealed record TrainingCheckpoint(
    int Epoch,
    ModuleState Model,
    OptimizerStateDictionary Optimizer,
    LRSchedulerStateDictionary Scheduler);
