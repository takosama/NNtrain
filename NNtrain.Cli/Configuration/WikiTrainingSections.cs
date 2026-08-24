namespace NNtrain;

/// <summary>
/// Groups checkpoint location and restart behavior in a training JSON file.
/// </summary>
sealed record WikiCheckpointConfiguration
{
    public string Directory { get; init; } = string.Empty;

    public string? FileName { get; init; }

    public bool Resume { get; init; }

    public bool AutoResume { get; init; }
}

/// <summary>
/// Groups optimizer and learning-rate scheduler settings in a training JSON
/// file.
/// </summary>
sealed record WikiOptimizationConfiguration
{
    public WikiOptimizerConfiguration? Optimizer { get; init; } = new();

    public WikiSchedulerConfiguration? Scheduler { get; init; } = new();
}

sealed record WikiOptimizerConfiguration
{
    public string Type { get; init; } =
        WikiTrainingConfiguration.NekoMuonOptimizer;

    public float LearningRate { get; init; } = 3e-4f;

    public float AuxiliaryLearningRate { get; init; } = 3e-4f;

    public float WeightDecay { get; init; } = 0.01f;

    public int NekoMuonNewtonSchulzInterval { get; init; } = 5;

    public int GainShareBlockDepth { get; init; } = 1;

    public float GainShareBeta1 { get; init; } = 0.9f;

    public float GainShareBeta2 { get; init; } = 0.999f;

    public float GainShareEpsilon { get; init; } = 1e-8f;

    public float GainShareRho { get; init; } = 0.95f;

    public float GainShareGamma { get; init; } = 1f;

    public float GainShareMinScale { get; init; } = 0.5f;

    public float GainShareMaxScale { get; init; } = 2f;

    public bool AdamWUseBFloat16FirstMoment { get; init; }

    public bool AdamWUseBFloat16SecondMoment { get; init; }
}

sealed record WikiSchedulerConfiguration
{
    public string Type { get; init; } =
        WikiTrainingConfiguration.WarmupCosineProgressScheduler;

    public float WarmupPercent { get; init; } = 20f;
}
