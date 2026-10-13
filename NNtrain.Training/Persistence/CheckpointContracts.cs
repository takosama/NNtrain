namespace NNtrain.Training.Persistence;

public enum CheckpointFaultPoint
{
    BeforeArtifactStage = 0,
    AfterArtifactStage = 1,
    AfterArtifactPublish = 2,
    BeforeManifestStage = 3,
    AfterManifestStage = 4,
    BeforeManifestPublish = 5,
    AfterManifestPublish = 6,
}

public sealed record CheckpointFaultContext(
    CheckpointFaultPoint Point,
    string Path,
    int ArtifactIndex);

/// <summary>
/// Deferred artifact payload. The callback receives the transaction-owned
/// staging path, allowing a model to stream directly into its existing
/// artifact name without first constructing an in-memory payload.
/// </summary>
public sealed record CheckpointArtifactWriteRequest(
    string ArtifactPath,
    Action<string> WriteStagedArtifact,
    bool PreservePreviousOnRollback = true);

/// <summary>Test and diagnostics hook called at durable publication boundaries.</summary>
public interface ICheckpointFaultInjector
{
    void OnCheckpointFaultPoint(CheckpointFaultContext context);
}
