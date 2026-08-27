using NNtrain.Training.Persistence;
using Xunit;

public sealed class CheckpointRepositoryTests
{
    [Theory]
    [InlineData(CheckpointFaultPoint.BeforeArtifactStage)]
    [InlineData(CheckpointFaultPoint.AfterArtifactStage)]
    [InlineData(CheckpointFaultPoint.AfterArtifactPublish)]
    [InlineData(CheckpointFaultPoint.BeforeManifestStage)]
    [InlineData(CheckpointFaultPoint.AfterManifestStage)]
    [InlineData(CheckpointFaultPoint.BeforeManifestPublish)]
    public void PreCommitFailureRestoresPreviousArtifactsAndManifest(
        CheckpointFaultPoint faultPoint)
    {
        using var directory = new TemporaryDirectory();
        string manifestPath = Path.Combine(directory.Root, "checkpoint.json");
        string artifactPath = Path.Combine(directory.Root, "weights.bin");
        string newArtifactPath = Path.Combine(directory.Root, "optimizer.bin");
        WriteGeneration(
            new CheckpointRepository(manifestPath),
            artifactPath,
            "old artifact",
            "old manifest");
        var failing = new CheckpointRepository(
            manifestPath,
            new ThrowAtFaultPoint(faultPoint));

        Assert.Throws<InjectedCheckpointFailure>(() =>
        {
            using CheckpointWriteTransaction transaction = failing.BeginWrite();
            transaction.PublishArtifact(
                artifactPath,
                path => File.WriteAllText(path, "new artifact"));
            transaction.PublishArtifact(
                newArtifactPath,
                path => File.WriteAllText(path, "new optimizer"));
            transaction.CommitManifest(
                path => File.WriteAllText(path, "new manifest"));
        });

        Assert.Equal("old manifest", File.ReadAllText(manifestPath));
        Assert.Equal("old artifact", File.ReadAllText(artifactPath));
        Assert.False(File.Exists(newArtifactPath));
        Assert.Empty(Directory.GetFiles(directory.Root, "*.backup"));
        Assert.Empty(Directory.GetFiles(directory.Root, "*.tmp"));
    }

    [Fact]
    public void AfterManifestPublishFailureKeepsNewGenerationActive()
    {
        using var directory = new TemporaryDirectory();
        string manifestPath = Path.Combine(directory.Root, "checkpoint.json");
        string artifactPath = Path.Combine(directory.Root, "weights.bin");
        WriteGeneration(
            new CheckpointRepository(manifestPath),
            artifactPath,
            "old artifact",
            "old manifest");
        var repository = new CheckpointRepository(
            manifestPath,
            new ThrowAtFaultPoint(
                CheckpointFaultPoint.AfterManifestPublish));

        Assert.Throws<InjectedCheckpointFailure>(() => WriteGeneration(
            repository,
            artifactPath,
            "new artifact",
            "new manifest"));

        Assert.Equal("new manifest", File.ReadAllText(manifestPath));
        Assert.Equal("new artifact", File.ReadAllText(artifactPath));
        Assert.Empty(Directory.GetFiles(directory.Root, "*.backup"));
    }

    [Fact]
    public void VersionRegistrySelectsRegisteredReaderAndWriter()
    {
        using var directory = new TemporaryDirectory();
        string manifestPath = Path.Combine(directory.Root, "checkpoint.txt");
        var registry = new CheckpointVersionRegistry<string>(
            path => int.Parse(File.ReadAllText(path).Split(':')[0]));
        registry.RegisterReader(
            new DelegateCheckpointVersionReader<string>(
                1,
                File.ReadAllText));
        registry.RegisterWriter(
            new DelegateCheckpointVersionWriter<string>(
                1,
                (transaction, value) => transaction.CommitManifest(
                    path => File.WriteAllText(path, value))));

        registry.Write(
            new CheckpointRepository(manifestPath),
            1,
            "1:payload");

        Assert.Equal("1:payload", registry.Read(manifestPath));
        Assert.False(registry.CanRead(2));
        Assert.Throws<NotSupportedException>(
            () => registry.Read(manifestPath, 2));
    }

    [Fact]
    public void DeferredArtifactRequestWritesAtTransactionStagingPath()
    {
        using var directory = new TemporaryDirectory();
        string manifestPath = Path.Combine(directory.Root, "checkpoint.json");
        string artifactPath = Path.Combine(directory.Root, "weights.bin");
        string? observedStagingPath = null;
        var repository = new CheckpointRepository(manifestPath);

        using (CheckpointWriteTransaction transaction = repository.BeginWrite())
        {
            transaction.PublishArtifact(
                new CheckpointArtifactWriteRequest(
                    artifactPath,
                    stagingPath =>
                    {
                        observedStagingPath = stagingPath;
                        File.WriteAllText(stagingPath, "streamed");
                    }));
            transaction.CommitManifest(
                stagingPath => File.WriteAllText(stagingPath, "manifest"));
        }

        Assert.NotNull(observedStagingPath);
        Assert.NotEqual(artifactPath, observedStagingPath);
        Assert.False(File.Exists(observedStagingPath));
        Assert.Equal("streamed", File.ReadAllText(artifactPath));
        Assert.Equal("manifest", File.ReadAllText(manifestPath));
    }

    private static void WriteGeneration(
        CheckpointRepository repository,
        string artifactPath,
        string artifact,
        string manifest)
    {
        using CheckpointWriteTransaction transaction = repository.BeginWrite();
        transaction.PublishArtifact(
            artifactPath,
            path => File.WriteAllText(path, artifact));
        transaction.CommitManifest(
            path => File.WriteAllText(path, manifest));
    }

    private sealed class ThrowAtFaultPoint(CheckpointFaultPoint point)
        : ICheckpointFaultInjector
    {
        public void OnCheckpointFaultPoint(CheckpointFaultContext context)
        {
            if (context.Point == point)
                throw new InjectedCheckpointFailure(context.Point);
        }
    }

    private sealed class InjectedCheckpointFailure(CheckpointFaultPoint point)
        : Exception(point.ToString());

    private sealed class TemporaryDirectory : IDisposable
    {
        internal TemporaryDirectory()
        {
            Root = Path.Combine(
                Path.GetTempPath(),
                $"NNtrain.CheckpointRepositoryTests-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Root);
        }

        internal string Root { get; }

        public void Dispose() => Directory.Delete(Root, recursive: true);
    }
}
