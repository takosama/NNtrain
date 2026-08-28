namespace NNtrain;

internal enum CanonicalTrainingTaskKind
{
    ImageClassification,
    WikiLanguageModel,
}

/// <summary>
/// A validated, path-resolved training configuration at the boundary between
/// JSON compatibility handling and the training application.
/// </summary>
internal abstract record CanonicalTrainingSpec(
    string ConfigurationPath,
    CanonicalTrainingTaskKind TaskKind,
    int? SourceSchemaVersion)
{
    internal bool UsesLegacySchema => SourceSchemaVersion is null;

    internal abstract TensorPrecisionMode PrecisionMode { get; }

    internal abstract int Bfp8BlockSize { get; }
}

internal sealed record CanonicalClassificationTrainingSpec(
    string ConfigurationPath,
    int? SourceSchemaVersion,
    TrainingConfiguration Configuration)
    : CanonicalTrainingSpec(
        ConfigurationPath,
        CanonicalTrainingTaskKind.ImageClassification,
        SourceSchemaVersion)
{
    internal override TensorPrecisionMode PrecisionMode
        => Configuration.GetPrecisionMode();

    internal override int Bfp8BlockSize
        => Configuration.GetBfp8BlockSize();
}

internal sealed record CanonicalWikiTrainingSpec(
    string ConfigurationPath,
    int? SourceSchemaVersion,
    WikiTrainingConfiguration Configuration)
    : CanonicalTrainingSpec(
        ConfigurationPath,
        CanonicalTrainingTaskKind.WikiLanguageModel,
        SourceSchemaVersion)
{
    internal override TensorPrecisionMode PrecisionMode
        => Configuration.GetPrecisionMode();

    internal override int Bfp8BlockSize
        => Configuration.Bfp8BlockSize;
}
