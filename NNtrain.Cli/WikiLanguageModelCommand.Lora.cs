using System.Security.Cryptography;

namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    internal static (ForgetMemoryDRNGpt Model, string Fingerprint) LoadLoraBase(
        string path, TensorPrecisionMode mode, int seed, int bfp8BlockSize = 32)
    {
        WikiModelCheckpoint metadata = LoadCheckpointMetadata(path);
        if (!IsCheckpointForgetMemoryDrn(metadata))
            throw new NotSupportedException("lora --model supports DRN checkpoint JSON only; other architectures are not supported.");
        // Construct in the requested training precision, then stream the saved
        // current master weights directly into it. Never restore base optimizer.
        var model = (ForgetMemoryDRNGpt)CreateModelStorage(metadata with { Dropout = 0 }, seed,
            mode == TensorPrecisionMode.Mix8_32 ? TensorDType.Float32 : mode.ToStorageDType());
        if (mode == TensorPrecisionMode.Mix8_32) model.to(mode, bfp8BlockSize);
        else model.SetPrecisionMode(mode);
        WikiModelCheckpoint loaded = LoadCheckpointForResume(path, model);
        if (loaded.FormatVersion < 7) model.load_state_dict(loaded.CurrentModel ?? loaded.Model);
        using var manifest = File.OpenRead(path);
        string hash = Convert.ToHexString(SHA256.HashData(manifest));
        if (metadata.FormatVersion >= 7)
        {
            using var artifact = File.OpenRead(GetCurrentModelArtifactPath(path, metadata.ArtifactSlot));
            hash += ":" + Convert.ToHexString(SHA256.HashData(artifact));
        }
        return (model, hash);
    }
}
