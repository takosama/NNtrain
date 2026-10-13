using System.IO;
using NNtrain.Audio;

namespace NNtrain.Gui;

internal static class AudioModelLoader
{
    internal static ILocalAsrModel Load(string directory, bool parakeet, int? device,
        CancellationToken ct)
    {
        if (parakeet)
        {
            string archive = Path.Combine(directory, "parakeet-tdt_ctc-0.6b-ja.nemo");
            if (File.Exists(archive) && (!File.Exists(Path.Combine(directory, "model_config.yaml")) ||
                !File.Exists(Path.Combine(directory, "tokenizer.vocab"))))
                NemoCheckpointConverter.ExtractModelAssets(archive, directory, ct);
            if (!File.Exists(Path.Combine(directory, "model.safetensors")))
                NemoCheckpointConverter.ConvertToFp16(archive, Path.Combine(directory, "model.safetensors"), ct);
        }
        ILocalAsrModel loaded = parakeet ? ParakeetCtcModel.Load(directory, ct) : NemotronAsrModel.Load(directory, ct);
        try
        {
            if (device is >= 0) loaded.EnableArc(device.Value, ct);
            return loaded;
        }
        catch
        {
            loaded.Dispose();
            throw;
        }
    }
}
