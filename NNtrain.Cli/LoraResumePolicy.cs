namespace NNtrain;

internal static class LoraResumePolicy
{
    internal static bool Resolve(LoraTrainingConfiguration config, string adapterPath, bool generation, TextWriter output)
    {
        bool exists = File.Exists(adapterPath);
        if (generation || config.Resume)
        {
            if (!exists) throw new FileNotFoundException("LoRA adapter checkpoint required for generation/resume was not found.", adapterPath);
            return true;
        }
        if (config.AutoResume)
        {
            output.WriteLine(exists
                ? $"auto-resume = restoring adapter checkpoint {adapterPath}"
                : $"auto-resume = no adapter checkpoint at {adapterPath}; starting a new adapter");
            // The caller must still validate/load the file. Never turn a corrupt or
            // incompatible checkpoint into permission to overwrite it with a fresh run.
            return exists;
        }
        if (exists) throw new IOException("Adapter already exists. Set autoResume=true or resume=true, or choose a new adapterPath; refusing to overwrite it.");
        return false;
    }
}
