namespace NNtrain;

internal static partial class Program
{
    internal static int Run(
        string[] args,
        TextWriter output,
        TextWriter error,
        bool openLossGraph = false)
        => Run(
            args,
            output,
            error,
            openLossGraph,
            ConfigLoader.Load);

    internal static int Run(
        string[] args,
        TextWriter output,
        TextWriter error,
        bool openLossGraph,
        Func<string, CanonicalTrainingSpec> loadConfiguration)
    {
        ArgumentNullException.ThrowIfNull(args);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(error);
        ArgumentNullException.ThrowIfNull(loadConfiguration);
        if (args.Length > 0 && string.Equals(args[0], "lora", StringComparison.OrdinalIgnoreCase))
            return LoraCommand.Run(args, output, error);
        if (args.Length > 0 && string.Equals(args[0], "qwen-gguf", StringComparison.OrdinalIgnoreCase))
            return QwenGgufCommand.Run(args, output, error);
        if (args.Length > 0 && string.Equals(args[0], "gemma-gguf", StringComparison.OrdinalIgnoreCase))
            return GemmaGgufCommand.Run(args, output, error);
        if (args.Length > 0 && string.Equals(args[0], "qwen-lora", StringComparison.OrdinalIgnoreCase))
            return QwenLoraCommand.Run(args, output, error);
        if (args.Length > 0 && string.Equals(args[0], "qwen-lora-prepare", StringComparison.OrdinalIgnoreCase))
            return QwenLoraPreparation.Run(args, output, error);

        if (!TrainingCommandArguments.TryParse(args, error, out TrainingCommandArguments? arguments))
            return 1;

        string configurationPath = arguments.ConfigurationPath;
        string? generatePrompt = arguments.GeneratePrompt;
        string? generationConfigurationPath = arguments.GenerationConfigurationPath;
        bool resumeFromCheckpoint = arguments.ResumeFromCheckpoint;
        bool autoResume = arguments.AutoResume;

        try
        {
            if (generationConfigurationPath is not null)
                return GenerateCommand.Run(generationConfigurationPath, output, error);
            CanonicalTrainingSpec canonical =
                loadConfiguration(configurationPath);
            if (canonical is CanonicalWikiTrainingSpec wiki)
            {
                if (generatePrompt is not null)
                {
                    return WikiLanguageModelCommand.Run(
                        wiki,
                        generatePrompt,
                        output,
                        error,
                        openLossGraph);
                }
                WikiTrainingConfiguration wikiConfig = wiki.Configuration;
                using TrainingRunGuard wikiRun = TrainingRunGuard.Begin(
                    wiki.ConfigurationPath,
                    wikiConfig.CheckpointPath);
                bool resumeWiki = TrainingResumePolicy.Resolve(
                    resumeFromCheckpoint,
                    autoResume || wikiConfig.AutoResume,
                    wikiRun,
                    wikiConfig.CheckpointPath,
                    output);
                int wikiExitCode = WikiLanguageModelCommand.Run(
                    wiki,
                    generatePrompt: null,
                    output,
                    error,
                    openLossGraph,
                    resumeWiki);
                if (wikiExitCode == 0)
                    wikiRun.Complete();
                return wikiExitCode;
            }
            if (generatePrompt is not null)
            {
                throw new ArgumentException(
                    "--generate can only be used with a gpt_rin_wiki_jp " +
                    "configuration.");
            }

            CanonicalClassificationTrainingSpec classification = canonical
                as CanonicalClassificationTrainingSpec
                ?? throw new InvalidDataException(
                    "Configuration is not an image-classification task.");
            return ClassificationTrainingCommand.Run(
                classification,
                configurationPath,
                resumeFromCheckpoint,
                autoResume,
                output,
                error,
                openLossGraph);
        }
        catch (Exception exception) when (
            exception is IOException
            or System.Text.Json.JsonException
            or ArgumentException)
        {
            error.WriteLine($"Error: {exception.Message}");
            return 2;
        }
    }
}
