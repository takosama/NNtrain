using System.Diagnostics.CodeAnalysis;

namespace NNtrain;

internal sealed record TrainingCommandArguments(
    string ConfigurationPath,
    string? GeneratePrompt,
    string? GenerationConfigurationPath,
    bool ResumeFromCheckpoint,
    bool AutoResume)
{
    internal static bool TryParse(
        string[] args,
        TextWriter error,
        [NotNullWhen(true)] out TrainingCommandArguments? arguments)
    {
        string configurationPath;
        string? generatePrompt = null;
        string? generationConfigurationPath = null;
        bool resumeFromCheckpoint = false;
        bool autoResume = false;
        if (args.Length == 0)
        {
            configurationPath = FindDefaultConfiguration();
        }
        else if (args.Length == 2
            && string.Equals(
                args[0],
                "--config",
                StringComparison.OrdinalIgnoreCase))
        {
            configurationPath = args[1];
        }
        else if (args.Length == 1
            && string.Equals(
                args[0],
                "--resume",
                StringComparison.OrdinalIgnoreCase))
        {
            configurationPath = FindDefaultConfiguration();
            resumeFromCheckpoint = true;
        }
        else if (args.Length == 1
            && string.Equals(
                args[0],
                "--auto-resume",
                StringComparison.OrdinalIgnoreCase))
        {
            configurationPath = FindDefaultConfiguration();
            autoResume = true;
        }
        else if (args.Length == 3
            && string.Equals(
                args[0],
                "--config",
                StringComparison.OrdinalIgnoreCase)
            && string.Equals(
                args[2],
                "--resume",
                StringComparison.OrdinalIgnoreCase))
        {
            configurationPath = args[1];
            resumeFromCheckpoint = true;
        }
        else if (args.Length == 3
            && string.Equals(
                args[0],
                "--config",
                StringComparison.OrdinalIgnoreCase)
            && string.Equals(
                args[2],
                "--auto-resume",
                StringComparison.OrdinalIgnoreCase))
        {
            configurationPath = args[1];
            autoResume = true;
        }
        else if (args.Length == 2
            && string.Equals(
                args[0],
                "--generate-config",
                StringComparison.OrdinalIgnoreCase))
        {
            configurationPath = string.Empty;
            generationConfigurationPath = args[1];
        }
        else if (args.Length == 4
            && string.Equals(
                args[0],
                "--config",
                StringComparison.OrdinalIgnoreCase)
            && string.Equals(
                args[2],
                "--generate",
                StringComparison.OrdinalIgnoreCase))
        {
            configurationPath = args[1];
            generatePrompt = args[3];
        }
        else
        {
            error.WriteLine(
                "Usage: NNtrain.Cli [--config <training-config.json>] " +
                "[--resume | --auto-resume | --generate <prompt>] " +
                "[--generate-config <generate.json>]");
            error.WriteLine("       NNtrain.Cli lora --model <DRN-checkpoint.json> [--config traning-lora.json] [--generate <prompt>]");
            error.WriteLine("       NNtrain.Cli gemma-gguf --model <gemma3-Q8_0.gguf> --prompt <text> [--max-new-tokens 128] [--context 4096] [--threads N] [--backend cpu|opencl] [--gpu-layers N] [--opencl-all-devices] [--temperature 0] [--top-p 1] [--top-k 1] [--no-stream]");
            arguments = null;
            return false;
        }

        arguments = new TrainingCommandArguments(
            configurationPath,
            generatePrompt,
            generationConfigurationPath,
            resumeFromCheckpoint,
            autoResume);
        return true;
    }

    internal static string FindDefaultConfiguration()
    {
        string[] fileNames =
        [
            "training.wiki-jp.json",
            "training.example.json",
        ];

        foreach (string startPath in new[]
        {
            Environment.CurrentDirectory,
            AppContext.BaseDirectory,
        })
        {
            DirectoryInfo? directory = new DirectoryInfo(startPath);
            while (directory is not null)
            {
                foreach (string fileName in fileNames)
                {
                    string candidate = Path.Combine(
                        directory.FullName,
                        fileName);
                    if (File.Exists(candidate))
                        return candidate;
                }

                directory = directory.Parent;
            }
        }

        return Path.Combine(
            Environment.CurrentDirectory,
            fileNames[0]);
    }
}
