using System.Diagnostics;
using System.Globalization;
using System.Text;
using NNtrain.Gemma;

namespace NNtrain;

internal static class GemmaGgufCommand
{
    internal static int Run(string[] args, TextWriter output, TextWriter error)
    {
        const string usage = "Usage: NNtrain.Cli gemma-gguf --model <gemma3-Q8_0.gguf> " +
            "--prompt <text> [--max-new-tokens 128] [--context 4096] " +
            "[--threads N] [--backend cpu|opencl] [--gpu-layers N] [--opencl-all-devices] [--tensor-split 53,47] " +
            "[--temperature 0] [--top-p 1] " +
            "[--top-k 1] [--no-stream]";
        try
        {
            string? modelPath = null;
            string? prompt = null;
            int maxNewTokens = 128;
            int contextSize = 4096;
            int? threads = null;
            int gpuLayers = 0;
            Gemma3Backend? backend = null;
            int topK = 1;
            float temperature = 0f;
            float topP = 1f;
            bool stream = true;
            bool allOpenClDevices = false;
            float[]? tensorSplits = null;

            for (int i = 1; i < args.Length; i++)
            {
                string option = args[i];
                if (option is "--help" or "-h")
                {
                    output.WriteLine(usage);
                    return 0;
                }
                if (option == "--no-stream")
                {
                    stream = false;
                    continue;
                }
                if (option == "--opencl-all-devices")
                {
                    allOpenClDevices = true;
                    continue;
                }
                if (i + 1 >= args.Length)
                    throw new ArgumentException($"Missing value for {option}.");
                string value = args[++i];
                switch (option)
                {
                    case "--model": modelPath = Path.GetFullPath(value); break;
                    case "--prompt": prompt = value; break;
                    case "--max-new-tokens": maxNewTokens = ParseGemmaInteger(value, option, allowZero: true); break;
                    case "--context": contextSize = ParseGemmaInteger(value, option); break;
                    case "--threads": threads = ParseGemmaInteger(value, option); break;
                    case "--gpu-layers": gpuLayers = ParseGemmaInteger(value, option, allowZero: true); break;
                    case "--tensor-split":
                        tensorSplits = value.Split(',').Select(part =>
                        {
                            if (!float.TryParse(part, NumberStyles.Float, CultureInfo.InvariantCulture, out float split)
                                || !float.IsFinite(split) || split <= 0)
                                throw new ArgumentException("--tensor-split requires comma-separated positive finite values.");
                            return split;
                        }).ToArray();
                        break;
                    case "--backend":
                        backend = value.ToLowerInvariant() switch
                        {
                            "cpu" => Gemma3Backend.Cpu,
                            "opencl" => Gemma3Backend.OpenCl,
                            _ => throw new ArgumentException("--backend must be cpu or opencl.")
                        };
                        break;
                    case "--top-k": topK = ParseGemmaInteger(value, option); break;
                    case "--temperature": temperature = ParseGemmaFloat(value, option, minExclusive: false); break;
                    case "--top-p": topP = ParseGemmaFloat(value, option, minExclusive: true); break;
                    default: throw new ArgumentException($"Unknown gemma-gguf option: {option}");
                }
            }
            if (modelPath is null || string.IsNullOrWhiteSpace(prompt))
                throw new ArgumentException(usage);
            if (topP > 1f)
                throw new ArgumentException("--top-p must be at most 1.");
            // An explicit GPU layer count uses OpenCL unless the caller requests CPU.
            Gemma3Backend selectedBackend = backend ??
                (gpuLayers > 0 ? Gemma3Backend.OpenCl : Gemma3Backend.Cpu);
            if (selectedBackend == Gemma3Backend.Cpu && gpuLayers > 0)
                throw new ArgumentException("--backend cpu requires --gpu-layers 0.");
            if (selectedBackend == Gemma3Backend.OpenCl && gpuLayers == 0)
                throw new ArgumentException("--backend opencl requires --gpu-layers N where N is positive.");
            if (allOpenClDevices && selectedBackend != Gemma3Backend.OpenCl)
                throw new ArgumentException("--opencl-all-devices requires the OpenCL backend.");
            if (tensorSplits is not null && (!allOpenClDevices || tensorSplits.Length < 2))
                throw new ArgumentException("--tensor-split requires --opencl-all-devices and at least two values.");

            var loadTimer = Stopwatch.StartNew();
            using Gemma3LocalModel model = Gemma3LocalModel.Load(
                modelPath, contextSize, gpuLayers, threads, selectedBackend, allOpenClDevices, tensorSplits);
            loadTimer.Stop();
            string formattedPrompt = Gemma3LocalModel.FormatConversation(
                [new Gemma3ChatTurn("user", prompt)]);
            output.WriteLine($"Gemma 3 Q8_0 GGUF: {modelPath}");
            string deviceSelection = selectedBackend == Gemma3Backend.Cpu ? "none"
                : allOpenClDevices ? "all Intel" : "selected";
            output.WriteLine($"backend: {(selectedBackend == Gemma3Backend.OpenCl ? "OpenCL" : "CPU")}, GPU layers: {gpuLayers}, OpenCL devices: {deviceSelection}");
            if (tensorSplits is not null)
                output.WriteLine($"tensor split: {string.Join(",", tensorSplits.Select(value => value.ToString(CultureInfo.InvariantCulture)))}");
            if (stream)
            {
                output.WriteLine("generated:");
                output.Flush();
            }
            var generationTimer = Stopwatch.StartNew();
            Gemma3GenerationResult result = model.GenerateAsync(
                formattedPrompt, maxNewTokens,
                chunk =>
                {
                    if (!stream) return;
                    output.Write(chunk);
                    output.Flush();
                }, CancellationToken.None, temperature, topP, topK)
                .GetAwaiter().GetResult();
            generationTimer.Stop();
            if (stream) output.WriteLine();
            else
            {
                output.WriteLine("generated:");
                output.WriteLine(result.Text);
            }
            output.Flush();

            double? decodeTokensPerSecond = result.CompletionTokens > 1
                && result.FirstTokenMilliseconds is double first
                && generationTimer.Elapsed.TotalMilliseconds > first
                    ? (result.CompletionTokens - 1) * 1000.0 /
                      (generationTimer.Elapsed.TotalMilliseconds - first)
                    : null;
            string firstToken = result.FirstTokenMilliseconds is double firstMs
                ? firstMs.ToString("F1", CultureInfo.InvariantCulture) + " ms"
                : "n/a";
            string decode = decodeTokensPerSecond is double rate
                ? rate.ToString("F2", CultureInfo.InvariantCulture) + " tok/s"
                : "n/a";
            error.WriteLine(FormattableString.Invariant(
                $"timing: load={loadTimer.Elapsed.TotalSeconds:F3} s, prompt={result.PromptTokens} tokens, prefill/first-token={firstToken}, decode={decode}, generated={result.CompletionTokens}, generation={generationTimer.Elapsed.TotalSeconds:F3} s"));
            error.Flush();
            return 0;
        }
        catch (Exception exception) when (exception is not OutOfMemoryException and not StackOverflowException)
        {
            error.WriteLine($"Error: {exception.Message}");
            return exception is ArgumentException ? 1 : 2;
        }
    }

    private static int ParseGemmaInteger(string value, string option, bool allowZero = false)
    {
        if (!int.TryParse(value.Normalize(NormalizationForm.FormKC),
                NumberStyles.Integer, CultureInfo.InvariantCulture, out int number)
            || (allowZero ? number < 0 : number <= 0))
            throw new ArgumentException($"{option} must be a {(allowZero ? "nonnegative" : "positive")} integer.");
        return number;
    }

    private static float ParseGemmaFloat(string value, string option, bool minExclusive)
    {
        if (!float.TryParse(value.Normalize(NormalizationForm.FormKC),
                NumberStyles.Float, CultureInfo.InvariantCulture, out float number)
            || !float.IsFinite(number)
            || (minExclusive ? number <= 0f : number < 0f))
            throw new ArgumentException($"{option} must be a finite {(minExclusive ? "positive" : "nonnegative")} number.");
        return number;
    }
}
