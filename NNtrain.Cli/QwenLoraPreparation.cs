using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace NNtrain;

/// <summary>CPU-only, lossless messages preparation. Never loads GPU model weights.</summary>
internal static class QwenLoraPreparation
{
    internal sealed record Repair(int Line, int LiteralBackslashesEscaped);
    internal sealed record RoleRepair(int Line, string Original, string Normalized);
    internal sealed record Conversion(string NormalizedJsonl, string TrainingJsonl,
        QwenLoraCommand.Example[] Examples, Repair[] Repairs, RoleRepair[] RoleRepairs, int MessageCount);
    private static readonly HashSet<string> LiteralCommands = new(StringComparer.Ordinal)
    {
        "beta", "delta", "Delta", "ell", "exists", "forall", "in", "iota", "land", "landR",
        "Longleftrightarrow", "Longrightarrow", "mathbf", "mathcal", "mathfrak",
        "mathrm", "mid", "neq", "Omega", "omega", "operatorname", "times", "varnothing", "widehat"
    };
    internal static string NormalizeMalformedLine(string line, out int repairs)
    {
        // First preserve every already valid JSON line exactly. Only malformed
        // strings with observed Markdown/LaTeX backslashes are normalized.
        try { using var parsed = JsonDocument.Parse(line); repairs = 0; return line; }
        catch (JsonException) { }
        var result = new StringBuilder(line.Length);
        repairs = 0;
        bool quoted = false;
        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (c == '"') quoted = !quoted;
            if (!quoted || c != '\\') { result.Append(c); continue; }
            if (i + 1 >= line.Length) throw new InvalidDataException("Dangling backslash; needs manual review.");
            char next = line[i + 1];
            int end = i + 1;
            while (end < line.Length && char.IsAsciiLetter(line[end])) end++;
            string command = line[(i + 1)..end];
            bool literal = next is '*' or '[' or ']' or '{' or '}' || LiteralCommands.Contains(command);
            if (literal)
            {
                // Retain the actual backslash, including \beta/\neq/\times
                // that would otherwise be misread as JSON control escapes.
                result.Append("\\\\"); repairs++; continue;
            }
            if (next is '"' or '\\' or '/' or 'b' or 'f' or 'n' or 'r' or 't' or 'u')
            {
                result.Append(c).Append(next); i++; continue;
            }
            throw new InvalidDataException($"Unrecognized escape \\{next}; needs manual review.");
        }
        string normalized = result.ToString();
        using var validation = JsonDocument.Parse(normalized);
        if (repairs == 0) throw new InvalidDataException("Malformed JSON is not an identified escape issue.");
        return normalized;
    }

    internal static Conversion ConvertMessages(string text)
    {
        var normalized = new StringBuilder();
        var training = new StringBuilder();
        var examples = new List<QwenLoraCommand.Example>();
        var repairs = new List<Repair>();
        var roleRepairs = new List<RoleRepair>();
        int lineNumber = 0, messageCount = 0;
        using var reader = new StringReader(text.TrimStart('\uFEFF'));
        while (reader.ReadLine() is { } line)
        {
            lineNumber++;
            if (string.IsNullOrWhiteSpace(line)) continue;
            string clean;
            try { clean = NormalizeMalformedLine(line, out int fixedCount); if (fixedCount > 0) repairs.Add(new(lineNumber, fixedCount)); }
            catch (Exception e) when (e is JsonException or InvalidDataException)
            { throw new InvalidDataException($"Data line {lineNumber}: {e.Message}", e); }
            using var document = JsonDocument.Parse(clean);
            if (document.RootElement.ValueKind != JsonValueKind.Object
                || document.RootElement.EnumerateObject().Count() != 1
                || !document.RootElement.TryGetProperty("messages", out var messages)
                || messages.ValueKind != JsonValueKind.Array || messages.GetArrayLength() < 2)
                throw new InvalidDataException($"Line {lineNumber}: expected messages only, at least two turns; nothing will be discarded.");
            var turns = new List<(string Role, string Content)>();
            foreach (var message in messages.EnumerateArray())
            {
                if (message.ValueKind != JsonValueKind.Object || message.EnumerateObject().Count() != 2
                    || !message.TryGetProperty("role", out var role) || role.ValueKind != JsonValueKind.String
                    || !message.TryGetProperty("content", out var content) || content.ValueKind != JsonValueKind.String)
                    throw new InvalidDataException($"Line {lineNumber}: unsupported message fields; nothing will be discarded.");
                string name = role.GetString()!;
                if (name == "assisutant" && messages.GetArrayLength() == 2
                    && turns.Count == 1 && turns[0].Role == "user")
                {
                    roleRepairs.Add(new(lineNumber, name, "assistant"));
                    name = "assistant";
                }
                if (name is not ("system" or "user" or "assistant"))
                    throw new InvalidDataException($"Line {lineNumber}: unsupported role '{name}'.");
                turns.Add((name, content.GetString()!));
            }
            if (turns[^1].Role != "assistant" || string.IsNullOrWhiteSpace(turns[^1].Content))
                throw new InvalidDataException($"Line {lineNumber}: final turn must be a nonempty assistant response.");
            if (roleRepairs.Count > 0 && roleRepairs[^1].Line == lineNumber)
            {
                var node = JsonNode.Parse(clean)!;
                node["messages"]![1]!["role"] = "assistant";
                clean = node.ToJsonString();
            }
            var prompt = new StringBuilder();
            foreach (var turn in turns.Take(turns.Count - 1))
                prompt.Append("<|im_start|>").Append(turn.Role).Append('\n').Append(turn.Content).Append("<|im_end|>\n");
            // Same non-thinking assistant prefix used by the existing QA80
            // ChatML file. Response content itself is kept exactly, not trimmed.
            prompt.Append("<|im_start|>assistant\n<think>\n\n</think>\n\n");
            var example = new QwenLoraCommand.Example(prompt.ToString(), turns[^1].Content);
            examples.Add(example); messageCount += turns.Count;
            normalized.Append(clean).Append('\n');
            training.Append(JsonSerializer.Serialize(example, LoraConfiguration.Json)).Append('\n');
        }
        if (examples.Count == 0) throw new InvalidDataException("No examples.");
        return new(normalized.ToString(), training.ToString(), examples.ToArray(), repairs.ToArray(), roleRepairs.ToArray(), messageCount);
    }

    internal static int Run(string[] args, TextWriter output, TextWriter error)
    {
        try
        {
            var values = new Dictionary<string, string>(StringComparer.Ordinal);
            for (int i = 1; i < args.Length; i++)
            {
                string key = args[i];
                if (key is not ("--model" or "--data" or "--output" or "--context" or "--epochs"))
                    throw new ArgumentException($"Unknown preparation argument: {key}");
                if (++i >= args.Length || !values.TryAdd(key, args[i])) throw new ArgumentException("Missing/duplicate argument.");
            }
            string model = Path.GetFullPath(values["--model"]), data = Path.GetFullPath(values["--data"]),
                directory = Path.GetFullPath(values["--output"]);
            if (Directory.Exists(directory) || File.Exists(directory)) throw new IOException("Preparation output must be a new path.");
            int requested = values.TryGetValue("--context", out string? context) ? int.Parse(context) : 0;
            int epochs = values.TryGetValue("--epochs", out string? epochValue) ? int.Parse(epochValue) : 2;
            if (requested < 0 || epochs <= 0) throw new ArgumentException("Context must be 0 (auto) or positive; epochs must be positive.");
            byte[] source = File.ReadAllBytes(data);
            var converted = ConvertMessages(new UTF8Encoding(false, true).GetString(source));
            var tokenizer = Qwen2GgufTokenizer.Load(model);
            if (tokenizer.EosTokenId is not int eos) throw new InvalidDataException("Model EOS missing.");
            var lengths = converted.Examples.Select((example, index) =>
            {
                int prompt = tokenizer.Encode(example.Prompt).Length;
                int response = tokenizer.Encode(example.Response).Length;
                return new { Example = index + 1, PromptTokens = prompt, ResponseTokens = response,
                    TotalTokensIncludingEos = checked(prompt + response + 1) };
            }).ToArray();
            int maximum = lengths.Max(row => row.TotalTokensIncludingEos);
            int limit = requested != 0 ? requested : maximum <= 4096 ? 4096 : maximum <= 8192 ? 8192 : maximum;
            if (limit < 2 || maximum > limit) throw new InvalidDataException($"Longest example is {maximum} tokens including EOS; requested context {limit} is too short. No truncation/padding. Use --context 0 for a sufficient proposed upper bound.");
            var descriptor = Qwen35Gguf.Inspect(model);
            if (limit > descriptor.ContextLength) throw new InvalidDataException("Examples exceed model context.");
            var config = new QwenLoraTrainingConfiguration
            {
                DataPath = "training.jsonl", AdapterPath = "adapter.bin", Devices = [0, 1], ContextLength = limit,
                MaxSteps = checked(epochs * converted.Examples.Length), SaveEverySteps = 5,
                Rank = 8, Alpha = 16, LearningRate = 0.0001f, WeightDecay = 0, GradientClip = 1, Seed = 1,
                Iq2ForwardPrecision = "exact", PromptPrefix = "", ResponsePrefix = "", ShuffleExamples = true,
                ShowLossGraph = true, OpenLossGraph = false, LossGraphPath = "loss.html"
            };
            int[] counts = lengths.Select(row => row.TotalTokensIncludingEos).ToArray();
            var options = config.ExecutionOptions(counts, resume: false);
            // Freeze the actual resolved choices so future resume uses the same
            // choices, rather than reapplying automatic defaults to a saved run.
            config = config with { UseMeasuredLengthDefaults = false, FusedAttentionRows = options.TrainingFusedAttentionRows,
                HostCheckpointBufferHandoff = options.TrainingHostCheckpointBufferHandoff,
                StreamedAttentionTileRows = options.TrainingStreamedAttentionTileRows,
                Iq2RollingTranspose = options.TrainingIQ2RollingTranspose };
            config.Validate();
            // Exact same dataset validator as training, CPU-only.
            byte[] trainingBytes = Encoding.UTF8.GetBytes(converted.TrainingJsonl);
            var reread = QwenLoraCommand.ReadExamples(trainingBytes, tokenizer, config);
            if (reread.Length != converted.Examples.Length) throw new InvalidDataException("Conversion count changed.");
            string json = JsonSerializer.Serialize(config, LoraConfiguration.Json);
            var reload = JsonSerializer.Deserialize<QwenLoraTrainingConfiguration>(json, LoraConfiguration.Json)!;
            reload.Validate();
            if (reload.ExecutionOptions(counts, true) != options) throw new InvalidDataException("Saved speed settings changed on resume.");
            Directory.CreateDirectory(directory);
            WriteNew(Path.Combine(directory, "source.jsonl"), source);
            WriteNew(Path.Combine(directory, "normalized.messages.jsonl"), Encoding.UTF8.GetBytes(converted.NormalizedJsonl));
            WriteNew(Path.Combine(directory, "training.jsonl"), trainingBytes);
            WriteNew(Path.Combine(directory, "train.json"), Encoding.UTF8.GetBytes(json));
            var report = new { Status = "prepared-cpu-only", Model = model, Source = data,
                SourceSha256 = Convert.ToHexString(SHA256.HashData(source)), TrainingSha256 = Convert.ToHexString(SHA256.HashData(trainingBytes)),
                Examples = converted.Examples.Length, Messages = converted.MessageCount, Repairs = converted.Repairs, RoleRepairs = converted.RoleRepairs,
                Epochs = epochs, MaxSteps = config.MaxSteps, ContextLength = limit, MaximumActualTokens = maximum,
                MinimumActualTokens = lengths.Min(row => row.TotalTokensIncludingEos), Over4096 = lengths.Count(row => row.TotalTokensIncludingEos > 4096),
                Padding = false, Packing = false, Truncation = false, Options = options, Lengths = lengths,
                Notes = "Source is preserved. Literal Markdown/LaTeX backslashes are retained; legitimate JSON newline escapes remain newlines. The isolated final-response role typo assisutant is normalized to assistant and recorded. Exact precision; rank8/alpha16/LR0.0001 and default 2 epochs are inherited/proposed from the earlier QA configuration, not recovered QA142 hyperparameters. No GPU model load or training. Variable lengths do not activate measured 4096/8192 speed defaults." };
            WriteNew(Path.Combine(directory, "preparation.json"), JsonSerializer.SerializeToUtf8Bytes(report, new JsonSerializerOptions { WriteIndented = true }));
            output.WriteLine($"Prepared {reread.Length} examples, {converted.MessageCount} messages; actual tokens {lengths.Min(r => r.TotalTokensIncludingEos)}..{maximum}, context upper bound {limit}, over4096={lengths.Count(r => r.TotalTokensIncludingEos > 4096)}.");
            output.WriteLine($"Config: {Path.Combine(directory, "train.json")}. CPU only; no padding/truncation or GPU work.");
            return 0;
        }
        catch (Exception e) { error.WriteLine($"Preparation failed: {e.Message}"); return 1; }
    }
    private static void WriteNew(string path, byte[] bytes)
    {
        using var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        stream.Write(bytes);
    }
}
