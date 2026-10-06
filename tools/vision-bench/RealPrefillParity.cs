using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using NNtrain;

internal static class RealPrefillParity
{
    internal static void Run(string[] args, string directory, int chunk, int pool, int deferred)
    {
        string modelPath = Path.GetFullPath("models/Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf");
        string? adapter = args.Contains("--lora")
            ? Path.GetFullPath("models/lora_rintya_qa142_Qwen3.8-27B-Uncensored-noMTP-IQ2_M_2epoch.gguf") : null;
        var tokenizer = Qwen2GgufTokenizer.Load(modelPath);
        var prompts = new List<Qwen35PromptToken[]>();
        var candidatePrompts = new List<Qwen35PromptToken[]>();
        var positions = new List<int>();
        bool xmxVisionLinear = args.Contains("--xmx"), xmxVisionAttention = args.Contains("--xmx-attention");
        bool flashVisionAttention = args.Contains("--flash-attention");
        void BuildPrompts(bool fast)
        {
        using (var encoder = Qwen35VisionEncoder.Load(Path.GetFullPath(
            "models/huggingface/unsloth/Qwen3.5-27B-GGUF/mmproj-F16.gguf"),
            useXmxLinear: fast && xmxVisionLinear, useXmxAttention: fast && xmxVisionAttention,
            useFlashAttention: fast && flashVisionAttention))
        for (int color = 0; color < 3; color++)
        {
            byte[] pixels = new byte[768 * 768 * 3];
            for (int i = 0; i < 768 * 768; i++) pixels[i * 3 + color] = 255;
            var embedding = encoder.Encode(Qwen35VisionPreprocessor.Prepare(pixels, 768, 768));
            var prompt = new List<Qwen35PromptToken>();
            int position = 0;
            AddText("<|im_start|>user\n<|vision_start|>");
            int imageId = tokenizer.Encode("<|image_pad|>").Single();
            for (int row = 0; row < embedding.GridHeight; row++)
            for (int col = 0; col < embedding.GridWidth; col++)
                prompt.Add(new(imageId, embedding.Values.AsSpan(
                    (row * embedding.GridWidth + col) * embedding.EmbeddingLength,
                    embedding.EmbeddingLength).ToArray(), new(position, position + row, position + col)));
            position += Math.Max(embedding.GridHeight, embedding.GridWidth);
            AddText("<|vision_end|>この画像の色を日本語で答えて。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
            if (fast) candidatePrompts.Add(prompt.ToArray());
            else { prompts.Add(prompt.ToArray()); positions.Add(position); }
            void AddText(string text)
            {
                foreach (int id in tokenizer.Encode(text)) prompt.Add(new(id, null, Qwen35Position.Scalar(position++)));
            }
        }
        }
        BuildPrompts(false);
        if (xmxVisionLinear || xmxVisionAttention || flashVisionAttention) BuildPrompts(true);
        else candidatePrompts = prompts;
        string referencePrefix = adapter is null ? "base-reference" : "lora-reference";
        string referenceManifest = Path.Combine(directory, referencePrefix + ".json");
        string sourceIdentity = Identity(modelPath) + "|" + (adapter is null ? "none" : Identity(adapter));
        using var digest = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (var prompt in prompts)
        foreach (var token in prompt)
        {
            digest.AppendData(BitConverter.GetBytes(token.TokenId));
            digest.AppendData(BitConverter.GetBytes(token.Position.Temporal));
            digest.AppendData(BitConverter.GetBytes(token.Position.Height));
            digest.AppendData(BitConverter.GetBytes(token.Position.Width));
            if (token.Embedding is not null) digest.AppendData(MemoryMarshal.AsBytes(token.Embedding.AsSpan()));
        }
        sourceIdentity += "|" + Convert.ToHexString(digest.GetHashAndReset());
        List<(int[] Tokens, float[] Logits)> reference;
        if (args.Contains("--reuse-reference"))
        {
            ReferenceCache cache = JsonSerializer.Deserialize<ReferenceCache>(File.ReadAllText(referenceManifest))!;
            if (cache.Identity != sourceIdentity) throw new InvalidDataException("Reference inputs changed; rerun without --reuse-reference.");
            reference = cache.Tokens.Select((tokens, color) =>
            {
                byte[] bytes = File.ReadAllBytes(Path.Combine(directory, $"{referencePrefix}-{color}.f32"));
                var logits = new float[bytes.Length / sizeof(float)];
                Buffer.BlockCopy(bytes, 0, logits, 0, bytes.Length);
                return (tokens, logits);
            }).ToList();
        }
        else
        {
            reference = Evaluate(false);
            for (int color = 0; color < reference.Count; color++)
                File.WriteAllBytes(Path.Combine(directory, $"{referencePrefix}-{color}.f32"),
                    MemoryMarshal.AsBytes(reference[color].Logits.AsSpan()).ToArray());
            File.WriteAllText(referenceManifest, JsonSerializer.Serialize(new ReferenceCache(sourceIdentity,
                reference.Select(row => row.Tokens).ToArray())));
        }
        var candidate = Evaluate(true);
        var comparisons = new List<object>();
        bool accepted = true;
        for (int color = 0; color < prompts.Count; color++)
        {
            double squareDifference = 0, squareReference = 0, max = 0;
            for (int i = 0; i < reference[color].Logits.Length; i++)
            {
                float a = reference[color].Logits[i], b = candidate[color].Logits[i];
                if (!float.IsFinite(a) || !float.IsFinite(b)) throw new ArithmeticException("Non-finite model logits.");
                double difference = (double)b - a;
                squareDifference += difference * difference; squareReference += (double)a * a;
                max = Math.Max(max, Math.Abs(difference));
            }
            double relativeL2 = Math.Sqrt(squareDifference / squareReference);
            bool tokensEqual = reference[color].Tokens.SequenceEqual(candidate[color].Tokens);
            bool pass = tokensEqual && relativeL2 <= 0.001 && max <= 0.02;
            accepted &= pass;
            comparisons.Add(new { color, prompt_tokens = prompts[color].Length, tokens_equal = tokensEqual,
                reference_tokens = reference[color].Tokens, candidate_tokens = candidate[color].Tokens,
                relative_l2 = relativeL2, max_abs_error = max, accepted = pass });
        }
        string output = JsonSerializer.Serialize(new { adapter, chunk, xmxVisionLinear, xmxVisionAttention, flashVisionAttention,
            ggufBslmPrefill = args.Contains("--gguf-bslm"), residentIq2Panels = args.Contains("--resident-iq2"),
            accepted, comparisons }, new JsonSerializerOptions { WriteIndented = true });
        string suffix = xmxVisionLinear || xmxVisionAttention || flashVisionAttention ? "-full-pipeline" : "";
        if (args.Contains("--resident-iq2")) suffix += "-resident";
        if (args.Contains("--gguf-bslm")) suffix += "-gguf-bslm";
        File.WriteAllText(Path.Combine(directory, (adapter is null ? "real-model-parity" : "real-model-lora-parity") + suffix + ".json"), output);
        Console.WriteLine(output);
        if (!accepted) throw new InvalidOperationException("Real model prefill parity gate failed.");

        List<(int[] Tokens, float[] Logits)> Evaluate(bool fast)
        {
            var options = new Qwen35ExecutionOptions
            {
                InferencePrefillChunkTokens = fast ? chunk : 16,
                InferenceProjectionRows = 4,
                InferenceXmxPrefill = fast,
                InferenceXmxFactoredPrefill = fast && args.Contains("--factored-prefill"),
                InferenceXmxGgufBslmPrefill = fast && args.Contains("--gguf-bslm"),
                InferenceResidentIq2Panels = fast && args.Contains("--resident-iq2"),
                InferenceXmxPackedPrefill = fast && args.Contains("--packed-prefill"),
                InferenceBatchRecurrent = fast,
                InferenceSubgroupRecurrentRms = fast && !args.Contains("--no-subgroup-rms"),
                InferenceBufferPoolMiB = fast ? pool : 64,
                InferenceDeferredReleaseMiB = fast ? deferred : 0
            };
            using var model = Qwen35QuantizedModel.Load(modelPath, [0, 1], options: options);
            if (adapter is not null) model.LoadLora(adapter);
            var results = new List<(int[], float[])>();
            for (int color = 0; color < prompts.Count; color++)
            {
                var timer = Stopwatch.StartNew();
                var prompt = fast ? candidatePrompts[color] : prompts[color];
                int[] ids = model.GenerateTokenIdsWithEmbeddings(prompt, 6, positions[color], CancellationToken.None);
                float[] logits = model.ForwardToken(tokenizer.Encode("赤")[0]);
                results.Add((ids[prompts[color].Length..], logits));
                Console.WriteLine($"parity fast={fast}, color={color}, ms={timer.Elapsed.TotalMilliseconds:F1}");
            }
            return results;
        }
    }

    private sealed record ReferenceCache(string Identity, int[][] Tokens);
    private static string Identity(string path)
    {
        var info = new FileInfo(path);
        return $"{info.FullName}:{info.Length}:{info.LastWriteTimeUtc.Ticks}";
    }
}
