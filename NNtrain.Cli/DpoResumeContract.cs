using System.Text.Json.Nodes;

namespace NNtrain;

internal static class DpoResumeContract
{
    internal static bool CompatibleContract(string saved, string current, bool allowPrecisionConversion = false)
        => ContractMismatches(saved, current, allowPrecisionConversion).Count == 0;

    internal static IReadOnlyList<string> ContractMismatches(string saved, string current, bool allowPrecisionConversion = false)
    {
        // Placement, buffering and microbatch partitioning may change; the effective update batch must not.
        var left = JsonNode.Parse(saved)!.AsObject();
        var right = JsonNode.Parse(current)!.AsObject();
        var differences = new List<string>();
        long Effective(JsonObject obj) => checked(
            (long)obj["batchSize"]!.GetValue<int>() * obj["gradientAccumulationSteps"]!.GetValue<int>());
        long savedBatch = Effective(left), currentBatch = Effective(right);
        if (savedBatch != currentBatch)
        {
            string Batch(JsonObject obj) => $"batchSize={obj["batchSize"]}, gradientAccumulationSteps={obj["gradientAccumulationSteps"]}";
            string difference = $"effective batch: saved={savedBatch} ({Batch(left)}), current={currentBatch} ({Batch(right)})";
            int currentMicrobatch = right["batchSize"]!.GetValue<int>();
            if (currentMicrobatch > 0 && savedBatch > 0 && savedBatch % currentMicrobatch == 0 && savedBatch / currentMicrobatch <= int.MaxValue)
                difference += $". To preserve {savedBatch} with batchSize={currentMicrobatch}, set gradientAccumulationSteps={savedBatch / currentMicrobatch}";
            differences.Add(difference);
        }
        foreach (string key in new[] { "batchSize", "gradientAccumulationSteps", "generationSlots", "completedQueueCapacity" })
        { left.Remove(key); right.Remove(key); }
        static bool Supported(string? mode) => mode is "float32" or "mix16_32" or "mix8_32";
        static bool HasSupportedPrecision(JsonObject contract) => contract["lora"] is not JsonObject lora ||
            Supported(lora["precisionMode"]?.GetValue<string>());
        void NormalizePrecision(JsonObject contract)
        {
            if (contract["lora"] is not JsonObject lora) return;
            string? mode = lora["precisionMode"]?.GetValue<string>();
            // Old non-BFP8 checkpoints predate this field; block size has no meaning for them.
            if (mode != "mix8_32" || allowPrecisionConversion) lora.Remove("bfp8BlockSize");
            else if (!lora.ContainsKey("bfp8BlockSize")) lora["bfp8BlockSize"] = 32;
            if (allowPrecisionConversion) lora.Remove("precisionMode");
        }
        bool savedPrecisionSupported = HasSupportedPrecision(left), currentPrecisionSupported = HasSupportedPrecision(right);
        if (!savedPrecisionSupported || !currentPrecisionSupported)
            differences.Add($"unsupported lora.precisionMode: saved={left["lora"]?["precisionMode"]?.ToJsonString() ?? "<missing>"}, current={right["lora"]?["precisionMode"]?.ToJsonString() ?? "<missing>"}");
        else
        { NormalizePrecision(left); NormalizePrecision(right); }
        void Compare(JsonObject a, JsonObject b, string prefix)
        {
            foreach (string key in a.Select(p => p.Key).Union(b.Select(p => p.Key)).Order(StringComparer.Ordinal))
            {
                string path = prefix + key;
                bool hasSaved = a.TryGetPropertyValue(key, out JsonNode? x), hasCurrent = b.TryGetPropertyValue(key, out JsonNode? y);
                if (hasSaved && hasCurrent && x is JsonObject xObject && y is JsonObject yObject)
                { Compare(xObject, yObject, path + "."); continue; }
                if (hasSaved && hasCurrent && JsonNode.DeepEquals(x, y)) continue;
                string difference = $"{path}: saved={(hasSaved ? x?.ToJsonString() ?? "null" : "<missing>")}, current={(hasCurrent ? y?.ToJsonString() ?? "null" : "<missing>")}";
                if (!allowPrecisionConversion && savedPrecisionSupported && currentPrecisionSupported &&
                    path is "lora.precisionMode" or "lora.bfp8BlockSize")
                    difference += ". Precision conversion requires allowPrecisionConversionOnResume=true";
                differences.Add(difference);
            }
        }
        Compare(left, right, "");
        return differences;
    }
}
